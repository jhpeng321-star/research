"""Checkpoint-agnostic POLICY14 preparation/training; never imports a simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import asdict, replace

import numpy as np
import torch

from .contract import RewardWeights
from .dreamer import DreamerConfig
from .lanes import LaneRules
from .objective14 import authoritative_reward
from .policy14 import Policy14Trainer, SmokeSampler, independent_score, make_controller_factory
from .real_returns14 import NPZRealReturnProvider, ScaledTeamCritic, warm_start_real_critic
from .start_states import checked_path
from .train_cli import GroupSignal, _load_starts, _read_config, policy_settings


def load_members(checkpoints, configs, device):
    """Strict weights; explicit supplied config controls model/physics semantics.

    JSON accepts flat train_config.json or nested resolved config. TOML inheritance
    is path-checked before every read. No dataset in a WM config is opened here.
    """
    from wms.config.schema import from_mapping
    from wms.config.train import TrainConfig
    from wms.eval.backend import load_backend
    models, receipts = [], []
    if len(checkpoints) != len(configs) or not checkpoints:
        raise ValueError("one WM config is required for each checkpoint (ensemble may have one member)")
    for checkpoint, config in zip(checkpoints, configs):
        checkpoint, config = checked_path(checkpoint), checked_path(config)
        raw = (json.loads(config.read_text(encoding="utf-8")) if config.suffix == ".json"
               else _read_config(config))
        raw = raw.get("config", raw)
        nested = "model" in raw or "schema_version" in raw
        if nested:
            cfg = from_mapping(TrainConfig, raw)
            flat, physics = cfg.to_legacy_flat(), asdict(cfg.physics)
        else:
            flat, physics = raw, None
        # to_legacy_flat intentionally omits default False switches. Fill the
        # complete explicit spec before overlaying a checkpoint, otherwise an old
        # embedded True (e.g. courtesy_off) silently defeats the supplied config.
        from wms.model.spec import ModelSpec
        defaults = asdict(ModelSpec.from_mapping(flat))
        flat = {**flat, **{key: value for key, value in defaults.items()
                          if isinstance(value, bool) or key == "follower_courtesy_intent"}}
        for sibling in ("physics.json", "TRAIN_RECEIPT.json", "lane_calib.json"):
            checked_path(checkpoint.with_name(sibling))
        backend = load_backend(checkpoint, device=device, model_config=flat, physics=physics)
        model = backend.model.eval().requires_grad_(False)
        models.append(model)
        receipts.append({"checkpoint": str(checkpoint), "config": str(config),
                         "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                         "courtesy_state_grad": model.follower_courtesy_state_grad,
                         "courtesy_intent": model.follower_courtesy_intent,
                         "courtesy_off": model.follower_courtesy_off,
                         "physics": asdict(model.physics)})
    return models, receipts


# --- S-GPUB-LOOP (10-26, main/owner decision): update-10 signal gate on a FROZEN rescore panel ---
# The default gate compares the independent rescore of ONE start (rescore batch = batch_size 1)
# at update 10 vs update 0, which cannot resolve a ~0.3 change of J ~ 750.  With --gate-panel the
# SAME criterion form (independent, critic-free J; update-10 must be better, i.e. lower, than
# update 0) is evaluated on a frozen panel of VAL start windows of the same TRAIN start corpus,
# paired per start.  Off (default) = the old gate, bit-identical.  The per-update 1-start rescore
# (entropy annealing, rescores.jsonl) is unchanged either way; panel scoring restores all RNG.
GATE_PANEL_SCHEMA = "policy14_gate_panel/1"


def _json_meta(meta):
    return {str(k): v for k, v in sorted(meta.items())
            if not str(k).startswith("_") and isinstance(v, (str, int, float, bool))}


def build_gate_panel(val, n=16, seed=965150):
    """Stratified (round-robin over sorted penetrations) VAL windows; own generator."""
    generator = torch.Generator().manual_seed(int(seed))
    orders = {}
    for penetration in val.penetrations:
        group = sorted(val.groups[penetration])
        orders[penetration] = [group[i] for i in torch.randperm(len(group), generator=generator).tolist()]
    picked, cursor = [], {p: 0 for p in val.penetrations}
    while len(picked) < n:
        progressed = False
        for penetration in val.penetrations:
            if len(picked) < n and cursor[penetration] < len(orders[penetration]):
                picked.append(orders[penetration][cursor[penetration]])
                cursor[penetration] += 1
                progressed = True
        if not progressed:
            raise ValueError(f"VAL has fewer than {n} start windows")
    return {"schema": GATE_PANEL_SCHEMA, "n": n, "panel_seed": int(seed),
            "selection": "VAL subset of the TRAIN start corpus (split_start_states, run seed); "
                         "round-robin over sorted penetrations, torch.randperm(panel_seed) per group",
            "starts": [{"index": int(i), "meta": _json_meta(val.metadata[i])} for i in picked]}


def load_gate_panel(path, val):
    doc = json.loads(checked_path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != GATE_PANEL_SCHEMA or not doc.get("starts"):
        raise ValueError("gate panel: wrong schema or empty")
    for start in doc["starts"]:
        index = int(start["index"])
        if index not in val.metadata or _json_meta(val.metadata[index]) != start["meta"]:
            raise ValueError(f"gate panel start {index} does not match this VAL corpus (frozen list differs)")
        world = str(start["meta"].get("world", start["meta"].get("world_id", "")))
        match = re.search(r"w(\d+)m", world)
        if match and 964000 <= int(match[1]) <= 964199:
            raise ValueError("gate panel contains an excluded evaluation seed")
    return doc


def score_gate_panel(trainer, val, panel, device):
    """Independent (critic-free) J per panel start; independent_score restores every RNG."""
    return [independent_score(trainer, val._batch([val._load(int(s["index"]))], device))["J"]
            for s in panel["starts"]]


def score_native_panel(trainer, val, panel, device):
    from .nativeinit import panel_gate
    scores = [independent_score(trainer, val._batch([val._load(int(s["index"]))], device),
                                native_reference=True) for s in panel["starts"]]
    return panel_gate([s["J"] for s in scores], [s["native_J"] for s in scores],
                      update=trainer.update_count)


def check_train_seeds(sampler):
    for meta in sampler.metadata.values():
        world = str(meta.get("world", meta.get("world_id", "")))
        match = re.search(r"w(\d+)m", world)
        if match and 964000 <= int(match[1]) <= 964199:
            raise ValueError("excluded traffic seed in start-state corpus")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/c4_policy14.toml")
    parser.add_argument("--wm-ckpt", nargs="+", required=True)
    parser.add_argument("--wm-config", nargs="+", required=True)
    parser.add_argument("--real-returns", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=965140)
    parser.add_argument("--max-updates", type=int, default=600)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--compact", action="store_true", help="print status/summary path only")
    parser.add_argument("--resume", default=None,
                        help="LOOP14: continue a c4_dreamer/3 trainer checkpoint (policy, critic, "
                             "optimisers, duals) on the supplied (accepted) WM; the WM stays frozen")
    parser.add_argument("--critic-warmup-updates", type=int, default=None,
                        help="override policy14.critic_warmup_updates (e.g. on fresh current-policy returns)")
    parser.add_argument("--real-credit", default=None,
                        help="S-SUMOCREDIT: TRAIN REAL_CREDIT.json (wms.control.real_credit14 fit); scales the "
                             "individual/group/coop/team counterfactual weights by the measured real-vs-imagined "
                             "credit multipliers in [0, 1] (absent = off, unchanged)")
    parser.add_argument("--gate-panel", default=None,
                        help="S-GPUB-LOOP: frozen gate-panel JSON; the update-10 signal gate is scored on it "
                             "(paired per start) instead of the single rescore start")
    parser.add_argument("--gate-panel-dump", default=None,
                        help="build the gate panel (VAL starts, panel seed 965150, n 16), write it there, exit")
    parser.add_argument("--horizon-gate", default=None,
                        help="WM-sha-bound COOPW horizon receipt for the current accepted loop WM")
    parser.add_argument("--native-init", action="store_true",
                        help="S-NATIVEINIT-1: native residual actor, paired native gate/learning curve")
    parser.add_argument("--longcritic-reliability", default=None)
    parser.add_argument("--longcritic-native-returns", default=None)
    parser.add_argument('--paircrit-receipt', default=None)
    parser.add_argument('--paircrit-policy-sha256', default=None)
    parser.add_argument('--looplong-boundary', default=None)
    parser.add_argument('--wm-swap-file', default=None)
    parser.add_argument('--phase-relative-signal-gate', action='store_true',
                         help='S-LOOPRESUME: apply the unchanged progress gate ten updates after resume')
    parser.add_argument('--prep-dry-run-updates', type=int, choices=(2,), default=None,
                        help='S-C4PREP: stop after exactly two scratch updates; no adoption receipt')
    args = parser.parse_args(argv)
    summary, output = {"status": "failed"}, None
    try:
        if args.seed < 0 or 964000 <= args.seed <= 964199:
            raise ValueError("excluded/invalid optimisation seed")
        if not 1 <= args.max_updates <= 600:
            raise ValueError("max-updates must be in 1..600")
        if args.smoke and (args.device != "cpu" or args.max_updates > 3):
            raise ValueError("smoke is CPU-only, at most three updates")
        raw = _read_config(args.config)
        if args.looplong_boundary:
            from .looplong import read_boundary
            prepared = read_boundary(args.looplong_boundary)
            if prepared.get('coop_credit') is False:
                for name in ('w_individual', 'w_group', 'w_coop', 'w_team'):
                    raw['counterfactual'][name] = [0.] * len(raw['counterfactual'][name])
        from .policy14_recovery import load_recovery
        recovery = load_recovery(raw, args)
        if args.native_init:
            from .nativeinit import PANEL_SHA256
            if args.smoke or not args.gate_panel:
                raise ValueError("native-init requires the real frozen TRAIN panel, not synthetic --smoke")
            if hashlib.sha256(checked_path(args.gate_panel).read_bytes()).hexdigest() != PANEL_SHA256:
                raise ValueError("native-init panel SHA mismatch")
            phase_updates = recovery['phase_updates'] if recovery else args.max_updates
            if phase_updates > 150 or phase_updates % 10:
                raise ValueError("native-init measures every 10 updates, maximum inherited cap150")
        if args.horizon_gate:
            raw["horizon"] = {**raw["horizon"], "gate_file": str(checked_path(args.horizon_gate))}
        switches = raw["policy14"]
        cfg = DreamerConfig(**raw["trainer"])
        if not switches["authoritative_j"] and cfg.real_return_mix:
            raise ValueError("real negative-J returns require authoritative_j")
        if cfg.gamma != 1.0 or cfg.imagination_grad_decay != 1.0:
            raise ValueError("POLICY14 requires undiscounted finite-horizon J and undamped action derivatives")
        if args.smoke:
            cfg = replace(cfg, batch_size=1, horizon=2, entry_pad_max=8,
                          checkpoint_every=0, validation_every=0)
        threads = int(raw["run"]["threads"])
        if not 1 <= threads <= 8:
            raise ValueError("threads must be in 1..8")
        torch.set_num_threads(threads)
        # Shared-GPU guard (as train_cli): env C4_GPU_MEM_FRACTION caps this process's CUDA
        # memory so it OOMs itself before starving a co-located run.  Unset = unchanged.
        frac = raw.get('run', {}).get('gpu_memory_fraction', os.environ.get("C4_GPU_MEM_FRACTION"))
        if frac and args.device == "cuda" and torch.cuda.is_available():
            if not 0 < float(frac) <= 1:
                raise ValueError('GPU memory fraction must be in (0, 1]')
            torch.cuda.set_per_process_memory_fraction(float(frac))
        summary['gpu_memory_fraction'] = float(frac) if frac else None
        from wms.core import speed_flags
        summary['diagnostic_grad_audit_every'] = speed_flags.every('grad_audit_every')
        candidate = checked_path(args.output_dir)
        candidate.mkdir(parents=True, exist_ok=False)
        output = candidate
        provider = NPZRealReturnProvider(args.real_returns, seed=args.seed)
        from .real_share import RealShareConfig, split_provider, exclude_development_starts, calibrate_share
        share_cfg = RealShareConfig(**raw.get('real_share', {}))
        share_dev = None
        if share_cfg.enabled:
            provider, share_dev = split_provider(provider)
        models, receipts = load_members(args.wm_ckpt, args.wm_config, args.device)
        from .policy14_probes import free_road_probe, gradient_probe, gradient_probe_13e
        from .dreamer import _restore_rng, _rng_state
        rng = _rng_state(torch.device(args.device))
        probe_models, probe_device = models, args.device
        if os.environ.get("C4_PROBES_DEVICE") == "cpu" and args.device != "cpu":
            # S-GPUB-LOOP (10-26): the readiness gates are float32 finite differences validated on
            # CPU; on CUDA a single float32 step in J flips the isolated-courtesy FD check of 13E b2
            # (CPU passes).  Opt-in: the same gates and tolerances on a CPU copy of the same weights.
            probe_models, _ = load_members(args.wm_ckpt, args.wm_config, "cpu")
            probe_device = "cpu"
        try:
            probes = {"gradient": gradient_probe(probe_models[0])} if switches["fd_gate"] else {}
            if switches["fd_gate"] and any(
                    getattr(m.physics, "emergency_braking", False)
                    or getattr(m.physics, "secure_gap_veto", False)
                    or getattr(m, "mainline_merge_yield_learnable", False) for m in probe_models[:1]):
                # POLICY14E: autograd vs FD of dJ/d(CAV accel) through the 13E envelope / gs / yield.
                probes["gradient_13e"] = gradient_probe_13e(probe_models[0])
            if switches["free_road_gate"]:
                probes["free_road"] = free_road_probe(probe_models[0])
        finally:
            _restore_rng(rng, torch.device(args.device))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.smoke:
            train, val = SmokeSampler(models[0]), SmokeSampler(models[0])
        else:
            train, val = _load_starts(raw["data"], models, args.seed, entries=True)
            check_train_seeds(train)
            check_train_seeds(val)
            if share_dev is not None:
                train = exclude_development_starts(train, set(share_dev.provenance['seeds'].tolist()))
        if args.gate_panel_dump:
            text = json.dumps(build_gate_panel(val), indent=1, sort_keys=True) + "\n"
            dump = checked_path(args.gate_panel_dump)
            if dump.exists():
                raise ValueError("gate panel dump exists (never overwritten)")
            dump.write_text(text, encoding="utf-8")
            summary.update(status="panel_dumped", gate_panel_dump=str(dump),
                           gate_panel_sha256=hashlib.sha256(text.encode()).hexdigest())
            (output / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary))
            return 0
        gate_panel = load_gate_panel(args.gate_panel, val) if args.gate_panel else None
        from .policy import CavPolicy, RsuGroupEncoder
        from .obs import build_observations
        from .safety import safe_accel
        from .counterfactual import CounterfactualConfig
        settings = policy_settings(raw)
        policy = CavPolicy(accel_max=settings["accel_max"],
                           lane_head=False if args.native_init else settings.get("lane_control") != "native",
                           init_zero_accel=False if args.native_init else settings.get("init_zero_accel", False),
                           residual_on_native=args.native_init)
        lane_section = dict(raw["lanes"])
        enabled = lane_section.pop("enabled", True)
        lane_rules = LaneRules.from_mapping(lane_section) if enabled else None
        if args.native_init:
            lane_rules = None
        # LOOP14C ramp merges (default off = legacy forced 6 m rule for every CAV).
        ramp_merge = {"policy_ramp_merge": switches.get("policy_ramp_merge", False),
                       "native_ramp_merge": switches.get("native_ramp_merge", "gap6")}
        if not isinstance(switches.get("native_lane_change", False), bool):
            raise ValueError("policy14.native_lane_change must be a boolean")
        if switches.get("native_lane_change", False):
            ramp_merge["native_lane_change"] = switches["native_lane_change"]
        if args.native_init:
            if not raw["counterfactual"].get("enabled"):
                raise ValueError("native-init requires paired native counterfactuals")
            if raw["counterfactual"].get("native_cav", "idm") not in ("idm", "head"):
                raise ValueError("native-init requires the native head or IDM reference")
            ramp_merge.update(policy_ramp_merge=False, native_lane_change=True, native_ramp_merge="wm_law")
        factory = make_controller_factory(**ramp_merge)
        if ramp_merge["policy_ramp_merge"] and lane_rules is None:
            raise ValueError("policy14.policy_ramp_merge needs [lanes] enabled")
        if (ramp_merge["native_ramp_merge"] == "wm_law" or ramp_merge.get("native_lane_change")) and any(
                getattr(m, "merge_pivot", None) is None for m in models):
            raise ValueError("policy14.native_ramp_merge='wm_law' needs first-passage WM members")
        # S-COOPW [uncertainty] / [exploration] / [horizon] (absent = unchanged; COOPW.md).
        coopw = {}
        from . import coopw as coopw_module
        for section, cls in (("uncertainty", "UncertaintyConfig"), ("exploration", "ExplorationConfig"),
                             ("horizon", "HorizonConfig")):
            if section in raw:
                coopw[section] = getattr(coopw_module, cls)(**raw[section])
                if section == "horizon":
                    summary["horizon_gate"] = coopw[section].bind(
                        [r["checkpoint_sha256"] for r in receipts])
                if coopw[section].enabled:
                    summary[section] = coopw[section].as_dict()
        # 13E's second tier is explicitly opt-in for policy CAVs, independently
        # of the WM's native lane physics. CAVCUTIN uses inferred follower b.
        trainer = Policy14Trainer(models, policy, GroupSignal(RsuGroupEncoder()),
                                  ScaledTeamCritic(entry_backlog=True,
                                      value_scale=float(switches["critic_value_scale"])), train,
                                  observation_builder=build_observations,
                                  reward_fn=authoritative_reward, safety=safe_accel,
                                  controller_factory=factory, config=cfg,
                                  reward_weights=RewardWeights(**raw["reward"]),
                                  device=args.device, output_dir=output, lane_rules=lane_rules,
                                  counterfactual=CounterfactualConfig(**raw["counterfactual"]),
                                  real_return_provider=provider,
                                  **coopw)
        trainer.marginal_lane_weight = float(switches["sampled_marginal_lane_weight"])
        from .paircrit import PairCriticConfig, BreakdownConfig, install
        pair_cfg = PairCriticConfig(**raw.get('paircrit', {}))
        risk_cfg = BreakdownConfig(**raw.get('breakdown', {}))
        if pair_cfg.enabled:
            pair_cfg.policy_returns = args.real_returns
            pair_cfg.native_returns = args.longcritic_native_returns or pair_cfg.native_returns
            pair_cfg.pairing_receipt = args.paircrit_receipt or pair_cfg.pairing_receipt
            pair_cfg.policy_sha256 = args.paircrit_policy_sha256 or pair_cfg.policy_sha256
            pair_cfg.reliability_file = args.longcritic_reliability or pair_cfg.reliability_file
        if pair_cfg.enabled and share_cfg.enabled:
            raise ValueError('paired regression replaces absolute real-share selection; set real_share.enabled=false')
        defer_install = False
        if args.resume and (pair_cfg.enabled or risk_cfg.enabled):
            prior = torch.load(checked_path(args.resume), map_location='cpu', weights_only=True)
            defer_install = 'paircrit' not in prior and 'breakdown' not in prior
            del prior
        if not defer_install:
            install(trainer, pair_cfg, risk_cfg, [r['checkpoint_sha256'] for r in receipts])
            if pair_cfg.enabled and not args.smoke:
                trainer.start_states = exclude_development_starts(
                    trainer.start_states, trainer.paircrit.provider.development_seeds)
        if args.native_init and getattr(trainer.exploration, 'long_persistence', False):
            policy.residual_persistent_exploration = True
        from .longcritic import LongCritic, LongCriticConfig
        long_cfg = LongCriticConfig(**raw.get("longcritic", {}))
        if long_cfg.enabled and not pair_cfg.enabled:
            long_cfg.reliability_file = args.longcritic_reliability or long_cfg.reliability_file
            long_cfg.native_returns = args.longcritic_native_returns or long_cfg.native_returns
            if not long_cfg.native_returns:
                raise ValueError("longcritic requires a separate real native-return anchor")
            native_provider = NPZRealReturnProvider(long_cfg.native_returns, seed=args.seed)
            if share_cfg.enabled:
                native_provider, _ = split_provider(native_provider)
            trainer.longcritic = LongCritic(trainer, long_cfg, native_provider,
                [r["checkpoint_sha256"] for r in receipts])
        if args.real_credit:
            from .real_credit14 import load_multipliers
            trainer.credit_multipliers = load_multipliers(checked_path(args.real_credit))
            summary["real_credit"] = {"file": str(args.real_credit),
                                      "sha256": hashlib.sha256(checked_path(args.real_credit).read_bytes()).hexdigest(),
                                      "multipliers": dict(trainer.credit_multipliers)}
        resumed = None
        if args.resume:
            resume = checked_path(args.resume)
            # Same ensemble size, possibly a refreshed (accepted) WM: dreamer.load_checkpoint's
            # loop-refresh path restores policy/critic/optimisers/duals and resets best scores.
            os.environ["WMS_LOOP_REFRESH"] = "0" if recovery else "1"
            trainer.load_checkpoint(resume)
            if recovery and trainer.update_count != recovery['saved_update']:
                raise ValueError('operational recovery checkpoint update mismatch')
            resumed = {"checkpoint": str(resume), "update_count": trainer.update_count,
                       "sha256": hashlib.sha256(resume.read_bytes()).hexdigest()}
        if defer_install:
            # Explicit boundary migration: restore the accepted actor first,
            # then reset only the newly introduced value/feature optimizers.
            install(trainer, pair_cfg, risk_cfg, [r['checkpoint_sha256'] for r in receipts])
            resumed['paircrit_boundary_reset'] = True
        if pair_cfg.enabled and not args.smoke:
            trainer.start_states = exclude_development_starts(
                trainer.start_states, trainer.paircrit.provider.development_seeds)
        if risk_cfg.enabled:
            if not args.native_init or not trainer.breakdown_receipt.get('calibration_qualified', False):
                raise ValueError('breakdown deployment requires native-init and an independently qualified calibration')
            # Fresh boundary calibration wins over a prior checkpoint's frozen head.
            trainer.group_encoder.risk.load_state_dict(trainer.breakdown_receipt['head'])
            trainer.critic.risk.load_state_dict(trainer.breakdown_receipt['head'])
        warm_updates = (int(switches["critic_warmup_updates"]) if args.critic_warmup_updates is None
                        else int(args.critic_warmup_updates))
        if warm_updates < 0:
            raise ValueError("critic warm-up updates must be >= 0")
        if args.looplong_boundary and prepared['flags'].get('long_credit'):
            trainer.long_credit_requested = True
            warmup = {'updates': 0, 'reason': 'looplong critic OFF; direct real-pair share 1.0'}
        else:
            warmup = (recovery['prior']['critic_warmup'] if recovery else
                      trainer.paircrit.warmup(trainer, pair_cfg.warmup_updates) if pair_cfg.enabled
                      else warm_start_real_critic(trainer, updates=warm_updates))
        if trainer.longcritic is not None and not getattr(trainer, 'long_credit_requested', False):
            if not trainer.longcritic.warm:
                summary["longcritic_native_warmup"] = trainer.longcritic.warmup(trainer)
            summary["longcritic"] = trainer.longcritic.receipt
        if recovery:
            if 'real_share' in recovery['prior']:
                summary['real_share'] = recovery['prior']['real_share']
                if trainer.config.real_return_mix != summary['real_share']['selected_share']:
                    raise ValueError('operational recovery fixed share mismatch')
            elif share_cfg.enabled:
                raise ValueError('operational recovery missing enabled share selection')
            if recovery.get('checkpoint_every_update'):
                trainer.config = replace(trainer.config, checkpoint_every=1)
            summary['operational_recovery'] = {k: v for k, v in recovery.items()
                                               if k not in ('prior', 'curve', 'updates')}
        elif share_cfg.enabled and not getattr(trainer, 'long_credit_requested', False):
            if args.smoke:
                raise ValueError('real-share selection requires real TRAIN starts')
            summary['real_share'] = calibrate_share(trainer, provider, share_dev,
                policy_sha256=resumed['sha256'] if resumed else None, config=share_cfg)
            (output / 'REAL_SHARE.json').write_text(json.dumps(summary['real_share'], indent=2), encoding='utf-8')
        if ramp_merge != {"policy_ramp_merge": False, "native_ramp_merge": "gap6"}:
            summary["ramp_merge"] = ramp_merge
        summary.update(seed=args.seed, smoke=args.smoke, device=args.device,
                       wm_members=receipts, probes=probes, probe_device=probe_device,
                       critic_warmup=warmup, resumed_from=resumed)
        if args.native_init:
            summary["native_actor_training"] = {
                "group": trainer.config.native_residual_train_group,
                "trunk": trainer.config.native_residual_train_trunk,
                "per_module_lr": trainer.config.actor_lr,
                "trainable_parameters": sum(p.numel() for p in trainer.actor_parameters),
                "migration": getattr(trainer, "native_training_migration", None)}
        # Independent development starts and paired draws, scored without a critic tail.
        rescore_batch = val.sample(cfg.batch_size, device=args.device)
        before = independent_score(trainer, rescore_batch)
        initial_score = before
        updates = list(recovery['updates']) if recovery else []
        summary.update(updates=updates, independent_score_before=initial_score)
        native_curve = list(recovery['curve']) if recovery else []
        native_start_update = recovery['phase_start'] if recovery else trainer.update_count
        if args.native_init:
            initial_native = score_native_panel(trainer, val, gate_panel, args.device)
            if not recovery:
                native_curve.append(initial_native)
            summary["native_curve"] = native_curve
            if not args.resume and any(d != 0.0 for d in initial_native["delta_per_start"]):
                raise ValueError("update-zero native identity failed on frozen panel")
            if args.resume and initial_native["passed"]:
                trainer.save_checkpoint(output / "policy_native_best.pt")
            if recovery:
                # Keep the original eligible best byte-for-byte until a better
                # measured checkpoint appears. Re-scoring still applies the veto.
                import shutil
                shutil.copyfile(recovery.get('best_checkpoint', args.resume), output / 'policy_native_best.pt')
                if not initial_native['passed']:
                    raise RuntimeError('native-relative recovery checkpoint veto')
        if gate_panel is not None:
            panel_before = score_gate_panel(trainer, val, gate_panel, args.device)
            summary["gate_panel"] = {"file": str(args.gate_panel), "n": len(panel_before),
                                     "sha256": hashlib.sha256(checked_path(args.gate_panel).read_bytes()).hexdigest(),
                                      "J_before": panel_before}
        if recovery:
            summary['gate_panel'] = recovery['prior']['gate_panel']
            # A saved checkpoint can precede the original phase-update10 gate.
            # Replaying that gate must still compare against the original phase start.
            panel_before = list(summary['gate_panel']['J_before'])
            (output / 'gate_panel_result.json').write_text(json.dumps(summary['gate_panel'], indent=1))
            (output / 'native_curve.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in native_curve))
        long_boundary = None
        if args.looplong_boundary:
            from .looplong import read_boundary, configure
            long_boundary = read_boundary(args.looplong_boundary)
            if [r['checkpoint_sha256'] for r in receipts] != [long_boundary['wm']['sha256']]:
                raise ValueError('looplong loaded WM mismatch')
            raw['_looplong_physics'] = receipts[0]['physics']
            if recovery:
                from .policy14_recovery import restore_looplong
                summary['looplong_recovery'] = restore_looplong(trainer, long_boundary, recovery, output)
            else:
                configure(trainer, long_boundary, raw, output)
            summary['looplong_initial_boundary'] = long_boundary
            cqr = getattr(trainer, 'long_credit', None)
            summary['looplong_long_credit'] = (
                getattr(trainer, 'long_credit_refusal', None) if cqr is None else
                {'artifact_sha256': cqr.get('artifact_sha256'),
                 'horizons': {h: {k: m.get(k) for k in ('qualified', 'correction_scale', 'rank', 'validation')}
                              for h, m in cqr['horizons'].items()}})
            (output / 'LOOPLONG_CREDIT.json').write_text(
                json.dumps(summary['looplong_long_credit'], indent=1, default=str), encoding='utf-8')
        elif args.wm_swap_file:
            raise ValueError('hot swap requires looplong boundary')
        if raw.get('data', {}).get('focus_cell_fraction') is not None:
            summary['start_focus'] = trainer.start_states.configure_focus_cells(
                float(raw['data']['focus_cell_fraction']))
            # MAIN's exact cell mixture supersedes the inherited rejection tilt.
            trainer._tilt_sampler = None
        summary['effective_config_before_updates'] = asdict(trainer.config)
        for _ in range(args.prep_dry_run_updates or args.max_updates):
            if (output / "STOP").exists():
                trainer.save_checkpoint(output / "policy_latest.pt")
                raise RuntimeError("own-run STOP requested; checkpoint saved")
            if long_boundary and args.wm_swap_file:
                from .looplong import maybe_swap
                swapped = maybe_swap(trainer, args.wm_swap_file, long_boundary, raw, output, load_members)
                if swapped:
                    long_boundary, receipts, event = swapped
                    models = trainer.world_models
                    # Scores from different WMs cannot compete for selection.
                    # Rescore the same frozen panel and restart the curve on the
                    # new WM; all actor/optimizer/phase counters are preserved.
                    native_curve.clear()
                    initial_native = score_native_panel(trainer, val, gate_panel, args.device)
                    native_curve.append(initial_native)
                    if initial_native['passed']:
                        trainer.save_checkpoint(output / 'policy_native_best.pt')
                    if gate_panel is not None:
                        panel_before = score_gate_panel(trainer, val, gate_panel, args.device)
                    before = independent_score(trainer, rescore_batch)
                    initial_score = before
                    summary.setdefault('wm_swaps', []).append(event)
            metrics = trainer.update()
            if recovery and recovery.get('checkpoint_every_update') and args.device.startswith('cuda'):
                metrics['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                metrics['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                with (output / 'cuda_peaks.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({k: metrics[k] for k in (
                        'update', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')}) + '\n')
            if not all(np.isfinite(metrics[k]) for k in ("actor_loss", "critic_loss", "actor_grad_norm", "critic_grad_norm")):
                raise ValueError("nonfinite update")
            if switches["independent_rescore"]:
                after = independent_score(trainer, rescore_batch)
                metrics["independent_J"] = after
                metrics["independent_J_decreased"] = after["J"] < before["J"]
                # Anneal only against observed, independently rescored unshaped
                # improvement, never merely because optimisation time elapsed.
                if metrics["independent_J_decreased"]:
                    trainer.config = replace(trainer.config, entropy_weight=max(
                        float(switches["entropy_floor"]), trainer.config.entropy_weight
                        * float(switches["entropy_decay_on_improvement"])))
                before = after
                with (output / "rescores.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"update": trainer.update_count, **after}, allow_nan=False) + "\n")
            updates.append(metrics)
            # Durable per-update telemetry (module gradients, residual size, memory)
            # so a long phase can be audited before RUN_SUMMARY exists.
            if args.device.startswith('cuda') and torch.cuda.is_available():
                metrics.setdefault('cuda_peak_allocated_bytes', torch.cuda.max_memory_allocated())
                metrics.setdefault('cuda_peak_reserved_bytes', torch.cuda.max_memory_reserved())
            with (output / "updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"t": __import__("time").time(), **metrics}, default=str) + "\n")
            if args.native_init and trainer.update_count % 10 == 0:
                native_result = score_native_panel(trainer, val, gate_panel, args.device)
                native_curve.append(native_result)
                with (output / "native_curve.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(native_result, allow_nan=False) + "\n")
                from .nativeinit import select_update
                if select_update(native_curve) == trainer.update_count:
                    trainer.save_checkpoint(output / "policy_native_best.pt")
            gate_failed = False
            gate_position = trainer.update_count - native_start_update if args.phase_relative_signal_gate else trainer.update_count
            if not args.smoke and gate_position == int(switches["signal_gate_updates"]):
                if gate_panel is not None:
                    panel_after = score_gate_panel(trainer, val, gate_panel, args.device)
                    deltas = [a - b for a, b in zip(panel_after, panel_before)]
                    mean_delta = sum(deltas) / len(deltas)
                    gate_failed = mean_delta >= 0.0
                    summary["gate_panel"].update(
                        J_after=panel_after, delta_per_start=deltas, mean_J_before=sum(panel_before) / len(deltas),
                        mean_J_after=sum(panel_after) / len(deltas), mean_delta=mean_delta,
                         n_improved=sum(d < 0 for d in deltas), update=trainer.update_count,
                        phase_update=gate_position,
                        passed=not gate_failed)
                    (output / "gate_panel_result.json").write_text(
                        json.dumps(summary["gate_panel"], indent=1, allow_nan=False), encoding="utf-8")
                else:
                    gate_failed = before["J"] >= initial_score["J"]
            if gate_failed:
                trainer.save_checkpoint(output / "policy_latest.pt")
                raise RuntimeError("early signal gate: independently rescored J did not decrease; stop before scaling")
        if args.prep_dry_run_updates:
            # Scratch updates are not eligible for phase selection or deployment.
            # Keep initial scoring/probes and the actual actor update unchanged.
            summary.update(status='completed_dry_run', updates=updates,
                           dry_run_updates=args.prep_dry_run_updates,
                           config=asdict(trainer.config), deployment_authorized=False,
                           wm_members=receipts, resumed_from=resumed)
            (output / 'RUN_SUMMARY.json').write_text(
                json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
            print(json.dumps({'status': summary['status'], 'summary_path': str(output / 'RUN_SUMMARY.json')}))
            return 0
        trainer.save_checkpoint(output / "policy_latest.pt")
        if long_boundary:
            summary['looplong_final_boundary'] = long_boundary
        if args.native_init:
            from .nativeinit import select_update
            import shutil
            chosen = select_update(native_curve)
            if chosen is None:
                raise RuntimeError("no trained checkpoint is <= native on all16 starts; keep native")
            shutil.copyfile(output / "policy_native_best.pt", output / "policy_latest.pt")
            selected = next(r for r in native_curve if r["update"] == chosen)
            receipt = {**selected, "checkpoint_sha256": hashlib.sha256(
                (output / "policy_latest.pt").read_bytes()).hexdigest(),
                "wm_sha256": [r["checkpoint_sha256"] for r in receipts],
                "wm_config_sha256": [r["config_sha256"] for r in receipts],
                "selection": "earliest minimum eligible measured mean delta", "measured_through": trainer.update_count}
            if risk_cfg.enabled:
                from .paircrit import breakdown_gate, load_risk
                # Keep the feature head frozen inside the actor. A separate,
                # external gate head can be qualified against the exact selected
                # candidate without changing its weights/hash or creating a
                # checkpoint -> calibration -> checkpoint hash dependency cycle.
                if not risk_cfg.gate_checkpoint:
                    raise RuntimeError('candidate-specific external risk calibration required')
                gate_head, gate_risk = load_risk(risk_cfg.gate_checkpoint, receipt['checkpoint_sha256'])
                if (not gate_risk.get('calibration_qualified')
                        or gate_risk['rule_sha256'] != trainer.breakdown_receipt['rule_sha256']):
                    raise RuntimeError('unqualified or different-rule candidate risk gate')
                gate_head = gate_head.to(args.device)
                risk_results = [breakdown_gate(gate_head,
                    val._batch([val._load(int(s['index']))], args.device).state)
                    for s in gate_panel['starts']]
                risk_result = {'horizons_s': [60, 120, 300], 'margin': 0.,
                    'policy_minus_native': [r['policy_minus_native'][0] for r in risk_results],
                    'passed': all(r['passed'] for r in risk_results), 'predictive_only': True,
                    'calibration_qualified': gate_risk['calibration_qualified'],
                    'rule_sha256': gate_risk['rule_sha256'],
                    'source_policy_sha256': gate_risk['policy_checkpoint_sha256']}
                receipt['breakdown'] = risk_result
                receipt['passed'] = receipt['passed'] and risk_result['passed']
                if not receipt['passed']:
                    raise RuntimeError('breakdown native-relative veto; keep native')
            summary["recommended_updates_per_cycle"] = chosen - native_start_update
            receipt.update(phase_start_update=native_start_update,
                           selected_phase_updates=chosen-native_start_update,
                           panel_horizon_s=trainer.config.horizon)
            (output / "native_gate_result.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
        if trainer.paircrit is not None:
            summary['paircrit_monitor'] = trainer.paircrit.diagnostics(trainer)
        summary.update(status="completed", seed=args.seed, smoke=args.smoke, device=args.device,
                       updates=updates, critic_warmup=warmup, config=asdict(trainer.config),
                       wm_members=receipts, probes=probes,
                       real_returns_file=str(provider.path), real_return_source=provider.provenance["source_kind"],
                       real_return_targets=(int(provider.indices.size) if hasattr(provider, 'indices')
                                            else int(provider._returns.numel())),
                       independent_score_before=initial_score, independent_score=before,
                       actor_from_scratch=resumed is None, resumed_from=resumed,
                       ensemble_epistemic_available=len(models) > 1,
                       deployment_calibration_supported=False,
                       lane_mask_hook=lane_rules.secure_gap_hook if lane_rules else None,
                       consequence_mask=lane_rules.consequence_mask if lane_rules else False,
                       synthetic_start_states=args.smoke)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError) as error:
        import traceback
        summary['traceback'] = traceback.format_exc()
        summary["error"] = f"{type(error).__name__}: {error}"
    if output is not None:
        (output / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    printed = ({"status": summary["status"], "summary_path": str(output / "RUN_SUMMARY.json") if output else None,
                **({"error": summary["error"]} if "error" in summary else {})}
               if args.compact else summary)
    print(json.dumps(printed, allow_nan=False))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
