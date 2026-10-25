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
        frac = os.environ.get("C4_GPU_MEM_FRACTION")
        if frac and args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(frac))
        candidate = checked_path(args.output_dir)
        candidate.mkdir(parents=True, exist_ok=False)
        output = candidate
        provider = NPZRealReturnProvider(args.real_returns, seed=args.seed)
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
        from .policy import CavPolicy, RsuGroupEncoder
        from .obs import build_observations
        from .safety import safe_accel
        from .counterfactual import CounterfactualConfig
        settings = policy_settings(raw)
        policy = CavPolicy(accel_max=settings["accel_max"],
                           lane_head=settings.get("lane_control") != "native",
                           init_zero_accel=settings.get("init_zero_accel", False))
        lane_section = dict(raw["lanes"])
        enabled = lane_section.pop("enabled", True)
        lane_rules = LaneRules.from_mapping(lane_section) if enabled else None
        # LOOP14C ramp merges (default off = legacy forced 6 m rule for every CAV).
        ramp_merge = {"policy_ramp_merge": switches.get("policy_ramp_merge", False),
                      "native_ramp_merge": switches.get("native_ramp_merge", "gap6")}
        factory = make_controller_factory(**ramp_merge)
        if ramp_merge["policy_ramp_merge"] and lane_rules is None:
            raise ValueError("policy14.policy_ramp_merge needs [lanes] enabled")
        if ramp_merge["native_ramp_merge"] == "wm_law" and any(
                getattr(m, "merge_pivot", None) is None for m in models):
            raise ValueError("policy14.native_ramp_merge='wm_law' needs first-passage WM members")
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
                                  real_return_provider=provider)
        trainer.marginal_lane_weight = float(switches["sampled_marginal_lane_weight"])
        resumed = None
        if args.resume:
            resume = checked_path(args.resume)
            # Same ensemble size, possibly a refreshed (accepted) WM: dreamer.load_checkpoint's
            # loop-refresh path restores policy/critic/optimisers/duals and resets best scores.
            os.environ["WMS_LOOP_REFRESH"] = "1"
            trainer.load_checkpoint(resume)
            resumed = {"checkpoint": str(resume), "update_count": trainer.update_count,
                       "sha256": hashlib.sha256(resume.read_bytes()).hexdigest()}
        warm_updates = (int(switches["critic_warmup_updates"]) if args.critic_warmup_updates is None
                        else int(args.critic_warmup_updates))
        if warm_updates < 0:
            raise ValueError("critic warm-up updates must be >= 0")
        warmup = warm_start_real_critic(trainer, updates=warm_updates)
        if ramp_merge != {"policy_ramp_merge": False, "native_ramp_merge": "gap6"}:
            summary["ramp_merge"] = ramp_merge
        summary.update(seed=args.seed, smoke=args.smoke, device=args.device,
                       wm_members=receipts, probes=probes, probe_device=probe_device,
                       critic_warmup=warmup, resumed_from=resumed)
        # Independent development starts and paired draws, scored without a critic tail.
        rescore_batch = val.sample(cfg.batch_size, device=args.device)
        before = independent_score(trainer, rescore_batch)
        initial_score = before
        updates = []
        summary.update(updates=updates, independent_score_before=initial_score)
        for _ in range(args.max_updates):
            if (output / "STOP").exists():
                trainer.save_checkpoint(output / "policy_latest.pt")
                raise RuntimeError("own-run STOP requested; checkpoint saved")
            metrics = trainer.update()
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
            if (not args.smoke and trainer.update_count == int(switches["signal_gate_updates"])
                    and before["J"] >= initial_score["J"]):
                trainer.save_checkpoint(output / "policy_latest.pt")
                raise RuntimeError("early signal gate: independently rescored J did not decrease; stop before scaling")
        trainer.save_checkpoint(output / "policy_latest.pt")
        summary.update(status="completed", seed=args.seed, smoke=args.smoke, device=args.device,
                       updates=updates, critic_warmup=warmup, config=asdict(cfg),
                       wm_members=receipts, probes=probes,
                       real_returns_file=str(provider.path), real_return_source=provider.provenance["source_kind"],
                       real_return_targets=int(provider._returns.numel()),
                       independent_score_before=initial_score, independent_score=before,
                       actor_from_scratch=resumed is None, resumed_from=resumed,
                       ensemble_epistemic_available=len(models) > 1,
                       deployment_calibration_supported=False,
                       lane_mask_hook=lane_rules.secure_gap_hook if lane_rules else None,
                       consequence_mask=lane_rules.consequence_mask if lane_rules else False,
                       synthetic_start_states=args.smoke)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
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
