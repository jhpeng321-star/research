"""Train C4 inside frozen WM-S: ``python -m wms.control.train_cli --config ...``.

Control modules are imported lazily; no dependency on SUMO. Paths are restricted
to this worktree, including resolved checkpoint, config and data paths.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import torch

from .contract import PHYSICS_DT_S, STEPS_PER_CONTROL, RewardWeights
from .dreamer import DreamerConfig, DreamerTrainer
from .start_states import checked_path, split_start_states


def _read_config(path, seen=()):
    """Validate every inheritance path before reading it."""
    from wms.core.toml import load
    from wms.config.schema import deep_merge

    path = checked_path(path)
    # Same depth cap as wms.config.load (16).  POLICYV5: was 8, which refused c4_wm_v5.toml (9 levels).
    if path in seen or len(seen) >= 16:
        raise ValueError("configuration inheritance cycle/depth exceeded")
    raw = load(path)
    parent = raw.pop("extends", None)
    return deep_merge(_read_config(path.parent / parent, (*seen, path)), raw) if parent else raw


class GroupSignal(torch.nn.Module):
    """Adapter: the trainer calls ``group_encoder(obs, state)``; the RSU encoder also needs the
    CAV positions and the RSU own-sensor field computed from the same state."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, obs, state_or_positions, rsu_field=None):
        from .rsu_field import build_rsu_field
        if isinstance(state_or_positions, dict):       # trainer's internal path
            state = state_or_positions
            return self.encoder(obs, state["s_m"], build_rsu_field(state))
        return self.encoder(obs, state_or_positions, rsu_field)   # bridge path


def policy_settings(raw) -> dict:
    """``[policy]`` section -> CavPolicy keyword arguments (ACCELCAP_NOTES.md, V6EXPLORE).

    ``accel_max_mps2`` = the actor's action cap; absent -> the fresh-actor default
    ``POLICY_ACCEL_MAX_MPS2`` (2.6).  A checkpoint loaded with --init-policy / --resume keeps its
    own stored bound (legacy 1.5 when it has none) and the trainer warns about the override.
    ``init_zero_accel`` (V6EXPLORE, default off) applies the neutral zero-mean acceleration
    start to the fresh actor; the key is returned only when enabled so existing digests of
    default configs stay unchanged."""
    from .contract import DIRECT_ACCEL_MAX_MPS2, POLICY_ACCEL_MAX_MPS2

    section = dict(raw.get("policy", {}))
    unknown = set(section) - {"accel_max_mps2", "init_zero_accel"}
    if unknown:
        raise ValueError(f"unknown [policy] keys: {sorted(unknown)}")
    accel_max = float(section.get("accel_max_mps2", POLICY_ACCEL_MAX_MPS2))
    if not 0.0 < accel_max <= DIRECT_ACCEL_MAX_MPS2:
        raise ValueError(f"[policy] accel_max_mps2 must be in (0, {DIRECT_ACCEL_MAX_MPS2}] "
                         "(the SUMO env's direct-command ceiling)")
    out = {"accel_max": accel_max}
    zero = section.get("init_zero_accel", False)
    if not isinstance(zero, bool):
        raise ValueError("[policy] init_zero_accel must be a boolean")
    if zero:
        out["init_zero_accel"] = True
    return out


def _load_components():
    names = {"obs": "build_observations", "reward": "team_reward", "policy": "CavPolicy",
             "safety": "safe_accel"}
    modules = {key: importlib.import_module(f"wms.control.{key}") for key in names}
    policy_module = modules["policy"]
    controller_factory = importlib.import_module("wms.control.bridge").make_cav_controller
    return {"observation_builder": modules["obs"].build_observations,
            "reward_fn": modules["reward"].team_reward, "safety": modules["safety"].safe_accel,
            "policy": policy_module.CavPolicy(),
            "group_encoder": GroupSignal(policy_module.RsuGroupEncoder()),
            "critic": policy_module.TeamCritic(), "controller_factory": controller_factory}


def _load_world_models(paths, device):
    load_backend = importlib.import_module("wms.eval.backend").load_backend

    models = []
    for path in paths:
        path = checked_path(path)
        # Validate the adjacent files too, including possible symlink targets.
        for sibling in ("physics.json", "TRAIN_RECEIPT.json"):
            checked_path(path.with_name(sibling))
        backend = load_backend(path, device=device)
        models.append(backend.model)
    return models


def _load_starts(data, models, seed, entries=False):
    from wms.config.train import TrainConfig
    from wms.config.schema import from_mapping
    TrajectoryDataset = importlib.import_module("wms.data.dataset").TrajectoryDataset
    resolve = importlib.import_module("wms.data.registry").resolve
    configure_train_sidecars = importlib.import_module("wms.data.sidecars").configure_train_sidecars

    source_config = checked_path(data.get("wm_config", "configs/r5b.toml"))
    source = from_mapping(TrainConfig, _read_config(source_config))
    spec = source.data
    # Deliberately construct only the original TRAIN corpus, not evaluation or switching sources.
    def path(name):
        return str(checked_path(resolve(name)))

    history_specs = {(getattr(model, "history_s", None), getattr(model, "history_stride_s", 1.0))
                     for model in models}
    if len(history_specs) != 1:
        raise ValueError("ensemble members must use the same observed-history specification")
    history_s, history_stride_s = history_specs.pop()
    split_kwargs = dict(
        val_fraction=float(data.get("val_fraction", 0.1)), seed=seed,
        excluded_penetrations=tuple(data.get("excluded_penetrations", [])),
        relabel=bool(data.get("relabel_cavs", False)),
        target_penetration=float(data.get("target_penetration", 0.2)))
    if getattr(spec, "c4_base", None) is not None:
        # POLICYV5 (V5DATA WM config): TRAIN-world windows of the C4 sources, no packs, no jam
        # scan (wms.control.c4_starts).  Legacy configs (c4_base unset) take the pack path below.
        from .c4_starts import build_c4_start_windows
        sources = data.get("start_sources")
        dataset = build_c4_start_windows(
            spec, path, history_s=history_s, history_stride_s=history_stride_s,
            entries=bool(entries), sources=None if sources is None else list(sources))
        weights = data.get("start_source_weights")
        if weights is not None:
            # C4 Dreamer loop: e.g. {v6_policy = 0.5, native_v5 = 0.25, genrand_plain = 0.25} so the
            # policy dreams from the states it actually visits in SUMO (default: uniform windows).
            selected = set(dataset.receipt["sources"])
            weights = {str(k): float(v) for k, v in dict(weights).items()}
            if set(weights) != selected or sum(weights.values()) <= 0.0:
                raise ValueError(f"[data] start_source_weights {sorted(weights)} must give one "
                                 f"weight per start source {sorted(selected)} (positive total)")
            split_kwargs["source_weights"] = weights
        return split_start_states(dataset, **split_kwargs)
    if data.get("start_sources") is not None or data.get("start_source_weights") is not None:
        raise ValueError("[data] start_sources needs a WM config with data.c4_base")

    roles = list(spec.packs)
    packs = [path(spec.packs[role]) for role in roles]
    identities = [path(spec.sidecars.identities[role]) for role in roles]
    overlays = [path(spec.sidecars.overlays[role]) for role in roles[1:]
                if role in spec.sidecars.overlays] or None
    windows = path(spec.windows.train) if spec.windows.train else None
    if not windows:
        raise ValueError("an explicit TRAIN window list is required")
    exclude = [path(name) for name in spec.windows.exclude] or None
    configure_train_sidecars(outcomes=path(spec.sidecars.outcomes) if spec.sidecars.outcomes else None,
                            identities=identities[0])
    dataset = TrajectoryDataset.from_real_packs(
        packs, identity_dirs=identities, overlays=overlays, horizon_steps=1,
        window_list=windows, exclude_window_list=exclude, start_shift_prob=0.0,
        history_s=history_s, history_stride_s=history_stride_s,
        identity_env_dir=identities[0],
        # Observed pre-t0 entry history per window (t <= t0 only) for imagined entries.
        entries_v2=bool(entries))
    train, val = split_start_states(dataset, **split_kwargs)
    return train, val


def _windows_by_source(sampler) -> dict:
    counts = {}
    for group in sampler.groups.values():
        for index in group:
            name = sampler.metadata[index]["start_source"]
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def parser():
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--config", default="configs/c4_policy.toml")
    out.add_argument("--device", choices=("cpu", "cuda"), default=None)
    out.add_argument("--max-updates", type=int)
    out.add_argument("--wm-ckpt", nargs="+")
    out.add_argument("--init-policy", help="initialise actor/group weights from a C4 checkpoint")
    out.add_argument("--resume", help="restore full trainer, optimisers, dual and RNG")
    out.add_argument("--output-dir")
    out.add_argument("--smoke", action="store_true", help="2 updates, batch 2, H 4, CPU")
    return out


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.perf_counter()
    output = None
    summary = {"status": "failed", "updates": 0}
    try:
        raw = _read_config(args.config)
        allowed = {"trainer", "run", "data", "world_model", "reward", "lanes", "counterfactual", "policy"}
        if set(raw) - allowed:
            raise ValueError(f"unknown C4 config sections: {sorted(set(raw) - allowed)}")
        run = raw.get("run", {})
        output = checked_path(args.output_dir or run.get("output_dir", "runs/active/C4-DREAMER"))
        output.mkdir(parents=True, exist_ok=True)
        cfg = DreamerConfig(**raw.get("trainer", {}))
        from .counterfactual import CounterfactualConfig
        from .lanes import LaneRules
        lane_section = dict(raw.get("lanes", {}))
        use_lanes = bool(lane_section.pop("enabled", True))
        lane_rules = LaneRules.from_mapping(lane_section) if use_lanes else None
        counterfactual = CounterfactualConfig(**raw.get("counterfactual", {}))
        device = args.device or run.get("device", "cpu")
        max_updates = args.max_updates if args.max_updates is not None else int(run.get("updates", 3000))
        if args.smoke:
            device, max_updates = "cpu", 2
            cfg = replace(cfg, batch_size=2, horizon=4, checkpoint_every=1, validation_every=1,
                          credit_profile_every=1 if cfg.credit_profile_every else cfg.credit_profile_every)
        if max_updates < 1:
            raise ValueError("max-updates must be positive")
        if args.init_policy and args.resume:
            raise ValueError("choose either --init-policy or --resume")
        seed = int(run.get("seed", 0))
        threads = int(run.get("threads", 1))
        if threads < 1:
            raise ValueError("threads must be positive")
        if args.smoke:
            threads = min(threads, 2)
        torch.set_num_threads(threads)
        # Shared-GPU guard: cap this process's CUDA memory so it OOMs itself before starving a
        # co-located world-model training run (env C4_GPU_MEM_FRACTION, e.g. 0.4).
        frac = os.environ.get("C4_GPU_MEM_FRACTION")
        if frac and str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(frac))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        paths = args.wm_ckpt or raw.get("world_model", {}).get("checkpoints", [])
        if not paths:
            raise ValueError("supply at least one local frozen WM checkpoint via --wm-ckpt")
        paths = [str(checked_path(path)) for path in paths]
        policy_kwargs = policy_settings(raw)
        components = _load_components()
        if hasattr(components["policy"], "set_accel_bounds"):      # fresh actor: configured cap
            components["policy"].set_accel_bounds(components["policy"].accel_min,
                                                  policy_kwargs["accel_max"])
        if policy_kwargs.get("init_zero_accel", False):            # V6EXPLORE neutral start
            if not hasattr(components["policy"], "init_zero_mean_accel"):
                raise ValueError("[policy] init_zero_accel requires a CavPolicy actor")
            components["policy"].init_zero_mean_accel()
        if cfg.critic_entry_backlog:     # SRCGRAD_NOTES.md: per-entrance backlog critic input
            components["critic"] = importlib.import_module("wms.control.policy").TeamCritic(
                entry_backlog=True)
        models = _load_world_models(paths, device)
        train, val = _load_starts(raw.get("data", {}), models, seed,
                                  entries=cfg.imagine_entries)
        trainer = DreamerTrainer(models, components.pop("policy"), components.pop("group_encoder"),
                                 components.pop("critic"), train, **components, config=cfg,
                                 reward_weights=RewardWeights(**raw.get("reward", {})),
                                 device=device, output_dir=output, lane_rules=lane_rules,
                                 counterfactual=counterfactual)
        if args.init_policy:
            trainer.load_checkpoint(args.init_policy, policy_only=True)
        if args.resume:
            trainer.load_checkpoint(args.resume)
            if args.smoke:
                trainer.config = replace(trainer.config, batch_size=2, horizon=4,
                                         checkpoint_every=1, validation_every=1,
                                         credit_profile_every=(1 if trainer.config.credit_profile_every
                                                               else trainer.config.credit_profile_every))
                max_updates = trainer.update_count + 2
        last = trainer.fit(max_updates, validation_states=val,
                           validation_batches=int(run.get("validation_batches", 1)))
        summary.update(status="completed", updates=trainer.update_count, device=device,
                       smoke=args.smoke, ensemble_checkpoints=paths, config=asdict(trainer.config),
                       lane_head=trainer.use_lanes, imagine_entries=trainer.config.imagine_entries,
                       lane_rules=asdict(lane_rules) if lane_rules is not None else None,
                       counterfactual=counterfactual.as_dict(),
                       policy_accel_bounds=[getattr(trainer.policy, "accel_min", None),
                                            getattr(trainer.policy, "accel_max", None)],
                       policy_init_zero_accel=bool(getattr(trainer.policy, "init_zero_accel",
                                                          False)),
                       physics_steps_per_control=STEPS_PER_CONTROL,
                       imagined_horizon_s=trainer.config.horizon * STEPS_PER_CONTROL * PHYSICS_DT_S,
                       train_windows=sum(map(len, train.groups.values())),
                       val_windows=sum(map(len, val.groups.values())),
                       penetrations=train.penetrations, last_update=last,
                       best_validation_return=(trainer.best_validation if
                                               np.isfinite(trainer.best_validation) else None),
                       best_coop_gain_val=(trainer.best_coop_validation if
                                           np.isfinite(trainer.best_coop_validation) else None))
        if hasattr(train.dataset, "start_meta"):     # POLICYV5: windows per C4 start source
            summary["start_windows_by_source"] = {
                role: _windows_by_source(sampler) for role, sampler in (("train", train),
                                                                        ("val", val))}
        code = 0
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        code = 1
    summary["wall_time_s"] = time.perf_counter() - started
    if output is not None:
        with (output / "RUN_SUMMARY.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
