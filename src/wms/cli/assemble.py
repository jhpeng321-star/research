"""Assemble the datasets and the model of a training run from a ``TrainConfig``.

This is the one place that turns the nested, logical-name configuration into the concrete
objects of the data and model layers (WP-I).  It reproduces the assembly of the legacy
``wms.cli`` train path call for call:

* every logical data name is resolved through :mod:`wms.data.registry` (never a literal path);
* the process-wide TRAIN sidecar hooks (``WM_N1_TRAIN_OUTCOMES`` / ``WM_N1_TRAIN_IDENTITIES``)
  are set exactly as the R3 chain exports them (outcomes dir; identities dir of the FIRST pack);
* train and validation windows come from ``TrajectoryDataset.from_real_packs`` with the
  window lists / exclusion lists, start-shift only on TRAIN, and the held-out / PI-wall item
  filter when configured;
* an optional switching/direct-output pack is mixed in (``mixdata``) or, without it, jam
  windows are over-sampled (``jam``);
* optional collected C4 SUMO episodes (``[data.c4_episodes]``, :mod:`wms.data.c4_episodes`)
  are mixed in last with their own sampling share (the Dreamer data loop);
* optional further C4 sources (``[[data.c4_sources]]``, PAIRLOSS: GENRAND plain / paired
  branches, :mod:`wms.data.c4_pairs`) with one split side per world across all C4 sources;
  with ``loss.paired_diff`` on, the TRAIN dataset is wrapped so a fraction of the batches are
  paired-branch batches;
* V5DATA ``data.c4_base``: no packs at all - the named C4 source is the base (share = 1 - the
  other sources' ratios) and validation is the C4 sources' held-out world split;
* the physics config and the model are built from the same flags the legacy ``_build_model``
  used (``ModelSpec.from_mapping(config.to_legacy_flat())``).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..config.train import TrainConfig
from ..core.errors import ConfigError

Resolver = Callable[[str], Path]
#: Physics step of the rollout; stage horizons are given in seconds.
SIM_DT_S = 0.5


@dataclass
class TrainingInputs:
    """Everything ``WmSTrainer`` needs, plus the resolved data paths for receipts."""

    train: Any
    val: Optional[Any]
    model: Any
    physics: Any
    resolved: dict[str, str] = field(default_factory=dict)


def default_resolver() -> Resolver:
    """Fail-closed resolver: the data must exist on disk."""
    from ..data.registry import require_present

    return require_present


def declared_data_names(cfg: TrainConfig) -> dict[str, str]:
    """``{label: logical name}`` of every dataset the configuration reads.

    Used by preflight (existence, restricted-token guard) and for the run receipt.
    """
    data = cfg.data
    names: dict[str, str] = {}
    for role, name in data.packs.items():
        names[f"pack[{role}]"] = name
    for role, name in data.sidecars.identities.items():
        names[f"identities[{role}]"] = name
    for role, name in data.sidecars.overlays.items():
        names[f"overlay[{role}]"] = name
    for label, name in (("outcomes", data.sidecars.outcomes),
                        ("driver_sidecar", data.sidecars.driver_sidecar),
                        ("windows.train", data.windows.train),
                        ("windows.val", data.windows.val),
                        ("physics_params", cfg.model.physics_params)):
        if name:
            names[label] = name
    for i, name in enumerate(data.windows.exclude):
        names[f"windows.exclude[{i}]"] = name
    sw = data.switching
    if sw.enabled:
        for label in ("pack", "identities", "driver_params", "windows_train", "windows_val",
                      "timeline_index", "zero_shot_heldout"):
            if label == "zero_shot_heldout" and sw.heldout_family_policy == "release":
                continue
            name = getattr(sw, label)
            if name:
                names[f"switching.{label}"] = name
    if data.c4_episodes.enabled:
        names["c4_episodes"] = data.c4_episodes.path
    for src in data.c4_sources:
        names[f"c4_sources[{src.name}]"] = src.path
    return names


def resolve_all(cfg: TrainConfig, resolver: Optional[Resolver] = None
                ) -> tuple[dict[str, Path], dict[str, str]]:
    """Resolve every declared name; returns ``(paths, errors)`` instead of raising."""
    from ..data.registry import resolve

    use = resolver or resolve
    paths: dict[str, Path] = {}
    errors: dict[str, str] = {}
    for label, name in declared_data_names(cfg).items():
        try:
            paths[label] = Path(use(name))
        except Exception as exc:  # noqa: BLE001 - reported per name by preflight
            errors[label] = f"{name}: {type(exc).__name__}: {exc}"
    return paths, errors


def build_physics(cfg: TrainConfig, resolver: Optional[Resolver] = None):
    """``PhysicsConfig`` from model flags and top-level integration options."""
    from ..physics.config import PhysicsConfig

    resolver = resolver or default_resolver()
    model = cfg.model
    sumo_idm = not model.geometry_only
    integration = {"stop_time_integration": cfg.physics.stop_time_integration,
                   "emergency_braking": cfg.physics.emergency_braking,
                   "secure_gap_veto": cfg.physics.secure_gap_veto,
                   "lane_change_secure_gap": bool(cfg.physics.lane_change_secure_gap),
                   "emergency_onset": str(cfg.physics.emergency_onset),
                   "emergency_decel_mps2": cfg.physics.emergency_decel_mps2,
                   "exit_at_road_end": bool(cfg.physics.exit_at_road_end),
                   "merge_end_wait": bool(cfg.physics.merge_end_wait),
                   "mainline_merge_yield": bool(cfg.physics.mainline_merge_yield),
                   "lane_learnable": bool(cfg.physics.lane_learnable),
                   "ramp_lane_end_stop": bool(cfg.physics.ramp_lane_end_stop),
                   "ramp_risk_retain": bool(cfg.physics.ramp_risk_retain),
                   "mainline_lane_mode": str(cfg.physics.mainline_lane_mode),
                   "mainline_courtesy_zone_m": float(cfg.physics.mainline_courtesy_zone_m)}
    # V8WM: config-level gap-rule / lane-end parameter overrides (empty = unchanged).
    integration.update({key: value if isinstance(value, (str, bool)) else float(value)
                        for key, value in cfg.physics.overrides.items()})
    if model.merge_cal:
        overrides = {"sumo_fidelity": bool(model.sumo_fidelity), "sumo_idm": sumo_idm,
                     "merge_cal": True, "lane_rule": model.lane_rule,
                     "lane_gap_rule": model.lane_gap_rule,
                     "gap_stopped_relax": bool(model.gap_stopped_relax), **integration}
        if model.physics_params:
            return PhysicsConfig.from_params_file(resolver(model.physics_params), **overrides)
        return PhysicsConfig(**overrides)
    return PhysicsConfig(sumo_fidelity=bool(model.sumo_fidelity), sumo_idm=sumo_idm,
                         lane_rule=model.lane_rule, lane_gap_rule=model.lane_gap_rule,
                         gap_stopped_relax=bool(model.gap_stopped_relax), **integration)


def build_model(cfg: TrainConfig, physics: Any):
    """The WM-S model (or a legacy black-box baseline) selected by ``cfg.model``."""
    from ..model.factory import build_model as factory
    from ..model.spec import ModelSpec

    spec = ModelSpec.from_mapping(cfg.to_legacy_flat())
    return factory(spec, physics, kind=cfg.model.kind)


def _horizon_steps(cfg: TrainConfig) -> int:
    return max(1, round(max(cfg.curriculum.unroll_s) / SIM_DT_S))


def _kept_summary(receipt: dict) -> str:
    kept, total = receipt.get("kept_by_pack", {}), receipt.get("total_by_pack", {})
    parts = ", ".join(f"{p} {kept.get(p, 0)}/{total.get(p, 0)}" for p in total)
    return (f"kept windows total {receipt.get('n_records')}/"
            f"{receipt.get('n_corpus_windows')} | {parts}")


def _junction_kwargs(data: Any) -> dict:
    """``junction_fix`` / ``controller_hygiene`` keywords for the pack datasets; empty when
    off (calls unchanged)."""
    out = {}
    mode = getattr(data, "junction_fix", "off")
    if mode != "off":
        out["junction_fix"] = mode
    hygiene = getattr(data, "controller_hygiene", "off")
    if hygiene != "off":
        out["controller_hygiene"] = hygiene
    return out


def build_datasets(cfg: TrainConfig, out_dir: Path, resolver: Optional[Resolver] = None
                   ) -> tuple[Any, Optional[Any], dict[str, str]]:
    """TRAIN dataset, validation dataset (or ``None``) and the resolved data paths."""
    from ..data import mixdata
    from ..data.dataset import TrajectoryDataset
    from ..data.sidecars import configure_train_sidecars
    from ..data.val_split import materialise_holdout
    from ..data.windows import ControllerExclusion

    resolver = resolver or default_resolver()
    data, resolved = cfg.data, {}

    def path(name: str) -> str:
        found = str(resolver(name))
        resolved[name] = found
        return found

    if data.c4_base is not None:
        # V5DATA: no packs / sidecars / window lists / switching pack; every TRAIN and
        # validation window comes from data.c4_sources (base source = data.c4_base).
        train, val = _add_c4_sources(cfg, None, None, path, horizon=_horizon_steps(cfg),
                                     pack_k=None, heldout=None,
                                     curriculum=tuple(cfg.curriculum.unroll_s))
        return train, val, resolved

    roles = list(data.packs)
    if not roles:
        raise ConfigError("data.packs is empty: at least one pack is required")
    packs = [path(data.packs[r]) for r in roles]
    identities = [path(data.sidecars.identities[r]) for r in roles]
    overlay_roles = [r for r in roles if r in data.sidecars.overlays]
    if overlay_roles and overlay_roles != roles[1:]:
        raise ConfigError("data.sidecars.overlays must name exactly the packs after the first "
                          f"(got {overlay_roles}, packs {roles})")
    overlays = [path(data.sidecars.overlays[r]) for r in overlay_roles] or None
    configure_train_sidecars(
        outcomes=path(data.sidecars.outcomes) if data.sidecars.outcomes else None,
        identities=identities[0])

    train_list = path(data.windows.train) if data.windows.train else None
    val_list = path(data.windows.val) if data.windows.val else None
    exclude = [path(x) for x in data.windows.exclude] or None
    out_dir = Path(out_dir)
    if data.windows.val_holdout_frac > 0.0:
        train_list, val_list = materialise_holdout(
            train_list, out_dir, frac=float(data.windows.val_holdout_frac),
            seed=int(data.windows.val_holdout_seed))

    sw = data.switching
    horizon = _horizon_steps(cfg)
    released = sw.enabled and sw.heldout_family_policy == "release"
    heldout = (mixdata.HeldOutGuard.load(path(sw.zero_shot_heldout))
               if sw.enabled and not released else None)
    leave_out = bool(data.windows.leave_pi_wall_out)
    guard = (ControllerExclusion(horizon_steps=horizon, heldout=heldout, leave_pi_wall_out=leave_out)
             if heldout or leave_out else None)

    def original(window_list: Optional[str], start_shift: float) -> TrajectoryDataset:
        return TrajectoryDataset.from_real_packs(
            packs, identity_dirs=identities, overlays=overlays, horizon_steps=horizon,
            start_shift_prob=start_shift, window_list=window_list,
            exclude_window_list=exclude, entries_v2=bool(data.entries_v2), item_filter=guard,
            history_s=data.history_s, history_stride_s=data.history_stride_s,
            **({"merge_pivot_history": True}
               if cfg.model.merge_decision == "first_passage" else {}),
            **_junction_kwargs(data))

    train = original(train_list, float(data.effective_start_shift_prob))
    pack_k = getattr(getattr(getattr(train, "_loader", None), "corpus", None), "k_slots", None)
    print("train " + _kept_summary(train.receipt), file=sys.stderr)
    val = None
    if val_list:
        val = original(val_list, 0.0)
        print("val   " + _kept_summary(val.receipt), file=sys.stderr)

    curriculum = tuple(cfg.curriculum.unroll_s)
    if sw.enabled:
        switch = mixdata.build_switch_datasets(
            pack=path(sw.pack), identities=path(sw.identities),
            window_list=path(sw.windows_train), index_path=path(sw.timeline_index),
            heldout_path=(None if released else path(sw.zero_shot_heldout)),
            horizon_steps=horizon,
            entries_v2=bool(data.entries_v2), family_policy=sw.heldout_family_policy,
            max_drop_frac=float(sw.heldout_max_drop_frac), val_holdout_frac=float(sw.val_frac),
            val_holdout_seed=int(sw.val_seed),
            val_window_list=path(sw.windows_val) if sw.windows_val else None,
            work_dir=out_dir, leave_pi_wall_out=leave_out,
            history_s=data.history_s, history_stride_s=data.history_stride_s,
            **_junction_kwargs(data))
    mixed = False
    if sw.enabled and switch["train"] is None:
        # DATAHYG strict removed (almost) every direct-mix item: train without the source.
        print("switching source DISABLED by data.controller_hygiene: "
              + json.dumps(switch["receipt"].get("controller_hygiene")), file=sys.stderr)
        train.receipt["switching_disabled_by_controller_hygiene"] = switch["receipt"]
    elif sw.enabled:
        train, val = mixdata.mix_datasets(
            train, switch, switch_ratio=float(sw.mix_ratio),
            event_oversample=float(sw.event_oversample), orig_val=val,
            jam_oversample=float(data.jam.oversample), jam_speed_mps=float(data.jam.speed_mps),
            jam_cache_dir=(data.jam.cache_dir or out_dir), **_jam_min(data))
        train.receipt["stage_sampling"] = train.describe_stages(curriculum)
        mixed = True
    if not mixed and float(data.jam.oversample) > 1.0:
        from ..data import jam

        report = jam.compute_jam_flags(
            train, speed_mps=float(data.jam.speed_mps),
            cache_path=os.path.join(data.jam.cache_dir or out_dir, "jam_original.json"),
            source_key=jam.dataset_source_key(train), **_jam_min(data, key="min_fraction"))
        report["oversample"] = float(data.jam.oversample)
        train = jam.JamOversampledDataset(train, report["flags"],
                                          oversample=float(data.jam.oversample), report=report)
        train.receipt["stage_sampling"] = train.describe_stages(curriculum)
    c4 = data.c4_episodes
    if c4.enabled:
        # Dreamer data loop: collected C4 SUMO episodes (python -m wms.data.c4_episodes)
        # become one more TRAIN source with sampling share ``mix_ratio``.
        from ..data.c4_episodes import add_c4_source, build_c4_datasets

        source = build_c4_datasets(
            path(c4.path), horizon_steps=horizon, history_s=data.history_s,
            history_stride_s=data.history_stride_s, entries_v2=bool(data.entries_v2),
            stride_s=float(c4.window_stride_s), start_s=c4.start_s, k_slots=pack_k,
            heldout=heldout, **({"merge_pivot_history": True}
                               if cfg.model.merge_decision == "first_passage" else {}))
        if not c4.use_val:
            source["val"] = None
        train, val = add_c4_source(train, val, source, mix_ratio=float(c4.mix_ratio),
                                   jam_speed_mps=float(data.jam.speed_mps), **_jam_min(data))
        train.receipt["stage_sampling"] = train.describe_stages(curriculum)
        print(f"c4_episodes train {len(source['train'])} windows, val "
              f"{0 if source['val'] is None else len(source['val'])} windows, "
              f"mix_ratio {float(c4.mix_ratio)}", file=sys.stderr)
    if data.c4_sources:
        train, val = _add_c4_sources(cfg, train, val, path, horizon=horizon, pack_k=pack_k,
                                     heldout=heldout, curriculum=curriculum)
    if getattr(data, "replay", None) is not None and data.replay.enabled:
        train = _wrap_replay(train, data.replay, curriculum)
    return train, val, resolved


def _wrap_replay(train: Any, replay_cfg: Any, curriculum: tuple) -> Any:
    """V10HORIZON: wrap the assembled TRAIN dataset in prioritised replay."""
    from ..data.replay_flags import compute_replay_flags
    from ..train.replay import PriorityReplayDataset

    low = float(getattr(replay_cfg, "low_share_oversample", 1.0)) > 1.0
    flags = compute_replay_flags(
        train, merge_end_m=float(replay_cfg.merge_zone_end_m),
        **({"low_share_max": float(replay_cfg.low_share_max)} if low else {}))
    replay = PriorityReplayDataset(
        train, merge_zone=flags["merge_zone"], cav_follower=flags["cav_follower"],
        brake_event=flags["brake_event"],
        merge_oversample=float(replay_cfg.merge_oversample),
        cav_oversample=float(replay_cfg.cav_follower_oversample),
        brake_oversample=float(replay_cfg.brake_oversample),
        error_alpha=float(replay_cfg.error_alpha),
        error_decay=float(replay_cfg.error_decay),
        error_max_boost=float(replay_cfg.error_max_boost),
        **({"low_share": flags["low_share"],
            "low_share_oversample": float(replay_cfg.low_share_oversample),
            "low_share_min_stage_s": float(replay_cfg.low_share_min_stage_s)} if low else {}))
    replay.receipt["replay"]["scan"] = {k: flags[k] for k in
                                        ("n_windows", "n_merge_zone", "n_cav_follower",
                                         "n_brake_event", "n_both", "merge_end_m",
                                         "merge_zone_s1_m", "n_low_share", "low_share_max")
                                        if k in flags}
    replay.receipt["replay"]["stage_sampling"] = replay.describe_stages(curriculum)
    print(f"priority replay: {flags['n_merge_zone']} merge-zone / "
          f"{flags['n_cav_follower']} CAV-follower / {flags['n_brake_event']} brake-event "
          f"of {flags['n_windows']} windows", file=sys.stderr)
    return replay


def _jam_min(data: Any, key: str = "jam_min_fraction") -> dict:
    """INFRAV5 ``data.jam.min_fraction`` kwarg, omitted at its legacy value 0 (unchanged calls)."""
    value = float(getattr(data.jam, "min_fraction", 0.0))
    return {key: value} if value > 0.0 else {}


def _longwm_paired_sources(loss: Any, data: Any) -> tuple[list[Any], Any]:
    """S-WMCOMBINED: the longwm paired pool = loop14_pairs (+ ``longwm_extra_paired``).

    Returns ``(sources, extra)``; ``extra`` is None in the default bit-identical mode
    where the pool is exactly ``loop14_pairs`` as before.
    """
    names = ('loop14_pairs',)
    extra = getattr(loss, 'longwm_extra_paired', None)
    if extra:
        names = names + (str(extra),)
    sources = [src for src in data.c4_sources if src.name in names]
    if len(sources) != len(names):
        raise ConfigError(f'loop longwm requires the paired c4 sources {names}')
    return sources, extra


def _add_c4_sources(cfg: TrainConfig, train: Any, val: Any, path: Callable[[str], str], *,
                    horizon: int, pack_k: Optional[int], heldout: Any, curriculum: tuple
                    ) -> tuple[Any, Any]:
    """PAIRLOSS: mix ``data.c4_sources`` and (``loss.paired_diff`` on) wrap paired batches."""
    from ..data.c4_episodes import build_c4_datasets, read_source_manifest
    from ..data.c4_pairs import (ConcatPairedSource, PairedBatchDataset, PairedBranchSource,
                                  add_c4_sources, mix_c4_sources, world_roles)

    data = cfg.data
    roots = {src.name: path(src.path) for src in data.c4_sources}
    docs = [read_source_manifest(r) for r in roots.values()]
    if data.c4_episodes.enabled:
        docs.append(read_source_manifest(path(data.c4_episodes.path)))
    roles, overridden = world_roles(docs)
    base = data.c4_base
    if base is not None:
        # V5DATA: every C4 source must stack with the base source (collation stacks slots).
        pack_k = int(read_source_manifest(roots[base])["k_slots"])
    common = dict(horizon_steps=horizon, history_s=data.history_s,
                  history_stride_s=data.history_stride_s, entries_v2=bool(data.entries_v2),
                  k_slots=pack_k, heldout=heldout, role_of=roles,
                  junction_fix=getattr(data, "junction_fix", "off"),
                  command_known=getattr(data, "controller_hygiene", "off") == "strict")
    if cfg.model.merge_decision == "first_passage":
        common["merge_pivot_history"] = True
    mixed = []
    order = sorted(data.c4_sources, key=lambda src: src.name != base)   # base first (V5DATA)
    for src in order:
        if src.mix_ratio <= 0.0:
            continue
        built = build_c4_datasets(roots[src.name], stride_s=float(src.window_stride_s),
                                  start_s=src.start_s, allow_empty=src.paired, **common)
        if not src.use_val:
            built["val"] = None
        if built["train"] is None:
            print(f"c4_sources[{src.name}]: no TRAIN window with the full {horizon} steps "
                  "(ordinary mix skipped)", file=sys.stderr)
            continue
        mixed.append((src.name, built, float(src.mix_ratio)))
        print(f"c4_sources[{src.name}] train {len(built['train'])} windows, val "
              f"{0 if built['val'] is None else len(built['val'])} windows, "
              f"mix_ratio {float(src.mix_ratio)}", file=sys.stderr)
    if base is not None:
        if not mixed or mixed[0][0] != base:
            raise ConfigError(f"data.c4_base {base!r}: the base source has no TRAIN window")
        train, val = mix_c4_sources(mixed, jam_oversample=float(data.jam.oversample),
                                    jam_speed_mps=float(data.jam.speed_mps), **_jam_min(data))
        train.receipt["c4_base"] = base
    else:
        train, val = add_c4_sources(train, val, mixed, jam_speed_mps=float(data.jam.speed_mps),
                                    **_jam_min(data))
    if hasattr(train, "describe_stages"):
        train.receipt["stage_sampling"] = train.describe_stages(curriculum)
    train.receipt["c4_world_split"] = {"n_worlds": len(roles),
                                       "n_val_worlds": sum(r == "val" for r in roles.values()),
                                       "overridden_to_val": overridden}
    pair = cfg.loss.paired_diff
    paired = [src for src in data.c4_sources if src.paired]
    extra_paired = None
    if cfg.loss.longwm_enabled:
        # Preserve ordinary replay shares; replace the paired-batch source with
        # THIS cycle's real POLICY/B tapes, never historical diagnostic pairs.
        paired, extra_paired = _longwm_paired_sources(cfg.loss, data)
    if (pair.enabled or cfg.loss.longwm_enabled) and paired:
        if len(paired) > 1 and not (cfg.loss.longwm_enabled and extra_paired):
            raise ConfigError("loss.paired_diff supports one paired c4 source")
        # Stage mode retains the full curriculum horizon for eligibility/window switching;
        # fixed mode keeps the original constructor call and branch-length cap.
        pair_kwargs = dict(common)
        if cfg.loss.longwm_enabled:
            pair_kwargs['arm_kinds'] = ('B', 'POLICY')
        if pair.horizon_mode == "stage":
            pair_kwargs.update(horizon_mode="stage", horizon_steps=_horizon_steps(cfg))
        sources = [PairedBranchSource(roots[s.name], name=s.name, **pair_kwargs)
                   for s in paired]
        source = sources[0] if len(sources) == 1 else ConcatPairedSource(sources)
        if cfg.loss.longwm_enabled:
            if source.horizon < 600 or any(not g['has_native_ref'] for g in source.groups):
                raise ConfigError('loop longwm needs complete real paired 300s native-reference tapes')
            source.longwm_pair = True
        train = PairedBatchDataset(train, source, fraction=float(pair.fraction),
                                   groups=int(pair.groups), max_arms=int(pair.max_arms),
                                   batch_size=int(cfg.optim.batch_size),
                                   seed=int(cfg.runtime.seed))
        print(f"paired batches: {source.receipt['n_groups']} groups / "
              f"{source.receipt['n_windows']} windows from c4_sources[{source.name}], horizon "
              f"{source.horizon} steps, fraction {float(pair.fraction)}", file=sys.stderr)
    return train, val


def assemble(cfg: TrainConfig, out_dir: Path, resolver: Optional[Resolver] = None
             ) -> TrainingInputs:
    """Datasets + physics + model of ``cfg`` (data are opened lazily; nothing is trained)."""
    resolver = resolver or default_resolver()
    train, val, resolved = build_datasets(cfg, out_dir, resolver)
    physics = build_physics(cfg, resolver)
    return TrainingInputs(train, val, build_model(cfg, physics), physics, resolved)


def window_counts(inputs: TrainingInputs) -> dict[str, dict[str, Any]]:
    """Kept windows per pack for the train and validation datasets (from their receipts)."""
    def counts(dataset: Any) -> dict[str, Any]:
        receipt = getattr(dataset, "receipt", {}) or {}
        out = {"n": len(dataset) if dataset is not None else 0,
               "kept_by_pack": dict(receipt.get("kept_by_pack", {}))}
        if "c4_sources" in receipt or "paired_batches" in receipt:
            # PAIRLOSS: windows and sampling share per mixed source
            out["by_source"] = {s["name"]: {"n": s["n"], "share": s["share"]}
                                for s in receipt.get("sources", [])}
        return out
    return {"train": counts(inputs.train), "val": counts(inputs.val)}


def describe(inputs: TrainingInputs) -> str:
    """One JSON line summarising the assembled run (printed by ``wms train``)."""
    return json.dumps({"windows": window_counts(inputs), "n_parameters": sum(
        p.numel() for p in inputs.model.parameters())}, sort_keys=True)
