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
    """``PhysicsConfig`` from the model flags (``--merge-cal-params`` semantics of the legacy CLI)."""
    from ..physics.config import PhysicsConfig

    resolver = resolver or default_resolver()
    model = cfg.model
    sumo_idm = not model.geometry_only
    if model.merge_cal:
        overrides = {"sumo_fidelity": bool(model.sumo_fidelity), "sumo_idm": sumo_idm,
                     "merge_cal": True, "lane_rule": model.lane_rule,
                     "lane_gap_rule": model.lane_gap_rule}
        if model.physics_params:
            return PhysicsConfig.from_params_file(resolver(model.physics_params), **overrides)
        return PhysicsConfig(**overrides)
    return PhysicsConfig(sumo_fidelity=bool(model.sumo_fidelity), sumo_idm=sumo_idm,
                         lane_rule=model.lane_rule, lane_gap_rule=model.lane_gap_rule)


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
            history_s=data.history_s, history_stride_s=data.history_stride_s)

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
            history_s=data.history_s, history_stride_s=data.history_stride_s)
        train, val = mixdata.mix_datasets(
            train, switch, switch_ratio=float(sw.mix_ratio),
            event_oversample=float(sw.event_oversample), orig_val=val,
            jam_oversample=float(data.jam.oversample), jam_speed_mps=float(data.jam.speed_mps),
            jam_cache_dir=(data.jam.cache_dir or out_dir))
        train.receipt["stage_sampling"] = train.describe_stages(curriculum)
    elif float(data.jam.oversample) > 1.0:
        from ..data import jam

        report = jam.compute_jam_flags(
            train, speed_mps=float(data.jam.speed_mps),
            cache_path=os.path.join(data.jam.cache_dir or out_dir, "jam_original.json"),
            source_key=jam.dataset_source_key(train))
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
            heldout=heldout)
        if not c4.use_val:
            source["val"] = None
        train, val = add_c4_source(train, val, source, mix_ratio=float(c4.mix_ratio),
                                   jam_speed_mps=float(data.jam.speed_mps))
        train.receipt["stage_sampling"] = train.describe_stages(curriculum)
        print(f"c4_episodes train {len(source['train'])} windows, val "
              f"{0 if source['val'] is None else len(source['val'])} windows, "
              f"mix_ratio {float(c4.mix_ratio)}", file=sys.stderr)
    return train, val, resolved


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
        return {"n": len(dataset) if dataset is not None else 0,
                "kept_by_pack": dict(receipt.get("kept_by_pack", {}))}
    return {"train": counts(inputs.train), "val": counts(inputs.val)}


def describe(inputs: TrainingInputs) -> str:
    """One JSON line summarising the assembled run (printed by ``wms train``)."""
    return json.dumps({"windows": window_counts(inputs), "n_parameters": sum(
        p.numel() for p in inputs.model.parameters())}, sort_keys=True)
