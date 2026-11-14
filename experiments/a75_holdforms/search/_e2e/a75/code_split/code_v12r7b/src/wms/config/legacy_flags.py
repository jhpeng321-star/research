"""Old ``python -m wms.cli train ...`` flags mapped onto the typed ``TrainConfig``.

``FLAGS`` is the single table: every legacy option, its argparse behaviour and
default (identical to the old parser) and the nested config path it sets.
Options with ``path=None`` need more than a plain assignment and are handled by
``_special`` below.  :func:`parse_legacy_argv` turns an old argv into a config,
so old chains keep working while presets replace them.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ..core.constants import N_CONTROLLERS
from ..core.errors import ConfigError
from .data import CAV_DRIVE_MODES
from .legacy_flat import set_path
from .schema import from_mapping
from .train import ROLLOUT_MODES, TrainConfig

COMMANDS = ("dry-run", "train")
_STORE_TRUE = {"action": "store_true", "default": False}


@dataclass(frozen=True)
class Flag:
    """One legacy option: option strings, argparse kwargs and target config path."""

    options: tuple[str, ...]
    dest: str
    path: Optional[str]
    kwargs: Mapping[str, Any]


def _f(options: str, dest: str, path: Optional[str], **kwargs: Any) -> Flag:
    return Flag(tuple(options.split()), dest, path, kwargs)


FLAGS: tuple[Flag, ...] = (
    _f("--pack", "pack", None, action="append", required=True),
    _f("--identities", "identities", None, action="append", default=None),
    _f("--outcomes", "outcomes", "data.sidecars.outcomes", default=None),
    _f("--overlay", "overlay", None, action="append", default=None),
    _f("--window-list", "window_list", "data.windows.train", default=None),
    _f("--exclude-window-list", "exclude_window_list", "data.windows.exclude",
       action="append", default=None),
    _f("--val-window-list", "val_window_list", "data.windows.val", default=None),
    _f("--val-holdout-frac", "val_holdout_frac", "data.windows.val_holdout_frac",
       type=float, default=0.0),
    _f("--val-holdout-seed", "val_holdout_seed", "data.windows.val_holdout_seed",
       type=int, default=0),
    _f("--select-by-long-horizon", "select_by_long_horizon", "selection.select_by_long_horizon",
       **_STORE_TRUE),
    _f("--val-every", "val_every", "selection.val_every", type=int, default=200),
    _f("--out", "out", "runtime.out_dir", default="output/wd01/wm_n1/wms/runs"),
    _f("--threads", "threads", "runtime.threads", type=int, default=6),
    _f("--unroll-s", "unroll_s", None, type=float, action="append", default=None),
    _f("--steps", "steps", None, type=int, default=None),
    _f("--batch-size", "batch_size", "optim.batch_size", type=int, default=4),
    _f("--sumo-fidelity", "sumo_fidelity", "model.sumo_fidelity", **_STORE_TRUE),
    _f("--geometry-only", "geometry_only", "model.geometry_only", **_STORE_TRUE),
    _f("--truncation-steps", "truncation_steps", "curriculum.truncation_steps", type=int, default=20),
    _f("--device", "device", "runtime.device", default="cpu"),
    _f("--start-shift-prob", "start_shift_prob", "data.start_shift_prob", type=float, default=0.0),
    _f("--history-s", "history_s", "data.history_s", type=float, default=None),
    _f("--history-stride-s", "history_stride_s", "data.history_stride_s", type=float, default=1.0),
    _f("--num-workers", "num_workers", None, type=int, default=None),
    _f("--gpu-share", "gpu_share", None, **_STORE_TRUE),
    _f("--stage-steps", "stage_steps", None, default=None),
    _f("--lr", "lr", "optim.lr", type=float, default=1.0e-3),
    _f("--seed", "seed", "runtime.seed", type=int, default=101),
    _f("--black-box", "black_box", None, **_STORE_TRUE),
    _f("--n-controllers", "n_controllers", "model.n_controllers", type=int, default=N_CONTROLLERS),
    _f("--controller-embed-dim", "controller_embed_dim", "model.controller_embed_dim",
       type=int, default=16),
    _f("--switch-window-fraction", "switch_window_fraction", "data.switching.window_fraction",
       type=float, default=0.0),
    _f("--checkpoint-residual", "checkpoint_residual", "model.checkpoint_residual", **_STORE_TRUE),
    _f("--blackbox-traj", "blackbox_traj", None, **_STORE_TRUE),
    _f("--merge-cal", "merge_cal", "model.merge_cal", **_STORE_TRUE),
    _f("--merge-cal-params", "merge_cal_params", "model.physics_params", default=None),
    _f("--traffic-loss", "traffic_loss", "loss.traffic.enabled", **_STORE_TRUE),
    _f("--traffic-weight", "traffic_weight", "loss.traffic.weight", type=float, default=0.1),
    _f("--traffic-bin-s", "traffic_bin_s", "loss.traffic.bin_s", type=float, default=30.0),
    _f("--traffic-ramp-start-s", "traffic_ramp_start_s", "loss.traffic.ramp_start_s",
       type=float, default=30.0),
    _f("--traffic-ramp-end-s", "traffic_ramp_end_s", "loss.traffic.ramp_end_s",
       type=float, default=120.0),
    _f("--state-noise-std-m", "state_noise_std_m", "data.state_noise_std_m", type=float, default=0.0),
    _f("--state-noise-std-mps", "state_noise_std_mps", "data.state_noise_std_mps",
       type=float, default=0.0),
    _f("--entries-v2", "entries_v2", "data.entries_v2", **_STORE_TRUE),
    _f("--entry-pad", "entry_pad", "data.entry_pad", type=int, default=8),
    _f("--rollout-mode", "rollout_mode", "model.rollout_mode", choices=ROLLOUT_MODES,
       default="residual"),
    _f("--lane-head", "lane_head", "model.lane_head", choices=("off", "history"), default="off"),
    _f("--residual-magnitude-weight", "residual_magnitude_weight", "loss.residual.hybrid_magnitude_weight",
       type=float, default=None),
    _f("--driver-sidecar", "driver_sidecar", "data.sidecars.driver_sidecar", default=None),
    _f("--projection-enabled --projection", "projection_enabled", "model.projection.enabled",
       **_STORE_TRUE),
    _f("--projection-min-gap-m", "projection_min_gap_m", "model.projection.min_gap_m",
       type=float, default=2.0),
    _f("--residual-cap-mps2", "residual_cap_mps2", "model.residual_cap_mps2", type=float, default=None),
    _f("--residual-soft-bound-mps2", "residual_soft_bound_mps2", "model.residual_soft_bound_mps2",
       type=float, default=None),
    _f("--residual-soft-bound-neg-mps2", "residual_soft_bound_neg_mps2",
       "model.residual_soft_bound_neg_mps2", type=float, default=None),
    _f("--residual-mag-weight", "residual_mag_weight", "loss.residual.residual_mag_weight",
       type=float, default=0.0),
    _f("--lane-rate-weight", "lane_rate_weight", "loss.lane_rate.weight", type=float, default=0.0),
    _f("--lane-rate-target", "lane_rate_target", "loss.lane_rate.target", type=float, default=0.0123),
    _f("--lane-logit-cap", "lane_logit_cap", "model.lane_logit_cap", type=float, default=None),
    _f("--stability-weight", "stability_weight", "loss.stability_weight", type=float, default=0.0),
    _f("--stability-margin", "stability_margin", "loss.stability_margin", type=float, default=1.0),
    _f("--stability-probes", "stability_probes", "loss.stability_probes", type=int, default=1),
    _f("--conservation-metric", "conservation_metric", "loss.conservation_enabled", **_STORE_TRUE),
    _f("--conservation-weight", "conservation_weight", "loss.conservation_weight",
       type=float, default=0.0),
    _f("--loss-share-logging", "loss_share_logging", "loss.loss_share_logging", **_STORE_TRUE),
    _f("--asym-speed-loss", "asym_speed_loss", "loss.braking.asym_speed_alpha", type=float, default=1.0),
    _f("--decel-weight", "decel_weight", "loss.braking.decel_weight", type=float, default=1.0),
    _f("--decel-threshold", "decel_threshold", "loss.braking.decel_threshold_mps2",
       type=float, default=3.0),
    _f("--propagation-loss", "propagation_loss", "loss.braking.propagation_weight",
       type=float, default=0.0),
    _f("--prop-window", "prop_window", "loss.braking.prop_window_s", type=float, default=3.0),
    _f("--prop-max-gap-m", "prop_max_gap_m", "loss.braking.prop_max_gap_m", type=float, default=None),
    _f("--residual-cap-neg-mps2", "residual_cap_neg_mps2", "model.residual_cap_neg_mps2",
       type=float, default=None),
    _f("--cav-drive", "cav_drive", "data.cav_drive", choices=CAV_DRIVE_MODES, default="teacher"),
    _f("--leave-pi-wall-out", "leave_pi_wall_out", "data.windows.leave_pi_wall_out", **_STORE_TRUE),
    _f("--switch-pack", "switch_pack", "data.switching.pack", default=None),
    _f("--switch-identities", "switch_identities", "data.switching.identities", default=None),
    _f("--switch-window-list", "switch_window_list", "data.switching.windows_train", default=None),
    _f("--switch-timeline-index", "switch_timeline_index", "data.switching.timeline_index", default=None),
    _f("--zero-shot-heldout", "zero_shot_heldout", "data.switching.zero_shot_heldout", default=None),
    _f("--heldout-family-policy", "heldout_family_policy", "data.switching.heldout_family_policy",
       choices=("drop", "refuse"), default="drop"),
    _f("--heldout-max-drop-frac", "heldout_max_drop_frac", "data.switching.heldout_max_drop_frac",
       type=float, default=0.85),
    _f("--switch-mix-ratio", "switch_mix_ratio", "data.switching.mix_ratio", type=float, default=0.5),
    _f("--event-oversample", "event_oversample", "data.switching.event_oversample",
       type=float, default=2.0),
    _f("--jam-oversample", "jam_oversample", "data.jam.oversample", type=float, default=1.0),
    _f("--jam-speed-mps", "jam_speed_mps", "data.jam.speed_mps", type=float, default=5.0),
    _f("--jam-cache-dir", "jam_cache_dir", "data.jam.cache_dir", default=None),
    _f("--switch-val-window-list", "switch_val_window_list", "data.switching.windows_val", default=None),
    _f("--switch-val-frac", "switch_val_frac", "data.switching.val_frac", type=float, default=0.1),
    _f("--switch-val-seed", "switch_val_seed", "data.switching.val_seed", type=int, default=0),
)

#: Every option string accepted on the legacy command line -> its config path
#: (``"<special>"`` for flags that need conversion, see ``_special``).
FLAG_TO_FIELD: dict[str, str] = {
    option: (flag.path or "<special>") for flag in FLAGS for option in flag.options
}

#: DataLoader workers implied by ``--gpu-share`` when ``--num-workers`` is absent.
GPU_SHARE_WORKERS = 4


def build_parser() -> argparse.ArgumentParser:
    """Argparse parser accepting exactly the old ``wms.cli`` flags with the old defaults."""
    parser = argparse.ArgumentParser(prog="wms.cli", description="legacy WM-S trainer flags",
                                     allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sub_parser = sub.add_parser(name, allow_abbrev=False)
        for flag in FLAGS:
            sub_parser.add_argument(*flag.options, dest=flag.dest, **dict(flag.kwargs))
    return parser


def parse_legacy_argv(argv: Sequence[str]) -> tuple[str, TrainConfig]:
    """Parse an old ``dry-run|train`` argv into ``(command, TrainConfig)``."""
    args = build_parser().parse_args(list(argv))
    tree = config_tree_from_args(args)
    return args.command, from_mapping(TrainConfig, tree)


def config_tree_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Nested config mapping equivalent to a parsed legacy namespace."""
    tree: dict[str, Any] = {}
    for flag in FLAGS:
        value = getattr(args, flag.dest)
        if flag.path is None or value is None:
            continue
        if isinstance(value, list):
            value = list(value)
        set_path(tree, flag.path, value)
    _special(args, tree)
    return tree


def _special(args: argparse.Namespace, tree: dict[str, Any]) -> None:
    """Flags whose value needs conversion or couples several config fields."""
    set_path(tree, "data.packs", {f"pack{i}": path for i, path in enumerate(args.pack)})
    _set_sidecars(args, tree)
    if args.unroll_s:
        set_path(tree, "curriculum.unroll_s", list(args.unroll_s))
    if args.stage_steps:
        set_path(tree, "curriculum.stage_steps", [int(x) for x in args.stage_steps.split(",")])
    set_path(tree, "curriculum.steps_per_epoch",
             args.steps if args.steps is not None else (2 if args.command == "dry-run" else 2000))
    num_workers = args.num_workers
    if args.gpu_share and num_workers is None:
        num_workers = GPU_SHARE_WORKERS
    set_path(tree, "runtime.num_workers", 0 if num_workers is None else num_workers)
    if args.black_box and args.blackbox_traj:
        raise ConfigError("--black-box and --blackbox-traj are mutually exclusive in the new config")
    if args.black_box or args.blackbox_traj:
        set_path(tree, "model.kind", "blackbox_id" if args.black_box else "blackbox_traj")


def _set_sidecars(args: argparse.Namespace, tree: dict[str, Any]) -> None:
    """Align the positional ``--identities``/``--overlay`` lists with the ``--pack`` roles."""
    roles = [f"pack{i}" for i in range(len(args.pack))]
    identities = args.identities or []
    if identities and len(identities) != len(roles):
        raise ConfigError("--identities must be supplied once per --pack")
    set_path(tree, "data.sidecars.identities", dict(zip(roles, identities)))
    overlays = args.overlay or []
    if overlays:
        if len(overlays) == len(roles) - 1:
            overlay_roles = roles[1:]
        elif len(overlays) == len(roles):
            overlay_roles = roles
        else:
            raise ConfigError("--overlay must be supplied once per extra --pack")
        set_path(tree, "data.sidecars.overlays", dict(zip(overlay_roles, overlays)))
