"""WM-S command line entry point.

Examples (from ``output/wd01/wm_n1``)::

    python -m wms.cli dry-run --pack TRAIN/sequence.npz
    python -m wms.cli train --pack TRAIN/pack_v3 --pack TRAIN/dspeed --out RUN
    python -m wms.cli train --black-box --window-list windows_train_noPI.json ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .blackbox_traj import BlackBoxTrajModel
from .model import N_CONTROLLERS, PhysicsConfig, WmSModel
from .train import (TrajectoryDataset, TrainDataRefusal, WmSTrainConfig,
                    WmSTrainer, dry_run)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wms", description="WM-S TRAIN-only trainer")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "train"):
        q = sub.add_parser(name)
        q.add_argument("--pack", action="append", required=True,
                       help="TRAIN trajectory pack; repeat for pack_v3/dspeed/damping")
        q.add_argument("--identities", action="append", default=None,
                       help="one read-only TRAIN identity sidecar per --pack")
        q.add_argument("--outcomes", default=None,
                       help="read-only TRAIN outcome sidecar for the mixed base pack")
        q.add_argument("--overlay", action="append", default=None,
                       help="one read-only overlay per extra pack")
        q.add_argument("--window-list", default=None, metavar="PATH",
                       help="C1 leave-controller-out TRAIN keep list "
                            "(P5 windows_train_noPI.json)")
        q.add_argument("--exclude-window-list", action="append", default=None,
                       metavar="PATH",
                       help="bare run-key exclusion list (pi_excluded_runs.json); "
                            "repeatable")
        q.add_argument("--val-window-list", default=None, metavar="PATH",
                       help="monitoring-only validation window list (windows_val.json)")
        q.add_argument("--val-holdout-frac", type=float, default=0.0,
                       help="hold out ~this fraction of TRAIN worlds from training "
                            "and monitor them via a derived TRAIN-scoped val list "
                            "(default 0 = off; requires --window-list and is "
                            "mutually exclusive with --val-window-list)")
        q.add_argument("--val-holdout-seed", type=int, default=0,
                       help="seed for --val-holdout-frac world selection")
        q.add_argument("--select-by-long-horizon", action="store_true", default=False,
                       help="select checkpoints by long-horizon validation (default off)")
        q.add_argument("--val-every", type=int, default=200,
                       help="validation cadence in optimizer steps (default 200)")
        q.add_argument("--out", default="output/wd01/wm_n1/wms/runs")
        q.add_argument("--threads", type=int, default=6)
        q.add_argument("--unroll-s", type=float, action="append", default=None)
        q.add_argument("--steps", type=int, default=None,
                       help="optimizer steps (default: 2 dry-run, 2000 train)")
        q.add_argument("--batch-size", type=int, default=4)
        q.add_argument("--sumo-fidelity", action="store_true")
        q.add_argument("--geometry-only", action="store_true",
                       help="disable SUMO vType IDM means while retaining frame flags")
        q.add_argument("--truncation-steps", type=int, default=20)
        q.add_argument("--device", default="cpu")
        q.add_argument("--start-shift-prob", type=float, default=0.0,
                       help="probability a window starts at a random later tick of its "
                            "run (real pre-t0 history; identity row unused)")
        q.add_argument("--history-s", type=float, default=None, metavar="H",
                       help="opt-in pre-t0 driver history duration (e.g. 30 or 60 s); "
                            "omitted keeps the exact legacy 10-tick path")
        q.add_argument("--history-stride-s", type=float, default=1.0,
                       help="long driver-history sampling interval (default 1 s; "
                            "multiple of 0.5 s, inert without --history-s)")
        q.add_argument("--num-workers", type=int, default=None,
                       help="DataLoader worker processes (default 0; --gpu-share uses 4)")
        q.add_argument("--gpu-share", action="store_true",
                       help="share one 48 GB GPU across 3 trainings (num-workers 4)")
        q.add_argument("--stage-steps", default=None,
                       help="comma list: optimizer steps per curriculum stage")
        q.add_argument("--lr", type=float, default=1.0e-3)
        q.add_argument("--seed", type=int, default=101)
        q.add_argument("--black-box", action="store_true",
                       help="black-box variant: CAVs are learned movers conditioned "
                            "on a controller-id embedding; no controller code runs")
        q.add_argument("--n-controllers", type=int, default=N_CONTROLLERS)
        q.add_argument("--controller-embed-dim", type=int, default=16)
        q.add_argument("--switch-window-fraction", type=float, default=0.0,
                       help="minimum training sample share from windows containing switches")
        q.add_argument("--checkpoint-residual", action="store_true",
                       help="gradient-checkpoint the residual net to bound CPU memory")
        q.add_argument("--blackbox-traj", action="store_true",
                       help="use the strong pure-learned black-box HDV baseline "
                            "(no IDM/MOBIL base; CAVs still teacher-forced); "
                            "default off keeps WM-S physics behaviour")
        q.add_argument("--merge-cal", action="store_true",
                       help="enable calibrated SUMO-like mandatory-merge physics "
                            "(rollout2.merge_cal; default OFF keeps the legacy "
                            "MOBIL-merge baseline byte-identical)")
        q.add_argument("--merge-cal-params", default=None, metavar="PATH",
                       help="TRAIN-fitted merge-cal params JSON (unknown keys filtered); "
                            "only applied together with --merge-cal")
        q.add_argument("--traffic-loss", action="store_true",
                       help="Run 2/B1: add the differentiable section x 30 s-bin "
                            "traffic-level loss (default off)")
        q.add_argument("--traffic-weight", type=float, default=0.1,
                       help="peak traffic-loss weight (reached at --traffic-ramp-end-s)")
        q.add_argument("--traffic-bin-s", type=float, default=30.0)
        q.add_argument("--traffic-ramp-start-s", type=float, default=30.0,
                       help="traffic loss is 0 before this horizon (per-car dominates)")
        q.add_argument("--traffic-ramp-end-s", type=float, default=120.0)
        q.add_argument("--state-noise-std-m", type=float, default=0.0,
                       help="Run 2/B4: Gaussian std (m) on predicted position during "
                            "training rollouts (default 0 = off)")
        q.add_argument("--state-noise-std-mps", type=float, default=0.0,
                       help="Run 2/B4: Gaussian std (m/s) on predicted speed (default 0)")
        q.add_argument("--entries-v2", action="store_true",
                       help="observed-history vehicle entries in the trainer: entrants "
                            "occupy free padding slots, move by physics (+ residual) and "
                            "are excluded from the per-car loss (default OFF)")
        q.add_argument("--entry-pad", type=int, default=8,
                       help="free padding slots per window for --entries-v2 entrants")
        q.add_argument("--rollout-mode",
                       choices=("residual", "driver_params", "hybrid"),
                       default="residual",
                       help="WM-S2: driver_params infers per-HDV IDM/MOBIL parameters "
                             "from observed pre-t0 history; hybrid keeps those params "
                             "AND the gated learned residual, conditioning the "
                             "residual net on the driver embedding (default residual)")
        q.add_argument("--lane-head", choices=("off", "history"), default="off",
                       help="learned history-conditioned lane-change head (default off)")
        q.add_argument("--residual-magnitude-weight", type=float, default=None,
                       help="hybrid: weight of the mean-residual-magnitude penalty used "
                            "to keep the driver encoder identifiable (default 1e-3 in "
                            "hybrid, 0 elsewhere)")
        q.add_argument("--driver-sidecar", default=None,
                       help="optional per-vehicle driver-parameter ground-truth sidecar "
                            "(hetdata truth CSV/JSON/JSONL/NPZ); evaluation only, never "
                            "used in the training loss")
        q.add_argument("--projection-enabled", "--projection", dest="projection_enabled",
                       action="store_true",
                       help="Run-2/A: differentiable physics feasibility projection "
                            "(min-gap / no-overlap) during rollout (default off)")
        q.add_argument("--projection-min-gap-m", type=float, default=2.0)
        q.add_argument("--residual-cap-mps2", type=float, default=None,
                       help="Run-2/B: tanh cap (m/s^2) on the learned residual (default off)")
        q.add_argument("--residual-soft-bound-mps2", type=float, default=None,
                       help="soft tanh bound (m/s^2) on the learned residual (default off)")
        q.add_argument("--residual-soft-bound-neg-mps2", type=float, default=None,
                       help="separate soft tanh bound (m/s^2) on the negative residual; "
                            "--residual-soft-bound-mps2 then bounds only the positive "
                            "side (default off = symmetric)")
        q.add_argument("--residual-mag-weight", type=float, default=0.0,
                       help="long-horizon residual-magnitude penalty weight (default off)")
        q.add_argument("--lane-rate-weight", type=float, default=0.0,
                       help="lane-change-rate penalty weight (default off)")
        q.add_argument("--lane-rate-target", type=float, default=0.0123,
                       help="target lane-change rate for --lane-rate-weight (default 0.0123)")
        q.add_argument("--lane-logit-cap", type=float, default=None,
                       help="Run-2/B: tanh cap on the learned lane-logit correction")
        q.add_argument("--stability-weight", type=float, default=0.0,
                       help="Run-2/B: contractivity regulariser weight (default off)")
        q.add_argument("--stability-margin", type=float, default=1.0)
        q.add_argument("--stability-probes", type=int, default=1)
        q.add_argument("--conservation-metric", dest="conservation_metric",
                       action="store_true",
                       help="Run-2/D: log the vehicle-count conservation metric "
                            "(default off)")
        q.add_argument("--conservation-weight", type=float, default=0.0,
                       help="Run-2/D: vehicle-conservation penalty weight (default off)")
        q.add_argument("--loss-share-logging", action="store_true",
                       help="Run 3: log each loss term's share of the total gradient "
                            "norm per curriculum stage (diagnostic; default off)")
        # Braking-response fix (wms/brakefix.py); every flag defaults OFF.
        q.add_argument("--asym-speed-loss", type=float, default=1.0, metavar="ALPHA",
                       help="penalise predicted-too-fast speed/position ALPHA x more "
                            "than too-slow (ALPHA > 1 enables; default 1 = off)")
        q.add_argument("--decel-weight", type=float, default=1.0, metavar="W",
                       help="per-HDV-step weight multiplier W on the speed/position "
                            "loss where the RECORDED acceleration < -A "
                            "(--decel-threshold); W > 1 enables (default 1 = off)")
        q.add_argument("--decel-threshold", type=float, default=3.0, metavar="A",
                       help="braking threshold A in m/s^2 (positive magnitude) used by "
                            "--decel-weight and --propagation-loss (default 3.0)")
        q.add_argument("--propagation-loss", type=float, default=0.0, metavar="W",
                       help="weight W of the backward-wave loss on followers whose "
                            "recorded leader brakes harder than -A (W > 0 enables)")
        q.add_argument("--prop-window", type=float, default=3.0, metavar="S",
                        help="seconds after each recorded leader braking step over which the "
                            "follower's predicted speed/accel is supervised (default 3)")
        q.add_argument("--prop-max-gap-m", type=float, default=None,
                       help="optional maximum recorded leader bumper gap (default unlimited)")
        q.add_argument("--residual-cap-neg-mps2", type=float, default=None,
                       help="separate tanh cap (m/s^2) on the NEGATIVE (braking) side "
                            "of the learned residual; --residual-cap-mps2 then caps "
                            "only the positive side (default off = symmetric)")
        # --- switching / direct-output data mixed with the original packs ----
        # (wms.mixdata; every flag below is inert unless --switch-pack is given)
        q.add_argument("--cav-drive", choices=("teacher", "closed_loop"),
                       default="teacher",
                       help="teacher (default, as before): every CAV follows the "
                            "recorded trajectory; closed_loop: execute the controller "
                            "timeline on predicted states (model plant)")
        q.add_argument("--leave-pi-wall-out", action="store_true",
                       help="drop any window containing PI or wall-owned CAVs; also monitoring")
        q.add_argument("--switch-pack", default=None, metavar="DIR",
                       help="switching/direct TRAIN pack (pack_switching_train) with "
                            "controller_timeline.npy sidecars; mixed with --pack")
        q.add_argument("--switch-identities", default=None, metavar="DIR",
                       help="identity sidecar of --switch-pack (identities_switching_train)")
        q.add_argument("--switch-window-list", default=None, metavar="PATH",
                       help="TRAIN window list of --switch-pack (windows_train_hetdata.json)")
        q.add_argument("--switch-timeline-index", default=None, metavar="PATH",
                       help="ACTION_TIMELINE_INDEX_train.json of --switch-pack")
        q.add_argument("--zero-shot-heldout", default=None, metavar="PATH",
                       help="ZERO_SHOT_HELDOUT.json (required with --switch-pack): "
                            "held-out worlds/direct family never enter training")
        q.add_argument("--heldout-family-policy", choices=("drop", "refuse"),
                       default="drop",
                       help="items with held-out-family direct CAV segments: drop "
                            "(default) or refuse the run; no bypass")
        q.add_argument("--heldout-max-drop-frac", type=float, default=0.85,
                       help="refuse if the held-family exclusion drops more than this "
                            "fraction of the switching items")
        q.add_argument("--switch-mix-ratio", type=float, default=0.5,
                       help="sampling share of the switching source (0<r<1)")
        q.add_argument("--event-oversample", type=float, default=2.0,
                       help="weight multiplier (>=1) of switching windows containing a "
                            "controller switch or a direct segment inside the stage horizon")
        q.add_argument("--jam-oversample", type=float, default=1.0,
                       help="weight multiplier (>=1) of training windows whose recorded "
                            "HDV speed drops below --jam-speed-mps in their future horizon "
                            "(default 1 = off)")
        q.add_argument("--jam-speed-mps", type=float, default=5.0,
                       help="jam threshold speed in m/s for --jam-oversample (default 5)")
        q.add_argument("--jam-cache-dir", default=None, metavar="DIR",
                       help="directory for the per-source jam-window sidecar JSON cache "
                            "(default: the run --out directory)")
        q.add_argument("--switch-val-window-list", default=None, metavar="PATH",
                       help="TRAIN-scoped monitoring list for the switching pack "
                            "(default: world split of --switch-window-list)")
        q.add_argument("--switch-val-frac", type=float, default=0.1,
                       help="fraction of switching worlds held out for monitoring "
                            "(0 = none); derived under <out>/switch_valsplit")
        q.add_argument("--switch-val-seed", type=int, default=0)
    return p


def _kept_summary(receipt: dict) -> str:
    """Compact ``pack kept/total`` summary of a dataset receipt (log line)."""
    kept = receipt.get("kept_by_pack", {})
    total = receipt.get("total_by_pack", {})
    parts = ", ".join(f"{p} {kept.get(p, 0)}/{total.get(p, 0)}" for p in total)
    return (f"kept windows total {receipt.get('n_records')}/"
            f"{receipt.get('n_corpus_windows')} | {parts}")


def _dataset(paths: list[str], identities: list[str], overlays: list[str] | None,
             horizon_steps: int, start_shift_prob: float = 0.0,
             exclude_keys: set[str] | None = None,
             select_windows: set[tuple[str, str]] | None = None,
             window_list: str | None = None,
             exclude_window_list: list[str] | None = None,
             entries_v2: bool = False, item_filter=None,
             history_s=None, history_stride_s=1.0) -> TrajectoryDataset:
    if not paths:
        raise ValueError("at least one --pack is required")
    if identities is None and len(paths) == 1 and paths[0].lower().endswith(".npz"):
        return TrajectoryDataset.from_pack(paths[0], history_s=history_s,
                                           history_stride_s=history_stride_s)
    if identities is None:
        raise ValueError("--identities is required for indexed TRAIN packs")
    if len(paths) != len(identities):
        raise ValueError("--identities must be supplied once per --pack")
    return TrajectoryDataset.from_real_packs(
        paths, identity_dirs=identities, overlays=overlays,
        horizon_steps=horizon_steps, start_shift_prob=start_shift_prob,
        exclude_keys=exclude_keys, select_windows=select_windows,
        window_list=window_list, exclude_window_list=exclude_window_list,
        entries_v2=entries_v2, item_filter=item_filter,
        history_s=history_s, history_stride_s=history_stride_s)


def _build_model(args, physics: PhysicsConfig, cfg: WmSTrainConfig) -> WmSModel:
    """WM-S physics model by default; the black-box baselines with their flags.

    ``--blackbox-traj`` selects the strong pure-learned HDV baseline; ``--black-box``
    selects the controller-id-embedding variant of WM-S; neither default is on.
    """
    if getattr(args, "blackbox_traj", False):
        if cfg.history_s is not None:
            raise ValueError("long history requires structured WM-S")
        return BlackBoxTrajModel(physics=physics)
    return WmSModel(physics=physics, black_box=cfg.black_box,
                    n_controllers=cfg.n_controllers,
                    controller_embed_dim=cfg.controller_embed_dim,
                    checkpoint_residual=cfg.checkpoint_residual,
                    rollout_mode=getattr(cfg, "rollout_mode", "residual"),
                    lane_head=getattr(cfg, "lane_head", "off"),
                    history_s=cfg.history_s, history_stride_s=cfg.history_stride_s)


def _check_switch_args(args) -> None:
    """All-or-nothing validation of the switching-data flags (fail closed)."""
    needed = {"--switch-identities": args.switch_identities,
              "--switch-window-list": args.switch_window_list,
              "--switch-timeline-index": args.switch_timeline_index,
              "--zero-shot-heldout": args.zero_shot_heldout}
    if not args.switch_pack:
        stray = [k for k, v in needed.items() if v]
        if stray:
            raise ValueError(f"{stray} given without --switch-pack")
        return
    missing = [k for k, v in needed.items() if not v]
    if missing:
        raise ValueError(f"--switch-pack requires {missing}")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        _check_switch_args(args)
        from .history import history_spec
        history_spec(args.history_s, args.history_stride_s)
        if args.history_s is not None and (args.black_box or args.blackbox_traj):
            raise ValueError("long history requires structured WM-S")
        if float(args.jam_oversample) < 1.0:
            raise ValueError("--jam-oversample must be >= 1")
        if float(args.jam_speed_mps) <= 0.0:
            raise ValueError("--jam-speed-mps must be positive")
        if args.outcomes:
            os.environ["WM_N1_TRAIN_OUTCOMES"] = str(args.outcomes)
        curriculum = tuple(args.unroll_s or (10.0, 30.0, 60.0, 120.0))
        max_steps = max(1, round(max(curriculum) / 0.5))
        train_window_list = args.window_list
        val_window_list = args.val_window_list
        if float(args.val_holdout_frac) > 0.0:
            if args.val_window_list:
                raise ValueError("--val-holdout-frac and --val-window-list are "
                                 "mutually exclusive")
            if not args.window_list:
                raise ValueError("--val-holdout-frac requires --window-list (the "
                                 "TRAIN corpus window list)")
            from .val_split import materialise_holdout

            train_window_list, val_window_list = materialise_holdout(
                args.window_list, args.out, frac=float(args.val_holdout_frac),
                seed=int(args.val_holdout_seed))
        num_workers = args.num_workers
        gpu_share_note = None
        if args.gpu_share:
            if num_workers is None:
                num_workers = 4
            gpu_share_note = {
                "enabled": True, "num_workers": int(num_workers),
                "note": ("gpu-share mode: run up to 3 trainings (num-workers 4) on "
                         "one 48 GB GPU; keep each process's memory modest")}
        from .train_guards import TrainingWindowGuard

        heldout = None
        if args.switch_pack:
            from .mixdata import HeldOutGuard

            heldout = HeldOutGuard.load(args.zero_shot_heldout)
        window_guard = (TrainingWindowGuard(horizon_steps=max_steps, heldout=heldout,
                                           leave_pi_wall_out=args.leave_pi_wall_out)
                        if heldout or args.leave_pi_wall_out else None)
        # A start-shift must not introduce held controller exposure after filtering.
        dataset = _dataset(args.pack, args.identities, args.overlay, max_steps,
                           0.0 if args.leave_pi_wall_out else float(args.start_shift_prob),
                           window_list=train_window_list,
                           exclude_window_list=args.exclude_window_list,
                           entries_v2=bool(args.entries_v2), item_filter=window_guard,
                           history_s=args.history_s, history_stride_s=args.history_stride_s)
        print("train " + _kept_summary(dataset.receipt), file=sys.stderr)
        val_dataset = None
        if val_window_list:
            val_dataset = _dataset(args.pack, args.identities, args.overlay,
                                   max_steps, 0.0, window_list=val_window_list,
                                   exclude_window_list=args.exclude_window_list,
                                   entries_v2=bool(args.entries_v2), item_filter=window_guard,
                                   history_s=args.history_s,
                                   history_stride_s=args.history_stride_s)
            print("val   " + _kept_summary(val_dataset.receipt), file=sys.stderr)
        if args.switch_pack:
            from . import mixdata

            switch = mixdata.build_switch_datasets(
                pack=args.switch_pack, identities=args.switch_identities,
                window_list=args.switch_window_list,
                index_path=args.switch_timeline_index,
                heldout_path=args.zero_shot_heldout, horizon_steps=max_steps,
                entries_v2=bool(args.entries_v2),
                family_policy=args.heldout_family_policy,
                max_drop_frac=float(args.heldout_max_drop_frac),
                val_holdout_frac=float(args.switch_val_frac),
                val_holdout_seed=int(args.switch_val_seed),
                val_window_list=args.switch_val_window_list, work_dir=args.out,
                leave_pi_wall_out=bool(args.leave_pi_wall_out),
                history_s=args.history_s, history_stride_s=args.history_stride_s)
            dataset, val_dataset = mixdata.mix_datasets(
                dataset, switch, switch_ratio=float(args.switch_mix_ratio),
                event_oversample=float(args.event_oversample), orig_val=val_dataset,
                jam_oversample=float(args.jam_oversample),
                jam_speed_mps=float(args.jam_speed_mps),
                jam_cache_dir=(args.jam_cache_dir or args.out))
            dataset.receipt["stage_sampling"] = dataset.describe_stages(curriculum)
            print("mix   " + json.dumps(
                {"n": len(dataset), "shares": dataset.shares,
                 "heldout": switch["receipt"]["heldout"],
                 "stage_sampling": dataset.receipt["stage_sampling"]}),
                file=sys.stderr)
        elif float(args.jam_oversample) > 1.0:
            from . import jamdata

            report = jamdata.compute_jam_flags(
                dataset, speed_mps=float(args.jam_speed_mps),
                cache_path=os.path.join(args.jam_cache_dir or args.out,
                                        "jam_original.json"),
                source_key=jamdata.dataset_source_key(dataset))
            report["oversample"] = float(args.jam_oversample)
            dataset = jamdata.JamOversampledDataset(
                dataset, report["flags"], oversample=float(args.jam_oversample),
                report=report)
            dataset.receipt["stage_sampling"] = dataset.describe_stages(curriculum)
            print("jam   " + json.dumps(
                {"n": len(dataset), "n_jam": int(report["n_jam"]),
                 "cache_hit": bool(report["cache_hit"])}), file=sys.stderr)
        if args.merge_cal:
            overrides = {"sumo_fidelity": bool(args.sumo_fidelity),
                         "sumo_idm": not bool(args.geometry_only),
                         "merge_cal": True}
            if args.merge_cal_params:
                physics = PhysicsConfig.from_params_file(args.merge_cal_params,
                                                         **overrides)
            else:
                physics = PhysicsConfig(**overrides)
        else:
            physics = PhysicsConfig(sumo_fidelity=bool(args.sumo_fidelity),
                                    sumo_idm=not bool(args.geometry_only))
        stage_steps = None
        if args.stage_steps:
            stage_steps = tuple(int(x) for x in args.stage_steps.split(","))
            if len(stage_steps) != len(curriculum):
                raise ValueError(
                    f"--stage-steps has {len(stage_steps)} values but --unroll-s has "
                    f"{len(curriculum)}; one positive count per curriculum stage")
        cfg = WmSTrainConfig(device=args.device, threads=int(args.threads),
                             history_s=args.history_s,
                             history_stride_s=args.history_stride_s,
                             stage_steps=stage_steps,
                             num_workers=(0 if num_workers is None else int(num_workers)),
                             lr=float(args.lr), seed=int(args.seed),
                             out_dir=args.out, unroll_curriculum_s=curriculum,
                             truncation_steps=int(args.truncation_steps),
                             black_box=bool(args.black_box),
                             n_controllers=int(args.n_controllers),
                             controller_embed_dim=int(args.controller_embed_dim),
                             checkpoint_residual=bool(args.checkpoint_residual),
                             entries_v2=bool(args.entries_v2),
                             entry_pad=int(args.entry_pad),
                             traffic_loss=bool(args.traffic_loss),
                             traffic_weight=float(args.traffic_weight),
                             traffic_bin_s=float(args.traffic_bin_s),
                             traffic_ramp_start_s=float(args.traffic_ramp_start_s),
                             traffic_ramp_end_s=float(args.traffic_ramp_end_s),
                             state_noise_std_m=float(args.state_noise_std_m),
                             state_noise_std_mps=float(args.state_noise_std_mps),
                             projection_enabled=bool(args.projection_enabled),
                             projection_min_gap_m=float(args.projection_min_gap_m),
                             residual_cap_mps2=(None if args.residual_cap_mps2 is None
                                                else float(args.residual_cap_mps2)),
                             residual_soft_bound_mps2=(
                                 None if args.residual_soft_bound_mps2 is None
                                 else float(args.residual_soft_bound_mps2)),
                             residual_soft_bound_neg_mps2=(
                                 None if args.residual_soft_bound_neg_mps2 is None
                                 else float(args.residual_soft_bound_neg_mps2)),
                             residual_mag_weight=float(args.residual_mag_weight),
                             lane_rate_weight=float(args.lane_rate_weight),
                             lane_rate_target=float(args.lane_rate_target),
                             lane_logit_cap=(None if args.lane_logit_cap is None
                                             else float(args.lane_logit_cap)),
                             stability_weight=float(args.stability_weight),
                             stability_margin=float(args.stability_margin),
                             stability_probes=int(args.stability_probes),
                             conservation_enabled=bool(args.conservation_metric),
                             conservation_weight=float(args.conservation_weight),
                             loss_share_logging=bool(args.loss_share_logging),
                             switch_window_fraction=float(args.switch_window_fraction),
                             cav_drive=str(args.cav_drive),
                             window_list=train_window_list,
                             exclude_window_list=args.exclude_window_list,
                             val_window_list=val_window_list,
                             select_by_long_horizon=bool(args.select_by_long_horizon),
                             val_every=int(args.val_every),
                             steps_per_epoch=(int(args.steps) if args.steps is not None
                                               else (2 if args.command == "dry-run" else 2000)),
                             batch_size=int(args.batch_size),
                               rollout_mode=str(args.rollout_mode),
                               lane_head=str(args.lane_head),
                               residual_magnitude_weight=(
                                   None if args.residual_magnitude_weight is None
                                   else float(args.residual_magnitude_weight)),
                                driver_sidecar=args.driver_sidecar,
                                asym_speed_alpha=float(args.asym_speed_loss),
                                decel_weight=float(args.decel_weight),
                                decel_threshold_mps2=float(args.decel_threshold),
                                propagation_weight=float(args.propagation_loss),
                                prop_window_s=float(args.prop_window),
                                prop_max_gap_m=args.prop_max_gap_m,
                                residual_cap_neg_mps2=(
                                    None if args.residual_cap_neg_mps2 is None
                                    else float(args.residual_cap_neg_mps2)))
        model = _build_model(args, physics, cfg)
        if args.command == "dry-run":
            result = dry_run(dataset, model=model, config=cfg)
        else:
            result = WmSTrainer(model, cfg).fit(dataset, val_dataset=val_dataset)
        if gpu_share_note is not None:
            result["gpu_share"] = gpu_share_note
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (TrainDataRefusal, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
