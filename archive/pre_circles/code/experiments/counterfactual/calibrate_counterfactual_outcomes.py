"""Calibrate the independent congestion label on no-control SUMO traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from counterfactual.calibration import calibrate_outcome_definition
from counterfactual.twin_replay import run_replay


from experiments import REPO_ROOT

PROJECT = REPO_ROOT
DEFAULT_SUMO_ROOT = Path(r"D:\research\highway_merge_shepherd")
DEFAULT_SUMO_EXE = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank queue/speed/persistence definitions using only no-control "
            "delay and discharge separation."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(101, 111)))
    parser.add_argument("--decision-steps", type=int, nargs="+", default=[60, 120, 180, 240])
    parser.add_argument("--outcome-steps", type=int, default=300)
    parser.add_argument("--episode-steps", type=int, default=540)
    parser.add_argument(
        "--queue-speed-candidates", type=float, nargs="+", default=[10.0, 12.5, 15.0]
    )
    parser.add_argument(
        "--queue-count-candidates", type=int, nargs="+", default=[10, 20, 30, 40, 50]
    )
    parser.add_argument(
        "--congestion-speed-candidates",
        type=float,
        nargs="+",
        default=[10.0, 12.5, 15.0, 17.5],
    )
    parser.add_argument("--sustain-step-candidates", type=int, nargs="+", default=[30, 60, 90])
    parser.add_argument("--minimum-seeds-to-freeze", type=int, default=10)
    parser.add_argument("--ttc-threshold", type=float, default=1.5)
    parser.add_argument("--hard-brake-threshold", type=float, default=-4.5)
    parser.add_argument("--cav-penetration", type=float, default=0.2)
    parser.add_argument("--sumo-step-length", type=float, default=0.5)
    parser.add_argument("--sumo-project-root", type=Path, default=DEFAULT_SUMO_ROOT)
    parser.add_argument("--sumo-config", default="osm.hyper.sumocfg")
    parser.add_argument(
        "--sumo-binary",
        default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "runs" / "probe" / "counterfactual_outcome_calibration.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required_steps = max(args.decision_steps) + args.outcome_steps
    episode_steps = max(args.episode_steps, required_steps)
    traces = []
    for index, seed in enumerate(args.seeds, start=1):
        print(
            f"[{index}/{len(args.seeds)}] seed={seed}: no-control replay to step {episode_steps}",
            flush=True,
        )
        traces.append(
            run_replay(
                project_root=args.sumo_project_root,
                sumo_binary=args.sumo_binary,
                sumo_config=args.sumo_config,
                seed=seed,
                decision_step=0,
                horizon_steps=episode_steps,
                prefix_action=0,
                post_action=0,
                cav_penetration=args.cav_penetration,
                sumo_step_length=args.sumo_step_length,
            )
        )

    report = calibrate_outcome_definition(
        traces,
        decision_steps=args.decision_steps,
        outcome_steps=args.outcome_steps,
        queue_speed_thresholds=args.queue_speed_candidates,
        congestion_queue_counts=args.queue_count_candidates,
        congestion_speed_thresholds=args.congestion_speed_candidates,
        sustain_steps=args.sustain_step_candidates,
        ttc_threshold_s=args.ttc_threshold,
        hard_brake_threshold_mps2=args.hard_brake_threshold,
        minimum_seed_count_for_freeze=args.minimum_seeds_to_freeze,
    )
    report["scenario"] = {
        "sumo_project_root": str(args.sumo_project_root),
        "sumo_config": args.sumo_config,
        "sumo_binary": args.sumo_binary,
        "cav_penetration": args.cav_penetration,
        "sumo_step_length": args.sumo_step_length,
        "episode_steps": episode_steps,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "seed_count": report["seed_count"],
                "window_count": report["window_count"],
                "recommended_definition": report["recommended_definition"],
                "recommended_diagnostics": report["recommended_diagnostics"],
            },
            indent=2,
        )
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
