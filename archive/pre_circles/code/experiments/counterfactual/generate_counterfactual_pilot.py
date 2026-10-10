"""Generate paired SUMO cases for treatment-effect analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from counterfactual.outcomes import OutcomeDefinition
from counterfactual.pilot import generate_paired_case, summarize_generation


from experiments import REPO_ROOT

PROJECT = REPO_ROOT
DEFAULT_SUMO_ROOT = Path(r"D:\research\highway_merge_shepherd")
DEFAULT_SUMO_EXE = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find a causal warning landmark, replay matched counterfactual "
            "branches, and label their independent future outcomes."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--search-steps", type=int, default=360)
    parser.add_argument(
        "--decision-steps",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional predeclared fixed-step grid. When omitted, use the first "
            "causal health-warning landmark."
        ),
    )
    parser.add_argument("--outcome-steps", type=int, default=300)
    parser.add_argument("--treatment-steps", type=int, default=120)
    parser.add_argument(
        "--actions",
        type=int,
        nargs="+",
        choices=[0, 1, 2],
        default=[0, 1, 2],
        help=(
            "Counterfactual arms to replay. Use '--actions 0 1' for the "
            "off-versus-safe-soft gate and faster generation."
        ),
    )
    parser.add_argument("--min-decision-step", type=int, default=60)
    parser.add_argument("--health-threshold", type=float, default=0.45)
    parser.add_argument("--allow-positive-trend", action="store_true")
    parser.add_argument(
        "--min-inband-scaled",
        type=float,
        default=0.0,
        help=(
            "Optional availability sensitivity filter; 0.2 corresponds to one "
            "CAV. The causal-landmark default is 0.0."
        ),
    )
    parser.add_argument("--cav-penetration", type=float, default=0.2)
    parser.add_argument("--sumo-step-length", type=float, default=0.5)
    parser.add_argument(
        "--soft-wall-tau",
        type=float,
        default=None,
        help=(
            "Optional soft-rung extra headway for a preserved-baseline ablation; "
            "the default keeps the supervisor's configured value."
        ),
    )
    parser.add_argument("--queue-speed-threshold", type=float, default=15.0)
    parser.add_argument("--congestion-queue-count", type=int, default=5)
    parser.add_argument("--congestion-speed-threshold", type=float, default=15.0)
    parser.add_argument("--sustain-steps", type=int, default=60)
    parser.add_argument("--ttc-threshold", type=float, default=1.5)
    parser.add_argument("--hard-brake-threshold", type=float, default=-4.5)
    parser.add_argument("--sumo-project-root", type=Path, default=DEFAULT_SUMO_ROOT)
    parser.add_argument("--sumo-config", default="osm.hyper.sumocfg")
    parser.add_argument(
        "--sumo-binary",
        default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "runs" / "probe" / "counterfactual_pilot.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT / "runs" / "probe" / "counterfactual_pilot.summary.json",
    )
    return parser.parse_args()


def _write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def main() -> int:
    args = parse_args()
    definition = OutcomeDefinition(
        queue_speed_threshold_mps=args.queue_speed_threshold,
        congestion_queue_count=args.congestion_queue_count,
        congestion_speed_threshold_mps=args.congestion_speed_threshold,
        sustain_steps=args.sustain_steps,
        ttc_threshold_s=args.ttc_threshold,
        hard_brake_threshold_mps2=args.hard_brake_threshold,
    )
    output = args.output.resolve()
    rows: list[Dict[str, Any]] = []
    work_items = (
        [(seed, step) for seed in args.seeds for step in args.decision_steps]
        if args.decision_steps
        else [(seed, None) for seed in args.seeds]
    )
    for index, (seed, fixed_step) in enumerate(work_items, start=1):
        selection = f"fixed_step={fixed_step}" if fixed_step is not None else "warning search"
        print(
            f"[{index}/{len(work_items)}] seed={seed}, {selection}: branching",
            flush=True,
        )
        row = generate_paired_case(
            project_root=args.sumo_project_root,
            sumo_binary=args.sumo_binary,
            sumo_config=args.sumo_config,
            seed=seed,
            search_steps=args.search_steps,
            outcome_steps=args.outcome_steps,
            treatment_steps=args.treatment_steps,
            decision_step_override=fixed_step,
            min_decision_step=args.min_decision_step,
            health_threshold=args.health_threshold,
            require_nonpositive_trend=not args.allow_positive_trend,
            min_inband_scaled=args.min_inband_scaled,
            cav_penetration=args.cav_penetration,
            sumo_step_length=args.sumo_step_length,
            soft_wall_tau=args.soft_wall_tau,
            actions=args.actions,
            definition=definition,
        )
        rows.append(row)
        _write_jsonl(output, rows)
        status = f"decision_step={row['decision_step']}" if row["eligible"] else row["reason"]
        print(f"[{index}/{len(work_items)}] seed={seed}: {status}", flush=True)

    summary = summarize_generation(rows)
    summary.update(
        {
            "schema_version": 2,
            "dataset": str(output),
            "configuration": {
                "search_steps": args.search_steps,
                "decision_steps": args.decision_steps,
                "outcome_steps": args.outcome_steps,
                "treatment_steps": args.treatment_steps,
                "actions": args.actions,
                "soft_wall_tau": args.soft_wall_tau,
                "outcome_definition": definition.to_dict(),
            },
        }
    )
    summary_output = args.summary_output.resolve()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {output}")
    print(f"wrote {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
