"""Run two identical off-control branches and verify deterministic replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from counterfactual.twin_replay import compare_traces, run_replay


from experiments import REPO_ROOT

PROJECT = REPO_ROOT
DEFAULT_SUMO_ROOT = Path(r"D:\research\highway_merge_shepherd")
DEFAULT_SUMO_EXE = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay two identical off->off SUMO branches from time zero and "
            "verify that their network-state fingerprints match at every step."
        )
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--decision-step", type=int, default=180)
    parser.add_argument("--horizon-steps", type=int, default=120)
    parser.add_argument("--cav-penetration", type=float, default=0.2)
    parser.add_argument("--sumo-step-length", type=float, default=0.5)
    parser.add_argument("--sumo-project-root", type=Path, default=DEFAULT_SUMO_ROOT)
    parser.add_argument(
        "--sumo-config",
        default="_light/osm.dense.light_m2400r400.sumocfg",
    )
    parser.add_argument(
        "--sumo-binary",
        default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"),
    )
    parser.add_argument("--precision-digits", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "runs" / "probe" / "counterfactual_twin_check.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    common = {
        "project_root": args.sumo_project_root,
        "sumo_binary": args.sumo_binary,
        "sumo_config": args.sumo_config,
        "seed": args.seed,
        "decision_step": args.decision_step,
        "horizon_steps": args.horizon_steps,
        "prefix_action": 0,
        "post_action": 0,
        "cav_penetration": args.cav_penetration,
        "sumo_step_length": args.sumo_step_length,
        "digits": args.precision_digits,
    }

    left = run_replay(**common)
    right = run_replay(**common)
    comparison = compare_traces(left, right)
    result = {
        "schema_version": 1,
        "purpose": "off-to-off deterministic replay prerequisite",
        "method": "full replay from time zero with common SUMO/CAV seed",
        "configuration": {
            key: str(value) if isinstance(value, Path) else value for key, value in common.items()
        },
        "comparison": comparison,
        "branch_a": left.summary(),
        "branch_b": right.summary(),
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")
    return 0 if comparison["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
