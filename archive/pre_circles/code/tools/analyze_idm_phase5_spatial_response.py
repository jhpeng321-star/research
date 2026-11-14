"""Paired absorber-to-corridor response audit for the Phase-5 pulse pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.io import atomic_write_json

LEVELS = ("absorber", "immediate_follower", "influence_region", "lane", "corridor")
LOWER_IS_BETTER = (
    "acceleration_energy_m2ps4",
    "integrated_negative_acceleration_mps",
    "severe_braking_count",
    "speed_variance_m2ps2",
    "mean_abs_jerk_mps3",
)


def _trace_by_step(branch: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["relative_step"]): row for row in branch.get("spatial_trace", [])}


def _paired_level_effects(
    zero: dict[str, Any], action: dict[str, Any], *, horizon_steps: int
) -> dict[str, Any]:
    zero_trace = _trace_by_step(zero)
    action_trace = _trace_by_step(action)
    result: dict[str, Any] = {}
    for level in LEVELS:
        metrics: dict[str, Any] = {}
        for metric in LOWER_IS_BETTER:
            differences = []
            for step in range(1, horizon_steps + 1):
                zero_value = zero_trace.get(step, {}).get(level, {}).get(metric)
                action_value = action_trace.get(step, {}).get(level, {}).get(metric)
                if zero_value is not None and action_value is not None:
                    differences.append(float(zero_value) - float(action_value))
            metrics[f"{metric}_improvement_sum"] = (
                float(sum(differences)) if differences else None
            )
            metrics[f"{metric}_paired_steps"] = len(differences)
        ttc_differences = []
        speed_differences = []
        acceleration_differences = []
        for step in range(1, horizon_steps + 1):
            zero_level = zero_trace.get(step, {}).get(level, {})
            action_level = action_trace.get(step, {}).get(level, {})
            zero_ttc, action_ttc = zero_level.get("minimum_ttc_s"), action_level.get(
                "minimum_ttc_s"
            )
            if zero_ttc is not None and action_ttc is not None:
                ttc_differences.append(float(action_ttc) - float(zero_ttc))
            for metric, values in (
                ("mean_speed_mps", speed_differences),
                ("mean_acceleration_mps2", acceleration_differences),
            ):
                zero_value, action_value = zero_level.get(metric), action_level.get(metric)
                if zero_value is not None and action_value is not None:
                    values.append(float(action_value) - float(zero_value))
        metrics["minimum_ttc_improvement_mean_s"] = (
            float(np.mean(ttc_differences)) if ttc_differences else None
        )
        metrics["minimum_ttc_paired_steps"] = len(ttc_differences)
        metrics["mean_speed_response_mps"] = (
            float(np.mean(speed_differences)) if speed_differences else None
        )
        metrics["mean_acceleration_response_mps2"] = (
            float(np.mean(acceleration_differences)) if acceleration_differences else None
        )
        result[level] = metrics
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    run_dir = args.summary.resolve().parent
    branches = {
        row["branch_id"]: row
        for path in (run_dir / "branches").glob("*.json")
        for row in [json.loads(path.read_text(encoding="utf-8"))]
    }
    rows = []
    for case in summary["cases"]:
        zero = branches[f"{case['case_id']}.zero"]
        for effect in case["effects"]:
            action = branches[f"{case['case_id']}.{effect['arm']}"]
            rows.append(
                {
                    "case_id": case["case_id"],
                    "demand": int(case["demand"]),
                    "seed": int(case["seed"]),
                    "landmark": case["landmark"],
                    "arm": effect["arm"],
                    "requested_residual_mps2": float(action["requested_residual_mps2"]),
                    "classification": effect["classification"],
                    "safety_composite_reference_units": float(
                        effect["safety_composite_reference_units"]
                    ),
                    "levels": _paired_level_effects(
                        zero,
                        action,
                        horizon_steps=int(action["safety_horizon_steps"]),
                    ),
                }
            )
    report = {
        "schema": "idm_phase5_spatial_response/1",
        "source": str(args.summary.resolve()),
        "source_protocol_hash": summary["protocol_hash"],
        "treatment_branch_count": len(rows),
        "levels": list(LEVELS),
        "lower_is_better_metrics": list(LOWER_IS_BETTER),
        "rows": rows,
    }
    atomic_write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
