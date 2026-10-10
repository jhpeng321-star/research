"""Post-run dose and causal-chain audit for the frozen v4 pulse experiment."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.io import atomic_write_json
from idm.action_local import classify_effect


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else None


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


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
    reason_counts: Counter[str] = Counter()
    for case in summary["cases"]:
        effect_by_arm = {effect["arm"]: effect for effect in case["effects"]}
        zero_branch = branches[f"{case['case_id']}.zero"]
        zero_commands = {
            (int(step["relative_step"]), command["vehicle_id"]): command
            for step in zero_branch["command_trace"]
            for command in step["commands"]
        }
        for arm, effect in sorted(effect_by_arm.items()):
            branch = branches[f"{case['case_id']}.{arm}"]
            residual = float(branch["requested_residual_mps2"])
            pulse_steps = int(summary["protocol"]["outcome_contract"]["pulse_steps"])
            treated_ids = set(branch.get("treated_role_vehicle_ids") or [])
            role_commands = [
                (int(step["relative_step"]), command)
                for step in branch["command_trace"]
                for command in step["commands"]
                if command.get("treatment_role", False)
            ]
            treated_commands = [
                (int(step["relative_step"]), command)
                for step in branch["command_trace"]
                for command in step["commands"]
                if command["vehicle_id"] in treated_ids
            ]
            pulse_role = [(step, command) for step, command in role_commands if step <= pulse_steps]
            realized_steps = len({step for step, _ in pulse_role})
            intended_requested_abs = abs(residual) * pulse_steps
            delivered_requested_abs = sum(
                abs(float(command["requested_residual_mps2"])) for _, command in pulse_role
            )
            filtered_pulse_abs = sum(
                abs(float(command["executed_residual_mps2"])) for _, command in pulse_role
            )
            filtered_total_abs = sum(
                abs(float(command["executed_residual_mps2"])) for _, command in treated_commands
            )
            paired_effective_values = []
            paired_filtered_values = []
            paired_command_count = 0
            unmatched_command_count = 0
            projection_deltas = []
            projected = 0
            branch_reasons: Counter[str] = Counter()
            for relative_step, command in treated_commands:
                executed_acceleration = command.get("executed_acceleration_mps2")
                if executed_acceleration is None:
                    continue
                desired = float(command["desired_acceleration_mps2"])
                projection_deltas.append(float(executed_acceleration) - desired)
                zero_command = zero_commands.get((relative_step, command["vehicle_id"]))
                zero_executed_acceleration = (
                    zero_command.get("executed_acceleration_mps2")
                    if zero_command is not None
                    else None
                )
                if zero_executed_acceleration is None:
                    unmatched_command_count += 1
                else:
                    paired_command_count += 1
                    paired_effective_values.append(
                        float(executed_acceleration) - float(zero_executed_acceleration)
                    )
                    paired_filtered_values.append(
                        float(command["executed_residual_mps2"])
                        - float(zero_command["executed_residual_mps2"])
                    )
                if command.get("projected"):
                    projected += 1
                for reason in command.get("projection_reasons") or []:
                    branch_reasons[str(reason)] += 1
                    reason_counts[str(reason)] += 1
            paired_effective_total_abs = sum(abs(value) for value in paired_effective_values)
            paired_filtered_total_abs = sum(abs(value) for value in paired_filtered_values)
            hard_constraints = {
                key: bool(value)
                for key, value in effect["constraints"].items()
                if key
                not in {
                    "nonrole_zero_residual",
                    "nonanchor_zero_residual",
                    "pulse_role_dose_at_least_80pct",
                }
            }
            minimum_dose_steps = math.ceil(0.8 * pulse_steps)
            dose_qualified = realized_steps >= minimum_dose_steps
            hard_pass = all(hard_constraints.values())
            classification = (
                classify_effect(
                    float(effect["safety_composite_reference_units"]),
                    deadband=float(summary["frozen_deadband"]),
                    constraints_pass=hard_pass,
                )
                if dose_qualified
                else "insufficient_dose"
            )
            geometry = branch["decision_geometry"]
            rows.append(
                {
                    "case_id": case["case_id"],
                    "demand": int(case["demand"]),
                    "seed": int(case["seed"]),
                    "landmark": case["landmark"],
                    "decision_step": int(case["decision_step"]),
                    "arm": arm,
                    "requested_residual_mps2": residual,
                    "intended_requested_abs_residual_seconds": intended_requested_abs,
                    "delivered_requested_abs_residual_seconds": delivered_requested_abs,
                    "filtered_pulse_abs_residual_seconds": filtered_pulse_abs,
                    "filtered_total_abs_residual_seconds": filtered_total_abs,
                    "paired_effective_abs_residual_seconds": paired_effective_total_abs,
                    "mean_paired_effective_residual_mps2": (
                        float(np.mean(paired_effective_values))
                        if paired_effective_values
                        else None
                    ),
                    "mean_abs_paired_effective_residual_mps2": (
                        float(np.mean(np.abs(paired_effective_values)))
                        if paired_effective_values
                        else None
                    ),
                    "maximum_abs_paired_effective_residual_mps2": (
                        max(abs(value) for value in paired_effective_values)
                        if paired_effective_values
                        else None
                    ),
                    "paired_filtered_abs_residual_seconds": paired_filtered_total_abs,
                    "paired_command_count": paired_command_count,
                    "unmatched_command_count": unmatched_command_count,
                    "paired_command_fraction": _safe_ratio(
                        paired_command_count, paired_command_count + unmatched_command_count
                    ),
                    "mean_abs_projection_delta_mps2": (
                        float(np.mean(np.abs(projection_deltas))) if projection_deltas else None
                    ),
                    "requested_role_steps": pulse_steps,
                    "minimum_qualified_role_steps": minimum_dose_steps,
                    "realized_role_steps": realized_steps,
                    "treatment_role_controlled_cav_seconds": len(role_commands),
                    "treated_vehicle_ids": sorted(treated_ids),
                    "handoff_count": max(len(treated_ids) - 1, 0),
                    "projection_count": projected,
                    "projection_frequency": (
                        projected / len(treated_commands) if treated_commands else None
                    ),
                    "projection_reason_counts": dict(branch_reasons),
                    "request_delivery_fraction": _safe_ratio(
                        delivered_requested_abs, intended_requested_abs
                    ),
                    "filter_pulse_retention_fraction": _safe_ratio(
                        filtered_pulse_abs, delivered_requested_abs
                    ),
                    "paired_effective_to_filtered_fraction": _safe_ratio(
                        paired_effective_total_abs, paired_filtered_total_abs
                    ),
                    "absorber_runway_m": geometry.get("runway_m"),
                    "initial_follower_present": geometry.get("follower_id") is not None,
                    "initial_follower_gap_m": geometry.get("follower_gap_m"),
                    "safety_composite_reference_units": float(
                        effect["safety_composite_reference_units"]
                    ),
                    "severe_braking_effect_reference_units": float(
                        effect["severe_braking_effect_reference_units"]
                    ),
                    "wave_effect_reference_units": float(effect["wave_effect_reference_units"]),
                    "low_ttc_excess_steps": int(effect["low_ttc_excess_steps"]),
                    "delay_rate_delta": effect["delay_rate_delta"],
                    "arrival_delta": int(effect["arrival_delta"]),
                    "merge_discharge_rate_delta": float(effect["merge_discharge_rate_delta"]),
                    "reward_delta": float(effect["reward_delta"]),
                    "hard_constraints": hard_constraints,
                    "hard_constraints_pass": hard_pass,
                    "dose_qualified": dose_qualified,
                    "classification": classification,
                }
            )
    qualified = [row for row in rows if row["dose_qualified"]]
    spatial_trace_available = bool(branches) and all(
        bool(branch.get("spatial_trace")) for branch in branches.values()
    )
    classification_counts = Counter(row["classification"] for row in rows)
    label_seed_coverage = {
        label: len({row["seed"] for row in qualified if row["classification"] == label})
        for label in ("helpful", "neutral", "harmful")
    }
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["demand"], row["landmark"], row["arm"])].append(row)
    grouped = {
        ".".join(map(str, key)): {
            "count": len(cell),
            "dose_qualified_count": sum(row["dose_qualified"] for row in cell),
            "classification_counts": dict(Counter(row["classification"] for row in cell)),
            "mean_request_delivery_fraction": _mean(cell, "request_delivery_fraction"),
            "mean_filter_pulse_retention_fraction": _mean(cell, "filter_pulse_retention_fraction"),
            "mean_paired_effective_to_filtered_fraction": _mean(
                cell, "paired_effective_to_filtered_fraction"
            ),
            "mean_safety_composite": _mean(cell, "safety_composite_reference_units"),
        }
        for key, cell in sorted(groups.items())
    }
    dose = np.asarray(
        [row["paired_effective_abs_residual_seconds"] for row in qualified], dtype=float
    )
    magnitude = np.asarray(
        [abs(row["safety_composite_reference_units"]) for row in qualified], dtype=float
    )
    dose_effect_rho = (
        float(spearmanr(dose, magnitude).statistic) if len(set(dose.tolist())) > 1 else None
    )
    report = {
        "schema": "idm_phase5_action_local_dose_audit/2",
        "source": str(args.summary.resolve()),
        "source_protocol_hash": summary["protocol_hash"],
        "treatment_branch_count": len(rows),
        "dose_qualified_branch_count": len(qualified),
        "dose_qualified_fraction": len(qualified) / len(rows) if rows else 0.0,
        "classification_counts": dict(classification_counts),
        "dose_qualified_classification_counts": dict(
            Counter(row["classification"] for row in qualified)
        ),
        "label_seed_coverage": label_seed_coverage,
        "dose_chain_means": {
            "intended_requested_abs_residual_seconds": _mean(
                rows, "intended_requested_abs_residual_seconds"
            ),
            "delivered_requested_abs_residual_seconds": _mean(
                rows, "delivered_requested_abs_residual_seconds"
            ),
            "filtered_pulse_abs_residual_seconds": _mean(
                rows, "filtered_pulse_abs_residual_seconds"
            ),
            "filtered_total_abs_residual_seconds": _mean(
                rows, "filtered_total_abs_residual_seconds"
            ),
            "paired_effective_abs_residual_seconds": _mean(
                rows, "paired_effective_abs_residual_seconds"
            ),
            "paired_filtered_abs_residual_seconds": _mean(
                rows, "paired_filtered_abs_residual_seconds"
            ),
            "paired_command_fraction": _mean(rows, "paired_command_fraction"),
            "request_delivery_fraction": _mean(rows, "request_delivery_fraction"),
            "filter_pulse_retention_fraction": _mean(rows, "filter_pulse_retention_fraction"),
            "paired_effective_to_filtered_fraction": _mean(
                rows, "paired_effective_to_filtered_fraction"
            ),
            "realized_role_steps": _mean(rows, "realized_role_steps"),
            "handoff_count": _mean(rows, "handoff_count"),
            "projection_frequency": _mean(rows, "projection_frequency"),
        },
        "projection_reason_counts": dict(reason_counts),
        "effect_magnitude_vs_executed_dose_spearman": dose_effect_rho,
        "grouped_results": grouped,
        "identity_and_safety": {
            "all_branch_decision_identity": summary["gates"]["all_branch_decision_identity"],
            "all_discovery_identity": summary["gates"]["all_discovery_identity"],
            "all_horizons_complete": summary["gates"]["all_horizons_complete"],
            "zero_collision_observations": summary["gates"]["zero_collision_observations"],
            "zero_teleport_observations": summary["gates"]["zero_teleport_observations"],
        },
        "spatial_observability": {
            "requested_filtered_projected_command_chain": True,
            "absorber_speed_acceleration_jerk_trace": spatial_trace_available,
            "immediate_follower_time_series": spatial_trace_available,
            "influence_region_time_series": spatial_trace_available,
            "lane_time_series": spatial_trace_available,
            "corridor_endpoints": True,
            "merge_service_endpoints": True,
            "interpretation": (
                "Spatial traces are available for local-to-corridor attribution."
                if spatial_trace_available
                else "This artifact can locate command-chain attenuation and corridor response, "
                "but cannot distinguish delayed/diluted follower and lane response."
            ),
        },
        "rows": rows,
    }
    atomic_write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
