"""Dose, spatial-response, and family gates for the structured IDM screen."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from experiments.io import atomic_write_json


def _trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _paired_command_dose(control: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    zero = {
        (int(row["relative_step"]), row["vehicle_id"]): row
        for row in control["command_rows"]
    }
    preprojection = []
    executed = []
    matched = 0
    for row in action["command_rows"]:
        pair = zero.get((int(row["relative_step"]), row["vehicle_id"]))
        if pair is None:
            continue
        left = row.get("projected_acceleration_mps2")
        right = pair.get("projected_acceleration_mps2")
        if left is None or right is None:
            continue
        matched += 1
        preprojection.append(
            float(row["desired_acceleration_mps2"])
            - float(pair["desired_acceleration_mps2"])
        )
        executed.append(float(left) - float(right))
    requested_dose = sum(abs(value) for value in preprojection)
    executed_dose = sum(abs(value) for value in executed)
    return {
        "paired_command_count": matched,
        "paired_preprojection_acceleration_seconds": requested_dose,
        "paired_executed_acceleration_seconds": executed_dose,
        "projection_survival_fraction": (
            executed_dose / requested_dose if requested_dose > 1e-12 else None
        ),
        "mean_signed_executed_acceleration_delta_mps2": (
            float(np.mean(executed)) if executed else None
        ),
    }


def _spatial_effects(control: dict[str, np.ndarray], action: dict[str, np.ndarray]) -> dict:
    result = {}
    for level in ("immediate_follower", "influence_region", "lane", "corridor"):
        for metric in (
            "acceleration_energy_m2ps4",
            "integrated_negative_acceleration_mps",
            "severe_braking_count",
            "hard_braking_count",
            "low_ttc_count",
        ):
            key = f"{level}.{metric}"
            if key not in control or key not in action:
                continue
            valid = np.isfinite(control[key]) & np.isfinite(action[key])
            result[f"{key}_improvement_sum"] = (
                float(np.sum(control[key][valid] - action[key][valid]))
                if np.any(valid)
                else None
            )
        key = f"{level}.minimum_ttc_s"
        if key in control and key in action:
            valid = np.isfinite(control[key]) & np.isfinite(action[key])
            result[f"{key}_improvement_mean"] = (
                float(np.mean(action[key][valid] - control[key][valid]))
                if np.any(valid)
                else None
            )
    return result


def _trace_integrity(trace: dict[str, np.ndarray]) -> dict[str, bool]:
    times = trace["simulation_time_s"]
    causal = bool(len(times) and np.allclose(np.diff(times), 1.0, atol=1e-9))
    follower_association = True
    command_chain = True
    for raw in trace["event_json"]:
        row = json.loads(str(raw))
        for absorber in row.get("absorbers", []):
            command = absorber.get("command") or {}
            command_chain &= all(
                key in command
                for key in (
                    "analytical_acceleration_mps2",
                    "requested_action",
                    "filtered_action",
                    "desired_acceleration_mps2",
                    "projected_acceleration_mps2",
                    "projection_reasons",
                )
            )
            follower = absorber.get("immediate_follower")
            if follower is None:
                continue
            follower_association &= (
                follower.get("lane_id") == absorber.get("lane_id")
                and follower.get("lane_position_m") is not None
                and absorber.get("lane_position_m") is not None
                and float(follower["lane_position_m"]) < float(absorber["lane_position_m"])
                and follower.get("gap_m") is not None
                and float(follower["gap_m"]) >= 0.0
            )
    return {
        "causal_timestamps": causal,
        "correct_follower_lane_association": bool(follower_association),
        "complete_absorber_command_chain": bool(command_chain),
    }


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
    trace_integrity = []
    trace_cache = {}
    for branch_id, branch in branches.items():
        trace_cache[branch_id] = _trace(run_dir / branch["trace_path"])
        trace_integrity.append(_trace_integrity(trace_cache[branch_id]))
    rows = []
    for case in summary["cases"]:
        control = branches[f"{case['case_id']}.nominal"]
        control_trace = trace_cache[control["branch_id"]]
        for effect in case["effects"]:
            action = branches[f"{case['case_id']}.{effect['arm']}"]
            dose = _paired_command_dose(control, action)
            row = {
                "case_id": case["case_id"],
                "demand": int(case["demand"]),
                "seed": int(case["seed"]),
                "landmark": case["landmark"],
                "arm": effect["arm"],
                "family": effect["family"],
                "value": float(effect["value"]),
                "classification": effect["classification"],
                "constraints": effect["constraints"],
                "constraints_pass": bool(effect["constraints_pass"]),
                "dose_qualified": bool(effect["constraints"]["dose_at_least_80pct"]),
                "safety_composite_reference_units": float(
                    effect["safety_composite_reference_units"]
                ),
                "delay_rate_delta": float(effect["delay_rate_delta"]),
                "arrival_delta": int(effect["arrival_delta"]),
                "merge_discharge_count_delta": int(effect["merge_discharge_count_delta"]),
                "projection_reason_counts": action["projection_reason_counts"],
                **dose,
                "spatial_effects": _spatial_effects(
                    control_trace, trace_cache[action["branch_id"]]
                ),
            }
            rows.append(row)
    family_reports = {}
    for family in ("gain", "reference_offset"):
        family_rows = [row for row in rows if row["family"] == family]
        eligible = [row for row in family_rows if row["dose_qualified"]]
        meaningful = [
            row
            for row in eligible
            if abs(row["safety_composite_reference_units"]) > summary["frozen_deadband"]
        ]
        helpful_seeds = {
            row["seed"] for row in eligible if row["classification"] == "helpful"
        }
        nonbeneficial_seeds = {
            row["seed"] for row in eligible if row["classification"] != "helpful"
        }
        survivals = [
            row["projection_survival_fraction"]
            for row in eligible
            if row["projection_survival_fraction"] is not None
        ]
        spatially_reaching = [
            row
            for row in eligible
            if any(
                abs(float(value)) > 1e-6
                for key, value in row["spatial_effects"].items()
                if value is not None
                and key.startswith(("immediate_follower.", "influence_region.", "lane."))
            )
        ]
        winning_values = set()
        by_case: dict[str, list[dict]] = defaultdict(list)
        for row in eligible:
            by_case[row["case_id"]].append(row)
        for cells in by_case.values():
            best = max(cells, key=lambda row: row["safety_composite_reference_units"])
            if best["safety_composite_reference_units"] > 0.0:
                winning_values.add(best["value"])
        dose = [row["paired_executed_acceleration_seconds"] for row in eligible]
        effect_magnitude = [abs(row["safety_composite_reference_units"]) for row in eligible]
        rho = (
            float(spearmanr(dose, effect_magnitude).statistic)
            if len(set(dose)) > 1 and len(dose) >= 3
            else None
        )
        gates = {
            "eligible_cases_present": bool(eligible),
            "effects_beyond_noise_at_least_30pct": (
                len(meaningful) / len(eligible) >= 0.30 if eligible else False
            ),
            "helpful_across_three_seeds": len(helpful_seeds) >= 3,
            "nonbeneficial_across_three_seeds": len(nonbeneficial_seeds) >= 3,
            "two_actions_win": len(winning_values) >= 2,
            "median_projection_survival_at_least_50pct": (
                float(np.median(survivals)) >= 0.50 if survivals else False
            ),
            "dose_effect_spearman_at_least_0_40": rho is not None and rho >= 0.40,
            "spatial_response_reaches_beyond_absorber": (
                len(spatially_reaching) / len(eligible) >= 0.30 if eligible else False
            ),
            "service_constraints_all_pass": all(
                row["constraints"][key]
                for row in eligible
                for key in (
                    "delay_within_budget",
                    "arrival_within_budget",
                    "discharge_within_budget",
                )
            ),
        }
        grouped = {}
        for demand in sorted({row["demand"] for row in family_rows}):
            for value in sorted({row["value"] for row in family_rows}):
                cell = [
                    row
                    for row in family_rows
                    if row["demand"] == demand and row["value"] == value
                ]
                qualified_cell = [row for row in cell if row["dose_qualified"]]
                grouped[f"m{demand}.v{value:+g}"] = {
                    "branch_count": len(cell),
                    "dose_qualified_count": len(qualified_cell),
                    "classification_counts": dict(
                        Counter(row["classification"] for row in qualified_cell)
                    ),
                    "helpful_seeds": sorted(
                        {
                            row["seed"]
                            for row in qualified_cell
                            if row["classification"] == "helpful"
                        }
                    ),
                    "mean_safety_composite": (
                        float(
                            np.mean(
                                [
                                    row["safety_composite_reference_units"]
                                    for row in qualified_cell
                                ]
                            )
                        )
                        if qualified_cell
                        else None
                    ),
                    "service_failure_counts": dict(
                        Counter(
                            key
                            for row in qualified_cell
                            for key in (
                                "delay_within_budget",
                                "arrival_within_budget",
                                "discharge_within_budget",
                            )
                            if not row["constraints"][key]
                        )
                    ),
                }
        family_reports[family] = {
            "branch_count": len(family_rows),
            "dose_qualified_count": len(eligible),
            "meaningful_effect_count": len(meaningful),
            "classification_counts": dict(Counter(row["classification"] for row in eligible)),
            "helpful_seed_count": len(helpful_seeds),
            "nonbeneficial_seed_count": len(nonbeneficial_seeds),
            "winning_values": sorted(winning_values),
            "median_projection_survival_fraction": (
                float(np.median(survivals)) if survivals else None
            ),
            "dose_effect_magnitude_spearman": rho,
            "spatially_reaching_count": len(spatially_reaching),
            "grouped_by_demand_and_value": grouped,
            "gates": gates,
            "passes": all(gates.values()),
        }
    instrumentation_gates = {
        key: all(row[key] for row in trace_integrity)
        for key in trace_integrity[0]
    } if trace_integrity else {}
    report = {
        "schema": "idm_structured_authority_analysis/1",
        "source": str(args.summary.resolve()),
        "source_protocol_hash": summary["protocol_hash"],
        "instrumentation_gates": instrumentation_gates,
        "screen_integrity_gates": summary["gates"],
        "family_reports": family_reports,
        "mode_aliases": summary["protocol"]["mode_aliases"],
        "rows": rows,
    }
    atomic_write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
