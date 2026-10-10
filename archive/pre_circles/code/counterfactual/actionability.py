"""Post-pilot diagnostics for a fixed counterfactual escalation action.

This module does not alter the oracle or select a deployment rule.  It exposes
why a frozen action passed or failed so a later treatment/supervisor revision
can be declared before another prospective cohort is generated.
"""

from __future__ import annotations

from collections import Counter
import math
from statistics import mean
from typing import Any, Mapping, Sequence

from counterfactual.headroom import TreatmentRepairCriteria


def dose_qualified(
    row: Mapping[str, Any],
    criteria: TreatmentRepairCriteria = TreatmentRepairCriteria(),
) -> bool:
    dose = row.get("treatment_dose") or {}
    geometry = dose.get("wall_geometry") or {}
    compact = geometry.get("geometry_mode") == "compact_moving"
    return bool(
        dose.get("activation_feasible")
        and float(dose.get("realized_duration_fraction", 0.0)) >= criteria.min_case_dose_fraction
        and float(dose.get("lane_coverage_fraction", 0.0))
        >= criteria.min_case_lane_coverage_fraction
        and (
            not compact
            or (
                bool(dose.get("assembly_completed"))
                and float(dose.get("holding_fraction_of_override", 0.0))
                >= criteria.min_holding_fraction_of_override
                and float(geometry.get("fraction_within_configured_span", 0.0))
                >= criteria.min_geometry_within_span_fraction
            )
        )
    )


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _feature_separation(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[bool]
) -> dict[str, Any]:
    positive = [row for row, label in zip(rows, labels) if label]
    negative = [row for row, label in zip(rows, labels) if not label]
    if len(positive) < 2 or len(negative) < 2:
        return {
            "positive_case_count": len(positive),
            "negative_case_count": len(negative),
            "top": [],
        }
    keys = sorted(set.intersection(*(set(row.get("features", {})) for row in rows)))
    records = []
    for key in keys:
        left = [_number(row["features"].get(key)) for row in positive]
        right = [_number(row["features"].get(key)) for row in negative]
        if not all(math.isfinite(value) for value in left + right):
            continue
        combined = left + right
        center = mean(combined)
        variance = mean((value - center) ** 2 for value in combined)
        if variance <= 1e-12:
            continue
        standardized = (mean(left) - mean(right)) / math.sqrt(variance)
        records.append(
            {
                "feature": key,
                "positive_mean": mean(left),
                "negative_mean": mean(right),
                "standardized_difference": standardized,
                "abs_standardized_difference": abs(standardized),
            }
        )
    records.sort(key=lambda item: item["abs_standardized_difference"], reverse=True)
    return {
        "positive_case_count": len(positive),
        "negative_case_count": len(negative),
        "top": records[:20],
    }


def _direct_event_counts(row: Mapping[str, Any]) -> dict[str, int]:
    treatment = row["arms"]["1"]["outcomes"]
    hard_events = list((treatment.get("hard_brake_attribution") or {}).get("event_starts") or [])
    ttc_events = list((treatment.get("low_ttc_attribution") or {}).get("events") or [])

    def near_wall(event: Mapping[str, Any]) -> bool:
        distance = _number(event.get("nearest_controlled_cav_same_road_m"))
        return math.isfinite(distance) and distance <= 50.0

    return {
        "hard_event_starts_assembly_or_hold": sum(
            event.get("phase") in ("assembling", "holding") for event in hard_events
        ),
        "hard_event_starts_direct_wall_context": sum(
            bool(event.get("controlled_wall_cav"))
            or bool(event.get("leader_controlled_wall_cav"))
            or near_wall(event)
            for event in hard_events
        ),
        "low_ttc_events_assembly_or_hold": sum(
            event.get("phase") in ("assembling", "holding") for event in ttc_events
        ),
        "low_ttc_events_direct_wall_context": sum(
            bool(event.get("follower_controlled_wall_cav"))
            or bool(event.get("leader_controlled_wall_cav"))
            or near_wall(event)
            for event in ttc_events
        ),
    }


def _service_noninferior(result: Mapping[str, Any]) -> bool:
    """Read both the legacy and current oracle spelling."""
    return bool(
        result.get(
            "service_noninferior",
            result.get("traffic_noninferior", False),
        )
    )


def summarize_actionability(
    rows: Sequence[Mapping[str, Any]],
    criteria: TreatmentRepairCriteria = TreatmentRepairCriteria(
        require_collapse_prevention_diversity=False
    ),
) -> dict[str, Any]:
    """Describe feasibility, safety, and benefit heterogeneity of one action."""
    feasible = [
        row for row in rows if bool((row.get("treatment_dose") or {}).get("activation_feasible"))
    ]
    completed = [row for row in feasible if dose_qualified(row, criteria)]
    classified = [row["safety_service_oracle"] for row in completed]
    safe_labels = [bool(result["safety_admissible"]) for result in classified]
    selected_labels = [bool(result["wall_selected"]) for result in classified]
    service_labels = [_service_noninferior(result) for result in classified]

    by_step: dict[str, Any] = {}
    for step in sorted({int(row["decision_step"]) for row in rows}):
        step_rows = [row for row in rows if int(row["decision_step"]) == step]
        step_feasible = [row for row in feasible if int(row["decision_step"]) == step]
        step_completed = [row for row in completed if int(row["decision_step"]) == step]
        step_oracle = [row["safety_service_oracle"] for row in step_completed]
        by_step[str(step)] = {
            "case_count": len(step_rows),
            "feasible_count": len(step_feasible),
            "dose_qualified_count": len(step_completed),
            "safe_count": sum(bool(result["safety_admissible"]) for result in step_oracle),
            "service_noninferior_count": sum(
                _service_noninferior(result) for result in step_oracle
            ),
            "selected_count": sum(bool(result["wall_selected"]) for result in step_oracle),
            "selected_seed_count": len(
                {
                    int(row["seed"])
                    for row in step_completed
                    if bool(row["safety_service_oracle"]["wall_selected"])
                }
            ),
        }

    refusal_reasons = Counter(
        reason for result in classified for reason in result.get("refusal_reasons", [])
    )
    unsafe_cases = []
    for row, result in zip(completed, classified):
        if bool(result["safety_admissible"]):
            continue
        unsafe_cases.append(
            {
                "case_id": str(row["case_id"]),
                "seed": int(row["seed"]),
                "decision_step": int(row["decision_step"]),
                "effects": dict(result["effects"]),
                "direct_attribution": _direct_event_counts(row),
            }
        )

    return {
        "case_count": len(rows),
        "independent_seed_count": len({int(row["seed"]) for row in rows}),
        "feasible_case_count": len(feasible),
        "dose_qualified_case_count": len(completed),
        "dose_qualified_fraction": len(completed) / len(feasible) if feasible else 0.0,
        "safe_case_count": sum(safe_labels),
        "safety_admissible_fraction": (sum(safe_labels) / len(completed) if completed else 0.0),
        "service_noninferior_case_count": sum(service_labels),
        "selected_case_count": sum(selected_labels),
        "selected_seed_count": len(
            {int(row["seed"]) for row, selected in zip(completed, selected_labels) if selected}
        ),
        "refusal_reason_counts": dict(sorted(refusal_reasons.items())),
        "by_decision_step": by_step,
        "predecision_separation": {
            "safe_vs_unsafe_completed": _feature_separation(completed, safe_labels),
            "selected_vs_not_selected_completed": _feature_separation(completed, selected_labels),
            "service_noninferior_vs_inferior_completed": _feature_separation(
                completed, service_labels
            ),
        },
        "unsafe_completed_cases": unsafe_cases,
        "interpretation_guardrail": (
            "development diagnostic only; do not select a final rule or claim "
            "prospective predictor performance from this report"
        ),
    }
