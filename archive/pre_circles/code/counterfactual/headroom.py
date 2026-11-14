"""Safety-first oracle rules for the v2 headroom pilot.

These are pilot screening rules, not the final locked-test success criteria.
They are frozen before seeds 111--122 are generated so the screen cannot be
redefined after seeing the paired outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import mean
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class HeadroomMargins:
    min_discharge_gain: int = 3
    max_discharge_loss: int = 2
    min_delay_gain_s: float = 200.0
    min_delay_gain_fraction: float = 0.02
    max_delay_loss_s: float = 300.0
    max_delay_loss_fraction: float = 0.05
    max_hard_brake_excess: int = 2
    max_hard_brake_excess_fraction: float = 0.10
    max_low_ttc_excess_steps: int = 2
    min_beneficial_seed_count: int = 3
    min_refusal_case_count: int = 3
    min_observable_standardized_difference: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TreatmentRepairCriteria:
    """Go/no-go thresholds for the corrected 12--20-seed action pilot."""

    min_case_dose_fraction: float = 0.80
    min_case_lane_coverage_fraction: float = 0.80
    min_dose_qualified_fraction: float = 0.80
    min_safety_admissible_fraction: float = 0.90
    min_beneficial_seed_count: int = 3
    min_nonbeneficial_seed_count: int = 3
    min_completed_seed_count: int = 6
    min_holding_fraction_of_override: float = 0.50
    min_geometry_within_span_fraction: float = 0.80
    require_collapse_prevention_diversity: bool = True
    min_collapse_prevention_seed_count: int = 3
    min_collapse_nonprevention_seed_count: int = 3
    min_collapse_prevention_case_fraction: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def classify_case(
    row: Mapping[str, Any],
    margins: HeadroomMargins = HeadroomMargins(),
) -> dict[str, Any]:
    """Apply the predeclared safety-first paired oracle to one case."""
    off = row["arms"]["0"]["outcomes"]
    wall = row["arms"]["1"]["outcomes"]
    engaged = bool(row["arms"]["1"].get("wall_engaged", False))

    delay_off = _number(off["incremental_delay_s"])
    delay_wall = _number(wall["incremental_delay_s"])
    delay_benefit = delay_off - delay_wall
    queue_benefit = _number(off["queue_area_vehicle_s"]) - _number(wall["queue_area_vehicle_s"])
    discharge_benefit = int(wall["merge_discharge_count"]) - int(off["merge_discharge_count"])
    hard_brake_excess = int(wall["hard_brake_vehicle_steps"]) - int(off["hard_brake_vehicle_steps"])
    low_ttc_excess = int(wall["low_ttc_exposure_steps"]) - int(off["low_ttc_exposure_steps"])
    collision_excess = int(wall["collision_unique_vehicle_count"]) - int(
        off["collision_unique_vehicle_count"]
    )
    teleport_excess = int(wall["starting_teleport_unique_vehicle_count"]) - int(
        off["starting_teleport_unique_vehicle_count"]
    )

    hard_brake_budget = max(
        margins.max_hard_brake_excess,
        math.ceil(margins.max_hard_brake_excess_fraction * int(off["hard_brake_vehicle_steps"])),
    )
    delay_gain_threshold = max(
        margins.min_delay_gain_s,
        margins.min_delay_gain_fraction * delay_off,
    )
    delay_loss_budget = max(
        margins.max_delay_loss_s,
        margins.max_delay_loss_fraction * delay_off,
    )
    safety_admissible = (
        collision_excess <= 0
        and teleport_excess <= 0
        and hard_brake_excess <= hard_brake_budget
        and low_ttc_excess <= margins.max_low_ttc_excess_steps
    )
    traffic_noninferior = (
        discharge_benefit >= -margins.max_discharge_loss and delay_benefit >= -delay_loss_budget
    )
    meaningful_benefit = (
        discharge_benefit >= margins.min_discharge_gain or delay_benefit >= delay_gain_threshold
    )
    wall_selected = engaged and safety_admissible and traffic_noninferior and meaningful_benefit

    refusal_reasons = []
    if not engaged:
        refusal_reasons.append("formation_or_runtime_veto")
    if not safety_admissible:
        refusal_reasons.append("safety_budget")
    if not traffic_noninferior:
        refusal_reasons.append("traffic_noninferiority")
    if not meaningful_benefit:
        refusal_reasons.append("insufficient_incremental_benefit")

    utility = delay_benefit / max(delay_gain_threshold, 1.0) + discharge_benefit / max(
        float(margins.min_discharge_gain), 1.0
    )
    return {
        "oracle_action": "DAMPING_PLUS_WALL" if wall_selected else "DAMPING_ONLY",
        "wall_selected": wall_selected,
        "wall_engaged": engaged,
        "safety_admissible": safety_admissible,
        "traffic_noninferior": traffic_noninferior,
        "meaningful_benefit": meaningful_benefit,
        "refusal_reasons": refusal_reasons,
        "effects": {
            "delay_benefit_s": delay_benefit,
            "queue_area_benefit_vehicle_s": queue_benefit,
            "merge_discharge_benefit": discharge_benefit,
            "hard_brake_excess_vehicle_steps": hard_brake_excess,
            "low_ttc_excess_steps": low_ttc_excess,
            "collision_excess_vehicle_count": collision_excess,
            "teleport_excess_vehicle_count": teleport_excess,
            "screening_utility": utility,
        },
        "budgets": {
            "delay_gain_threshold_s": delay_gain_threshold,
            "delay_loss_budget_s": delay_loss_budget,
            "hard_brake_excess_budget": hard_brake_budget,
            "low_ttc_excess_budget_steps": margins.max_low_ttc_excess_steps,
        },
    }


def _observable_separation(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[bool]
) -> dict[str, Any]:
    selected = [row for row, label in zip(rows, labels) if label]
    refused = [row for row, label in zip(rows, labels) if not label]
    candidates = []
    if len(selected) < 2 or len(refused) < 2:
        return {"passed": False, "reason": "fewer_than_two_cases_per_action", "top": []}

    keys = (
        sorted(
            set.intersection(*(set(row.get("features", {})) for row in rows if row.get("features")))
        )
        if rows and all(row.get("features") for row in rows)
        else []
    )
    for key in keys:
        left = [_number(row["features"].get(key)) for row in selected]
        right = [_number(row["features"].get(key)) for row in refused]
        left = [value for value in left if math.isfinite(value)]
        right = [value for value in right if math.isfinite(value)]
        if len(left) < 2 or len(right) < 2:
            continue
        all_values = left + right
        center = mean(all_values)
        variance = mean((value - center) ** 2 for value in all_values)
        if variance <= 1e-12:
            continue
        standardized = (mean(left) - mean(right)) / math.sqrt(variance)
        candidates.append(
            {
                "feature": key,
                "selected_mean": mean(left),
                "refused_mean": mean(right),
                "standardized_difference": standardized,
                "abs_standardized_difference": abs(standardized),
            }
        )
    candidates.sort(key=lambda item: item["abs_standardized_difference"], reverse=True)
    top = candidates[:10]
    return {
        "passed": bool(top) and top[0]["abs_standardized_difference"] >= 0.5,
        "reason": "univariate_pilot_screen_only",
        "top": top,
    }


def summarize_headroom(
    rows: Sequence[Mapping[str, Any]],
    margins: HeadroomMargins = HeadroomMargins(),
) -> dict[str, Any]:
    eligible = [row for row in rows if bool(row.get("eligible", False))]
    oracle = [classify_case(row, margins) for row in eligible]
    selections = [bool(result["wall_selected"]) for result in oracle]
    beneficial_seeds = sorted(
        {int(row["seed"]) for row, result in zip(eligible, oracle) if result["wall_selected"]}
    )
    refusal_count = sum(not selected for selected in selections)
    engaged_count = sum(bool(result["wall_engaged"]) for result in oracle)
    unsafe_count = sum(
        bool(result["wall_engaged"]) and not bool(result["safety_admissible"]) for result in oracle
    )
    observable = _observable_separation(eligible, selections)
    selected_utility = sum(
        float(result["effects"]["screening_utility"])
        for result in oracle
        if result["wall_selected"]
    )

    checks = {
        "both_actions_selected": bool(any(selections) and any(not x for x in selections)),
        "beneficial_walls_span_independent_seeds": (
            len(beneficial_seeds) >= margins.min_beneficial_seed_count
        ),
        "always_wall_has_higher_intervention_burden": (
            engaged_count > sum(selections) and refusal_count >= margins.min_refusal_case_count
        ),
        "oracle_materially_improves_over_damping": (
            len(beneficial_seeds) >= margins.min_beneficial_seed_count and selected_utility > 0.0
        ),
        "observable_pre_treatment_separation": bool(observable["passed"]),
    }
    return {
        "passed": bool(eligible) and all(checks.values()),
        "checks": checks,
        "margins": margins.to_dict(),
        "case_count": len(rows),
        "eligible_case_count": len(eligible),
        "independent_seed_count": len({int(row["seed"]) for row in eligible}),
        "wall_engaged_case_count": engaged_count,
        "wall_selected_case_count": sum(selections),
        "damping_selected_case_count": refusal_count,
        "safe_beneficial_seed_count": len(beneficial_seeds),
        "safe_beneficial_seeds": beneficial_seeds,
        "unsafe_engagement_count": unsafe_count,
        "selected_screening_utility_sum": selected_utility,
        "observable_separation": observable,
        "oracle_rows": oracle,
    }


def summarize_treatment_repair(
    rows: Sequence[Mapping[str, Any]],
    criteria: TreatmentRepairCriteria = TreatmentRepairCriteria(),
    *,
    effects_exceed_repeat_noise: Optional[bool] = None,
) -> dict[str, Any]:
    """Evaluate treatment consistency before any predictor-scale generation."""
    from counterfactual.safety_service_oracle import (
        classify_safety_service_case,
    )

    eligible = [row for row in rows if bool(row.get("eligible", False))]
    # Dose accounting starts when formation is declared feasible and treatment
    # begins.  A later assembly/geometry failure may make the causal row
    # ineligible for benefit modeling, but it must remain in the treatment-dose
    # denominator; otherwise failed treatments make delivery look better.
    supervisor_admitted = [
        row for row in rows if bool((row.get("predecision_supervisor") or {}).get("admitted", True))
    ]
    feasible = [
        row
        for row in supervisor_admitted
        if bool((row.get("treatment_dose") or {}).get("activation_feasible"))
    ]

    def dose_qualified(row: Mapping[str, Any]) -> bool:
        dose = row.get("treatment_dose") or {}
        geometry = dose.get("wall_geometry") or {}
        compact = geometry.get("geometry_mode") == "compact_moving"
        compact_phase_ok = not compact or (
            bool(dose.get("assembly_completed"))
            and float(dose.get("holding_fraction_of_override", 0.0))
            >= criteria.min_holding_fraction_of_override
            and float(geometry.get("fraction_within_configured_span", 0.0))
            >= criteria.min_geometry_within_span_fraction
        )
        return compact_phase_ok and (
            float(dose.get("realized_duration_fraction", 0.0)) >= criteria.min_case_dose_fraction
            and float(dose.get("lane_coverage_fraction", 0.0))
            >= criteria.min_case_lane_coverage_fraction
        )

    completed = [row for row in feasible if dose_qualified(row)]
    classifications = [
        (
            row,
            row.get("safety_service_oracle") or classify_safety_service_case(row),
        )
        for row in completed
    ]
    safe = [(row, result) for row, result in classifications if bool(result["safety_admissible"])]
    beneficial_seeds = sorted(
        {int(row["seed"]) for row, result in classifications if bool(result["wall_selected"])}
    )
    nonbeneficial_seeds = sorted(
        {int(row["seed"]) for row, result in classifications if not bool(result["wall_selected"])}
    )
    completed_seeds = sorted({int(row["seed"]) for row in completed})
    dose_qualified_fraction = len(completed) / len(feasible) if feasible else 0.0
    safety_admissible_fraction = len(safe) / len(completed) if completed else 0.0
    excess_collision_or_teleport = any(
        int(result["effects"]["collision_excess_vehicle_count"]) > 0
        or int(result["effects"]["teleport_excess_vehicle_count"]) > 0
        for _row, result in classifications
    )
    prevention_cohort = [
        row
        for row in completed
        if not bool((row.get("predecision_congestion") or {}).get("sustained_congestion", False))
    ]
    collapse_prevention_cases = [
        row
        for row in prevention_cohort
        if bool(row["arms"]["0"]["outcomes"].get("sustained_congestion"))
        and not bool(row["arms"]["1"]["outcomes"].get("sustained_congestion"))
    ]
    collapse_nonprevention_cases = [
        row for row in prevention_cohort if row not in collapse_prevention_cases
    ]
    collapse_prevention_seeds = sorted({int(row["seed"]) for row in collapse_prevention_cases})
    collapse_nonprevention_seeds = sorted(
        {int(row["seed"]) for row in collapse_nonprevention_cases}
    )
    collapse_prevention_fraction = (
        len(collapse_prevention_cases) / len(prevention_cohort) if prevention_cohort else 0.0
    )
    collapse_checks = {
        "collapse_prevention_spans_seeds": (
            len(collapse_prevention_seeds) >= criteria.min_collapse_prevention_seed_count
        ),
        "collapse_nonprevention_spans_seeds": (
            len(collapse_nonprevention_seeds) >= criteria.min_collapse_nonprevention_seed_count
        ),
        "collapse_prevention_case_fraction": (
            collapse_prevention_fraction >= criteria.min_collapse_prevention_case_fraction
        ),
    }
    checks = {
        "feasible_activations_exist": bool(feasible),
        "most_feasible_activations_receive_80pct_dose": (
            dose_qualified_fraction >= criteria.min_dose_qualified_fraction
        ),
        "zero_excess_collisions_or_teleports": not excess_collision_or_teleport,
        "safety_admissible_fraction": (
            safety_admissible_fraction >= criteria.min_safety_admissible_fraction
        ),
        "beneficial_completed_treatments_span_seeds": (
            len(beneficial_seeds) >= criteria.min_beneficial_seed_count
        ),
        "nonbeneficial_completed_treatments_span_seeds": (
            len(nonbeneficial_seeds) >= criteria.min_nonbeneficial_seed_count
        ),
        "completed_treatments_span_independent_seeds": (
            len(completed_seeds) >= criteria.min_completed_seed_count
        ),
        "meaningful_effects_exceed_repeat_noise": (
            effects_exceed_repeat_noise is True
        ),
    }
    if criteria.require_collapse_prevention_diversity:
        checks.update(collapse_checks)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "criteria": criteria.to_dict(),
        "eligible_case_count": len(eligible),
        "supervisor_admitted_case_count": len(supervisor_admitted),
        "feasible_activation_count": len(feasible),
        "dose_qualified_case_count": len(completed),
        "dose_qualified_seed_count": len(completed_seeds),
        "dose_qualified_seeds": completed_seeds,
        "dose_qualified_fraction": dose_qualified_fraction,
        "safety_admissible_case_count": len(safe),
        "safety_admissible_fraction": safety_admissible_fraction,
        "beneficial_completed_seed_count": len(beneficial_seeds),
        "beneficial_completed_seeds": beneficial_seeds,
        "nonbeneficial_completed_seed_count": len(nonbeneficial_seeds),
        "nonbeneficial_completed_seeds": nonbeneficial_seeds,
        "collapse_prevention_case_count": len(collapse_prevention_cases),
        "collapse_prevention_seed_count": len(collapse_prevention_seeds),
        "collapse_prevention_seeds": collapse_prevention_seeds,
        "collapse_nonprevention_case_count": len(collapse_nonprevention_cases),
        "collapse_nonprevention_seed_count": len(collapse_nonprevention_seeds),
        "collapse_nonprevention_seeds": collapse_nonprevention_seeds,
        "precollapse_completed_case_count": len(prevention_cohort),
        "collapse_prevention_case_fraction": collapse_prevention_fraction,
        "collapse_checks": collapse_checks,
        "repeat_noise_evidence_provided": effects_exceed_repeat_noise is not None,
    }
