"""Safety/service oracle for counterfactual escalation development.

This oracle is a transparent development-stage amendment adopted after the
hyper/p20 and hyper/p30 outcomes were inspected.  It never overwrites the
original delay-oriented headroom label.  Delay is a noninferiority guardrail;
meaningful benefit comes from braking/TTC improvement or merge service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from counterfactual.dataset_contract import (
    MIN_MODEL_DEVELOPMENT_GROUPS,
    MIN_WALL_SELECTED_CASES,
    MIN_WALL_SELECTED_GROUPS,
)


ORACLE_VERSION = "counterfactual_escalation_safety_service_oracle/1"


@dataclass(frozen=True)
class SafetyServiceMargins:
    min_discharge_gain: int = 3
    max_discharge_loss: int = 2
    max_delay_loss_s: float = 300.0
    max_delay_loss_fraction: float = 0.05
    max_hard_brake_exposure_excess: int = 2
    max_hard_brake_exposure_excess_fraction: float = 0.10
    max_hard_brake_onset_excess: int = 2
    max_hard_brake_unique_vehicle_excess: int = 2
    max_low_ttc_excess_steps: int = 2
    min_hard_brake_onset_reduction: int = 2
    min_low_ttc_exposure_reduction_steps: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _hard_brake_stat(outcome: Mapping[str, Any], key: str) -> int:
    attribution = outcome.get("hard_brake_attribution") or {}
    if key not in attribution:
        raise ValueError(f"outcome lacks hard-brake attribution field {key!r}")
    return int(attribution[key])


def classify_safety_service_case(
    row: Mapping[str, Any],
    margins: SafetyServiceMargins = SafetyServiceMargins(),
) -> dict[str, Any]:
    """Classify one paired case without consulting its legacy oracle label."""
    off = row["arms"]["0"]["outcomes"]
    wall = row["arms"]["1"]["outcomes"]
    engaged = bool(row["arms"]["1"].get("wall_engaged", False))

    delay_benefit = _number(off["incremental_delay_s"]) - _number(wall["incremental_delay_s"])
    discharge_benefit = int(wall["merge_discharge_count"]) - int(off["merge_discharge_count"])
    hard_brake_exposure_excess = int(wall["hard_brake_vehicle_steps"]) - int(
        off["hard_brake_vehicle_steps"]
    )
    hard_brake_onset_excess = _hard_brake_stat(wall, "event_start_count") - _hard_brake_stat(
        off, "event_start_count"
    )
    hard_brake_unique_vehicle_excess = _hard_brake_stat(
        wall, "unique_vehicle_count"
    ) - _hard_brake_stat(off, "unique_vehicle_count")
    low_ttc_excess = int(wall["low_ttc_exposure_steps"]) - int(off["low_ttc_exposure_steps"])
    collision_excess = int(wall["collision_unique_vehicle_count"]) - int(
        off["collision_unique_vehicle_count"]
    )
    teleport_excess = int(wall["starting_teleport_unique_vehicle_count"]) - int(
        off["starting_teleport_unique_vehicle_count"]
    )

    delay_loss_budget = max(
        margins.max_delay_loss_s,
        margins.max_delay_loss_fraction * _number(off["incremental_delay_s"]),
    )
    hard_brake_exposure_budget = max(
        margins.max_hard_brake_exposure_excess,
        math.ceil(
            margins.max_hard_brake_exposure_excess_fraction * int(off["hard_brake_vehicle_steps"])
        ),
    )
    safety_admissible = (
        collision_excess <= 0
        and teleport_excess <= 0
        and hard_brake_exposure_excess <= hard_brake_exposure_budget
        and hard_brake_onset_excess <= margins.max_hard_brake_onset_excess
        and hard_brake_unique_vehicle_excess <= margins.max_hard_brake_unique_vehicle_excess
        and low_ttc_excess <= margins.max_low_ttc_excess_steps
    )
    service_noninferior = (
        discharge_benefit >= -margins.max_discharge_loss and delay_benefit >= -delay_loss_budget
    )
    benefit_components = {
        "hard_brake_onset_reduction": (
            hard_brake_onset_excess <= -margins.min_hard_brake_onset_reduction
        ),
        "low_ttc_exposure_reduction": (
            low_ttc_excess <= -margins.min_low_ttc_exposure_reduction_steps
        ),
        "merge_discharge_gain": discharge_benefit >= margins.min_discharge_gain,
    }
    meaningful_benefit = any(benefit_components.values())
    wall_selected = engaged and safety_admissible and service_noninferior and meaningful_benefit

    refusal_reasons = []
    if not engaged:
        refusal_reasons.append("formation_or_runtime_veto")
    if not safety_admissible:
        refusal_reasons.append("safety_budget")
    if not service_noninferior:
        refusal_reasons.append("service_noninferiority")
    if not meaningful_benefit:
        refusal_reasons.append("insufficient_safety_or_service_benefit")

    return {
        "oracle_version": ORACLE_VERSION,
        "oracle_action": "DAMPING_PLUS_WALL" if wall_selected else "DAMPING_ONLY",
        "wall_selected": wall_selected,
        "wall_engaged": engaged,
        "safety_admissible": safety_admissible,
        "service_noninferior": service_noninferior,
        "meaningful_benefit": meaningful_benefit,
        "benefit_components": benefit_components,
        "refusal_reasons": refusal_reasons,
        "effects": {
            "delay_benefit_s": delay_benefit,
            "merge_discharge_benefit": discharge_benefit,
            "hard_brake_exposure_excess_vehicle_steps": (hard_brake_exposure_excess),
            "hard_brake_onset_excess": hard_brake_onset_excess,
            "hard_brake_unique_vehicle_excess": (hard_brake_unique_vehicle_excess),
            "low_ttc_excess_steps": low_ttc_excess,
            "collision_excess_vehicle_count": collision_excess,
            "teleport_excess_vehicle_count": teleport_excess,
        },
        "budgets": {
            "delay_loss_budget_s": delay_loss_budget,
            "hard_brake_exposure_excess_budget": hard_brake_exposure_budget,
            "hard_brake_onset_excess_budget": (margins.max_hard_brake_onset_excess),
            "hard_brake_unique_vehicle_excess_budget": (
                margins.max_hard_brake_unique_vehicle_excess
            ),
            "low_ttc_excess_budget_steps": margins.max_low_ttc_excess_steps,
        },
    }


def summarize_staged_safety_service(
    rows: Sequence[Mapping[str, Any]],
    margins: SafetyServiceMargins = SafetyServiceMargins(),
) -> dict[str, Any]:
    """Report formation, safety, and benefit stages with seed-group counts."""
    eligible = [row for row in rows if bool(row.get("eligible", False))]
    classified = [classify_safety_service_case(row, margins) for row in eligible]

    def _seeds(predicate) -> list[int]:
        return sorted(
            {int(row["seed"]) for row, result in zip(eligible, classified) if predicate(result)}
        )

    engaged = [result for result in classified if result["wall_engaged"]]
    safe = [result for result in engaged if result["safety_admissible"]]
    unsafe = [result for result in engaged if not result["safety_admissible"]]
    selected = [result for result in classified if result["wall_selected"]]
    selected_seeds = _seeds(lambda result: bool(result["wall_selected"]))
    independent_seeds = sorted({int(row["seed"]) for row in eligible})
    checks = {
        "minimum_independent_development_groups": (
            len(independent_seeds) >= MIN_MODEL_DEVELOPMENT_GROUPS
        ),
        "minimum_selected_independent_groups": (len(selected_seeds) >= MIN_WALL_SELECTED_GROUPS),
        "minimum_selected_cases": len(selected) >= MIN_WALL_SELECTED_CASES,
    }
    return {
        "oracle_version": ORACLE_VERSION,
        "margins": margins.to_dict(),
        "eligible_case_count": len(eligible),
        "independent_seed_count": len(independent_seeds),
        "formation_or_runtime_veto_case_count": len(classified) - len(engaged),
        "engaged_case_count": len(engaged),
        "engaged_seed_count": len(_seeds(lambda result: result["wall_engaged"])),
        "safe_engagement_case_count": len(safe),
        "unsafe_engagement_case_count": len(unsafe),
        "unsafe_engagement_seed_count": len(
            _seeds(lambda result: result["wall_engaged"] and not result["safety_admissible"])
        ),
        "selected_case_count": len(selected),
        "selected_seed_count": len(selected_seeds),
        "selected_seeds": selected_seeds,
        "safe_not_selected_case_count": len(safe) - len(selected),
        "model_fit_adequacy_checks": checks,
        "model_fit_permitted": all(checks.values()),
        "rows": classified,
    }


def evaluate_safety_service_baselines(
    rows: Sequence[Mapping[str, Any]],
    margins: SafetyServiceMargins = SafetyServiceMargins(),
) -> dict[str, Any]:
    """Compare simple development rules against the paired oracle."""
    eligible = [row for row in rows if bool(row.get("eligible", False))]
    pairs = [(row, classify_safety_service_case(row, margins)) for row in eligible]
    oracle_total = sum(bool(result["wall_selected"]) for _, result in pairs)

    def _never(_row: Mapping[str, Any]) -> bool:
        return False

    def _always(_row: Mapping[str, Any]) -> bool:
        return True

    def _late(row: Mapping[str, Any]) -> bool:
        return int(row["decision_step"]) >= 180

    def _step_240(row: Mapping[str, Any]) -> bool:
        return int(row["decision_step"]) == 240

    def _p30_late(row: Mapping[str, Any]) -> bool:
        return (
            abs(float(row["scenario"]["cav_penetration"]) - 0.30) < 1e-9
            and int(row["decision_step"]) >= 180
        )

    rules = {
        "never_wall": _never,
        "always_command_wall": _always,
        "late_step_180_or_240": _late,
        "step_240_only": _step_240,
        "p30_late": _p30_late,
    }
    reports = {}
    for name, rule in rules.items():
        commanded = [(row, result) for row, result in pairs if rule(row)]
        engaged = [(row, result) for row, result in commanded if result["wall_engaged"]]
        beneficial = [(row, result) for row, result in commanded if result["wall_selected"]]
        reports[name] = {
            "command_count": len(commanded),
            "engagement_count": len(engaged),
            "unsafe_engagement_count": sum(
                not result["safety_admissible"] for _, result in engaged
            ),
            "beneficial_engagement_count": len(beneficial),
            "beneficial_precision_among_engagements": (
                len(beneficial) / len(engaged) if engaged else None
            ),
            "paired_oracle_recall": (len(beneficial) / oracle_total if oracle_total else None),
            "realized_override_steps": sum(
                int(row["arms"]["1"].get("wall_override_step_count", 0)) for row, _ in engaged
            ),
        }
    reports["paired_oracle"] = {
        "command_count": oracle_total,
        "engagement_count": oracle_total,
        "unsafe_engagement_count": 0,
        "beneficial_engagement_count": oracle_total,
        "beneficial_precision_among_engagements": 1.0 if oracle_total else None,
        "paired_oracle_recall": 1.0 if oracle_total else None,
        "realized_override_steps": sum(
            int(row["arms"]["1"].get("wall_override_step_count", 0))
            for row, result in pairs
            if result["wall_selected"]
        ),
    }
    return {
        "oracle_version": ORACLE_VERSION,
        "eligible_case_count": len(eligible),
        "paired_oracle_selected_case_count": oracle_total,
        "baselines": reports,
    }
