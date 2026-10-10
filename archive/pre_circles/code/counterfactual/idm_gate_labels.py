"""Frozen safety-first Gate-B labels for the IDM escalation predictor.

This module is versioned separately from the legacy headroom oracle.  Delay is
a service noninferiority guardrail, not a stand-alone reason to command a wall;
queue area is reported but never selects treatment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


CONTRACT_VERSION = "idm_escalation_gate_b_safety_service/1"


@dataclass(frozen=True)
class IDMGateBMargins:
    min_hard_brake_onset_reduction: int = 2
    min_hard_brake_onset_reduction_fraction: float = 0.10
    min_hard_brake_exposure_reduction: int = 3
    min_hard_brake_exposure_reduction_fraction: float = 0.10
    min_low_ttc_reduction_steps: int = 2
    min_discharge_gain: int = 3
    max_discharge_loss: int = 2
    max_delay_loss_s: float = 300.0
    max_delay_loss_fraction: float = 0.05
    max_hard_brake_excess: int = 2
    max_hard_brake_excess_fraction: float = 0.10
    max_low_ttc_excess_steps: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _hard_brake_onsets(outcome: Mapping[str, Any]) -> int:
    attribution = outcome.get("hard_brake_attribution") or {}
    return int(attribution.get("event_start_count", 0))


def classify_idm_gate_b_case(
    row: Mapping[str, Any],
    margins: IDMGateBMargins = IDMGateBMargins(),
) -> dict[str, Any]:
    off = row["arms"]["0"]["outcomes"]
    wall = row["arms"]["1"]["outcomes"]
    engaged = bool(row["arms"]["1"].get("wall_engaged", False))

    delay_off = _number(off["incremental_delay_s"])
    delay_wall = _number(wall["incremental_delay_s"])
    delay_benefit = delay_off - delay_wall
    queue_benefit = _number(off["queue_area_vehicle_s"]) - _number(
        wall["queue_area_vehicle_s"]
    )
    discharge_benefit = int(wall["merge_discharge_count"]) - int(
        off["merge_discharge_count"]
    )
    hard_brake_excess = int(wall["hard_brake_vehicle_steps"]) - int(
        off["hard_brake_vehicle_steps"]
    )
    onset_off = _hard_brake_onsets(off)
    onset_wall = _hard_brake_onsets(wall)
    onset_excess = onset_wall - onset_off
    low_ttc_excess = int(wall["low_ttc_exposure_steps"]) - int(
        off["low_ttc_exposure_steps"]
    )
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
    delay_loss_budget = max(
        margins.max_delay_loss_s,
        margins.max_delay_loss_fraction * delay_off,
    )
    onset_benefit_threshold = max(
        margins.min_hard_brake_onset_reduction,
        math.ceil(margins.min_hard_brake_onset_reduction_fraction * onset_off),
    )
    exposure_benefit_threshold = max(
        margins.min_hard_brake_exposure_reduction,
        math.ceil(
            margins.min_hard_brake_exposure_reduction_fraction
            * int(off["hard_brake_vehicle_steps"])
        ),
    )

    safety_admissible = bool(
        collision_excess <= 0
        and teleport_excess <= 0
        and hard_brake_excess <= hard_brake_budget
        and low_ttc_excess <= margins.max_low_ttc_excess_steps
    )
    traffic_noninferior = bool(
        discharge_benefit >= -margins.max_discharge_loss
        and delay_benefit >= -delay_loss_budget
    )
    safety_benefit = bool(
        -onset_excess >= onset_benefit_threshold
        or -hard_brake_excess >= exposure_benefit_threshold
        or -low_ttc_excess >= margins.min_low_ttc_reduction_steps
    )
    throughput_benefit = discharge_benefit >= margins.min_discharge_gain
    meaningful_benefit = bool(safety_benefit or throughput_benefit)
    wall_selected = bool(
        engaged and safety_admissible and traffic_noninferior and meaningful_benefit
    )

    refusal_reasons = []
    if not engaged:
        refusal_reasons.append("formation_or_runtime_veto")
    if not safety_admissible:
        refusal_reasons.append("safety_budget")
    if not traffic_noninferior:
        refusal_reasons.append("service_noninferiority")
    if not meaningful_benefit:
        refusal_reasons.append("insufficient_safety_or_discharge_benefit")
    return {
        "label_contract_version": CONTRACT_VERSION,
        "oracle_action": "DAMPING_PLUS_WALL" if wall_selected else "DAMPING_ONLY",
        "wall_selected": wall_selected,
        "wall_engaged": engaged,
        "safety_admissible": safety_admissible,
        "traffic_noninferior": traffic_noninferior,
        "meaningful_benefit": meaningful_benefit,
        "safety_benefit": safety_benefit,
        "throughput_benefit": throughput_benefit,
        "delay_is_standalone_benefit": False,
        "queue_area_is_selection_criterion": False,
        "refusal_reasons": refusal_reasons,
        "effects": {
            "delay_benefit_s": delay_benefit,
            "queue_area_benefit_vehicle_s": queue_benefit,
            "merge_discharge_benefit": discharge_benefit,
            "hard_brake_onset_excess": onset_excess,
            "hard_brake_excess_vehicle_steps": hard_brake_excess,
            "low_ttc_excess_steps": low_ttc_excess,
            "collision_excess_vehicle_count": collision_excess,
            "teleport_excess_vehicle_count": teleport_excess,
        },
        "budgets": {
            "delay_loss_budget_s": delay_loss_budget,
            "hard_brake_excess_budget": hard_brake_budget,
            "low_ttc_excess_budget_steps": margins.max_low_ttc_excess_steps,
            "hard_brake_onset_benefit_threshold": onset_benefit_threshold,
            "hard_brake_exposure_benefit_threshold": exposure_benefit_threshold,
            "low_ttc_benefit_threshold_steps": margins.min_low_ttc_reduction_steps,
            "discharge_benefit_threshold": margins.min_discharge_gain,
        },
    }


def contract_manifest() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "margins": IDMGateBMargins().to_dict(),
        "primary_benefits": [
            "hard_brake_onset_reduction",
            "hard_brake_exposure_reduction",
            "low_ttc_exposure_reduction",
            "merge_discharge_gain",
        ],
        "service_guardrails": ["delay_noninferiority", "merge_discharge_noninferiority"],
        "delay_is_standalone_benefit": False,
        "queue_area_is_selection_criterion": False,
    }
