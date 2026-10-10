"""Post-hoc sensitivity analysis for counterfactual wall-win definitions.

This module deliberately does not change the frozen readiness/oracle label.
It asks how the paired off/soft evidence changes when upstream queue and SUMO
timeLoss are treated as reported intervention costs rather than hard vetoes.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Mapping, Sequence


EPSILON = 1e-9


@dataclass(frozen=True)
class WinRule:
    """Metadata for one post-hoc definition of a traffic-effective wall."""

    name: str
    title: str
    description: str


WIN_RULES = (
    WinRule(
        "current_pareto",
        "Frozen Pareto rule",
        "No worse pre-merge congestion, queue area, delay, or merge discharge; "
        "strictly better on at least one.",
    ),
    WinRule(
        "delay_margin_2pct",
        "2% delay non-inferiority",
        "Frozen rule except aggregate delay may increase by at most 2% when "
        "another outcome strictly improves.",
    ),
    WinRule(
        "delay_margin_5pct",
        "5% delay non-inferiority",
        "Frozen rule except aggregate delay may increase by at most 5% when "
        "another outcome strictly improves.",
    ),
    WinRule(
        "delay_excluded",
        "Delay excluded",
        "No worse pre-merge congestion, queue area, or merge discharge; "
        "delay is reported but does not determine the label.",
    ),
    WinRule(
        "stability_throughput",
        "Pre-merge stability + throughput",
        "No induced pre-merge sustained congestion and no merge-discharge "
        "loss; strictly improve congestion state or discharge. Queue and "
        "delay are reported costs.",
    ),
    WinRule(
        "bottleneck_service_exact",
        "Merge service, exact throughput",
        "No loss of mean merge speed or merge discharge; strictly improve "
        "at least one. Pre-merge queue, pre-merge congestion, and delay are "
        "reported costs.",
    ),
    WinRule(
        "bottleneck_service_1veh",
        "Merge service, one-vehicle tolerance",
        "Mean merge speed must not fall and merge discharge may fall by at "
        "most one vehicle over the outcome horizon; strictly improve speed "
        "or discharge.",
    ),
    WinRule(
        "collapse_rescue_1veh",
        "Collapse rescue, one-vehicle tolerance",
        "The soft wall must turn an off-arm pre-merge congestion event into "
        "a non-event, with at most one fewer merge discharge.",
    ),
)

WIN_RULE_BY_NAME = {rule.name: rule for rule in WIN_RULES}


def _outcomes(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    return row["arms"]["0"]["outcomes"], row["arms"]["1"]["outcomes"]


def treatment_effects(row: Mapping[str, Any]) -> Dict[str, float]:
    """Return paired effects, with positive values favorable unless named cost."""
    off, soft = _outcomes(row)
    off_delay = float(off["incremental_delay_s"])
    off_queue = float(off["queue_area_vehicle_s"])
    delay_cost = float(soft["incremental_delay_s"]) - off_delay
    queue_cost = float(soft["queue_area_vehicle_s"]) - off_queue
    return {
        "sustained_congestion_reduction": float(
            int(bool(off["sustained_congestion"])) - int(bool(soft["sustained_congestion"]))
        ),
        "queue_area_cost_vehicle_s": queue_cost,
        "queue_area_relative_cost": queue_cost / max(abs(off_queue), EPSILON),
        "delay_cost_s": delay_cost,
        "delay_relative_cost": delay_cost / max(abs(off_delay), EPSILON),
        "merge_discharge_gain": float(soft["merge_discharge_count"])
        - float(off["merge_discharge_count"]),
        "merge_speed_gain_mps": float(soft["mean_merge_speed_mps"])
        - float(off["mean_merge_speed_mps"]),
        "pre_merge_speed_gain_mps": float(soft["mean_pre_merge_speed_mps"])
        - float(off["mean_pre_merge_speed_mps"]),
    }


def is_traffic_win(row: Mapping[str, Any], rule_name: str) -> bool:
    """Apply one traffic-only win rule to a paired off/soft case."""
    if rule_name not in WIN_RULE_BY_NAME:
        raise ValueError(f"unknown win rule: {rule_name}")

    effect = treatment_effects(row)
    congestion = effect["sustained_congestion_reduction"]
    queue_cost = effect["queue_area_cost_vehicle_s"]
    delay_cost = effect["delay_cost_s"]
    delay_relative_cost = effect["delay_relative_cost"]
    discharge = effect["merge_discharge_gain"]
    merge_speed = effect["merge_speed_gain_mps"]

    if rule_name == "current_pareto":
        no_worse = congestion >= 0 and queue_cost <= 0 and delay_cost <= 0 and discharge >= 0
        strict = congestion > 0 or queue_cost < 0 or delay_cost < 0 or discharge > 0
        return no_worse and strict

    if rule_name in {"delay_margin_2pct", "delay_margin_5pct"}:
        margin = 0.02 if rule_name == "delay_margin_2pct" else 0.05
        no_worse = (
            congestion >= 0
            and queue_cost <= 0
            and delay_relative_cost <= margin + EPSILON
            and discharge >= 0
        )
        strict = congestion > 0 or queue_cost < 0 or delay_cost < 0 or discharge > 0
        return no_worse and strict

    if rule_name == "delay_excluded":
        no_worse = congestion >= 0 and queue_cost <= 0 and discharge >= 0
        strict = congestion > 0 or queue_cost < 0 or discharge > 0
        return no_worse and strict

    if rule_name == "stability_throughput":
        return congestion >= 0 and discharge >= 0 and (congestion > 0 or discharge > 0)

    if rule_name == "bottleneck_service_exact":
        return merge_speed >= 0 and discharge >= 0 and (merge_speed > 0 or discharge > 0)

    if rule_name == "bottleneck_service_1veh":
        return merge_speed >= 0 and discharge >= -1 and (merge_speed > 0 or discharge > 0)

    if rule_name == "collapse_rescue_1veh":
        return congestion > 0 and discharge >= -1

    raise AssertionError(f"unhandled win rule: {rule_name}")


def _average(outcomes: Sequence[Mapping[str, Any]], key: str) -> float:
    return mean(float(outcome[key]) for outcome in outcomes) if outcomes else 0.0


def policy_summary(
    rows: Sequence[Mapping[str, Any]],
    readiness_cases: Mapping[str, Mapping[str, Any]],
    actions: Sequence[int],
) -> Dict[str, Any]:
    """Summarize the realized paired policy chosen by an oracle label."""
    outcomes = [row["arms"][str(action)]["outcomes"] for row, action in zip(rows, actions)]
    return {
        "case_count": len(outcomes),
        "soft_count": sum(action == 1 for action in actions),
        "unsafe_soft_count": sum(
            action == 1
            and not bool(readiness_cases[str(row["case_id"])]["actions"]["1"]["safety_admissible"])
            for row, action in zip(rows, actions)
        ),
        "sustained_congestion_count": sum(
            bool(outcome["sustained_congestion"]) for outcome in outcomes
        ),
        "mean_queue_area_vehicle_s": _average(outcomes, "queue_area_vehicle_s"),
        "mean_incremental_delay_s": _average(outcomes, "incremental_delay_s"),
        "mean_merge_discharge_count": _average(outcomes, "merge_discharge_count"),
        "mean_merge_speed_mps": _average(outcomes, "mean_merge_speed_mps"),
        "mean_pre_merge_speed_mps": _average(outcomes, "mean_pre_merge_speed_mps"),
        "mean_hard_brake_vehicle_steps": _average(outcomes, "hard_brake_vehicle_steps"),
        "mean_low_ttc_exposure_steps": _average(outcomes, "low_ttc_exposure_steps"),
        "total_collision_unique_vehicle_count": sum(
            int(outcome.get("collision_unique_vehicle_count", 0)) for outcome in outcomes
        ),
        "total_starting_teleport_unique_vehicle_count": sum(
            int(outcome.get("starting_teleport_unique_vehicle_count", 0)) for outcome in outcomes
        ),
    }


def _positive_cost_summary(values: Sequence[float]) -> Dict[str, Any]:
    positive = [float(value) for value in values if value > EPSILON]
    return {
        "case_count": len(positive),
        "mean": mean(positive) if positive else 0.0,
        "max": max(positive, default=0.0),
    }


def _delta_vs_off(policy: Mapping[str, Any], off: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sustained_congestion_cases_avoided": int(off["sustained_congestion_count"])
        - int(policy["sustained_congestion_count"]),
        "mean_queue_area_change_vehicle_s": float(policy["mean_queue_area_vehicle_s"])
        - float(off["mean_queue_area_vehicle_s"]),
        "mean_incremental_delay_change_s": float(policy["mean_incremental_delay_s"])
        - float(off["mean_incremental_delay_s"]),
        "mean_merge_discharge_gain": float(policy["mean_merge_discharge_count"])
        - float(off["mean_merge_discharge_count"]),
        "mean_merge_speed_gain_mps": float(policy["mean_merge_speed_mps"])
        - float(off["mean_merge_speed_mps"]),
        "mean_pre_merge_speed_gain_mps": float(policy["mean_pre_merge_speed_mps"])
        - float(off["mean_pre_merge_speed_mps"]),
        "mean_hard_brake_change_vehicle_steps": float(policy["mean_hard_brake_vehicle_steps"])
        - float(off["mean_hard_brake_vehicle_steps"]),
        "mean_low_ttc_change_steps": float(policy["mean_low_ttc_exposure_steps"])
        - float(off["mean_low_ttc_exposure_steps"]),
    }


def analyze_split(
    rows: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare all win definitions on one completed paired dataset."""
    eligible = [row for row in rows if bool(row.get("eligible"))]
    readiness_cases = {str(case["case_id"]): case for case in readiness.get("cases", [])}
    missing = [
        str(row["case_id"]) for row in eligible if str(row["case_id"]) not in readiness_cases
    ]
    if missing:
        raise ValueError(f"readiness report is missing {len(missing)} paired cases")

    off_actions = [0] * len(eligible)
    soft_actions = [1] * len(eligible)
    policies = {
        "always_off": policy_summary(eligible, readiness_cases, off_actions),
        "always_soft": policy_summary(eligible, readiness_cases, soft_actions),
    }

    rule_details: Dict[str, Any] = {}
    selected_ids_by_rule: Dict[str, set[str]] = {}
    for rule in WIN_RULES:
        case_details = []
        actions = []
        for row in eligible:
            case_id = str(row["case_id"])
            effect = treatment_effects(row)
            traffic_win = is_traffic_win(row, rule.name)
            safety_admissible = bool(readiness_cases[case_id]["actions"]["1"]["safety_admissible"])
            selected = traffic_win and safety_admissible
            actions.append(1 if selected else 0)
            case_details.append(
                {
                    "case_id": case_id,
                    "seed": int(row["seed"]),
                    "decision_step": int(row["decision_step"]),
                    "traffic_win": bool(traffic_win),
                    "safety_admissible": safety_admissible,
                    "selected_soft": bool(selected),
                    "effects": effect,
                }
            )

        selected_cases = [case for case in case_details if case["selected_soft"]]
        selected_effects = [case["effects"] for case in selected_cases]
        policy = policy_summary(eligible, readiness_cases, actions)
        selected_ids_by_rule[rule.name] = {str(case["case_id"]) for case in selected_cases}
        rule_details[rule.name] = {
            "title": rule.title,
            "description": rule.description,
            "traffic_win_count": sum(case["traffic_win"] for case in case_details),
            "selected_safe_soft_count": len(selected_cases),
            "safety_rejected_traffic_win_count": sum(
                case["traffic_win"] and not case["safety_admissible"] for case in case_details
            ),
            "selected_rescue_count": sum(
                case["effects"]["sustained_congestion_reduction"] > 0 for case in selected_cases
            ),
            "selected_induced_pre_merge_congestion_count": sum(
                case["effects"]["sustained_congestion_reduction"] < 0 for case in selected_cases
            ),
            "selected_discharge_loss_count": sum(
                case["effects"]["merge_discharge_gain"] < 0 for case in selected_cases
            ),
            "selected_merge_speed_loss_count": sum(
                case["effects"]["merge_speed_gain_mps"] < 0 for case in selected_cases
            ),
            "selected_delay_cost_s": _positive_cost_summary(
                [effect["delay_cost_s"] for effect in selected_effects]
            ),
            "selected_delay_relative_cost": _positive_cost_summary(
                [effect["delay_relative_cost"] for effect in selected_effects]
            ),
            "selected_queue_cost_vehicle_s": _positive_cost_summary(
                [effect["queue_area_cost_vehicle_s"] for effect in selected_effects]
            ),
            "selected_queue_relative_cost": _positive_cost_summary(
                [effect["queue_area_relative_cost"] for effect in selected_effects]
            ),
            "policy": policy,
            "policy_delta_vs_always_off": _delta_vs_off(policy, policies["always_off"]),
            "cases": case_details,
        }

    current_ids = selected_ids_by_rule["current_pareto"]
    for name, details in rule_details.items():
        selected_ids = selected_ids_by_rule[name]
        details["added_safe_wins_vs_current"] = sorted(selected_ids - current_ids)
        details["removed_safe_wins_vs_current"] = sorted(current_ids - selected_ids)

    return {
        "case_count": len(rows),
        "eligible_case_count": len(eligible),
        "baseline_policies": policies,
        "rules": rule_details,
    }
