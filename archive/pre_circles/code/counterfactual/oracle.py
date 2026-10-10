"""Oracle-headroom analysis for paired counterfactual treatment arms."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Dict, Mapping, Sequence, Tuple


ACTION_NAMES = {0: "off", 1: "soft", 2: "strong"}


def _outcome(row: Mapping[str, Any], action: int) -> Mapping[str, Any]:
    return row["arms"][str(action)]["outcomes"]


def _traffic_score(outcome: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        int(bool(outcome["sustained_congestion"])),
        float(outcome["queue_area_vehicle_s"]),
        float(outcome["incremental_delay_s"]),
        -int(outcome["merge_discharge_count"]),
        int(outcome["duty_steps"]),
    )


def _safety_excess(
    outcome: Mapping[str, Any], off: Mapping[str, Any]
) -> Tuple[int, int, int, int, int]:
    collision_excess = max(
        0,
        int(outcome.get("collision_unique_vehicle_count", 0))
        - int(off.get("collision_unique_vehicle_count", 0)),
    )
    teleport_excess = max(
        0,
        int(outcome.get("starting_teleport_unique_vehicle_count", 0))
        - int(off.get("starting_teleport_unique_vehicle_count", 0)),
    )
    hard_brake_excess = max(
        0,
        int(outcome["hard_brake_vehicle_steps"]) - int(off["hard_brake_vehicle_steps"]),
    )
    low_ttc_excess = max(
        0,
        int(outcome["low_ttc_exposure_steps"]) - int(off["low_ttc_exposure_steps"]),
    )
    return (
        int(
            collision_excess > 0
            or teleport_excess > 0
            or hard_brake_excess > 0
            or low_ttc_excess > 0
        ),
        collision_excess,
        teleport_excess,
        hard_brake_excess,
        low_ttc_excess,
    )


def oracle_score(row: Mapping[str, Any], action: int) -> Tuple[Any, ...]:
    """Safety-first, then congestion/delay, with lower actuation as tie-break."""
    off = _outcome(row, 0)
    outcome = _outcome(row, action)
    return (*_safety_excess(outcome, off), *_traffic_score(outcome), int(action))


def choose_oracle_action(row: Mapping[str, Any]) -> int:
    if not row.get("eligible"):
        raise ValueError("oracle action is undefined for an ineligible case")
    return min((0, 1, 2), key=lambda action: oracle_score(row, action))


def choose_traffic_oracle_action(row: Mapping[str, Any]) -> int:
    """Choose the traffic-optimal arm without applying a safety constraint."""
    if not row.get("eligible"):
        raise ValueError("oracle action is undefined for an ineligible case")
    return min((0, 1, 2), key=lambda action: (*_traffic_score(_outcome(row, action)), action))


def is_dominant_traffic_benefit(row: Mapping[str, Any], action: int) -> bool:
    """Whether treatment improves off without a queue/delay/discharge trade-off."""
    if action not in (1, 2):
        return False
    off = _outcome(row, 0)
    treatment = _outcome(row, action)
    no_worse = (
        int(bool(treatment["sustained_congestion"])) <= int(bool(off["sustained_congestion"]))
        and float(treatment["queue_area_vehicle_s"]) <= float(off["queue_area_vehicle_s"])
        and float(treatment["incremental_delay_s"]) <= float(off["incremental_delay_s"])
        and int(treatment["merge_discharge_count"]) >= int(off["merge_discharge_count"])
    )
    strictly_better = (
        int(bool(treatment["sustained_congestion"])) < int(bool(off["sustained_congestion"]))
        or float(treatment["queue_area_vehicle_s"]) < float(off["queue_area_vehicle_s"])
        or float(treatment["incremental_delay_s"]) < float(off["incremental_delay_s"])
        or int(treatment["merge_discharge_count"]) > int(off["merge_discharge_count"])
    )
    return no_worse and strictly_better


def choose_least_intensive_beneficial_action(row: Mapping[str, Any]) -> int:
    """Choose soft before strong when either dominantly improves traffic."""
    for action in (1, 2):
        if is_dominant_traffic_benefit(row, action):
            return action
    return 0


def pareto_actions(row: Mapping[str, Any]) -> list[int]:
    """Return actions not dominated jointly on traffic, safety, and duty."""

    def vector(action: int) -> Tuple[float, ...]:
        outcome = _outcome(row, action)
        return (
            float(bool(outcome["sustained_congestion"])),
            float(outcome.get("collision_unique_vehicle_count", 0)),
            float(outcome.get("starting_teleport_unique_vehicle_count", 0)),
            float(outcome["hard_brake_vehicle_steps"]),
            float(outcome["low_ttc_exposure_steps"]),
            float(outcome["queue_area_vehicle_s"]),
            float(outcome["incremental_delay_s"]),
            -float(outcome["merge_discharge_count"]),
            float(outcome["duty_steps"]),
        )

    vectors = {action: vector(action) for action in (0, 1, 2)}
    frontier = []
    for action, candidate in vectors.items():
        dominated = any(
            other != action
            and all(left <= right for left, right in zip(vectors[other], candidate))
            and any(left < right for left, right in zip(vectors[other], candidate))
            for other in vectors
        )
        if not dominated:
            frontier.append(action)
    return frontier


def treatment_effects(row: Mapping[str, Any], action: int) -> Dict[str, Any]:
    """Return positive-valued benefits of one treatment relative to off."""
    if action not in (1, 2):
        raise ValueError("treatment effect is defined for soft or strong")
    off = _outcome(row, 0)
    treatment = _outcome(row, action)
    return {
        "action": action,
        "action_name": ACTION_NAMES[action],
        "avoids_sustained_congestion": bool(
            off["sustained_congestion"] and not treatment["sustained_congestion"]
        ),
        "induces_sustained_congestion": bool(
            not off["sustained_congestion"] and treatment["sustained_congestion"]
        ),
        "queue_area_reduction_vehicle_s": float(off["queue_area_vehicle_s"])
        - float(treatment["queue_area_vehicle_s"]),
        "delay_reduction_s": float(off["incremental_delay_s"])
        - float(treatment["incremental_delay_s"]),
        "additional_merge_discharge": int(treatment["merge_discharge_count"])
        - int(off["merge_discharge_count"]),
        "collision_vehicle_reduction": int(off.get("collision_unique_vehicle_count", 0))
        - int(treatment.get("collision_unique_vehicle_count", 0)),
        "starting_teleport_vehicle_reduction": int(
            off.get("starting_teleport_unique_vehicle_count", 0)
        )
        - int(treatment.get("starting_teleport_unique_vehicle_count", 0)),
        "hard_brake_reduction_vehicle_steps": int(off["hard_brake_vehicle_steps"])
        - int(treatment["hard_brake_vehicle_steps"]),
        "low_ttc_reduction_steps": int(off["low_ttc_exposure_steps"])
        - int(treatment["low_ttc_exposure_steps"]),
        "safety_excess": _safety_excess(treatment, off)[0] > 0,
    }


def classify_case(row: Mapping[str, Any], oracle_action: int) -> str:
    off = _outcome(row, 0)
    if oracle_action != 0:
        chosen = _outcome(row, oracle_action)
        if off["sustained_congestion"] and not chosen["sustained_congestion"]:
            return "rescue"
        if off["sustained_congestion"]:
            return "mitigation"
        return "preventive_benefit"

    beneficial_actions = [action for action in (1, 2) if is_dominant_traffic_benefit(row, action)]
    if beneficial_actions and all(
        _safety_excess(_outcome(row, action), off)[0] for action in beneficial_actions
    ):
        return "safety_rejected"
    treatment_scores = [_traffic_score(_outcome(row, action)) for action in (1, 2)]
    if all(score > _traffic_score(off) for score in treatment_scores):
        return "harmful_treatments_only"
    return "no_dominant_traffic_benefit"


def _policy_summary(rows: Sequence[Mapping[str, Any]], actions: Sequence[int]) -> Dict[str, Any]:
    outcomes = [_outcome(row, action) for row, action in zip(rows, actions)]
    if not outcomes:
        return {
            "case_count": 0,
            "sustained_congestion_count": 0,
            "mean_queue_area_vehicle_s": None,
            "mean_incremental_delay_s": None,
            "mean_merge_discharge_count": None,
            "mean_hard_brake_vehicle_steps": None,
            "mean_collision_unique_vehicle_count": None,
            "mean_starting_teleport_unique_vehicle_count": None,
            "mean_low_ttc_exposure_steps": None,
            "mean_duty_fraction": None,
        }
    return {
        "case_count": len(outcomes),
        "sustained_congestion_count": sum(
            bool(outcome["sustained_congestion"]) for outcome in outcomes
        ),
        "mean_queue_area_vehicle_s": mean(
            float(outcome["queue_area_vehicle_s"]) for outcome in outcomes
        ),
        "mean_incremental_delay_s": mean(
            float(outcome["incremental_delay_s"]) for outcome in outcomes
        ),
        "mean_merge_discharge_count": mean(
            int(outcome["merge_discharge_count"]) for outcome in outcomes
        ),
        "mean_hard_brake_vehicle_steps": mean(
            int(outcome["hard_brake_vehicle_steps"]) for outcome in outcomes
        ),
        "mean_collision_unique_vehicle_count": mean(
            int(outcome.get("collision_unique_vehicle_count", 0)) for outcome in outcomes
        ),
        "mean_starting_teleport_unique_vehicle_count": mean(
            int(outcome.get("starting_teleport_unique_vehicle_count", 0)) for outcome in outcomes
        ),
        "mean_low_ttc_exposure_steps": mean(
            int(outcome["low_ttc_exposure_steps"]) for outcome in outcomes
        ),
        "mean_duty_fraction": mean(float(outcome["duty_fraction"]) for outcome in outcomes),
    }


def analyze_oracle(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    eligible = [row for row in rows if row.get("eligible")]
    ineligible = [row for row in rows if not row.get("eligible")]
    oracle_actions = [choose_oracle_action(row) for row in eligible]
    traffic_oracle_actions = [choose_traffic_oracle_action(row) for row in eligible]
    benefit_oracle_actions = [choose_least_intensive_beneficial_action(row) for row in eligible]
    classifications = [classify_case(row, action) for row, action in zip(eligible, oracle_actions)]
    policies = {
        "always_off": _policy_summary(eligible, [0] * len(eligible)),
        "always_soft": _policy_summary(eligible, [1] * len(eligible)),
        "always_strong": _policy_summary(eligible, [2] * len(eligible)),
        "counterfactual_oracle": _policy_summary(eligible, oracle_actions),
        "traffic_oracle_unconstrained": _policy_summary(eligible, traffic_oracle_actions),
        "least_intensive_benefit_oracle": _policy_summary(eligible, benefit_oracle_actions),
    }
    off = policies["always_off"]
    oracle = policies["counterfactual_oracle"]
    case_details = []
    for row, action, traffic_action, benefit_action, classification in zip(
        eligible,
        oracle_actions,
        traffic_oracle_actions,
        benefit_oracle_actions,
        classifications,
    ):
        case_details.append(
            {
                "seed": int(row["seed"]),
                "decision_step": int(row["decision_step"]),
                "oracle_action": action,
                "oracle_action_name": ACTION_NAMES[action],
                "traffic_oracle_action": traffic_action,
                "traffic_oracle_action_name": ACTION_NAMES[traffic_action],
                "least_intensive_benefit_action": benefit_action,
                "least_intensive_benefit_action_name": ACTION_NAMES[benefit_action],
                "pareto_actions": pareto_actions(row),
                "pareto_action_names": [
                    ACTION_NAMES[pareto_action] for pareto_action in pareto_actions(row)
                ],
                "classification": classification,
                "soft_effect_vs_off": treatment_effects(row, 1),
                "strong_effect_vs_off": treatment_effects(row, 2),
            }
        )

    return {
        "schema_version": 1,
        "case_count": len(rows),
        "eligible_case_count": len(eligible),
        "ineligible_case_count": len(ineligible),
        "oracle_definition": {
            "priority": [
                "no collision, teleport, hard-brake, or low-TTC excess versus off",
                "avoid sustained congestion",
                "minimize queue area",
                "minimize incremental delay",
                "maximize merge discharge",
                "minimize treatment duty",
            ],
            "note": "This is an upper-bound selector using outcomes, not a deployable gate.",
        },
        "oracle_action_counts": {
            ACTION_NAMES[action]: oracle_actions.count(action) for action in (0, 1, 2)
        },
        "traffic_oracle_action_counts": {
            ACTION_NAMES[action]: traffic_oracle_actions.count(action) for action in (0, 1, 2)
        },
        "least_intensive_benefit_action_counts": {
            ACTION_NAMES[action]: benefit_oracle_actions.count(action) for action in (0, 1, 2)
        },
        "case_classification_counts": dict(sorted(Counter(classifications).items())),
        "policies": policies,
        "oracle_headroom_vs_off": {
            "sustained_congestion_cases_avoided": int(
                off["sustained_congestion_count"] - oracle["sustained_congestion_count"]
            ),
            "mean_queue_area_reduction_vehicle_s": None
            if not eligible
            else float(off["mean_queue_area_vehicle_s"])
            - float(oracle["mean_queue_area_vehicle_s"]),
            "mean_incremental_delay_reduction_s": None
            if not eligible
            else float(off["mean_incremental_delay_s"]) - float(oracle["mean_incremental_delay_s"]),
        },
        "traffic_oracle_headroom_vs_off": {
            "sustained_congestion_cases_avoided": int(
                off["sustained_congestion_count"]
                - policies["traffic_oracle_unconstrained"]["sustained_congestion_count"]
            ),
            "mean_queue_area_reduction_vehicle_s": None
            if not eligible
            else float(off["mean_queue_area_vehicle_s"])
            - float(policies["traffic_oracle_unconstrained"]["mean_queue_area_vehicle_s"]),
            "mean_incremental_delay_reduction_s": None
            if not eligible
            else float(off["mean_incremental_delay_s"])
            - float(policies["traffic_oracle_unconstrained"]["mean_incremental_delay_s"]),
            "warning": (
                "Unconstrained traffic upper bound; inspect safety deltas before "
                "claiming deployable benefit."
            ),
        },
        "least_intensive_benefit_headroom_vs_off": {
            "sustained_congestion_cases_avoided": int(
                off["sustained_congestion_count"]
                - policies["least_intensive_benefit_oracle"]["sustained_congestion_count"]
            ),
            "mean_queue_area_reduction_vehicle_s": None
            if not eligible
            else float(off["mean_queue_area_vehicle_s"])
            - float(policies["least_intensive_benefit_oracle"]["mean_queue_area_vehicle_s"]),
            "mean_incremental_delay_reduction_s": None
            if not eligible
            else float(off["mean_incremental_delay_s"])
            - float(policies["least_intensive_benefit_oracle"]["mean_incremental_delay_s"]),
            "warning": "Traffic dominance only; safety constraint is applied separately.",
        },
        "cases": case_details,
        "ineligible_seeds": [int(row["seed"]) for row in ineligible],
    }
