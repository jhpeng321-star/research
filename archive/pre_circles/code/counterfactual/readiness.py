"""Readiness audit combining calibration, traffic benefit, and safety budgets."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Sequence

from counterfactual.oracle import ACTION_NAMES, is_dominant_traffic_benefit


def _freeze_candidate(calibration: Mapping[str, Any]) -> Mapping[str, Any] | None:
    requirements = calibration["freeze_requirements"]
    if int(calibration["seed_count"]) < int(requirements["minimum_seed_count"]):
        return None
    candidates = [
        candidate
        for candidate in calibration["top_candidates"]
        if candidate["valid_two_class_split"]
        and int(candidate["positive_window_count"]) >= int(requirements["minimum_positive_windows"])
        and int(candidate["negative_window_count"]) >= int(requirements["minimum_negative_windows"])
    ]
    return candidates[0] if candidates else None


def _same_definition(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "queue_speed_threshold_mps",
        "congestion_queue_count",
        "congestion_speed_threshold_mps",
        "sustain_steps",
    )
    return all(float(left[key]) == float(right[key]) for key in keys)


def evaluate_readiness(
    calibration: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_paired_cases: int = 10,
    minimum_off_labels: int = 3,
    minimum_active_labels: int = 3,
) -> Dict[str, Any]:
    """Determine whether paired data support training a deployable gate."""
    candidate = _freeze_candidate(calibration)
    state = calibration["empirical_no_control_state"]
    hard_brake_limit = float(state["window_hard_brake_rate_vehicle_steps_per_s"]["p95"])
    low_ttc_limit = float(state["window_low_ttc_exposure_fraction"]["p95"])
    label_definition = candidate["definition"] if candidate else None
    eligible_rows = [row for row in rows if row.get("eligible")]

    cases = []
    selected_actions = []
    definition_mismatches = []
    for row in eligible_rows:
        case_id = str(row.get("case_id", f"seed-{row['seed']}-step-{row['decision_step']}"))
        row_definition = row["outcome_definition"]
        definition_match = bool(
            label_definition and _same_definition(row_definition, label_definition)
        )
        if not definition_match:
            definition_mismatches.append(case_id)

        action_results = {}
        deployable_actions = []
        available_actions = [action for action in (1, 2) if str(action) in row.get("arms", {})]
        for action in available_actions:
            outcome = row["arms"][str(action)]["outcomes"]
            duration_s = max(
                1e-9,
                float(outcome["observed_outcome_steps"]) * float(outcome["step_length_s"]),
            )
            hard_brake_rate = float(outcome["hard_brake_vehicle_steps"]) / duration_s
            low_ttc_fraction = float(outcome["low_ttc_exposure_steps"]) / max(
                1, int(outcome["observed_outcome_steps"])
            )
            collision_free = int(outcome.get("collision_unique_vehicle_count", 0)) == 0
            teleport_free = int(outcome.get("starting_teleport_unique_vehicle_count", 0)) == 0
            safety_admissible = (
                collision_free
                and teleport_free
                and hard_brake_rate <= hard_brake_limit
                and low_ttc_fraction <= low_ttc_limit
            )
            traffic_beneficial = is_dominant_traffic_benefit(row, action)
            deployable = definition_match and traffic_beneficial and safety_admissible
            if deployable:
                deployable_actions.append(action)
            action_results[str(action)] = {
                "action_name": ACTION_NAMES[action],
                "traffic_beneficial": traffic_beneficial,
                "hard_brake_rate_vehicle_steps_per_s": hard_brake_rate,
                "hard_brake_p95_limit": hard_brake_limit,
                "low_ttc_exposure_fraction": low_ttc_fraction,
                "low_ttc_p95_limit": low_ttc_limit,
                "collision_free": collision_free,
                "teleport_free": teleport_free,
                "safety_admissible": safety_admissible,
                "deployable": deployable,
            }
        selected = min(deployable_actions) if deployable_actions else 0
        selected_actions.append(selected)
        cases.append(
            {
                "case_id": case_id,
                "seed": int(row["seed"]),
                "decision_step": int(row["decision_step"]),
                "definition_match": definition_match,
                "available_actions": [0, *available_actions],
                "selected_action": selected,
                "selected_action_name": ACTION_NAMES[selected],
                "actions": action_results,
            }
        )

    counts = Counter(selected_actions)
    enough_cases = len(eligible_rows) >= minimum_paired_cases
    enough_off = counts[0] >= minimum_off_labels
    enough_active = counts[1] + counts[2] >= minimum_active_labels
    label_ready = candidate is not None
    safety_headroom = counts[1] + counts[2] > 0
    ready = (
        label_ready
        and not definition_mismatches
        and enough_cases
        and enough_off
        and enough_active
        and safety_headroom
    )
    blockers = []
    if not label_ready:
        blockers.append("no outcome definition meets calibration freeze counts")
    if definition_mismatches:
        blockers.append("paired cases do not all use the freeze-candidate definition")
    if not enough_cases:
        blockers.append(f"only {len(eligible_rows)} paired cases; require {minimum_paired_cases}")
    if not safety_headroom:
        blockers.append("no active treatment passes both traffic and safety budgets")
    elif not enough_active:
        blockers.append(
            f"only {counts[1] + counts[2]} active labels; require {minimum_active_labels}"
        )
    if not enough_off:
        blockers.append(f"only {counts[0]} off labels; require {minimum_off_labels}")

    return {
        "schema_version": 1,
        "ready_to_train_deployable_gate": ready,
        "status": "ready" if ready else "not_ready",
        "blockers": blockers,
        "label_freeze_candidate": label_definition,
        "label_candidate_diagnostics": candidate,
        "safety_budget": {
            "source": "no-control calibration p95",
            "hard_brake_rate_vehicle_steps_per_s_max": hard_brake_limit,
            "low_ttc_exposure_fraction_max": low_ttc_limit,
            "collision_unique_vehicle_count_max": 0,
            "starting_teleport_unique_vehicle_count_max": 0,
        },
        "paired_case_count": len(eligible_rows),
        "definition_mismatch_case_ids": definition_mismatches,
        "selected_action_counts": {ACTION_NAMES[action]: counts[action] for action in (0, 1, 2)},
        "training_sample_requirements": {
            "minimum_paired_cases": minimum_paired_cases,
            "minimum_off_labels": minimum_off_labels,
            "minimum_active_labels": minimum_active_labels,
        },
        "cases": cases,
    }
