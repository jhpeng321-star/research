"""Leakage-safe calibration of independent congestion outcome thresholds.

Only no-control traces are accepted. Candidate event definitions are ranked by
whether their positive windows have higher incremental delay and lower merge
discharge than their negative windows. Health and treatment outcomes are not
used to select the definition.
"""

from __future__ import annotations

from itertools import product
import math
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from config.constants import PRE_MERGE_EDGE
from counterfactual.outcomes import OutcomeDefinition, summarize_branch
from counterfactual.twin_replay import ReplayTrace, StepSnapshot


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _quantiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {
        f"p{int(percentile * 100):02d}": _percentile(values, percentile)
        for percentile in (0.05, 0.25, 0.5, 0.75, 0.95)
    }


def _pre_merge_speeds(snapshot: StepSnapshot) -> list[float]:
    return [
        float(row[4])
        for row in snapshot.vehicles
        if str(row[1]) == PRE_MERGE_EDGE and math.isfinite(float(row[4]))
    ]


def _event_from_snapshots(
    snapshots: Sequence[StepSnapshot], definition: OutcomeDefinition
) -> tuple[bool, Optional[int]]:
    run_length = 0
    for snapshot in snapshots:
        speeds = _pre_merge_speeds(snapshot)
        queue_count = sum(speed < definition.queue_speed_threshold_mps for speed in speeds)
        mean_speed = mean(speeds) if speeds else None
        congested = (
            queue_count >= definition.congestion_queue_count
            and mean_speed is not None
            and mean_speed <= definition.congestion_speed_threshold_mps
        )
        run_length = run_length + 1 if congested else 0
        if run_length >= definition.sustain_steps:
            return True, snapshot.relative_step + 1 - definition.sustain_steps
    return False, None


def _standardized_separation(
    positives: Sequence[float], negatives: Sequence[float], *, higher_positive: bool
) -> float:
    combined = [*positives, *negatives]
    scale = pstdev(combined) if len(combined) > 1 else 0.0
    if scale <= 1e-12:
        return 0.0
    difference = mean(positives) - mean(negatives)
    return (difference if higher_positive else -difference) / scale


def _candidate_result(
    definition: OutcomeDefinition,
    windows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    labels = []
    onset_steps = []
    for window in windows:
        event, onset = _event_from_snapshots(window["snapshots"], definition)
        labels.append(event)
        onset_steps.append(onset)

    positive = [window for window, label in zip(windows, labels) if label]
    negative = [window for window, label in zip(windows, labels) if not label]
    unique_seeds = {int(window["seed"]) for window in windows}
    positive_seeds = {int(window["seed"]) for window in positive}
    negative_seeds = {int(window["seed"]) for window in negative}
    minimum_class_seed_count = 2 if len(unique_seeds) >= 3 else 1
    prevalence = len(positive) / len(windows) if windows else 0.0
    valid = (
        len(positive) >= 2
        and len(negative) >= 2
        and len(positive_seeds) >= minimum_class_seed_count
        and len(negative_seeds) >= minimum_class_seed_count
    )
    delay_separation = 0.0
    discharge_separation = 0.0
    if valid:
        delay_separation = _standardized_separation(
            [float(window["delay_rate_s_per_s"]) for window in positive],
            [float(window["delay_rate_s_per_s"]) for window in negative],
            higher_positive=True,
        )
        discharge_separation = _standardized_separation(
            [float(window["discharge_rate_veh_per_s"]) for window in positive],
            [float(window["discharge_rate_veh_per_s"]) for window in negative],
            higher_positive=False,
        )

    prevalence_penalty = abs(prevalence - 0.35)
    score = (
        delay_separation + 0.5 * discharge_separation - 0.5 * prevalence_penalty
        if valid
        else -1_000.0 - prevalence_penalty
    )
    seed_rates = {}
    for seed in sorted({int(window["seed"]) for window in windows}):
        selected = [label for window, label in zip(windows, labels) if int(window["seed"]) == seed]
        seed_rates[str(seed)] = sum(selected) / len(selected) if selected else 0.0
    return {
        "definition": definition.to_dict(),
        "score": score,
        "valid_two_class_split": valid,
        "positive_window_count": len(positive),
        "negative_window_count": len(negative),
        "positive_seed_count": len(positive_seeds),
        "negative_seed_count": len(negative_seeds),
        "minimum_seed_groups_per_class": minimum_class_seed_count,
        "positive_prevalence": prevalence,
        "delay_standardized_separation": delay_separation,
        "discharge_standardized_separation": discharge_separation,
        "positive_rate_by_seed": seed_rates,
        "window_labels": [
            {
                "seed": int(window["seed"]),
                "decision_step": int(window["decision_step"]),
                "sustained_congestion": bool(label),
                "congestion_onset_step": onset,
            }
            for window, label, onset in zip(windows, labels, onset_steps)
        ],
    }


def _candidate_definitions(
    *,
    queue_speed_thresholds: Iterable[float],
    congestion_queue_counts: Iterable[int],
    congestion_speed_thresholds: Iterable[float],
    sustain_steps: Iterable[int],
    ttc_threshold_s: float,
    hard_brake_threshold_mps2: float,
) -> Iterable[OutcomeDefinition]:
    for queue_speed, queue_count, mean_speed, persistence in product(
        queue_speed_thresholds,
        congestion_queue_counts,
        congestion_speed_thresholds,
        sustain_steps,
    ):
        yield OutcomeDefinition(
            queue_speed_threshold_mps=float(queue_speed),
            congestion_queue_count=int(queue_count),
            congestion_speed_threshold_mps=float(mean_speed),
            sustain_steps=int(persistence),
            ttc_threshold_s=float(ttc_threshold_s),
            hard_brake_threshold_mps2=float(hard_brake_threshold_mps2),
        )


def calibrate_outcome_definition(
    traces: Sequence[ReplayTrace],
    *,
    decision_steps: Sequence[int],
    outcome_steps: int,
    queue_speed_thresholds: Sequence[float] = (10.0, 12.5, 15.0),
    congestion_queue_counts: Sequence[int] = (10, 20, 30, 40, 50),
    congestion_speed_thresholds: Sequence[float] = (10.0, 12.5, 15.0, 17.5),
    sustain_steps: Sequence[int] = (30, 60, 90),
    ttc_threshold_s: float = 1.5,
    hard_brake_threshold_mps2: float = -4.5,
    minimum_seed_count_for_freeze: int = 10,
) -> Dict[str, Any]:
    """Rank independent event definitions using only off-control traces."""
    if not traces:
        raise ValueError("at least one no-control trace is required")
    if outcome_steps <= 0:
        raise ValueError("outcome_steps must be positive")
    if any(trace.post_action != 0 or trace.prefix_action != 0 for trace in traces):
        raise ValueError("calibration accepts only off-control traces")

    base_definition = OutcomeDefinition(
        ttc_threshold_s=ttc_threshold_s,
        hard_brake_threshold_mps2=hard_brake_threshold_mps2,
    )
    windows = []
    all_pre_merge_speeds = []
    all_queue_counts_at_15 = []
    for trace in traces:
        for snapshot in trace.snapshots:
            speeds = _pre_merge_speeds(snapshot)
            all_pre_merge_speeds.extend(speeds)
            all_queue_counts_at_15.append(sum(speed < 15.0 for speed in speeds))
        for decision_step in decision_steps:
            if decision_step + outcome_steps > trace.completed_steps:
                continue
            summary = summarize_branch(
                trace,
                decision_step=decision_step,
                outcome_steps=outcome_steps,
                definition=base_definition,
            )
            duration_s = max(
                1e-9,
                float(summary["observed_outcome_steps"]) * float(summary["step_length_s"]),
            )
            snapshots = tuple(
                snapshot
                for snapshot in trace.snapshots
                if decision_step <= snapshot.relative_step <= decision_step + outcome_steps
            )
            windows.append(
                {
                    "seed": trace.seed,
                    "decision_step": int(decision_step),
                    "snapshots": snapshots,
                    "delay_rate_s_per_s": float(summary["incremental_delay_s"]) / duration_s,
                    "discharge_rate_veh_per_s": float(summary["merge_discharge_count"])
                    / duration_s,
                    "hard_brake_rate_vehicle_steps_per_s": float(
                        summary["hard_brake_vehicle_steps"]
                    )
                    / duration_s,
                    "low_ttc_exposure_fraction": float(summary["low_ttc_exposure_steps"])
                    / max(1, int(summary["observed_outcome_steps"])),
                    "collision_unique_vehicle_count": int(
                        summary["collision_unique_vehicle_count"]
                    ),
                    "starting_teleport_unique_vehicle_count": int(
                        summary["starting_teleport_unique_vehicle_count"]
                    ),
                    "decision_health": float(
                        trace.snapshots[decision_step].supervisor_observation[0]
                    ),
                }
            )
    if not windows:
        raise ValueError("no complete calibration windows exist in the traces")

    ranked = [
        _candidate_result(definition, windows)
        for definition in _candidate_definitions(
            queue_speed_thresholds=queue_speed_thresholds,
            congestion_queue_counts=congestion_queue_counts,
            congestion_speed_thresholds=congestion_speed_thresholds,
            sustain_steps=sustain_steps,
            ttc_threshold_s=ttc_threshold_s,
            hard_brake_threshold_mps2=hard_brake_threshold_mps2,
        )
    ]
    ranked.sort(key=lambda result: float(result["score"]), reverse=True)
    seeds = sorted({trace.seed for trace in traces})
    valid = [result for result in ranked if result["valid_two_class_split"]]
    freeze_eligible = [
        result
        for result in valid
        if result["positive_window_count"] >= 5 and result["negative_window_count"] >= 5
    ]
    if len(seeds) >= minimum_seed_count_for_freeze and freeze_eligible:
        recommended = freeze_eligible[0]
        recommendation_policy = "highest_score_meeting_freeze_requirements"
    else:
        recommended = valid[0] if valid else ranked[0]
        recommendation_policy = "highest_screening_score"
    freeze_ready = len(seeds) >= minimum_seed_count_for_freeze and recommended in freeze_eligible
    return {
        "schema_version": 1,
        "calibration_scope": "no_control_only",
        "selection_inputs": [
            "future incremental delay rate",
            "future merge discharge rate",
        ],
        "explicitly_excluded_from_selection": [
            "health",
            "soft outcomes",
            "strong outcomes",
        ],
        "seed_count": len(seeds),
        "seeds": seeds,
        "trace_digests": {str(trace.seed): trace.trace_digest for trace in traces},
        "decision_steps": [int(step) for step in decision_steps],
        "outcome_steps": int(outcome_steps),
        "window_count": len(windows),
        "candidate_count": len(ranked),
        "empirical_no_control_state": {
            "pre_merge_vehicle_speed_mps": _quantiles(all_pre_merge_speeds),
            "slow_vehicle_count_below_15_mps": _quantiles(all_queue_counts_at_15),
            "window_delay_rate_s_per_s": _quantiles(
                [float(window["delay_rate_s_per_s"]) for window in windows]
            ),
            "window_discharge_rate_veh_per_s": _quantiles(
                [float(window["discharge_rate_veh_per_s"]) for window in windows]
            ),
            "window_hard_brake_rate_vehicle_steps_per_s": _quantiles(
                [float(window["hard_brake_rate_vehicle_steps_per_s"]) for window in windows]
            ),
            "window_low_ttc_exposure_fraction": _quantiles(
                [float(window["low_ttc_exposure_fraction"]) for window in windows]
            ),
            "collision_window_count": sum(
                int(window["collision_unique_vehicle_count"]) > 0 for window in windows
            ),
            "teleport_window_count": sum(
                int(window["starting_teleport_unique_vehicle_count"]) > 0 for window in windows
            ),
        },
        "recommended_definition": recommended["definition"],
        "recommendation_policy": recommendation_policy,
        "recommended_diagnostics": {
            key: value
            for key, value in recommended.items()
            if key not in {"definition", "window_labels"}
        },
        "recommended_window_labels": recommended["window_labels"],
        "freeze_ready": freeze_ready,
        "freeze_requirements": {
            "minimum_seed_count": minimum_seed_count_for_freeze,
            "minimum_positive_windows": 5,
            "minimum_negative_windows": 5,
        },
        "status": "ready_to_freeze" if freeze_ready else "screening_only",
        "top_candidates": [
            {key: value for key, value in result.items() if key != "window_labels"}
            for result in ranked[:20]
        ],
    }
