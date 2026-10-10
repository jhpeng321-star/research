"""Outcome labels that are independent of the supervisor health signal.

The treatment gate may consume the health trajectory as a feature, but its
label must be grounded in future traffic outcomes.  This module derives those
labels directly from paired SUMO vehicle traces.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from statistics import median
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from config.constants import (
    DEFAULT_VEH_LENGTH,
    MERGE_EDGE,
    PRE_MERGE_EDGE,
    PRE_MERGE_LANE_LENGTH,
)
from counterfactual.twin_replay import ReplayTrace, Scalar, StepSnapshot, VehicleState


@dataclass(frozen=True)
class OutcomeDefinition:
    """Frozen event and safety thresholds for one counterfactual dataset."""

    queue_speed_threshold_mps: float = 15.0
    congestion_queue_count: int = 5
    congestion_speed_threshold_mps: float = 15.0
    sustain_steps: int = 60
    ttc_threshold_s: float = 1.5
    hard_brake_threshold_mps2: float = -4.5
    vehicle_length_m: float = DEFAULT_VEH_LENGTH

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _number(value: Scalar) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


def _vehicle_map(snapshot: StepSnapshot) -> Dict[str, VehicleState]:
    return {str(row[0]): row for row in snapshot.vehicles}


def _mean(values: Iterable[float]) -> Optional[float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _low_ttc_pairs(
    vehicles: Sequence[VehicleState], definition: OutcomeDefinition
) -> Tuple[Tuple[str, str, float], ...]:
    by_lane: Dict[str, list[VehicleState]] = {}
    for row in vehicles:
        by_lane.setdefault(str(row[2]), []).append(row)

    events = []
    for lane_rows in by_lane.values():
        ordered = sorted(lane_rows, key=lambda row: _number(row[3]))
        for follower, leader in zip(ordered, ordered[1:]):
            follower_speed = _number(follower[4])
            leader_speed = _number(leader[4])
            closing_speed = follower_speed - leader_speed
            if not math.isfinite(closing_speed) or closing_speed <= 0.0:
                continue
            gap = _number(leader[3]) - _number(follower[3]) - definition.vehicle_length_m
            ttc = max(0.0, gap) / closing_speed
            if ttc < definition.ttc_threshold_s:
                events.append((str(follower[0]), str(leader[0]), float(ttc)))
    return tuple(events)


def step_metrics(snapshot: StepSnapshot, definition: OutcomeDefinition) -> Dict[str, Any]:
    """Derive one-step traffic, safety, and CAV-coverage measurements."""
    pre_merge = [row for row in snapshot.vehicles if str(row[1]) == PRE_MERGE_EDGE]
    merge = [row for row in snapshot.vehicles if str(row[1]) == MERGE_EDGE]
    pre_speeds = [_number(row[4]) for row in pre_merge]
    merge_speeds = [_number(row[4]) for row in merge]
    low_ttc = _low_ttc_pairs(snapshot.vehicles, definition)
    hard_brakes = [
        str(row[0])
        for row in snapshot.vehicles
        if _number(row[5]) <= definition.hard_brake_threshold_mps2
    ]

    vehicles = _vehicle_map(snapshot)
    active_rows = [
        vehicles[vehicle_id] for vehicle_id in snapshot.active_cavs if vehicle_id in vehicles
    ]
    active_pre_merge = [row for row in active_rows if str(row[1]) == PRE_MERGE_EDGE]
    return {
        "relative_step": snapshot.relative_step,
        "sim_time": _number(snapshot.sim_time),
        "queue_count": sum(speed < definition.queue_speed_threshold_mps for speed in pre_speeds),
        "pre_merge_vehicle_count": len(pre_merge),
        "pre_merge_mean_speed_mps": _mean(pre_speeds),
        "merge_mean_speed_mps": _mean(merge_speeds),
        "hard_brake_vehicle_ids": tuple(hard_brakes),
        "low_ttc_pairs": low_ttc,
        "collision_vehicle_ids": snapshot.collision_vehicle_ids,
        "starting_teleport_vehicle_ids": snapshot.starting_teleport_vehicle_ids,
        "active_cav_count": len(active_rows),
        "active_cav_pre_merge_count": len(active_pre_merge),
        "active_cav_pre_merge_lane_count": len({str(row[2]) for row in active_pre_merge}),
    }


def _step_length(snapshots: Sequence[StepSnapshot]) -> float:
    differences = []
    for left, right in zip(snapshots, snapshots[1:]):
        difference = _number(right.sim_time) - _number(left.sim_time)
        if math.isfinite(difference) and difference > 0.0:
            differences.append(difference)
    return float(median(differences)) if differences else 0.0


def _sustained_congestion(
    metrics: Sequence[Mapping[str, Any]], definition: OutcomeDefinition
) -> Tuple[bool, Optional[int]]:
    run_length = 0
    for index, metric in enumerate(metrics):
        speed = metric["pre_merge_mean_speed_mps"]
        congested = (
            metric["queue_count"] >= definition.congestion_queue_count
            and speed is not None
            and float(speed) <= definition.congestion_speed_threshold_mps
        )
        run_length = run_length + 1 if congested else 0
        if run_length >= definition.sustain_steps:
            onset_index = index + 1 - definition.sustain_steps
            return True, int(metrics[onset_index]["relative_step"])
    return False, None


def summarize_predecision_congestion(
    trace: ReplayTrace,
    *,
    decision_step: int,
    definition: OutcomeDefinition = OutcomeDefinition(),
) -> Dict[str, Any]:
    """Label collapse already present using only snapshots available at decision."""
    snapshots = [
        snapshot
        for snapshot in trace.snapshots
        if int(snapshot.relative_step) <= int(decision_step)
    ]
    metrics = [step_metrics(snapshot, definition) for snapshot in snapshots]
    sustained, onset_step = _sustained_congestion(metrics, definition)
    return {
        "sustained_congestion": sustained,
        "congestion_onset_step": onset_step,
        "observed_through_step": int(decision_step),
        "outcome_definition": definition.to_dict(),
    }


_WALL_BANDS_M = {
    1: (200.0, 400.0),
    2: (200.0, 800.0),
}


def _audit_by_step(trace: ReplayTrace) -> Dict[int, Mapping[str, Any]]:
    return {
        int(audit["relative_step"]): audit
        for audit in trace.controller_audits
        if audit.get("relative_step") is not None
    }


def _controlled_wall_cavs(
    snapshot: StepSnapshot,
    audit: Optional[Mapping[str, Any]] = None,
) -> set[str]:
    """Return the exact controlled CAVs, with a legacy-band fallback."""
    if audit is not None and "wall_override_ids" in audit:
        return {str(vehicle_id) for vehicle_id in audit["wall_override_ids"]}
    band = _WALL_BANDS_M.get(int(snapshot.action))
    if band is None:
        return set()
    lower, upper = band
    active = set(snapshot.active_cavs)
    return {
        str(row[0])
        for row in snapshot.vehicles
        if str(row[0]) in active
        and str(row[1]) == PRE_MERGE_EDGE
        and lower <= PRE_MERGE_LANE_LENGTH - _number(row[3]) <= upper
    }


def hard_brake_attribution(
    trace: ReplayTrace,
    snapshots: Sequence[StepSnapshot],
    definition: OutcomeDefinition,
    *,
    event_sample_limit: int = 200,
) -> Dict[str, Any]:
    """Attribute hard-brake exposures and event onsets to traffic context."""
    exposure_counts: Dict[str, Counter[str]] = {
        "vehicle_class": Counter(),
        "phase": Counter(),
        "road_id": Counter(),
        "lane_id": Counter(),
    }
    onset_counts: Dict[str, Counter[str]] = {
        "vehicle_class": Counter(),
        "phase": Counter(),
        "road_id": Counter(),
        "lane_id": Counter(),
    }
    exposure_flags: Counter[str] = Counter()
    onset_flags: Counter[str] = Counter()
    unique_vehicle_ids: set[str] = set()
    event_samples: list[Dict[str, Any]] = []
    event_start_count = 0
    if not snapshots:
        raise ValueError("hard-brake attribution requires at least one snapshot")
    previous_hard_brakes = {
        str(row[0])
        for row in snapshots[0].vehicles
        if _number(row[5]) <= definition.hard_brake_threshold_mps2
    }
    audits = _audit_by_step(trace)
    treatment_end = (
        math.inf
        if trace.treatment_steps is None
        else snapshots[0].relative_step + trace.treatment_steps
    )

    for previous, current in zip(snapshots, snapshots[1:]):
        previous_map = _vehicle_map(previous)
        current_map = _vehicle_map(current)
        active_cavs = set(current.active_cavs)
        audit = audits.get(int(current.relative_step))
        controlled_cavs = _controlled_wall_cavs(current, audit)

        leaders: Dict[str, VehicleState] = {}
        by_lane: Dict[str, list[VehicleState]] = {}
        for row in current.vehicles:
            by_lane.setdefault(str(row[2]), []).append(row)
        for lane_rows in by_lane.values():
            ordered = sorted(lane_rows, key=lambda row: _number(row[3]))
            for follower, leader in zip(ordered, ordered[1:]):
                leaders[str(follower[0])] = leader

        controlled_rows = [
            current_map[vehicle_id] for vehicle_id in controlled_cavs if vehicle_id in current_map
        ]
        current_hard_brakes = {
            str(row[0])
            for row in current.vehicles
            if _number(row[5]) <= definition.hard_brake_threshold_mps2
        }
        treatment_state = str(audit.get("treatment_state", "")).lower() if audit is not None else ""
        phase = treatment_state or (
            "treatment_window" if current.relative_step <= treatment_end else "post_release"
        )

        for vehicle_id in sorted(current_hard_brakes):
            row = current_map[vehicle_id]
            vehicle_class = "cav" if vehicle_id in active_cavs else "hdv"
            road_id = str(row[1])
            lane_id = str(row[2])
            previous_row = previous_map.get(vehicle_id)
            lane_changed = previous_row is not None and str(previous_row[2]) != lane_id
            controlled_wall_cav = vehicle_id in controlled_cavs

            leader = leaders.get(vehicle_id)
            leader_id = str(leader[0]) if leader is not None else None
            leader_gap_m = None
            leader_ttc_s = None
            leader_controlled_wall_cav = leader_id in controlled_cavs
            if leader is not None:
                leader_gap_m = max(
                    0.0,
                    _number(leader[3]) - _number(row[3]) - definition.vehicle_length_m,
                )
                closing_speed = _number(row[4]) - _number(leader[4])
                if math.isfinite(closing_speed) and closing_speed > 0.0:
                    leader_ttc_s = leader_gap_m / closing_speed

            same_road_distances = [
                abs(_number(other[3]) - _number(row[3]))
                for other in controlled_rows
                if str(other[1]) == road_id and str(other[0]) != vehicle_id
            ]
            same_lane_distances = [
                abs(_number(other[3]) - _number(row[3]))
                for other in controlled_rows
                if str(other[2]) == lane_id and str(other[0]) != vehicle_id
            ]
            nearest_same_road_m = min(same_road_distances, default=None)
            nearest_same_lane_m = min(same_lane_distances, default=None)
            within_50m_same_road = nearest_same_road_m is not None and nearest_same_road_m <= 50.0
            within_50m_same_lane = nearest_same_lane_m is not None and nearest_same_lane_m <= 50.0
            event_start = vehicle_id not in previous_hard_brakes

            unique_vehicle_ids.add(vehicle_id)
            for counter_name, value in (
                ("vehicle_class", vehicle_class),
                ("phase", phase),
                ("road_id", road_id),
                ("lane_id", lane_id),
            ):
                exposure_counts[counter_name][value] += 1
                if event_start:
                    onset_counts[counter_name][value] += 1
            for flag_name, flag in (
                ("controlled_wall_cav", controlled_wall_cav),
                ("leader_controlled_wall_cav", leader_controlled_wall_cav),
                ("within_50m_controlled_cav_same_road", within_50m_same_road),
                ("within_50m_controlled_cav_same_lane", within_50m_same_lane),
                ("lane_changed", lane_changed),
            ):
                if flag:
                    exposure_flags[flag_name] += 1
                    if event_start:
                        onset_flags[flag_name] += 1

            if event_start:
                event_start_count += 1
            if event_start and len(event_samples) < event_sample_limit:
                event_samples.append(
                    {
                        "relative_step": current.relative_step,
                        "sim_time_s": _number(current.sim_time),
                        "phase": phase,
                        "applied_action": int(current.action),
                        "vehicle_id": vehicle_id,
                        "vehicle_class": vehicle_class,
                        "road_id": road_id,
                        "lane_id": lane_id,
                        "lane_position_m": _number(row[3]),
                        "speed_mps": _number(row[4]),
                        "acceleration_mps2": _number(row[5]),
                        "lane_changed": lane_changed,
                        "controlled_wall_cav": controlled_wall_cav,
                        "leader_id": leader_id,
                        "leader_gap_m": leader_gap_m,
                        "leader_ttc_s": leader_ttc_s,
                        "leader_controlled_wall_cav": leader_controlled_wall_cav,
                        "nearest_controlled_cav_same_road_m": nearest_same_road_m,
                        "nearest_controlled_cav_same_lane_m": nearest_same_lane_m,
                    }
                )
        previous_hard_brakes = current_hard_brakes

    return {
        "hard_brake_threshold_mps2": definition.hard_brake_threshold_mps2,
        "exposure_vehicle_steps": sum(exposure_counts["vehicle_class"].values()),
        "unique_vehicle_count": len(unique_vehicle_ids),
        "event_start_count": event_start_count,
        "event_sample_truncated": event_start_count > len(event_samples),
        "exposure_counts": {
            name: dict(sorted(counter.items())) for name, counter in exposure_counts.items()
        },
        "event_start_counts": {
            name: dict(sorted(counter.items())) for name, counter in onset_counts.items()
        },
        "exposure_context_counts": dict(sorted(exposure_flags.items())),
        "event_start_context_counts": dict(sorted(onset_flags.items())),
        "event_starts": event_samples,
    }


def low_ttc_attribution(
    trace: ReplayTrace,
    snapshots: Sequence[StepSnapshot],
    definition: OutcomeDefinition,
    *,
    event_sample_limit: int = 500,
) -> Dict[str, Any]:
    """Attribute low-TTC pair exposure to treatment phase and wall context."""
    if not snapshots:
        raise ValueError("low-TTC attribution requires at least one snapshot")
    audits = _audit_by_step(trace)
    previous_pairs = {
        (follower, leader)
        for follower, leader, _ttc in _low_ttc_pairs(snapshots[0].vehicles, definition)
    }
    treatment_end = (
        math.inf
        if trace.treatment_steps is None
        else snapshots[0].relative_step + trace.treatment_steps
    )
    exposure_counts: Dict[str, Counter[str]] = {
        "phase": Counter(),
        "pair_class": Counter(),
        "road_id": Counter(),
        "lane_id": Counter(),
    }
    exposure_flags: Counter[str] = Counter()
    onset_flags: Counter[str] = Counter()
    event_samples: list[Dict[str, Any]] = []
    event_start_count = 0
    unique_pairs: set[Tuple[str, str]] = set()

    for previous, current in zip(snapshots, snapshots[1:]):
        current_map = _vehicle_map(current)
        previous_map = _vehicle_map(previous)
        active_cavs = set(current.active_cavs)
        audit = audits.get(int(current.relative_step))
        controlled_cavs = _controlled_wall_cavs(current, audit)
        controlled_rows = [
            current_map[vehicle_id] for vehicle_id in controlled_cavs if vehicle_id in current_map
        ]
        treatment_state = str(audit.get("treatment_state", "")).lower() if audit is not None else ""
        phase = treatment_state or (
            "treatment_window" if current.relative_step <= treatment_end else "post_release"
        )
        current_pairs = _low_ttc_pairs(current.vehicles, definition)
        current_pair_ids = {(follower, leader) for follower, leader, _ttc in current_pairs}

        for follower_id, leader_id, ttc_s in current_pairs:
            follower = current_map[follower_id]
            leader = current_map[leader_id]
            follower_class = "cav" if follower_id in active_cavs else "hdv"
            leader_class = "cav" if leader_id in active_cavs else "hdv"
            pair_class = f"{follower_class}_following_{leader_class}"
            road_id = str(follower[1])
            lane_id = str(follower[2])
            follower_controlled = follower_id in controlled_cavs
            leader_controlled = leader_id in controlled_cavs
            previous_follower = previous_map.get(follower_id)
            follower_lane_changed = (
                previous_follower is not None and str(previous_follower[2]) != lane_id
            )
            same_road_distances = [
                abs(_number(other[3]) - _number(follower[3]))
                for other in controlled_rows
                if str(other[1]) == road_id
            ]
            same_lane_distances = [
                abs(_number(other[3]) - _number(follower[3]))
                for other in controlled_rows
                if str(other[2]) == lane_id
            ]
            nearest_same_road_m = min(same_road_distances, default=None)
            nearest_same_lane_m = min(same_lane_distances, default=None)
            within_50m_same_road = nearest_same_road_m is not None and nearest_same_road_m <= 50.0
            within_50m_same_lane = nearest_same_lane_m is not None and nearest_same_lane_m <= 50.0
            pair = (follower_id, leader_id)
            event_start = pair not in previous_pairs
            unique_pairs.add(pair)
            exposure_counts["phase"][phase] += 1
            exposure_counts["pair_class"][pair_class] += 1
            exposure_counts["road_id"][road_id] += 1
            exposure_counts["lane_id"][lane_id] += 1
            for name, flag in (
                ("follower_controlled_wall_cav", follower_controlled),
                ("leader_controlled_wall_cav", leader_controlled),
                ("within_50m_controlled_cav_same_road", within_50m_same_road),
                ("within_50m_controlled_cav_same_lane", within_50m_same_lane),
                ("follower_lane_changed", follower_lane_changed),
            ):
                if flag:
                    exposure_flags[name] += 1
                    if event_start:
                        onset_flags[name] += 1
            if event_start:
                event_start_count += 1
            if len(event_samples) < event_sample_limit:
                gap_m = max(
                    0.0,
                    _number(leader[3]) - _number(follower[3]) - definition.vehicle_length_m,
                )
                event_samples.append(
                    {
                        "relative_step": current.relative_step,
                        "sim_time_s": _number(current.sim_time),
                        "phase": phase,
                        "applied_action": int(current.action),
                        "event_start": event_start,
                        "follower_id": follower_id,
                        "leader_id": leader_id,
                        "pair_class": pair_class,
                        "road_id": road_id,
                        "lane_id": lane_id,
                        "gap_m": gap_m,
                        "ttc_s": float(ttc_s),
                        "follower_speed_mps": _number(follower[4]),
                        "leader_speed_mps": _number(leader[4]),
                        "follower_acceleration_mps2": _number(follower[5]),
                        "leader_acceleration_mps2": _number(leader[5]),
                        "follower_controlled_wall_cav": follower_controlled,
                        "leader_controlled_wall_cav": leader_controlled,
                        "follower_lane_changed": follower_lane_changed,
                        "nearest_controlled_cav_same_road_m": nearest_same_road_m,
                        "nearest_controlled_cav_same_lane_m": nearest_same_lane_m,
                    }
                )
        previous_pairs = current_pair_ids

    return {
        "ttc_threshold_s": definition.ttc_threshold_s,
        "exposure_pair_steps": sum(exposure_counts["phase"].values()),
        "unique_pair_count": len(unique_pairs),
        "event_start_count": event_start_count,
        "event_sample_truncated": (sum(exposure_counts["phase"].values()) > len(event_samples)),
        "exposure_counts": {
            name: dict(sorted(counter.items())) for name, counter in exposure_counts.items()
        },
        "exposure_context_counts": dict(sorted(exposure_flags.items())),
        "event_start_context_counts": dict(sorted(onset_flags.items())),
        "events": event_samples,
    }


def summarize_branch(
    trace: ReplayTrace,
    *,
    decision_step: Optional[int] = None,
    outcome_steps: Optional[int] = None,
    definition: OutcomeDefinition = OutcomeDefinition(),
) -> Dict[str, Any]:
    """Summarize post-decision outcomes for one replayed treatment arm."""
    decision = trace.decision_step if decision_step is None else int(decision_step)
    final_step = decision + outcome_steps if outcome_steps is not None else math.inf
    snapshots = [
        snapshot for snapshot in trace.snapshots if decision <= snapshot.relative_step <= final_step
    ]
    if not snapshots or snapshots[0].relative_step != decision:
        raise ValueError(f"trace does not contain decision step {decision}")

    metrics = [step_metrics(snapshot, definition) for snapshot in snapshots]
    dt = _step_length(snapshots)
    sustained, onset_step = _sustained_congestion(metrics, definition)

    baseline_time_loss = {str(row[0]): _number(row[6]) for row in snapshots[0].vehicles}
    previous_time_loss = dict(baseline_time_loss)
    incremental_delay_s = 0.0
    discharged_ids: set[str] = set()
    hard_brake_vehicle_steps = 0
    low_ttc_exposure_steps = 0
    low_ttc_pairs: set[Tuple[str, str]] = set()
    min_ttc_s: Optional[float] = None
    collision_vehicle_ids: set[str] = set()
    collision_vehicle_steps = 0
    starting_teleport_vehicle_ids: set[str] = set()

    for previous, current, metric in zip(snapshots, snapshots[1:], metrics[1:]):
        current_map = _vehicle_map(current)
        for vehicle_id, row in current_map.items():
            cumulative = _number(row[6])
            prior = previous_time_loss.get(vehicle_id, 0.0)
            if math.isfinite(cumulative) and math.isfinite(prior):
                incremental_delay_s += max(0.0, cumulative - prior)
        previous_time_loss = {
            vehicle_id: _number(row[6]) for vehicle_id, row in current_map.items()
        }

        previous_map = _vehicle_map(previous)
        for vehicle_id, previous_row in previous_map.items():
            current_row = current_map.get(vehicle_id)
            if (
                str(previous_row[1]) == MERGE_EDGE
                and current_row is not None
                and str(current_row[1]) != MERGE_EDGE
            ):
                discharged_ids.add(vehicle_id)

        hard_brake_vehicle_steps += len(metric["hard_brake_vehicle_ids"])
        collision_vehicle_steps += len(metric["collision_vehicle_ids"])
        collision_vehicle_ids.update(metric["collision_vehicle_ids"])
        starting_teleport_vehicle_ids.update(metric["starting_teleport_vehicle_ids"])
        if metric["low_ttc_pairs"]:
            low_ttc_exposure_steps += 1
        for follower, leader, ttc in metric["low_ttc_pairs"]:
            low_ttc_pairs.add((follower, leader))
            min_ttc_s = ttc if min_ttc_s is None else min(min_ttc_s, ttc)

    queue_counts = [int(metric["queue_count"]) for metric in metrics]
    pre_speeds = [
        float(metric["pre_merge_mean_speed_mps"])
        for metric in metrics
        if metric["pre_merge_mean_speed_mps"] is not None
    ]
    merge_speeds = [
        float(metric["merge_mean_speed_mps"])
        for metric in metrics
        if metric["merge_mean_speed_mps"] is not None
    ]
    post_decision_snapshots = snapshots[1:]
    duty_steps = sum(snapshot.action != 0 for snapshot in post_decision_snapshots)
    active_counts = [int(metric["active_cav_count"]) for metric in metrics]
    active_lane_counts = [int(metric["active_cav_pre_merge_lane_count"]) for metric in metrics]

    return {
        "action": trace.post_action,
        "decision_step": decision,
        "observed_outcome_steps": max(0, len(snapshots) - 1),
        "terminated_early": trace.terminated_early,
        "step_length_s": dt,
        "sustained_congestion": sustained,
        "congestion_onset_step": onset_step,
        "queue_area_vehicle_s": sum(queue_counts[1:]) * dt,
        "max_queue_count": max(queue_counts, default=0),
        "mean_queue_count": sum(queue_counts) / len(queue_counts),
        "min_pre_merge_speed_mps": min(pre_speeds) if pre_speeds else None,
        "mean_pre_merge_speed_mps": _mean(pre_speeds),
        "mean_merge_speed_mps": _mean(merge_speeds),
        "incremental_delay_s": incremental_delay_s,
        "merge_discharge_count": len(discharged_ids),
        "hard_brake_vehicle_steps": hard_brake_vehicle_steps,
        "hard_brake_attribution": hard_brake_attribution(trace, snapshots, definition),
        "collision_vehicle_steps": collision_vehicle_steps,
        "collision_unique_vehicle_count": len(collision_vehicle_ids),
        "starting_teleport_unique_vehicle_count": len(starting_teleport_vehicle_ids),
        "low_ttc_exposure_steps": low_ttc_exposure_steps,
        "low_ttc_unique_pair_count": len(low_ttc_pairs),
        "low_ttc_attribution": low_ttc_attribution(trace, snapshots, definition),
        "min_ttc_s": min_ttc_s,
        "duty_steps": duty_steps,
        "duty_fraction": duty_steps / len(post_decision_snapshots)
        if post_decision_snapshots
        else 0.0,
        "mean_active_cav_count": _mean(active_counts),
        "max_active_cav_pre_merge_lane_count": max(active_lane_counts, default=0),
        "outcome_definition": definition.to_dict(),
    }
