"""Pure, leakage-safe feature extraction for the frozen IDM shadow screen.

The functions in this module do not launch SUMO and do not approximate either
controller.  They summarize (a) the causal damping prefix and (b) the exact
ten-second damping/wall twins produced by deterministic time-zero replay.

Vehicle identifiers are used transiently to link trajectories and wall
members.  They are never emitted in a model feature name or tensor.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from config.constants import (
    DEFAULT_VEH_LENGTH,
    DOWNSTREAM_EDGE,
    DOWNSTREAM_LANE_LENGTH,
    MERGE_EDGE,
    MERGE_DROPPED_LANE_LENGTH,
    PRE_MERGE_EDGE,
    PRE_MERGE_LANE_LENGTH,
    RAMP_EDGE,
    RAMP_LANE_LENGTH,
)
from counterfactual.idm_shadow_observation_v2_contract import (
    HISTORY_CAPTURE_S,
    HISTORY_SAMPLE_INTERVAL_S,
    PRIMARY_HISTORY_S,
    PROHIBITED_INPUT_TOKENS,
    SHADOW_HORIZON_S,
    SHADOW_REPORT_TIMES_S,
)
from counterfactual.twin_replay import ReplayTrace, StepSnapshot


PREDECISION_SCHEMA = "idm_shadow_predecision_record_v2/1"
SHADOW_SCHEMA = "idm_shadow_twin_features_v2/1"
CAPTURE_PHASE = "after_identical_damping_before_wall_override"
TTC_CENSOR_S = 120.0
HARD_BRAKE_MPS2 = -4.5
LOW_TTC_S = 3.0
MAX_ANONYMOUS_SLOTS = 256

_REGION_BY_EDGE = {
    PRE_MERGE_EDGE: 0,
    RAMP_EDGE: 1,
    MERGE_EDGE: 2,
    DOWNSTREAM_EDGE: 3,
}
_EDGE_LENGTH_M = {
    PRE_MERGE_EDGE: PRE_MERGE_LANE_LENGTH,
    RAMP_EDGE: RAMP_LANE_LENGTH,
    MERGE_EDGE: MERGE_DROPPED_LANE_LENGTH,
    DOWNSTREAM_EDGE: DOWNSTREAM_LANE_LENGTH,
}
_TENSOR_FEATURE_NAMES = (
    "present",
    "window_start_present",
    "region_code",
    "lane_ordinal",
    "position_fraction",
    "distance_to_region_end_fraction",
    "speed_mps",
    "acceleration_mps2",
    "front_gap_m",
    "closing_speed_mps",
    "is_controlled_cav",
    "route_index",
    "type_normal",
    "type_aggressive",
    "type_cautious",
    "type_other",
    "entry_event",
    "lane_change_event",
    "exit_event",
)


class ShadowObservationError(ValueError):
    """Raised when the frozen causal or exact-twin contract is violated."""


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center_x = (len(values) - 1) / 2.0
    center_y = _mean(values)
    denominator = sum((index - center_x) ** 2 for index in range(len(values)))
    if denominator <= 0.0:
        return 0.0
    return sum(
        (index - center_x) * (value - center_y)
        for index, value in enumerate(values)
    ) / denominator


def _quantile(values: Sequence[float], probability: float) -> tuple[float, float]:
    """Return a linearly interpolated quantile and its availability mask."""
    if len(values) < 2:
        return 0.0, 0.0
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 1.0


def _lane_ordinal(lane_id: Any) -> int:
    try:
        return int(str(lane_id).rsplit("_", 1)[-1])
    except ValueError:
        return 0


def _row_key(row: Sequence[Any]) -> tuple[int, int, float, str]:
    return (
        _REGION_BY_EDGE.get(str(row[1]), 4),
        _lane_ordinal(row[2]),
        _finite(row[3]),
        str(row[0]),
    )


def _snapshot_map(trace: ReplayTrace) -> dict[int, StepSnapshot]:
    snapshots: dict[int, StepSnapshot] = {}
    for snapshot in trace.snapshots:
        step = int(snapshot.relative_step)
        if step in snapshots:
            raise ShadowObservationError(f"duplicate relative step {step}")
        snapshots[step] = snapshot
    return snapshots


def _audit_map(trace: ReplayTrace) -> dict[int, Mapping[str, Any]]:
    audits: dict[int, Mapping[str, Any]] = {}
    for index, audit in enumerate(trace.controller_audits):
        step = int(audit.get("relative_step", index))
        if step in audits:
            raise ShadowObservationError(f"duplicate controller audit step {step}")
        audits[step] = audit
    return audits


def _canonical_snapshot(snapshot: StepSnapshot) -> str:
    """Order-invariant physical signature used only for replay integrity."""
    vehicles = [tuple(row) for row in sorted(snapshot.vehicles, key=_row_key)]
    payload = {
        "relative_step": int(snapshot.relative_step),
        "sim_time": snapshot.sim_time,
        "action": int(snapshot.action),
        "supervisor_observation": list(snapshot.supervisor_observation),
        "vehicles": vehicles,
        "active_cavs": sorted(str(value) for value in snapshot.active_cavs),
        "collisions": sorted(str(value) for value in snapshot.collision_vehicle_ids),
        "teleports": sorted(str(value) for value in snapshot.starting_teleport_vehicle_ids),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _exact_history(
    trace: ReplayTrace,
    decision_step: int,
    seconds: int,
) -> tuple[StepSnapshot, ...]:
    snapshots = _snapshot_map(trace)
    # A closed interval [-seconds, 0] contains seconds / interval + 1 samples.
    count = int(seconds / HISTORY_SAMPLE_INTERVAL_S) + 1
    steps = tuple(
        decision_step - (count - 1 - index) * HISTORY_SAMPLE_INTERVAL_S
        for index in range(count)
    )
    if any(step not in snapshots for step in steps):
        missing = [step for step in steps if step not in snapshots]
        raise ShadowObservationError(
            f"missing {seconds}s causal history steps: {missing[:5]}"
        )
    selected = tuple(snapshots[step] for step in steps)
    times = [_finite(snapshot.sim_time) for snapshot in selected]
    for before, after in zip(times, times[1:]):
        if abs((after - before) - HISTORY_SAMPLE_INTERVAL_S) > 1.0e-6:
            raise ShadowObservationError("causal history is not sampled at exact 1s intervals")
    if any(int(snapshot.relative_step) > decision_step for snapshot in selected):
        raise ShadowObservationError("future sample entered causal history")
    return selected


def _exact_report_snapshots(
    trace: ReplayTrace,
    decision_step: int,
) -> dict[int, StepSnapshot]:
    snapshots = _snapshot_map(trace)
    result: dict[int, StepSnapshot] = {}
    decision = snapshots.get(decision_step)
    if decision is None:
        raise ShadowObservationError("decision snapshot is missing")
    decision_time = _finite(decision.sim_time)
    for offset in SHADOW_REPORT_TIMES_S:
        step = decision_step + int(offset / HISTORY_SAMPLE_INTERVAL_S)
        snapshot = snapshots.get(step)
        if snapshot is None:
            raise ShadowObservationError(f"missing exact shadow report at +{offset}s")
        if abs((_finite(snapshot.sim_time) - decision_time) - offset) > 1.0e-6:
            raise ShadowObservationError(f"shadow report +{offset}s has wrong simulation time")
        result[offset] = snapshot
    if max(result) != SHADOW_HORIZON_S:
        raise ShadowObservationError("shadow horizon does not end at exactly 10s")
    return result


def _lane_pairs(snapshot: StepSnapshot) -> list[tuple[Sequence[Any], Sequence[Any], float, float]]:
    lanes: dict[tuple[str, str], list[Sequence[Any]]] = defaultdict(list)
    for row in snapshot.vehicles:
        lanes[(str(row[1]), str(row[2]))].append(row)
    pairs = []
    for rows in lanes.values():
        ordered = sorted(rows, key=lambda row: (_finite(row[3]), str(row[0])))
        for follower, leader in zip(ordered, ordered[1:]):
            gap = max(0.0, _finite(leader[3]) - _finite(follower[3]) - DEFAULT_VEH_LENGTH)
            closing = _finite(follower[4]) - _finite(leader[4])
            pairs.append((follower, leader, gap, closing))
    return pairs


def _ttc(gap: float, closing_speed: float) -> tuple[float, float]:
    if closing_speed <= 0.0:
        return TTC_CENSOR_S, 1.0
    return min(TTC_CENSOR_S, max(0.0, gap) / closing_speed), 0.0


def _entries(
    snapshots: Sequence[StepSnapshot],
    edge: str,
) -> int:
    count = 0
    for previous, current in zip(snapshots, snapshots[1:]):
        old = previous.vehicle_map()
        for vehicle_id, row in current.vehicle_map().items():
            if str(row[1]) != edge:
                continue
            prior = old.get(vehicle_id)
            if prior is None or str(prior[1]) != edge:
                count += 1
    return count


def _merge_discharge(snapshots: Sequence[StepSnapshot]) -> int:
    return _edge_transition_count(snapshots, MERGE_EDGE, DOWNSTREAM_EDGE)


def _edge_transition_count(
    snapshots: Sequence[StepSnapshot],
    source_edge: str,
    destination_edge: str,
) -> int:
    count = 0
    for previous, current in zip(snapshots, snapshots[1:]):
        old = previous.vehicle_map()
        for vehicle_id, row in current.vehicle_map().items():
            prior = old.get(vehicle_id)
            if (
                prior is not None
                and str(prior[1]) == source_edge
                and str(row[1]) == destination_edge
            ):
                count += 1
    return count


def _snapshot_metrics(snapshot: StepSnapshot) -> dict[str, float]:
    rows_by_edge: dict[str, list[Sequence[Any]]] = defaultdict(list)
    influence_rows = [row for row in snapshot.vehicles if str(row[1]) in _REGION_BY_EDGE]
    for row in influence_rows:
        rows_by_edge[str(row[1])].append(row)
    pairs = [
        row for row in _lane_pairs(snapshot) if str(row[0][1]) in _REGION_BY_EDGE
    ]
    gaps = [gap for _, _, gap, _ in pairs]
    ttcs = [_ttc(gap, closing)[0] for _, _, gap, closing in pairs]
    uncensored = [1.0 - _ttc(gap, closing)[1] for _, _, gap, closing in pairs]
    gap_q10, gap_q10_mask = _quantile(gaps, 0.10)
    gap_q50, gap_q50_mask = _quantile(gaps, 0.50)
    ttc_q10, ttc_q10_mask = _quantile(ttcs, 0.10)
    result: dict[str, float] = {
        "vehicle_count": float(len(influence_rows)),
        "hard_brake_count": float(
            sum(_finite(row[5]) <= HARD_BRAKE_MPS2 for row in influence_rows)
        ),
        "low_ttc_pair_count": float(sum(value < LOW_TTC_S for value in ttcs)),
        "minimum_gap_m": min(gaps) if gaps else 0.0,
        "minimum_gap_available": float(bool(gaps)),
        "minimum_ttc_s": min(ttcs) if ttcs else TTC_CENSOR_S,
        "minimum_ttc_available": float(bool(ttcs)),
        "ttc_uncensored_fraction": _mean(uncensored),
        "gap_q10_m": gap_q10,
        "gap_q10_available": gap_q10_mask,
        "gap_q50_m": gap_q50,
        "gap_q50_available": gap_q50_mask,
        "ttc_q10_s": ttc_q10,
        "ttc_q10_available": ttc_q10_mask,
        "queue_proxy_count": float(sum(_finite(row[4]) < 5.0 for row in influence_rows)),
    }
    for name, edge in (
        ("premerge", PRE_MERGE_EDGE),
        ("ramp", RAMP_EDGE),
        ("merge", MERGE_EDGE),
        ("downstream", DOWNSTREAM_EDGE),
    ):
        rows = rows_by_edge.get(edge, [])
        speeds = [_finite(row[4]) for row in rows]
        accelerations = [_finite(row[5]) for row in rows]
        result.update(
            {
                f"{name}_count": float(len(rows)),
                f"{name}_mean_speed_mps": _mean(speeds),
                f"{name}_speed_std_mps": _std(speeds),
                f"{name}_mean_acceleration_mps2": _mean(accelerations),
                f"{name}_speed_available": float(bool(rows)),
            }
        )
    return result


def _max_consecutive_braking(snapshot: StepSnapshot) -> int:
    by_lane: dict[tuple[str, str], list[Sequence[Any]]] = defaultdict(list)
    for row in snapshot.vehicles:
        if str(row[1]) not in _REGION_BY_EDGE:
            continue
        by_lane[(str(row[1]), str(row[2]))].append(row)
    longest = 0
    for rows in by_lane.values():
        run = 0
        for row in sorted(rows, key=lambda item: (_finite(item[3]), str(item[0]))):
            run = run + 1 if _finite(row[5]) < -0.5 else 0
            longest = max(longest, run)
    return longest


def _history_features(snapshots: Sequence[StepSnapshot], window_s: int) -> dict[str, float]:
    metrics = [_snapshot_metrics(snapshot) for snapshot in snapshots]
    prefix = f"history{window_s}"
    features: dict[str, float] = {}
    path_names = (
        "premerge_count",
        "premerge_mean_speed_mps",
        "premerge_speed_std_mps",
        "ramp_count",
        "ramp_mean_speed_mps",
        "merge_count",
        "merge_mean_speed_mps",
        "downstream_count",
        "downstream_mean_speed_mps",
        "hard_brake_count",
        "low_ttc_pair_count",
        "minimum_gap_m",
        "queue_proxy_count",
    )
    for name in path_names:
        path = [row[name] for row in metrics]
        features[f"{prefix}_{name}_mean"] = _mean(path)
        features[f"{prefix}_{name}_change"] = path[-1] - path[0]
        features[f"{prefix}_{name}_slope"] = _slope(path)

    front_positions: list[float] = []
    compression_flags: list[float] = []
    chain_lengths: list[float] = []
    for snapshot in snapshots:
        slow = [
            _finite(row[3])
            for row in snapshot.vehicles
            if str(row[1]) == PRE_MERGE_EDGE and _finite(row[4]) < 15.0
        ]
        front_positions.append(max(slow) if slow else PRE_MERGE_LANE_LENGTH)
        premerge_gaps = [
            gap
            for follower, _, gap, _ in _lane_pairs(snapshot)
            if str(follower[1]) == PRE_MERGE_EDGE
        ]
        compression_flags.append(float(bool(premerge_gaps) and _mean(premerge_gaps) < 20.0))
        chain_lengths.append(float(_max_consecutive_braking(snapshot)))

    first_pairs = _lane_pairs(snapshots[0])
    last_pairs = _lane_pairs(snapshots[-1])
    first_gaps = [gap for follower, _, gap, _ in first_pairs if str(follower[1]) == PRE_MERGE_EDGE]
    last_gaps = [gap for follower, _, gap, _ in last_pairs if str(follower[1]) == PRE_MERGE_EDGE]
    current_rows = [row for row in snapshots[-1].vehicles if str(row[1]) == PRE_MERGE_EDGE]
    current_speeds = [_finite(row[4]) for row in current_rows]
    current_gaps = [gap for follower, _, gap, _ in last_pairs if str(follower[1]) == PRE_MERGE_EDGE]
    gap_mean = _mean(current_gaps)
    gap_std = _std(current_gaps)
    successive = []
    for follower, leader, _, _ in last_pairs:
        if str(follower[1]) == PRE_MERGE_EDGE:
            successive.append((_finite(follower[4]), _finite(leader[4])))
    correlation = 0.0
    if len(successive) >= 2:
        left = [row[0] for row in successive]
        right = [row[1] for row in successive]
        left_center = _mean(left)
        right_center = _mean(right)
        denominator = math.sqrt(
            sum((value - left_center) ** 2 for value in left)
            * sum((value - right_center) ** 2 for value in right)
        )
        if denominator > 1.0e-12:
            correlation = sum(
                (a - left_center) * (b - right_center) for a, b in successive
            ) / denominator

    pre_entries = _entries(snapshots, PRE_MERGE_EDGE)
    ramp_entries = _entries(snapshots, RAMP_EDGE)
    discharge = _merge_discharge(snapshots)
    duration = max(1.0, _finite(snapshots[-1].sim_time) - _finite(snapshots[0].sim_time))

    amplitude_path = []
    gap_mean_path = []
    platoon_size_path = []
    platoon_length_path = []
    first_downstream_braking = None
    first_upstream_braking = None
    first_destabilizing_lane = None
    prior_lane_speeds: dict[int, float] = {}
    lane_change_count = 0
    yield_proxy_count = 0
    cumulative_compression = 0.0
    previous_gap_mean = None
    previous_vehicle_map: dict[str, Sequence[Any]] = {}
    feasible_band_counts = []
    feasible_band_entries = 0
    feasible_band_exits = 0
    previous_band_members: set[str] = set()
    for time_index, snapshot in enumerate(snapshots):
        pre_rows = [row for row in snapshot.vehicles if str(row[1]) == PRE_MERGE_EDGE]
        speeds = [_finite(row[4]) for row in pre_rows]
        amplitude_path.append(max(speeds) - min(speeds) if speeds else 0.0)
        pairs_now = [
            (follower, leader, gap, closing)
            for follower, leader, gap, closing in _lane_pairs(snapshot)
            if str(follower[1]) == PRE_MERGE_EDGE
        ]
        gaps_now = [row[2] for row in pairs_now]
        mean_gap_now = _mean(gaps_now)
        gap_mean_path.append(mean_gap_now)
        if previous_gap_mean is not None:
            cumulative_compression += max(0.0, previous_gap_mean - mean_gap_now)
        previous_gap_mean = mean_gap_now
        compact_pairs = [row for row in pairs_now if row[2] <= 30.0]
        platoon_vehicle_ids = {
            str(vehicle[0])
            for follower, leader, _, _ in compact_pairs
            for vehicle in (follower, leader)
        }
        platoon_size_path.append(float(len(platoon_vehicle_ids)))
        platoon_positions = [
            _finite(row[3]) for row in pre_rows if str(row[0]) in platoon_vehicle_ids
        ]
        platoon_length_path.append(
            max(platoon_positions) - min(platoon_positions) if len(platoon_positions) >= 2 else 0.0
        )
        for row in pre_rows:
            if _finite(row[5]) < -0.5:
                if _finite(row[3]) >= 0.5 * PRE_MERGE_LANE_LENGTH and first_downstream_braking is None:
                    first_downstream_braking = time_index
                if _finite(row[3]) < 0.5 * PRE_MERGE_LANE_LENGTH and first_upstream_braking is None:
                    first_upstream_braking = time_index
            if _finite(row[3]) >= PRE_MERGE_LANE_LENGTH - 200.0 and _finite(row[5]) < -1.0:
                yield_proxy_count += 1
        lane_speeds: dict[int, list[float]] = defaultdict(list)
        for row in pre_rows:
            lane_speeds[_lane_ordinal(row[2])].append(_finite(row[4]))
        current_lane_speeds = {lane: _mean(values) for lane, values in lane_speeds.items()}
        if first_destabilizing_lane is None and prior_lane_speeds:
            candidates = [
                (prior_lane_speeds[lane] - speed, lane)
                for lane, speed in current_lane_speeds.items()
                if lane in prior_lane_speeds and prior_lane_speeds[lane] - speed > 2.0
            ]
            if candidates:
                first_destabilizing_lane = max(candidates)[1]
        prior_lane_speeds = current_lane_speeds
        current_map = snapshot.vehicle_map()
        if previous_vehicle_map:
            lane_change_count += sum(
                vehicle_id in previous_vehicle_map
                and (
                    str(previous_vehicle_map[vehicle_id][1]),
                    str(previous_vehicle_map[vehicle_id][2]),
                )
                != (str(row[1]), str(row[2]))
                for vehicle_id, row in current_map.items()
            )
        active = set(str(value) for value in snapshot.active_cavs)
        band_members = {
            str(row[0])
            for row in pre_rows
            if str(row[0]) in active
            and 200.0 <= PRE_MERGE_LANE_LENGTH - _finite(row[3]) <= 400.0
        }
        if time_index > 0:
            feasible_band_entries += len(band_members - previous_band_members)
            feasible_band_exits += len(previous_band_members - band_members)
        feasible_band_counts.append(float(len(band_members)))
        previous_band_members = band_members
        previous_vehicle_map = current_map

    current_premerge_pairs = [
        (follower, leader, gap, closing)
        for follower, leader, gap, closing in last_pairs
        if str(follower[1]) == PRE_MERGE_EDGE
    ]
    current_premerge_ttcs = [_ttc(gap, closing)[0] for _, _, gap, closing in current_premerge_pairs]
    chain_gaps = [gap for _, _, gap, _ in current_premerge_pairs]
    headways = [
        gap / max(0.1, _finite(follower[4]))
        for follower, _, gap, _ in current_premerge_pairs
    ]
    headway_q10, headway_q10_mask = _quantile(headways, 0.10)
    headway_q50, headway_q50_mask = _quantile(headways, 0.50)
    headway_q90, headway_q90_mask = _quantile(headways, 0.90)
    first_headways = [
        gap / max(0.1, _finite(follower[4]))
        for follower, _, gap, _ in first_pairs
        if str(follower[1]) == PRE_MERGE_EDGE
    ]
    acceleration_by_lane: dict[str, list[Sequence[Any]]] = defaultdict(list)
    for row in current_rows:
        acceleration_by_lane[str(row[2])].append(row)
    amplification_values = []
    for rows in acceleration_by_lane.values():
        values = [
            abs(_finite(row[5]))
            for row in sorted(rows, key=lambda item: (_finite(item[3]), str(item[0])))
        ]
        if len(values) >= 2 and values[-1] > 1.0e-6:
            amplification_values.append(max(values) / values[-1])
    braking_lag_available = first_downstream_braking is not None and first_upstream_braking is not None
    braking_lag = (
        float(first_upstream_braking - first_downstream_braking)
        if braking_lag_available
        else 0.0
    )

    ramp_rows = [row for row in snapshots[-1].vehicles if str(row[1]) == RAMP_EDGE]
    mainline_rows = [row for row in snapshots[-1].vehicles if str(row[1]) == PRE_MERGE_EDGE]
    arrival_differences = []
    relative_speeds = []
    for ramp_row in ramp_rows:
        ramp_time = max(0.0, RAMP_LANE_LENGTH - _finite(ramp_row[3])) / max(
            0.1, _finite(ramp_row[4])
        )
        for mainline_row in mainline_rows:
            main_time = max(0.0, PRE_MERGE_LANE_LENGTH - _finite(mainline_row[3])) / max(
                0.1, _finite(mainline_row[4])
            )
            arrival_differences.append(abs(ramp_time - main_time))
            relative_speeds.append(_finite(ramp_row[4]) - _finite(mainline_row[4]))
    merge_interaction_available = bool(arrival_differences)
    features.update(
        {
            f"{prefix}_wave_front_speed_mps": _slope(front_positions),
            f"{prefix}_wave_amplitude_mps": (
                max(current_speeds) - min(current_speeds) if current_speeds else 0.0
            ),
            f"{prefix}_wave_amplitude_change_mps": amplitude_path[-1] - amplitude_path[0],
            f"{prefix}_wave_amplitude_growth_flag": float(amplitude_path[-1] > amplitude_path[0]),
            f"{prefix}_wave_recovery_flag": float(
                amplitude_path[-1] < amplitude_path[0]
                and metrics[-1]["premerge_mean_speed_mps"] > metrics[0]["premerge_mean_speed_mps"]
            ),
            f"{prefix}_wave_compression_persistence": _mean(compression_flags),
            f"{prefix}_braking_propagation_lag_s": braking_lag,
            f"{prefix}_braking_propagation_lag_available": float(braking_lag_available),
            f"{prefix}_follower_consecutive_braking_max": max(chain_lengths, default=0.0),
            f"{prefix}_follower_chain_minimum_gap_m": min(chain_gaps, default=0.0),
            f"{prefix}_follower_chain_gap_available": float(bool(chain_gaps)),
            f"{prefix}_follower_chain_minimum_ttc_s": min(current_premerge_ttcs, default=TTC_CENSOR_S),
            f"{prefix}_follower_chain_ttc_available": float(bool(current_premerge_ttcs)),
            f"{prefix}_follower_string_amplification": max(amplification_values, default=0.0),
            f"{prefix}_follower_string_amplification_available": float(bool(amplification_values)),
            f"{prefix}_platoon_gap_mean_m": gap_mean,
            f"{prefix}_platoon_gap_cv": gap_std / max(1.0e-6, gap_mean),
            f"{prefix}_platoon_gap_change_m": _mean(last_gaps) - _mean(first_gaps),
            f"{prefix}_headway_variance_s2": _std(headways) ** 2,
            f"{prefix}_headway_q10_s": headway_q10,
            f"{prefix}_headway_q10_available": headway_q10_mask,
            f"{prefix}_headway_q50_s": headway_q50,
            f"{prefix}_headway_q50_available": headway_q50_mask,
            f"{prefix}_headway_q90_s": headway_q90,
            f"{prefix}_headway_q90_available": headway_q90_mask,
            f"{prefix}_headway_mean_change_s": _mean(headways) - _mean(first_headways),
            f"{prefix}_platoon_size_change": platoon_size_path[-1] - platoon_size_path[0],
            f"{prefix}_platoon_length_change_m": platoon_length_path[-1] - platoon_length_path[0],
            f"{prefix}_platoon_density_change_per_m": (
                platoon_size_path[-1] / max(1.0, platoon_length_path[-1])
                - platoon_size_path[0] / max(1.0, platoon_length_path[0])
            ),
            f"{prefix}_platoon_speed_cv": _std(current_speeds) / max(1.0e-6, _mean(current_speeds)),
            f"{prefix}_successive_speed_correlation": correlation,
            f"{prefix}_merge_arrival_gap_s": min(arrival_differences, default=0.0),
            f"{prefix}_merge_arrival_gap_available": float(merge_interaction_available),
            f"{prefix}_ramp_mainline_relative_speed_mps": (
                relative_speeds[arrival_differences.index(min(arrival_differences))]
                if arrival_differences
                else 0.0
            ),
            f"{prefix}_lane_change_event_count": float(lane_change_count),
            f"{prefix}_forced_yield_proxy_count": float(yield_proxy_count),
            f"{prefix}_cumulative_mainline_compression_m": cumulative_compression,
            f"{prefix}_first_destabilizing_lane_ordinal": float(first_destabilizing_lane or 0),
            f"{prefix}_first_destabilizing_lane_available": float(first_destabilizing_lane is not None),
            f"{prefix}_feasible_position_cav_count_change": feasible_band_counts[-1] - feasible_band_counts[0],
            f"{prefix}_feasible_position_cav_entry_count": float(feasible_band_entries),
            f"{prefix}_feasible_position_cav_exit_count": float(feasible_band_exits),
            f"{prefix}_mainline_arrival_rate_vph": pre_entries * 3600.0 / duration,
            f"{prefix}_ramp_arrival_rate_vph": ramp_entries * 3600.0 / duration,
            f"{prefix}_merge_discharge_rate_vph": discharge * 3600.0 / duration,
            f"{prefix}_excess_demand_vehicle_count": float(pre_entries + ramp_entries - discharge),
            f"{prefix}_downstream_storage_fraction": min(
                1.0,
                metrics[-1]["downstream_count"] * DEFAULT_VEH_LENGTH / (2.0 * DOWNSTREAM_LANE_LENGTH),
            ),
            f"{prefix}_interlane_speed_imbalance_mps": _interlane_speed_imbalance(snapshots[-1]),
            f"{prefix}_controlled_cav_coverage_fraction": (
                len(snapshots[-1].active_cavs) / max(1, len(snapshots[-1].vehicles))
            ),
        }
    )
    return features


def _interlane_speed_imbalance(snapshot: StepSnapshot) -> float:
    speeds: dict[str, list[float]] = defaultdict(list)
    for row in snapshot.vehicles:
        if str(row[1]) == PRE_MERGE_EDGE:
            speeds[str(row[2])].append(_finite(row[4]))
    lane_means = [_mean(values) for values in speeds.values() if values]
    return max(lane_means) - min(lane_means) if len(lane_means) >= 2 else 0.0


def _anonymous_history_tensor(snapshots: Sequence[StepSnapshot]) -> dict[str, Any]:
    slot_by_vehicle: dict[str, int] = {}
    for snapshot in snapshots:
        new_rows = [
            row
            for row in snapshot.vehicles
            if str(row[1]) in _REGION_BY_EDGE and str(row[0]) not in slot_by_vehicle
        ]
        for row in sorted(new_rows, key=_row_key):
            if len(slot_by_vehicle) >= MAX_ANONYMOUS_SLOTS:
                raise ShadowObservationError(
                    "influence-region trajectory exceeds anonymous slot capacity"
                )
            slot_by_vehicle[str(row[0])] = len(slot_by_vehicle)

    values: list[list[list[float]]] = []
    masks: list[list[float]] = []
    event_masks: list[list[float]] = []
    previous: dict[str, Sequence[Any]] = {}
    capacity = len(slot_by_vehicle)
    for frame_index, snapshot in enumerate(snapshots):
        current = {
            str(row[0]): row
            for row in snapshot.vehicles
            if str(row[1]) in _REGION_BY_EDGE
        }
        frame = [[0.0] * len(_TENSOR_FEATURE_NAMES) for _ in range(capacity)]
        present = [0.0] * capacity
        event_valid = [0.0] * capacity
        lane_rows: dict[tuple[str, str], list[Sequence[Any]]] = defaultdict(list)
        for row in snapshot.vehicles:
            lane_rows[(str(row[1]), str(row[2]))].append(row)
        front: dict[str, tuple[float, float]] = {}
        for rows in lane_rows.values():
            ordered = sorted(rows, key=lambda row: (_finite(row[3]), str(row[0])))
            for follower, leader in zip(ordered, ordered[1:]):
                front[str(follower[0])] = (
                    max(0.0, _finite(leader[3]) - _finite(follower[3]) - DEFAULT_VEH_LENGTH),
                    _finite(follower[4]) - _finite(leader[4]),
                )
        active = set(str(value) for value in snapshot.active_cavs)
        for vehicle_id, row in current.items():
            slot = slot_by_vehicle.get(vehicle_id)
            if slot is None:
                continue
            edge = str(row[1])
            length = _EDGE_LENGTH_M.get(edge, PRE_MERGE_LANE_LENGTH)
            gap, closing = front.get(vehicle_id, (0.0, 0.0))
            prior = previous.get(vehicle_id)
            entry = float(frame_index > 0 and prior is None)
            lane_change = float(
                prior is not None
                and (str(prior[1]), str(prior[2])) != (str(row[1]), str(row[2]))
            )
            vehicle_type = str(row[8]).lower()
            normal = float("normal" in vehicle_type or "cav" in vehicle_type)
            aggressive = float("aggress" in vehicle_type)
            cautious = float("caut" in vehicle_type)
            other = float(not (normal or aggressive or cautious))
            frame[slot] = [
                1.0,
                float(frame_index == 0),
                float(_REGION_BY_EDGE.get(edge, 4)),
                float(_lane_ordinal(row[2])),
                _finite(row[3]) / max(1.0, length),
                max(0.0, length - _finite(row[3])) / max(1.0, length),
                _finite(row[4]),
                _finite(row[5]),
                gap,
                closing,
                float(vehicle_id in active),
                _finite(row[7]),
                normal,
                aggressive,
                cautious,
                other,
                entry,
                lane_change,
                0.0,
            ]
            present[slot] = 1.0
            event_valid[slot] = float(frame_index > 0)
        for vehicle_id in set(previous) - set(current):
            slot = slot_by_vehicle.get(vehicle_id)
            if slot is not None:
                frame[slot][-1] = 1.0
                event_valid[slot] = 1.0
        values.append(frame)
        masks.append(present)
        event_masks.append(event_valid)
        previous = current
    channel_names = list(_TENSOR_FEATURE_NAMES)
    return {
        "relative_times_s": [
            _finite(snapshot.sim_time) - _finite(snapshots[-1].sim_time)
            for snapshot in snapshots
        ],
        "slot_feature_names": channel_names,
        "channel_names": channel_names,
        "values": values,
        "tensor_values": values,
        "presence_mask": masks,
        "tensor_mask": masks,
        "event_mask": event_masks,
        "anonymous_slot_count": len(slot_by_vehicle),
        "slot_capacity": capacity,
        "maximum_slot_capacity": MAX_ANONYMOUS_SLOTS,
        "truncated_vehicle_count": 0,
        "sequence_available": True,
    }


def _safe_feasibility_features(row: Mapping[str, Any]) -> dict[str, float]:
    """Whitelist predecision scan fields; every unlisted field is ignored."""
    formation = row.get("formation") if isinstance(row.get("formation"), Mapping) else {}
    coverage = row.get("lane_coverage") if isinstance(row.get("lane_coverage"), Mapping) else {}
    projected = row.get("projected_dose") if isinstance(row.get("projected_dose"), Mapping) else {}
    safety = row.get("activation_safety") if isinstance(row.get("activation_safety"), Mapping) else {}
    return {
        "preflight_feasible": float(bool(formation.get("feasible", False))),
        "preflight_member_count": float(len(formation.get("controlled_cav_ids", ()) or ())),
        "preflight_occupied_lane_count": float(len(formation.get("occupied_lane_ids", ()) or ())),
        "preflight_required_lane_count": float(len(formation.get("required_lane_ids", ()) or ())),
        "preflight_longitudinal_span_m": _finite(formation.get("longitudinal_span_m")),
        "preflight_predicted_max_span_m": _finite(formation.get("predicted_max_span_m")),
        "preflight_predicted_terminal_span_m": _finite(
            formation.get("predicted_terminal_span_m")
        ),
        "preflight_lane_coverage_fraction": _finite(coverage.get("coverage_fraction")),
        "preflight_full_lane_coverage": float(
            bool(coverage.get("full_occupied_lane_coverage", False))
        ),
        "preflight_projected_override_fraction": _finite(
            projected.get("projected_duration_fraction")
        ),
        "preflight_safety_admissible": float(bool(safety.get("safe", False))),
        "preflight_min_acceleration_mps2": _finite(safety.get("min_acceleration_mps2")),
        "preflight_min_front_gap_m": _finite(safety.get("min_front_gap_m")),
        "preflight_min_rear_gap_m": _finite(safety.get("min_rear_gap_m")),
        "preflight_min_front_ttc_s": _finite(safety.get("min_front_ttc_s"), TTC_CENSOR_S),
        "preflight_min_rear_ttc_s": _finite(safety.get("min_rear_ttc_s"), TTC_CENSOR_S),
        "preflight_admitted": float(bool(row.get("admitted", False))),
    }


def _masked_scalar(field: Any) -> tuple[float, bool]:
    if not isinstance(field, Mapping) or not bool(field.get("available", False)):
        return 0.0, False
    value = field.get("value")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    return (result, True) if math.isfinite(result) else (0.0, False)


def _masked_text(field: Any) -> tuple[str, bool]:
    if not isinstance(field, Mapping) or not bool(field.get("available", False)):
        return "", False
    value = field.get("value")
    return (str(value), value is not None)


def _summary_with_mask(prefix: str, values: Sequence[float]) -> dict[str, float]:
    q10, q10_mask = _quantile(values, 0.10)
    return {
        f"{prefix}_mean": _mean(values),
        f"{prefix}_minimum": min(values, default=0.0),
        f"{prefix}_q10": q10,
        f"{prefix}_q10_available": q10_mask,
        f"{prefix}_available_fraction": float(bool(values)),
    }


def _live_predecision_features(record: Mapping[str, Any]) -> dict[str, float]:
    """Anonymize the enriched live record through an explicit physical whitelist."""
    if not record:
        return {"live_record_available": 0.0}
    if record.get("capture_phase") != CAPTURE_PHASE:
        raise ShadowObservationError("live record has the wrong capture phase")
    if record.get("internal_identifiers_are_linkage_metadata_only") is not True:
        raise ShadowObservationError("live record does not declare linkage-only identifiers")
    raw_vehicles = record.get("vehicles")
    if not isinstance(raw_vehicles, list):
        raise ShadowObservationError("live record vehicles are missing")

    influence: list[tuple[Mapping[str, Any], str, int]] = []
    by_internal: dict[str, Mapping[str, Any]] = {}
    for row in raw_vehicles:
        if not isinstance(row, Mapping):
            continue
        road, road_ok = _masked_text(row.get("road"))
        lane_index_value, lane_ok = _masked_scalar(row.get("lane_index"))
        if not road_ok or road not in _REGION_BY_EDGE:
            continue
        lane_index = int(lane_index_value) if lane_ok else 0
        influence.append((row, road, lane_index))
        identity = row.get("internal_vehicle_id")
        if identity is not None:
            by_internal[str(identity)] = row

    features: dict[str, float] = {
        "live_record_available": 1.0,
        "live_influence_vehicle_count": float(len(influence)),
    }
    region_names = {
        PRE_MERGE_EDGE: "live_premerge",
        RAMP_EDGE: "live_ramp",
        MERGE_EDGE: "live_merge",
        DOWNSTREAM_EDGE: "live_downstream",
    }
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row, road, lane_index in influence:
        groups[region_names[road]].append(row)
        if road == PRE_MERGE_EDGE and 0 <= lane_index <= 2:
            groups[f"live_premerge_lane{lane_index}"].append(row)
    for group_name in (*region_names.values(), "live_premerge_lane0", "live_premerge_lane1", "live_premerge_lane2"):
        rows = groups.get(group_name, [])
        features[f"{group_name}_vehicle_count"] = float(len(rows))
        for link_name in ("leader", "follower"):
            gaps = []
            relative_speeds = []
            for row in rows:
                link = row.get(link_name)
                if not isinstance(link, Mapping) or not bool(link.get("available", False)):
                    continue
                gap = link.get("net_gap_m")
                relative = link.get("relative_speed_mps")
                try:
                    gap_value = float(gap)
                    if math.isfinite(gap_value):
                        gaps.append(gap_value)
                except (TypeError, ValueError):
                    pass
                try:
                    relative_value = float(relative)
                    if math.isfinite(relative_value):
                        relative_speeds.append(relative_value)
                except (TypeError, ValueError):
                    pass
            features.update(_summary_with_mask(f"{group_name}_{link_name}_gap_m", gaps))
            features.update(
                _summary_with_mask(
                    f"{group_name}_{link_name}_relative_speed_mps", relative_speeds
                )
            )

    parameter_names = (
        "tau_s",
        "min_gap_m",
        "accel_mps2",
        "decel_mps2",
        "emergency_decel_mps2",
        "apparent_decel_mps2",
        "speed_factor",
        "allowed_speed_mps",
        "max_speed_mps",
        "action_step_length_s",
    )
    for name in parameter_names:
        values = []
        for row, _, _ in influence:
            parameters = row.get("parameters")
            value, available = _masked_scalar(
                parameters.get(name) if isinstance(parameters, Mapping) else None
            )
            if available:
                values.append(value)
        features.update(
            {
                f"live_idm_{name}_mean": _mean(values),
                f"live_idm_{name}_std": _std(values),
                f"live_idm_{name}_range": max(values) - min(values) if values else 0.0,
                f"live_idm_{name}_available_fraction": len(values) / max(1, len(influence)),
            }
        )

    damping = record.get("damping") if isinstance(record.get("damping"), Mapping) else {}
    controlled = record.get("controlled_cav_internal_ids") or ()
    commands = []
    diagnostics = damping.get("projection_diagnostics")
    if isinstance(diagnostics, Mapping) and isinstance(diagnostics.get("commands"), list):
        commands = [row for row in diagnostics["commands"] if isinstance(row, Mapping)]
    authority = []
    saturation = []
    projection_survival = []
    for command in commands:
        nominal = command.get("nominal_acceleration_mps2")
        desired = command.get("desired_acceleration_mps2")
        projected = command.get("projected")
        projected_acceleration = (
            projected.get("acceleration_mps2") if isinstance(projected, Mapping) else None
        )
        if nominal is not None and projected_acceleration is not None:
            authority.append(abs(_finite(projected_acceleration) - _finite(nominal)))
        if desired is not None and projected_acceleration is not None:
            saturation.append(
                float(abs(_finite(projected_acceleration) - _finite(desired)) > 1.0e-6)
            )
        if isinstance(projected, Mapping):
            projection_survival.append(float(not bool(projected.get("constraint_conflict", False))))
    features.update(
        {
            "live_damping_controlled_count": float(len(controlled)),
            "live_damping_command_count": float(len(commands)),
            "live_damping_authority_mean_mps2": _mean(authority),
            "live_damping_authority_maximum_mps2": max(authority, default=0.0),
            "live_damping_authority_available": float(bool(authority)),
            "live_damping_projection_survival_fraction": _mean(projection_survival),
            "live_damping_command_saturation_fraction": _mean(saturation),
        }
    )

    proposed = record.get("proposed_wall") if isinstance(record.get("proposed_wall"), Mapping) else {}
    members = tuple(str(value) for value in (proposed.get("member_internal_ids") or ()))
    member_distances = []
    for identity in members:
        row = by_internal.get(identity)
        if row is None:
            continue
        value, available = _masked_scalar(row.get("distance_to_merge_m"))
        if available:
            member_distances.append(value)
    reserve_distances = []
    reserve_offsets = []
    reserves = proposed.get("ranked_same_lane_reserves")
    if isinstance(reserves, Mapping):
        for lane_rows in reserves.values():
            for row in lane_rows or ():
                if not isinstance(row, Mapping):
                    continue
                try:
                    reserve_distances.append(float(row["distance_to_merge_m"]))
                    reserve_offsets.append(float(row["distance_from_primary_reference_m"]))
                except (KeyError, TypeError, ValueError):
                    continue
    release_boundary = _finite(proposed.get("release_boundary_m"))
    features.update(
        {
            "live_wall_member_count": float(len(members)),
            "live_wall_member_distance_mean_m": _mean(member_distances),
            "live_wall_member_distance_span_m": (
                max(member_distances) - min(member_distances) if member_distances else 0.0
            ),
            "live_wall_member_distance_available": float(bool(member_distances)),
            "live_wall_reserve_count": float(len(reserve_distances)),
            "live_wall_reserve_distance_mean_m": _mean(reserve_distances),
            "live_wall_reserve_reference_offset_mean_m": _mean(reserve_offsets),
            "live_wall_release_boundary_m": release_boundary,
            "live_wall_member_boundary_margin_minimum_m": (
                min((value - release_boundary for value in member_distances), default=0.0)
            ),
        }
    )
    return features


def validate_model_feature_names(features: Mapping[str, Any]) -> None:
    """Fail closed if a learned-input name contains a prohibited token."""
    for name, value in features.items():
        lowered = str(name).lower()
        matched = [token for token in PROHIBITED_INPUT_TOKENS if token in lowered]
        if matched:
            raise ShadowObservationError(
                f"prohibited model feature name {name!r} contains {matched[0]!r}"
            )
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ShadowObservationError(f"model feature {name!r} is not a finite scalar")


def model_feature_vector(record: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Return a deterministic name-sorted tensor without identifiers."""
    features = record.get("model_features")
    if not isinstance(features, Mapping):
        raise ShadowObservationError("record does not contain model_features")
    validate_model_feature_names(features)
    names = tuple(sorted(str(name) for name in features))
    return names, tuple(float(features[name]) for name in names)


def capture_predecision_record(
    control_trace: ReplayTrace,
    *,
    decision_step: int,
    feasibility_row: Mapping[str, Any],
    live_predecision_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture 20/60-second causal history from the damping-only prefix."""
    history20 = _exact_history(control_trace, decision_step, PRIMARY_HISTORY_S)
    history60 = _exact_history(control_trace, decision_step, HISTORY_CAPTURE_S)
    physical20 = _history_features(history20, PRIMARY_HISTORY_S)
    physical60 = _history_features(history60, HISTORY_CAPTURE_S)
    preflight = _safe_feasibility_features(feasibility_row)
    live_physical = _live_predecision_features(live_predecision_record or {})
    # The primary tensor uses 20 seconds.  Sixty-second summaries remain a
    # separately named sensitivity block and are never silently pooled here.
    model_features = {**physical20, **preflight, **live_physical}
    validate_model_feature_names(model_features)
    return {
        "schema": PREDECISION_SCHEMA,
        "capture_phase": CAPTURE_PHASE,
        "decision_step": int(decision_step),
        "feature_blocks": {
            "sectional_transitional_instability": {
                key: value
                for key, value in physical20.items()
                if "wave_" in key or "interlane_" in key or "brak" in key
            },
            "merge_pressure_breakdown_state": {
                key: value
                for key, value in {**physical20, **preflight}.items()
                if any(token in key for token in ("arrival", "discharge", "excess", "storage", "preflight"))
            },
            "trajectory_physical_20s": physical20,
            "trajectory_physical_60s": physical60,
            "live_predecision_physical": live_physical,
        },
        "model_features": model_features,
        "sensitivity_model_features_60s": physical60,
        "history_20s": _anonymous_history_tensor(history20),
        "history_60s": _anonymous_history_tensor(history60),
        "integrity": {
            "causal_cutoff_passed": True,
            "sample_interval_s": float(HISTORY_SAMPLE_INTERVAL_S),
            "primary_sample_count": len(history20),
            "captured_sample_count": len(history60),
            "latest_relative_step": int(history60[-1].relative_step),
            "internal_identity_exported": False,
        },
    }


def _member_ids(feasibility_row: Mapping[str, Any], audit: Mapping[str, Any]) -> tuple[str, ...]:
    values = audit.get("wall_member_ids") or audit.get("wall_override_ids")
    if not values:
        formation = feasibility_row.get("formation")
        if isinstance(formation, Mapping):
            values = formation.get("controlled_cav_ids")
    return tuple(sorted(str(value) for value in (values or ())))


def _member_safety(snapshot: StepSnapshot, member_ids: Iterable[str]) -> dict[str, float]:
    member_set = set(member_ids)
    pairs = _lane_pairs(snapshot)
    front_by_follower = {str(follower[0]): (gap, closing) for follower, _, gap, closing in pairs}
    rear_by_leader = {str(leader[0]): (follower, gap, closing) for follower, leader, gap, closing in pairs}
    front_gap: list[float] = []
    rear_gap: list[float] = []
    front_ttc: list[float] = []
    rear_ttc: list[float] = []
    follower_acceleration: list[float] = []
    for member_id in member_set:
        if member_id in front_by_follower:
            gap, closing = front_by_follower[member_id]
            front_gap.append(gap)
            front_ttc.append(_ttc(gap, closing)[0])
        if member_id in rear_by_leader:
            follower, gap, closing = rear_by_leader[member_id]
            rear_gap.append(gap)
            # For the rear pair, ``closing`` already means follower minus wall.
            rear_ttc.append(_ttc(gap, closing)[0])
            follower_acceleration.append(_finite(follower[5]))
    return {
        "member_count": float(len(member_set)),
        "minimum_front_gap_m": min(front_gap) if front_gap else 0.0,
        "front_gap_available": float(bool(front_gap)),
        "minimum_rear_gap_m": min(rear_gap) if rear_gap else 0.0,
        "rear_gap_available": float(bool(rear_gap)),
        "minimum_front_ttc_s": min(front_ttc) if front_ttc else TTC_CENSOR_S,
        "front_ttc_available": float(bool(front_ttc)),
        "minimum_rear_ttc_s": min(rear_ttc) if rear_ttc else TTC_CENSOR_S,
        "rear_ttc_available": float(bool(rear_ttc)),
        "minimum_follower_acceleration_mps2": (
            min(follower_acceleration) if follower_acceleration else 0.0
        ),
        "follower_acceleration_available": float(bool(follower_acceleration)),
        "affected_follower_count": float(len(follower_acceleration)),
        "required_follower_deceleration_mps2": max(
            0.0, -(min(follower_acceleration) if follower_acceleration else 0.0)
        ),
    }


def _premerge_lane_report(snapshot: StepSnapshot) -> dict[str, float]:
    pairs = _lane_pairs(snapshot)
    features: dict[str, float] = {}
    for lane_ordinal in range(3):
        lane_id = f"{PRE_MERGE_EDGE}_{lane_ordinal}"
        lane_pairs = [row for row in pairs if str(row[0][2]) == lane_id]
        gaps = [row[2] for row in lane_pairs]
        ttcs = [_ttc(row[2], row[3])[0] for row in lane_pairs]
        speeds = [
            _finite(row[4])
            for row in snapshot.vehicles
            if str(row[1]) == PRE_MERGE_EDGE and str(row[2]) == lane_id
        ]
        gap_q10, gap_mask = _quantile(gaps, 0.10)
        ttc_q10, ttc_mask = _quantile(ttcs, 0.10)
        features.update(
            {
                f"lane{lane_ordinal}_gap_q10_m": gap_q10,
                f"lane{lane_ordinal}_gap_q10_available": gap_mask,
                f"lane{lane_ordinal}_ttc_q10_s": ttc_q10,
                f"lane{lane_ordinal}_ttc_q10_available": ttc_mask,
                f"lane{lane_ordinal}_mean_speed_mps": _mean(speeds),
                f"lane{lane_ordinal}_speed_available": float(bool(speeds)),
            }
        )
    return features


def _member_boundary_margin(
    snapshot: StepSnapshot,
    member_ids: Iterable[str],
    release_boundary_m: float,
) -> tuple[float, float]:
    vehicle_map = snapshot.vehicle_map()
    margins = []
    for member_id in member_ids:
        row = vehicle_map.get(str(member_id))
        if row is not None and str(row[1]) == PRE_MERGE_EDGE:
            margins.append(
                PRE_MERGE_LANE_LENGTH - _finite(row[3]) - release_boundary_m
            )
    return (min(margins), 1.0) if margins else (0.0, 0.0)


def _numeric_projection_audit(audit: Mapping[str, Any]) -> dict[str, float]:
    coverage = audit.get("lane_coverage") if isinstance(audit.get("lane_coverage"), Mapping) else {}
    deltas = audit.get("control_deltas") if isinstance(audit.get("control_deltas"), Mapping) else {}
    authorities: list[float] = []
    projection_survival: list[float] = []
    saturation: list[float] = []
    for detail in deltas.values():
        if not isinstance(detail, Mapping):
            continue
        speed = detail.get("pre_override_speed_mps")
        projected = detail.get("member_projected_speed_target_mps")
        raw = detail.get("member_raw_speed_target_mps")
        if speed is not None and projected is not None:
            authorities.append(max(0.0, _finite(speed) - _finite(projected)))
        if raw is not None and projected is not None:
            difference = abs(_finite(raw) - _finite(projected))
            projection_survival.append(float(difference <= 1.0e-6))
            saturation.append(float(difference > 1.0e-6))
    reserves = audit.get("reserve_ids_by_lane")
    reserve_count = 0
    if isinstance(reserves, Mapping):
        reserve_count = sum(len(values or ()) for values in reserves.values())
    handoffs = audit.get("handoff_counts_by_lane")
    handoff_count = 0.0
    if isinstance(handoffs, Mapping):
        handoff_count = sum(_finite(value) for value in handoffs.values())
    state = str(audit.get("treatment_state", "")).upper()
    return {
        "wall_active": float(bool(audit.get("wall_active", False))),
        "assembly_state": float(state == "ASSEMBLING"),
        "hold_state": float(state == "HOLDING"),
        "release_state": float(state in {"TAPERING", "RELEASING", "RELEASED", "COOLDOWN"}),
        "assembly_elapsed_s": _finite(audit.get("assembly_elapsed_s")),
        "member_span_m": _finite(audit.get("current_member_span_m")),
        "member_speed_spread_mps": _finite(audit.get("member_speed_spread_mps")),
        "lane_coverage_fraction": _finite(coverage.get("coverage_fraction")),
        "available_authority_mean_mps": _mean(authorities),
        "available_authority_min_mps": min(authorities) if authorities else 0.0,
        "authority_available": float(bool(authorities)),
        "projection_survival_fraction": _mean(projection_survival),
        "command_saturation_fraction": _mean(saturation),
        "same_lane_reserve_count": float(reserve_count),
        "handoff_count": handoff_count,
    }


def _audit_integrity(
    control: ReplayTrace,
    wall: ReplayTrace,
    decision_step: int,
) -> dict[str, Any]:
    control_audits = _audit_map(control)
    wall_audits = _audit_map(wall)
    out_of_band_ok = True
    target_only_ok = True
    checked_nonmembers = 0
    checked_members = 0
    for step in range(decision_step + 1, decision_step + SHADOW_HORIZON_S + 1):
        audit = wall_audits.get(step, {})
        wall_ids = set(str(value) for value in (audit.get("wall_override_ids") or ()))
        member_ids = set(str(value) for value in (audit.get("wall_member_ids") or ()))
        target_only_ok = target_only_ok and wall_ids == member_ids
        checked_members += len(wall_ids)
        base = audit.get("base_actions") if isinstance(audit.get("base_actions"), Mapping) else {}
        applied = audit.get("applied_env_actions") if isinstance(audit.get("applied_env_actions"), Mapping) else {}
        nonmembers = set(str(value) for value in (audit.get("out_of_band_damping_ids") or ()))
        for vehicle_id in nonmembers:
            if vehicle_id in base and vehicle_id in applied:
                checked_nonmembers += 1
                out_of_band_ok = out_of_band_ok and _json_equal(base[vehicle_id], applied[vehicle_id])
        control_audit = control_audits.get(step, {})
        target_only_ok = target_only_ok and not bool(control_audit.get("wall_override_ids", ()))
    return {
        "target_only_wall_override": bool(target_only_ok),
        "out_of_band_damping_identity": bool(out_of_band_ok),
        "checked_wall_member_commands": int(checked_members),
        "checked_out_of_band_commands": int(checked_nonmembers),
    }


def _shadow_lifecycle_features(
    audits: Mapping[int, Mapping[str, Any]],
    decision_step: int,
) -> dict[str, float]:
    """Summarize every audit in the exact prospective ten-second shadow."""
    rows = [
        audits.get(step, {})
        for step in range(decision_step + 1, decision_step + SHADOW_HORIZON_S + 1)
    ]
    active = [float(bool(row.get("wall_active", False))) for row in rows]
    states = [str(row.get("treatment_state", "")).upper() for row in rows]
    hold_offsets = [
        index + 1 for index, state in enumerate(states) if state == "HOLDING"
    ]
    spans = [
        _finite(row.get("current_member_span_m"))
        for row in rows
        if row.get("current_member_span_m") is not None
    ]
    coverage = []
    projection_survival = []
    saturation = []
    reserve_counts = []
    handoff_counts = []
    member_changes = 0
    speed_convergence_offsets = []
    safety_stop = False
    for index, row in enumerate(rows, start=1):
        numeric = _numeric_projection_audit(row)
        if bool(row.get("wall_active", False)):
            coverage.append(numeric["lane_coverage_fraction"])
        if numeric["authority_available"]:
            projection_survival.append(numeric["projection_survival_fraction"])
            saturation.append(numeric["command_saturation_fraction"])
        reserve_counts.append(numeric["same_lane_reserve_count"])
        handoff_counts.append(numeric["handoff_count"])
        member_changes += len(row.get("member_changes") or ())
        tolerance = _finite(row.get("compact_assembly_speed_tolerance_mps"), 2.0)
        spread = row.get("member_speed_spread_mps")
        if row.get("wall_member_ids") and spread is not None and _finite(spread) <= tolerance:
            speed_convergence_offsets.append(index)
        safety_stop = safety_stop or bool(row.get("abort_reason"))
    release_states = {"TAPERING", "RELEASING", "RELEASED", "COOLDOWN"}
    return {
        "shadow_10s_projected_override_fraction": _mean(active),
        "shadow_10s_any_assembly_success": float(bool(hold_offsets)),
        "shadow_10s_first_hold_time_s": float(hold_offsets[0] if hold_offsets else 0),
        "shadow_10s_first_hold_time_available": float(bool(hold_offsets)),
        "shadow_10s_speed_convergence_time_s": float(
            speed_convergence_offsets[0] if speed_convergence_offsets else 0
        ),
        "shadow_10s_speed_convergence_time_available": float(
            bool(speed_convergence_offsets)
        ),
        "shadow_10s_maximum_member_span_m": max(spans, default=0.0),
        "shadow_10s_minimum_lane_coverage_fraction": min(coverage, default=0.0),
        "shadow_10s_projection_survival_fraction": _mean(projection_survival),
        "shadow_10s_command_saturation_fraction": _mean(saturation),
        "shadow_10s_member_change_count": float(member_changes),
        "shadow_10s_handoff_count": max(handoff_counts, default=0.0),
        "shadow_10s_minimum_same_lane_reserve_count": min(reserve_counts, default=0.0),
        "shadow_10s_lifecycle_release_flag": float(any(state in release_states for state in states)),
        "shadow_10s_lifecycle_safety_stop_flag": float(safety_stop),
    }


def _json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(right, sort_keys=True, default=str)


def shadow_observation_features(
    control_trace: ReplayTrace,
    treatment_trace: ReplayTrace,
    *,
    decision_step: int,
    feasibility_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract exact 1/2/5/10-second twin features from stored replay traces."""
    control_map = _snapshot_map(control_trace)
    wall_map = _snapshot_map(treatment_trace)
    for step in range(0, decision_step + 1):
        if step not in control_map or step not in wall_map:
            raise ShadowObservationError("twin is missing part of the common prefix")
        if _canonical_snapshot(control_map[step]) != _canonical_snapshot(wall_map[step]):
            raise ShadowObservationError(f"twin prefix mismatch at relative step {step}")
    control_reports = _exact_report_snapshots(control_trace, decision_step)
    wall_reports = _exact_report_snapshots(treatment_trace, decision_step)
    control_audits = _audit_map(control_trace)
    wall_audits = _audit_map(treatment_trace)

    decision_control = control_map[decision_step]
    control_path = [decision_control]
    wall_path = [wall_map[decision_step]]
    model_features: dict[str, float] = {}
    initial_members = _member_ids(feasibility_row, wall_audits.get(decision_step, {}))
    initial_member_safety = _member_safety(wall_map[decision_step], initial_members)
    release_boundary_m = _finite(
        wall_audits.get(decision_step + 1, {}).get("compact_release_boundary_m")
    )
    for offset in SHADOW_REPORT_TIMES_S:
        control_snapshot = control_reports[offset]
        wall_snapshot = wall_reports[offset]
        control_path.append(control_snapshot)
        wall_path.append(wall_snapshot)
        control_metrics = _snapshot_metrics(control_snapshot)
        wall_metrics = _snapshot_metrics(wall_snapshot)
        control_lane_metrics = _premerge_lane_report(control_snapshot)
        wall_lane_metrics = _premerge_lane_report(wall_snapshot)
        wall_audit = wall_audits.get(decision_step + offset, {})
        member_ids = _member_ids(feasibility_row, wall_audit)
        member_safety = _member_safety(wall_snapshot, member_ids)
        if (
            initial_member_safety["rear_gap_available"]
            and member_safety["rear_gap_available"]
        ):
            member_safety["follower_compression_m"] = max(
                0.0,
                initial_member_safety["minimum_rear_gap_m"]
                - member_safety["minimum_rear_gap_m"],
            )
            member_safety["follower_compression_available"] = 1.0
        else:
            member_safety["follower_compression_m"] = 0.0
            member_safety["follower_compression_available"] = 0.0
        boundary_margin, boundary_mask = _member_boundary_margin(
            wall_snapshot, member_ids, release_boundary_m
        )
        member_safety["release_boundary_margin_m"] = boundary_margin
        member_safety["release_boundary_margin_available"] = boundary_mask
        audit_metrics = _numeric_projection_audit(wall_audit)
        for name in (
            "premerge_count",
            "premerge_mean_speed_mps",
            "ramp_count",
            "ramp_mean_speed_mps",
            "merge_count",
            "merge_mean_speed_mps",
            "downstream_count",
            "downstream_mean_speed_mps",
            "hard_brake_count",
            "low_ttc_pair_count",
            "minimum_gap_m",
            "minimum_ttc_s",
            "queue_proxy_count",
        ):
            model_features[f"shadow_t{offset}_damping_{name}"] = control_metrics[name]
            model_features[f"shadow_t{offset}_wall_{name}"] = wall_metrics[name]
            model_features[f"shadow_t{offset}_wall_minus_damping_{name}"] = (
                wall_metrics[name] - control_metrics[name]
            )
        for name, value in member_safety.items():
            model_features[f"shadow_t{offset}_{name}"] = value
        for name in sorted(control_lane_metrics):
            model_features[f"shadow_t{offset}_damping_{name}"] = control_lane_metrics[name]
            model_features[f"shadow_t{offset}_wall_{name}"] = wall_lane_metrics[name]
            if not name.endswith("available"):
                model_features[f"shadow_t{offset}_wall_minus_damping_{name}"] = (
                    wall_lane_metrics[name] - control_lane_metrics[name]
                )
        for name, value in audit_metrics.items():
            model_features[f"shadow_t{offset}_{name}"] = value

    # Cumulative physical response uses exact observed crossings in each twin.
    control_full = [
        control_map[step]
        for step in range(decision_step, decision_step + SHADOW_HORIZON_S + 1)
    ]
    wall_full = [
        wall_map[step]
        for step in range(decision_step, decision_step + SHADOW_HORIZON_S + 1)
    ]
    cumulative = {
        "mainline_entries": (_entries(control_full, PRE_MERGE_EDGE), _entries(wall_full, PRE_MERGE_EDGE)),
        "ramp_entries": (_entries(control_full, RAMP_EDGE), _entries(wall_full, RAMP_EDGE)),
        "mainline_merge_inflow": (
            _edge_transition_count(control_full, PRE_MERGE_EDGE, MERGE_EDGE),
            _edge_transition_count(wall_full, PRE_MERGE_EDGE, MERGE_EDGE),
        ),
        "ramp_merge_inflow": (
            _edge_transition_count(control_full, RAMP_EDGE, MERGE_EDGE),
            _edge_transition_count(wall_full, RAMP_EDGE, MERGE_EDGE),
        ),
        "merge_discharge": (_merge_discharge(control_full), _merge_discharge(wall_full)),
    }
    for name, (damping_value, wall_value) in cumulative.items():
        model_features[f"shadow_10s_damping_{name}"] = float(damping_value)
        model_features[f"shadow_10s_wall_{name}"] = float(wall_value)
        model_features[f"shadow_10s_wall_minus_damping_{name}"] = float(
            wall_value - damping_value
        )
    model_features["shadow_10s_projected_inflow_reduction"] = float(
        (
            cumulative["mainline_merge_inflow"][0]
            + cumulative["ramp_merge_inflow"][0]
        )
        - (
            cumulative["mainline_merge_inflow"][1]
            + cumulative["ramp_merge_inflow"][1]
        )
    )
    model_features["shadow_10s_projected_discharge_change"] = float(
        cumulative["merge_discharge"][1] - cumulative["merge_discharge"][0]
    )
    model_features.update(_shadow_lifecycle_features(wall_audits, decision_step))
    validate_model_feature_names(model_features)
    audit_integrity = _audit_integrity(control_trace, treatment_trace, decision_step)
    return {
        "schema": SHADOW_SCHEMA,
        "decision_step": int(decision_step),
        "report_times_s": list(SHADOW_REPORT_TIMES_S),
        "feature_blocks": {"proposed_wall_shadow_10s": dict(model_features)},
        "model_features": model_features,
        "integrity": {
            "prefix_identity": True,
            "exact_report_times": True,
            "exact_horizon_s": SHADOW_HORIZON_S,
            "internal_identity_exported": False,
            **audit_integrity,
        },
    }


__all__ = [
    "CAPTURE_PHASE",
    "MAX_ANONYMOUS_SLOTS",
    "PREDECISION_SCHEMA",
    "SHADOW_SCHEMA",
    "ShadowObservationError",
    "capture_predecision_record",
    "model_feature_vector",
    "shadow_observation_features",
    "validate_model_feature_names",
]
