"""Paired off/soft/strong data generation for a treatment-effect gate."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from config.constants import (
    DEFAULT_VEH_LENGTH,
    DOWNSTREAM_EDGE,
    DOWNSTREAM_EDGE_NUM_LANES,
    DOWNSTREAM_LANE_LENGTH,
    PRE_MERGE_EDGE,
    PRE_MERGE_LANE_LENGTH,
)
from counterfactual.outcomes import OutcomeDefinition, summarize_branch, step_metrics
from counterfactual.twin_replay import ReplayTrace, StepSnapshot, run_replay


OBSERVATION_NAMES = (
    "health",
    "health_delta_1",
    "merge_density_ema_scaled",
    "merge_mean_speed_scaled",
    "pre_merge_queue_scaled",
    "queue_spillback",
    "wall_band_cav_count_scaled",
    "pre_merge_vehicle_count_scaled",
    "current_rung_scaled",
    "engagement_duty",
    "episode_progress",
    "merge_inflow_scaled",
    "health_lag_5",
    "health_lag_10",
    "health_lag_20",
    "health_lag_30",
)
TRAFFIC_PATTERN_LAGS = (5, 10, 20, 30, 60)
LOCAL_DISTANCE_BINS_M = (
    ("0_100m", 0.0, 100.0),
    ("100_200m", 100.0, 200.0),
    ("200_400m", 200.0, 400.0),
    ("400_800m", 400.0, 800.0),
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _linear_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center_x = (len(values) - 1) / 2.0
    center_y = _mean(values)
    denominator = sum((index - center_x) ** 2 for index in range(len(values)))
    if denominator <= 0.0:
        return 0.0
    return (
        sum((index - center_x) * (value - center_y) for index, value in enumerate(values))
        / denominator
    )


def _lag1_autocorrelation(values: Sequence[float]) -> float:
    """Finite lag-1 autocorrelation; zero denotes insufficient/constant data."""

    if len(values) < 3:
        return 0.0
    left = [float(value) for value in values[:-1]]
    right = [float(value) for value in values[1:]]
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator > 1.0e-12 else 0.0


def _same_lane_mean_gap(snapshot: StepSnapshot, road_id: str) -> Optional[float]:
    gaps: list[float] = []
    by_lane: Dict[str, list[Any]] = {}
    for row in snapshot.vehicles:
        if str(row[1]) == road_id:
            by_lane.setdefault(str(row[2]), []).append(row)
    for rows in by_lane.values():
        ordered = sorted(rows, key=lambda row: _finite(row[3]))
        gaps.extend(
            max(0.0, _finite(leader[3]) - _finite(follower[3]) - DEFAULT_VEH_LENGTH)
            for follower, leader in zip(ordered, ordered[1:])
        )
    return _mean(gaps) if gaps else None


def _edge_safety_features(
    snapshot: StepSnapshot,
    road_id: str,
    definition: OutcomeDefinition,
) -> Dict[str, float]:
    rows = [row for row in snapshot.vehicles if str(row[1]) == road_id]
    hard_brakes = sum(
        _finite(row[5]) <= definition.hard_brake_threshold_mps2 for row in rows
    )
    low_ttc = 0
    by_lane: Dict[str, list[Any]] = {}
    for row in rows:
        by_lane.setdefault(str(row[2]), []).append(row)
    for lane_rows in by_lane.values():
        ordered = sorted(lane_rows, key=lambda row: _finite(row[3]))
        for follower, leader in zip(ordered, ordered[1:]):
            closing = _finite(follower[4]) - _finite(leader[4])
            if closing <= 0.0:
                continue
            gap = max(0.0, _finite(leader[3]) - _finite(follower[3]) - DEFAULT_VEH_LENGTH)
            ttc = gap / closing
            low_ttc += int(ttc < definition.ttc_threshold_s)
    return {
        "hard_brake_count": float(hard_brakes),
        "low_ttc_pair_count": float(low_ttc),
    }


def _causal_window(
    snapshots: Sequence[StepSnapshot], now_s: float, window_s: float
) -> list[StepSnapshot]:
    return [
        snapshot
        for snapshot in snapshots
        if now_s - window_s < _finite(snapshot.sim_time) <= now_s
    ]


def _edge_entry_count(snapshots: Sequence[StepSnapshot], road_id: str) -> int:
    count = 0
    for previous, current in zip(snapshots, snapshots[1:]):
        old = previous.vehicle_map()
        for vehicle_id, row in current.vehicle_map().items():
            if str(row[1]) != road_id:
                continue
            prior = old.get(vehicle_id)
            if prior is None or str(prior[1]) != road_id:
                count += 1
    return count


def observation_v2_features(
    trace: ReplayTrace,
    *,
    decision_step: int,
    definition: OutcomeDefinition,
    dose_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Frozen leakage-safe observation-v2 aggregates through the decision.

    Windows are in simulation seconds, not array indices. Every input is read
    from the common control prefix at or before the declared decision.
    """

    if decision_step < 0 or decision_step >= len(trace.snapshots):
        raise ValueError(f"decision step {decision_step} is outside the trace")
    prefix = list(trace.snapshots[: decision_step + 1])
    current = prefix[-1]
    now_s = _finite(current.sim_time)
    current_rows = [row for row in current.vehicles if str(row[1]) == DOWNSTREAM_EDGE]
    current_speeds = [_finite(row[4]) for row in current_rows]
    current_safety = _edge_safety_features(current, DOWNSTREAM_EDGE, definition)
    lane_km = DOWNSTREAM_EDGE_NUM_LANES * DOWNSTREAM_LANE_LENGTH / 1000.0
    road_space_m = DOWNSTREAM_EDGE_NUM_LANES * DOWNSTREAM_LANE_LENGTH

    windows = {seconds: _causal_window(prefix, now_s, seconds) for seconds in (20.0, 60.0)}
    downstream_paths: Dict[float, Dict[str, list[float]]] = {}
    for seconds, snapshots in windows.items():
        density_path: list[float] = []
        speed_path: list[float] = []
        variance_path: list[float] = []
        hard_brake_path: list[float] = []
        low_ttc_path: list[float] = []
        for snapshot in snapshots:
            rows = [row for row in snapshot.vehicles if str(row[1]) == DOWNSTREAM_EDGE]
            speeds = [_finite(row[4]) for row in rows]
            safety = _edge_safety_features(snapshot, DOWNSTREAM_EDGE, definition)
            density_path.append(len(rows) / lane_km if lane_km else 0.0)
            speed_path.append(_mean(speeds))
            variance_path.append(_std(speeds) ** 2)
            hard_brake_path.append(safety["hard_brake_count"])
            low_ttc_path.append(safety["low_ttc_pair_count"])
        downstream_paths[seconds] = {
            "density": density_path,
            "speed": speed_path,
            "variance": variance_path,
            "hard_brake": hard_brake_path,
            "low_ttc": low_ttc_path,
        }

    premerge_paths: Dict[float, Dict[str, list[float]]] = {}
    for seconds, snapshots in windows.items():
        speed_path: list[float] = []
        gap_path: list[float] = []
        for snapshot in snapshots:
            speeds = [
                _finite(row[4])
                for row in snapshot.vehicles
                if str(row[1]) == PRE_MERGE_EDGE
            ]
            if speeds:
                speed_path.append(_mean(speeds))
            gap = _same_lane_mean_gap(snapshot, PRE_MERGE_EDGE)
            if gap is not None:
                gap_path.append(float(gap))
        premerge_paths[seconds] = {"speed": speed_path, "gap": gap_path}

    discharge_rates = {
        seconds: (
            _edge_entry_count(snapshots, DOWNSTREAM_EDGE) * 3600.0 / seconds
            if snapshots
            else 0.0
        )
        for seconds, snapshots in windows.items()
    }
    upstream_flow_60 = (
        _edge_entry_count(windows[60.0], PRE_MERGE_EDGE) * 3600.0 / 60.0
        if windows[60.0]
        else 0.0
    )
    context = dict(dose_context or {})
    proposed_wall_speed = _finite(context.get("proposed_wall_speed_mps", 0.0))
    release_boundary = _finite(context.get("release_boundary_m", 0.0))
    selected_ids = tuple(str(value) for value in context.get("selected_member_ids", ()))
    current_map = current.vehicle_map()
    selected_distances = [
        PRE_MERGE_LANE_LENGTH - _finite(current_map[vehicle_id][3])
        for vehicle_id in selected_ids
        if vehicle_id in current_map and str(current_map[vehicle_id][1]) == PRE_MERGE_EDGE
    ]
    release_margin = (
        min(selected_distances) - release_boundary if selected_distances else 0.0
    )
    p20 = downstream_paths[20.0]
    p60 = downstream_paths[60.0]
    pre20 = premerge_paths[20.0]
    pre60 = premerge_paths[60.0]
    return {
        "obs_v2_postmerge_vehicle_count": float(len(current_rows)),
        "obs_v2_postmerge_density_veh_per_km_lane": len(current_rows) / lane_km,
        "obs_v2_postmerge_occupancy_proxy": (
            len(current_rows) * DEFAULT_VEH_LENGTH / road_space_m if road_space_m else 0.0
        ),
        "obs_v2_postmerge_mean_speed_mps": _mean(current_speeds),
        "obs_v2_postmerge_harmonic_speed_mps": (
            len(current_speeds) / sum(1.0 / max(speed, 0.1) for speed in current_speeds)
            if current_speeds
            else 0.0
        ),
        "obs_v2_postmerge_speed_variance_m2ps2": _std(current_speeds) ** 2,
        "obs_v2_postmerge_hard_brake_count": current_safety["hard_brake_count"],
        "obs_v2_postmerge_low_ttc_pair_count": current_safety["low_ttc_pair_count"],
        "obs_v2_postmerge_discharge_rate_20s_vph": discharge_rates[20.0],
        "obs_v2_postmerge_discharge_rate_60s_vph": discharge_rates[60.0],
        "obs_v2_postmerge_density_slope_60s": _linear_slope(p60["density"]),
        "obs_v2_postmerge_speed_slope_60s_mps_per_sample": _linear_slope(p60["speed"]),
        "obs_v2_postmerge_speed_variance_slope_60s": _linear_slope(p60["variance"]),
        "obs_v2_postmerge_hard_brake_active_fraction_60s": (
            sum(value > 0.0 for value in p60["hard_brake"]) / len(p60["hard_brake"])
            if p60["hard_brake"]
            else 0.0
        ),
        "obs_v2_postmerge_low_ttc_active_fraction_60s": (
            sum(value > 0.0 for value in p60["low_ttc"]) / len(p60["low_ttc"])
            if p60["low_ttc"]
            else 0.0
        ),
        "obs_v2_postmerge_discharge_short_minus_long_vph": (
            discharge_rates[20.0] - discharge_rates[60.0]
        ),
        "obs_v2_premerge_speed_variance_20s": _std(pre20["speed"]) ** 2,
        "obs_v2_premerge_speed_lag1_autocorr_20s": _lag1_autocorrelation(pre20["speed"]),
        "obs_v2_premerge_speed_lag1_autocorr_60s": _lag1_autocorrelation(pre60["speed"]),
        "obs_v2_premerge_gap_available_20s": float(bool(pre20["gap"])),
        "obs_v2_premerge_gap_variance_20s_m2": _std(pre20["gap"]) ** 2,
        "obs_v2_premerge_gap_lag1_autocorr_20s": _lag1_autocorrelation(pre20["gap"]),
        "obs_v2_premerge_gap_lag1_autocorr_60s": _lag1_autocorrelation(pre60["gap"]),
        "obs_v2_dose_proposed_wall_speed_mps": proposed_wall_speed,
        "obs_v2_dose_release_boundary_margin_m": release_margin,
        "obs_v2_dose_upstream_flow_60s_vph": upstream_flow_60,
        "obs_v2_dose_flow_speed_product": upstream_flow_60 * proposed_wall_speed,
    }


def _local_group_features(
    prefix: str,
    rows: Sequence[Any],
    *,
    active_cavs: set[str],
    definition: OutcomeDefinition,
) -> Dict[str, Any]:
    speeds = [_finite(row[4]) for row in rows]
    accelerations = [_finite(row[5]) for row in rows]
    cav_count = sum(str(row[0]) in active_cavs for row in rows)
    return {
        f"{prefix}_vehicle_count": len(rows),
        f"{prefix}_cav_count": cav_count,
        f"{prefix}_cav_fraction": cav_count / len(rows) if rows else 0.0,
        f"{prefix}_queue_count": sum(
            speed < definition.queue_speed_threshold_mps for speed in speeds
        ),
        f"{prefix}_mean_speed_mps": _mean(speeds),
        f"{prefix}_min_speed_mps": min(speeds, default=0.0),
        f"{prefix}_speed_std_mps": _std(speeds),
        f"{prefix}_mean_acceleration_mps2": _mean(accelerations),
        f"{prefix}_min_acceleration_mps2": min(accelerations, default=0.0),
        f"{prefix}_hard_brake_count": sum(
            acceleration <= definition.hard_brake_threshold_mps2 for acceleration in accelerations
        ),
    }


def local_traffic_features(snapshot: StepSnapshot, definition: OutcomeDefinition) -> Dict[str, Any]:
    """Describe the local shock and feasible soft-wall formation at time t."""
    active = set(snapshot.active_cavs)
    pre_merge_rows = [row for row in snapshot.vehicles if str(row[1]) == PRE_MERGE_EDGE]
    result: Dict[str, Any] = {}
    for lane_index in range(3):
        lane_id = f"{PRE_MERGE_EDGE}_{lane_index}"
        lane_rows = [row for row in pre_merge_rows if str(row[2]) == lane_id]
        result.update(
            _local_group_features(
                f"local_lane_{lane_index}",
                lane_rows,
                active_cavs=active,
                definition=definition,
            )
        )

    bin_rows: Dict[str, list[Any]] = {}
    for name, lower, upper in LOCAL_DISTANCE_BINS_M:
        rows = [
            row
            for row in pre_merge_rows
            if lower <= PRE_MERGE_LANE_LENGTH - _finite(row[3]) < upper
        ]
        bin_rows[name] = rows
        result.update(
            _local_group_features(
                f"local_bin_{name}",
                rows,
                active_cavs=active,
                definition=definition,
            )
        )

    near_speeds = [_finite(row[4]) for row in bin_rows["0_100m"] + bin_rows["100_200m"]]
    upstream_speeds = [_finite(row[4]) for row in bin_rows["200_400m"]]
    result["local_downstream_minus_upstream_speed_mps"] = (
        _mean(near_speeds) - _mean(upstream_speeds) if near_speeds and upstream_speeds else 0.0
    )

    by_lane: Dict[str, list[Any]] = {}
    for row in pre_merge_rows:
        by_lane.setdefault(str(row[2]), []).append(row)
    soft_cavs = {
        str(row[0])
        for row in pre_merge_rows
        if str(row[0]) in active and 200.0 <= PRE_MERGE_LANE_LENGTH - _finite(row[3]) <= 400.0
    }
    leader_gaps: list[float] = []
    follower_gaps: list[float] = []
    leader_cav_count = 0
    follower_cav_count = 0
    leader_count = 0
    follower_count = 0
    soft_cav_speeds: list[float] = []
    neighboring_hdv_speeds: list[float] = []
    for lane_rows in by_lane.values():
        ordered = sorted(lane_rows, key=lambda row: _finite(row[3]))
        for index, row in enumerate(ordered):
            if str(row[0]) not in soft_cavs:
                continue
            soft_cav_speeds.append(_finite(row[4]))
            if index + 1 < len(ordered):
                leader = ordered[index + 1]
                leader_count += 1
                leader_gaps.append(
                    max(
                        0.0,
                        _finite(leader[3]) - _finite(row[3]) - DEFAULT_VEH_LENGTH,
                    )
                )
                if str(leader[0]) in active:
                    leader_cav_count += 1
                else:
                    neighboring_hdv_speeds.append(_finite(leader[4]))
            if index > 0:
                follower = ordered[index - 1]
                follower_count += 1
                follower_gaps.append(
                    max(
                        0.0,
                        _finite(row[3]) - _finite(follower[3]) - DEFAULT_VEH_LENGTH,
                    )
                )
                if str(follower[0]) in active:
                    follower_cav_count += 1
                else:
                    neighboring_hdv_speeds.append(_finite(follower[4]))

    result.update(
        {
            "formation_soft_cav_count": len(soft_cavs),
            "formation_leader_gap_min_m": min(leader_gaps, default=0.0),
            "formation_leader_gap_mean_m": _mean(leader_gaps),
            "formation_follower_gap_min_m": min(follower_gaps, default=0.0),
            "formation_follower_gap_mean_m": _mean(follower_gaps),
            "formation_leader_cav_fraction": (
                leader_cav_count / leader_count if leader_count else 0.0
            ),
            "formation_follower_cav_fraction": (
                follower_cav_count / follower_count if follower_count else 0.0
            ),
            "formation_soft_cav_speed_std_mps": _std(soft_cav_speeds),
            "formation_cav_minus_neighbor_hdv_speed_mps": (
                _mean(soft_cav_speeds) - _mean(neighboring_hdv_speeds)
                if soft_cav_speeds and neighboring_hdv_speeds
                else 0.0
            ),
        }
    )
    metric = step_metrics(snapshot, definition)
    low_ttc_values = [float(row[2]) for row in metric["low_ttc_pairs"]]
    result.update(
        {
            "decision_hard_brake_vehicle_count": len(metric["hard_brake_vehicle_ids"]),
            "decision_low_ttc_pair_count": len(metric["low_ttc_pairs"]),
            "decision_min_low_ttc_s": min(low_ttc_values, default=99.0),
        }
    )
    return result


def placement_features(snapshot: StepSnapshot) -> Dict[str, Any]:
    """Describe realized CAV penetration and wall-forming geometry at time t."""
    active = set(snapshot.active_cavs)
    pre_merge_rows = [row for row in snapshot.vehicles if str(row[1]) == PRE_MERGE_EDGE]
    cav_rows = [row for row in pre_merge_rows if str(row[0]) in active]
    distances = [PRE_MERGE_LANE_LENGTH - _finite(row[3]) for row in cav_rows]
    lane_rows: Dict[str, list[Any]] = {}
    for row in cav_rows:
        lane_rows.setdefault(str(row[2]), []).append(row)

    same_lane_spacings = []
    for rows in lane_rows.values():
        positions = sorted(_finite(row[3]) for row in rows)
        same_lane_spacings.extend(right - left for left, right in zip(positions, positions[1:]))

    result: Dict[str, Any] = {
        "placement_network_cav_fraction": (
            len(active) / snapshot.vehicle_count if snapshot.vehicle_count else 0.0
        ),
        "placement_pre_merge_cav_count": len(cav_rows),
        "placement_pre_merge_cav_fraction": (
            len(cav_rows) / len(pre_merge_rows) if pre_merge_rows else 0.0
        ),
        "placement_pre_merge_cav_lane_count": len(lane_rows),
        "placement_cav_distance_to_merge_min_m": min(distances, default=None),
        "placement_cav_distance_to_merge_max_m": max(distances, default=None),
        "placement_cav_distance_to_merge_mean_m": (
            sum(distances) / len(distances) if distances else None
        ),
        "placement_longitudinal_span_m": (max(distances) - min(distances) if distances else 0.0),
        "placement_max_same_lane_cav_spacing_m": max(same_lane_spacings, default=0.0),
    }
    for lane_index in range(3):
        lane_id = f"{PRE_MERGE_EDGE}_{lane_index}"
        lane_distances = [
            PRE_MERGE_LANE_LENGTH - _finite(row[3]) for row in lane_rows.get(lane_id, [])
        ]
        result[f"placement_lane_{lane_index}_cav_count"] = len(lane_distances)
        result[f"placement_lane_{lane_index}_nearest_cav_to_merge_m"] = min(
            lane_distances, default=None
        )

    for name, lower, upper in (
        ("soft_band", 200.0, 400.0),
        ("strong_band", 200.0, 800.0),
    ):
        rows = [
            row for row in cav_rows if lower <= PRE_MERGE_LANE_LENGTH - _finite(row[3]) <= upper
        ]
        lane_count = len({str(row[2]) for row in rows})
        result[f"placement_{name}_cav_count"] = len(rows)
        result[f"placement_{name}_lane_count"] = lane_count
        result[f"placement_{name}_lane_coverage_fraction"] = lane_count / 3.0
    return result


def find_decision_step(
    trace: ReplayTrace,
    *,
    min_step: int = 10,
    health_threshold: float = 0.45,
    require_nonpositive_trend: bool = True,
    min_inband_scaled: float = 0.0,
) -> Optional[int]:
    """Find a pre-outcome warning landmark using only information at time t."""
    for snapshot in trace.snapshots:
        if snapshot.relative_step < min_step:
            continue
        observation = snapshot.supervisor_observation
        if len(observation) < len(OBSERVATION_NAMES):
            raise ValueError("supervisor observation is shorter than 16 dimensions")
        health = _finite(observation[0], default=1.0)
        trend = _finite(observation[1])
        inband = _finite(observation[6])
        if health > health_threshold:
            continue
        if require_nonpositive_trend and trend > 0.0:
            continue
        if inband < min_inband_scaled:
            continue
        return snapshot.relative_step
    return None


def features_from_snapshot(snapshot: StepSnapshot, definition: OutcomeDefinition) -> Dict[str, Any]:
    """Build causal-at-decision features, including health-path shape."""
    observation = [_finite(value) for value in snapshot.supervisor_observation]
    if len(observation) != len(OBSERVATION_NAMES):
        raise ValueError(
            f"expected {len(OBSERVATION_NAMES)} supervisor features, got {len(observation)}"
        )
    features: Dict[str, Any] = dict(zip(OBSERVATION_NAMES, observation))
    chronological_health = [
        features["health_lag_30"],
        features["health_lag_20"],
        features["health_lag_10"],
        features["health_lag_5"],
        features["health"],
    ]
    changes = [right - left for left, right in zip(chronological_health, chronological_health[1:])]
    features.update(
        {
            "health_path_chronological": chronological_health,
            "health_path_mean": sum(chronological_health) / len(chronological_health),
            "health_path_range": max(chronological_health) - min(chronological_health),
            "health_change_30": features["health"] - features["health_lag_30"],
            "health_falling_segments": sum(change < 0.0 for change in changes),
            "health_rising_segments": sum(change > 0.0 for change in changes),
            "vehicle_count": snapshot.vehicle_count,
            "network_mean_speed_mps": _finite(snapshot.mean_speed),
            "active_cav_count": len(snapshot.active_cavs),
        }
    )
    features.update(placement_features(snapshot))
    features.update(local_traffic_features(snapshot, definition))
    features.update(
        {
            f"decision_{key}": value
            for key, value in step_metrics(snapshot, definition).items()
            if key
            not in {
                "hard_brake_vehicle_ids",
                "low_ttc_pairs",
                "collision_vehicle_ids",
                "starting_teleport_vehicle_ids",
                "relative_step",
                "sim_time",
            }
        }
    )
    return features


def features_from_trace(
    trace: ReplayTrace,
    *,
    decision_step: int,
    definition: OutcomeDefinition,
    dose_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build snapshot plus causal traffic-history features at a decision."""
    if decision_step < 0 or decision_step >= len(trace.snapshots):
        raise ValueError(f"decision step {decision_step} is outside the trace")
    current = trace.snapshots[decision_step]
    features = features_from_snapshot(current, definition)

    for lag in TRAFFIC_PATTERN_LAGS:
        lag_step = max(0, decision_step - lag)
        snapshot = trace.snapshots[lag_step]
        metric = step_metrics(snapshot, definition)
        placement = placement_features(snapshot)
        prefix = f"traffic_lag_{lag}"
        features.update(
            {
                f"{prefix}_actual_step": lag_step,
                f"{prefix}_queue_count": metric["queue_count"],
                f"{prefix}_pre_merge_mean_speed_mps": metric["pre_merge_mean_speed_mps"],
                f"{prefix}_merge_mean_speed_mps": metric["merge_mean_speed_mps"],
                f"{prefix}_active_cav_count": metric["active_cav_count"],
                f"{prefix}_active_cav_pre_merge_lane_count": metric[
                    "active_cav_pre_merge_lane_count"
                ],
                f"{prefix}_pre_merge_cav_fraction": placement["placement_pre_merge_cav_fraction"],
                f"{prefix}_cav_longitudinal_span_m": placement["placement_longitudinal_span_m"],
                f"{prefix}_soft_band_cav_count": placement["placement_soft_band_cav_count"],
                f"{prefix}_soft_band_lane_coverage_fraction": placement[
                    "placement_soft_band_lane_coverage_fraction"
                ],
                f"{prefix}_strong_band_cav_count": placement["placement_strong_band_cav_count"],
                f"{prefix}_strong_band_lane_coverage_fraction": placement[
                    "placement_strong_band_lane_coverage_fraction"
                ],
                f"{prefix}_inflow_scaled": _finite(snapshot.supervisor_observation[11]),
                f"{prefix}_density_scaled": _finite(snapshot.supervisor_observation[2]),
            }
        )

    window_start = max(0, decision_step - 60)
    pattern_snapshots = trace.snapshots[window_start : decision_step + 1]
    pattern_metrics = [step_metrics(snapshot, definition) for snapshot in pattern_snapshots]
    queue_path = [float(metric["queue_count"]) for metric in pattern_metrics]
    pre_speed_path = [
        float(metric["pre_merge_mean_speed_mps"])
        for metric in pattern_metrics
        if metric["pre_merge_mean_speed_mps"] is not None
    ]
    merge_speed_path = [
        float(metric["merge_mean_speed_mps"])
        for metric in pattern_metrics
        if metric["merge_mean_speed_mps"] is not None
    ]
    inflow_path = [_finite(snapshot.supervisor_observation[11]) for snapshot in pattern_snapshots]
    inband_path = [_finite(snapshot.supervisor_observation[6]) for snapshot in pattern_snapshots]
    hard_brake_id_path = [
        set(str(value) for value in metric["hard_brake_vehicle_ids"]) for metric in pattern_metrics
    ]
    hard_brake_counts = [len(values) for values in hard_brake_id_path]
    low_ttc_pair_counts = [len(metric["low_ttc_pairs"]) for metric in pattern_metrics]
    pre_merge_speed_std_path = []
    for snapshot in pattern_snapshots:
        speeds = [_finite(row[4]) for row in snapshot.vehicles if str(row[1]) == PRE_MERGE_EDGE]
        pre_merge_speed_std_path.append(_std(speeds))

    def path_summary(name: str, values: Sequence[float]) -> Dict[str, Any]:
        if not values:
            return {
                f"pattern60_{name}_mean": None,
                f"pattern60_{name}_min": None,
                f"pattern60_{name}_max": None,
                f"pattern60_{name}_range": None,
                f"pattern60_{name}_change": None,
            }
        return {
            f"pattern60_{name}_mean": sum(values) / len(values),
            f"pattern60_{name}_min": min(values),
            f"pattern60_{name}_max": max(values),
            f"pattern60_{name}_range": max(values) - min(values),
            f"pattern60_{name}_change": values[-1] - values[0],
        }

    features.update(path_summary("queue_count", queue_path))
    features.update(path_summary("pre_merge_speed_mps", pre_speed_path))
    features.update(path_summary("merge_speed_mps", merge_speed_path))
    features.update(path_summary("inflow_scaled", inflow_path))
    features.update(path_summary("pre_merge_speed_std_mps", pre_merge_speed_std_path))
    hard_brake_onsets = 0
    previous_ids = hard_brake_id_path[0] if hard_brake_id_path else set()
    for current_ids in hard_brake_id_path[1:]:
        hard_brake_onsets += len(current_ids - previous_ids)
        previous_ids = current_ids
    features.update(
        {
            "pattern60_queue_count_slope_per_step": _linear_slope(queue_path),
            "pattern60_pre_merge_speed_slope_mps_per_step": _linear_slope(pre_speed_path),
            "pattern60_hard_brake_vehicle_steps": sum(hard_brake_counts),
            "pattern60_hard_brake_onset_count": hard_brake_onsets,
            "pattern60_hard_brake_active_fraction": (
                sum(value > 0 for value in hard_brake_counts) / len(hard_brake_counts)
                if hard_brake_counts
                else 0.0
            ),
            "pattern60_low_ttc_pair_steps": sum(low_ttc_pair_counts),
            "pattern60_low_ttc_active_fraction": (
                sum(value > 0 for value in low_ttc_pair_counts) / len(low_ttc_pair_counts)
                if low_ttc_pair_counts
                else 0.0
            ),
        }
    )
    features["pattern60_inband_available_fraction"] = (
        sum(value > 0.0 for value in inband_path) / len(inband_path) if inband_path else 0.0
    )
    features["pattern_window_start_step"] = window_start
    features.update(
        observation_v2_features(
            trace,
            decision_step=decision_step,
            definition=definition,
            dose_context=dose_context,
        )
    )
    return features


def _assert_identical_prefix(traces: Mapping[int, ReplayTrace], decision_step: int) -> str:
    actions = sorted(traces)
    reference = traces[actions[0]]
    prefix = [snapshot.digest for snapshot in reference.snapshots[: decision_step + 1]]
    for action in actions[1:]:
        candidate = [snapshot.digest for snapshot in traces[action].snapshots[: decision_step + 1]]
        if candidate != prefix:
            raise RuntimeError(
                f"counterfactual prefix diverged before treatment for action {action}"
            )
    return reference.snapshots[decision_step].digest


def generate_paired_case(
    *,
    project_root: Path,
    sumo_binary: str,
    sumo_config: str,
    seed: int,
    search_steps: int,
    outcome_steps: int,
    treatment_steps: int,
    decision_step_override: Optional[int] = None,
    min_decision_step: int = 10,
    health_threshold: float = 0.45,
    require_nonpositive_trend: bool = True,
    min_inband_scaled: float = 0.0,
    cav_penetration: float = 0.2,
    sumo_step_length: float = 0.5,
    soft_wall_tau: Optional[float] = None,
    actions: Sequence[int] = (0, 1, 2),
    definition: OutcomeDefinition = OutcomeDefinition(),
) -> Dict[str, Any]:
    """Generate one paired counterfactual case or an explicit ineligible record."""
    replay_actions = tuple(dict.fromkeys(int(action) for action in actions))
    if 0 not in replay_actions:
        raise ValueError("counterfactual actions must include the off arm 0")
    if any(action not in (0, 1, 2) for action in replay_actions):
        raise ValueError("counterfactual actions must be drawn from 0, 1, and 2")
    common = {
        "project_root": Path(project_root),
        "sumo_binary": str(sumo_binary),
        "sumo_config": str(sumo_config),
        "seed": int(seed),
        "cav_penetration": float(cav_penetration),
        "sumo_step_length": float(sumo_step_length),
        "supervisor_overrides": (
            {"rung_soft_wall_tau": float(soft_wall_tau)} if soft_wall_tau is not None else {}
        ),
    }
    eligibility_trace: Optional[ReplayTrace] = None
    if decision_step_override is None:
        eligibility_trace = run_replay(
            **common,
            decision_step=0,
            horizon_steps=int(search_steps),
            prefix_action=0,
            post_action=0,
        )
        decision_step = find_decision_step(
            eligibility_trace,
            min_step=min_decision_step,
            health_threshold=health_threshold,
            require_nonpositive_trend=require_nonpositive_trend,
            min_inband_scaled=min_inband_scaled,
        )
        eligibility = {
            "selection_mode": "first_causal_warning",
            "min_decision_step": int(min_decision_step),
            "health_threshold": float(health_threshold),
            "require_nonpositive_trend": bool(require_nonpositive_trend),
            "min_inband_scaled": float(min_inband_scaled),
        }
    else:
        decision_step = int(decision_step_override)
        if decision_step < 0:
            raise ValueError("decision_step_override must be non-negative")
        eligibility = {
            "selection_mode": "predeclared_fixed_step",
            "fixed_decision_step": decision_step,
            "note": "Step selected before replay; no future outcome was consulted.",
        }
    base = {
        "schema_version": 2,
        "seed": int(seed),
        "scenario": {
            "project_root": str(Path(project_root)),
            "sumo_binary": str(sumo_binary),
            "sumo_config": str(sumo_config),
            "cav_penetration": float(cav_penetration),
            "sumo_step_length": float(sumo_step_length),
            "soft_wall_tau": (float(soft_wall_tau) if soft_wall_tau is not None else None),
            "actions": list(replay_actions),
        },
        "eligibility": eligibility,
        "search_trace_digest": (
            eligibility_trace.trace_digest if eligibility_trace is not None else None
        ),
    }
    if decision_step is None:
        base.update(
            {
                "eligible": False,
                "reason": "no_causal_warning_landmark_in_search_window",
                "search_steps_completed": (
                    eligibility_trace.completed_steps if eligibility_trace is not None else 0
                ),
            }
        )
        return base

    traces = {
        action: run_replay(
            **common,
            decision_step=decision_step,
            horizon_steps=int(outcome_steps),
            prefix_action=0,
            post_action=action,
            treatment_steps=int(treatment_steps),
        )
        for action in replay_actions
    }
    prefix_digest = _assert_identical_prefix(traces, decision_step)
    decision_snapshot = traces[0].snapshots[decision_step]
    base.update(
        {
            "eligible": True,
            "case_id": f"seed-{int(seed)}-step-{decision_step}",
            "decision_step": decision_step,
            "decision_sim_time_s": _finite(decision_snapshot.sim_time),
            "decision_state_digest": prefix_digest,
            "features": features_from_trace(
                traces[0], decision_step=decision_step, definition=definition
            ),
            "treatment_steps": int(treatment_steps),
            "outcome_steps": int(outcome_steps),
            "outcome_definition": definition.to_dict(),
            "arms": {
                str(action): {
                    "trace_digest": trace.trace_digest,
                    "outcomes": summarize_branch(
                        trace,
                        decision_step=decision_step,
                        outcome_steps=outcome_steps,
                        definition=definition,
                    ),
                }
                for action, trace in traces.items()
            },
        }
    )
    return base


def summarize_generation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    reasons = Counter(
        str(row.get("reason", "eligible")) if not row.get("eligible") else "eligible"
        for row in rows
    )
    return {
        "case_count": len(rows),
        "eligible_case_count": sum(bool(row.get("eligible")) for row in rows),
        "ineligible_case_count": sum(not bool(row.get("eligible")) for row in rows),
        "status_counts": dict(sorted(reasons.items())),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "case_ids": [str(row["case_id"]) for row in rows if row.get("case_id")],
    }
