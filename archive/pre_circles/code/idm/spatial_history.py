"""Diagnostic-only spatial RSU history for IDM authority replays.

This module reads the ambient :class:`TrafficSnapshot` and never writes to the
environment.  Its outputs are intentionally excluded from actor observations,
candidate ranking, rewards, and controller state until a later, separately
frozen observation-ablation contract promotes them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import math

import numpy as np
import traci

from env.snapshot import get_active_snapshot


SPATIAL_HISTORY_SCHEMA = "idm_spatial_rsu_history/1"


@dataclass(frozen=True)
class SpatialHistoryContract:
    sample_interval_s: float = 1.0
    retained_history_s: float = 30.0
    causal_windows_s: tuple[float, ...] = (10.0, 20.0, 30.0)
    mainline_upstream_coverage_m: float = 2000.0
    band_width_m: float = 50.0
    ramp_coverage_m: float = 400.0
    downstream_coverage_m: float = 500.0
    slow_speed_threshold_mps: float = 15.0
    free_flow_threshold_mps: float = 20.0
    slow_cluster_min_vehicles: int = 2
    slow_cluster_max_separation_m: float = 75.0
    queue_anchor_distance_m: float = 250.0
    severe_braking_threshold_mps2: float = -3.0
    low_ttc_threshold_s: float = 1.5

    def __post_init__(self) -> None:
        positive = {
            "sample_interval_s": self.sample_interval_s,
            "retained_history_s": self.retained_history_s,
            "mainline_upstream_coverage_m": self.mainline_upstream_coverage_m,
            "band_width_m": self.band_width_m,
            "ramp_coverage_m": self.ramp_coverage_m,
            "downstream_coverage_m": self.downstream_coverage_m,
            "slow_cluster_max_separation_m": self.slow_cluster_max_separation_m,
            "queue_anchor_distance_m": self.queue_anchor_distance_m,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.slow_cluster_min_vehicles < 2:
            raise ValueError("slow_cluster_min_vehicles must be at least two")
        if any(float(value) <= 0.0 for value in self.causal_windows_s):
            raise ValueError("causal windows must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SPATIAL_HISTORY_SCHEMA, **asdict(self)}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _lane_length(lane_id: str, provided: Mapping[str, float] | None) -> float | None:
    if provided is not None and lane_id in provided:
        return _finite(provided[lane_id])
    try:
        return _finite(traci.lane.getLength(lane_id))
    except Exception:
        return None


def _vehicle_rows(
    snapshot: Any,
    *,
    pre_edge: str,
    ramp_edge: str,
    lane_lengths: Mapping[str, float] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_id in snapshot.ordered_ids():
        vehicle_id = str(raw_id)
        road_id = snapshot.road(vehicle_id)
        lane_id = snapshot.lane_id(vehicle_id)
        lane_position = _finite(snapshot.lane_pos(vehicle_id))
        speed = _finite(snapshot.speed(vehicle_id))
        acceleration = _finite(snapshot.accel(vehicle_id))
        if road_id is None or lane_id is None or lane_position is None or speed is None:
            continue
        distance_to_merge = None
        if road_id in {pre_edge, ramp_edge}:
            length = _lane_length(str(lane_id), lane_lengths)
            if length is not None:
                distance_to_merge = max(length - lane_position, 0.0)
        leader = snapshot.leader(vehicle_id)
        leader_id = str(leader[0]) if leader and leader[0] else None
        leader_gap = _finite(leader[1]) if leader else None
        leader_speed = _finite(snapshot.speed(leader_id)) if leader_id else None
        closing = None if leader_speed is None else speed - leader_speed
        ttc = (
            max(leader_gap, 0.0) / closing
            if leader_gap is not None and closing is not None and closing > 1e-6
            else None
        )
        rows.append(
            {
                "vehicle_id": vehicle_id,
                "road_id": str(road_id),
                "lane_id": str(lane_id),
                "lane_position_m": lane_position,
                "distance_to_merge_m": distance_to_merge,
                "speed_mps": speed,
                "acceleration_mps2": acceleration,
                "ttc_s": ttc,
            }
        )
    return rows


def _band_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    start_m: float,
    end_m: float,
    contract: SpatialHistoryContract,
) -> dict[str, Any]:
    speeds = [float(row["speed_mps"]) for row in rows]
    accelerations = [
        float(row["acceleration_mps2"])
        for row in rows
        if row.get("acceleration_mps2") is not None
    ]
    ttc_values = [float(row["ttc_s"]) for row in rows if row.get("ttc_s") is not None]
    width = end_m - start_m
    mean_speed = float(np.mean(speeds)) if speeds else None
    return {
        "start_distance_m": float(start_m),
        "end_distance_m": float(end_m),
        "vehicle_count": len(rows),
        "linear_density_veh_per_m": len(rows) / width,
        "mean_speed_mps": mean_speed,
        "p10_speed_mps": float(np.percentile(speeds, 10)) if speeds else None,
        "speed_variance_m2ps2": float(np.var(speeds)) if speeds else None,
        "mean_acceleration_mps2": (
            float(np.mean(accelerations)) if accelerations else None
        ),
        "acceleration_energy_m2ps4": (
            sum(value * value for value in accelerations) if accelerations else None
        ),
        "severe_braking_count": sum(
            value < contract.severe_braking_threshold_mps2 for value in accelerations
        ),
        "low_ttc_count": sum(
            value < contract.low_ttc_threshold_s for value in ttc_values
        ),
        # This is not a detector flow measurement.  The explicit name prevents
        # it from being cited as one.
        "density_speed_flow_proxy_veh_per_s": (
            len(rows) * mean_speed / width if mean_speed is not None else None
        ),
    }


def _band_profile(
    rows: Sequence[Mapping[str, Any]],
    *,
    coverage_m: float,
    contract: SpatialHistoryContract,
) -> list[dict[str, Any]]:
    count = int(math.ceil(coverage_m / contract.band_width_m))
    output = []
    for index in range(count):
        start = index * contract.band_width_m
        end = min((index + 1) * contract.band_width_m, coverage_m)
        members = [
            row
            for row in rows
            if row.get("distance_to_merge_m") is not None
            and start <= float(row["distance_to_merge_m"]) < end
        ]
        output.append(
            _band_summary(members, start_m=start, end_m=end, contract=contract)
        )
    return output


def _slow_vehicle_clusters(
    rows: Sequence[Mapping[str, Any]],
    contract: SpatialHistoryContract,
) -> list[dict[str, Any]]:
    slow = sorted(
        (
            row
            for row in rows
            if row.get("distance_to_merge_m") is not None
            and float(row["speed_mps"]) < contract.slow_speed_threshold_mps
        ),
        key=lambda row: float(row["distance_to_merge_m"]),
    )
    clusters: list[list[Mapping[str, Any]]] = []
    for row in slow:
        distance = float(row["distance_to_merge_m"])
        if not clusters:
            clusters.append([row])
            continue
        previous = float(clusters[-1][-1]["distance_to_merge_m"])
        if distance - previous <= contract.slow_cluster_max_separation_m:
            clusters[-1].append(row)
        else:
            clusters.append([row])
    output = []
    for members in clusters:
        if len(members) < contract.slow_cluster_min_vehicles:
            continue
        distances = [float(row["distance_to_merge_m"]) for row in members]
        output.append(
            {
                "vehicle_ids": [str(row["vehicle_id"]) for row in members],
                "vehicle_count": len(members),
                "downstream_edge_distance_m": min(distances),
                "upstream_edge_distance_m": max(distances),
                "connected_to_bottleneck": (
                    min(distances) <= contract.queue_anchor_distance_m
                ),
                "mean_speed_mps": float(
                    np.mean([float(row["speed_mps"]) for row in members])
                ),
            }
        )
    return output


def _fronts_by_lane(
    mainline_rows: Sequence[Mapping[str, Any]],
    contract: SpatialHistoryContract,
) -> dict[str, dict[str, Any]]:
    lanes = sorted({str(row["lane_id"]) for row in mainline_rows})
    output: dict[str, dict[str, Any]] = {}
    for lane_id in lanes:
        clusters = _slow_vehicle_clusters(
            [row for row in mainline_rows if str(row["lane_id"]) == lane_id],
            contract,
        )
        queues = [row for row in clusters if row["connected_to_bottleneck"]]
        detached = [row for row in clusters if not row["connected_to_bottleneck"]]
        queue_tail = (
            max(float(row["upstream_edge_distance_m"]) for row in queues)
            if queues
            else None
        )
        wave_front = (
            max(float(row["upstream_edge_distance_m"]) for row in detached)
            if detached
            else None
        )
        output[lane_id] = {
            "queue_tail_distance_m": queue_tail,
            "detached_wave_front_distance_m": wave_front,
            "clusters": clusters,
            "front_detection_reason": (
                "detached_slow_cluster"
                if wave_front is not None
                else "queue_connected_slow_cluster"
                if queue_tail is not None
                else "no_qualified_slow_cluster"
            ),
        }
    return output


def _road_summary(
    rows: Sequence[Mapping[str, Any]], contract: SpatialHistoryContract
) -> dict[str, Any]:
    speeds = [float(row["speed_mps"]) for row in rows]
    accelerations = [
        float(row["acceleration_mps2"])
        for row in rows
        if row.get("acceleration_mps2") is not None
    ]
    return {
        "vehicle_count": len(rows),
        "mean_speed_mps": float(np.mean(speeds)) if speeds else None,
        "speed_variance_m2ps2": float(np.var(speeds)) if speeds else None,
        "acceleration_energy_m2ps4": (
            sum(value * value for value in accelerations) if accelerations else None
        ),
        "severe_braking_count": sum(
            value < contract.severe_braking_threshold_mps2 for value in accelerations
        ),
    }


def capture_corridor_frame(
    env: Any,
    contract: SpatialHistoryContract,
    *,
    snapshot: Any | None = None,
    simulation_time_s: float | None = None,
    lane_lengths: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Capture one read-only, spatially resolved corridor frame."""
    snapshot = get_active_snapshot() if snapshot is None else snapshot
    if snapshot is None:
        return {"schema": SPATIAL_HISTORY_SCHEMA, "snapshot_available": False}
    if simulation_time_s is None:
        simulation_time_s = float(traci.simulation.getTime())
    pre_edge = str(env._pre_edge)
    ramp_edge = str(env._ramp_edge)
    merge_edge = str(env._merge_edge)
    down_edge = str(env._shepherd_engine.down_edge)
    rows = _vehicle_rows(
        snapshot,
        pre_edge=pre_edge,
        ramp_edge=ramp_edge,
        lane_lengths=lane_lengths,
    )
    mainline = [row for row in rows if row["road_id"] == pre_edge]
    ramp = [row for row in rows if row["road_id"] == ramp_edge]
    lane_ids = sorted({str(row["lane_id"]) for row in mainline})
    return {
        "schema": SPATIAL_HISTORY_SCHEMA,
        "snapshot_available": True,
        "simulation_time_s": float(simulation_time_s),
        "contract": contract.to_dict(),
        "mainline": {
            "all_lanes": _band_profile(
                mainline,
                coverage_m=contract.mainline_upstream_coverage_m,
                contract=contract,
            ),
            "by_lane": {
                lane_id: _band_profile(
                    [row for row in mainline if str(row["lane_id"]) == lane_id],
                    coverage_m=contract.mainline_upstream_coverage_m,
                    contract=contract,
                )
                for lane_id in lane_ids
            },
        },
        "ramp": {
            "bands": _band_profile(
                ramp,
                coverage_m=contract.ramp_coverage_m,
                contract=contract,
            ),
            "summary": _road_summary(ramp, contract),
        },
        "merge": _road_summary(
            [row for row in rows if row["road_id"] == merge_edge], contract
        ),
        "downstream": _road_summary(
            [row for row in rows if row["road_id"] == down_edge], contract
        ),
        "fronts_by_lane": _fronts_by_lane(mainline, contract),
    }


def _fit_front_velocity(
    samples: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    """Fit upstream-positive front velocity using only supplied causal samples."""
    if len(samples) < 2:
        return {"velocity_mps": None, "sample_count": len(samples), "r_squared": None}
    times = np.asarray([float(row[0]) for row in samples], dtype=float)
    positions = np.asarray([float(row[1]) for row in samples], dtype=float)
    if np.ptp(times) <= 1e-9:
        return {"velocity_mps": None, "sample_count": len(samples), "r_squared": None}
    slope, intercept = np.polyfit(times, positions, 1)
    predicted = slope * times + intercept
    residual = float(np.sum((positions - predicted) ** 2))
    total = float(np.sum((positions - float(np.mean(positions))) ** 2))
    r_squared = 1.0 if total <= 1e-12 and residual <= 1e-12 else (
        None if total <= 1e-12 else 1.0 - residual / total
    )
    return {
        "velocity_mps": float(slope),
        "sample_count": len(samples),
        "r_squared": r_squared,
    }


class SpatialRSUHistory:
    """Retain causal spatial frames and attach 10/20/30-second front fits."""

    def __init__(self, contract: SpatialHistoryContract | None = None):
        self.contract = contract or SpatialHistoryContract()
        self._frames: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._frames.clear()

    def observe(self, frame: Mapping[str, Any]) -> dict[str, Any] | None:
        timestamp = _finite(frame.get("simulation_time_s"))
        if timestamp is None:
            raise ValueError("frame requires a finite simulation_time_s")
        if self._frames:
            previous = float(self._frames[-1]["simulation_time_s"])
            if timestamp < previous:
                raise ValueError("spatial history timestamps must be nondecreasing")
            if timestamp - previous + 1e-9 < self.contract.sample_interval_s:
                return None
        stored = dict(frame)
        self._frames.append(stored)
        cutoff = timestamp - self.contract.retained_history_s
        self._frames = [
            row for row in self._frames if float(row["simulation_time_s"]) >= cutoff
        ]
        stored["front_velocity_by_lane"] = self._front_velocity(timestamp)
        return stored

    def _front_velocity(self, timestamp: float) -> dict[str, dict[str, Any]]:
        lanes = sorted(
            {
                str(lane_id)
                for frame in self._frames
                for lane_id in (frame.get("fronts_by_lane") or {})
            }
        )
        output: dict[str, dict[str, Any]] = {}
        for lane_id in lanes:
            output[lane_id] = {}
            for window in self.contract.causal_windows_s:
                cutoff = timestamp - float(window)
                samples = []
                for frame in self._frames:
                    time_s = float(frame["simulation_time_s"])
                    if time_s < cutoff or time_s > timestamp:
                        continue
                    front = (frame.get("fronts_by_lane") or {}).get(lane_id, {})
                    position = front.get("detached_wave_front_distance_m")
                    if position is None:
                        position = front.get("queue_tail_distance_m")
                    if position is not None:
                        samples.append((time_s, float(position)))
                output[lane_id][f"{int(window)}s"] = _fit_front_velocity(samples)
        return output

    def window(self, seconds: float) -> list[dict[str, Any]]:
        if not self._frames:
            return []
        latest = float(self._frames[-1]["simulation_time_s"])
        cutoff = latest - float(seconds)
        return [
            dict(frame)
            for frame in self._frames
            if float(frame["simulation_time_s"]) >= cutoff
        ]
