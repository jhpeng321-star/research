"""Compact causal spatial-response instrumentation for IDM authority screens."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import traci

from env.snapshot import get_active_snapshot
from idm.phase_diagnostics import upstream_follower_states, vehicle_safety_events


def _vehicle_state(
    snapshot: Any,
    vehicle_id: str,
    previous_acceleration: Mapping[str, float],
) -> dict[str, Any] | None:
    if not snapshot.has(vehicle_id):
        return None
    position = snapshot.position(vehicle_id)
    speed = snapshot.speed(vehicle_id)
    acceleration = snapshot.accel(vehicle_id)
    lane_position = snapshot.lane_pos(vehicle_id)
    if speed is None or acceleration is None:
        return None
    acceleration = float(acceleration)
    previous = previous_acceleration.get(vehicle_id)
    return {
        "vehicle_id": vehicle_id,
        "road_id": snapshot.road(vehicle_id),
        "lane_id": snapshot.lane_id(vehicle_id),
        "lane_position_m": None if lane_position is None else float(lane_position),
        "position_xy_m": None if position is None else [float(position[0]), float(position[1])],
        "speed_mps": float(speed),
        "acceleration_mps2": acceleration,
        "jerk_mps3": None if previous is None else acceleration - float(previous),
    }


def _population(
    snapshot: Any,
    vehicle_ids: list[str],
    previous_acceleration: Mapping[str, float],
    *,
    severe_threshold_mps2: float,
    hard_threshold_mps2: float,
    low_ttc_threshold_s: float,
) -> dict[str, Any]:
    states = [
        state
        for vehicle_id in vehicle_ids
        for state in [_vehicle_state(snapshot, vehicle_id, previous_acceleration)]
        if state is not None
    ]
    accelerations = [float(state["acceleration_mps2"]) for state in states]
    speeds = [float(state["speed_mps"]) for state in states]
    ttc_values: list[float] = []
    for state in states:
        leader = snapshot.leader(state["vehicle_id"])
        if not leader or not leader[0] or snapshot.speed(str(leader[0])) is None:
            continue
        closing = float(state["speed_mps"]) - float(snapshot.speed(str(leader[0])))
        if closing > 1e-6:
            ttc_values.append(max(float(leader[1]), 0.0) / closing)
    return {
        "vehicle_ids": [state["vehicle_id"] for state in states],
        "vehicle_count": len(states),
        "mean_speed_mps": float(np.mean(speeds)) if speeds else None,
        "speed_variance_m2ps2": float(np.var(speeds)) if speeds else None,
        "acceleration_energy_m2ps4": sum(value * value for value in accelerations),
        "integrated_negative_acceleration_mps": sum(
            max(-value, 0.0) for value in accelerations
        ),
        "severe_braking_count": sum(
            value < severe_threshold_mps2 for value in accelerations
        ),
        "hard_braking_count": sum(value < hard_threshold_mps2 for value in accelerations),
        "minimum_ttc_s": min(ttc_values) if ttc_values else None,
        "low_ttc_count": sum(value < low_ttc_threshold_s for value in ttc_values),
    }


def capture_spatial_response(
    env: Any,
    *,
    relative_step: int,
    role_ids: set[str],
    frozen_lane_id: str,
    previous_acceleration: Mapping[str, float],
    commands: list[dict[str, Any]],
    follower_count: int = 5,
) -> dict[str, Any]:
    """Capture one post-action step from absorber through merge service."""
    snapshot = get_active_snapshot()
    if snapshot is None:
        return {"relative_step": relative_step, "snapshot_available": False}
    lane_rows = [
        (float(position), vehicle_id)
        for vehicle_id in snapshot.ordered_ids()
        for position in [snapshot.lane_pos(vehicle_id)]
        if snapshot.lane_id(vehicle_id) == frozen_lane_id and position is not None
    ]
    lane_rows.sort(reverse=True)
    lane_ids = [vehicle_id for _, vehicle_id in lane_rows]
    command_by_id = {str(command["vehicle_id"]): command for command in commands}
    absorbers = []
    immediate_follower_ids: list[str] = []
    influence_ids: list[str] = []
    for role_id in sorted(role_ids):
        state = _vehicle_state(snapshot, role_id, previous_acceleration)
        if state is None:
            continue
        command = command_by_id.get(role_id, {})
        state["command"] = {
            "analytical_acceleration_mps2": command.get("nominal_acceleration_mps2"),
            "action_family": command.get("action_family"),
            "requested_action": command.get("requested_action"),
            "filtered_action": command.get("filtered_action"),
            "desired_acceleration_mps2": command.get("desired_acceleration_mps2"),
            "projected_acceleration_mps2": (command.get("projected") or {}).get(
                "acceleration_mps2"
            ),
            "projection_reasons": list((command.get("projected") or {}).get("reasons") or []),
            "analytical_reference_mps": (command.get("reference") or {}).get(
                "filtered_reference_mps"
            ),
            "effective_reference_mps": (command.get("reference") or {}).get(
                "effective_reference_mps"
            ),
        }
        follower_id = None
        follower_gap = None
        if role_id in lane_ids:
            index = lane_ids.index(role_id)
            if index + 1 < len(lane_ids):
                follower_id = lane_ids[index + 1]
                role_position = snapshot.lane_pos(role_id)
                follower_position = snapshot.lane_pos(follower_id)
                if role_position is not None and follower_position is not None:
                    follower_gap = max(float(role_position) - float(follower_position) - 5.0, 0.0)
                influence_ids.extend(lane_ids[index + 1 : index + 1 + follower_count])
        follower_state = (
            _vehicle_state(snapshot, follower_id, previous_acceleration) if follower_id else None
        )
        if follower_state is not None:
            immediate_follower_ids.append(follower_id)
            closing = float(follower_state["speed_mps"]) - float(state["speed_mps"])
            follower_state["gap_m"] = follower_gap
            follower_state["ttc_s"] = (
                follower_gap / closing
                if follower_gap is not None and closing > 1e-6
                else None
            )
        state["immediate_follower"] = follower_state
        absorbers.append(state)
    influence_ids = list(dict.fromkeys(influence_ids))[:follower_count]
    corridor_edges = {
        env._pre_edge,
        env._merge_edge,
        env._ramp_edge,
        env._shepherd_engine.down_edge,
    }
    corridor_ids = [
        vehicle_id
        for vehicle_id in snapshot.ordered_ids()
        if snapshot.road(vehicle_id) in corridor_edges
    ]
    severe_threshold = float(env.config.get("idm_severe_brake_threshold_mps2", -3.0))
    hard_threshold = -4.5
    low_ttc_threshold = float(env.config.get("corridor_ttc_threshold", 1.5))
    population_kwargs = {
        "severe_threshold_mps2": severe_threshold,
        "hard_threshold_mps2": hard_threshold,
        "low_ttc_threshold_s": low_ttc_threshold,
    }
    slow_lane_positions = [
        float(position)
        for position, vehicle_id in lane_rows
        if snapshot.speed(vehicle_id) is not None and float(snapshot.speed(vehicle_id)) < 15.0
    ]
    premerge_speeds = [
        float(speed)
        for vehicle_id in snapshot.ordered_ids()
        if snapshot.road(vehicle_id) == env._pre_edge
        for speed in [snapshot.speed(vehicle_id)]
        if speed is not None
    ]
    collapse_step = bool(
        premerge_speeds
        and sum(speed < 15.0 for speed in premerge_speeds) >= 5
        and float(np.mean(premerge_speeds)) <= 15.0
    )
    follower_trajectories: list[dict[str, Any]] = []
    relation_by_vehicle: dict[str, str] = {}
    for role_id in sorted(role_ids):
        relation_by_vehicle[role_id] = "target"
        for row in upstream_follower_states(
            snapshot,
            role_id,
            count=follower_count,
        ):
            follower_trajectories.append(row)
            relation_by_vehicle.setdefault(
                str(row["vehicle_id"]),
                str(row["relation_to_target"]),
            )
    simulation_time_s = float(traci.simulation.getTime())
    safety_events = vehicle_safety_events(
        snapshot,
        corridor_ids,
        relative_step=relative_step,
        simulation_time_s=simulation_time_s,
        relation_by_vehicle=relation_by_vehicle,
        severe_threshold_mps2=severe_threshold,
        hard_threshold_mps2=hard_threshold,
        low_ttc_threshold_s=low_ttc_threshold,
    )
    return {
        "relative_step": int(relative_step),
        "simulation_time_s": simulation_time_s,
        "snapshot_available": True,
        "frozen_lane_id": frozen_lane_id,
        "absorbers": absorbers,
        "upstream_followers": follower_trajectories,
        "safety_events": safety_events,
        "immediate_follower": _population(
            snapshot,
            list(dict.fromkeys(immediate_follower_ids)),
            previous_acceleration,
            **population_kwargs,
        ),
        "influence_region": _population(
            snapshot,
            influence_ids,
            previous_acceleration,
            **population_kwargs,
        ),
        "lane": {
            **_population(snapshot, lane_ids, previous_acceleration, **population_kwargs),
            "low_speed_vehicle_count": len(slow_lane_positions),
            "low_speed_wave_span_m": (
                max(slow_lane_positions) - min(slow_lane_positions)
                if len(slow_lane_positions) >= 2
                else 0.0
            ),
        },
        "corridor": {
            **_population(snapshot, corridor_ids, previous_acceleration, **population_kwargs),
            "delay_rate_s_per_vehicle": env._corridor_delay_rate_live(),
            "ttc_count": env._corridor_ttc_count(),
        },
        "merge_service": {
            "discharge_rate_veh_per_s": float(env._flow_tracker.rate()),
            "arrived_this_step": int(traci.simulation.getArrivedNumber()),
            "unfinished_vehicle_count": int(traci.simulation.getMinExpectedNumber()),
            "collapse_condition": collapse_step,
        },
    }
