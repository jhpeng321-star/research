"""TraCI helpers for direct-control policies: kinematic speed and merge detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set

import traci

# SUMO lane-change state bits (see TraCI constants).
LCA_LEFT = 1
LCA_RIGHT = 2
LCA_STRATEGIC = 256
LCA_URGENT = 512

_CONFIGURED: Set[str] = set()


@dataclass
class MergeIntent:
    """A vehicle in an adjacent lane wants to merge into the CAV's lane."""

    vehicle_id: str
    delta_x: float
    v_merging: float


def compute_a_req(v_current: float, v_target: float, delta_x: float) -> float:
    """Required constant acceleration to reach v_target over distance delta_x (m/s²)."""
    delta_x = max(delta_x, 1.0)
    return (v_target**2 - v_current**2) / (2.0 * delta_x)


def configure_cav(cav_id: str) -> None:
    """One-time TraCI setup so speed commands are respected."""
    if cav_id in _CONFIGURED:
        return
    if cav_id not in traci.vehicle.getIDList():
        return
    traci.vehicle.setSpeedMode(cav_id, 31)
    traci.vehicle.setLaneChangeMode(cav_id, 1621)
    _CONFIGURED.add(cav_id)


def reset_configured() -> None:
    """Clear the configure cache (for tests)."""
    _CONFIGURED.clear()


def apply_kinematic_speed(
    cav_id: str,
    v_current: float,
    a_req: float,
    max_decel: float,
    dt: float = 1.0,
) -> None:
    """Apply a_req over one step via slowDown, clamped to max_decel."""
    a_clamped = max(-max_decel, a_req)
    v_next = max(0.0, v_current + a_clamped * dt)
    traci.vehicle.slowDown(cav_id, v_next, duration=dt)


def release_to_sumo(cav_id: str, tau: float = 1.0) -> None:
    """Return longitudinal control to SUMO with normal headway."""
    traci.vehicle.setTau(cav_id, tau)
    traci.vehicle.setSpeed(cav_id, -1)


def _lane_change_toward(cav_lane: int, other_lane: int, lc_state: int) -> bool:
    """True if other vehicle's lane-change state moves toward cav_lane."""
    if other_lane == cav_lane:
        return False
    if other_lane < cav_lane and (lc_state & LCA_LEFT):
        return True
    if other_lane > cav_lane and (lc_state & LCA_RIGHT):
        return True
    return False


def _best_lane_targets_cav(cav_id: str, other_id: str) -> bool:
    """True if SUMO routing prefers the CAV's lane for the other vehicle."""
    try:
        cav_lane_id = traci.vehicle.getLaneID(cav_id)
        for lane_id, _, _, _ in traci.vehicle.getBestLanes(other_id):
            if lane_id == cav_lane_id:
                return True
    except Exception:
        return False
    return False


def _wants_cav_lane(cav_id: str, other_id: str) -> bool:
    """True if other_id is trying to move into cav_id's lane."""
    try:
        cav_lane = traci.vehicle.getLaneIndex(cav_id)
        other_lane = traci.vehicle.getLaneIndex(other_id)
    except Exception:
        return False

    if other_lane == cav_lane:
        return False

    try:
        lc_state = traci.vehicle.getLaneChangeState(other_id)
    except Exception:
        lc_state = 0

    if _lane_change_toward(cav_lane, other_lane, lc_state):
        return True
    return _best_lane_targets_cav(cav_id, other_id)


def _longitudinal_gap(cav_id: str, other_id: str) -> float:
    """Longitudinal separation along the road (m), minimum 1 m."""
    for getter in (
        traci.vehicle.getLeftLeaders,
        traci.vehicle.getRightLeaders,
        traci.vehicle.getLeftFollowers,
        traci.vehicle.getRightFollowers,
    ):
        try:
            for entry in getter(cav_id, 200.0):
                if entry[0] == other_id:
                    return max(1.0, entry[1])
        except Exception:
            continue

    try:
        cav_edge = traci.vehicle.getRoadID(cav_id)
        other_edge = traci.vehicle.getRoadID(other_id)
        cav_pos = traci.vehicle.getLanePosition(cav_id)
        other_pos = traci.vehicle.getLanePosition(other_id)
    except Exception:
        return float("inf")

    if cav_edge == other_edge:
        return max(1.0, abs(other_pos - cav_pos))

    return float("inf")


def _adjacent_candidates(cav_id: str, lateral_radius: float) -> list:
    """Return nearby vehicles in lanes adjacent to the CAV."""
    candidates = []
    try:
        cav_edge = traci.vehicle.getRoadID(cav_id)
        cav_lane = traci.vehicle.getLaneIndex(cav_id)
        cav_pos = traci.vehicle.getLanePosition(cav_id)
    except Exception:
        return candidates

    for veh_id in traci.vehicle.getIDList():
        if veh_id == cav_id:
            continue
        try:
            if traci.vehicle.getRoadID(veh_id) != cav_edge:
                continue
            if abs(traci.vehicle.getLaneIndex(veh_id) - cav_lane) != 1:
                continue
            if abs(traci.vehicle.getLanePosition(veh_id) - cav_pos) > lateral_radius:
                continue
            candidates.append(veh_id)
        except Exception:
            continue

    for getter in (
        traci.vehicle.getLeftLeaders,
        traci.vehicle.getRightLeaders,
        traci.vehicle.getLeftFollowers,
        traci.vehicle.getRightFollowers,
    ):
        try:
            for entry in getter(cav_id, lateral_radius):
                other_id = entry[0]
                if other_id != cav_id and other_id not in candidates:
                    candidates.append(other_id)
        except Exception:
            continue

    return candidates


def find_merge_intent(cav_id: str, lateral_radius: float = 50.0) -> Optional[MergeIntent]:
    """Return the most urgent adjacent-lane vehicle trying to merge into cav_id's lane."""
    best: Optional[MergeIntent] = None

    for other_id in _adjacent_candidates(cav_id, lateral_radius):
        if not _wants_cav_lane(cav_id, other_id):
            continue
        delta_x = _longitudinal_gap(cav_id, other_id)
        if delta_x == float("inf"):
            continue
        try:
            v_merging = traci.vehicle.getSpeed(other_id)
        except Exception:
            continue

        intent = MergeIntent(vehicle_id=other_id, delta_x=delta_x, v_merging=v_merging)
        if best is None or intent.delta_x < best.delta_x:
            best = intent

    return best
