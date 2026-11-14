"""Consolidated, defensive TraCI readers + small numeric helpers.

Replaces the three near-duplicate proximity scanners and the two copies of
``leader_ttc`` / ``percentile`` that used to live in reward.py,
selfish_reward.py, action.py and evaluation/travel_time.py. Every TraCI call
here is wrapped so a vehicle leaving the network mid-step never raises.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import traci

from config.constants import (
    DEFAULT_VEH_LENGTH,
    MERGE_DROPPED_LANE_INDEX,
    MERGE_DROPPED_LANE_LENGTH,
    MERGE_EDGE,
    MERGE_SURVIVING_LANE_INDEX,
)
from env.snapshot import get_active_snapshot

# Look-ahead distance used for leader / adjacency queries (metres).
# When a snapshot is active it serves these queries from its subscription
# data; the VAR_LEADER subscription is parameterized with this same distance
# (see env.snapshot.SUBSCRIPTION_LOOK_AHEAD).
DEFAULT_LOOK_AHEAD = 200.0


# --- leaders / TTC --------------------------------------------------------
def _leader_info_from_snapshot(veh_id: str, snap) -> Optional[Tuple[str, float, float]]:
    """Snapshot equivalent of the direct getLeader path below (same values)."""
    leader = snap.leader(veh_id)
    if leader is None:
        return None
    try:
        leader_id, distance = leader
    except (TypeError, ValueError):
        return None
    # A no-leader record ("", -1) and leaders that just left the network are
    # filtered exactly like the direct path's getIDList() membership check.
    if leader_id not in snap.id_set:
        return None
    leader_speed = snap.speed(leader_id)
    if leader_speed is None:
        return None
    return leader_id, distance, leader_speed


def leader_info(
    veh_id: str, look_ahead: float = DEFAULT_LOOK_AHEAD
) -> Optional[Tuple[str, float, float]]:
    """Return ``(leader_id, gap, leader_speed)`` or ``None`` if no leader."""
    snap = get_active_snapshot()
    if snap is not None and look_ahead == snap.look_ahead:
        return _leader_info_from_snapshot(veh_id, snap)
    try:
        info = traci.vehicle.getLeader(veh_id, look_ahead)
    except Exception:
        return None
    if info is None:
        return None
    leader_id, distance = info
    if leader_id not in traci.vehicle.getIDList():
        return None
    try:
        leader_speed = traci.vehicle.getSpeed(leader_id)
    except Exception:
        return None
    return leader_id, distance, leader_speed


def _ego_speed(veh_id: str) -> Optional[float]:
    """Current speed; None mirrors the direct getter raising."""
    snap = get_active_snapshot()
    if snap is not None:
        return snap.speed(veh_id)
    try:
        return traci.vehicle.getSpeed(veh_id)
    except Exception:
        return None


def leader_ttc(veh_id: str, look_ahead: float = DEFAULT_LOOK_AHEAD) -> float:
    """Time-to-collision to the vehicle ahead; ``+inf`` if not closing."""
    info = leader_info(veh_id, look_ahead)
    if info is None:
        return float("inf")
    _, gap, leader_speed = info
    v_ego = _ego_speed(veh_id)
    if v_ego is None:
        return float("inf")
    closing = v_ego - leader_speed
    if closing <= 0.01:
        return float("inf")
    return float(gap) / closing


def _vehicle_state(veh_id: str) -> Optional[Tuple[str, int, float, float]]:
    """(edge, lane_index, lane_pos, speed) or None (left network / error)."""
    snap = get_active_snapshot()
    if snap is not None:
        edge = snap.road(veh_id)
        lane_idx = snap.lane_index(veh_id)
        pos = snap.lane_pos(veh_id)
        speed = snap.speed(veh_id)
        if edge is None or lane_idx is None or pos is None or speed is None:
            return None
        return edge, int(lane_idx), float(pos), float(speed)
    try:
        edge = traci.vehicle.getRoadID(veh_id)
        lane_idx = int(traci.vehicle.getLaneIndex(veh_id))
        pos = float(traci.vehicle.getLanePosition(veh_id))
        speed = float(traci.vehicle.getSpeed(veh_id))
    except Exception:
        return None
    return edge, lane_idx, pos, speed


def side_ttc(
    veh_id: str,
    radius: float = 50.0,
    merge_edge: str = MERGE_EDGE,
    dropped_lane: int = MERGE_DROPPED_LANE_INDEX,
    surviving_lane: int = MERGE_SURVIVING_LANE_INDEX,
) -> float:
    """Min side-conflict TTC (s) over converging-lane neighbors; ``+inf`` if none.

    A side conflict exists only where two vehicles must race for the SAME
    lane: on the merge edge (3 lanes -> 2 downstream), the dropped lane ends,
    so pairs on (dropped_lane, surviving_lane) are converging. Parallel pairs
    on continuing lanes are ignored — they never have to share a lane.

    Conflict time = net longitudinal gap / relative speed. Pairs already
    overlapped (|dpos| < veh length) count as TTC 0 — a side collision in
    progress. Only approaching pairs (the faster one closing the gap) count.
    """
    ego = _vehicle_state(veh_id)
    if ego is None:
        return float("inf")
    ego_edge, ego_lane, ego_pos, ego_speed = ego
    if ego_edge != merge_edge:
        return float("inf")

    best = float("inf")
    for nbr_id in vehicles_within_radius(veh_id, radius, same_edge_only=True):
        nbr = _vehicle_state(nbr_id)
        if nbr is None:
            continue
        nbr_edge, nbr_lane, nbr_pos, nbr_speed = nbr
        if nbr_lane == ego_lane:
            continue  # same lane is leader_ttc's job
        lanes = {ego_lane, nbr_lane}
        # Converging pair = the dropped lane and its surviving neighbor lane.
        if lanes != {dropped_lane, surviving_lane}:
            continue
        d = abs(nbr_pos - ego_pos) - DEFAULT_VEH_LENGTH
        if d > 20.0:
            continue
        if d <= 0.0:
            return 0.0  # longitudinally overlapped on converging lanes
        # Closing speed: whichever vehicle is behind must be faster.
        if ego_pos < nbr_pos:
            closing = ego_speed - nbr_speed
        else:
            closing = nbr_speed - ego_speed
        if closing <= 0.1:
            continue
        best = min(best, d / closing)
    return best


def _eta_to_lane_end(pos: float, speed: float, pos_end: float) -> float:
    """Estimated time (s) to reach the lane end; +inf if already past it."""
    remaining = pos_end - pos
    if remaining < 0.0:
        return float("inf")
    return remaining / max(speed, 0.5)


def merge_arrival_pet(
    veh_id: str,
    radius: float = 150.0,
    merge_edge: str = MERGE_EDGE,
    dropped_lane: int = MERGE_DROPPED_LANE_INDEX,
    surviving_lane: int = MERGE_SURVIVING_LANE_INDEX,
    pos_end: float = MERGE_DROPPED_LANE_LENGTH,
) -> float:
    """Min post-encroachment time (s) at the lane-drop; ``+inf`` if no conflict.

    For the vehicle and each neighbor on the complementary converging lane
    (one on the dropped lane, one on the surviving lane, both on the merge
    edge), PET = |ETA_ego - ETA_nbr| where ETA is the estimated arrival time
    at the lane end. Small PET = the pair reaches the merge point together
    (arrival-coordination failure); large PET = healthy zipper sequencing.
    """
    ego = _vehicle_state(veh_id)
    if ego is None:
        return float("inf")
    ego_edge, ego_lane, ego_pos, ego_speed = ego
    if ego_edge != merge_edge or ego_lane not in (dropped_lane, surviving_lane):
        return float("inf")
    eta_ego = _eta_to_lane_end(ego_pos, ego_speed, pos_end)
    if eta_ego == float("inf"):
        return float("inf")

    best = float("inf")
    for nbr_id in vehicles_within_radius(veh_id, radius, same_edge_only=True):
        nbr = _vehicle_state(nbr_id)
        if nbr is None:
            continue
        nbr_edge, nbr_lane, nbr_pos, nbr_speed = nbr
        if {ego_lane, nbr_lane} != {dropped_lane, surviving_lane}:
            continue
        eta_nbr = _eta_to_lane_end(nbr_pos, nbr_speed, pos_end)
        if eta_nbr == float("inf"):
            continue
        best = min(best, abs(eta_ego - eta_nbr))
    return best


def leader_ttc_detailed(
    veh_id: str, look_ahead: float = DEFAULT_LOOK_AHEAD
) -> Tuple[float, float, float]:
    """Return ``(ttc, gap, leader_speed)``; ttc/gap are +inf with no leader."""
    info = leader_info(veh_id, look_ahead)
    if info is None:
        return float("inf"), float("inf"), 0.0
    _, gap, leader_speed = info
    v_ego = _ego_speed(veh_id)
    if v_ego is None:
        return float("inf"), gap, leader_speed
    closing = v_ego - leader_speed
    if closing <= 0.01:
        return float("inf"), gap, leader_speed
    return gap / closing, gap, leader_speed


# --- proximity scanners ---------------------------------------------------
def vehicles_within_radius(cav_id: str, radius: float, same_edge_only: bool = True) -> List[str]:
    """Nearby vehicle IDs within ``radius`` (longitudinal, same edge by default)."""
    snap = get_active_snapshot()
    if snap is not None:
        ego_edge = snap.road(cav_id)
        ego_pos = snap.lane_pos(cav_id)
        if ego_edge is None or ego_pos is None:
            return []
        nearby_snap: List[str] = []
        for veh_id in snap.ordered_ids():
            if veh_id == cav_id:
                continue
            veh_edge = snap.road(veh_id)
            veh_pos = snap.lane_pos(veh_id)
            if veh_edge is None or veh_pos is None:
                continue
            if same_edge_only and veh_edge != ego_edge:
                continue
            if abs(veh_pos - ego_pos) > radius:
                continue
            nearby_snap.append(veh_id)
        return nearby_snap

    try:
        ego_edge = traci.vehicle.getRoadID(cav_id)
        ego_pos = traci.vehicle.getLanePosition(cav_id)
    except Exception:
        return []
    nearby: List[str] = []
    for veh_id in traci.vehicle.getIDList():
        if veh_id == cav_id:
            continue
        try:
            if same_edge_only and traci.vehicle.getRoadID(veh_id) != ego_edge:
                continue
            if abs(traci.vehicle.getLanePosition(veh_id) - ego_pos) > radius:
                continue
        except Exception:
            continue
        nearby.append(veh_id)
    return nearby


def ahead_lane_speeds(
    cav_id: str,
    near: float = 150.0,
    far: float = 200.0,
) -> List[float]:
    """Speeds of vehicles in the ``[near, far]`` m band ahead on the same lane.

    Falls back to ``[0, far]`` if the primary band is empty (mirrors the
    flow-stabilization behaviour in the Shepherd action engine).
    """
    snap = get_active_snapshot()
    if snap is not None:
        cav_lane = snap.lane_id(cav_id)
        cav_pos = snap.lane_pos(cav_id)
        cav_edge = snap.road(cav_id)
        if cav_lane is None or cav_pos is None or cav_edge is None:
            return []
        primary_s: List[float] = []
        fallback_s: List[float] = []
        for veh_id in snap.ordered_ids():
            if veh_id == cav_id:
                continue
            veh_lane = snap.lane_id(veh_id)
            veh_edge = snap.road(veh_id)
            veh_pos = snap.lane_pos(veh_id)
            speed = snap.speed(veh_id)
            if veh_lane is None or veh_edge is None or veh_pos is None or speed is None:
                continue
            rel_pos = veh_pos - cav_pos
            if veh_lane != cav_lane and veh_edge != cav_edge:
                continue
            if rel_pos <= 0:  # only ahead
                continue
            if veh_lane == cav_lane:
                if near <= rel_pos <= far:
                    primary_s.append(speed)
                elif 0 < rel_pos <= far:
                    fallback_s.append(speed)
        return primary_s if primary_s else fallback_s

    try:
        cav_lane = traci.vehicle.getLaneID(cav_id)
        cav_pos = traci.vehicle.getLanePosition(cav_id)
        cav_edge = traci.vehicle.getRoadID(cav_id)
    except Exception:
        return []

    primary: List[float] = []
    fallback: List[float] = []
    for veh_id in traci.vehicle.getIDList():
        if veh_id == cav_id:
            continue
        try:
            veh_lane = traci.vehicle.getLaneID(veh_id)
            veh_edge = traci.vehicle.getRoadID(veh_id)
            rel_pos = traci.vehicle.getLanePosition(veh_id) - cav_pos
            speed = traci.vehicle.getSpeed(veh_id)
        except Exception:
            continue
        # Prefer same lane; allow same edge as a looser fallback.
        if veh_lane != cav_lane and veh_edge != cav_edge:
            continue
        if rel_pos <= 0:  # only ahead
            continue
        if veh_lane == cav_lane:
            if near <= rel_pos <= far:
                primary.append(speed)
            elif 0 < rel_pos <= far:
                fallback.append(speed)
    return primary if primary else fallback


# --- small numeric helpers (shared by reward + evaluation) ----------------
def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile. Empty input → NaN (caller decides meaning)."""
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = max(1, min(len(sorted_vals), math.ceil((p / 100.0) * len(sorted_vals))))
    return sorted_vals[rank - 1]


def percentile_or_zero(values: Sequence[float], p: float) -> float:
    """Like :func:`percentile` but ``0.0`` for empty input (reward semantics)."""
    if not values:
        return 0.0
    return percentile(values, p)
