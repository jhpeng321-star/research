"""Stopped-HDV incident obstacle: free-gap placement + traci insertion.

Registration section 3: at t0 a stopped HDV is placed on the mainline right
lane at s_down +40 m (A) or -150 m (B); if the slot is occupied, the nearest
free gap of >= 15 m within +/-20 m is used and the shift is logged.  The
vehicle is held stopped for the whole 300 s window.

The gap search is a pure function (unit-tested without SUMO); the traci
insertion is a thin, lazily-imported wrapper that the runner injects a fake
for in the tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    from . import constants as C
except ImportError:  # script execution
    import constants as C  # type: ignore


class ObstacleError(ValueError):
    """An incident-obstacle contract violation; never silently relaxed."""


def occupied_intervals(vehicles) -> list[tuple[float, float]]:
    """[(rear, front)] intervals from ``[(front_pos, length), ...]``."""
    intervals = []
    for front, length in vehicles:
        intervals.append((float(front) - float(length), float(front)))
    return intervals


def _overlaps(candidate: tuple[float, float],
              intervals: list[tuple[float, float]], min_gap: float) -> bool:
    rear, front = candidate
    for other_rear, other_front in intervals:
        if not (front + min_gap <= other_rear or rear - min_gap >= other_front):
            return True
    return False


def find_free_slot(vehicles, desired_front_m: float, *, length: float,
                   min_gap: float = C.OBSTACLE_MIN_GAP_M,
                   window: float = C.OBSTACLE_SEARCH_WINDOW_M,
                   step: float = 1.0) -> dict:
    """Nearest free placement for a ``length``-m obstacle at ``desired``.

    ``vehicles`` is ``[(front_pos_m, length_m), ...]`` on the lane.  The
    search offsets 0, +step, -step, +2*step, ... within +/-window and returns
    the first offset whose interval clears every vehicle by ``min_gap`` on
    both sides.  Raises :class:`ObstacleError` when no slot qualifies.
    """
    if length <= 0 or min_gap < 0 or window < 0:
        raise ObstacleError("invalid obstacle geometry")
    intervals = occupied_intervals(vehicles)
    offsets = [0.0]
    k = 1
    while k * step <= window + 1e-9:
        offsets.extend([k * step, -k * step])
        k += 1
    for offset in offsets:
        front = float(desired_front_m) + offset
        candidate = (front - float(length), front)
        if not _overlaps(candidate, intervals, float(min_gap)):
            return {
                "position_m": front,
                "shift_m": float(offset),
                "length_m": float(length),
                "min_gap_m": float(min_gap),
                "shifted": abs(offset) > 1e-9,
            }
    raise ObstacleError(
        f"no free gap of >= {min_gap} m within +/-{window} m of "
        f"{desired_front_m:.3f} m on the incident lane")


def incident_lane_id(s_down_m: float) -> str:
    """The SUMO lane id of the incident position (mainline right lane)."""
    if float(s_down_m) >= 0.0:
        return C.INCIDENT_LANE_EDGE_LANE["merge"]   # 1487606179_1
    return C.INCIDENT_LANE_EDGE_LANE["entry"]       # 1191554636_0


def incident_lane_base(s_down_m: float) -> float:
    """The lane's coordinate offset so ``lane_pos = s_down + base``.

    The merge right lane starts at the merge-edge start (s_down = 0); the
    entry right lane has origin s = 0, i.e. base = +2644.22.
    """
    if float(s_down_m) >= 0.0:
        return 0.0
    return C.MAIN_RIGHT_LANE_OFFSET_M


def _lane_vehicles(api, lane_id: str) -> list[tuple[float, float]]:
    out = []
    for vid in api.vehicle.getLastStepVehicleIDs(lane_id):
        out.append((float(api.vehicle.getLanePosition(vid)),
                    float(api.vehicle.getLength(vid))))
    return out


def place_obstacle(api, *, lane_id: str, desired_s_down_m: float,
                   s_down_base_m: float, obstacle_id: str,
                   route_id: str = "mainline_route",
                   type_id: str = C.OBSTACLE_TYPE,
                   length: float = C.OBSTACLE_LENGTH_M,
                   min_gap: float = C.OBSTACLE_MIN_GAP_M,
                   window: float = C.OBSTACLE_SEARCH_WINDOW_M) -> dict:
    """Find the free slot and insert the stopped obstacle via traci.

    ``desired_s_down_m`` is merge-relative; ``s_down_base_m`` is the lane's
    downstream origin (0 for the entry/merge right lane), so the traci
    lane position is ``desired_s_down_m + s_down_base_m``.
    """
    desired_lane_pos = float(desired_s_down_m) + float(s_down_base_m)
    slot = find_free_slot(_lane_vehicles(api, lane_id), desired_lane_pos,
                          length=length, min_gap=min_gap, window=window)
    api.vehicle.add(obstacle_id, route_id, type_id=type_id,
                    departPos=str(slot["position_m"]), departSpeed="0")
    if hasattr(api.vehicle, "setStop"):
        api.vehicle.setStop(obstacle_id, lane_id,
                            pos=slot["position_m"], duration=C.OBSTACLE_HOLD_S)
    if hasattr(api.vehicle, "moveTo"):
        api.vehicle.moveTo(obstacle_id, lane_id, slot["position_m"])
    if hasattr(api.vehicle, "setSpeed"):
        api.vehicle.setSpeed(obstacle_id, 0.0)
    record = dict(slot)
    record.update({
        "obstacle_id": obstacle_id,
        "lane_id": lane_id,
        "desired_s_down_m": float(desired_s_down_m),
        "desired_lane_pos_m": desired_lane_pos,
        "s_down_base_m": float(s_down_base_m),
        "hold_s": C.OBSTACLE_HOLD_S,
    })
    return record


def hold_obstacle(api, obstacle_id: str) -> None:
    """Pin the obstacle at 0 m/s on the current physics tick."""
    api.vehicle.setSpeed(obstacle_id, 0.0)
