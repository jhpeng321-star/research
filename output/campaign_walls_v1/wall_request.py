"""WM-59 wall REQUEST rule: ONLINE feasibility from CAV report frames only.

The wall pilot's planner side (owner rules, PLAN_DRAFT 1.2/1.3): a
feasible pair is >= 2 live CAVs on the SAME 2-lane mainline edge
(1191554636 or 1487606178), in DIFFERENT lanes, with longitudinal span
<= ``wall_group.MAX_WALL_SPAN_M`` (75 m).  Everything here is evaluated
from the SensorFrame-like report objects the dispatcher already consumes
(``traffic_shepherd.wall_group`` report-content-only contract) - never
from a physics snapshot, never from truth.

Member choice: the feasible pair nearest upstream of the merge - the
smallest MEAN ``distance_to_merge_m`` of the pair (the registered sensor
coordinate counts DOWN toward the merge, so smaller = nearer; ties by
smaller span, then lexicographic member ids) - one member per lane,
ordered by lane index (the ``WallAssignment`` lane encoding).

Wall speed: ``factor x the members' mean reported speed at request``,
floored at ``SPEED_FLOOR_MPS`` = 2.5 m/s (always strictly above the
dispatcher's ``SAFE_SPEED_MIN_MPS`` = 2.0 safety floor).

Request timing: early = the first branch tick >= t0 with a feasible pair;
late = the first feasible tick >= t0 + 60.  If no feasible pair appears
within ``REQUEST_WINDOW_S`` (120 s) after the target time, the branch is
recorded ``not_feasible`` and continues native.
"""
from __future__ import annotations

from math import isfinite

from campaign_walls_v1.common import (
    ARM_SPEED_FACTORS,
    ARM_TIMINGS,
    MAINLINE_2LANE_EDGES,
    REQUEST_WINDOW_S,
    SPEED_FLOOR_MPS,
    TIMING_OFFSET_S,
    WallsPilotError,
    arm_speed_factor,
    arm_timing,
    wall_id_for,
)

#: re-exported so the driver and the tests share one import surface
__all__ = [
    "arm_speed_factor",
    "arm_timing",
    "wall_id_for",
    "edge_of_lane",
    "target_and_deadline",
    "wall_speed_for",
    "feasible_pairs",
    "choose_pair",
    "build_wall_plan",
    "FeasibilityError",
]


class FeasibilityError(WallsPilotError):
    """A wall-request rule violation (never silently relaxed)."""


def edge_of_lane(lane_id) -> str | None:
    """The SUMO edge id encoded in a lane id ("1191554636_0" ->
    "1191554636"); None for internal (junction) lanes and lane-less
    frames.  The report-only edge identity: SUMO lane ids are
    ``<edge_id>_<lane_index>`` and internal lanes start with ":"."""
    if not isinstance(lane_id, str) or not lane_id:
        return None
    if lane_id.startswith(":"):
        return None
    edge, sep, index = lane_id.rpartition("_")
    if not sep or not index.isdigit():
        return None
    return edge


def target_and_deadline(t0_s: float, timing: str) -> tuple[float, float]:
    """(target_time_s, deadline_s) for the arm's request timing.

    early: target t0; late: target t0 + 60.  The deadline is the target
    plus the 120 s feasibility window; the driver records ``not_feasible``
    at the first tick strictly after it.
    """
    if timing not in ARM_TIMINGS:
        raise FeasibilityError(f"unknown request timing {timing!r}")
    target = float(t0_s) + TIMING_OFFSET_S[timing]
    return target, target + REQUEST_WINDOW_S


def wall_speed_for(factor: float, member_speeds: dict) -> float:
    """factor x the members' mean reported speed, floored at 2.5 m/s."""
    if factor not in ARM_SPEED_FACTORS.values():
        raise FeasibilityError(f"unknown wall speed factor {factor!r}")
    speeds = [float(s) for s in member_speeds.values()]
    if not speeds or any(not isfinite(s) or s < 0 for s in speeds):
        raise FeasibilityError("member speeds must be finite and nonnegative")
    mean = sum(speeds) / len(speeds)
    return round(max(SPEED_FLOOR_MPS, factor * mean), 6)


def feasible_pairs(frames: dict) -> list[dict]:
    """All feasible pairs from the report frames, nearest-the-merge first.

    A pair: two live CAVs whose frames place them on the SAME 2-lane
    mainline edge, in DIFFERENT lanes, with |s_a - s_b| <=
    MAX_WALL_SPAN_M (75 m; ``wall_group.downstream_position_m``'s
    s_m = -distance_to_merge_m coordinate).  Frames without a lane id on
    a mainline edge or without a finite distance_to_merge_m are never
    candidates (conservative: the request must be provably feasible).
    """
    from traffic_shepherd.wall_group import (  # noqa: PLC0415 - frozen lib
        MAX_WALL_SPAN_M,
        downstream_position_m,
    )

    by_edge_lane: dict[tuple, list[dict]] = {}
    for vehicle_id, frame in frames.items():
        if frame is None:
            continue
        edge = edge_of_lane(getattr(frame, "lane_id", None))
        if edge is None or edge not in MAINLINE_2LANE_EDGES:
            continue
        s_m = downstream_position_m(frame)
        if s_m is None:
            continue
        lane = getattr(frame, "lane_index", None)
        if lane is None:
            continue
        by_edge_lane.setdefault((edge, int(lane)), []).append({
            "vehicle_id": str(vehicle_id),
            "s_m": float(s_m),
            "distance_to_merge_m": -float(s_m),
            "speed_mps": float(getattr(frame, "speed_mps", 0.0) or 0.0),
            "lane_index": int(lane),
        })
    pairs: list[dict] = []
    edges = sorted({edge for (edge, _lane) in by_edge_lane})
    for edge in edges:
        lane_ids = sorted(lane for (e, lane) in by_edge_lane if e == edge)
        if len(lane_ids) < 2:
            continue  # a 2-lane cross-section needs two distinct lanes
        for index_a, lane_a in enumerate(lane_ids):
            for lane_b in lane_ids[index_a + 1:]:
                for a in by_edge_lane[(edge, lane_a)]:
                    for b in by_edge_lane[(edge, lane_b)]:
                        span = abs(a["s_m"] - b["s_m"])
                        if span > float(MAX_WALL_SPAN_M):
                            continue
                        members = sorted([a, b], key=lambda m: m["lane_index"])
                        pairs.append({
                            "edge": edge,
                            "members": tuple(m["vehicle_id"] for m in members),
                            "lanes": [m["lane_index"] for m in members],
                            "span_m": round(span, 6),
                            "mean_distance_to_merge_m": round(
                                (a["distance_to_merge_m"]
                                 + b["distance_to_merge_m"]) / 2.0, 6),
                            "member_speeds": {m["vehicle_id"]: m["speed_mps"]
                                              for m in members},
                        })
    pairs.sort(key=lambda p: (p["mean_distance_to_merge_m"], p["span_m"],
                              p["members"]))
    return pairs


def choose_pair(frames: dict) -> dict | None:
    """The feasible pair nearest upstream of the merge, or None."""
    pairs = feasible_pairs(frames)
    return pairs[0] if pairs else None


def build_wall_plan(*, run_name: str, cav_ids, pair: dict, factor: float,
                    hold_s: float, now_s: float, plan_factory=None):
    """The strict-roster dispatcher plan that requests the wall.

    Members are assigned ``wall-v2`` and carry the single ``WallAssignment``
    (phase ``assemble``, hold_s drawn from the branch seed); every other
    roster CAV is named ``native`` (non-members NATIVE - the owner's
    pairing decision).  ``plan_factory`` defaults to the frozen
    ``AssignmentPlan`` (injectable in tests).
    """
    members = tuple(str(v) for v in pair["members"])
    if len(members) < 2 or len(set(members)) != len(members):
        raise FeasibilityError(f"infeasible member set {members!r}")
    roster = [str(v) for v in cav_ids]
    unknown = [v for v in members if v not in roster]
    if unknown:
        raise FeasibilityError(f"wall members outside the roster: {unknown}")
    speed = wall_speed_for(factor, pair["member_speeds"])
    if plan_factory is None:
        from traffic_shepherd.mixed_assignment import (  # noqa: PLC0415
            AssignmentPlan,
        )
        plan_factory = AssignmentPlan
    from traffic_shepherd.mixed_assignment import WallAssignment  # noqa: PLC0415

    wall_id = wall_id_for(run_name)
    assignments = {v: "wall-v2" for v in members}
    assignments.update({v: "native" for v in roster if v not in members})
    return plan_factory(
        assignments=assignments,
        plan_id=f"{run_name}:wall@{now_s:g}",
        commit_horizon_s=0.0,
        walls=(WallAssignment(wall_id=wall_id, members=members,
                              wall_speed_mps=speed, phase="assemble",
                              hold_s=float(hold_s)),),
        strict_roster=True,
    ), wall_id, speed
