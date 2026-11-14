"""WM-59 wall REQUEST rule v2.2 'merge-release': ONLINE feasibility from
CAV report frames only.

The wall pilot's planner side (owner spec OWNER_SPEC_V22_20261001.md,
owner-confirmed 2026-10-01, BINDING): a feasible pair is >= 2 live CAVs
on the upstream mainline ENTRY edge 1191554636 ONLY (OPTION A: the exit
edge is never used for members), in DIFFERENT lanes, with longitudinal
span <= ``REQUEST_SPAN_CAP_M`` (75 m - the 2026-09-30 A2 revert to the
wall_group.MAX_WALL_SPAN_M cap), and - once the wall speed is known -
with the LEAD member's distance to the MERGE START
(``MERGE_START_S - s_road(lead)``) >= ``D_min``:

    D_min = X + v_wall * (t_assemble + H_min)
          = 0   + v_wall * (10 s        + 60 s) = 70 * v_wall   [m]

(v2.2 law: the wall releases at the merge START, so the road past the
merge is irrelevant to the request; the a-priori ``t_assemble`` is the
J4 assembly-time MEDIAN 10 s (147-request J4 geometry, no outcomes) and
``H_min`` is the smallest registered fixed-arm hold (60 s - one planner
decision interval).  This SUPERSEDES the v2.1 A4 rule
``road_left >= v_wall x 80 s`` and with it the road-end release the A4
rule protected; it also supersedes the wallrt eee755f edge-aware
``road_left_m`` repair, whose per-edge formula is subsumed by the
cumulative edge-aware coordinate ``s_road``.)  Everything here is
evaluated from the SensorFrame-like report objects the dispatcher
already consumes (``traffic_shepherd.wall_group`` report-content-only
contract) - never from a physics snapshot, never from truth.

Member choice: the feasible pair of MINIMAL span; ties broken by the
MOST merge distance (the lead member's), then lexicographic member ids
- one member per lane, ordered by lane index (the ``WallAssignment``
lane encoding).  ``s_road`` is the edge-aware cumulative road
coordinate (entry ``position_m``; merge 2644.2 + position; exit 2719.6
+ position; internal edges via the merge-distance coordinate), so the
span and the D_min distance are correct across edges.

Wall speed: ``factor x the members' mean reported speed at request``,
floored at ``SPEED_FLOOR_MPS`` = 2.5 m/s (always strictly above the
dispatcher's ``SAFE_SPEED_MIN_MPS`` = 2.0 safety floor).

Request timing: early = the first branch tick >= t0 with a feasible pair;
late = the first feasible tick >= t0 + 60.  If no feasible pair appears
within ``REQUEST_WINDOW_S`` (120 s) after the target time, the branch is
recorded ``not_feasible`` and continues native (a pair that exists but
fails D_min is refused at the proposal with the logged
``dmin_unreachable`` reason - see ``dmin_refusal``).

SMOKE-ONLY wide-pair window: ``feasible_pairs``/``choose_pair`` accept an
optional ``min_span_m`` (the WIDE-PAIR smoke mode passes 40 m) so the
request waits for a pair whose span lands in [min_span_m,
REQUEST_SPAN_CAP_M] instead of taking the minimal-span pair.  Real
branches never pass it (the driver refuses it without the smoke horizon
cap).
"""
from __future__ import annotations

from math import isfinite

from campaign_walls_v1.common import (
    ARM_SPEED_FACTORS,
    ARM_TIMINGS,
    ASSEMBLY_TIME_ASSUMED_S,
    MEMBER_EDGES,
    MERGE_RELEASE_POINT_S,
    MERGE_START_S,
    MIN_MEANINGFUL_HOLD_S,
    MERGE_RELEASE_X_M,
    NETWORK_END_S,
    REQUEST_SPAN_CAP_M,
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
    "s_road",
    "road_left_m",
    "min_proposal_distance_m",
    "hold_cap_s",
    "dmin_refusal",
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


def s_road(frame):
    """Edge-aware cumulative road coordinate of a report frame (single
    source of truth: ``traffic_shepherd.wall_group.frame_s_road``, the
    v2.2 coordinate the release point and every span comparison run on).
    None when the frame carries neither a mainline edge position nor a
    finite merge-distance coordinate."""
    from traffic_shepherd.wall_group import frame_s_road  # noqa: PLC0415

    return frame_s_road(frame)


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


def min_proposal_distance_m(wall_speed_mps: float) -> float:
    """The v2.2 proposer minimum ``D_min`` (owner binding):

        D_min = X + v_wall * t_assemble + v_wall * H_min

    with the owner override X = ``MERGE_RELEASE_X_M`` = 0.0 m, the
    a-priori J4 assembly time ``ASSEMBLY_TIME_ASSUMED_S`` = 10 s (the
    J4 median, mechanics-only, no outcome) and the minimal meaningful
    hold ``MIN_MEANINGFUL_HOLD_S`` = 60 s (one 60 s planner decision
    interval = the smallest registered fixed-arm draw).  A request whose
    LEAD member is closer to the merge start than D_min cannot assemble
    AND hold meaningfully before the merge -> refused at the proposal.
    """
    speed = float(wall_speed_mps)
    if not isfinite(speed) or speed <= 0:
        raise FeasibilityError("wall_speed_mps must be finite and positive")
    return round(float(MERGE_RELEASE_X_M)
                 + speed * (ASSEMBLY_TIME_ASSUMED_S + MIN_MEANINGFUL_HOLD_S),
                 6)


def hold_cap_s(hold_s: float, lead_s_road_m: float,
               wall_speed_mps: float) -> float:
    """The v2.2 hold cap (owner binding: the hold ALWAYS ends at the
    merge start at the latest): the registered draw ``hold_s`` capped by
    the time the lead member needs to reach the release point at the
    wall speed, ``(MERGE_RELEASE_POINT_S - s_road(lead)) / v_wall``.
    Registered draws stay recorded as drawn; this is the a-priori
    effective hold used for the cap logging (the session's
    ``merge_release`` enforces the same cap on the live positions)."""
    speed = float(wall_speed_mps)
    if not isfinite(speed) or speed <= 0:
        raise FeasibilityError("wall_speed_mps must be finite and positive")
    to_release = max(0.0, float(MERGE_RELEASE_POINT_S)
                     - float(lead_s_road_m))
    return round(min(float(hold_s), to_release / speed), 6)


def road_left_m(frame):
    """Road ahead of a member up to the NETWORK END, edge-aware (v2.2):
    ``NETWORK_END_S - s_road(frame)``.  Supersedes the v2.1 entry-edge
    assumption AND the wallrt eee755f per-edge repair (both subsumed by
    ``s_road``).  Kept for telemetry comparability with the J4 rows; the
    v2.2 request rule gates on the LEAD member's distance to the MERGE
    START, not on road_left.  None when the frame has no coordinate."""
    s = s_road(frame)
    if s is None:
        return None
    return round(float(NETWORK_END_S) - float(s), 6)


def _candidates(frames: dict) -> dict:
    """Entry-edge candidates keyed by (edge, lane index), from report
    frames only (edge from the lane id, coordinate = s_road)."""
    by_edge_lane: dict[tuple, list[dict]] = {}
    for vehicle_id, frame in frames.items():
        if frame is None:
            continue
        edge = edge_of_lane(getattr(frame, "lane_id", None))
        if edge is None or edge not in MEMBER_EDGES:
            continue
        s = s_road(frame)
        if s is None:
            continue
        lane = getattr(frame, "lane_index", None)
        if lane is None:
            continue
        by_edge_lane.setdefault((edge, int(lane)), []).append({
            "vehicle_id": str(vehicle_id),
            "s_road_m": float(s),
            "speed_mps": float(getattr(frame, "speed_mps", 0.0) or 0.0),
            "lane_index": int(lane),
        })
    return by_edge_lane


def _pairs_from_candidates(by_edge_lane: dict,
                           factor: float | None) -> tuple[list[dict], dict]:
    """All span-feasible pairs (best first) with their D_min verdict;
    returns (pairs, refusal) where ``refusal`` describes the best pair
    that failed ONLY the D_min gate (empty dict when none)."""
    pairs: list[dict] = []
    best_refusal: dict = {}
    edges = sorted({edge for (edge, _lane) in by_edge_lane})
    for edge in edges:
        lane_ids = sorted(lane for (e, lane) in by_edge_lane if e == edge)
        if len(lane_ids) < 2:
            continue  # a 2-lane cross-section needs two distinct lanes
        for index_a, lane_a in enumerate(lane_ids):
            for lane_b in lane_ids[index_a + 1:]:
                for a in by_edge_lane[(edge, lane_a)]:
                    for b in by_edge_lane[(edge, lane_b)]:
                        span = abs(a["s_road_m"] - b["s_road_m"])
                        if span > float(REQUEST_SPAN_CAP_M) + 1e-9:
                            continue
                        members = sorted([a, b], key=lambda m: m["lane_index"])
                        lead_s = max(a["s_road_m"], b["s_road_m"])
                        lead_distance = float(MERGE_START_S) - lead_s
                        speeds = {m["vehicle_id"]: m["speed_mps"]
                                  for m in members}
                        speed = (wall_speed_for(factor, speeds)
                                 if factor is not None else None)
                        dmin = (min_proposal_distance_m(speed)
                                if speed is not None else None)
                        pair = {
                            "edge": edge,
                            "members": tuple(m["vehicle_id"] for m in members),
                            "lanes": [m["lane_index"] for m in members],
                            "span_m": round(span, 6),
                            "mean_distance_to_merge_m": round(
                                float(MERGE_START_S)
                                - (a["s_road_m"] + b["s_road_m"]) / 2.0, 6),
                            "road_left_m": round(float(NETWORK_END_S)
                                                 - lead_s, 6),
                            "lead_s_road_m": round(lead_s, 6),
                            "lead_distance_to_merge_m": round(lead_distance,
                                                              6),
                            "member_speeds": speeds,
                            "wall_speed_mps": speed,
                            "dmin_m": dmin,
                        }
                        if (factor is not None
                                and lead_distance < float(dmin) - 1e-9):
                            # v2.2 D_min refusal: assembly + a minimal
                            # meaningful hold cannot complete before the
                            # merge start -> refused at the proposal.
                            if (not best_refusal
                                    or lead_distance > best_refusal[
                                        "lead_distance_to_merge_m"]):
                                best_refusal = {
                                    "reason": "dmin_unreachable",
                                    "edge": edge,
                                    "members": list(pair["members"]),
                                    "span_m": pair["span_m"],
                                    "lead_s_road_m": pair["lead_s_road_m"],
                                    "lead_distance_to_merge_m":
                                        pair["lead_distance_to_merge_m"],
                                    "dmin_m": dmin,
                                    "wall_speed_mps": speed,
                                }
                            continue
                        pairs.append(pair)
    pairs.sort(key=lambda p: (p["span_m"], -p["lead_distance_to_merge_m"],
                              p["members"]))
    return pairs, best_refusal


def feasible_pairs(frames: dict, factor: float | None = None,
                   min_span_m: float | None = None) -> list[dict]:
    """All feasible pairs from the report frames, best first.

    A pair: two live CAVs whose frames place them on the upstream entry
    edge 1191554636, in DIFFERENT lanes, with |s_road_a - s_road_b| <=
    REQUEST_SPAN_CAP_M (75 m; the 2026-09-30 A2 revert of the 6 m
    tightening), and - with a wall-speed ``factor`` - the v2.2 owner
    rule: the LEAD member's distance to the merge start
    (MERGE_START_S - s_road(lead)) >= D_min = v_wall x
    (ASSEMBLY_TIME_ASSUMED_S + MIN_MEANINGFUL_HOLD_S)  (assembly + a
    minimal meaningful hold must fit before the merge; supersedes the
    v2.1 A4 road-left rule).  ``min_span_m`` (SMOKE-ONLY wide-pair
    window) additionally requires span >= min_span_m, so the request
    selects a WIDE pair instead of the minimal-span one.  Frames
    without a lane id on the entry edge or without a finite s_road
    coordinate are never candidates (conservative: the request must be
    provably feasible).  Order: minimal span, then most lead merge
    distance, then member ids.
    """
    pairs, _refusal = _pairs_from_candidates(_candidates(frames), factor)
    if min_span_m is not None:
        pairs = [p for p in pairs
                 if p["span_m"] >= float(min_span_m) - 1e-9]
    return pairs


def dmin_refusal(frames: dict, factor: float) -> dict | None:
    """The v2.2 proposal refusal reason, or None.

    When lane-distinct pairs within the span cap exist RIGHT NOW but
    every one of them fails the D_min gate (the wall could not assemble
    AND hold meaningfully before the merge start), returns the best
    such pair's refusal record (``reason="dmin_unreachable"`` with the
    lead merge distance, D_min and the wall speed) for the driver's
    logged refusal.  None when there is no span-feasible pair at all
    (the plain request-window ``not_feasible`` applies)."""
    if factor is None:
        return None
    _pairs, refusal = _pairs_from_candidates(_candidates(frames), factor)
    return refusal or None


def choose_pair(frames: dict, factor: float | None = None,
                min_span_m: float | None = None) -> dict | None:
    """The minimal-span feasible pair (ties: most lead merge distance),
    or None.  ``min_span_m`` restricts the search to WIDE pairs
    (smoke-only)."""
    pairs = feasible_pairs(frames, factor, min_span_m=min_span_m)
    return pairs[0] if pairs else None


def build_wall_plan(*, run_name: str, cav_ids, pair: dict, factor: float,
                    hold_s: float, now_s: float, plan_factory=None,
                    phase: str = "assemble", wall_speed_mps: float | None = None):
    """The strict-roster dispatcher plan that requests the wall.

    Members are assigned ``wall-v2`` and carry the single ``WallAssignment``
    (phase ``assemble``, hold_s drawn from the branch seed - recorded AS
    DRAWN; the v2.2 merge-start cap is enforced at run time by the wall
    session's ``merge_release`` and logged by the driver, never
    pre-shrunk here); every other roster CAV is named ``native``
    (non-members NATIVE - the owner's pairing decision).
    ``plan_factory`` defaults to the frozen ``AssignmentPlan``
    (injectable in tests).
    """
    members = tuple(str(v) for v in pair["members"])
    if len(members) < 2 or len(set(members)) != len(members):
        raise FeasibilityError(f"infeasible member set {members!r}")
    roster = [str(v) for v in cav_ids]
    unknown = [v for v in members if v not in roster]
    if unknown:
        raise FeasibilityError(f"wall members outside the roster: {unknown}")
    speed = (float(wall_speed_mps) if wall_speed_mps is not None
             else wall_speed_for(factor, pair["member_speeds"]))
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
        plan_id=f"{run_name}:{'wall' if phase == 'assemble' else phase}@{now_s:g}",
        commit_horizon_s=0.0,
        walls=(WallAssignment(wall_id=wall_id, members=members,
                              wall_speed_mps=speed, phase=phase,
                              hold_s=float(hold_s)),),
        strict_roster=True,
    ), wall_id, speed
