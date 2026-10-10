"""Wall law v2.2 'merge-release' (WALL22-I integration of REGISTRATION_WALL_V22).

Law selection: the pinned module :mod:`traffic_shepherd.wall_group` stays
at ``wall-v2.1-closure`` byte-for-byte (J4 reproducibility); THIS module
adds law v2.2 ALONGSIDE it.  A v2.2 wall is requested by setting
``law_version="wall-v2.2-merge-release"`` on the dispatcher's
``WallAssignment`` (``traffic_shepherd.mixed_assignment``); ``apply_plan``
then instantiates :class:`WallGroupSessionV22` instead of the pinned
``WallGroupSession``.  Every existing caller that does not name a law
version keeps the pinned v2.1 behaviour unchanged.

Law v2.2 'merge-release' (owner-confirmed 2026-10-01, BINDING; the
verbatim owner text and the registered constants live in
REGISTRATION_WALL_V22.md sections 1-2; design note
DESIGN_V22_MERGE_RELEASE.md; zero SUMO was spent here):

1. RELEASE POINT = the START OF THE MERGE (X = 0 m before the merge edge
   start).  In HOLD only, the session fires ``merge_release`` exactly once
   when the LEAD member's edge-aware cumulative coordinate ``s_road``
   reaches ``MERGE_RELEASE_POINT_S`` (= 2644.2 m), handing the members
   back to native that tick - a RELEASE kind, never an abort.
2. HOLD = planner decides, CAPPED: the dispatcher/planner may keep or
   release earlier, but the wall ALWAYS hands back at the merge start at
   the latest (the session's ``merge_release`` is the hard cap).
3. SHAPE = unchanged v2.1-closure assembly/hold pacing (one member per
   lane, 2 abreast, bounded absolute acceleration).
4. Defensive rule: an ASSEMBLY whose lead reaches the release point aborts
   ``assembly_past_release`` (never a merge release) - the v2.2 proposer's
   D_min gate (``campaign_walls_v1.wall_request``) refuses such requests
   a priori; this is the hard guarantee.

Positions run on ``frame_s_road`` (edge-aware cumulative road coordinate
from the entry edge origin) instead of the edge-blind
``downstream_position_m``: span, rear surveillance and the release point
stay correct across the entry/merge/exit edges.  This subsumes the wallrt
eee755f per-edge ``road_left_m`` repair; the v2.1 road-end release is
superseded (the driver retires its outcome outside the v2.2 outcome set).

Report content only: ``tick`` consumes SensorFrame-like objects and never
sees a physics snapshot or truth.
"""
from __future__ import annotations

from math import isfinite
from typing import Mapping

from traffic_shepherd import wall_group
from traffic_shepherd.wall_group import (
    ASSEMBLY_TIMEOUT_S,
    PHASE_ASSEMBLY,
    PHASE_HANDBACK,
    PHASE_HOLD,
    assembly_target_mps,
    _pace,
)

#: law v2.2 'merge-release': the wall meters inflow UPSTREAM of the merge
#: and hands back to native at the merge START (owner-confirmed 2026-10-01).
WALL_LAW_VERSION = "wall-v2.2-merge-release"

# --- merge-release geometry (cumulative s_road from the entry edge origin) --
ENTRY_EDGE_ID = "1191554636"
MERGE_EDGE_ID = "1487606179"
EXIT_EDGE_ID = "1487606178"
ENTRY_EDGE_LENGTH_M = 2644.2
MERGE_EDGE_LENGTH_M = 75.4
EXIT_EDGE_LENGTH_M = 477.9
MERGE_START_S = ENTRY_EDGE_LENGTH_M                    # 2644.2 m
MERGE_END_S = MERGE_START_S + MERGE_EDGE_LENGTH_M      # 2719.6 m
NETWORK_END_S = MERGE_END_S + EXIT_EDGE_LENGTH_M       # 3197.5 m
#: owner binding: release X = 0 m before the merge edge start.
MERGE_RELEASE_X_M = 0.0
MERGE_RELEASE_POINT_S = MERGE_START_S - MERGE_RELEASE_X_M
#: defensive abort: assembly must never carry the lead member to the release
#: point (the proposer refuses such requests; this guarantees it even if the
#: reports drift between proposal and assembly).
REASON_ASSEMBLY_PAST_RELEASE = "assembly_past_release"


def edge_of_lane(lane_id) -> str | None:
    """SUMO edge id encoded in a lane id ("1191554636_0" -> "1191554636");
    None for internal (junction) lanes and lane-less frames."""
    if not isinstance(lane_id, str) or not lane_id:
        return None
    if lane_id.startswith(":"):
        return None
    edge, sep, index = lane_id.rpartition("_")
    if not sep or not index.isdigit():
        return None
    return edge


def frame_s_road(frame) -> float | None:
    """Edge-aware cumulative road coordinate s_road of a report frame.

    Uses the frame's CURRENT edge (``edge_id`` property or the edge parsed
    from ``lane_id``) and its lane position; falls back to the registered
    ``distance_to_merge_m`` coordinate for internal/unknown edges.  This is
    the v2.2 replacement for the edge-blind ``position_m`` assumption: a
    member on the merge or exit edge is located correctly instead of being
    read as if it were still on the 2644.2 m entry edge.
    """
    edge = getattr(frame, "edge_id", None)
    if edge is None:
        edge = edge_of_lane(getattr(frame, "lane_id", None))
    position = getattr(frame, "position_m", None)
    if position is not None and isfinite(float(position)):
        if edge == ENTRY_EDGE_ID:
            return float(position)
        if edge == MERGE_EDGE_ID:
            return MERGE_START_S + float(position)
        if edge == EXIT_EDGE_ID:
            return MERGE_END_S + float(position)
    distance = getattr(frame, "distance_to_merge_m", None)
    if distance is not None and isfinite(float(distance)):
        return MERGE_START_S - float(distance)
    return None


class WallGroupSessionV22(wall_group.WallGroupSession):
    """The law v2.2 'merge-release' wall session.

    Identical ownership/lifecycle contract to the pinned
    ``WallGroupSession`` (assembly -> hold -> handback, bounded absolute
    acceleration, safety aborts in every phase, exclusive member
    ownership); the differences are exactly the registered v2.2 mechanics:
    the edge-aware ``s_road`` coordinate everywhere, the HOLD-only
    ``merge_release`` handback at ``MERGE_RELEASE_POINT_S``, and the
    ``assembly_past_release`` defensive abort.  Wall events log this
    module's ``WALL_LAW_VERSION``.
    """

    def __init__(self, wall_id: str, *, members: frozenset,
                 wall_speed_mps: float):
        super().__init__(wall_id, members=members,
                         wall_speed_mps=wall_speed_mps)
        self.law_version = WALL_LAW_VERSION

    # ------------------------------------------------------------------ api
    def tick(self, frames: Mapping, now_s: float, *,
             control_dt_s: float = 1.0) -> dict:
        """One control tick (v2.2 law): returns ``{member: absolute
        acceleration m/s^2}`` wall commands, applying the declared safety
        aborts (every phase), the ``assembly_past_release`` defence, the
        assembly -> hold transition and the HOLD-only ``merge_release``
        handback.  Input is report content only."""
        commands: dict[str, float] = {}
        if self.phase == PHASE_HANDBACK:
            return commands
        abort_reason = self._safety_abort(frames)
        if abort_reason is not None:
            self.abort(abort_reason, now_s)
            return commands
        speeds = {v: float(frames[v].speed_mps) for v in self.members}
        positions, _position_error = self._member_positions(frames)
        if positions is None:
            # unreachable: _safety_abort already returned the declared
            # member_position_unavailable abort before this point
            return commands
        if self.phase == PHASE_ASSEMBLY:
            if self.assembly_since is None:
                self.assembly_since = now_s
                self.events.append({"kind": "wall_event", "wall_id": self.wall_id,
                                    "event": "assembly_start", "timestamp_s": now_s,
                                    "wall_speed_mps": self.wall_speed_mps,
                                    "law_version": WALL_LAW_VERSION})
            if now_s - self.assembly_since >= ASSEMBLY_TIMEOUT_S:
                self.abort("assembly_timeout", now_s)
                return commands
            lane_probe = self._lane_coverage(frames)
            if lane_probe is not None and len(lane_probe) != len(self.members):
                self.abort("lane_coverage_violation", now_s)
                return commands
            front_pos = max(positions.values())
            # v2.2 defensive rule: assembly must never carry the lead member
            # to the merge release point - abort (the proposer already
            # refuses such requests, this is the hard guarantee).
            if front_pos >= MERGE_RELEASE_POINT_S - 1e-9:
                self.abort(REASON_ASSEMBLY_PAST_RELEASE, now_s)
                return commands
            for v, speed in speeds.items():
                # wall_v2 pacing: a = clip((v_target - v)/dt, -3, +1.5) toward
                # the closure target (front: v_wall; rear: v_wall + Delta)
                commands[v] = _pace(
                    assembly_target_mps(self.wall_speed_mps, front_pos - positions[v]),
                    speed, control_dt_s)
            if self._assembly_complete(speeds, positions):
                self.phase = PHASE_HOLD
                self.assembled_at_s = now_s
                self.events.append({"kind": "wall_event", "wall_id": self.wall_id,
                                    "event": "assembled", "timestamp_s": now_s,
                                    "members": sorted(self.members),
                                    "wall_speed_mps": self.wall_speed_mps,
                                    "law_version": WALL_LAW_VERSION})
        elif self.phase == PHASE_HOLD:
            # v2.2 MERGE RELEASE: in HOLD only, fire exactly once when the
            # LEAD member (max s_road) reaches the release point, and hand
            # the members back to native at the merge start.  This is the
            # hard cap on the hold (owner binding).
            if max(positions.values()) >= MERGE_RELEASE_POINT_S - 1e-9:
                self.merge_release(now_s)
                return commands
            for v, speed in speeds.items():
                commands[v] = _pace(self.wall_speed_mps, speed, control_dt_s)
        return commands

    def merge_release(self, now_s: float) -> bool:
        """v2.2 MERGE-RELEASE: a distinct HOLD-only handback at the merge
        release point.  Never valid in assembly (that is the declared
        ``assembly_past_release`` abort); fires once."""
        if self.phase != PHASE_HOLD:
            return False
        self.phase = PHASE_HANDBACK
        self.events.append({"kind": "wall_event", "wall_id": self.wall_id,
                            "event": "merge_release", "timestamp_s": now_s,
                            "release_kind": "merge_release",
                            "release_point_s": MERGE_RELEASE_POINT_S,
                            "merge_start_s": MERGE_START_S,
                            "handback": "native"})
        return True

    # -------------------------------------------------------------- helpers
    def _member_positions(self, frames: Mapping):
        """Member s_road positions (edge-aware cumulative coordinate; see
        frame_s_road).  Returns (positions, None) or (None,
        "member_position_unavailable"): a member without a finite
        coordinate aborts the wall LOUDLY.  v2.2 uses the edge-aware
        coordinate, so the span and the merge-release point are correct
        across the entry/merge/exit edges (the old edge-blind
        ``position_m`` read a member on the exit edge as if it were still
        on the entry edge)."""
        positions = {}
        for v in self.members:
            frame = frames.get(v)
            s = frame_s_road(frame)
            if s is None:
                return None, "member_position_unavailable"
            positions[v] = s
        return positions, None

    def _safety_abort(self, frames: Mapping) -> str | None:
        """Declared safety aborts, evaluated in EVERY phase (including
        assembly): a missing member frame, a member span beyond the abort
        cap, and a non-member directly behind the wall and faster than it
        (rear-end risk).  v2.2: every position comparison runs on the
        edge-aware cumulative ``s_road`` coordinate."""
        for v in self.members:
            if v not in frames or frames[v] is None:
                return "member_frame_missing"
        positions, position_error = self._member_positions(frames)
        if positions is None:
            # WM-43 A6: fail LOUDLY with the declared reason - never fall
            # back to a non-comparable coordinate or skip the span check.
            return position_error
        span = max(positions.values()) - min(positions.values())
        if span > wall_group.ABORT_SPAN_M:
            return "member_span_exceeded"
        # s_road increases DOWNSTREAM: the rear (most upstream) member is
        # min(s_road) and "directly behind the wall" is the s_road window
        # [rear - REAR_GAP_M, rear].
        rear = min(positions.values())
        for v, frame in frames.items():
            if v in self.members:
                continue
            pos = frame_s_road(frame)
            speed = getattr(frame, "speed_mps", None)
            if pos is None or speed is None or not isfinite(float(speed)):
                continue  # unplaced non-member: surveillance skips it
            if rear - wall_group.REAR_GAP_M <= pos <= rear \
                    and float(speed) > self.wall_speed_mps + wall_group.REAR_SPEED_EXCESS_MPS:
                return f"non_member_rear_end_risk:{v}"
        return None
