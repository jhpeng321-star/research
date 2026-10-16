"""Planner-driven wall groups for the per-CAV dispatcher (WM-40).

A wall is a GROUP action over a planner-chosen member set that spans all
lanes where it forms (one member per lane).  The planner sets the members,
their lane order and the wall speed (any value strictly above
``SAFE_SPEED_MIN_MPS``); the hold duration is NOT fixed -- the planner may
release the wall any time after assembly completes, and assembly cannot be
cancelled by the planner.  SAFETY ABORTS are allowed in every phase
(including assembly) and hand members back to NATIVE immediately.

Lifecycle: ``assembly`` -> ``hold`` -> ``handback``.  Pacing and the
assembly-completion test reuse the registered wall-v2 conventions
(``output/wall_v2_integration_design/wall_v2_suffix.py``): requests are
absolute accelerations ``a = clip((v_target - v)/dt, -3, +1.5)`` m/s^2,
always inside the ownership clip, and assembly completes on speed +
longitudinal span alignment.  The old wall-v2 trigger/gating rules (E2/E3
entry conditions, rearm deadband) are NOT part of this session -- the
planner decides, and the declared safe-speed floor is a SAFETY limit,
never a trigger.

ASSEMBLY CLOSURE RULE ("Option B", law version
``WALL_LAW_VERSION``).  Pacing every member to the same speed can never close
a longitudinal gap, so wide requests (up to MAX_WALL_SPAN_M) never assembled.
During ASSEMBLY the FRONT member (largest downstream position) holds
``v_wall``; every REAR member targets ``v_wall + min(2.0 m/s, 0.1 1/s * gap)``
where ``gap`` is its own distance behind the front, until span <=
ASSEMBLY_SPAN_TOL_M.  The target is still turned into an absolute
acceleration by the same bounded clip [-3, +1.5] m/s^2 (``_pace``).  The
completion test (speed within 1 m/s of v_wall AND span <= 6 m) and the HOLD
phase (all members paced to ``v_wall``) are unchanged.  A new safety abort
``assembly_timeout`` fires when assembly has not completed 60 s after
``assembly_start``.  The Part A emulator mirror
(output/wd01/partA_ctrl/emulators.py::wall_pace) implements the identical law.

Report content only: ``tick`` consumes SensorFrame-like objects and never
sees a physics snapshot or truth.
"""
from __future__ import annotations

from math import isfinite
from typing import Mapping

#: SAFETY limit (not a trigger): the planner may set any wall speed strictly
#: above this floor; requests at or below it are refused with a reason.
SAFE_SPEED_MIN_MPS = 2.0

#: Feasibility span cap for the requested member set (wl04/CV-26 scale).
MAX_WALL_SPAN_M = 75.0
#: Longitudinal span above which an ACTIVE wall aborts.
ABORT_SPAN_M = 150.0
#: Assembly alignment tolerances (speed and longitudinal span).
ASSEMBLY_SPEED_TOL_MPS = 1.0
ASSEMBLY_SPAN_TOL_M = 6.0
#: Bounded pacing (ownership clips to [-3, 1.5]; mirrors wall_v2 pacing).
BUFFER_DECEL_MPS2 = 3.0
COMFORT_ACCEL_MPS2 = 1.5
#: Non-member rear-end surveillance gap behind the rearmost member.
REAR_GAP_M = 10.0
#: A non-member this much faster than the wall inside the rear gap is a
#: rear-end risk -> safety abort.
REAR_SPEED_EXCESS_MPS = 2.0
#: Assembly closure (Option B): rear members target
#: v_wall + min(MAX_DELTA, GAIN * gap_to_front).
ASSEMBLY_CLOSE_MAX_DELTA_MPS = 2.0
ASSEMBLY_CLOSE_GAIN_PER_S = 0.1
#: Safety abort when assembly has not completed this long after assembly_start.
ASSEMBLY_TIMEOUT_S = 60.0
WALL_LAW_VERSION = "wall-v2.1-closure"

PHASE_ASSEMBLY = "assembly"
PHASE_HOLD = "hold"
PHASE_HANDBACK = "handback"


class WallRequestError(ValueError):
    """An infeasible wall request; the message carries the reason."""


def validate_wall_request(members, wall_speed_mps: float, active_walls: Mapping) -> frozenset:
    """Validate a planner wall request; return the normalized member set.

    Refusals (each with a reason in the message): fewer than two members,
    a member already owned by another ACTIVE wall (a released or aborted
    member may join a later wall), and a wall speed at or below the declared
    safety floor.
    """
    members = frozenset(members)
    if len(members) < 2:
        raise WallRequestError("a wall needs at least two members (it spans several lanes)")
    for wall_id, other in active_walls.items():
        overlap = members & other.members
        if overlap:
            raise WallRequestError(
                f"members {sorted(overlap)} already belong to active wall {wall_id}")
    speed = wall_speed_mps
    if isinstance(speed, bool) or not isfinite(float(speed)):
        raise WallRequestError("wall_speed_mps must be a finite number")
    if float(speed) <= SAFE_SPEED_MIN_MPS:
        raise WallRequestError(
            f"wall_speed_mps {speed} is at or below the safety floor "
            f"{SAFE_SPEED_MIN_MPS} m/s (any value strictly above it is valid)")
    return members


class WallGroupSession:
    """One planner-driven wall: assembly -> hold -> (planner or safety)
    handback.  Members are exclusively owned by their wall session while it
    is active; the dispatcher hands them back to NATIVE on release or abort."""

    def __init__(self, wall_id: str, *, members: frozenset, wall_speed_mps: float):
        self.wall_id = wall_id
        self.members = frozenset(members)
        self.wall_speed_mps = float(wall_speed_mps)
        self.phase = PHASE_ASSEMBLY
        self.abort_reason: str | None = None
        self.assembly_since: float | None = None
        self.assembled_at_s: float | None = None
        self.events: list[dict] = []

    # ------------------------------------------------------------------ api
    def tick(self, frames: Mapping, now_s: float, *,
             control_dt_s: float = 1.0) -> dict:
        """One control tick: returns ``{member: absolute acceleration m/s^2}``
        wall commands, applying the declared safety aborts (every phase) and
        the assembly -> hold transition.  Input is report content only."""
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
            for v, speed in speeds.items():
                commands[v] = _pace(self.wall_speed_mps, speed, control_dt_s)
        return commands

    def release(self, now_s: float) -> bool:
        """Planner handback; refused during assembly (the planner cannot
        cancel assembly; SAFETY aborts remain available in every phase).
        Returns True when the handback is accepted."""
        if self.phase != PHASE_HOLD:
            return False
        self.phase = PHASE_HANDBACK
        self.events.append({"kind": "wall_event", "wall_id": self.wall_id,
                            "event": "released", "timestamp_s": now_s,
                            "handback": "native"})
        return True

    def abort(self, reason: str, now_s: float) -> None:
        """Declared safety abort; valid in EVERY phase including assembly."""
        self.abort_reason = reason
        self.phase = PHASE_HANDBACK
        self.events.append({"kind": "wall_event", "wall_id": self.wall_id,
                            "event": "safety_abort", "timestamp_s": now_s,
                            "reason": reason, "handback": "native"})

    @property
    def active(self) -> bool:
        return self.phase in (PHASE_ASSEMBLY, PHASE_HOLD)

    # -------------------------------------------------------------- helpers
    def _member_positions(self, frames: Mapping):
        """Member s_m positions (downstream-increasing; see
        downstream_position_m).  Returns (positions, None) or
        (None, "member_position_unavailable"): a member without a finite
        coordinate aborts the wall LOUDLY - there is deliberately NO
        fallback to the lane-local position_m (not comparable across
        edges; r3 false member_span_exceeded) or to any odometer.
        """
        positions = {}
        for v in self.members:
            frame = frames.get(v)
            s = downstream_position_m(frame)
            if s is None:
                return None, "member_position_unavailable"
            positions[v] = s
        return positions, None

    def _lane_coverage(self, frames: Mapping) -> set | None:
        """Distinct lane identities across members, or None when the frames
        carry no lane information (the check is then skipped)."""
        lanes = set()
        for v in self.members:
            lane = getattr(frames[v], "lane_index", None)
            if lane is None:
                lane = getattr(frames[v], "lane_id", None)
            if lane is None:
                return None
            lanes.add(lane)
        return lanes

    def _safety_abort(self, frames: Mapping) -> str | None:
        """Declared safety aborts, evaluated in EVERY phase (including
        assembly): a missing member frame, a member span beyond the abort
        cap, and a non-member directly behind the wall and faster than it
        (rear-end risk, e.g. a PI CAV commanded toward 31.29 m/s)."""
        for v in self.members:
            if v not in frames or frames[v] is None:
                return "member_frame_missing"
        positions, position_error = self._member_positions(frames)
        if positions is None:
            # WM-43 A6: fail LOUDLY with the declared reason - never fall
            # back to a non-comparable coordinate or skip the span check.
            return position_error
        span = max(positions.values()) - min(positions.values())
        if span > ABORT_SPAN_M:
            return "member_span_exceeded"
        # s_m increases DOWNSTREAM: the rear (most upstream) member is
        # min(s_m) and "directly behind the wall" is the s_m window
        # [rear - REAR_GAP_M, rear] (r5 sign defect: with the raw
        # distance_to_merge_m these comparisons silently looked AHEAD).
        rear = min(positions.values())
        for v, frame in frames.items():
            if v in self.members:
                continue
            pos = downstream_position_m(frame)
            speed = getattr(frame, "speed_mps", None)
            if pos is None or speed is None or not isfinite(float(speed)):
                continue  # unplaced non-member: surveillance skips it
            if rear - REAR_GAP_M <= pos <= rear \
                    and float(speed) > self.wall_speed_mps + REAR_SPEED_EXCESS_MPS:
                return f"non_member_rear_end_risk:{v}"
        return None

    def _assembly_complete(self, member_speeds: Mapping[str, float],
                           positions: Mapping[str, float] | None) -> bool:
        if set(member_speeds) != set(self.members):
            return False
        if any(abs(speed - self.wall_speed_mps) > ASSEMBLY_SPEED_TOL_MPS
               for speed in member_speeds.values()):
            return False
        if positions is None:
            return False
        span = max(positions.values()) - min(positions.values())
        return span <= ASSEMBLY_SPAN_TOL_M


def downstream_position_m(frame):
    """s_m = -distance_to_merge_m: the DOWNSTREAM-INCREASING longitudinal
    coordinate used by EVERY wall_group position comparison.

    distance_to_merge_m (the registered sensor coordinate, defined for
    pre_merge/merge/downstream in sensing.observation_map + capture)
    counts DOWN toward the merge point: pre_merge [0, 2644.22] decreasing,
    merge [-75.44, 0], downstream [-553.33, -75.44]; negative values are
    valid post-merge positions, not errors.  Negating it gives s_m, which
    increases in the direction of travel and is comparable across all
    mainline edges - lane-local position_m is NOT (the r3 false
    member_span_exceeded), and getDistance() is an odometer (insertion-
    offset trap).  Sign contract (audited, Amendment A6): rear member =
    min(s_m); "behind the wall" = s_m in [rear - REAR_GAP_M, rear].
    Returns None when the frame carries no finite coordinate; MEMBERS
    without one trigger the declared member_position_unavailable abort,
    non-members are skipped by the surveillance (conservative).
    """
    pos = getattr(frame, "distance_to_merge_m", None)
    if pos is None:
        return None
    value = float(pos)
    if not isfinite(value):
        return None
    return -value


def assembly_target_mps(wall_speed_mps: float, gap_to_front_m: float) -> float:
    """Assembly closure target speed of a member ``gap_to_front_m`` behind the
    front member (front: gap 0 -> exactly ``wall_speed_mps``)."""
    delta = min(ASSEMBLY_CLOSE_MAX_DELTA_MPS,
                ASSEMBLY_CLOSE_GAIN_PER_S * max(0.0, gap_to_front_m))
    return wall_speed_mps + delta


def _pace(target_mps: float, speed_mps: float, dt_s: float) -> float:
    """wall_v2 pacing: bounded absolute acceleration toward the wall speed."""
    raw = (target_mps - speed_mps) / dt_s
    return float(min(max(raw, -BUFFER_DECEL_MPS2), COMFORT_ACCEL_MPS2))
