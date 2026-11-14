"""INC1 response schedules R0-R4 (registration section 4).

Each schedule returns, per control tick, a :class:`ControlPlan` of

* ``wall``    -- CAV id -> absolute acceleration to issue through
  ``CommandOwnership``'s wall path (the ONLY writer; R1/R2/R3 are all
  expressed as ownership-owned commands, never a private traci write), and
* ``native``  -- the remaining live CAVs to release to native.

CAVs not in ``wall`` and not in ``native`` run the frozen CIRCLES damping
(R3's purpose); HDVs are always untouched IDM and never appear in a plan.

R4 is NOT a :class:`ControlPlan` schedule: per
``REGISTRATION_WM_N1_INC1_ADDENDUM_1.md`` (A1.2/A1.3) it is driven through the
patched frozen v2.2 driver by :mod:`inc1.r4_driver` (the optional
``post_restore_hook`` + ``branch_length_s_override`` kwargs on top of
``walls_design/merge_release/copies/post/wall_driver.py``).  ``make_schedule
("R4")`` therefore refuses - there is no private re-implementation of the
wall law here.

R1 with no qualifying CAV is infeasible (A1.3): it falls back to R0 with a
log and is excluded from the world's feasible set (``arm_feasibility``).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    from . import constants as C
except ImportError:  # script execution
    import constants as C  # type: ignore

MAINLINE_LANES = (0, 1)  # lane_phys main_left / main_right


class ScheduleError(ValueError):
    """A schedule contract violation; never silently relaxed."""


@dataclass(frozen=True)
class CavState:
    vehicle_id: str
    speed_mps: float
    s_down_m: float
    lane_phys: int
    allowed_speed_mps: float = 30.0
    #: the CAV's native IDM acceleration at t0 (the released/native
    #: car-following acceleration reported by the simulator).  Used as R2's
    #: upper acceleration bound; ``None`` means unknown -> never accelerate.
    native_accel_mps2: float | None = None
    #: measured acceleration and current same-lane leader (bumper gap m,
    #: leader speed m/s); consumed only by the R5/R6 shared generators.
    accel_mps2: float = 0.0
    leader_gap_m: float | None = None
    leader_v_mps: float | None = None
    #: False once the CAV has left the network (drops out of R5/R6 membership)
    present: bool = True


@dataclass
class ControlPlan:
    wall: dict = field(default_factory=dict)
    native: tuple = ()
    events: list = field(default_factory=list)


def _native_all(states) -> ControlPlan:
    return ControlPlan(wall={}, native=tuple(sorted(s.vehicle_id for s in states)))


def _clip(accel: float, lo: float, hi: float) -> float:
    """Bound a commanded acceleration to the registered interval [lo, hi]."""
    lo, hi = float(lo), float(hi)
    if hi < lo:
        hi = lo
    return float(max(lo, min(hi, float(accel))))


def _validate_plan(states, plan: ControlPlan) -> None:
    ids = {s.vehicle_id for s in states}
    if not set(plan.wall) <= ids or not set(plan.native) <= ids:
        raise ScheduleError("plan names a non-live CAV")
    if set(plan.wall) & set(plan.native):
        raise ScheduleError("wall and native masks must be disjoint")


# ------------------------------------------------------------------- R0/R3
class SubsetSchedule:
    """Controls a fixed subset of CAVs directly (wall accel or damping)."""

    def __init__(self, selected: list[str], *, controller: str,
                 obstacle_s_down: float, source: str):
        self.selected = sorted(selected)
        self.controller = controller          # "damping" or "schedule"
        self.obstacle_s_down = float(obstacle_s_down)
        self.source = source
        self.events: list = []

    def step(self, now_s: float, states) -> ControlPlan:
        ids = {s.vehicle_id for s in states}
        selected = [v for v in self.selected if v in ids]
        native = tuple(sorted(ids - set(selected)))
        plan = ControlPlan(wall={}, native=native)
        _validate_plan(states, plan)
        return plan


# ----------------------------------------------------------------------- R1
class R1NearestStop:
    """Nearest upstream CAV (either mainline lane, <=500 m): -2.0 m/s^2,
    hold 30 s, then native.  No qualifier -> the arm equals R0 (logged).

    The issued command is bounded to the registered interval
    ``[R1_BRAKE_MPS2, 0]`` s (a -2.0 m/s^2 decel at most, never a positive
    push); the hold is a 0.0 command for exactly ``R1_HOLD_S``.
    """

    def __init__(self, states, obstacle_s_down: float):
        self.obstacle_s_down = float(obstacle_s_down)
        candidates = [
            s for s in states
            if s.lane_phys in MAINLINE_LANES
            and s.s_down_m < self.obstacle_s_down
            and (self.obstacle_s_down - s.s_down_m) <= C.R1_RANGE_M
        ]
        self.selected = (max(candidates, key=lambda s: s.s_down_m).vehicle_id
                         if candidates else None)
        self.phase = "braking"
        self.hold_start_s = None
        self.events: list = []
        if self.selected is None:
            self.events.append({"kind": "r1_no_qualifier",
                                "falls_back_to": "R0"})

    def step(self, now_s: float, states) -> ControlPlan:
        by_id = {s.vehicle_id: s for s in states}
        if self.selected is None or self.selected not in by_id:
            return _native_all(states)
        state = by_id[self.selected]
        if self.phase == "braking" and state.speed_mps <= 0.05:
            self.phase = "holding"
            self.hold_start_s = float(now_s)
            self.events.append({"kind": "r1_stopped", "vehicle_id": self.selected,
                                "timestamp_s": float(now_s)})
        if self.phase == "holding":
            if now_s - self.hold_start_s >= C.R1_HOLD_S - 1e-9:
                self.phase = "done"
            else:
                return self._plan_with(states, self.selected, 0.0)
        if self.phase == "braking":
            return self._plan_with(states, self.selected, C.R1_BRAKE_MPS2)
        return _native_all(states)

    @staticmethod
    def _plan_with(states, selected, accel) -> ControlPlan:
        ids = {s.vehicle_id for s in states}
        native = tuple(sorted(ids - {selected}))
        bounded = _clip(accel, C.R1_BRAKE_MPS2, 0.0)
        plan = ControlPlan(wall={selected: bounded}, native=native)
        _validate_plan(states, plan)
        return plan


# ----------------------------------------------------------------------- R2
class R2EarlySlow:
    """CAVs with s_down in [obs-1000, obs-200) at t0 track 15 m/s
    (decel <= 1.5 m/s^2) for 120 s, then native.

    The issued command is bounded to ``[-R2_DECEL_MAX_MPS2, native]`` where
    ``native`` is the CAV's own native IDM acceleration captured at t0: R2
    never commands a harder decel than 1.5 m/s^2 nor a stronger acceleration
    than the vehicle's native model would apply.
    """

    def __init__(self, states, obstacle_s_down: float, t0_s: float = C.T0_S):
        self.obstacle_s_down = float(obstacle_s_down)
        self.t0_s = float(t0_s)
        lo = self.obstacle_s_down - C.R2_UPSTREAM_LO_M
        hi = self.obstacle_s_down - C.R2_UPSTREAM_HI_M
        self.controlled = sorted(
            s.vehicle_id for s in states if lo <= s.s_down_m < hi)
        by_id = {s.vehicle_id: s for s in states}
        self.native_accel_mps2 = {
            vid: (C.R2_ACCEL_MAX_DEFAULT_MPS2
                  if by_id[vid].native_accel_mps2 is None
                  else float(by_id[vid].native_accel_mps2))
            for vid in self.controlled
        }
        self.events: list = []

    def _upper_bound(self, vid: str) -> float:
        if vid not in self.native_accel_mps2:
            return C.R2_ACCEL_MAX_DEFAULT_MPS2
        # never let the native cap fall below the registered decel floor
        return max(self.native_accel_mps2[vid], -C.R2_DECEL_MAX_MPS2)

    def step(self, now_s: float, states) -> ControlPlan:
        if now_s - self.t0_s >= C.R2_DURATION_S - 1e-9:
            return _native_all(states)
        by_id = {s.vehicle_id: s for s in states}
        controlled = [v for v in self.controlled if v in by_id]
        ids = set(by_id)
        native = tuple(sorted(ids - set(controlled)))
        wall = {}
        for vid in controlled:
            state = by_id[vid]
            desired = (C.R2_TARGET_MPS - state.speed_mps) / C.CONTROL_DT_S
            wall[vid] = _clip(desired, -C.R2_DECEL_MAX_MPS2,
                              self._upper_bound(vid))
        plan = ControlPlan(wall=wall, native=native)
        _validate_plan(states, plan)
        return plan


# ----------------------------------------------------------------------- R3
def r3_schedule(states, obstacle_s_down: float) -> SubsetSchedule:
    """Upstream CAVs (s_down < obstacle) run the frozen CIRCLES damping."""
    selected = [s.vehicle_id for s in states
                if s.s_down_m < float(obstacle_s_down)]
    return SubsetSchedule(selected, controller="damping",
                          obstacle_s_down=obstacle_s_down, source="R3")


# ----------------------------------------------------------------------- R4
def r4_requires_driver_path(*_args, **_kwargs):
    """R4 is driven by :mod:`inc1.r4_driver`, not as a ControlPlan.

    The frozen v2.2 wall law is never re-implemented here; the patched
    driver path (A1.2) owns the wall request, assembly and handback.
    """
    raise ScheduleError(
        "R4 (frozen wall law v2.2) is driven through inc1.r4_driver via the "
        "registered driver hook patch (ADDENDUM_1 A1.2); it is not a "
        "ControlPlan schedule and is never re-implemented here.")


#: kept for callers that used to guard the blocked arm; now a no-op that
#: simply points at the driver path.
r4_blocked = r4_requires_driver_path


def arm_feasibility(arm: str, schedule) -> tuple[bool, str | None]:
    """``(feasible, reason)`` for a per-CAV arm from its built schedule.

    R1 with no qualifying CAV falls back to R0 (logged ``r1_no_qualifier``)
    and is INFEASIBLE per A1.3, so the world's feasible set excludes it
    rather than counting a duplicate of R0.  R5/R6 carry their own verdict
    from the shared generator (A2.2/A2.3).  Every other ControlPlan arm is
    feasible by construction; R4 feasibility is classified by
    :func:`inc1.r4_driver.classify_branch` from the run receipts.
    """
    if arm == "R1" and isinstance(schedule, R1NearestStop):
        if schedule.selected is None:
            return False, "r1_no_qualifier"
    feasible = getattr(schedule, "feasible", None)
    if feasible is not None and arm in ("R5", "R6"):
        return bool(feasible), getattr(schedule, "infeasible_reason", None)
    return True, None


def make_schedule(arm: str, states, obstacle_s_down: float, *,
                  t0_s: float = C.T0_S):
    """Build the schedule for ``arm`` (R0-R3, R5/R6; R4 refuses).

    R5/R6 are backed by the ONE shared ``inc1_gen`` generator through
    :class:`inc1.scripted.ScriptedSchedule`; nothing is re-implemented here.
    """
    if arm == "R0":
        return SubsetSchedule([], controller="native",
                              obstacle_s_down=obstacle_s_down, source="R0")
    if arm == "R1":
        return R1NearestStop(states, obstacle_s_down)
    if arm == "R2":
        return R2EarlySlow(states, obstacle_s_down, t0_s=t0_s)
    if arm == "R3":
        return r3_schedule(states, obstacle_s_down)
    if arm == "R4":
        return r4_requires_driver_path(states, obstacle_s_down)
    if arm in ("R5", "R6"):
        from . import scripted  # lazy: avoids a schedules<->scripted cycle
        return scripted.make_scripted_schedule(arm, states, obstacle_s_down,
                                               t0_s=t0_s)
    raise ScheduleError(f"unknown arm {arm!r}")
