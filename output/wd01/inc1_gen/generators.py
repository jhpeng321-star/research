"""Scripted multi-CAV response generators R5 and R6 (addendum 2, A2.2/A2.3).

ONE shared module used both by the SUMO arm (through ``CommandOwnership``,
longitudinal commands only) and by the WM rollout as the in-the-loop CAV
mover.  Pure, deterministic, closed-loop rules; numpy-only (no traci, no
torch); zero SUMO; nothing reads an outcome.

Common semantics (A2.1, verbatim rules):

* limits for EVERY scripted command: ``decel <= 1.5 m/s^2`` and
  ``accel <= the native IDM acceleration of that vehicle at that tick``;
* membership fixed at t0 from the CAVs present; a member that leaves the
  network drops out (then native);
* ``d = s_obs - s_down(CAV)`` on the mainline;
* no lateral command; lane changes stay SUMO-native (LC2013);
* each :meth:`step` returns the per-tick ``{vehicle_id: accel_cmd}`` for the
  still-tracked members plus the native set (all other live CAVs), and an
  infeasible reason when applicable.

Registered readings implemented here (documented in ``INC1_GEN.md``):

* target-speed tracking commands ``(v_tgt - v) / control_dt``, clipped to the
  common limits (the same convention as the registered R2 SUMO schedule);
* when the native IDM bound is below the decel floor the clip interval
  degenerates to ``[-1.5, max(native, -1.5)]`` (never below -1.5), the same
  resolution the registered R2 schedule uses;
* the R6 headway controller is the one-gain relaxation
  ``a_hc = k * (max(0, gap) / h - v)`` with the registered gain ``k = 0.5
  s^-1`` and headway ``h = 3.0 s`` (gain units s^-1 on a speed error);
* the R6 blocked-lane approacher target is ``min(12 m/s, v + native * dt)``:
  "min(12 m/s, native)" operationalised so the vehicle is never pushed harder
  than its native IDM bound at this tick.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .states import ObstacleRecord, VehState

# ------------------------------------------------------------- common limits
DECEL_MAX_MPS2 = 1.5            # decel <= 1.5 m/s^2 (magnitude bound)
NATIVE_UNKNOWN_CAP_MPS2 = 0.0   # unknown native accel -> never accelerate
EPS_S = 1e-9

# ------------------------------------------------------------------- R5 (A2.2)
R5_NAME = "R5"
R5_LANES = (0, 1)               # mainline lanes
R5_D_MIN_M = 100.0              # d in [100, 1500) m at t0
R5_D_MAX_M = 1500.0
R5_DURATION_S = 120.0           # then native
R5_V_BASE_MPS = 10.0            # v_tgt(d) = 10 + 12*(d-100)/1400, clipped
R5_V_SPAN_MPS = 12.0
R5_D_SPAN_M = 1400.0
R5_MIN_MEMBERS = 2              # feasible iff >= 2 members

# ------------------------------------------------------------------- R6 (A2.3)
R6_NAME = "R6"
R6_OPENER_LANE = 0              # adjacent lane (lane_phys 0)
R6_OPENER_D_MIN_M = 0.0         # d in [0, 600) m at t0
R6_OPENER_D_MAX_M = 600.0
R6_HEADWAY_S = 3.0              # tracked time headway
R6_HEADWAY_GAIN_1PS = 0.5       # headway controller gain 0.5 s^-1
R6_APPROACH_LANE = 1            # blocked lane (obstacle's lane)
R6_APPROACH_D_MIN_M = 30.0      # d in [30, 600) m at t0
R6_APPROACH_D_MAX_M = 600.0
R6_APPROACH_SPEED_MPS = 12.0    # tracks min(12 m/s, native)
R6_DURATION_S = 180.0           # then native
R6_MIN_MEMBERS = 2              # >= 2 members in total ...
R6_MIN_OPENERS = 1              # ... with at least 1 gap opener


def clip_command(accel: float, native_accel: float | None) -> float:
    """The common limits (A2.1) applied to one raw command.

    Result lies in ``[-1.5, max(native, -1.5)]`` m/s^2: never a harder decel
    than 1.5 m/s^2 and never a stronger push than the native IDM bound (an
    unknown native bound counts as 0.0 -> never accelerate).
    """
    cap = (NATIVE_UNKNOWN_CAP_MPS2 if native_accel is None
           else float(native_accel))
    hi = max(cap, -DECEL_MAX_MPS2)
    return float(max(-DECEL_MAX_MPS2, min(hi, float(accel))))


def native_cap(state: VehState) -> float:
    """The usable native IDM bound of one vehicle at this tick."""
    return (NATIVE_UNKNOWN_CAP_MPS2 if state.native_accel is None
            else float(state.native_accel))


# --------------------------------------------------------------- step output
@dataclass
class GeneratorStep:
    """One control tick of a scripted generator."""

    t_s: float
    #: per-tick {vehicle_id: accel_cmd} for the still-tracked members
    commands: dict[str, float] = field(default_factory=dict)
    #: all other live CAVs released to native (never an HDV)
    native: tuple[str, ...] = ()
    feasible: bool = True
    infeasible_reason: str | None = None
    events: tuple[dict, ...] = ()


class _ScriptedGenerator:
    """Shared membership/window/limits machinery for R5 and R6."""

    name = "scripted"
    duration_s = 0.0

    def __init__(self, t0_states, obstacle: ObstacleRecord, *,
                 t0_s: float, control_dt_s: float = 1.0) -> None:
        self.obstacle = obstacle
        self.t0_s = float(t0_s)
        self.control_dt_s = float(control_dt_s)
        self._members: tuple[str, ...] = ()
        self._dropped: set[str] = set()
        self._window_closed = False
        self.events: list[dict] = []
        self.infeasible_reason: str | None = None
        self.feasible: bool = True
        self._latch_membership(t0_states)

    # -- membership ------------------------------------------------------
    def _latch_membership(self, t0_states) -> None:
        """Membership is fixed at t0 from the CAVs present (A2.1)."""
        raise NotImplementedError

    def _select(self, state: VehState, *, lanes, d_min: float,
                d_max: float) -> bool:
        if not state.present or not state.is_cav:
            return False
        if state.lane_phys not in lanes:
            return False
        d = self.obstacle.distance_to(state)
        return bool(d_min <= d < d_max)

    # -- per-tick --------------------------------------------------------
    def step(self, now_s: float, states) -> GeneratorStep:
        now_s = float(now_s)
        live = {s.vehicle_id: s for s in states if s.present and s.is_cav}
        for vid in self._members:
            if vid not in live and vid not in self._dropped:
                self._dropped.add(vid)
                self.events.append({"kind": "member_left", "generator": self.name,
                                    "vehicle_id": vid, "t_s": now_s})
        plan = GeneratorStep(t_s=now_s, feasible=self.feasible,
                             infeasible_reason=self.infeasible_reason)
        if not self.feasible:
            plan.native = tuple(sorted(live))
            return plan
        if now_s - self.t0_s >= self.duration_s - EPS_S and not self._window_closed:
            self._window_closed = True
            self.events.append({"kind": "window_closed", "generator": self.name,
                                "t_s": now_s})
        if self._window_closed:
            plan.native = tuple(sorted(live))
            return plan
        for vid in self._members:
            if vid in self._dropped or vid not in live:
                continue
            state = live[vid]
            raw = self._raw_command(state, now_s)
            plan.commands[vid] = clip_command(raw, state.native_accel)
        plan.native = tuple(sorted(set(live) - set(plan.commands)))
        return plan

    def _raw_command(self, state: VehState, now_s: float) -> float:
        """The UNclipped scripted command for one tracked member."""
        raise NotImplementedError


def staggered_target(d: float) -> float:
    """The R5 staggered target ``clip(10 + 12*(d-100)/1400, 10, 22)`` m/s."""
    raw = R5_V_BASE_MPS + R5_V_SPAN_MPS * (float(d) - R5_D_MIN_M) / R5_D_SPAN_M
    return float(min(R5_V_BASE_MPS + R5_V_SPAN_MPS,
                     max(R5_V_BASE_MPS, raw)))


def headway_accel(gap: float, v: float, *, headway_s: float = R6_HEADWAY_S,
                  gain_1ps: float = R6_HEADWAY_GAIN_1PS) -> float:
    """The R6 gap-opener headway controller.

    One-gain relaxation on the headway-implied speed: the vehicle relaxes its
    speed toward ``gap / h`` (the 3.0 s time-headway speed) with the
    registered gain 0.5 s^-1.  ``gap`` is the bumper gap to the CURRENT
    leader; a gap is never treated as negative.
    """
    implied = max(0.0, float(gap)) / float(headway_s)
    return float(gain_1ps) * (implied - float(v))


class R5Harmonisation(_ScriptedGenerator):
    """R5: coordinated upstream speed harmonisation (A2.2).

    Members: every CAV on mainline lanes 0-1 with d in [100, 1500) m at t0.
    For 120 s after t0 each member tracks the staggered target
    ``v_tgt(d) = 10 + 12*(d-100)/1400`` (clipped to [10, 22]), with d updated
    each tick; members nearer the obstacle are slowed more.  Feasible iff at
    least 2 members.
    """

    name = R5_NAME
    duration_s = R5_DURATION_S

    def _latch_membership(self, t0_states) -> None:
        members = sorted(
            s.vehicle_id for s in t0_states
            if self._select(s, lanes=R5_LANES, d_min=R5_D_MIN_M,
                            d_max=R5_D_MAX_M))
        self._members = tuple(members)
        n = len(self._members)
        self.events.append({"kind": "membership", "generator": self.name,
                            "members": list(self._members), "n": n})
        if n < R5_MIN_MEMBERS:
            self.feasible = False
            self.infeasible_reason = "infeasible_fewer_than_two_members"
            self.events.append({"kind": "infeasible", "generator": self.name,
                                "reason": self.infeasible_reason,
                                "n_members": n})

    def _raw_command(self, state: VehState, now_s: float) -> float:
        d = self.obstacle.distance_to(state)
        v_tgt = staggered_target(d)
        return (v_tgt - state.v) / self.control_dt_s


class R6GapOpening(_ScriptedGenerator):
    """R6: coordinated lane-clearing and gap opening (A2.3).

    The blocked lane is the obstacle's lane (lane_phys 1, right); the
    adjacent lane is lane_phys 0.  For 180 s after t0, then native:

    * gap openers: every CAV in lane 0 with d in [0, 600) m at t0 tracks a
      3.0 s time headway to its current leader; the command is
      ``min(native IDM accel, headway controller gain 0.5 s^-1)`` within the
      common limits;
    * blocked-lane approachers: every CAV in lane 1 with d in [30, 600) m at
      t0 tracks ``min(12 m/s, native)``; its lane change stays SUMO-native.

    Feasible iff at least 2 members in total with at least 1 gap opener.
    """

    name = R6_NAME
    duration_s = R6_DURATION_S

    def __init__(self, t0_states, obstacle: ObstacleRecord, *,
                 t0_s: float, control_dt_s: float = 1.0) -> None:
        self.openers: tuple[str, ...] = ()
        self.approachers: tuple[str, ...] = ()
        super().__init__(t0_states, obstacle, t0_s=t0_s,
                         control_dt_s=control_dt_s)

    def _latch_membership(self, t0_states) -> None:
        openers = sorted(
            s.vehicle_id for s in t0_states
            if self._select(s, lanes=(R6_OPENER_LANE,),
                            d_min=R6_OPENER_D_MIN_M, d_max=R6_OPENER_D_MAX_M))
        approachers = sorted(
            s.vehicle_id for s in t0_states
            if self._select(s, lanes=(R6_APPROACH_LANE,),
                            d_min=R6_APPROACH_D_MIN_M,
                            d_max=R6_APPROACH_D_MAX_M))
        self.openers = tuple(openers)
        self.approachers = tuple(approachers)
        self._members = tuple(sorted(self.openers + self.approachers))
        n_open, n_app = len(self.openers), len(self.approachers)
        self.events.append({"kind": "membership", "generator": self.name,
                            "openers": list(self.openers),
                            "approachers": list(self.approachers),
                            "n_openers": n_open, "n_approachers": n_app})
        if n_open + n_app < R6_MIN_MEMBERS:
            self.feasible = False
            self.infeasible_reason = "infeasible_fewer_than_two_members"
        elif n_open < R6_MIN_OPENERS:
            self.feasible = False
            self.infeasible_reason = "infeasible_no_gap_opener"
        if not self.feasible:
            self.events.append({"kind": "infeasible", "generator": self.name,
                                "reason": self.infeasible_reason,
                                "n_openers": n_open, "n_approachers": n_app})

    def _raw_command(self, state: VehState, now_s: float) -> float:
        cap = native_cap(state)
        if state.vehicle_id in self.openers:
            if state.leader_gap is None:
                # no current leader: the headway term is vacuous -> native
                return cap
            return min(cap, headway_accel(state.leader_gap, state.v))
        v_tgt = min(R6_APPROACH_SPEED_MPS, state.v + cap * self.control_dt_s)
        return (v_tgt - state.v) / self.control_dt_s


def make_generator(arm: str, t0_states, obstacle: ObstacleRecord, *,
                   t0_s: float, control_dt_s: float = 1.0
                   ) -> _ScriptedGenerator:
    """Build the scripted generator for ``arm`` (``"R5"`` or ``"R6"``)."""
    if arm == R5_NAME:
        return R5Harmonisation(t0_states, obstacle, t0_s=t0_s,
                               control_dt_s=control_dt_s)
    if arm == R6_NAME:
        return R6GapOpening(t0_states, obstacle, t0_s=t0_s,
                            control_dt_s=control_dt_s)
    raise ValueError(f"unknown scripted arm {arm!r} (expected 'R5' or 'R6')")
