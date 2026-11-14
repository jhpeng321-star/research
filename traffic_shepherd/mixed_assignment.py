"""Per-CAV controller assignment with hysteresis: the planner dispatcher.

``MixedAssignmentSession`` sits between a planner (which produces an
``AssignmentPlan`` coloring: one controller mode per CAV, plus optional
planner-driven walls) and the existing command-ownership machinery. It
exposes the same ``step()`` interface as ``FleetSwitchSession``
(``step(owner, frames, *, now_s, live_ids, disappeared_ids=(),
reappeared_ids=())``) so it can sit in a workflow where a fleet session
sits, but it routes each live designated CAV according to its CURRENT
COMMITTED assignment instead of one fleet-wide mode:

- ``damping`` CAVs stay on the ownership default: the identical CIRCLES
  instance ``CommandOwnership`` was constructed with decides them;
- ``native`` CAVs are passed to ``CommandOwnership`` via ``native_ids``
  (release override to native IDM; no commands written);
- ``pi-external-u`` CAVs are routed through the roster-bound PI session as a
  SUBSET (see ``_PISubsetProxyOwner`` for the supported-path rationale);
- ``wall-v2`` CAVs are members of a planner-driven wall group
  (:class:`~traffic_shepherd.wall_group.WallGroupSession`): a GROUP action
  with planner-set members and wall speed, exclusive member ownership for
  the wall episode, and NATIVE handback on release or safety abort
  (WM-40; the former documented stub is replaced).  A CAV belongs to at
  most one active wall; non-members stay free per-CAV (native / damping /
  pi-external-u).

Native is the default assignment everywhere (mixed_assignment/3): a CAV
absent from a lenient plan targets native, and the initial placement of
every roster CAV is native until the first plan commits.  Plans may set
``strict_roster`` to demand a complete assignments map instead.

Switch clocks (contract change, WM-14 repair A: the previous refresh-on-any-
differing-plan rule let a planner that repeats the same request once per tick
starve that request forever). Each CAV carries two independent clocks:

- **Execution dwell**: an assignment-changing switch (or the initial
  placement) at time ``t`` with horizon ``H`` locks that CAV against further
  switches until ``t + H``; both timestamps are frozen at the execution, so
  later horizon-only plan changes can never postpone an already-earned
  switch.  The horizon is CONFIGURABLE per plan; the default is 0
  (planner-driven switching checks continuously).  Wall lifecycle actions
  (assembly commit, handback) are NOT per-CAV switches and bypass both
  clocks.
- **Pending-target stability**: the first plan that names a target different
  from the committed assignment starts that target's stability clock at the
  plan time with the plan's horizon frozen as the stability requirement.
  Repeating the same pending target -- under any plan id, or with a changed
  horizon -- never restarts the clock; a change to a DIFFERENT target
  restarts it from the new plan time with the new horizon; returning to the
  committed assignment cancels the pending switch entirely.

A pending switch executes on the first control tick satisfying both clocks
(dwell from the previous execution and stability of the pending target).
The first plan ever applied keeps its immediate placement semantics; it
counts as the CAV's execution for dwell purposes. Telemetry records the
pending/committed state and both deadlines per live CAV, plus per-CAV wall
membership/phase and wall events.

Plan times must be monotonic non-decreasing (equal times keep the existing
same-tick plan-before-step order; a plan earlier than its predecessor is
rejected).  A control step at time ``now`` must not execute a tick earlier
than the last submitted plan time: a plan submitted at ``t`` is authoritative
from ``t`` onward, so ``step(now)`` with ``now < t`` is rejected while the
equal-time plan-before-step order stays valid.

Wall locks (WM-40): once a wall is staged it cannot be dropped while it is
assembling, and its members cannot be reassigned, until the wall session
reports assembled or safety-aborted; a violating plan is REJECTED with an
error, never silently ignored.  After assembly the planner may release at
any time (explicit ``release`` phase, or the wall's absence from the plan);
release hands members back to NATIVE, as does any safety abort (available
in every phase, including assembly).

PI lifecycle: while the committed assignment routes a CAV away from
``pi-external-u``, that CAV is simply outside the PI subset each tick, so the
adapter's own lifecycle path drops its recurrence state; re-entry
re-initializes from the measured speed (the same invalidate mechanism
``FleetSwitchSession`` uses on PI exit). A departed CAV's PI state is
invalidated through ``departed_ids``; a reappeared CAV re-enters under its
committed plan assignment and its id reuse is signalled through
``id_reuse_ids``.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from .fleet_switch import MODES
from .wall_group import (PHASE_ASSEMBLY, PHASE_HANDBACK, PHASE_HOLD,
                         WallGroupSession, validate_wall_request)

VERSION = "mixed_assignment/3"

DEFAULT_COMMIT_HORIZON_S = 0.0

WALL_PLAN_PHASES = frozenset(("assemble", "hold", "release"))

# --- wall LAW selection (WALL22-I integration, 2026-10-01) -----------------
#: The pinned v2.1 law (``traffic_shepherd/wall_group.py``, byte-frozen at
#: ``wall-v2.1-closure``) remains the DEFAULT: every plan that does not name
#: a law version stages the exact pre-WALL22-I session behaviour.  Law v2.2
#: 'merge-release' (its registered sibling
#: ``traffic_shepherd/wall_group_v22.py``, REGISTRATION_WALL_V22) is used
#: ONLY when the :class:`WallAssignment` names it explicitly.
WALL_LAW_VERSION_V21 = "wall-v2.1-closure"
WALL_LAW_VERSION_V22 = "wall-v2.2-merge-release"
WALL_LAW_VERSIONS = (WALL_LAW_VERSION_V21, WALL_LAW_VERSION_V22)
#: accepted aliases (the registration's short names) -> canonical versions
WALL_LAW_VERSION_ALIASES = {
    "v2.1-closure": WALL_LAW_VERSION_V21,
    "v2.2-merge-release": WALL_LAW_VERSION_V22,
}


def normalize_wall_law_version(law_version) -> str:
    """``None`` -> the pinned v2.1 default; a canonical version or its
    short alias -> the canonical version; anything else raises
    ``ValueError`` (a typo must fail loudly, never fall back)."""
    if law_version is None:
        return WALL_LAW_VERSION_V21
    name = str(law_version)
    if name in WALL_LAW_VERSIONS:
        return name
    if name in WALL_LAW_VERSION_ALIASES:
        return WALL_LAW_VERSION_ALIASES[name]
    raise ValueError(
        f"unknown wall law_version {law_version!r} (expected one of "
        f"{list(WALL_LAW_VERSIONS)} or aliases "
        f"{sorted(WALL_LAW_VERSION_ALIASES)})")


@dataclass(frozen=True)
class WallAssignment:
    """One planner wall in a plan (``mixed_assignment/3`` schema).

    ``members`` is ordered by lane index, exactly one member per lane of the
    cross-section (the order is the planner's lane encoding; the session
    verifies actual lane coverage from frames at the first tick).  At most
    one of ``hold_s`` (seconds after assembly completes) and ``release_at_s``
    (absolute episode time) may be given: the default release point when the
    planner stops updating; the planner may always release earlier, never
    during assembly.  ``phase`` states the planner intent: ``assemble`` /
    ``hold`` continue the wall (``hold`` is accepted at any session phase),
    ``release`` hands back after assembly.
    """

    wall_id: str
    members: tuple[str, ...]
    wall_speed_mps: float
    phase: str = "assemble"
    hold_s: float | None = None
    release_at_s: float | None = None
    #: WALL22-I: the wall law this wall runs (``normalize_wall_law_version``):
    #: ``None``/``"wall-v2.1-closure"`` stages the pinned v2.1 session
    #: (byte-identical default); ``"wall-v2.2-merge-release"`` stages the
    #: v2.2 merge-release session.  Fixed at creation; a later plan naming a
    #: different law for an active wall is refused.
    law_version: str | None = None

    def __post_init__(self):
        if not isinstance(self.wall_id, str) or not self.wall_id.strip():
            raise ValueError("wall id must be a nonempty string")
        members = tuple(self.members)
        if (len(members) < 2 or len(set(members)) != len(members)
                or any(not isinstance(v, str) or not v for v in members)):
            raise ValueError(
                "wall members must be at least two distinct nonempty cav ids "
                "(a wall spans several lanes, one member per lane)")
        object.__setattr__(self, "members", members)
        speed = self.wall_speed_mps
        if isinstance(speed, bool) or not isfinite(float(speed)):
            raise ValueError("wall speed must be a finite number")
        object.__setattr__(self, "wall_speed_mps", float(speed))
        if self.wall_speed_mps <= 0:
            raise ValueError("wall speed must be positive")
        if self.phase not in WALL_PLAN_PHASES:
            raise ValueError("wall phase must be assemble, hold, or release")
        terms = (self.hold_s, self.release_at_s)
        if (terms[0] is None) == (terms[1] is None):
            raise ValueError("exactly one of hold_s and release_at_s is required")
        if self.hold_s is not None:
            if (isinstance(self.hold_s, bool) or not isfinite(float(self.hold_s))
                    or float(self.hold_s) <= 0):
                raise ValueError("wall hold_s must be finite and positive")
            object.__setattr__(self, "hold_s", float(self.hold_s))
        if self.release_at_s is not None:
            if (isinstance(self.release_at_s, bool) or not isfinite(float(self.release_at_s))
                    or float(self.release_at_s) < 0):
                raise ValueError("wall release_at_s must be finite and nonnegative")
            object.__setattr__(self, "release_at_s", float(self.release_at_s))
        object.__setattr__(self, "law_version",
                           normalize_wall_law_version(self.law_version))


@dataclass(frozen=True)
class AssignmentPlan:
    """One planner coloring: controller mode per CAV, plus wall groups.

    ``assignments`` maps cav_id -> mode (a member of
    :data:`traffic_shepherd.fleet_switch.MODES`); under the default lenient
    mode a CAV absent from the map targets NATIVE (mixed_assignment/3;
    native is the default everywhere).  With ``strict_roster`` the plan must
    name every roster CAV instead -- incomplete maps are rejected at
    ``apply_plan`` (the dispatcher owns the roster).  ``walls`` carries the
    planner's wall groups; the union of wall members must equal the CAVs
    assigned ``wall-v2``.  ``commit_horizon_s`` is the hysteresis horizon
    this plan demands (0 = planner-driven switching without dwell).
    """

    assignments: Mapping[str, str]
    plan_id: str
    commit_horizon_s: float = DEFAULT_COMMIT_HORIZON_S
    walls: tuple[WallAssignment, ...] = ()
    strict_roster: bool = False

    def __post_init__(self):
        if not isinstance(self.plan_id, str) or not self.plan_id.strip():
            raise ValueError("plan id must be a nonempty string")
        horizon = self.commit_horizon_s
        if (isinstance(horizon, bool) or not isfinite(float(horizon)) or float(horizon) < 0):
            raise ValueError("commit horizon must be finite and nonnegative "
                             "(0 = planner-driven switching without dwell)")
        object.__setattr__(self, "commit_horizon_s", float(horizon))
        if not isinstance(self.strict_roster, bool):
            raise ValueError("strict_roster must be a bool")
        if not isinstance(self.assignments, Mapping) or isinstance(self.assignments, str):
            raise ValueError("assignments must be a mapping of cav id to mode")
        normalized = dict(self.assignments)
        if any(not isinstance(v, str) or not v for v in normalized):
            raise ValueError("plan cav ids must be nonempty strings")
        if any(mode not in MODES for mode in normalized.values()):
            raise ValueError("plan modes must be native, damping, pi-external-u, or wall-v2")
        object.__setattr__(self, "assignments", normalized)
        walls = tuple(self.walls)
        if not all(isinstance(w, WallAssignment) for w in walls):
            raise ValueError("walls must be WallAssignment entries")
        ids = [w.wall_id for w in walls]
        if len(ids) != len(set(ids)):
            raise ValueError("wall ids must be unique within a plan")
        seen: set[str] = set()
        for w in walls:
            overlap = seen & set(w.members)
            if overlap:
                raise ValueError(
                    f"members {sorted(overlap)} are assigned to two walls of one plan")
            seen.update(w.members)
        wall_assigned = {v for v, mode in normalized.items() if mode == "wall-v2"}
        if wall_assigned != seen:
            raise ValueError("wall-v2 assignments and the union of wall members must coincide")
        object.__setattr__(self, "walls", walls)


class _ModeDispatchController:
    """Per-vehicle ``decide`` dispatch, valid only inside a mixed PI write.

    PI-assigned frames go to the prepared PI adapter decisions; every other
    frame goes to the identical damping instance the ownership default uses.
    A ``None`` frame carries no vehicle identity, so it goes to damping:
    both controllers return the identical ``missing_observation`` fallback
    there, the owner releases the vehicle to native either way, and the PI
    adapter has staged no proposal for a vehicle it classified as missing.
    """

    def __init__(self, damping_controller, pi_adapter, pi_ids):
        self._damping = damping_controller
        self._pi_adapter = pi_adapter
        self._pi_ids = frozenset(pi_ids)

    @property
    def config(self):
        return self._damping.config

    def decide(self, frame, *, now_s):
        if frame is not None and frame.vehicle_id in self._pi_ids:
            return self._pi_adapter.decide(frame, now_s=now_s)
        return self._damping.decide(frame, now_s=now_s)


class _PISubsetProxyOwner:
    """Owner view that lets the roster-bound PI adapter drive a subset tick.

    Why a proxy: the registered PI adapter (``PIOwnerAdapter``) drives its
    whole tick itself. It requires ``owner.controller is adapter`` and
    ``owner.cav_ids == adapter.cav_ids``, classifies its ``live_ids`` into
    prepared/fallback PI decisions, then performs the single real
    ``CommandOwnership`` write, verifying a persisted record callback for
    every live vehicle. It has no damping category, and the ownership clock
    admits exactly one write per control tick, so mixed damping+PI routing
    cannot be expressed by calling ``PISession.step`` with the real owner.

    The supported path chosen here: hand the adapter this proxy, which
    presents the adapter identity and the full roster, then forwards the
    adapter's single write to the real owner with the full live roster, the
    real native mask, the real wall commands, a mode-dispatching controller
    and a record callback filtered to the PI subset. Every adapter invariant
    (one write per tick, one callback per prepared vehicle, contiguous tick
    clock) is preserved without modifying any existing module.  Wall
    commands are staged by the dispatcher before the adapter tick and are
    forwarded verbatim; the adapter may not add its own.
    """

    def __init__(self, real_owner, pi_adapter, damping_controller):
        self._real = real_owner
        self._pi_adapter = pi_adapter
        self._damping = damping_controller
        self._pi_ids: frozenset = frozenset()
        self._real_frames: Mapping = {}
        self._real_live: frozenset = frozenset()
        self._real_native: frozenset = frozenset()
        self._real_wall: dict = {}
        self._writes_this_tick = 0

    # Identity the adapter checks, delegated to the real owner.
    @property
    def controller(self):
        return self._pi_adapter

    @property
    def cav_ids(self):
        return self._real.cav_ids

    @property
    def cav_speed_mode(self):
        return self._real.cav_speed_mode

    @property
    def _last(self):
        # The adapter's callback verifies ``owner._last[vehicle] is record``;
        # the real owner updates its map before invoking the callback.
        return self._real._last

    def begin_tick(self, *, real_frames, real_live, native_ids, pi_ids,
                   wall_commands=()):
        # Filter frames to the live roster before the real ownership write:
        # the real owner rejects a sensor mask containing departed vehicles
        # (same reason wall_v2_suffix filters; the PI adapter itself receives
        # the raw mapping and validates its own subset).
        self._real_frames = {v: f for v, f in real_frames.items() if v in real_live}
        self._real_live = frozenset(real_live)
        self._real_native = frozenset(native_ids)
        self._real_wall = dict(wall_commands)
        self._pi_ids = frozenset(pi_ids)
        self._writes_this_tick = 0

    def end_tick(self):
        self._real_frames = {}
        self._real_live = frozenset()
        self._real_native = frozenset()
        self._real_wall = {}
        self._pi_ids = frozenset()
        self._writes_this_tick = 0

    def step(self, frames, *, now_s, live_ids, native_ids=(), wall_commands=None,
             record_callback=None):
        """Only the PI adapter calls this, once per mixed tick."""
        if self._writes_this_tick:
            raise RuntimeError("PI adapter attempted a second ownership write in one mixed tick")
        if native_ids:
            raise RuntimeError("mixed PI subset routing must not carry a native mask")
        if record_callback is None:
            raise RuntimeError("PI adapter must supply its record callback")
        if (wall_commands or {}) != self._real_wall:
            raise RuntimeError("PI adapter attempted wall commands outside the "
                               "dispatcher-staged wall set")
        pi_expected = frozenset(live_ids)
        if not pi_expected <= self._pi_ids:
            raise RuntimeError("PI adapter live set escaped the committed PI subset")

        def forward(record):
            if record.vehicle_id in pi_expected:
                record_callback(record)

        dispatch = _ModeDispatchController(self._damping, self._pi_adapter, self._pi_ids)
        saved = self._real.controller
        self._real.controller = dispatch
        self._writes_this_tick += 1
        try:
            return self._real.step(self._real_frames, now_s=now_s, live_ids=self._real_live,
                                   native_ids=self._real_native,
                                   wall_commands=self._real_wall,
                                   record_callback=forward)
        finally:
            self._real.controller = saved


class _MemberRoster:
    """Validation-only stub exposing ``.members`` like a wall session."""

    __slots__ = ("members",)

    def __init__(self, members: frozenset):
        self.members = members


class MixedAssignmentSession:
    """Route each live designated CAV by its committed plan assignment."""

    def __init__(self, *, circles_controller, pi_session, cav_ids):
        self.circles_controller = circles_controller
        self.pi_session = pi_session
        self.cav_ids = frozenset(cav_ids)
        if not self.cav_ids:
            raise ValueError("mixed assignment requires a nonempty CAV roster")
        if pi_session is not None and frozenset(pi_session.cav_ids) != self.cav_ids:
            raise ValueError("PI session roster must equal the mixed assignment roster")
        self._committed: dict[str, str] = {v: "native" for v in self.cav_ids}
        self._pending: dict[str, str | None] = {v: None for v in self.cav_ids}
        self._pending_since: dict[str, float | None] = {v: None for v in self.cav_ids}
        self._pending_horizon: dict[str, float | None] = {v: None for v in self.cav_ids}
        self._last_exec: dict[str, float | None] = {v: None for v in self.cav_ids}
        self._exec_horizon: dict[str, float | None] = {v: None for v in self.cav_ids}
        self._commit_until: dict[str, float | None] = {v: None for v in self.cav_ids}
        self._plan_id: str | None = None
        self._horizon_s = DEFAULT_COMMIT_HORIZON_S
        self._last_plan_time: float | None = None
        self._first_plan_applied = False
        self._pi_engaged = False
        self._proxy: _PISubsetProxyOwner | None = None
        self._owner = None
        self._stopped = False
        self._last_tick: float | None = None
        self._emit = None
        self._walls: dict[str, WallGroupSession] = {}
        self._wall_terms: dict[str, tuple] = {}
        self._staged_releases: set[str] = set()
        self.assignment_records: list[dict] = []
        self.wall_records: list[dict] = []

    # ------------------------------------------------------------- planner
    def apply_plan(self, plan: AssignmentPlan, *, now_s: float) -> None:
        """Stage a planner coloring (and wall groups) under the contract.

        The first plan ever applied commits immediately (initial placement is
        not a switch) and starts every CAV's execution dwell.  Every later
        plan updates each CAV's PENDING target only: repeating the currently
        pending target never restarts its stability clock, a different
        target restarts the clock from this plan time with this plan's
        horizon frozen, returning to the committed assignment cancels the
        pending switch, and execution happens in ``step`` once both the
        dwell and stability requirements hold.  Under the lenient default a
        CAV absent from the plan targets native; ``strict_roster`` plans
        must name every roster CAV.  Plan times must be monotonic
        non-decreasing.

        Wall groups bypass the per-CAV clocks: staging a wall commits its
        members to ``wall-v2`` (assembly starts at the next control tick),
        the members stay locked to their wall until it reports assembled or
        safety-aborted, and a violating plan is rejected here with an error.
        """
        if self._stopped:
            raise RuntimeError("mixed assignment session is stopped")
        if not isinstance(plan, AssignmentPlan):
            raise TypeError("plan must be an AssignmentPlan")
        try:
            now_s = float(now_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("plan time must be finite and nonnegative") from exc
        if not isfinite(now_s) or now_s < 0:
            raise ValueError("plan time must be finite and nonnegative")
        if self._last_plan_time is not None and now_s < self._last_plan_time - 1e-9:
            raise ValueError("plan times must be monotonic non-decreasing")
        unknown = set(plan.assignments) - self.cav_ids
        if unknown:
            raise ValueError(f"plan assigns CAVs outside the roster: {sorted(unknown)}")
        if plan.strict_roster and set(plan.assignments) != set(self.cav_ids):
            missing = sorted(self.cav_ids - set(plan.assignments))
            raise ValueError(f"strict_roster plan is missing assignments for {missing}")
        if "pi-external-u" in plan.assignments.values() and self.pi_session is None:
            raise ValueError("plan assigns pi-external-u but no PI session is bound")
        self._reconcile_walls(plan)
        self._horizon_s = plan.commit_horizon_s
        self._plan_id = plan.plan_id
        self._last_plan_time = now_s
        wall_members = self._active_wall_members()
        if not self._first_plan_applied:
            for v in self.cav_ids:
                mode = "wall-v2" if v in wall_members else plan.assignments.get(v, "native")
                self._committed[v] = mode
                self._pending[v] = None
                self._pending_since[v] = None
                self._pending_horizon[v] = None
                self._last_exec[v] = now_s
                self._exec_horizon[v] = plan.commit_horizon_s
                self._commit_until[v] = now_s + plan.commit_horizon_s
            self._first_plan_applied = True
            return
        for v in self.cav_ids:
            if v in wall_members:
                # Wall membership is a group action, not a per-CAV switch:
                # it commits at the next tick and clears any pending target.
                if self._committed[v] != "wall-v2":
                    self._committed[v] = "wall-v2"
                self._pending[v] = None
                self._pending_since[v] = None
                self._pending_horizon[v] = None
                continue
            if self._committed[v] == "wall-v2":
                # Member of a wall staged for release: handback to native is
                # decided by the wall lifecycle, not by this plan's clocks.
                continue
            want = plan.assignments.get(v, "native")
            if want == self._committed[v]:
                # Returning to (or holding) the committed assignment cancels
                # any pending switch.
                self._pending[v] = None
                self._pending_since[v] = None
                self._pending_horizon[v] = None
            elif want == self._pending[v]:
                # Repeating the pending target (any plan id, any horizon)
                # never restarts its stability clock; the horizon stays the
                # one frozen when this target first became pending.
                pass
            else:
                # A real target change (re)starts stability from now.
                self._pending[v] = want
                self._pending_since[v] = now_s
                self._pending_horizon[v] = plan.commit_horizon_s
            self._commit_until[v] = self._earliest_execution(v)

    def _reconcile_walls(self, plan: AssignmentPlan) -> None:
        """Validate the plan's wall groups against the active wall sessions.

        Locks enforced here (a violating plan raises, never silently
        degrades): the members of an active wall cannot change; an
        assembling wall cannot be dropped; an explicit ``release`` phase is
        refused while the wall is still assembling.  After assembly the
        planner may release at any time, explicitly or by omitting the wall.
        Wall speed may be updated at any phase (planner decides; the floor
        was validated by the plan).  All validation runs before any state
        mutates, keeping ``apply_plan`` atomic.
        """
        plan_walls = {w.wall_id: w for w in plan.walls}
        # -- validation pass (no mutation) --
        for wall_id, session in self._walls.items():
            assignment = plan_walls.get(wall_id)
            if assignment is None:
                if session.phase == PHASE_ASSEMBLY:
                    raise ValueError(
                        f"wall {wall_id} is assembling: it cannot be dropped "
                        "(safety aborts remain available in every phase)")
                continue  # absence after hold is a release
            if frozenset(assignment.members) != session.members:
                raise ValueError(
                    f"members of active wall {wall_id} are locked until it reports "
                    "assembled or safety-aborted")
            if assignment.phase == "release" and session.phase == PHASE_ASSEMBLY:
                raise ValueError(
                    f"planner cannot cancel wall {wall_id} assembly; safety aborts "
                    "remain available in every phase")
            existing_law = getattr(session, "law_version", WALL_LAW_VERSION_V21)
            if assignment.law_version != existing_law:
                raise ValueError(
                    f"wall {wall_id} cannot switch law version "
                    f"{existing_law!r} -> {assignment.law_version!r}: a wall's "
                    "law is fixed when it is staged")
        new_walls = [w for w in plan.walls if w.wall_id not in self._walls]
        overlap_space = dict(self._walls)
        for w in new_walls:
            overlap_space[w.wall_id] = _MemberRoster(frozenset(w.members))
        for w in new_walls:
            others = {k: s for k, s in overlap_space.items() if k != w.wall_id}
            validate_wall_request(w.members, w.wall_speed_mps, others)
        # -- mutation pass --
        for wall_id, session in self._walls.items():
            assignment = plan_walls.get(wall_id)
            if assignment is None:
                self._staged_releases.add(wall_id)
                continue
            session.wall_speed_mps = assignment.wall_speed_mps
            self._wall_terms[wall_id] = (assignment.hold_s, assignment.release_at_s)
            if assignment.phase == "release":
                self._staged_releases.add(wall_id)
            else:
                # A same-tick later plan may revoke a staged release.
                self._staged_releases.discard(wall_id)
        for assignment in new_walls:
            # Feasibility was fully validated above; stage the sessions now.
            if assignment.law_version == WALL_LAW_VERSION_V22:
                # WALL22-I: law v2.2 'merge-release' (explicitly named on
                # the assignment; every other value keeps the pinned v2.1
                # session class and its exact pre-WALL22-I behaviour).
                from .wall_group_v22 import WallGroupSessionV22  # noqa: PLC0415
                self._walls[assignment.wall_id] = WallGroupSessionV22(
                    assignment.wall_id, members=frozenset(assignment.members),
                    wall_speed_mps=assignment.wall_speed_mps)
            else:
                self._walls[assignment.wall_id] = WallGroupSession(
                    assignment.wall_id, members=frozenset(assignment.members),
                    wall_speed_mps=assignment.wall_speed_mps)
            self._wall_terms[assignment.wall_id] = (assignment.hold_s, assignment.release_at_s)

    def _active_wall_members(self) -> frozenset:
        members: set[str] = set()
        for session in self._walls.values():
            members.update(session.members)
        return frozenset(members)

    def _earliest_execution(self, v: str) -> float | None:
        """Earliest time the next switch could execute for CAV ``v``."""
        deadlines = []
        if self._last_exec[v] is not None:
            deadlines.append(self._last_exec[v] + self._exec_horizon[v])
        if self._pending[v] is not None:
            deadlines.append(self._pending_since[v] + self._pending_horizon[v])
        return max(deadlines) if deadlines else None

    # ------------------------------------------------------------ telemetry
    def bind_emit(self, emit) -> None:
        """Wire the workflow telemetry sink, before the first tick (wall pattern)."""
        if self._last_tick is not None:
            raise RuntimeError("mixed assignment emit sink must be bound before the first tick")
        if emit is not None and not callable(emit):
            raise TypeError("emit must be callable or None")
        self._emit = emit

    def _emit_row(self, row: dict) -> None:
        if self._emit is not None:
            self._emit(row["kind"], {k: v for k, v in row.items() if k != "kind"})

    def _controller_state(self, owner, vehicle_id: str) -> dict:
        """Expose controller internal state for the world model's
        controller_state input: the CIRCLES damping path's last issued
        target speed (the frozen controller is stateless per decide) and
        the PI external-U running target for PI-routed CAVs."""
        record = owner._last.get(vehicle_id)
        damping_target = (record.issued_target_speed_mps
                          if record is not None and record.owner == "damping" else None)
        pi_target = None
        if self.pi_session is not None:
            pi_state = getattr(getattr(self.pi_session, "adapter", None), "pi", None)
            state = getattr(pi_state, "_state", None)
            if state:
                entry = state.get(vehicle_id)
                if entry:
                    pi_target = entry[0]
        return {"damping_target_speed_mps": damping_target,
                "pi_running_target_mps": pi_target}

    def _emit_rows(self, now_s, live, intent) -> None:
        for v in sorted(live):
            wall_id = self._wall_of(v)
            session = self._walls.get(wall_id) if wall_id else None
            if session is not None and session.phase == PHASE_HOLD:
                in_phase_from = session.assembled_at_s
            else:
                in_phase_from = session.assembly_since if session is not None else None
            row = dict(kind="mixed_assignment", timestamp_s=now_s, plan_id=self._plan_id,
                       vehicle_id=v, action=intent[v], wall_id=wall_id,
                       wall_phase=(session.phase if session is not None else None),
                       time_in_phase_s=(None if in_phase_from is None else now_s - in_phase_from),
                       wall_speed_mps=(session.wall_speed_mps if session is not None else None),
                       locked=session is not None,
                       committed_until_s=self._commit_until[v],
                       pending_target=self._pending[v], pending_since_s=self._pending_since[v],
                       stability_deadline_s=(None if self._pending[v] is None else
                                             self._pending_since[v] + self._pending_horizon[v]),
                       last_execution_s=self._last_exec[v],
                       dwell_deadline_s=(None if self._last_exec[v] is None else
                                         self._last_exec[v] + self._exec_horizon[v]),
                       controller_state=self._controller_state(self._owner, v))
            self.assignment_records.append(row)
            self._emit_row(row)

    def _wall_of(self, vehicle_id: str) -> str | None:
        for wall_id, session in self._walls.items():
            if vehicle_id in session.members:
                return wall_id
        return None

    # ------------------------------------------------------------- wall api
    def _tick_walls(self, now_s: float, frames: Mapping) -> dict[str, float]:
        """Advance every active wall one control tick.

        Applies the staged planner releases, the hold-term auto releases and
        the in-session safety aborts; members of a wall that hands back this
        tick are returned to NATIVE immediately (lifecycle action, not a
        per-CAV switch).  Returns the union of member wall commands
        (absolute accelerations) for this tick.
        """
        wall_accels: dict[str, float] = {}
        for wall_id in sorted(self._walls):
            session = self._walls[wall_id]
            hold_s, release_at_s = self._wall_terms[wall_id]
            if session.phase == PHASE_HOLD:
                due = wall_id in self._staged_releases
                if (not due and release_at_s is not None and now_s >= release_at_s):
                    due = True
                if (not due and hold_s is not None and session.assembled_at_s is not None
                        and now_s - session.assembled_at_s >= hold_s):
                    due = True
                if due:
                    session.release(now_s)
            commands = session.tick(frames, now_s)
            if session.phase == PHASE_HANDBACK:
                self._handback(wall_id, now_s)
            else:
                wall_accels.update(commands)
            while session.events:
                event = session.events.pop(0)
                self.wall_records.append(event)
                if self._emit is not None:
                    self._emit("wall_event",
                               {k: v for k, v in event.items() if k != "kind"})
        self._staged_releases.clear()
        return wall_accels

    def _handback(self, wall_id: str, now_s: float) -> None:
        session = self._walls.pop(wall_id)
        self._wall_terms.pop(wall_id, None)
        self._staged_releases.discard(wall_id)
        for v in session.members:
            if self._committed.get(v) == "wall-v2":
                self._committed[v] = "native"
            self._pending[v] = None
            self._pending_since[v] = None
            self._pending_horizon[v] = None

    # ------------------------------------------------------------- clocks
    def _advance_commits(self, now_s: float) -> None:
        """Execute every pending switch whose dwell AND stability clocks hold.

        The dwell clock is frozen at the previous execution (``last_exec +
        exec_horizon``) and the stability clock at the moment the pending
        target first differed (``pending_since + pending_horizon``); neither
        can be pushed forward by later horizon-only plan changes. On
        execution the pending target commits and the dwell re-locks from
        this execution with the pending target's frozen horizon.
        """
        for v in sorted(self.cav_ids):
            if self._pending[v] is None:
                continue
            stability_ok = now_s >= self._pending_since[v] + self._pending_horizon[v]
            dwell_ok = (self._last_exec[v] is None
                        or now_s >= self._last_exec[v] + self._exec_horizon[v])
            if stability_ok and dwell_ok:
                self._committed[v] = self._pending[v]
                self._last_exec[v] = now_s
                self._exec_horizon[v] = self._pending_horizon[v]
                self._pending[v] = None
                self._pending_since[v] = None
                self._pending_horizon[v] = None
                self._commit_until[v] = self._earliest_execution(v)

    # ------------------------------------------------------- single writer
    @staticmethod
    def _assert_single_writer(paths: Mapping[str, str], records, live_designated) -> None:
        """Assert the per-tick single-writer invariant; fail loudly otherwise."""
        if set(paths) != set(live_designated):
            raise RuntimeError("assignment intent must cover every live designated CAV exactly once")
        written = [record.vehicle_id for record in records]
        if len(written) != len(set(written)) or set(written) != set(live_designated):
            raise RuntimeError("each live designated CAV must be written by exactly one "
                               "controller path per tick")

    # ---------------------------------------------------------------- step
    def step(self, owner, frames, *, now_s, live_ids, disappeared_ids=(), reappeared_ids=()):
        """One control tick; same interface and clock contract as a fleet session."""
        if self._stopped:
            raise RuntimeError("mixed assignment session is stopped")
        try:
            now_s = float(now_s)
            if not isfinite(now_s) or now_s < 0:
                raise ValueError("mixed assignment time must be finite and nonnegative")
            if abs(now_s - round(now_s)) > 1e-9:
                raise ValueError("mixed assignment steps must lie on the one-second grid")
            if self._last_tick is not None and abs(now_s - self._last_tick - 1.) > 1e-9:
                raise ValueError("mixed assignment requires every one-second control tick")
            if (self._last_plan_time is not None
                    and now_s < self._last_plan_time - 1e-9):
                raise ValueError("step time precedes the last submitted plan time")
            if self._owner is None:
                if owner.cav_ids != self.cav_ids:
                    raise ValueError("mixed assignment owner roster differs from registered CAVs")
                if owner.controller is not self.circles_controller:
                    raise ValueError("ownership default controller is not the registered "
                                     "damping instance")
                self._owner = owner
            elif owner is not self._owner:
                raise ValueError("mixed assignment cannot replace its CommandOwnership instance")
            live = frozenset(live_ids) & self.cav_ids
            wall_accels = {v: a for v, a in self._tick_walls(now_s, frames).items()
                           if v in live}
            self._advance_commits(now_s)
            intent = {v: self._committed[v] for v in live}
            paths = {}
            for v, mode in intent.items():
                if mode == "wall-v2" and v not in wall_accels and self._wall_of(v) is None:
                    # defensive: committed wall members always have an active wall
                    raise RuntimeError(f"wall-v2 CAV {v} has no active wall session")
                paths[v] = ("native" if mode == "native"
                            else "pi-external-u" if mode == "pi-external-u"
                            else "wall-v2" if mode == "wall-v2" else "damping")
            native_ids = {v for v in live if intent[v] == "native"}
            pi_active = frozenset(v for v in live if intent[v] == "pi-external-u")
            if self._pi_engaged or pi_active:
                records = self._step_via_pi(frames, now_s=now_s, live=live,
                                            native_ids=native_ids, pi_active=pi_active,
                                            wall_accels=wall_accels,
                                            disappeared_ids=disappeared_ids,
                                            reappeared_ids=reappeared_ids)
            else:
                records = owner.step(frames, now_s=now_s, live_ids=live_ids,
                                     native_ids=native_ids, wall_commands=wall_accels)
            self._assert_single_writer(paths, records, live)
            self._last_tick = now_s
            self._emit_rows(now_s, live, intent)
            return records
        except BaseException:
            self._stopped = True
            raise

    def _step_via_pi(self, frames, *, now_s, live, native_ids, pi_active, wall_accels,
                     disappeared_ids, reappeared_ids):
        """Route the tick through the PI adapter with the committed PI subset.

        Once engaged, the adapter is ticked every control tick (even with an
        empty subset) because it requires contiguous one-second ticks; an
        empty-subset tick writes the whole roster through the dispatching
        controller with an empty PI set, i.e. plain damping/native behavior.
        Wall commands are staged on the proxy and forwarded with the single
        ownership write.
        """
        if self._proxy is None:
            self._proxy = _PISubsetProxyOwner(self._owner, self.pi_session.adapter,
                                              self.circles_controller)
        self._pi_engaged = True
        proxy = self._proxy
        pi_frames = {v: frames[v] for v in pi_active if v in frames}
        # PIFIX (2026-10-01): the registered PI adapter classifies its tick as
        # ``live - native - wall`` and REQUIRES every wall command id to be a
        # subset of ``live_ids`` (native & wall disjoint, (native|wall) <=
        # live), so a tick carrying both a PI CAV and an active wall crashed
        # ("live/native/wall masks must contain only disjoint registered
        # CAVs", PL-EXP1d w8526 @1761 s).  Route the committed PI subset UNION
        # the active wall members as the adapter's live mask; wall ids are
        # excluded from PI proposal preparation exactly as before.  The proxy
        # still performs the single real ownership write over the full roster
        # with the real native mask and the dispatcher-staged wall commands, so
        # each CAV keeps its exact command (wall / PI / native) and nothing is
        # issued twice.
        pi_live = pi_active | frozenset(wall_accels)
        proxy.begin_tick(real_frames=frames, real_live=live, native_ids=native_ids,
                         pi_ids=pi_live, wall_commands=wall_accels)
        try:
            # Full disappearance/reappearance sets, mirroring the fleet
            # session's PI branch; ids outside the adapter's live mask are
            # harmless lifecycle no-ops for the adapter.
            return self.pi_session.step(proxy, pi_frames, now_s=now_s, live_ids=pi_live,
                                        native_ids=(), wall_commands=wall_accels,
                                        departed_ids=tuple(disappeared_ids),
                                        id_reuse_ids=tuple(reappeared_ids))
        finally:
            proxy.end_tick()
