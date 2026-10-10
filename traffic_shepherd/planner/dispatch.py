"""PlannerDecision -> dispatcher adapter (PL01-B, WM-40 alignment).

Target: the FINAL ``mixed_assignment/3`` dispatcher (WM-40, merged in
the research-dense tree): a full per-CAV map (``assignments``; absent CAV = NATIVE
in lenient mode, but the planner always sends the complete map and runs
``strict_roster=True``), a tuple of walls ``{wall_id, lane-ordered members
(>= 2), wall_speed_mps > 2.0, phase, exactly one of hold_s / release_at_s}``,
and ``commit_horizon_s = 0`` (the planner re-decides every tick, dwell 0).

Release encoding (v3): after hold, a released wall is OMITTED from ``walls``
and its members are assigned NATIVE in the same plan (omission after hold =
release). A wall carrying phase ``"release"`` is never emitted -- the adapter
raises :class:`WallAssignmentError` for it, and it must never fire for an
ASSEMBLING wall (an assembling wall is neither released nor omitted; it ends
only by the dispatcher's own safety abort, reported via controller_state).

:data:`DispatcherPlanV3` / :data:`DispatcherWallV3` mirror the final v3 field
names locally (all schema names live in this one module). The fallback emits
today's ``mixed_assignment/2`` ``AssignmentPlan`` shape for plans with NO
wall (full map, commit_horizon_s = config.v2_commit_horizon_s, the minimal
positive horizon) and raises :class:`WallAssignmentError` for any wall --
mixed_assignment/2 executes walls only as a documented stub and rejects a
planner-specified member set. :func:`to_mixed_assignment_v3_real` builds the
REAL ``traffic_shepherd.mixed_assignment`` v3 objects (lazy import) and
raises when the installed module's VERSION is not ``mixed_assignment/3``.

Zero simulation: no SUMO/traci/libsumo import anywhere in this package.
"""
import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from .config import PlannerConfig
from .interface import PlanCandidate, plan_key

DISPATCH_V3_SCHEMA = "mixed_assignment/3"
DISPATCH_V2_SCHEMA = "mixed_assignment/2"
WALL_PHASES_V3 = ("assemble", "hold", "release")
#: mixed_assignment/3 per-CAV mode vocabulary (fleet_switch.MODES).
WALL_MODES_V3 = frozenset(("native", "damping", "pi-external-u", "wall-v2"))
#: Dispatcher wall-speed floor: ``wall_speed_mps`` must be strictly above it.
SAFE_WALL_SPEED_MIN_MPS = 2.0


class WallAssignmentError(ValueError):
    """A plan cannot be encoded for the target dispatcher schema.

    Raised for planner plans that still carry an explicit ``release`` wall
    (v3 encodes releases after hold by wall omission with members -> native),
    and for the mixed_assignment/2 fallback when a wall is present (v2's
    wall execution path is a documented stub; dispatch walls via
    :func:`to_dispatcher_v3` / :func:`to_mixed_assignment_v3_real`).
    """


@dataclass(frozen=True)
class DispatcherWallV3:
    """One wall in the mixed_assignment/3 walls tuple (final WM-40 names)."""

    wall_id: str
    members: Tuple[str, ...]  # lane-ordered, exactly one CAV per lane; >= 2
    wall_speed_mps: float
    phase: str = "assemble"  # assemble | hold | release
    hold_s: Optional[float] = None  # seconds AFTER assembly completes
    release_at_s: Optional[float] = None  # ABSOLUTE episode time

    def __post_init__(self):
        if not isinstance(self.wall_id, str) or not self.wall_id.strip():
            raise ValueError("wall_id must be a non-empty string")
        members = tuple(self.members)
        if (len(members) < 2 or len(set(members)) != len(members)
                or any(not isinstance(m, str) or not m for m in members)):
            raise ValueError(
                "wall members must be at least two distinct non-empty cav ids "
                "(a wall spans a cross-section, lane-ordered, one per lane)")
        object.__setattr__(self, "members", members)
        speed = self.wall_speed_mps
        if isinstance(speed, bool):
            raise ValueError("wall_speed_mps must be a finite number > 2.0")
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            raise ValueError("wall_speed_mps must be a finite number > 2.0") from None
        if not math.isfinite(speed) or speed <= SAFE_WALL_SPEED_MIN_MPS:
            raise ValueError(
                f"wall_speed_mps must be finite and > {SAFE_WALL_SPEED_MIN_MPS} "
                f"(got {speed})")
        object.__setattr__(self, "wall_speed_mps", speed)
        if self.phase not in WALL_PHASES_V3:
            raise ValueError(
                f"phase must be one of {WALL_PHASES_V3}, got {self.phase!r}")
        if (self.hold_s is None) == (self.release_at_s is None):
            raise ValueError("exactly one of hold_s / release_at_s must be set")
        if self.hold_s is not None:
            hold = float(self.hold_s)
            if isinstance(self.hold_s, bool) or not math.isfinite(hold) or hold <= 0:
                raise ValueError("hold_s must be finite and positive")
            object.__setattr__(self, "hold_s", hold)
        if self.release_at_s is not None:
            release_at = float(self.release_at_s)
            if (isinstance(self.release_at_s, bool) or not math.isfinite(release_at)
                    or release_at < 0):
                raise ValueError(
                    "release_at_s must be finite and >= 0 (absolute episode time)")
            object.__setattr__(self, "release_at_s", release_at)


@dataclass(frozen=True)
class DispatcherPlanV3:
    """The mixed_assignment/3 dispatcher plan (final WM-40 names, strict)."""

    assignments: Mapping[str, str]  # FULL map; strict_roster: absent CAV is an error
    plan_id: str
    commit_horizon_s: float = 0.0  # WM-40: dwell 0, the planner re-decides
    walls: Tuple[DispatcherWallV3, ...] = ()
    strict_roster: bool = True
    schema: str = DISPATCH_V3_SCHEMA

    def __post_init__(self):
        if not isinstance(self.plan_id, str) or not self.plan_id.strip():
            raise ValueError("plan_id must be a non-empty string")
        horizon = self.commit_horizon_s
        if (isinstance(horizon, bool) or not math.isfinite(float(horizon))
                or float(horizon) < 0):
            raise ValueError("commit_horizon_s must be finite and >= 0")
        object.__setattr__(self, "commit_horizon_s", float(horizon))
        if not isinstance(self.strict_roster, bool):
            raise ValueError("strict_roster must be a bool")
        if (not isinstance(self.assignments, Mapping)
                or isinstance(self.assignments, str)):
            raise ValueError("assignments must be a mapping of cav id to mode")
        assignments = dict(self.assignments)
        invalid = sorted(c for c, mode in assignments.items()
                         if mode not in WALL_MODES_V3)
        if invalid:
            raise ValueError(
                f"invalid assignment modes for {invalid}; "
                f"vocabulary is {sorted(WALL_MODES_V3)}")
        object.__setattr__(self, "assignments", assignments)
        walls = tuple(self.walls)
        if not all(isinstance(w, DispatcherWallV3) for w in walls):
            raise ValueError("walls must be DispatcherWallV3 entries")
        ids = [w.wall_id for w in walls]
        if len(ids) != len(set(ids)):
            raise ValueError("wall ids must be unique within a plan")
        seen: set = set()
        for wall in walls:
            overlap = seen & set(wall.members)
            if overlap:
                raise ValueError(
                    f"members {sorted(overlap)} are assigned to two walls "
                    "of one plan")
            seen.update(wall.members)
        wall_assigned = {c for c, mode in assignments.items() if mode == "wall-v2"}
        if wall_assigned != seen:
            raise ValueError(
                "wall-v2 assignments and the union of wall members must coincide")
        object.__setattr__(self, "walls", walls)


@dataclass(frozen=True)
class AssignmentPlanV2:
    """Local mirror of today's ``mixed_assignment/2`` AssignmentPlan shape.

    Fields mirror the live dispatcher: ``assignments`` (full per-CAV mode
    map), ``plan_id``, positive ``commit_horizon_s``, and the wall fields
    (``wall_members`` + ``wall_candidate`` ``{pair, wall_speed_mps, hold_s}``)
    which this adapter never fills -- v2 rejects walls.
    """

    assignments: Mapping[str, str]
    plan_id: str
    commit_horizon_s: float
    wall_members: frozenset = frozenset()
    wall_candidate: Optional[Mapping] = None

    def __post_init__(self):
        if not isinstance(self.plan_id, str) or not self.plan_id.strip():
            raise ValueError("plan id must be a nonempty string")
        if float(self.commit_horizon_s) <= 0:
            raise ValueError("commit horizon must be finite and positive")
        if self.wall_members or self.wall_candidate is not None:
            raise WallAssignmentError(
                "mixed_assignment/2 rejects walls: the wall execution path is "
                "a documented stub; dispatch walls via mixed_assignment/3 "
                "(to_dispatcher_v3)")


def plan_id_for(plan: PlanCandidate) -> str:
    """Deterministic plan id: sha256 of the canonical plan key (first 12 hex)."""
    digest = hashlib.sha256(repr(plan_key(plan)).encode("utf-8")).hexdigest()
    return f"planner-{digest[:12]}"


def to_dispatcher_v3(decision, config: Optional[PlannerConfig] = None,
                     *, t0_s: Optional[float] = None) -> DispatcherPlanV3:
    """PlannerDecision -> mixed_assignment/3 plan (strict, full map, dwell 0).

    Wall members leave ``assignments`` in the planner's plan candidate; here
    they are (re-)assigned ``wall-v2`` so the union of wall members coincides
    with the wall-v2 assignments (v3 plan invariant).

    Release encoding: toward predict() the planner keeps a released wall in
    the candidate as ``phase="release"`` (release now) with its locked
    members; toward the WM-40 dispatcher the release stays encoded as
    OMISSION after hold -- this adapter drops release-phase walls and
    assigns their members native in ``assignments`` (an ASSEMBLING wall is
    never released or omitted; it ends only by the dispatcher's safety
    abort, which the loop encodes the same way).

    Dispatcher time (contract §8.4): predict ``release_at_s`` is an offset
    from t0; pass ``t0_s`` to emit the ABSOLUTE episode time
    ``t0_s + release_at_s``. ``hold_s`` is already relative to the assembly
    and passes through unchanged.
    """
    cfg = config if config is not None else PlannerConfig()
    assignments = dict(decision.plan.per_cav)
    walls = []
    for wall in decision.plan.walls:
        if wall.phase == "release":
            # omission after hold: the wall disappears from the dispatcher
            # plan and its members are native
            assignments.update({member: "native" for member in wall.members})
            continue
        assignments.update({member: "wall-v2" for member in wall.members})
        release_at = wall.release_at_s
        if release_at is not None and t0_s is not None:
            release_at = float(t0_s) + float(release_at)
        walls.append(DispatcherWallV3(
            wall_id=wall.wall_id, members=tuple(wall.members),
            wall_speed_mps=wall.speed_mps, phase=wall.phase,
            hold_s=wall.hold_s, release_at_s=release_at))
    return DispatcherPlanV3(assignments=assignments,
                            plan_id=plan_id_for(decision.plan),
                            commit_horizon_s=cfg.v3_commit_horizon_s,
                            walls=tuple(walls))


def installed_mixed_assignment_v3():
    """The installed :mod:`traffic_shepherd.mixed_assignment` module, REFUSING
    any version other than ``mixed_assignment/3``.

    The ONE version guard shared by every real v3 plan construction
    (:func:`to_mixed_assignment_v3_real` and
    :func:`dispatcher_plan_to_mixed_assignment_v3_real`): raises
    :class:`WallAssignmentError` for anything else -- never a silent /2
    fallback."""
    import importlib

    ma = importlib.import_module("traffic_shepherd.mixed_assignment")
    version = getattr(ma, "VERSION", None)
    if version != DISPATCH_V3_SCHEMA:
        raise WallAssignmentError(
            f"installed traffic_shepherd.mixed_assignment VERSION "
            f"{version!r} is not {DISPATCH_V3_SCHEMA!r}: the planner targets "
            "the final WM-40 dispatcher schema; real v3 plan construction "
            "refuses any other installed version")
    return ma


def dispatcher_plan_to_mixed_assignment_v3_real(dispatcher_plan: DispatcherPlanV3):
    """The mirrored :class:`DispatcherPlanV3` -> the REAL ``mixed_assignment/3``
    ``AssignmentPlan`` (walls EXECUTED, not refused).

    Builds the real ``AssignmentPlan``/``WallAssignment`` objects from the
    already-validated mirror (final WM-40 field names), through the shared
    :func:`installed_mixed_assignment_v3` version guard. This is the
    conversion the PL-04 closed-loop runner materializes EVERY dispatcher
    plan through (master ``apply_plan`` + the branch plan factory)."""
    ma = installed_mixed_assignment_v3()
    walls = tuple(
        ma.WallAssignment(wall_id=wall.wall_id, members=wall.members,
                          wall_speed_mps=wall.wall_speed_mps, phase=wall.phase,
                          hold_s=wall.hold_s, release_at_s=wall.release_at_s)
        for wall in dispatcher_plan.walls)
    return ma.AssignmentPlan(assignments=dict(dispatcher_plan.assignments),
                             plan_id=str(dispatcher_plan.plan_id),
                             commit_horizon_s=float(dispatcher_plan.commit_horizon_s),
                             walls=walls,
                             strict_roster=bool(dispatcher_plan.strict_roster))


def to_mixed_assignment_v3_real(decision, config: Optional[PlannerConfig] = None,
                                *, t0_s: Optional[float] = None):
    """PlannerDecision -> the REAL ``mixed_assignment/3`` AssignmentPlan.

    Lazily imports :mod:`traffic_shepherd.mixed_assignment` and builds the
    real ``AssignmentPlan``/``WallAssignment`` objects (final WM-40 field
    names, mirrored in :data:`DispatcherPlanV3`/:data:`DispatcherWallV3`,
    which run the same planner-side validation first). Raises
    :class:`WallAssignmentError` when the installed module's VERSION is not
    ``mixed_assignment/3``.
    """
    ma = installed_mixed_assignment_v3()
    mirror = to_dispatcher_v3(decision, config, t0_s=t0_s)
    walls = tuple(
        ma.WallAssignment(wall_id=wall.wall_id, members=wall.members,
                          wall_speed_mps=wall.wall_speed_mps, phase=wall.phase,
                          hold_s=wall.hold_s, release_at_s=wall.release_at_s)
        for wall in mirror.walls)
    return ma.AssignmentPlan(assignments=dict(mirror.assignments),
                             plan_id=mirror.plan_id,
                             commit_horizon_s=mirror.commit_horizon_s,
                             walls=walls,
                             strict_roster=mirror.strict_roster)


def to_mixed_assignment_v2(decision, config: Optional[PlannerConfig] = None) -> AssignmentPlanV2:
    """PlannerDecision -> today's mixed_assignment/2 shape (0 walls only).

    Raises :class:`WallAssignmentError` for any wall (v2 rejects walls).
    """
    cfg = config if config is not None else PlannerConfig()
    if decision.plan.walls:
        raise WallAssignmentError(
            f"plan carries {len(decision.plan.walls)} wall(s) "
            f"({', '.join(w.wall_id for w in decision.plan.walls)}): "
            "mixed_assignment/2 rejects walls; use to_dispatcher_v3")
    return AssignmentPlanV2(
        assignments=dict(decision.plan.per_cav),
        plan_id=plan_id_for(decision.plan),
        commit_horizon_s=cfg.v2_commit_horizon_s)
