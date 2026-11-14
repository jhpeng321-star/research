"""Planner-side types mirroring the frozen world-model contract wd01_predict/1.

Mirrors the WD-01 worktree's frozen PREDICT_CONTRACT_v1.md
(sha256 2619390e95ae2d37fcdffd6dc1fefeb5141ef9d2dc1cfb2676b9c02559bd83a8,
verified 2026-09-25; read-only, never edited). Types and structural
validation only -- no numerics, no simulation imports (zero-simulation rule).

PL01-A scope: ``WallSpec`` exists here exactly as the contract declares it,
but wall proposal / lifecycle logic is PL01-B
(output/planner_v1/PL01_PLAN.md, task sequence). The decision loop accepts a
current plan that already contains walls with locked members and leaves them
untouched.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

CONTRACT_VERSION = "wd01_predict/1"
CONTRACT_VERSION_V1_1 = "wd01_predict/1.1"
#: predict contract v2 (the WD-01 worktree's PREDICT_CONTRACT_v2.md,
#: sha256 db475371...0775, the post-review text with REQUIRED
#: future_has_closing_pair; a delta on v1/v1.1, both frozen). Dual-version:
#: v1, v1.1 and v2 are all supported.
#: Gating item G-2 (§8, 2026-09-27): v2 meta additionally carries
#: ``relative_safety_validated: bool`` and
#: ``relative_safety_validation_id`` (str; non-None iff the flag is True;
#: a MISSING field is treated as False -- fail closed). Enforced by
#: safety.enforce_future_gate next to G-1.
CONTRACT_VERSION_V2 = "wd01_predict/2"
CONTINUATIONS: Tuple[str, ...] = ("planner", "native")

#: Contract v2.1 "Part A v2 general" (the WD-01 worktree's frozen
#: PREDICT_CONTRACT_v2_1.md, sha256
#: ba52820a6559cfe766daa1be4aa29c479ec7b5c9e9760c1b5ac21106b2b361d2,
#: verified 2026-09-29; read-only, never edited). It amends v2: M = 1 is
#: the deterministic Part A v2 general's ensemble size (a Part B ensemble
#: may keep M > 1); the planner accepts M = 1 EVERYWHERE (abstain counts
#: over K; the switch/veto rules pair over all M x K cells -- all existing
#: statistics flatten over cells, so M = 1 needs no special case).
#: ``StateSample.arrivals`` stays OPTIONAL (None or empty is never refused).
#: The four per-cell fields and four meta fields below are OPTIONAL on
#: EVERY predictor; NO veto and NO abstain depends on them -- the planner
#: LOGS them (closed_loop decisions.jsonl / RUN_RECEIPT) and that is all.
#: The planner decides on TOTAL delay (delay_vs_native_s); a cross-
#: prediction comparison of ``delay_part_a_s`` is legal only between equal
#: ``part_a_scope`` values (rules.compare_part_a raises the NAMED
#: PartAScopeMismatchError otherwise); totals are always comparable.
PART_A_SCOPES: Tuple[str, ...] = ("tracked", "all_vehicles")
PART_A_V21_CELL_FIELDS: Tuple[str, ...] = (
    "n_tracked",
    "n_exited_tracked",
    "n_free_road_assumed",
    "untracked_arrivals_expected",
)
PART_A_V21_META_FIELDS: Tuple[str, ...] = (
    "part_a_model",
    "param_set_id",
    "recon_schema_version",
    "part_a_scope",
)

#: Per-CAV action vocabulary of ``PlanCandidate.per_cav`` (contract section 3.3).
#: The order is also the deterministic tie-break order (native first).
ACTIONS: Tuple[str, ...] = ("native", "damping", "pi-external-u")
ACTION_RANK = {name: rank for rank, name in enumerate(ACTIONS)}

WALL_PHASES: Tuple[str, ...] = ("assemble", "hold", "release")

#: ``abort_reason`` int8 codes (contract section 4.4).
ABORT_REASONS = {0: "none", 1: "safety", 2: "assembly_timeout"}

TRAJECTORY_COLUMNS = 4  # (t_s, s_m, lane_id, speed_mps)
LANE_MAIN_LEFT, LANE_MAIN_RIGHT, LANE_ACCEL = 0.0, 1.0, 2.0


@dataclass(frozen=True)
class WallSpec:
    """One wall (contract section 3.3). Members are ordered by lane index.

    Exactly one of ``release_at_s`` / ``hold_s`` must be set (contract
    validation rule; violations raise, never guess).
    """

    wall_id: str
    members: Tuple[str, ...]
    speed_mps: float
    phase: str
    release_at_s: Optional[float] = None
    hold_s: Optional[float] = None

    def __post_init__(self):
        if not self.wall_id:
            raise ValueError("wall_id must be non-empty")
        if not self.members:
            raise ValueError("a wall needs at least one member")
        if len(set(self.members)) != len(self.members):
            raise ValueError("wall members must be unique")
        if self.phase not in WALL_PHASES:
            raise ValueError(f"phase must be one of {WALL_PHASES}, got {self.phase!r}")
        if (self.release_at_s is None) == (self.hold_s is None):
            raise ValueError("exactly one of release_at_s / hold_s must be set")


@dataclass(frozen=True)
class PlanCandidate:
    """A full plan (contract section 3.3): v1 plans are constant from t0.

    ``per_cav`` is the FULL map of every CAV that is not a wall member; wall
    members never appear in it. Structural violations raise.
    """

    per_cav: Mapping[str, str]
    walls: Tuple[WallSpec, ...] = ()
    plan_version: int = 1

    def __post_init__(self):
        if self.plan_version != 1:
            raise ValueError("only plan_version 1 exists (wd01_predict/1)")
        object.__setattr__(self, "per_cav", dict(self.per_cav))
        for cav, action in self.per_cav.items():
            if action not in ACTIONS:
                raise ValueError(f"per_cav[{cav!r}] action {action!r} not in {ACTIONS}")
        seen = set()
        for wall in self.walls:
            if wall.wall_id in seen:
                raise ValueError(f"duplicate wall_id {wall.wall_id!r}")
            seen.add(wall.wall_id)
        members = [m for w in self.walls for m in w.members]
        if len(set(members)) != len(members):
            raise ValueError("wall membership sets must be disjoint")
        overlap = set(members) & set(self.per_cav)
        if overlap:
            raise ValueError(f"wall members must not appear in per_cav: {sorted(overlap)}")

    @property
    def free_cavs(self) -> Tuple[str, ...]:
        """CAVs this plan assigns individually (everyone but wall members)."""
        return tuple(sorted(self.per_cav))


class StateSample(Protocol):
    """One state sample (contract section 3.1): structural view only.

    ``hybrid`` is the reconstructor hybrid state, ``arrivals`` the arrival
    draw (common random numbers across plans), ``meta`` carries
    ``sample_id``, ``time_s``, ``merge_observed`` and ``coverage``. The
    planner reads only ``meta.merge_observed`` (abstain rule).

    Contract v2.1 section 2: ``arrivals`` is OPTIONAL -- ``None`` or an
    empty payload is accepted and never refused anywhere on the planner
    side (Part A v2 general ignores it and inserts no vehicles; predictors
    that use it, like perfect_sim, simply see no draws to install). The
    planner never reads it in any decision rule.

    PL01-E4 interface request to WD-01: the contract state sample has no
    observed-frame-age field, so the planner reads an OPTIONAL ``frames``
    mapping ({vehicle_id: last_frame_time_s}) here for the wall-proposer
    liveness filter; when absent, the hybrid ``tracked`` observed-vehicle
    layer counts as a frame at the sample time.
    """

    hybrid: Any
    arrivals: Any
    meta: Mapping[str, Any]
    frames: Optional[Mapping[str, float]] = None


@dataclass(frozen=True)
class WallPrediction:
    """Per-wall output block (contract section 4.4).

    v2 additive heads (None = not reported; REQUIRED for a v2 wall):
    ``p_assembly_completes`` / ``expected_assembly_time_s`` (NaN only where
    p_assembly_completes == 0) / ``p_assembly_abort`` — all [M, K]; no
    native_ versions (contract v2 section 3).
    """

    assembled: np.ndarray  # bool [M, K]
    assembled_at_s: np.ndarray  # float [M, K], NaN where not assembled
    safety_abort: np.ndarray  # bool [M, K]
    abort_reason: np.ndarray  # int8 [M, K] (0 none / 1 safety / 2 assembly_timeout)
    # -- contract v2 future-wall heads (None = not reported) --
    p_assembly_completes: Optional[np.ndarray] = None  # float [M, K]
    expected_assembly_time_s: Optional[np.ndarray] = None  # float [M, K]
    p_assembly_abort: Optional[np.ndarray] = None  # float [M, K]


@dataclass(frozen=True)
class Prediction:
    """World-model output for ONE plan (contract section 4).

    Every [M, K] array is aligned across plans and the shared native
    reference (paired comparison).
    """

    delay_vs_native_s: np.ndarray  # [M, K]; negative = saving
    plan_delay_s: np.ndarray  # [M, K]
    native_delay_s: np.ndarray  # [M, K], shared per call
    delay_part_a_s: np.ndarray  # [M, K]
    delay_part_b_s: np.ndarray  # [M, K]
    collision: np.ndarray  # bool [M, K], Part A window
    min_ttc_s: np.ndarray  # [M, K]
    ttc_lt3_exposure_s: np.ndarray  # [M, K]
    native_ttc_lt3_exposure_s: np.ndarray  # [M, K]
    hard_brake_s: np.ndarray  # [M, K]
    native_hard_brake_s: np.ndarray  # [M, K]
    native_collision: np.ndarray  # bool [M, K]
    native_min_ttc_s: np.ndarray  # [M, K]
    trajectories_tracked: Mapping[str, np.ndarray] = field(default_factory=dict)
    native_trajectories_tracked: Mapping[str, np.ndarray] = field(default_factory=dict)
    trajectories_hidden: Optional[Any] = None
    walls: Mapping[str, WallPrediction] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)
    # -- contract v1.1 optional diagnostics (None = not reported; a 1.1
    # prediction that omits them is invalid and fails closed in safety.py) --
    cell_valid: Optional[np.ndarray] = None  # bool [M, K]
    native_cell_valid: Optional[np.ndarray] = None  # bool [M, K]
    has_closing_pair: Optional[np.ndarray] = None  # bool [M, K]
    native_has_closing_pair: Optional[np.ndarray] = None  # bool [M, K]
    # -- contract v2 (wd01_predict/2): future heads beyond the commit
    # window. None = not reported (v1/v1.1); REQUIRED on a v2 prediction
    # (missing -> fail closed, safety.py). All [M, K] with native_ twins.
    future_ttc_lt3_exposure_s: Optional[np.ndarray] = None
    native_future_ttc_lt3_exposure_s: Optional[np.ndarray] = None
    future_hard_brake_s: Optional[np.ndarray] = None
    native_future_hard_brake_s: Optional[np.ndarray] = None
    p_future_collision: Optional[np.ndarray] = None
    native_p_future_collision: Optional[np.ndarray] = None
    future_min_ttc_q05_s: Optional[np.ndarray] = None
    native_future_min_ttc_q05_s: Optional[np.ndarray] = None
    future_cell_valid: Optional[np.ndarray] = None  # bool [M, K]
    native_future_cell_valid: Optional[np.ndarray] = None  # bool [M, K]
    # REQUIRED on v2 (post-review contract, sha db475371...0775): NaN q05
    # passes only in a valid future cell where this is False.
    future_has_closing_pair: Optional[np.ndarray] = None  # bool [M, K]
    native_future_has_closing_pair: Optional[np.ndarray] = None  # bool [M, K]
    # -- contract v2.1 (Part A v2 general) OPTIONAL per-cell fields; None =
    # not reported (perfect_sim and the mock omit them and stay valid).
    # [M, K] when present. LOGGING-ONLY: no veto and no abstain reads them.
    n_tracked: Optional[np.ndarray] = None  # vehicles tracked at t
    n_exited_tracked: Optional[np.ndarray] = None  # tracked, left before t+D
    n_free_road_assumed: Optional[np.ndarray] = None  # optimistic-bias flag
    untracked_arrivals_expected: Optional[np.ndarray] = None  # informational
    # -- TTC<1.5 s exposure (T-A, owner addendum 2): OPTIONAL additions on
    # top of the frozen v2.1 field set. ``ttc_lt3_exposure_s`` keeps its
    # frozen 3 s meaning; these carry the SAME quantity at the 1.5 s
    # threshold (the primary relative-TTC metric, PlannerConfig.
    # relative_ttc_threshold_s). None = not reported (safety.py fails
    # closed with a named error only when the config SELECTS them).
    ttc_lt1p5_exposure_s: Optional[np.ndarray] = None  # [M, K]
    native_ttc_lt1p5_exposure_s: Optional[np.ndarray] = None  # [M, K]
    future_ttc_lt1p5_exposure_s: Optional[np.ndarray] = None  # [M, K] (v2)
    native_future_ttc_lt1p5_exposure_s: Optional[np.ndarray] = None  # [M, K]


def part_a_scope(prediction: "Prediction") -> Optional[str]:
    """The prediction's meta ``part_a_scope`` (contract v2.1 section 3),
    None when the predictor does not report it."""
    scope = (prediction.meta or {}).get("part_a_scope")
    return None if scope is None else str(scope)


class WorldModel(Protocol):
    """The predict contract API (contract section 2).

    ``predict_many`` is the primitive; ``predict`` is one plan and defaults to
    the first element of ``predict_many``. The native reference is computed
    ONCE per call on the same samples and shared by every returned Prediction.
    """

    def predict_many(
        self,
        state_samples: Sequence[StateSample],
        plans: Sequence[PlanCandidate],
        *,
        controller_state: Optional[Mapping[str, Mapping[str, Any]]] = None,
        horizon_s: float = 600.0,
        trajectory_s: Optional[float] = None,
        # contract v2 (wd01_predict/2): the plan candidate is the action for
        # the commit window [t0, t0+commit_s]; after commit_s the branch
        # continues under ``continuation`` ("planner" pi_k or "native").
        # Delta is prediction semantics only -- the planner still re-decides
        # every tick. v1/v1.1 implementations ignore both.
        commit_s: float = 30.0,
        continuation: str = "planner",
    ) -> "list[Prediction]": ...

    def predict(
        self,
        state_samples: Sequence[StateSample],
        plan: PlanCandidate,
        controller_state: Optional[Mapping[str, Mapping[str, Any]]] = None,
        **kw,
    ) -> Prediction:
        return self.predict_many(state_samples, [plan], controller_state=controller_state, **kw)[0]


def plan_key(plan: PlanCandidate) -> tuple:
    """Canonical hashable key of a plan's content (insertion-order independent)."""
    walls = tuple(
        (
            w.wall_id,
            tuple(w.members),
            float(w.speed_mps),
            w.phase,
            None if w.release_at_s is None else float(w.release_at_s),
            None if w.hold_s is None else float(w.hold_s),
        )
        for w in plan.walls
    )
    return (int(plan.plan_version), tuple(sorted(plan.per_cav.items())), walls)


def locked_entries(controller_state):
    """The ``locked: true`` entries of a controller_state (contract section 3.2).

    A locked CAV may appear only as a member of its existing wall. Raises if a
    locked entry does not name its wall.
    """
    if controller_state is None:
        return {}
    locked = {}
    for cav, entry in controller_state.items():
        if not entry.get("locked"):
            continue
        wall_id = entry.get("wall_id")
        if not wall_id:
            raise ValueError(f"locked CAV {cav!r} must name its wall_id")
        locked[cav] = entry
    return locked


def with_actions(plan: PlanCandidate, actions: Mapping[str, str]) -> PlanCandidate:
    """A copy of ``plan`` with per-CAV actions replaced; walls carried untouched.

    Passing a wall member raises (contract rule: members never appear in
    per_cav), which structurally enforces the rule that locked wall members
    are never re-assigned.
    """
    per_cav = dict(plan.per_cav)
    per_cav.update(actions)
    return PlanCandidate(per_cav=per_cav, walls=plan.walls, plan_version=plan.plan_version)


def with_wall(plan: PlanCandidate, wall: WallSpec) -> PlanCandidate:
    """A copy of ``plan`` with ``wall`` added; its members leave per_cav.

    The members must currently be free (present in per_cav): wall members
    never appear in per_cav (contract section 3.3).
    """
    missing = [m for m in wall.members if m not in plan.per_cav]
    if missing:
        raise ValueError(f"wall members must be free CAVs, not free: {missing}")
    per_cav = {c: a for c, a in plan.per_cav.items() if c not in wall.members}
    return PlanCandidate(
        per_cav=per_cav, walls=plan.walls + (wall,), plan_version=plan.plan_version
    )


def without_wall(plan: PlanCandidate, wall_id: str) -> PlanCandidate:
    """A copy of ``plan`` without wall ``wall_id``; members return to native.

    N1 handback (contract section 5): dissolve the wall and put its members
    back into per_cav at native (the next individual check re-optimizes them).
    """
    wall = next((w for w in plan.walls if w.wall_id == wall_id), None)
    if wall is None:
        raise ValueError(f"wall {wall_id!r} is not in the plan")
    walls = tuple(w for w in plan.walls if w.wall_id != wall_id)
    per_cav = dict(plan.per_cav)
    for member in wall.members:
        per_cav[member] = "native"
    return PlanCandidate(per_cav=per_cav, walls=walls, plan_version=plan.plan_version)


def with_wall_phase(plan: PlanCandidate, wall_id: str, phase: str) -> PlanCandidate:
    """A copy of ``plan`` with the phase of wall ``wall_id`` replaced."""
    if phase not in WALL_PHASES:
        raise ValueError(f"phase must be one of {WALL_PHASES}, got {phase!r}")
    wall = next((w for w in plan.walls if w.wall_id == wall_id), None)
    if wall is None:
        raise ValueError(f"wall {wall_id!r} is not in the plan")
    walls = tuple(replace(wall, phase=phase) if w.wall_id == wall_id else w for w in plan.walls)
    return PlanCandidate(per_cav=dict(plan.per_cav), walls=walls, plan_version=plan.plan_version)


def with_wall_release(plan: PlanCandidate, wall_id: str) -> PlanCandidate:
    """A copy of ``plan`` with wall ``wall_id`` recast as a release candidate
    (predict() encoding, contract section 5/§8.4): the wall STAYS in the plan
    with its locked members -- phase becomes "release" with
    ``release_at_s=0.0`` (release now, offset from t0) and ``hold_s=None`` --
    and the members never move into per_cav while they are locked (§3.3).
    Members return to per_cav as native only after the dispatcher reports the
    handback done (members unlocked)."""
    wall = next((w for w in plan.walls if w.wall_id == wall_id), None)
    if wall is None:
        raise ValueError(f"wall {wall_id!r} is not in the plan")
    released = replace(wall, phase="release", release_at_s=0.0, hold_s=None)
    walls = tuple(released if w.wall_id == wall_id else w for w in plan.walls)
    return PlanCandidate(per_cav=dict(plan.per_cav), walls=walls, plan_version=plan.plan_version)
