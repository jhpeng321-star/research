"""Validators for the predict() contract v1: §3.3 plan rules, §3.1 sample
rules, §4 output rules. Validators raise; they never return a silent guess.

Every violation is a subclass of ValueError, so callers may catch
`ContractViolation` precisely or ValueError generically.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .contract import (
    ABORT_REASONS,
    ACTIONS,
    CONTINUATION_NATIVE,
    CONTINUATIONS,
    CONTRACT_VERSION_V1_1,
    CONTRACT_VERSION_V2,
    ENTRANCE_MAINLINE,
    ENTRANCE_RAMP,
    LANE_IDS,
    PART_A_SCOPES,
    PER_CAV_ACTIONS,
    PHASES,
    PLAN_VERSION,
    SUPPORTED_CONTRACT_VERSIONS,
    ControllerStateEntry,
    ControllerState,
    HiddenTrajectorySet,
    Meta,
    PlanCandidate,
    Prediction,
    SampleMeta,
    StateSample,
    TRAJ_COLS,
    WallOutcome,
    WallSpec,
)

# tolerance for the §4.1 arithmetic identities
_DELAY_RTOL = 1e-6
_DELAY_ATOL = 1e-9


class ContractViolation(ValueError):
    """Base class for every contract-validation failure."""


class PlanViolation(ContractViolation):
    """A §3.3 plan rule (or §3.2 controller-state shape) is violated."""


class StateSamplesViolation(ContractViolation):
    """A §3.1 state-sample rule is violated."""


class PredictionViolation(ContractViolation):
    """A §4 output rule is violated."""


def _is_real(x: Any) -> bool:
    return isinstance(x, Real) and not isinstance(x, (bool, np.bool_))


def _is_bool(x: Any) -> bool:
    return isinstance(x, (bool, np.bool_))


def _require(cond: Any, msg: str, exc: type) -> None:
    if not cond:
        raise exc(msg)


# ---------------------------------------------------------------------------
# controller state (§3.2) — checked by validate_plan because plan rules refer
# to locked/wall entries.
# ---------------------------------------------------------------------------

def _as_entry(cav_id: str, e: Any) -> ControllerStateEntry:
    if isinstance(e, ControllerStateEntry):
        return e
    if isinstance(e, Mapping):
        try:
            return ControllerStateEntry.from_raw(e)
        except TypeError as err:
            raise PlanViolation(f"controller_state[{cav_id!r}]: {err}") from err
    raise PlanViolation(
        f"controller_state[{cav_id!r}]: expected ControllerStateEntry or mapping, "
        f"got {type(e).__name__}"
    )


def _check_controller_state(controller_state: ControllerState) -> None:
    _require(
        isinstance(controller_state, Mapping),
        "controller_state must be a mapping cav_id -> ControllerStateEntry",
        PlanViolation,
    )
    for cav_id, raw in controller_state.items():
        _require(
            isinstance(cav_id, str) and cav_id,
            f"controller_state key must be a non-empty str, got {cav_id!r}",
            PlanViolation,
        )
        e = _as_entry(cav_id, raw)
        _require(
            e.action in ACTIONS,
            f"controller_state[{cav_id!r}].action {e.action!r} not in {ACTIONS}",
            PlanViolation,
        )
        _require(
            e.wall_id is None or (isinstance(e.wall_id, str) and e.wall_id),
            f"controller_state[{cav_id!r}].wall_id must be None or non-empty str",
            PlanViolation,
        )
        _require(
            e.wall_phase is None or e.wall_phase in PHASES,
            f"controller_state[{cav_id!r}].wall_phase {e.wall_phase!r} not in {PHASES} or None",
            PlanViolation,
        )
        _require(
            _is_real(e.time_in_phase_s) and math.isfinite(e.time_in_phase_s) and e.time_in_phase_s >= 0,
            f"controller_state[{cav_id!r}].time_in_phase_s must be finite >= 0",
            PlanViolation,
        )
        _require(
            e.wall_speed_mps is None
            or (_is_real(e.wall_speed_mps) and math.isfinite(e.wall_speed_mps) and e.wall_speed_mps >= 0),
            f"controller_state[{cav_id!r}].wall_speed_mps must be None or finite >= 0",
            PlanViolation,
        )
        _require(_is_bool(e.locked), f"controller_state[{cav_id!r}].locked must be bool", PlanViolation)
        # triple consistency: action=="wall" <=> wall_id set <=> wall_phase set
        is_wall = e.action == "wall"
        _require(
            is_wall == (e.wall_id is not None) == (e.wall_phase is not None),
            f"controller_state[{cav_id!r}]: action/wall_id/wall_phase inconsistent "
            f"(action={e.action!r}, wall_id={e.wall_id!r}, wall_phase={e.wall_phase!r}); "
            "a wall member carries all three, everyone else none",
            PlanViolation,
        )
        _require(
            not (e.locked and e.wall_id is None),
            f"controller_state[{cav_id!r}]: locked=True but wall_id is None",
            PlanViolation,
        )
        # §9 (v1.1): optional lane_id, one of the WM-30 physical lanes or None
        _require(
            e.lane_id is None
            or (
                isinstance(e.lane_id, Integral)
                and not isinstance(e.lane_id, bool)
                and e.lane_id in LANE_IDS
            ),
            f"controller_state[{cav_id!r}].lane_id must be None or one of "
            f"{LANE_IDS} (0 main_left, 1 main_right, 2 accel), got {e.lane_id!r}",
            PlanViolation,
        )


def _as_wall_spec(i: int, w: Any) -> WallSpec:
    if isinstance(w, WallSpec):
        return w
    if isinstance(w, Mapping):
        try:
            return WallSpec.from_raw(w)
        except TypeError as err:
            raise PlanViolation(f"walls[{i}]: {err}") from err
    raise PlanViolation(f"walls[{i}]: expected WallSpec or mapping, got {type(w).__name__}")


def _as_plan(plan: Any) -> PlanCandidate:
    if isinstance(plan, PlanCandidate):
        return plan
    if isinstance(plan, Mapping):
        try:
            return PlanCandidate.from_raw(plan)
        except TypeError as err:
            raise PlanViolation(f"plan: {err}") from err
    raise PlanViolation(f"plan: expected PlanCandidate or mapping, got {type(plan).__name__}")


# ---------------------------------------------------------------------------
# §3.3 plan rules
# ---------------------------------------------------------------------------

def validate_plan(plan: Any, controller_state: ControllerState) -> None:
    """Validate one PlanCandidate against the controller state. Raises
    PlanViolation on the first violated §3.3 rule; returns None on success."""
    _check_controller_state(controller_state)
    plan = _as_plan(plan)

    # plan_version
    _require(
        isinstance(plan.plan_version, Integral) and not isinstance(plan.plan_version, bool),
        "[plan_version] must be int, got {pv!r}".format(pv=plan.plan_version),
        PlanViolation,
    )
    _require(
        plan.plan_version == PLAN_VERSION,
        f"[plan_version] v1 contract requires plan_version == {PLAN_VERSION}, got {plan.plan_version}",
        PlanViolation,
    )

    # walls: shape, uniqueness, membership hygiene
    _require(isinstance(plan.walls, (tuple, list)), "[walls] must be a tuple of WallSpec", PlanViolation)
    seen_wall_ids = set()
    seen_members = {}  # cav_id -> wall_id, for disjointness
    for i, raw in enumerate(plan.walls):
        w = _as_wall_spec(i, raw)
        tag = f"[walls[{i}] {w.wall_id!r}]"
        _require(isinstance(w.wall_id, str) and w.wall_id, f"{tag} wall_id must be non-empty str", PlanViolation)
        _require(
            w.wall_id not in seen_wall_ids,
            f"{tag} duplicate wall_id {w.wall_id!r}",
            PlanViolation,
        )
        seen_wall_ids.add(w.wall_id)
        _require(isinstance(w.members, (tuple, list)), f"{tag} members must be a tuple", PlanViolation)
        _require(len(w.members) > 0, f"{tag} members must be non-empty", PlanViolation)
        for m in w.members:
            _require(
                isinstance(m, str) and m,
                f"{tag} member ids must be non-empty str, got {m!r}",
                PlanViolation,
            )
            _require(
                m in controller_state,
                f"{tag} member {m!r} is not a known CAV (missing from controller_state)",
                PlanViolation,
            )
            if m in seen_members:
                raise PlanViolation(
                    f"{tag} member {m!r} already belongs to wall {seen_members[m]!r}; "
                    "[walls] wall membership sets must be disjoint"
                )
            seen_members[m] = w.wall_id
        _require(
            len(set(w.members)) == len(w.members),
            f"{tag} duplicate member inside {w.members}",
            PlanViolation,
        )
        # §9 (v1.1): when lane_id is present for every member, members must be
        # ordered by ascending lane index; if any lane_id is missing the check
        # is skipped (lane order is then the planner's responsibility, §8.3)
        lanes = [_as_entry(m, controller_state[m]).lane_id for m in w.members]
        if all(ln is not None for ln in lanes):
            _require(
                all(a < b for a, b in zip(lanes, lanes[1:])),
                f"{tag} members {w.members} are not ordered by ascending lane_id "
                f"{lanes} (§9)",
                PlanViolation,
            )
        _require(
            _is_real(w.speed_mps) and math.isfinite(w.speed_mps) and w.speed_mps > 0,
            f"{tag} speed_mps must be finite > 0 (a moving wall), got {w.speed_mps!r}",
            PlanViolation,
        )
        _require(
            w.phase in PHASES,
            f"{tag} phase {w.phase!r} not in {PHASES}",
            PlanViolation,
        )
        _require(
            (w.release_at_s is None) != (w.hold_s is None),
            f"{tag} exactly one of release_at_s / hold_s must be set "
            f"(release_at_s={w.release_at_s!r}, hold_s={w.hold_s!r})",
            PlanViolation,
        )
        _require(
            w.release_at_s is None
            or (_is_real(w.release_at_s) and math.isfinite(w.release_at_s) and w.release_at_s >= 0),
            f"{tag} release_at_s must be None or finite >= 0 (relative to t0)",
            PlanViolation,
        )
        _require(
            w.hold_s is None or (_is_real(w.hold_s) and w.hold_s >= 0),
            f"{tag} hold_s must be None or >= 0 (math.inf allowed for the native "
            "indefinite-hold encoding)",
            PlanViolation,
        )

    # per_cav: the FULL map of every non-member CAV, allowed actions only
    _require(
        isinstance(plan.per_cav, Mapping),
        "[per_cav] must be a mapping cav_id -> action",
        PlanViolation,
    )
    all_members = set(seen_members)
    for cav_id, action in plan.per_cav.items():
        _require(
            isinstance(cav_id, str) and cav_id,
            f"[per_cav] key must be non-empty str, got {cav_id!r}",
            PlanViolation,
        )
        _require(
            action in PER_CAV_ACTIONS,
            f"[per_cav][{cav_id!r}] action {action!r} not in {PER_CAV_ACTIONS}",
            PlanViolation,
        )
        _require(
            cav_id not in all_members,
            f"[per_cav] wall member {cav_id!r} must never appear in per_cav",
            PlanViolation,
        )
    expected_non_members = set(controller_state) - all_members
    missing = expected_non_members - set(plan.per_cav)
    _require(
        not missing,
        f"[per_cav] is not the FULL map: non-member CAVs missing from per_cav: {sorted(missing)}",
        PlanViolation,
    )
    extra = set(plan.per_cav) - expected_non_members
    _require(
        not extra,
        f"[per_cav] has entries that are not non-member CAVs: {sorted(extra)}",
        PlanViolation,
    )

    # locked CAVs may appear only as members of their existing wall
    for cav_id, raw in controller_state.items():
        e = _as_entry(cav_id, raw)
        if not e.locked:
            continue
        if cav_id not in seen_members:
            raise PlanViolation(
                f"[locked] CAV {cav_id!r} is locked to wall {e.wall_id!r} but is not a "
                "member of any wall in the plan; locked CAVs may appear only as members "
                "of their existing wall"
            )
        if seen_members[cav_id] != e.wall_id:
            raise PlanViolation(
                f"[locked] CAV {cav_id!r} is locked to wall {e.wall_id!r} but the plan "
                f"places it in wall {seen_members[cav_id]!r}"
            )


# ---------------------------------------------------------------------------
# §3.1 state samples
# ---------------------------------------------------------------------------

def _as_state_sample(k: int, s: Any) -> StateSample:
    if isinstance(s, StateSample):
        return s
    if isinstance(s, Mapping):
        try:
            return StateSample.from_raw(s)
        except TypeError as err:
            raise StateSamplesViolation(f"state_samples[{k}]: {err}") from err
    raise StateSamplesViolation(
        f"state_samples[{k}]: expected StateSample or mapping, got {type(s).__name__}"
    )


def _check_arrivals(arrivals: Any) -> None:
    from .contract import Arrival, ArrivalDraw

    if isinstance(arrivals, ArrivalDraw):
        _require(
            isinstance(arrivals.demand_model_id, str) and arrivals.demand_model_id,
            "arrivals draw: demand_model_id must be non-empty str",
            StateSamplesViolation,
        )
        _require(
            isinstance(arrivals.seed, Integral) and not isinstance(arrivals.seed, bool),
            "arrivals draw: seed must be int",
            StateSamplesViolation,
        )
        return
    _require(
        isinstance(arrivals, (tuple, list)),
        "arrivals must be a tuple of Arrival rows or an ArrivalDraw",
        StateSamplesViolation,
    )
    for i, a in enumerate(arrivals):
        if isinstance(a, Mapping):
            try:
                a = Arrival(**a)
            except TypeError as err:
                raise StateSamplesViolation(f"arrivals[{i}]: {err}") from err
        _require(
            isinstance(a, Arrival),
            f"arrivals[{i}]: expected Arrival row, got {type(a).__name__}",
            StateSamplesViolation,
        )
        _require(
            _is_real(a.t_s) and math.isfinite(a.t_s) and a.t_s >= 0,
            f"arrivals[{i}].t_s must be finite >= 0",
            StateSamplesViolation,
        )
        # §8.9: entrance labels are fixed to the boundary section names
        _require(
            a.entrance in (ENTRANCE_MAINLINE, ENTRANCE_RAMP),
            f"arrivals[{i}].entrance must be {ENTRANCE_MAINLINE!r} or "
            f"{ENTRANCE_RAMP!r} (§8), got {a.entrance!r}",
            StateSamplesViolation,
        )
        _require(
            _is_real(a.speed_mps) and math.isfinite(a.speed_mps) and a.speed_mps >= 0,
            f"arrivals[{i}].speed_mps must be finite >= 0",
            StateSamplesViolation,
        )
        _require(_is_bool(a.is_cav), f"arrivals[{i}].is_cav must be bool", StateSamplesViolation)
        _require(
            isinstance(a.vehicle_type, str) and a.vehicle_type,
            f"arrivals[{i}].vehicle_type must be non-empty str",
            StateSamplesViolation,
        )
        # §8.14(b): arrival_id is OPTIONAL; when present it is the plan/native
        # pairing key and must be a non-empty str.
        _require(
            a.arrival_id is None or (isinstance(a.arrival_id, str) and a.arrival_id),
            f"arrivals[{i}].arrival_id must be None or a non-empty str (§8.14)",
            StateSamplesViolation,
        )


def validate_state_samples(state_samples: Sequence[Any]) -> Tuple[StateSample, ...]:
    """Validate the K-sample sequence (§3.1) and return it as a tuple.
    K >= 1; all samples share one time_s; meta fields are well-typed and
    sample_ids unique."""
    _require(
        isinstance(state_samples, (tuple, list)) and len(state_samples) >= 1,
        "state_samples must be a non-empty sequence of StateSample (K >= 1)",
        StateSamplesViolation,
    )
    samples = tuple(_as_state_sample(k, s) for k, s in enumerate(state_samples))

    time_s = None
    seen_ids = set()
    for k, s in enumerate(samples):
        tag = f"state_samples[{k}]"
        _require(s.hybrid is not None, f"{tag}.hybrid must not be None", StateSamplesViolation)
        _check_arrivals(s.arrivals)
        meta = s.meta
        if isinstance(meta, Mapping):
            try:
                meta = SampleMeta.from_raw(meta)
            except TypeError as err:
                raise StateSamplesViolation(f"{tag}.meta: {err}") from err
        _require(
            isinstance(meta, SampleMeta),
            f"{tag}.meta must be a SampleMeta",
            StateSamplesViolation,
        )
        _require(
            isinstance(meta.sample_id, str) and meta.sample_id,
            f"{tag}.meta.sample_id must be non-empty str",
            StateSamplesViolation,
        )
        _require(
            meta.sample_id not in seen_ids,
            f"{tag}.meta.sample_id {meta.sample_id!r} is not unique across the K samples",
            StateSamplesViolation,
        )
        seen_ids.add(meta.sample_id)
        _require(
            _is_real(meta.time_s) and math.isfinite(meta.time_s),
            f"{tag}.meta.time_s must be a finite number",
            StateSamplesViolation,
        )
        if time_s is None:
            time_s = meta.time_s
        else:
            _require(
                meta.time_s == time_s,
                f"{tag}.meta.time_s {meta.time_s!r} differs from sample 0 time_s "
                f"{time_s!r}; all K samples of one call share the same time_s",
                StateSamplesViolation,
            )
        _require(
            _is_bool(meta.merge_observed),
            f"{tag}.meta.merge_observed must be bool",
            StateSamplesViolation,
        )
        _require(
            _is_real(meta.coverage) and math.isfinite(meta.coverage) and 0.0 <= meta.coverage <= 1.0,
            f"{tag}.meta.coverage must be a float in [0, 1]",
            StateSamplesViolation,
        )
    return samples


# ---------------------------------------------------------------------------
# §4 output rules
# ---------------------------------------------------------------------------

def _as_meta(meta: Any) -> Meta:
    if isinstance(meta, Meta):
        return meta
    if isinstance(meta, Mapping):
        try:
            return Meta.from_raw(meta)
        except TypeError as err:
            raise PredictionViolation(f"meta: {err}") from err
    raise PredictionViolation(f"meta: expected Meta or mapping, got {type(meta).__name__}")


def _mk_array(
    arr: Any,
    name: str,
    shape: Tuple[int, ...],
    kind: str,
    *,
    allow_nan: bool = False,
    values: Optional[Sequence[int]] = None,
) -> None:
    """One §4 array check: type, shape, dtype, NaN policy, allowed values."""
    where = f"Prediction.{name}"
    _require(isinstance(arr, np.ndarray), f"{where} must be a numpy array, got {type(arr).__name__}", PredictionViolation)
    _require(arr.shape == tuple(shape), f"{where} must have shape {shape}, got {arr.shape}", PredictionViolation)
    if kind == "f":
        _require(arr.dtype.kind == "f", f"{where} must be a float dtype, got {arr.dtype}", PredictionViolation)
        if not allow_nan:
            _require(
                bool(np.isfinite(arr).all()),
                f"{where} must be finite (NaN/inf not allowed)",
                PredictionViolation,
            )
    elif kind == "b":
        _require(arr.dtype == np.bool_, f"{where} must be a bool array, got {arr.dtype}", PredictionViolation)
    elif kind == "i8":
        _require(arr.dtype == np.int8, f"{where} must be int8, got {arr.dtype}", PredictionViolation)
        if values is not None:
            uniq = set(np.unique(arr).tolist())
            _require(
                uniq <= set(values),
                f"{where} values {sorted(uniq)} not within allowed {sorted(values)}",
                PredictionViolation,
            )
    else:  # pragma: no cover - internal guard
        raise AssertionError(f"unknown kind {kind}")


def _bool_array(arr: Any, name: str, M: int, K: int) -> np.ndarray:
    """§9 required cell-validity output. A missing one is a validation error —
    it is never defaulted to True (fail closed)."""
    _require(
        arr is not None,
        f"{name} is REQUIRED in contract-1.1 outputs; a missing {name} is a "
        "validation error and is never defaulted to True (§9)",
        PredictionViolation,
    )
    _mk_array(arr, name, (M, K), "b")
    return arr


def _finite_on(arr: np.ndarray, valid: np.ndarray, name: str) -> None:
    """§9: every float in a valid cell is finite; invalid cells may hold NaN."""
    if valid.all():
        ok = bool(np.isfinite(arr).all())
    else:
        ok = bool(np.isfinite(arr[valid]).all())
    _require(
        ok,
        f"{name} must be finite wherever the cell is valid; NaN may appear only "
        "in invalid cells (§9)",
        PredictionViolation,
    )


def _nan_iff_no_closing_pair(
    min_ttc: np.ndarray, closing: np.ndarray, valid: np.ndarray, name: str
) -> None:
    """§9: in a valid cell, min_ttc_s is NaN if and only if has_closing_pair is
    False (a NaN with no pair passes the TTC floor; a NaN with a closing pair
    is a hidden failure and is rejected)."""
    is_nan = np.isnan(min_ttc)
    _require(
        bool(np.array_equal(is_nan[valid], ~closing[valid])),
        f"{name}: in a valid cell, NaN if and only if has_closing_pair is False "
        "(§9); NaN never signals a failed, invalid or early-stopped prediction",
        PredictionViolation,
    )


# §3 (v2) future-safety outputs: bool validity/pair books first, then the
# [M, K] floats. All are None on v1/v1.1 outputs and REQUIRED on v2 outputs.
_V2_BOOL_ARRAYS = (
    "future_cell_valid",
    "native_future_cell_valid",
    "future_has_closing_pair",
    "native_future_has_closing_pair",
)
_V2_FLOAT_ARRAYS = (
    "future_ttc_lt3_exposure_s",
    "future_hard_brake_s",
    "p_future_collision",
    "future_min_ttc_q05_s",
    "native_future_ttc_lt3_exposure_s",
    "native_future_hard_brake_s",
    "native_p_future_collision",
    "native_future_min_ttc_q05_s",
)
# §4 (v2) meta additions; None (False for the validated flag) on v1/v1.1.
_V2_META_FIELDS = (
    "commit_s",
    "continuation",
    "continuation_policy_id",
    "value_model_id",
    "value_model_sha256",
    "future_min_ttc_validation_id",
)


def _prob_in_unit_interval(arr: np.ndarray, valid: np.ndarray, name: str) -> None:
    """§3 (v2): probabilities are finite in [0, 1] in valid cells."""
    v = np.asarray(arr)[valid]
    ok = bool(np.isfinite(v).all()) and bool(((v >= 0.0) & (v <= 1.0)).all())
    _require(
        ok,
        f"{name} must be a finite probability in [0, 1] wherever the cell is "
        "valid (§3); NaN never signals a failed prediction",
        PredictionViolation,
    )


def _reject_newer_version_arrays(pred: Prediction, version: str) -> None:
    """§9 / v2 dual-version discipline: a v1/v1.1 output carries none of the
    v2-only arrays (a newer output validated by an older path is a version
    mismatch, mirroring the native_unlocked_* handling)."""
    for f_name in _V2_BOOL_ARRAYS + _V2_FLOAT_ARRAYS:
        _require(
            getattr(pred, f_name) is None,
            f"{f_name} set on a {version!r} output; the §3 v2 future-safety "
            "arrays belong to 'wd01_predict/2' outputs",
            PredictionViolation,
        )


def _check_tracked_trajectories(
    traj: Any, name: str, shape: Tuple[int, int, int, int]
) -> None:
    """§4.3 tracked trajectories: {vehicle_id: float[M,K,T,4]}; per (m,k) the
    NaN region is a whole-row trailing suffix (a vehicle that left the section
    is NaN from the exit step on) and step 0 (t0) is always real."""
    where = f"Prediction.{name}"
    _require(
        isinstance(traj, Mapping) and len(traj) > 0,
        f"{where} must be a non-empty mapping vehicle_id -> array",
        PredictionViolation,
    )
    for vid, arr in traj.items():
        _require(
            isinstance(vid, str) and vid,
            f"{where} keys must be non-empty vehicle ids, got {vid!r}",
            PredictionViolation,
        )
        _mk_array(arr, f"{name}[{vid!r}]", shape, "f", allow_nan=True)
        nan_row = np.isnan(arr[..., 0])  # [M, K, T] from the t_s column
        # whole-row NaN (all four columns), never partial columns
        any_nan = np.isnan(arr).any(axis=-1)
        all_nan = np.isnan(arr).all(axis=-1)
        _require(
            bool(np.array_equal(any_nan, all_nan)),
            f"{where}[{vid!r}]: NaN must cover the whole (t_s, s_m, lane_id, speed_mps) "
            "row, not individual columns",
            PredictionViolation,
        )
        # suffix pattern: once NaN, always NaN (no holes, no re-entry)
        started = np.cumsum(nan_row, axis=-1) > 0
        _require(
            bool(np.array_equal(nan_row, started)),
            f"{where}[{vid!r}]: NaN steps must be a trailing suffix per (member, sample); "
            "found holes or NaN before the exit step",
            PredictionViolation,
        )
        _require(
            not bool(nan_row[..., 0].any()),
            f"{where}[{vid!r}]: a tracked vehicle is present at t0; step 0 must not be NaN",
            PredictionViolation,
        )


def _check_optional_unlocked(pred: Prediction, M: int, K: int) -> None:
    """§4.5: the five native_unlocked_* arrays are present (all of them) exactly
    when meta.native_unlocked_available is True."""
    fields = {
        "native_unlocked_delay_s": "f",
        "native_unlocked_min_ttc_s": "f",
        "native_unlocked_ttc_lt3_exposure_s": "f",
        "native_unlocked_hard_brake_s": "f",
        "native_unlocked_collision": "b",
    }
    present = {k: getattr(pred, k) is not None for k in fields}
    if pred.meta.native_unlocked_available:
        _require(
            all(present.values()),
            "[native_unlocked] meta.native_unlocked_available=True but missing: "
            f"{sorted(k for k, p in present.items() if not p)}",
            PredictionViolation,
        )
        for k, kind in fields.items():
            _mk_array(getattr(pred, k), k, (M, K), kind, allow_nan=(k.endswith("min_ttc_s")))
    else:
        _require(
            not any(present.values()),
            "[native_unlocked] meta.native_unlocked_available=False but set: "
            f"{sorted(k for k, p in present.items() if p)}",
            PredictionViolation,
        )


def _check_v2_1_optional(pred: Prediction, M: int, K: int, meta: Meta) -> None:
    """§4 (v2.1) OPTIONAL tracked-scope fields: validated exactly when
    present, absent = valid for every predictor (e.g. perfect_sim).
    Per-cell counts are float [M, K], finite and >= 0; meta strings are
    non-empty; part_a_scope is one of PART_A_SCOPES.  No veto and no
    abstain may be derived from n_free_road_assumed (a threshold would be
    an owner decision)."""
    where = "Prediction."
    for name in ("n_tracked", "n_exited_tracked", "n_free_road_assumed",
                 "untracked_arrivals_expected"):
        arr = getattr(pred, name)
        if arr is None:
            continue
        _mk_array(arr, name, (M, K), "f")
        _require(
            bool((arr >= 0).all()),
            f"{where}{name} must be >= 0 everywhere (a count / expectation)",
            PredictionViolation,
        )
    for name in ("part_a_model", "param_set_id", "recon_schema_version"):
        v = getattr(meta, name)
        _require(
            v is None or (isinstance(v, str) and v),
            f"meta.{name} must be None or a non-empty str (§4 v2.1)",
            PredictionViolation,
        )
    scope = meta.part_a_scope
    _require(
        scope is None or scope in PART_A_SCOPES,
        f"meta.part_a_scope must be None or one of {PART_A_SCOPES}, "
        f"got {scope!r} (§3 v2.1: part_a values are compared across "
        "predictors only when the scope is equal)",
        PredictionViolation,
    )


def validate_prediction(
    pred: Any,
    M: int,
    K: int,
    n_plans: int,
    trajectory_s: Optional[float] = None,
) -> None:
    """Validate one Prediction (§4).

    M, K, n_plans are the call dimensions; n_plans is call context only (no
    per-Prediction field depends on it, but wrong call sites are caught early).
    trajectory_s is the caller's requested trajectory length; when None the T
    axis is derived from meta.part_a_horizon_s. T == trajectory_s + 1 (t0
    included)."""
    _require(isinstance(pred, Prediction), "pred must be a Prediction", PredictionViolation)
    for name, v in (("M", M), ("K", K), ("n_plans", n_plans)):
        _require(
            isinstance(v, Integral) and not isinstance(v, bool) and v >= 1,
            f"{name} must be a positive int, got {v!r}",
            PredictionViolation,
        )
    _require(
        trajectory_s is None or (_is_real(trajectory_s) and trajectory_s >= 0),
        "trajectory_s must be None or >= 0",
        PredictionViolation,
    )

    meta = _as_meta(pred.meta)

    # --- §4.6 meta (§9: both supported versions accepted; the output is
    # validated against the version it declares) ---
    _require(
        meta.contract_version in SUPPORTED_CONTRACT_VERSIONS,
        f"meta.contract_version must be one of {SUPPORTED_CONTRACT_VERSIONS}, "
        f"got {meta.contract_version!r}",
        PredictionViolation,
    )
    is_v11 = meta.contract_version == CONTRACT_VERSION_V1_1
    is_v2 = meta.contract_version == CONTRACT_VERSION_V2
    _require(
        isinstance(meta.model_id, str) and meta.model_id,
        "meta.model_id must be non-empty str",
        PredictionViolation,
    )
    _require(
        isinstance(meta.weights_sha256, (tuple, list)) and len(meta.weights_sha256) == M,
        f"meta.weights_sha256 must have one entry per member (M={M}), got len "
        f"{len(meta.weights_sha256)}",
        PredictionViolation,
    )
    _require(
        all(isinstance(w, str) and w for w in meta.weights_sha256),
        "meta.weights_sha256 entries must be non-empty str",
        PredictionViolation,
    )
    _require(
        isinstance(meta.sample_ids, (tuple, list)) and len(meta.sample_ids) == K,
        f"meta.sample_ids must have length K={K}, got len {len(meta.sample_ids)}",
        PredictionViolation,
    )
    _require(
        isinstance(meta.member_ids, (tuple, list)) and len(meta.member_ids) == M,
        f"meta.member_ids must have length M={M}, got len {len(meta.member_ids)}",
        PredictionViolation,
    )
    _require(
        all(isinstance(x, str) and x for x in list(meta.sample_ids) + list(meta.member_ids)),
        "meta.sample_ids / meta.member_ids entries must be non-empty str",
        PredictionViolation,
    )
    _require(
        _is_real(meta.part_a_horizon_s) and meta.part_a_horizon_s > 0,
        "meta.part_a_horizon_s must be > 0",
        PredictionViolation,
    )
    if is_v2:
        # §3 (v2): the v1 window fields cover the commit window Δ, so
        # safety_window_s == commit_s (not part_a_horizon_s) on v2 outputs.
        _require(
            _is_real(meta.safety_window_s) and meta.safety_window_s == meta.commit_s,
            f"meta.safety_window_s ({meta.safety_window_s!r}) must equal "
            f"meta.commit_s ({meta.commit_s!r}) on a 'wd01_predict/2' output (§3)",
            PredictionViolation,
        )
    else:
        _require(
            _is_real(meta.safety_window_s) and meta.safety_window_s == meta.part_a_horizon_s,
            f"meta.safety_window_s ({meta.safety_window_s!r}) must equal "
            f"meta.part_a_horizon_s ({meta.part_a_horizon_s!r})",
            PredictionViolation,
        )
    _require(
        _is_real(meta.horizon_s) and meta.horizon_s > 0,
        "meta.horizon_s must be > 0",
        PredictionViolation,
    )
    _require(
        isinstance(meta.native_kept_walls, (tuple, list))
        and all(isinstance(w, str) and w for w in meta.native_kept_walls),
        "meta.native_kept_walls must be a tuple of wall_id strings",
        PredictionViolation,
    )
    _require(
        _is_bool(meta.native_unlocked_available),
        "meta.native_unlocked_available must be bool",
        PredictionViolation,
    )
    _require(
        isinstance(meta.device, str) and meta.device,
        "meta.device must be non-empty str (informational)",
        PredictionViolation,
    )
    _require(
        _is_real(meta.wall_clock_s) and math.isfinite(meta.wall_clock_s) and meta.wall_clock_s >= 0,
        "meta.wall_clock_s must be finite >= 0",
        PredictionViolation,
    )

    if is_v2:
        # --- §4 (v2) meta additions; REQUIRED, never defaulted (fail closed) ---
        _require(
            _is_real(meta.commit_s) and meta.commit_s > 0,
            f"meta.commit_s must be a finite > 0 on a v2 output, got {meta.commit_s!r}",
            PredictionViolation,
        )
        _require(
            meta.commit_s <= meta.part_a_horizon_s,
            f"meta.commit_s ({meta.commit_s!r}) must not exceed "
            f"meta.part_a_horizon_s ({meta.part_a_horizon_s!r}): the commit "
            "window is prediction-internal (§1)",
            PredictionViolation,
        )
        _require(
            meta.continuation in CONTINUATIONS,
            f"meta.continuation must be one of {CONTINUATIONS}, got {meta.continuation!r}",
            PredictionViolation,
        )
        _require(
            (meta.continuation_policy_id is None) == (meta.continuation == CONTINUATION_NATIVE),
            "meta.continuation_policy_id must be None exactly when continuation "
            f"is 'native' (got continuation={meta.continuation!r}, "
            f"continuation_policy_id={meta.continuation_policy_id!r}) (§4)",
            PredictionViolation,
        )
        _require(
            meta.continuation_policy_id is None
            or (isinstance(meta.continuation_policy_id, str) and meta.continuation_policy_id),
            "meta.continuation_policy_id must be a non-empty str when set (§4)",
            PredictionViolation,
        )
        _require(
            isinstance(meta.value_model_id, str) and meta.value_model_id,
            "meta.value_model_id must be a non-empty str on a v2 output (§4)",
            PredictionViolation,
        )
        _require(
            isinstance(meta.value_model_sha256, (tuple, list))
            and len(meta.value_model_sha256) == M
            and all(isinstance(w, str) and w for w in meta.value_model_sha256),
            f"meta.value_model_sha256 must have one non-empty str entry per "
            f"member (M={M}) on a v2 output (§4)",
            PredictionViolation,
        )
        _require(
            _is_bool(meta.future_min_ttc_validated),
            "meta.future_min_ttc_validated must be bool on a v2 output (§4)",
            PredictionViolation,
        )
        _require(
            (meta.future_min_ttc_validation_id is None) != meta.future_min_ttc_validated,
            "meta.future_min_ttc_validation_id must be set exactly when "
            "meta.future_min_ttc_validated is True: it points at the ledger/"
            "report record that validated the future_min_ttc_q05_s head (§4); "
            f"got flag={meta.future_min_ttc_validated!r}, "
            f"id={meta.future_min_ttc_validation_id!r}",
            PredictionViolation,
        )
        _require(
            meta.future_min_ttc_validation_id is None
            or (
                isinstance(meta.future_min_ttc_validation_id, str)
                and meta.future_min_ttc_validation_id
            ),
            "meta.future_min_ttc_validation_id must be a non-empty str when set (§4)",
            PredictionViolation,
        )
    else:
        # v1/v1.1 outputs carry none of the §4 v2 meta additions (a missing
        # one is the v1 shape; a set one is a version mismatch).
        for f_name in _V2_META_FIELDS:
            _require(
                getattr(meta, f_name) is None,
                f"meta.{f_name} set on a {meta.contract_version!r} output; the "
                "§4 v2 meta additions belong to 'wd01_predict/2' outputs",
                PredictionViolation,
            )
        _require(
            not meta.future_min_ttc_validated,
            "meta.future_min_ttc_validated must be False on a "
            f"{meta.contract_version!r} output; future-min-TTC validation is a "
            "'wd01_predict/2' concept (§4)",
            PredictionViolation,
        )

    # --- §4.1 / §4.2 [M, K] arrays; §9 cell validity (v1.1) ---
    # shape/dtype here; finiteness is a per-cell rule (§9), so NaN passes
    # _mk_array and is policed below per contract version
    _mk_array(pred.plan_delay_s, "plan_delay_s", (M, K), "f", allow_nan=True)
    _mk_array(pred.native_delay_s, "native_delay_s", (M, K), "f", allow_nan=True)
    _mk_array(pred.delay_vs_native_s, "delay_vs_native_s", (M, K), "f", allow_nan=True)
    _mk_array(pred.delay_part_a_s, "delay_part_a_s", (M, K), "f", allow_nan=True)
    _mk_array(pred.delay_part_b_s, "delay_part_b_s", (M, K), "f", allow_nan=True)
    _mk_array(pred.collision, "collision", (M, K), "b")  # bool: defined by dtype (§9)
    _mk_array(pred.native_collision, "native_collision", (M, K), "b")
    _mk_array(pred.min_ttc_s, "min_ttc_s", (M, K), "f", allow_nan=True)
    _mk_array(pred.native_min_ttc_s, "native_min_ttc_s", (M, K), "f", allow_nan=True)
    _mk_array(pred.ttc_lt3_exposure_s, "ttc_lt3_exposure_s", (M, K), "f", allow_nan=True)
    _mk_array(pred.native_ttc_lt3_exposure_s, "native_ttc_lt3_exposure_s", (M, K), "f", allow_nan=True)
    _mk_array(pred.hard_brake_s, "hard_brake_s", (M, K), "f", allow_nan=True)
    _mk_array(pred.native_hard_brake_s, "native_hard_brake_s", (M, K), "f", allow_nan=True)

    if is_v2:
        # §3 (v2) future-safety arrays: shape/dtype first (NaN passes here and
        # is policed per cell below), then the per-cell validity rules.
        for f_name in _V2_BOOL_ARRAYS:
            _require(
                getattr(pred, f_name) is not None,
                f"{f_name} is REQUIRED on a 'wd01_predict/2' output; a missing "
                f"{f_name} is a validation error and is never defaulted "
                "(§3 v2: fail closed)",
                PredictionViolation,
            )
        future_cell_valid = _bool_array(pred.future_cell_valid, "future_cell_valid", M, K)
        native_future_cell_valid = _bool_array(
            pred.native_future_cell_valid, "native_future_cell_valid", M, K
        )
        future_closing = _bool_array(
            pred.future_has_closing_pair, "future_has_closing_pair", M, K
        )
        native_future_closing = _bool_array(
            pred.native_future_has_closing_pair, "native_future_has_closing_pair", M, K
        )
        for f_name in _V2_FLOAT_ARRAYS:
            _mk_array(getattr(pred, f_name), f_name, (M, K), "f", allow_nan=True)

    if is_v11 or is_v2:
        cell_valid = _bool_array(pred.cell_valid, "cell_valid", M, K)
        native_cell_valid = _bool_array(pred.native_cell_valid, "native_cell_valid", M, K)
        closing = _bool_array(pred.has_closing_pair, "has_closing_pair", M, K)
        native_closing = _bool_array(
            pred.native_has_closing_pair, "native_has_closing_pair", M, K
        )
        # §9 validator rules: in a VALID cell, min_ttc_s is NaN iff
        # has_closing_pair is False; every other float is finite; collision
        # flags are defined (guaranteed by the bool dtype above). Invalid
        # cells may carry NaN in any field.
        _finite_on(pred.plan_delay_s, cell_valid, "plan_delay_s")
        _finite_on(pred.delay_part_a_s, cell_valid, "delay_part_a_s")
        _finite_on(pred.ttc_lt3_exposure_s, cell_valid, "ttc_lt3_exposure_s")
        _finite_on(pred.hard_brake_s, cell_valid, "hard_brake_s")
        _finite_on(pred.native_delay_s, native_cell_valid, "native_delay_s")
        _finite_on(pred.native_ttc_lt3_exposure_s, native_cell_valid, "native_ttc_lt3_exposure_s")
        _finite_on(pred.native_hard_brake_s, native_cell_valid, "native_hard_brake_s")
        _nan_iff_no_closing_pair(pred.min_ttc_s, closing, cell_valid, "min_ttc_s")
        _nan_iff_no_closing_pair(
            pred.native_min_ttc_s, native_closing, native_cell_valid, "native_min_ttc_s"
        )
        both_valid = cell_valid & native_cell_valid

        if is_v2:
            # §3 (v2): future_cell_valid covers delay_part_b_s and every
            # future_* float; plan/native totals are finite wherever both of
            # their parts are (their composition).
            _finite_on(pred.delay_part_b_s, future_cell_valid, "delay_part_b_s")
            _finite_on(pred.plan_delay_s, cell_valid & future_cell_valid, "plan_delay_s")
            _finite_on(
                pred.native_delay_s,
                native_cell_valid & native_future_cell_valid,
                "native_delay_s",
            )
            _finite_on(
                pred.future_ttc_lt3_exposure_s, future_cell_valid, "future_ttc_lt3_exposure_s"
            )
            _finite_on(pred.future_hard_brake_s, future_cell_valid, "future_hard_brake_s")
            _finite_on(
                pred.native_future_ttc_lt3_exposure_s,
                native_future_cell_valid,
                "native_future_ttc_lt3_exposure_s",
            )
            _finite_on(
                pred.native_future_hard_brake_s,
                native_future_cell_valid,
                "native_future_hard_brake_s",
            )
            _prob_in_unit_interval(pred.p_future_collision, future_cell_valid, "p_future_collision")
            _prob_in_unit_interval(
                pred.native_p_future_collision,
                native_future_cell_valid,
                "native_p_future_collision",
            )
            # §3 (v2) NaN rules: in a valid future cell every future_* float is
            # finite EXCEPT future_min_ttc_q05_s (and native_), which is NaN
            # ONLY where future_has_closing_pair is False.
            _nan_iff_no_closing_pair(
                pred.future_min_ttc_q05_s, future_closing, future_cell_valid, "future_min_ttc_q05_s"
            )
            _nan_iff_no_closing_pair(
                pred.native_future_min_ttc_q05_s,
                native_future_closing,
                native_future_cell_valid,
                "native_future_min_ttc_q05_s",
            )
            # §4.1 identities on cells valid on both sides AND future-valid on
            # both sides (an invalid cell may hold NaN on either side).
            identity_cells = both_valid & future_cell_valid & native_future_cell_valid
            _finite_on(pred.delay_vs_native_s, identity_cells, "delay_vs_native_s")
        else:
            # a 1.1 output carries none of the §3 v2 future arrays
            _reject_newer_version_arrays(pred, meta.contract_version)
            _finite_on(pred.delay_part_b_s, cell_valid, "delay_part_b_s")
            _finite_on(pred.delay_vs_native_s, both_valid, "delay_vs_native_s")
            identity_cells = both_valid
    else:
        # §9: a 1.0 output has no cell_valid and is accepted only through this
        # v1 validator path; carrying §9 or v2 fields on a 1.0 output is a
        # version mismatch (mirrors the native_unlocked_* handling).
        for f_name in (
            "cell_valid",
            "native_cell_valid",
            "has_closing_pair",
            "native_has_closing_pair",
        ):
            _require(
                getattr(pred, f_name) is None,
                f"{f_name} set on a {meta.contract_version!r} output; §9 cell-validity "
                "fields belong to 1.1 outputs (a 1.0 output has none)",
                PredictionViolation,
            )
        _reject_newer_version_arrays(pred, meta.contract_version)
        # §8.7 (v1): min_ttc_s may be NaN (no closing-pair bookkeeping in v1);
        # every other float must be finite — NaN never signals a failure.
        for name, arr in (
            ("plan_delay_s", pred.plan_delay_s),
            ("native_delay_s", pred.native_delay_s),
            ("delay_vs_native_s", pred.delay_vs_native_s),
            ("delay_part_a_s", pred.delay_part_a_s),
            ("delay_part_b_s", pred.delay_part_b_s),
            ("ttc_lt3_exposure_s", pred.ttc_lt3_exposure_s),
            ("native_ttc_lt3_exposure_s", pred.native_ttc_lt3_exposure_s),
            ("hard_brake_s", pred.hard_brake_s),
            ("native_hard_brake_s", pred.native_hard_brake_s),
        ):
            _require(
                bool(np.isfinite(arr).all()),
                f"{name} must be finite in v1 (§8.7: NaN never signals a failed "
                "prediction)",
                PredictionViolation,
            )
        identity_cells = np.ones((M, K), dtype=bool)

    # §4.1 identities (§8.8 tolerances), restricted in v1.1 to cells valid on
    # both sides (an invalid cell may hold NaN, which cannot enter an identity)
    _require(
        bool(
            np.allclose(
                pred.delay_vs_native_s[identity_cells],
                pred.plan_delay_s[identity_cells] - pred.native_delay_s[identity_cells],
                rtol=_DELAY_RTOL,
                atol=_DELAY_ATOL,
            )
        ),
        "delay_vs_native_s != plan_delay_s - native_delay_s",
        PredictionViolation,
    )
    _require(
        bool(
            np.allclose(
                pred.delay_part_a_s[identity_cells] + pred.delay_part_b_s[identity_cells],
                pred.plan_delay_s[identity_cells],
                rtol=_DELAY_RTOL,
                atol=_DELAY_ATOL,
            )
        ),
        "delay_part_a_s + delay_part_b_s != plan_delay_s (diagnostic split must sum)",
        PredictionViolation,
    )

    # --- §4.3 trajectories; T = trajectory_s + 1 ---
    if trajectory_s is None:
        # §2 (v2): trajectory_s=None means commit_s on a v2 output (v1/v1.1:
        # the model's own part_a_horizon_s).
        base = meta.commit_s if is_v2 else meta.part_a_horizon_s
        T = int(round(float(base))) + 1
    else:
        T = int(round(float(trajectory_s))) + 1
    shape = (M, K, T, TRAJ_COLS)
    _check_tracked_trajectories(pred.trajectories_tracked, "trajectories_tracked", shape)
    _check_tracked_trajectories(
        pred.native_trajectories_tracked, "native_trajectories_tracked", shape
    )
    if pred.trajectories_hidden is not None:
        hidden = pred.trajectories_hidden
        _require(
            isinstance(hidden, (tuple, list)) and len(hidden) == K,
            f"trajectories_hidden must be a per-k sequence of length K={K}",
            PredictionViolation,
        )
        for k, hs in enumerate(hidden):
            if isinstance(hs, HiddenTrajectorySet):
                arr, ids = hs.traj, hs.ids
            elif isinstance(hs, tuple) and len(hs) == 2:
                arr, ids = hs
            else:
                raise PredictionViolation(
                    f"trajectories_hidden[{k}] must be a HiddenTrajectorySet "
                    f"(array [n_k, T, 4] + ids), got {type(hs).__name__}"
                )
            _mk_array(
                arr,
                f"trajectories_hidden[{k}].traj",
                (arr.shape[0], T, TRAJ_COLS),
                "f",
            )
            _require(
                isinstance(ids, (tuple, list))
                and len(ids) == arr.shape[0]
                and all(isinstance(x, str) and x for x in ids),
                f"trajectories_hidden[{k}].ids must have one non-empty str id per hidden row",
                PredictionViolation,
            )

    # --- §4.4 walls ---
    _require(
        isinstance(pred.walls, Mapping),
        "walls must be a mapping wall_id -> WallOutcome",
        PredictionViolation,
    )
    for wall_id, wo in pred.walls.items():
        _require(
            isinstance(wall_id, str) and wall_id,
            f"walls key must be non-empty str, got {wall_id!r}",
            PredictionViolation,
        )
        tag = f"walls[{wall_id!r}]"
        _require(
            isinstance(wo, WallOutcome),
            f"{tag} must be a WallOutcome",
            PredictionViolation,
        )
        _mk_array(wo.assembled, f"{tag}.assembled", (M, K), "b")
        _mk_array(wo.assembled_at_s, f"{tag}.assembled_at_s", (M, K), "f", allow_nan=True)
        _mk_array(wo.safety_abort, f"{tag}.safety_abort", (M, K), "b")
        _mk_array(
            wo.abort_reason,
            f"{tag}.abort_reason",
            (M, K),
            "i8",
            values=ABORT_REASONS,
        )
        _require(
            bool(np.array_equal(np.isnan(wo.assembled_at_s), ~wo.assembled)),
            f"{tag}.assembled_at_s must be NaN exactly where assembled is False",
            PredictionViolation,
        )
        _require(
            bool(np.all(np.asarray(wo.assembled_at_s)[wo.assembled] >= 0)),
            f"{tag}.assembled_at_s must be >= 0 where assembled",
            PredictionViolation,
        )
        _require(
            bool(
                np.array_equal(
                    wo.abort_reason != ABORT_REASONS[0],
                    wo.safety_abort,
                )
            ),
            f"{tag}.abort_reason non-zero must coincide exactly with safety_abort",
            PredictionViolation,
        )

        if is_v2:
            # §3 (v2) future-wall heads [M, K]: REQUIRED (all three) on v2
            # outputs, never defaulted; no native_ versions exist.
            for f_name in (
                "p_assembly_completes",
                "expected_assembly_time_s",
                "p_assembly_abort",
            ):
                _require(
                    getattr(wo, f_name) is not None,
                    f"{tag}.{f_name} is REQUIRED on a 'wd01_predict/2' output "
                    "(§3 v2: fail closed, never defaulted)",
                    PredictionViolation,
                )
            _mk_array(wo.p_assembly_completes, f"{tag}.p_assembly_completes", (M, K), "f", allow_nan=True)
            _mk_array(
                wo.expected_assembly_time_s, f"{tag}.expected_assembly_time_s", (M, K), "f", allow_nan=True
            )
            _mk_array(wo.p_assembly_abort, f"{tag}.p_assembly_abort", (M, K), "f", allow_nan=True)
            _prob_in_unit_interval(
                wo.p_assembly_completes, future_cell_valid, f"{tag}.p_assembly_completes"
            )
            _prob_in_unit_interval(wo.p_assembly_abort, future_cell_valid, f"{tag}.p_assembly_abort")
            # NaN signals nothing except "never completes": in a valid future
            # cell, NaN is allowed ONLY where p_assembly_completes == 0 (at
            # p == 0 a finite expected time is still legitimate).
            nan_time = np.isnan(wo.expected_assembly_time_s)
            bad_nan = nan_time[future_cell_valid] & (
                wo.p_assembly_completes[future_cell_valid] != 0.0
            )
            _require(
                not bool(bad_nan.any()),
                f"{tag}.expected_assembly_time_s must be NaN only where "
                "p_assembly_completes == 0 in valid future cells (§3); a NaN "
                "with p > 0 is a hidden failure, never a signal",
                PredictionViolation,
            )
        else:
            for f_name in (
                "p_assembly_completes",
                "expected_assembly_time_s",
                "p_assembly_abort",
            ):
                _require(
                    getattr(wo, f_name) is None,
                    f"{tag}.{f_name} set on a {meta.contract_version!r} output; the "
                    "§3 v2 future-wall heads belong to 'wd01_predict/2' outputs",
                    PredictionViolation,
                )

    # --- §4.5 optional native-unlocked diagnostics ---
    _check_optional_unlocked(pred, M, K)

    # --- §4 (v2.1) optional tracked-scope fields ---
    _check_v2_1_optional(pred, M, K, meta)


__all__ = [
    "ContractViolation",
    "PlanViolation",
    "StateSamplesViolation",
    "PredictionViolation",
    "validate_plan",
    "validate_state_samples",
    "validate_prediction",
]
