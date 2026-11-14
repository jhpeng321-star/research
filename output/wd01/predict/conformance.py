"""Conformance kit for predict() implementations (contract v1).

run_conformance(predictor) builds a synthetic call (several CAVs, one wall in
each phase with locked members, K >= 3 samples, several plans — including the
§5 native reference and the §4.5 unlocked variant) and checks:

1. inputs_pass_validators        §3.1/§3.3 fixtures themselves are valid
2. output_order_matches_plans    one Prediction per plan; Prediction[i]
                                 corresponds to plans[i] (single-plan anchor
                                 calls + swap test)
3. mk_alignment_across_plans     identical member_ids / sample_ids in every
                                 Prediction of the call (paired [M, K] axes)
4. native_shared_within_call     identical native_* arrays, native
                                 trajectories and native_kept_walls in every
                                 Prediction of the call (§2: native computed
                                 ONCE per call)
5. validators_pass               validate_prediction for every Prediction (§4)
6. deterministic_on_repeat_call  second identical call -> identical outputs
                                 (meta.wall_clock_s excluded: informational)
7. predict_matches_predict_many  §2 identity
8. native_kept_walls_match_reference
                                 meta.native_kept_walls equals §5 resolution

Raises ConformanceError listing every failed check; returns a
ConformanceReport when all pass. Pure numpy; no simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields, is_dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .contract import (
    CONTINUATION_NATIVE,
    CONTINUATION_PLANNER,
    CONTINUATIONS,
    CONTRACT_VERSION,
    CONTRACT_VERSION_V1_1,
    DEFAULT_COMMIT_S,
    DEFAULT_HORIZON_S,
    PHASE_ASSEMBLE,
    PHASE_HOLD,
    PHASE_RELEASE,
    SUPPORTED_CONTRACT_VERSIONS,
    Arrival,
    ControllerStateEntry,
    Meta,
    PlanCandidate,
    Prediction,
    SampleMeta,
    StateSample,
    WallSpec,
)
from .native_ref import native_reference, native_unlocked_reference
from .validate import (
    _V2_BOOL_ARRAYS,
    _V2_FLOAT_ARRAYS,
    validate_plan,
    validate_prediction,
    validate_state_samples,
)

T0_S = 1000.0
SECTION_LEN_M = 1000.0

CHECKS: Tuple[str, ...] = (
    "inputs_pass_validators",
    "output_order_matches_plans",
    "mk_alignment_across_plans",
    "native_shared_within_call",
    "validators_pass",
    "deterministic_on_repeat_call",
    "predict_matches_predict_many",
    "native_kept_walls_match_reference",
)

# §2/§3 (v2) kit: the same 8 checks on the commit-window API, plus the two
# v2-specific checks (future outputs present/aligned, meta matches the call).
V2_CHECKS: Tuple[str, ...] = CHECKS + (
    "v2_future_outputs_present",
    "v2_meta_matches_call",
)

# the frozen v1 family that run_conformance_both_versions exercises (§9)
_V1_FAMILY_VERSIONS = (CONTRACT_VERSION, CONTRACT_VERSION_V1_1)

_NATIVE_ARRAY_FIELDS = (
    "native_delay_s",
    "native_collision",
    "native_min_ttc_s",
    "native_ttc_lt3_exposure_s",
    "native_hard_brake_s",
    # §9 cell-validity outputs; None == None on a 1.0 call, so unconditional
    "native_cell_valid",
    "native_has_closing_pair",
)

# §3 (v2): the native future-safety outputs join the shared-native check
_NATIVE_ARRAY_FIELDS_V2 = _NATIVE_ARRAY_FIELDS + (
    "native_future_cell_valid",
    "native_future_has_closing_pair",
    "native_future_ttc_lt3_exposure_s",
    "native_future_hard_brake_s",
    "native_p_future_collision",
    "native_future_min_ttc_q05_s",
)


class ConformanceError(RuntimeError):
    """One or more conformance checks failed; .failures lists them."""

    def __init__(self, failures: Sequence[str]):
        self.failures = tuple(failures)
        super().__init__("conformance failed: " + "; ".join(self.failures))


@dataclass(frozen=True)
class ConformanceReport:
    predictor_id: str
    checks: Tuple[str, ...]
    n_plans: int
    n_samples: int
    M: int
    K: int
    horizon_s: float
    trajectory_s: Optional[float]


# --- synthetic fixtures -----------------------------------------------------

def build_cav_ids() -> Tuple[str, ...]:
    return ("cav_a0", "cav_a1", "cav_h0", "cav_n0", "cav_n1", "cav_r0")


def build_controller_state() -> Mapping[str, ControllerStateEntry]:
    """Six CAVs: wall W1 assembling (two locked members), wall W2 holding,
    wall W3 releasing, two free CAVs. §5 resolution keeps exactly W1."""
    return {
        "cav_a0": ControllerStateEntry("wall", "W1", PHASE_ASSEMBLE, 12.0, 5.0, True),
        "cav_a1": ControllerStateEntry("wall", "W1", PHASE_ASSEMBLE, 12.0, 5.0, True),
        "cav_h0": ControllerStateEntry("wall", "W2", PHASE_HOLD, 40.0, 6.0, True),
        "cav_r0": ControllerStateEntry("wall", "W3", PHASE_RELEASE, 3.0, 8.0, True),
        "cav_n0": ControllerStateEntry("native"),
        "cav_n1": ControllerStateEntry("native"),
    }


def build_state_samples(k: int = 3, t0: float = T0_S) -> list:
    """K >= 3 samples sharing t0; hybrid uses the minimal synthetic mapping the
    reference predictor reads ({'tracked': {id: (s_m, lane_id, speed_mps)},
    'section_len_m': float}). Tracked ids are stable across samples; several
    vehicles exit the section inside the Part A window (NaN-tail coverage)."""
    if k < 3:
        raise ValueError("the conformance kit needs k >= 3 samples")
    tracked = {
        "cav_a0": (20.0, 0, 22.0),
        "cav_a1": (30.0, 1, 21.0),
        "cav_h0": (60.0, 0, 24.0),
        "cav_r0": (80.0, 1, 25.0),
        "cav_n0": (45.0, 0, 28.0),
        "cav_n1": (120.0, 1, 26.0),
        "obs_0": (200.0, 0, 29.0),
        "obs_1": (250.0, 1, 23.0),
    }
    arrivals = tuple(
        Arrival(t0 + dt, entrance, speed, is_cav, vtype)
        for dt, entrance, speed, is_cav, vtype in (
            (15.0, "mainline", 28.0, False, "car"),
            (90.0, "mainline", 30.0, True, "cav"),
            (300.0, "ramp", 22.0, False, "car"),
            (550.0, "mainline", 26.0, False, "truck"),
        )
    )
    return [
        StateSample(
            hybrid={
                "t0": t0,
                "section_len_m": SECTION_LEN_M,
                "tracked": dict(tracked),
            },
            arrivals=arrivals,
            meta=SampleMeta(f"s{i}", t0, i != 1, 0.75),
        )
        for i in range(k)
    ]


def build_plans(controller_state: Mapping[str, ControllerStateEntry]) -> list:
    """Four valid plans: §5 native reference, §4.5 unlocked variant, a
    keep-W1-plus-release-the-rest plan with a damping CAV, and a new-wall plan
    assembled from the two free CAVs (per_cav empty: every CAV is a member)."""
    ids = build_cav_ids()
    native, _ = native_reference(controller_state, ids)
    unlocked = native_unlocked_reference(controller_state, ids)

    keep_w1 = WallSpec(
        wall_id="W1",
        members=("cav_a0", "cav_a1"),
        speed_mps=5.0,
        phase=PHASE_ASSEMBLE,
        release_at_s=None,
        hold_s=30.0,
    )
    p_hold = PlanCandidate(
        plan_version=1,
        per_cav={"cav_n0": "damping", "cav_n1": "native"},
        walls=(keep_w1,) + tuple(
            w for w in native.walls if w.phase == PHASE_RELEASE
        ),
    )
    new_wall = WallSpec(
        wall_id="W4",
        members=("cav_n0", "cav_n1"),
        speed_mps=7.0,
        phase=PHASE_ASSEMBLE,
        release_at_s=None,
        hold_s=20.0,
    )
    p_new = PlanCandidate(
        plan_version=1,
        per_cav={},
        walls=(keep_w1, new_wall)
        + tuple(w for w in native.walls if w.phase == PHASE_RELEASE),
    )
    return [native, unlocked, p_hold, p_new]


# --- deep equality (wall_clock_s excluded: informational §4.6) --------------

def _arr_eq(x: Any, y: Any) -> bool:
    if isinstance(x, np.ndarray):
        return bool(
            np.array_equal(x, y, equal_nan=x.dtype.kind == "f")
        )
    return x == y


def _pred_equal(a: Any, b: Any) -> bool:
    """Structural deep equality over Prediction-like objects; numpy arrays via
    array_equal (NaN-aware), meta.wall_clock_s ignored."""
    if is_dataclass(a) and is_dataclass(b):
        if type(a) is not type(b):
            return False
        for f in dc_fields(a):
            if isinstance(a, Meta) and f.name == "wall_clock_s":
                continue  # informational (§4.6); excluded from equality
            if not _pred_equal(getattr(a, f.name), getattr(b, f.name)):
                return False
        return True
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            return False
        return all(_pred_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_pred_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return _arr_eq(a, b)
    return a == b


# --- the kit -----------------------------------------------------------------

def _run_kit(
    predictor: Any,
    *,
    samples: list,
    call_kwargs: Mapping,
    traj_s_of: Any,
    native_fields: Sequence[str],
    checks: Sequence[str],
    extra_checks: Sequence[Tuple[str, Any]] = (),
) -> ConformanceReport:
    """Shared kit core for both contract generations.

    `call_kwargs` goes to predict_many verbatim (v1: horizon_s/trajectory_s;
    v2: + commit_s/continuation). `traj_s_of(meta)` derives the T axis for the
    validator pass (part_a_horizon_s in v1/v1.1; commit_s in v2 when
    trajectory_s was not given). `extra_checks` entries are (name, fn(preds))
    appended after the shared checks."""
    controller_state = build_controller_state()
    plans = build_plans(controller_state)
    n_plans = len(plans)

    failures = []

    def check(name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — collect, then raise once
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    def _inputs() -> None:
        validate_state_samples(samples)
        for p in plans:
            validate_plan(p, controller_state)

    check("inputs_pass_validators", _inputs)

    preds = predictor.predict_many(samples, plans, controller_state, **call_kwargs)

    def _order() -> None:
        if not isinstance(preds, list):
            raise AssertionError(f"predict_many must return a list, got {type(preds)}")
        if len(preds) != n_plans:
            raise AssertionError(f"expected {n_plans} Predictions, got {len(preds)}")
        # single-plan anchors fix f(plan_i) without assuming an order
        for idx in (0, n_plans - 1):
            alone = predictor.predict_many(
                samples, [plans[idx]], controller_state, **call_kwargs
            )
            if len(alone) != 1:
                raise AssertionError(
                    f"predict_many with one plan returned {len(alone)} Predictions"
                )
            if not _pred_equal(alone[0], preds[idx]):
                raise AssertionError(
                    f"Prediction[{idx}] does not correspond to plans[{idx}] "
                    "(single-plan anchor mismatch)"
                )
        # swap test: a Prediction must depend only on its own plan
        swapped = predictor.predict_many(
            samples, list(reversed(plans)), controller_state, **call_kwargs
        )
        for i, p in enumerate(reversed(preds)):
            if not _pred_equal(p, swapped[i]):
                raise AssertionError(
                    f"Prediction[{i}] changed when other plans were reordered"
                )

    check("output_order_matches_plans", _order)

    def _alignment() -> None:
        m0 = preds[0].meta.member_ids
        k0 = preds[0].meta.sample_ids
        for j, p in enumerate(preds[1:], start=1):
            if p.meta.member_ids != m0:
                raise AssertionError(f"Prediction[{j}].meta.member_ids differs across plans")
            if p.meta.sample_ids != k0:
                raise AssertionError(f"Prediction[{j}].meta.sample_ids differs across plans")
            if getattr(p, "native_delay_s").shape != preds[0].native_delay_s.shape:
                raise AssertionError(f"Prediction[{j}] native_delay_s shape differs")

    check("mk_alignment_across_plans", _alignment)

    def _native_shared() -> None:
        for j, p in enumerate(preds[1:], start=1):
            for f in native_fields:
                if not _arr_eq(getattr(preds[0], f), getattr(p, f)):
                    raise AssertionError(f"native {f} differs between Prediction 0 and {j}")
            if not _pred_equal(
                preds[0].native_trajectories_tracked, p.native_trajectories_tracked
            ):
                raise AssertionError(
                    f"native_trajectories_tracked differs between Prediction 0 and {j}"
                )
            if preds[0].meta.native_kept_walls != p.meta.native_kept_walls:
                raise AssertionError(
                    f"meta.native_kept_walls differs between Prediction 0 and {j}"
                )

    check("native_shared_within_call", _native_shared)

    def _validators() -> None:
        traj_s = traj_s_of(preds[0].meta)
        for j, p in enumerate(preds):
            validate_prediction(
                p,
                len(p.meta.member_ids),
                len(p.meta.sample_ids),
                n_plans,
                traj_s,
            )

    check("validators_pass", _validators)

    def _determinism() -> None:
        preds2 = predictor.predict_many(
            samples, plans, controller_state, **call_kwargs
        )
        for j, (a, b) in enumerate(zip(preds, preds2)):
            if not _pred_equal(a, b):
                raise AssertionError(f"repeat call changed Prediction[{j}]")

    check("deterministic_on_repeat_call", _determinism)

    def _predict_identity() -> None:
        single = predictor.predict(
            samples, plans[0], controller_state, **call_kwargs
        )
        if not _pred_equal(single, preds[0]):
            raise AssertionError(
                "predict(samples, plan, ...) != predict_many(samples, [plan], ...)[0]"
            )

    check("predict_matches_predict_many", _predict_identity)

    def _kept_walls() -> None:
        _, kept = native_reference(controller_state, build_cav_ids())
        got = preds[0].meta.native_kept_walls
        if list(got) != list(kept):
            raise AssertionError(f"meta.native_kept_walls {list(got)} != §5 resolution {list(kept)}")

    check("native_kept_walls_match_reference", _kept_walls)

    for name, fn in extra_checks:
        check(name, lambda fn=fn: fn(preds))

    if failures:
        raise ConformanceError(failures)
    return ConformanceReport(
        predictor_id=getattr(predictor, "model_id", type(predictor).__name__),
        checks=tuple(checks),
        n_plans=n_plans,
        n_samples=len(samples),
        M=len(preds[0].meta.member_ids),
        K=len(preds[0].meta.sample_ids),
        horizon_s=call_kwargs["horizon_s"],
        trajectory_s=call_kwargs.get("trajectory_s"),
    )


def run_conformance(
    predictor: Any,
    *,
    horizon_s: float = DEFAULT_HORIZON_S,
    trajectory_s: Optional[float] = None,
    k_samples: int = 3,
) -> ConformanceReport:
    """Run every conformance check against `predictor` (v1/v1.1 API). Raises
    ConformanceError on the first batch of failures (all checks run; every
    failure is listed). Returns a ConformanceReport when all pass."""
    return _run_kit(
        predictor,
        samples=build_state_samples(k=k_samples),
        call_kwargs=dict(horizon_s=horizon_s, trajectory_s=trajectory_s),
        traj_s_of=lambda meta: (
            trajectory_s if trajectory_s is not None else meta.part_a_horizon_s
        ),
        native_fields=_NATIVE_ARRAY_FIELDS,
        checks=CHECKS,
    )


def run_conformance_v2(
    predictor: Any,
    *,
    horizon_s: float = DEFAULT_HORIZON_S,
    commit_s: float = DEFAULT_COMMIT_S,
    continuation: str = CONTINUATION_PLANNER,
    trajectory_s: Optional[float] = None,
    k_samples: int = 3,
) -> ConformanceReport:
    """The full kit on the §2 (v2) commit-window API (continuation
    "planner"|"native"), plus the two v2-specific checks: the REQUIRED
    future-safety outputs are present and [M, K]-aligned in every Prediction,
    and meta matches the call (commit_s, continuation, the policy-id rule and
    safety_window_s == commit_s). Raises ConformanceError listing every
    failure; returns a ConformanceReport when all pass."""
    return _run_kit(
        predictor,
        samples=build_state_samples(k=k_samples),
        call_kwargs=dict(
            horizon_s=horizon_s,
            commit_s=commit_s,
            continuation=continuation,
            trajectory_s=trajectory_s,
        ),
        traj_s_of=lambda meta: (
            trajectory_s if trajectory_s is not None else meta.commit_s
        ),
        native_fields=_NATIVE_ARRAY_FIELDS_V2,
        checks=V2_CHECKS,
        extra_checks=(
            ("v2_future_outputs_present", _check_v2_future_outputs),
            (
                "v2_meta_matches_call",
                lambda preds: _check_v2_meta_matches_call(
                    preds, commit_s=commit_s, continuation=continuation
                ),
            ),
        ),
    )


def _check_v2_future_outputs(preds: Sequence[Prediction]) -> None:
    """Kit check: every REQUIRED §3 (v2) future-safety output is present and
    [M, K]-aligned in every Prediction of the call (never defaulted)."""
    m_k = preds[0].native_delay_s.shape
    for j, p in enumerate(preds):
        for f in _V2_BOOL_ARRAYS + _V2_FLOAT_ARRAYS:
            arr = getattr(p, f)
            if not isinstance(arr, np.ndarray) or arr.shape != m_k:
                raise AssertionError(
                    f"Prediction[{j}].{f} is missing or not [M, K] {m_k} "
                    "(§3 v2: REQUIRED, never defaulted)"
                )


def _check_v2_meta_matches_call(
    preds: Sequence[Prediction], *, commit_s: float, continuation: str
) -> None:
    """Kit check: meta carries the requested commit semantics."""
    for j, p in enumerate(preds):
        if p.meta.commit_s != commit_s or p.meta.continuation != continuation:
            raise AssertionError(
                f"Prediction[{j}].meta (commit_s={p.meta.commit_s!r}, "
                f"continuation={p.meta.continuation!r}) does not match the call "
                f"(commit_s={commit_s!r}, continuation={continuation!r})"
            )
        if p.meta.safety_window_s != p.meta.commit_s:
            raise AssertionError(
                f"Prediction[{j}].meta.safety_window_s "
                f"({p.meta.safety_window_s!r}) != meta.commit_s "
                f"({p.meta.commit_s!r}) on a v2 output (§3)"
            )
        if (p.meta.continuation_policy_id is None) != (p.meta.continuation == CONTINUATION_NATIVE):
            raise AssertionError(
                f"Prediction[{j}].meta.continuation_policy_id is not None "
                "exactly when continuation == 'native' (§4)"
            )


def broken_predictors() -> "list[tuple[str, Any]]":
    """§9 conformance fixtures: predictors that MUST be rejected by the kit.

    Each subclass of the reference predictor emits a well-formed-looking
    1.1 call whose safety bookkeeping is broken in exactly the way §9
    forbids. Tests (and any implementation's CI) assert that run_conformance
    raises ConformanceError for every one of them."""
    import dataclasses

    import numpy as np

    from .contract import CONTRACT_VERSION_V1_1
    from .reference_const_speed import ConstSpeedPredictor

    class NanMinTtcWithClosingPair(ConstSpeedPredictor):
        """NaN min_ttc_s where has_closing_pair is True: a hidden failure
        dressed up as a missing pair. §9 requires rejection."""

        def __init__(self):
            super().__init__(contract_version=CONTRACT_VERSION_V1_1)

        def predict_many(self, state_samples, plans, controller_state, **kw):
            out = super().predict_many(state_samples, plans, controller_state, **kw)
            return [
                dataclasses.replace(
                    p,
                    min_ttc_s=np.where(
                        np.isfinite(p.min_ttc_s), np.nan, p.min_ttc_s
                    ),
                )
                for p in out
            ]

    class NanSignalledFailure(ConstSpeedPredictor):
        """A failure signalled only through NaN while cell_valid stays True.
        §9: NaN never signals a failed prediction; cell_valid must."""

        def __init__(self):
            super().__init__(contract_version=CONTRACT_VERSION_V1_1)

        def predict_many(self, state_samples, plans, controller_state, **kw):
            out = super().predict_many(state_samples, plans, controller_state, **kw)
            return [
                dataclasses.replace(
                    p, plan_delay_s=np.full(p.plan_delay_s.shape, np.nan)
                )
                for p in out
            ]

    class MissingCellValid(ConstSpeedPredictor):
        """A 1.1 output without cell_valid. §9: consumers must reject a
        missing cell_valid, never default it to True."""

        def __init__(self):
            super().__init__(contract_version=CONTRACT_VERSION_V1_1)

        def predict_many(self, state_samples, plans, controller_state, **kw):
            out = super().predict_many(state_samples, plans, controller_state, **kw)
            return [dataclasses.replace(p, cell_valid=None) for p in out]

    return [
        ("nan_min_ttc_with_closing_pair", NanMinTtcWithClosingPair()),
        ("nan_signalled_failure", NanSignalledFailure()),
        ("missing_cell_valid", MissingCellValid()),
    ]


def broken_predictors_v2() -> "list[tuple[str, Any]]":
    """§3/§5 conformance fixtures: v2 predictors that MUST be rejected by
    run_conformance_v2. Each subclass of the v2 reference predictor emits a
    well-formed-looking wd01_predict/2 call broken in exactly one way the
    contract forbids; the kit catches each through the validators_pass check."""
    import dataclasses

    import numpy as np

    from .reference_const_speed import ConstSpeedPredictorV2

    class V2MissingFutureCellValid(ConstSpeedPredictorV2):
        """A v2 output without future_cell_valid. §3: REQUIRED, never
        defaulted to True (fail closed)."""

        def predict_many(self, state_samples, plans, controller_state, **kw):
            out = super().predict_many(state_samples, plans, controller_state, **kw)
            return [dataclasses.replace(p, future_cell_valid=None) for p in out]

    class V2NanFutureMinTtcWithClosingPair(ConstSpeedPredictorV2):
        """NaN future_min_ttc_q05_s where future_has_closing_pair is True:
        a hidden failure dressed up as a missing pair."""

        def predict_many(self, state_samples, plans, controller_state, **kw):
            out = super().predict_many(state_samples, plans, controller_state, **kw)
            return [
                dataclasses.replace(
                    p,
                    future_min_ttc_q05_s=np.full(
                        p.future_min_ttc_q05_s.shape, np.nan, dtype=np.float64
                    ),
                )
                for p in out
            ]

    class V2NanFutureDelayWhileValid(ConstSpeedPredictorV2):
        """A continuation failure signalled only through NaN delay_part_b_s
        while future_cell_valid stays True. §3/§9: NaN never signals failure."""

        def predict_many(self, state_samples, plans, controller_state, **kw):
            out = super().predict_many(state_samples, plans, controller_state, **kw)
            return [
                dataclasses.replace(
                    p, delay_part_b_s=np.full(p.delay_part_b_s.shape, np.nan)
                )
                for p in out
            ]

    class V2PolicyIdWithNativeContinuation(ConstSpeedPredictorV2):
        """continuation='native' with continuation_policy_id set: §4 says the
        id is None exactly when the continuation is native."""

        def predict_many(self, state_samples, plans, controller_state, **kw):
            out = super().predict_many(state_samples, plans, controller_state, **kw)
            return [
                dataclasses.replace(
                    p,
                    meta=dataclasses.replace(
                        p.meta,
                        continuation=CONTINUATION_NATIVE,
                        continuation_policy_id="pi/must-not-be-here",
                    ),
                )
                for p in out
            ]

    class V2ValidatedWithoutId(ConstSpeedPredictorV2):
        """future_min_ttc_validated=True without a validation id: §4 says the
        id points at the validating record and must exist when the flag is
        True (the planner refuses live mode on an unattributable head)."""

        def predict_many(self, state_samples, plans, controller_state, **kw):
            out = super().predict_many(state_samples, plans, controller_state, **kw)
            return [
                dataclasses.replace(
                    p,
                    meta=dataclasses.replace(
                        p.meta,
                        future_min_ttc_validated=True,
                        future_min_ttc_validation_id=None,
                    ),
                )
                for p in out
            ]

    class V2WallExpectedTimeNanWithPPositive(ConstSpeedPredictorV2):
        """expected_assembly_time_s NaN while p_assembly_completes > 0: the
        NaN must mean 'never completes', never hide a failure (§3)."""

        def predict_many(self, state_samples, plans, controller_state, **kw):
            out = super().predict_many(state_samples, plans, controller_state, **kw)
            patched = []
            for p in out:
                walls = dict(p.walls)
                wall_id = next(iter(walls))
                walls[wall_id] = dataclasses.replace(
                    walls[wall_id],
                    expected_assembly_time_s=np.full(
                        walls[wall_id].expected_assembly_time_s.shape,
                        np.nan,
                        dtype=np.float64,
                    ),
                )
                patched.append(dataclasses.replace(p, walls=walls))
            return patched

    return [
        ("v2_missing_future_cell_valid", V2MissingFutureCellValid()),
        ("v2_nan_future_min_ttc_with_closing_pair", V2NanFutureMinTtcWithClosingPair()),
        ("v2_nan_future_delay_while_future_valid", V2NanFutureDelayWhileValid()),
        ("v2_policy_id_with_native_continuation", V2PolicyIdWithNativeContinuation()),
        ("v2_validated_flag_true_without_id", V2ValidatedWithoutId()),
        ("v2_wall_expected_assembly_time_nan_with_p_positive", V2WallExpectedTimeNanWithPPositive()),
    ]


def run_conformance_both_versions(predictor_factory) -> "dict[str, ConformanceReport]":
    """§9: run the full kit once per FROZEN v1-family contract version
    ('wd01_predict/1' and 'wd01_predict/1.1'). The v2 commit-window kit is
    run_conformance_v2 (run per continuation, see the kit __main__)."""
    reports = {}
    for version in _V1_FAMILY_VERSIONS:
        reports[version] = run_conformance(predictor_factory(version))
    return reports


if __name__ == "__main__":
    import json

    from .reference_const_speed import ConstSpeedPredictor, ConstSpeedPredictorV2

    reports = run_conformance_both_versions(
        lambda version: ConstSpeedPredictor(contract_version=version)
    )
    v2_reports = {
        continuation: run_conformance_v2(
            ConstSpeedPredictorV2(), continuation=continuation
        )
        for continuation in CONTINUATIONS
    }
    print(
        json.dumps(
            {
                "status": "PASS",
                "versions": {
                    version: {
                        "predictor": report.predictor_id,
                        "checks": list(report.checks),
                        "n_plans": report.n_plans,
                        "n_samples": report.n_samples,
                        "M": report.M,
                        "K": report.K,
                    }
                    for version, report in reports.items()
                },
                "v2": {
                    continuation: {
                        "predictor": report.predictor_id,
                        "checks": list(report.checks),
                        "n_plans": report.n_plans,
                        "n_samples": report.n_samples,
                        "M": report.M,
                        "K": report.K,
                    }
                    for continuation, report in v2_reports.items()
                },
            },
            indent=2,
        )
    )
