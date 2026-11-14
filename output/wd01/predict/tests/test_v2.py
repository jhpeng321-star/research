"""PREDICT_CONTRACT_v2 (wd01_predict/2) — positive and negative tests for
every v2 rule: §1/§2 commit-window API, §3 outputs (delay split, future
safety, validity books, NaN rules, future walls), §4 meta additions.
§8.14/§8.15 (v1 clarifications) validator rules are covered at the bottom."""
import dataclasses

import numpy as np
import pytest

from predict import (
    CONTRACT_VERSION,
    CONTRACT_VERSION_V1_1,
    CONTRACT_VERSION_V2,
    DEFAULT_COMMIT_S,
    CONTINUATION_NATIVE,
    CONTINUATION_PLANNER,
    CONTINUATIONS,
    SUPPORTED_CONTRACT_VERSIONS,
    Arrival,
    SampleMeta,
    StateSample,
    PredictionViolation,
    StateSamplesViolation,
    build_controller_state,
    build_plans,
    build_state_samples,
    validate_prediction,
    validate_state_samples,
)
from predict.conformance import V2_CHECKS, broken_predictors_v2, run_conformance_v2
from predict.conformance import ConformanceError
from predict.reference_const_speed import ConstSpeedPredictorV2

CS = build_controller_state()
SAMPLES = build_state_samples(k=3)
PLANS = build_plans(CS)
N_PLANS = 2
V2 = ConstSpeedPredictorV2().predict_many(SAMPLES, PLANS[:N_PLANS], CS)[0]
V2_NATIVE = ConstSpeedPredictorV2(continuation=CONTINUATION_NATIVE).predict_many(
    SAMPLES, PLANS[:1], CS
)[0]
M, K = V2.collision.shape


def rep(pred, **kw):
    return dataclasses.replace(pred, **kw)


def repmeta(pred, **kw):
    return rep(pred, meta=dataclasses.replace(pred.meta, **kw))


def copied(arr):
    return np.array(arr, copy=True)


def walls_with(pred, wall_id, **kw):
    wo = dataclasses.replace(pred.walls[wall_id], **kw)
    return {**pred.walls, wall_id: wo}


# --- §1/§2: constants, commit-window API ---------------------------------------

def test_v2_constants():
    assert CONTRACT_VERSION_V2 == "wd01_predict/2"
    assert DEFAULT_COMMIT_S == 30.0
    assert CONTINUATIONS == ("planner", "native")
    assert SUPPORTED_CONTRACT_VERSIONS == (
        CONTRACT_VERSION,
        CONTRACT_VERSION_V1_1,
        CONTRACT_VERSION_V2,
    )


def test_v2_reference_output_passes_validation():
    assert V2.meta.contract_version == CONTRACT_VERSION_V2
    assert V2.meta.commit_s == DEFAULT_COMMIT_S == 30.0
    assert V2.meta.safety_window_s == V2.meta.commit_s  # §3: window == Δ
    assert V2.meta.continuation == CONTINUATION_PLANNER
    assert V2.meta.continuation_policy_id
    assert V2.meta.value_model_id and V2.meta.value_model_sha256 == V2.meta.weights_sha256
    assert V2.meta.future_min_ttc_validated is False
    assert V2.meta.future_min_ttc_validation_id is None
    assert validate_prediction(V2, M, K, N_PLANS) is None


def test_v2_native_continuation_has_no_policy_id():
    assert V2_NATIVE.meta.continuation == CONTINUATION_NATIVE
    assert V2_NATIVE.meta.continuation_policy_id is None
    assert validate_prediction(V2_NATIVE, M, K, 1) is None


def test_v2_part_a_covers_commit_window():
    """§3: part_a is the delay inside [t0, t0+Δ], part_b the continuation
    value; the diagnostic split still sums to the total."""
    frac = V2.meta.commit_s / V2.meta.horizon_s
    assert np.allclose(V2.delay_part_a_s, V2.plan_delay_s * frac)
    assert np.allclose(V2.delay_part_b_s, V2.plan_delay_s * (1.0 - frac))
    assert validate_prediction(V2, M, K, N_PLANS) is None


def test_v2_trajectory_s_none_means_commit_s():
    arr = V2.trajectories_tracked["cav_a0"]
    assert arr.shape[2] == int(V2.meta.commit_s) + 1  # T = commit_s + 1
    with pytest.raises(PredictionViolation, match="shape"):
        validate_prediction(V2, M, K, N_PLANS, trajectory_s=60.0)
    longer = ConstSpeedPredictorV2().predict_many(
        SAMPLES, PLANS[:1], CS, trajectory_s=60.0
    )[0]
    assert longer.trajectories_tracked["cav_a0"].shape[2] == 61
    validate_prediction(longer, M, K, 1, trajectory_s=60.0)


def test_v2_per_call_commit_and_continuation():
    pred = ConstSpeedPredictorV2().predict_many(
        SAMPLES, PLANS[:1], CS, commit_s=10.0, continuation=CONTINUATION_NATIVE
    )[0]
    assert pred.meta.commit_s == 10.0
    assert pred.meta.safety_window_s == 10.0
    assert pred.meta.continuation_policy_id is None
    assert pred.trajectories_tracked["cav_a0"].shape[2] == 11
    validate_prediction(pred, M, K, 1, trajectory_s=10.0)


@pytest.mark.parametrize("commit_s", [0.0, -1.0, float("nan"), 61.0])
def test_v2_commit_s_must_be_positive_and_within_part_a(commit_s):
    with pytest.raises(ValueError, match="commit_s"):
        ConstSpeedPredictorV2().predict_many(SAMPLES, PLANS[:1], CS, commit_s=commit_s)


# --- §4: meta additions (REQUIRED in v2, never defaulted) -----------------------

@pytest.mark.parametrize(
    "kw,fragment",
    [
        ({"commit_s": None}, "commit_s"),
        ({"commit_s": 0.0}, "commit_s"),
        ({"continuation": "free-for-all"}, "continuation"),
        ({"continuation": None}, "continuation"),
        ({"continuation": CONTINUATION_PLANNER, "continuation_policy_id": None},
         "continuation_policy_id"),
        ({"continuation": CONTINUATION_NATIVE, "continuation_policy_id": "pi/9"},
         "continuation_policy_id"),
        ({"value_model_id": ""}, "value_model_id"),
        ({"value_model_sha256": ("a", "b")}, "value_model_sha256"),
        ({"future_min_ttc_validated": None}, "future_min_ttc_validated"),
        ({"future_min_ttc_validated": True, "future_min_ttc_validation_id": None},
         "future_min_ttc_validation_id"),
        ({"future_min_ttc_validated": False, "future_min_ttc_validation_id": "ledger/1"},
         "future_min_ttc_validation_id"),
    ],
)
def test_v2_meta_rules_negative(kw, fragment):
    with pytest.raises(PredictionViolation, match=fragment):
        validate_prediction(repmeta(V2, **kw), M, K, N_PLANS)


def test_v2_meta_validated_with_id_is_accepted():
    ok = repmeta(
        V2,
        future_min_ttc_validated=True,
        future_min_ttc_validation_id="PREDICT_LEDGER/2026-09-26/min_ttc_v1",
    )
    assert validate_prediction(ok, M, K, N_PLANS) is None


def test_v1_v1_1_outputs_must_not_carry_v2_meta():
    from predict.reference_const_speed import ConstSpeedPredictor

    v1 = ConstSpeedPredictor().predict_many(SAMPLES, PLANS[:1], CS)[0]
    v11 = ConstSpeedPredictor(contract_version=CONTRACT_VERSION_V1_1).predict_many(
        SAMPLES, PLANS[:1], CS
    )[0]
    for base in (v1, v11):
        with pytest.raises(PredictionViolation, match="commit_s"):
            validate_prediction(repmeta(base, commit_s=30.0), M, K, 1)
        with pytest.raises(PredictionViolation, match="continuation"):
            validate_prediction(
                repmeta(base, continuation=CONTINUATION_PLANNER), M, K, 1
            )
        with pytest.raises(PredictionViolation, match="future_min_ttc_validated"):
            validate_prediction(
                repmeta(base, future_min_ttc_validated=True), M, K, 1
            )


# --- §3: future-safety outputs REQUIRED, never defaulted ------------------------

_V2_BOOL_FIELDS = (
    "future_cell_valid",
    "native_future_cell_valid",
    "future_has_closing_pair",
    "native_future_has_closing_pair",
)
_V2_FLOAT_FIELDS = (
    "future_ttc_lt3_exposure_s",
    "future_hard_brake_s",
    "p_future_collision",
    "future_min_ttc_q05_s",
    "native_future_ttc_lt3_exposure_s",
    "native_future_hard_brake_s",
    "native_p_future_collision",
    "native_future_min_ttc_q05_s",
)


@pytest.mark.parametrize("field", _V2_BOOL_FIELDS)
def test_v2_validity_books_required_never_defaulted(field):
    with pytest.raises(PredictionViolation, match="REQUIRED"):
        validate_prediction(rep(V2, **{field: None}), M, K, N_PLANS)


@pytest.mark.parametrize("field", _V2_FLOAT_FIELDS)
def test_v2_future_floats_required(field):
    with pytest.raises(PredictionViolation, match="numpy array"):
        validate_prediction(rep(V2, **{field: None}), M, K, N_PLANS)


@pytest.mark.parametrize("field", _V2_BOOL_FIELDS + _V2_FLOAT_FIELDS)
def test_v1_v1_1_outputs_must_not_carry_v2_arrays(field):
    from predict.reference_const_speed import ConstSpeedPredictor

    v1 = ConstSpeedPredictor().predict_many(SAMPLES, PLANS[:1], CS)[0]
    junk = np.ones((M, K), dtype=bool if field in _V2_BOOL_FIELDS else np.float64)
    with pytest.raises(PredictionViolation, match="wd01_predict/2"):
        validate_prediction(rep(v1, **{field: junk}), M, K, 1)


# --- §3: NaN policy on the future fields ----------------------------------------

def test_v2_future_min_ttc_nan_iff_no_closing_pair():
    # NaN q05 with a closing pair: hidden failure, rejected
    nan_q05 = np.full((M, K), np.nan)
    with pytest.raises(PredictionViolation, match="future_min_ttc_q05_s"):
        validate_prediction(rep(V2, future_min_ttc_q05_s=nan_q05), M, K, N_PLANS)
    # finite q05 with NO closing pair: also rejected (iff)
    no_pair = rep(V2, future_has_closing_pair=np.zeros((M, K), dtype=bool))
    with pytest.raises(PredictionViolation, match="if and only if"):
        validate_prediction(no_pair, M, K, N_PLANS)
    # NaN q05 with no pair, valid future cells: correct, accepted
    ok = rep(
        no_pair,
        future_min_ttc_q05_s=nan_q05,
    )
    assert validate_prediction(ok, M, K, N_PLANS) is None


def test_v2_native_future_min_ttc_nan_iff_no_closing_pair():
    nan_q05 = np.full((M, K), np.nan)
    with pytest.raises(PredictionViolation, match="native_future_min_ttc_q05_s"):
        validate_prediction(rep(V2, native_future_min_ttc_q05_s=nan_q05), M, K, N_PLANS)
    ok = rep(
        V2,
        native_future_has_closing_pair=np.zeros((M, K), dtype=bool),
        native_future_min_ttc_q05_s=nan_q05,
    )
    assert validate_prediction(ok, M, K, N_PLANS) is None


def test_v2_nan_continuation_delay_while_future_valid_is_rejected():
    """A continuation failure signalled ONLY through NaN delay_part_b_s while
    future_cell_valid stays True: rejected (NaN never signals failure)."""
    part_b = copied(V2.delay_part_b_s)
    part_b[0, 0] = np.nan
    with pytest.raises(PredictionViolation, match="delay_part_b_s"):
        validate_prediction(rep(V2, delay_part_b_s=part_b), M, K, N_PLANS)


def test_v2_invalid_future_cell_may_carry_nan_anywhere():
    """future_cell_valid=False removes every future finiteness obligation for
    that cell (fail closed downstream); window obligations are untouched."""
    future_valid = copied(V2.future_cell_valid)
    future_valid[0, 1] = False
    pred = rep(
        V2,
        future_cell_valid=future_valid,
        delay_part_b_s=np.where(future_valid, V2.delay_part_b_s, np.nan),
        future_ttc_lt3_exposure_s=np.where(
            future_valid, V2.future_ttc_lt3_exposure_s, np.nan
        ),
        future_min_ttc_q05_s=np.where(future_valid, V2.future_min_ttc_q05_s, np.nan),
    )
    assert validate_prediction(pred, M, K, N_PLANS) is None
    # ... but a NaN in a still-valid future cell is rejected
    exposure = copied(pred.future_ttc_lt3_exposure_s)
    exposure[0, 0] = np.nan
    with pytest.raises(PredictionViolation, match="future_ttc_lt3_exposure_s"):
        validate_prediction(rep(pred, future_ttc_lt3_exposure_s=exposure), M, K, N_PLANS)


def test_v2_native_future_validity_mirrors_plan_side():
    native_valid = copied(V2.native_future_cell_valid)
    native_valid[0, 2] = False
    pred = rep(
        V2,
        native_future_cell_valid=native_valid,
        native_future_hard_brake_s=np.where(
            native_valid, V2.native_future_hard_brake_s, np.nan
        ),
    )
    assert validate_prediction(pred, M, K, N_PLANS) is None
    bad = copied(pred.native_future_hard_brake_s)
    bad[0, 0] = np.nan  # NaN in a still-valid native future cell
    with pytest.raises(PredictionViolation, match="native_future_hard_brake_s"):
        validate_prediction(rep(pred, native_future_hard_brake_s=bad), M, K, N_PLANS)


def test_v2_identity_cells_need_window_and_future_validity():
    """§4.1 identities are checked on cells valid on both sides AND
    future-valid on both sides; a future-invalid cell may hold NaN part_b."""
    future_valid = copied(V2.future_cell_valid)
    native_future_valid = copied(V2.native_future_cell_valid)
    future_valid[0, 0] = False
    native_future_valid[0, 0] = False
    pred = rep(
        V2,
        future_cell_valid=future_valid,
        native_future_cell_valid=native_future_valid,
        delay_part_b_s=np.where(future_valid, V2.delay_part_b_s, np.nan),
    )
    assert validate_prediction(pred, M, K, N_PLANS) is None
    # part_a no longer sums where part_b is NaN-by-invalidity: still accepted,
    # but bumping part_a on a FULLY valid cell breaks the identity
    part_a = copied(pred.delay_part_a_s)
    part_a[0, 1] += 1.0
    with pytest.raises(PredictionViolation, match="must sum"):
        validate_prediction(rep(pred, delay_part_a_s=part_a), M, K, N_PLANS)


@pytest.mark.parametrize("field", ["p_future_collision", "native_p_future_collision"])
def test_v2_future_collision_probabilities_bounded(field):
    bad = copied(getattr(V2, field))
    bad[0, 0] = 1.5
    with pytest.raises(PredictionViolation, match=field):
        validate_prediction(rep(V2, **{field: bad}), M, K, N_PLANS)
    nan = copied(getattr(V2, field))
    nan[0, 0] = np.nan
    with pytest.raises(PredictionViolation, match=field):
        validate_prediction(rep(V2, **{field: nan}), M, K, N_PLANS)
    edge = copied(getattr(V2, field))
    edge[0, 0], edge[0, 1] = 0.0, 1.0
    assert validate_prediction(rep(V2, **{field: edge}), M, K, N_PLANS) is None


# --- §3: future-wall heads --------------------------------------------------------

def test_v2_future_wall_heads_present_and_valid():
    for wall_id, wo in V2.walls.items():
        assert wo.p_assembly_completes.shape == (M, K)
        assert np.all((wo.p_assembly_completes >= 0) & (wo.p_assembly_completes <= 1))
        assert np.all((wo.p_assembly_abort >= 0) & (wo.p_assembly_abort <= 1))
        # p == 1 everywhere in the stub, so no NaN expected_assembly_time
        assert not np.isnan(wo.expected_assembly_time_s).any()
    assert validate_prediction(V2, M, K, N_PLANS) is None


def test_v2_expected_assembly_time_nan_iff_p_zero():
    wall_id = next(iter(V2.walls))
    # p == 0 with NaN expected time: legitimate ("never completes")
    done = np.zeros((M, K), dtype=np.float64)
    pred = rep(
        V2,
        walls=walls_with(
            V2,
            wall_id,
            p_assembly_completes=done,
            expected_assembly_time_s=np.full((M, K), np.nan),
        ),
    )
    assert validate_prediction(pred, M, K, N_PLANS) is None
    # NaN exactly where p == 0 (finite elsewhere): legitimate, accepted
    mixed_p = np.ones((M, K), dtype=np.float64)
    mixed_p[0, 1] = 0.0
    mixed_t = np.full((M, K), 5.0, dtype=np.float64)
    mixed_t[0, 1] = np.nan  # p == 0 -> 'never completes' -> NaN allowed
    good = rep(
        V2,
        walls=walls_with(
            V2,
            wall_id,
            p_assembly_completes=mixed_p,
            expected_assembly_time_s=mixed_t,
        ),
    )
    assert validate_prediction(good, M, K, N_PLANS) is None
    # NaN where p > 0: hidden failure, rejected
    leaked = copied(mixed_t)
    leaked[0, 2] = np.nan  # p[0,2] == 1 but time NaN -> rejected
    with pytest.raises(PredictionViolation, match="expected_assembly_time_s"):
        validate_prediction(
            rep(
                V2,
                walls=walls_with(
                    V2,
                    wall_id,
                    p_assembly_completes=mixed_p,
                    expected_assembly_time_s=leaked,
                ),
            ),
            M,
            K,
            N_PLANS,
        )


def test_v2_future_wall_probability_bounds_enforced():
    wall_id = next(iter(V2.walls))
    bad_p = copied(V2.walls[wall_id].p_assembly_completes)
    bad_p[0, 0] = -0.1
    with pytest.raises(PredictionViolation, match="p_assembly_completes"):
        validate_prediction(
            rep(V2, walls=walls_with(V2, wall_id, p_assembly_completes=bad_p)),
            M, K, N_PLANS,
        )
    bad_abort = copied(V2.walls[wall_id].p_assembly_abort)
    bad_abort[0, 0] = 1.1
    with pytest.raises(PredictionViolation, match="p_assembly_abort"):
        validate_prediction(
            rep(V2, walls=walls_with(V2, wall_id, p_assembly_abort=bad_abort)),
            M, K, N_PLANS,
        )


def test_v1_v1_1_outputs_must_not_carry_future_wall_heads():
    from predict.reference_const_speed import ConstSpeedPredictor

    v1 = ConstSpeedPredictor().predict_many(SAMPLES, PLANS[:1], CS)[0]
    wall_id = next(iter(v1.walls))
    junk = {
        "p_assembly_completes": np.ones((M, K)),
        "expected_assembly_time_s": np.ones((M, K)),
        "p_assembly_abort": np.zeros((M, K)),
    }
    with pytest.raises(PredictionViolation, match="wd01_predict/2"):
        validate_prediction(
            rep(v1, walls=walls_with(v1, wall_id, **junk)), M, K, 1
        )


# --- §8.14: arrivals before t0, optional arrival_id ------------------------------

def test_s8_14_arrival_before_t0_is_accepted():
    t0 = SAMPLES[0].meta.time_s
    early = dataclasses.replace(
        SAMPLES[0],
        arrivals=(Arrival(t0 - 30.0, "mainline", 26.0, False, "car"),),
    )
    out = validate_state_samples([early])
    assert out[0].arrivals[0].t_s == t0 - 30.0 < t0
    with pytest.raises(StateSamplesViolation, match="t_s"):
        validate_state_samples(
            [dataclasses.replace(SAMPLES[0], arrivals=(
                Arrival(-1.0, "mainline", 26.0, False, "car"),))]
        )


def test_s8_14_arrival_id_optional_and_validated():
    t0 = SAMPLES[0].meta.time_s
    row = Arrival(t0 + 5.0, "ramp", 22.0, False, "car", arrival_id="arr-0007")
    out = validate_state_samples([dataclasses.replace(SAMPLES[0], arrivals=(row,))])
    assert out[0].arrivals[0].arrival_id == "arr-0007"
    assert Arrival(t0 + 5.0, "ramp", 22.0, False, "car").arrival_id is None
    with pytest.raises(StateSamplesViolation, match="arrival_id"):
        validate_state_samples(
            [dataclasses.replace(SAMPLES[0], arrivals=(
                Arrival(t0 + 5.0, "ramp", 22.0, False, "car", arrival_id=""),))]
        )
    # raw-dict rows gain the optional key the same way
    raw = StateSample.from_raw(
        {
            "hybrid": {"t0": 0.0, "section_len_m": 1000.0, "tracked": {}},
            "arrivals": [
                {"t_s": 1.0, "entrance": "ramp", "speed_mps": 20.0, "is_cav": True,
                 "vehicle_type": "cav", "arrival_id": "arr-1"}
            ],
            "meta": {"sample_id": "sX", "time_s": 0.0, "merge_observed": False,
                     "coverage": 0.5},
        }
    )
    assert raw.arrivals[0].arrival_id == "arr-1"


# --- §8.15: merge_observed bool; builder maps None -> False ----------------------

def test_s8_15_merge_observed_must_be_bool_fail_closed():
    bad = dataclasses.replace(
        SAMPLES[0],
        meta=SampleMeta("sX", SAMPLES[0].meta.time_s, None, 0.75),  # hand-built None
    )
    with pytest.raises(StateSamplesViolation, match="merge_observed"):
        validate_state_samples([bad])


def test_s8_15_builder_maps_none_to_false():
    """The StateSample builder maps hybrid-state merge_observed=None (no RSU
    inputs) to False — fail closed, counts toward ABSTAIN (§8.15)."""
    meta = SampleMeta.from_raw(
        {"sample_id": "sX", "time_s": 0.0, "merge_observed": None, "coverage": 0.5}
    )
    assert meta.merge_observed is False
    sample = StateSample.from_raw(
        {
            "hybrid": {"t0": 0.0, "section_len_m": 1000.0, "tracked": {}},
            "arrivals": (),
            "meta": {"sample_id": "sX", "time_s": 0.0, "merge_observed": None,
                     "coverage": 0.5},
        }
    )
    assert sample.meta.merge_observed is False
    assert validate_state_samples([sample])[0].meta.merge_observed is False


# --- the v2 conformance kit --------------------------------------------------------

def test_v2_kit_passes_both_continuations():
    for continuation in CONTINUATIONS:
        report = run_conformance_v2(ConstSpeedPredictorV2(), continuation=continuation)
        assert report.checks == V2_CHECKS
        assert report.predictor_id == "reference_const_speed/2"
        assert (report.n_plans, report.n_samples, report.M, report.K) == (4, 3, 1, 3)
        assert report.trajectory_s is None  # commit-window default


@pytest.mark.parametrize(
    "name", [n for n, _ in broken_predictors_v2()], ids=[n for n, _ in broken_predictors_v2()]
)
def test_v2_kit_rejects_broken_predictors(name):
    predictor = dict(broken_predictors_v2())[name]
    with pytest.raises(ConformanceError) as err:
        run_conformance_v2(predictor)
    assert "validators_pass" in str(err.value) or "v2_" in str(err.value)
