"""§4 Prediction rules — positive and negative tests for every rule, exercised
on the reference predictor's output."""
import dataclasses

import numpy as np
import pytest

from predict import (
    PredictionViolation,
    build_controller_state,
    build_plans,
    build_state_samples,
    validate_prediction,
)
from predict.reference_const_speed import ConstSpeedPredictor

CS = build_controller_state()
SAMPLES = build_state_samples(k=3)
PLANS = build_plans(CS)
N_PLANS = 2
PRED = ConstSpeedPredictor().predict_many(SAMPLES, PLANS[:N_PLANS], CS)[0]
M, K = len(PRED.meta.member_ids), len(PRED.meta.sample_ids)

_MK_ARRAY_FIELDS = (
    "delay_vs_native_s",
    "plan_delay_s",
    "native_delay_s",
    "delay_part_a_s",
    "delay_part_b_s",
    "collision",
    "min_ttc_s",
    "ttc_lt3_exposure_s",
    "hard_brake_s",
    "native_collision",
    "native_min_ttc_s",
    "native_ttc_lt3_exposure_s",
    "native_hard_brake_s",
)


def repred(**kw):
    return dataclasses.replace(PRED, **kw)


def repmeta(**kw):
    return repred(meta=dataclasses.replace(PRED.meta, **kw))


def walls_with(wall_id, **kw):
    wo = dataclasses.replace(PRED.walls[wall_id], **kw)
    return {**PRED.walls, wall_id: wo}


def copied(arr):
    return np.array(arr, copy=True)


# --- positives ----------------------------------------------------------------

def test_valid_prediction_passes():
    assert validate_prediction(PRED, M, K, N_PLANS, None) is None


def test_valid_prediction_passes_with_explicit_trajectory_s():
    assert validate_prediction(PRED, M, K, N_PLANS, trajectory_s=60.0) is None


def test_every_array_field_is_numpy():
    for name in _MK_ARRAY_FIELDS:
        assert isinstance(getattr(PRED, name), np.ndarray), name
    for name in ("trajectories_tracked", "native_trajectories_tracked"):
        traj = getattr(PRED, name)
        assert traj, name
        assert all(isinstance(v, np.ndarray) for v in traj.values()), name
    for wo in PRED.walls.values():
        for name in ("assembled", "assembled_at_s", "safety_abort", "abort_reason"):
            assert isinstance(getattr(wo, name), np.ndarray), name


def test_min_ttc_may_be_nan_when_no_pairs_exist():
    nan = np.full((M, K), np.nan)
    assert validate_prediction(repred(min_ttc_s=nan, native_min_ttc_s=nan), M, K, N_PLANS) is None


# --- §4.1/§4.2 [M, K] arrays ---------------------------------------------------

def test_trajectory_s_beyond_window_fails_t_check():
    with pytest.raises(PredictionViolation, match="shape"):
        validate_prediction(PRED, M, K, N_PLANS, trajectory_s=120.0)


def test_arrays_must_be_numpy():
    with pytest.raises(PredictionViolation, match="numpy array"):
        validate_prediction(repred(plan_delay_s=[[1.0, 2.0, 3.0]]), M, K, N_PLANS)


def test_delay_dtype_must_be_float():
    bad = PRED.plan_delay_s.astype(np.int64)
    with pytest.raises(PredictionViolation, match="float dtype"):
        validate_prediction(repred(plan_delay_s=bad), M, K, N_PLANS)


def test_array_shape_must_be_mk():
    bad = PRED.plan_delay_s[:, :2]
    with pytest.raises(PredictionViolation, match=r"got \(1, 2\)"):
        validate_prediction(repred(plan_delay_s=bad), M, K, N_PLANS)


def test_bool_fields_must_have_bool_dtype():
    bad = PRED.collision.astype(np.float64)
    with pytest.raises(PredictionViolation, match="bool array"):
        validate_prediction(repred(collision=bad), M, K, N_PLANS)


def test_delay_and_exposure_fields_must_be_finite():
    bad = copied(PRED.hard_brake_s)
    bad[0, 0] = np.nan
    with pytest.raises(PredictionViolation, match="finite"):
        validate_prediction(repred(hard_brake_s=bad), M, K, N_PLANS)


def test_delay_vs_native_must_equal_plan_minus_native():
    bad = copied(PRED.delay_vs_native_s) + 1.0
    with pytest.raises(PredictionViolation, match="delay_vs_native_s"):
        validate_prediction(repred(delay_vs_native_s=bad), M, K, N_PLANS)


def test_part_a_plus_part_b_must_sum_to_plan_delay():
    bad = copied(PRED.delay_part_a_s) + 2.0
    with pytest.raises(PredictionViolation, match="must sum"):
        validate_prediction(repred(delay_part_a_s=bad), M, K, N_PLANS)


# --- §4.3 trajectories ---------------------------------------------------------

def _traj(name):
    arr = copied(PRED.trajectories_tracked["cav_a0"])
    traj = dict(PRED.trajectories_tracked)
    traj["cav_a0"] = arr
    return repred(**{name: traj}), arr


def test_trajectory_nan_must_be_a_trailing_suffix():
    p, arr = _traj("trajectories_tracked")
    arr[0, 0, 5, :] = np.nan  # hole in the middle
    with pytest.raises(PredictionViolation, match="suffix|holes"):
        validate_prediction(p, M, K, N_PLANS)


def test_trajectory_nan_must_cover_whole_rows():
    p, arr = _traj("trajectories_tracked")
    arr[0, 0, 40, 3] = np.nan  # speed column only
    with pytest.raises(PredictionViolation, match="whole"):
        validate_prediction(p, M, K, N_PLANS)


def test_trajectory_step0_must_be_real():
    p, arr = _traj("trajectories_tracked")
    arr[0, 0, 0, :] = np.nan
    with pytest.raises(PredictionViolation, match="suffix|t0|step 0"):
        validate_prediction(p, M, K, N_PLANS)


def test_trajectories_must_be_non_empty_mapping():
    with pytest.raises(PredictionViolation, match="non-empty mapping"):
        validate_prediction(repred(trajectories_tracked={}), M, K, N_PLANS)


def test_trajectory_vehicle_ids_must_be_strings():
    traj = {5: PRED.trajectories_tracked["cav_a0"]}
    with pytest.raises(PredictionViolation, match="vehicle ids"):
        validate_prediction(repred(trajectories_tracked=traj), M, K, N_PLANS)


# --- §4.4 walls -----------------------------------------------------------------

def test_abort_reason_must_be_int8():
    wo = PRED.walls["W1"]
    bad = wo.abort_reason.astype(np.int32)
    with pytest.raises(PredictionViolation, match="int8"):
        validate_prediction(repred(walls=walls_with("W1", abort_reason=bad)), M, K, N_PLANS)


def test_abort_reason_values_are_restricted():
    bad = copied(PRED.walls["W1"].abort_reason)
    bad[0, 0] = 3
    with pytest.raises(PredictionViolation, match="allowed"):
        validate_prediction(repred(walls=walls_with("W1", abort_reason=bad)), M, K, N_PLANS)


def test_abort_reason_must_agree_with_safety_abort():
    bad = copied(PRED.walls["W1"].abort_reason)
    bad[0, 0] = 1  # safety abort while safety_abort is False
    with pytest.raises(PredictionViolation, match="safety_abort"):
        validate_prediction(repred(walls=walls_with("W1", abort_reason=bad)), M, K, N_PLANS)


def test_assembled_at_s_nan_iff_not_assembled():
    wo = PRED.walls["W1"]
    assembled = copied(wo.assembled)
    assembled[0, 1] = False  # keep assembled_at_s finite -> violation
    with pytest.raises(PredictionViolation, match="NaN exactly"):
        validate_prediction(
            repred(walls=walls_with("W1", assembled=assembled)), M, K, N_PLANS
        )
    at = copied(wo.assembled_at_s)
    at[0, 0] = -1.0  # assembled but negative timestamp
    with pytest.raises(PredictionViolation, match=">= 0"):
        validate_prediction(repred(walls=walls_with("W1", assembled_at_s=at)), M, K, N_PLANS)


def test_wall_outcome_must_be_typed():
    with pytest.raises(PredictionViolation, match="WallOutcome"):
        validate_prediction(repred(walls={"W1": dict(PRED.walls["W1"].__dict__)}), M, K, N_PLANS)


# --- §4.5 optional native-unlocked diagnostics -----------------------------------

def test_unlocked_fields_absent_when_unavailable():
    assert PRED.meta.native_unlocked_available is False
    with pytest.raises(PredictionViolation, match="available=False"):
        validate_prediction(
            repred(native_unlocked_delay_s=copied(PRED.native_delay_s)), M, K, N_PLANS
        )


UNLOCKED = repred(
    native_unlocked_delay_s=copied(PRED.native_delay_s),
    native_unlocked_min_ttc_s=copied(PRED.native_min_ttc_s),
    native_unlocked_ttc_lt3_exposure_s=copied(PRED.native_ttc_lt3_exposure_s),
    native_unlocked_hard_brake_s=copied(PRED.native_hard_brake_s),
    native_unlocked_collision=copied(PRED.native_collision),
    meta=dataclasses.replace(PRED.meta, native_unlocked_available=True),
)


def test_full_unlocked_set_passes_when_available():
    assert validate_prediction(UNLOCKED, M, K, N_PLANS) is None


def test_partial_unlocked_set_fails_when_available():
    with pytest.raises(PredictionViolation, match="missing"):
        validate_prediction(
            dataclasses.replace(UNLOCKED, native_unlocked_collision=None), M, K, N_PLANS
        )


# --- §4.6 meta -------------------------------------------------------------------

def test_contract_version_is_checked():
    # a literal outside SUPPORTED_CONTRACT_VERSIONS (v2 joined the set, so the
    # probe version moved off the real namespace entirely)
    with pytest.raises(PredictionViolation, match="contract_version"):
        validate_prediction(repmeta(contract_version="wd01_predict/9"), M, K, N_PLANS)


def test_safety_window_equals_part_a_horizon():
    with pytest.raises(PredictionViolation, match="safety_window_s"):
        validate_prediction(repmeta(safety_window_s=61.0), M, K, N_PLANS)


def test_weights_sha256_has_one_entry_per_member():
    with pytest.raises(PredictionViolation, match="one entry per member"):
        validate_prediction(repmeta(weights_sha256=("a", "b")), M, K, N_PLANS)


def test_sample_ids_length_k():
    with pytest.raises(PredictionViolation, match="length K"):
        validate_prediction(repmeta(sample_ids=("s0", "s1")), M, K, N_PLANS)


def test_member_ids_length_m():
    with pytest.raises(PredictionViolation, match="length M"):
        validate_prediction(repmeta(member_ids=("a", "b")), M, K, N_PLANS)


@pytest.mark.parametrize(
    "kw,fragment",
    [
        ({"horizon_s": 0.0}, "horizon_s"),
        ({"wall_clock_s": -1.0}, "wall_clock_s"),
        ({"model_id": ""}, "model_id"),
        ({"device": ""}, "device"),
        ({"native_kept_walls": "W1"}, "native_kept_walls"),
        ({"part_a_horizon_s": 0.0}, "part_a_horizon_s"),
    ],
)
def test_meta_value_rules(kw, fragment):
    with pytest.raises(PredictionViolation, match=fragment):
        validate_prediction(repmeta(**kw), M, K, N_PLANS)
