"""§9 FAIL CLOSED consumer rule — safety_veto (plan-side reference helper)."""
import dataclasses

import numpy as np
import pytest

from predict import (
    CONTRACT_VERSION_V1_1,
    build_controller_state,
    build_plans,
    build_state_samples,
    safety_veto,
    validate_prediction,
)
from predict.reference_const_speed import ConstSpeedPredictor

CS = build_controller_state()
SAMPLES = build_state_samples(k=3)
PLANS = build_plans(CS)
PRED = ConstSpeedPredictor().predict_many(SAMPLES, PLANS[:1], CS)[0]
V11 = ConstSpeedPredictor(contract_version=CONTRACT_VERSION_V1_1).predict_many(
    SAMPLES, PLANS[:1], CS
)[0]


def rep(pred, **kw):
    return dataclasses.replace(pred, **kw)


def test_s9_clean_cells_are_not_vetoed():
    veto = safety_veto(PRED)
    assert veto.shape == PRED.collision.shape and veto.dtype == np.bool_
    assert not veto.any()


def test_s9_invalid_cell_fails_closed_before_min_ttc():
    cell_valid = np.array(V11.cell_valid, copy=True)
    cell_valid[0, 0] = False
    veto = safety_veto(rep(V11, cell_valid=cell_valid))
    assert bool(veto[0, 0]) and not veto[0, 1:].any()
    native_cell_valid = np.array(V11.native_cell_valid, copy=True)
    native_cell_valid[0, 1] = False
    veto = safety_veto(rep(V11, native_cell_valid=native_cell_valid))
    assert bool(veto[0, 1]) and not veto[0, 0]


def test_s9_nan_in_any_delay_field_fails_closed_both_versions():
    for pred in (PRED, V11):
        delay = np.array(pred.plan_delay_s, copy=True)
        delay[0, 0] = np.nan
        assert bool(safety_veto(rep(pred, plan_delay_s=delay))[0, 0])
        native = np.array(pred.native_delay_s, copy=True)
        native[0, 1] = np.nan
        assert bool(safety_veto(rep(pred, native_delay_s=native))[0, 1])
        vs = np.array(pred.delay_vs_native_s, copy=True)
        vs[0, 2] = np.nan
        assert bool(safety_veto(rep(pred, delay_vs_native_s=vs))[0, 2])
        for f in ("delay_part_a_s", "delay_part_b_s"):
            arr = np.array(getattr(pred, f), copy=True)
            arr[0, 0] = np.nan
            assert bool(safety_veto(rep(pred, **{f: arr}))[0, 0])


def test_s9_collision_and_native_collision_veto():
    collision = np.array(PRED.collision, copy=True)
    collision[0, 0] = True
    assert bool(safety_veto(rep(PRED, collision=collision))[0, 0])
    native = np.array(PRED.native_collision, copy=True)
    native[0, 1] = True
    assert bool(safety_veto(rep(PRED, native_collision=native))[0, 1])


def test_s9_wall_safety_abort_fails_the_plan():
    from predict import WallOutcome

    walls = dict(PRED.walls)
    wall_id = next(iter(walls))
    abort = np.zeros(walls[wall_id].safety_abort.shape, dtype=bool)
    abort[0, 0] = True
    walls[wall_id] = dataclasses.replace(walls[wall_id], safety_abort=abort)
    assert bool(safety_veto(rep(PRED, walls=walls))[0, 0])


def test_s9_nan_min_ttc_with_no_pair_is_not_a_veto():
    """§9: a NaN min_ttc in a valid cell with has_closing_pair False passes the
    TTC floor — the veto rule must NOT fire on it (the planner reads min_ttc
    only after the vetoes)."""
    pred = rep(
        V11,
        min_ttc_s=np.full(V11.min_ttc_s.shape, np.nan),
        has_closing_pair=np.zeros(V11.has_closing_pair.shape, dtype=bool),
    )
    validate_prediction(pred, *pred.collision.shape, 1)
    assert not safety_veto(pred).any()


def test_s9_veto_rejects_non_predictions():
    with pytest.raises(TypeError, match="Prediction"):
        safety_veto({"plan_delay_s": 0.0})
