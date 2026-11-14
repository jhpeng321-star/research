"""§5 (v2) consumer rules — safety_veto_v2 aligned EXACTLY to the planner's
PL-01 reading (2026-09-27): one test per rule clause, incl. the owner's
future-collision vetoes, gating item G-1, and live_mode_allowed."""
import dataclasses
import inspect

import numpy as np
import pytest

from predict import (
    CONTRACT_VERSION_V1_1,
    CONTRACT_VERSION_V2,
    CONTINUATION_PLANNER,
    build_controller_state,
    build_plans,
    build_state_samples,
    live_mode_allowed,
    safety_veto_v2,
    validate_prediction,
)
from predict.reference_const_speed import ConstSpeedPredictor, ConstSpeedPredictorV2

CS = build_controller_state()
SAMPLES = build_state_samples(k=3)
PLANS = build_plans(CS)
V2 = ConstSpeedPredictorV2().predict_many(SAMPLES, PLANS[:1], CS)[0]
V1 = ConstSpeedPredictor().predict_many(SAMPLES, PLANS[:1], CS)[0]
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


# --- rule 1: fail closed ----------------------------------------------------------

def test_v2_clean_cells_are_not_vetoed():
    veto = safety_veto_v2(V2)
    assert veto.shape == V2.collision.shape and veto.dtype == np.bool_
    assert not veto.any()


def test_v2_rejects_non_v2_predictions():
    with pytest.raises(TypeError, match="Prediction"):
        safety_veto_v2({"plan_delay_s": 0.0})
    with pytest.raises(ValueError, match="wd01_predict/2"):
        safety_veto_v2(V1)
    v11 = ConstSpeedPredictor(contract_version=CONTRACT_VERSION_V1_1).predict_many(
        SAMPLES, PLANS[:1], CS
    )[0]
    with pytest.raises(ValueError, match="wd01_predict/2"):
        safety_veto_v2(v11)


def test_v2_missing_required_future_field_is_rejected_never_defaulted():
    with pytest.raises(ValueError, match="REQUIRED"):
        safety_veto_v2(rep(V2, future_cell_valid=None))
    with pytest.raises(ValueError, match="REQUIRED"):
        safety_veto_v2(rep(V2, native_p_future_collision=None))


# --- rule 1: fail closed ----------------------------------------------------------

def test_v2_plan_and_native_invalid_cells_fail_closed():
    cell_valid = copied(V2.cell_valid)
    cell_valid[0, 0] = False
    veto = safety_veto_v2(rep(V2, cell_valid=cell_valid))
    assert bool(veto[0, 0]) and not veto[0, 1:].any()
    native_valid = copied(V2.native_cell_valid)
    native_valid[0, 1] = False
    veto = safety_veto_v2(rep(V2, native_cell_valid=native_valid))
    assert bool(veto[0, 1]) and not veto[0, 0] and not veto[0, 2]


def test_v2_invalid_future_cells_fail_closed():
    future_valid = copied(V2.future_cell_valid)
    future_valid[0, 0] = False
    veto = safety_veto_v2(rep(V2, future_cell_valid=future_valid))
    assert bool(veto[0, 0]) and not veto[0, 1:].any()
    native_future_valid = copied(V2.native_future_cell_valid)
    native_future_valid[0, 2] = False
    veto = safety_veto_v2(rep(V2, native_future_cell_valid=native_future_valid))
    assert bool(veto[0, 2]) and not veto[0, :2].any()


def test_v2_nan_delay_fails_closed_including_continuation_part():
    part_b = copied(V2.delay_part_b_s)
    part_b[0, 1] = np.nan
    assert bool(safety_veto_v2(rep(V2, delay_part_b_s=part_b))[0, 1])
    plan = copied(V2.plan_delay_s)
    plan[0, 0] = np.nan
    assert bool(safety_veto_v2(rep(V2, plan_delay_s=plan))[0, 0])


def test_v2_window_collisions_fail_closed():
    collision = copied(V2.collision)
    collision[0, 0] = True
    assert bool(safety_veto_v2(rep(V2, collision=collision))[0, 0])
    # amended (owner decision 2026-09-27): a NATIVE collision no longer
    # vetoes at the new default — it is logged
    native_collision = copied(V2.native_collision)
    native_collision[0, 1] = True
    assert not safety_veto_v2(rep(V2, native_collision=native_collision)).any()


def test_v2_native_collision_veto_flag_both_ways():
    # amended (owner decision 2026-09-27, contract §7): the default is now
    # native_collision_vetoes=False; the planner can re-enable the veto
    native_collision = copied(V2.native_collision)
    native_collision[0, 1] = True
    assert not safety_veto_v2(rep(V2, native_collision=native_collision)).any()
    assert bool(safety_veto_v2(
        rep(V2, native_collision=native_collision), native_collision_vetoes=True
    )[0, 1])


def test_v2_wall_safety_abort_fails_the_plan_cell():
    wall_id = next(iter(V2.walls))
    abort = copied(V2.walls[wall_id].safety_abort)
    abort[0, 2] = True
    veto = safety_veto_v2(rep(V2, walls=walls_with(V2, wall_id, safety_abort=abort)))
    assert bool(veto[0, 2]) and not veto[0, :2].any()


# --- rule 2: absolute window gate --------------------------------------------------

def test_v2_window_ttc_floor_with_closing_pair():
    min_ttc = copied(V2.min_ttc_s)
    min_ttc[0, 0] = 1.2  # below the 1.5 s floor, pair present -> veto
    assert bool(safety_veto_v2(rep(V2, min_ttc_s=min_ttc))[0, 0])
    at_floor = copied(V2.min_ttc_s)
    at_floor[0, 0] = 1.5  # exactly at the floor passes
    assert not safety_veto_v2(rep(V2, min_ttc_s=at_floor)).any()
    # amended (PL-01 2026-09-27): the NATIVE window TTC is no longer
    # floor-checked — native_min_ttc_s < 1.5 does NOT veto
    native_min_ttc = copied(V2.native_min_ttc_s)
    native_min_ttc[0, 2] = 0.4
    assert not safety_veto_v2(rep(V2, native_min_ttc_s=native_min_ttc)).any()


def test_v2_nan_plan_ttc_with_pair_vetoes():
    # PL-01 rule 2: a plan NaN min TTC with a closing pair vetoes (fail closed)
    nan_ttc = copied(V2.min_ttc_s)
    nan_ttc[0, 2] = np.nan
    veto = safety_veto_v2(rep(V2, min_ttc_s=nan_ttc))
    assert bool(veto[0, 2]) and not veto[0, :2].any()


def test_v2_native_ttc_not_floor_checked_but_nan_with_pair_vetoes():
    # PL-01 rule 2 native clause: a finite native TTC below 1.5 s does NOT
    # veto; only a NaN native TTC with a closing pair vetoes
    low = copied(V2.native_min_ttc_s)
    low[:] = 0.4
    assert not safety_veto_v2(rep(V2, native_min_ttc_s=low)).any()
    nan_ttc = copied(V2.native_min_ttc_s)
    nan_ttc[0, 1] = np.nan
    veto = safety_veto_v2(rep(V2, native_min_ttc_s=nan_ttc))
    assert bool(veto[0, 1]) and not veto[0, 0] and not veto[0, 2]


def test_v2_nan_window_ttc_with_no_pair_passes_the_floor():
    no_pair = np.zeros((M, K), dtype=bool)
    pred = rep(
        V2,
        min_ttc_s=np.full((M, K), np.nan),
        has_closing_pair=no_pair,
        native_min_ttc_s=np.full((M, K), np.nan),
        native_has_closing_pair=no_pair,
    )
    assert validate_prediction(pred, M, K, 1) is None
    assert not safety_veto_v2(pred).any()


# --- rule 3: owner's future-collision vetoes (absolute + paired) --------------------

def test_v2_owner_absolute_future_collision_veto():
    p = copied(V2.p_future_collision)
    p[0, 2] = 0.02  # > p_max = 0.01 in ANY cell vetoes the whole call
    veto = safety_veto_v2(rep(V2, p_future_collision=p))
    assert veto.all()
    # exactly at p_max (and paired with an equal native p): the veto is
    # STRICTLY above, so the call survives
    at_bound_plan = copied(V2.p_future_collision)
    at_bound_native = copied(V2.native_p_future_collision)
    at_bound_plan[0, 2] = 0.01
    at_bound_native[0, 2] = 0.01
    survived = rep(
        V2, p_future_collision=at_bound_plan, native_p_future_collision=at_bound_native
    )
    assert not safety_veto_v2(survived).any()


def test_v2_owner_p_max_keyword_bound():
    # p_max is a keyword: the absolute veto fires only strictly above it
    # (paired plan/native keep the paired veto quiet)
    p = copied(V2.p_future_collision)
    p[0, 1] = 0.02  # above the default p_max = 0.01 ...
    pred = rep(
        V2,
        p_future_collision=p,
        native_p_future_collision=copied(p),  # equal native: no paired veto
    )
    assert safety_veto_v2(pred).all()
    assert not safety_veto_v2(pred, p_max=0.05).any()


def test_v2_owner_paired_future_collision_veto_with_tolerance():
    # paired case kept below p_max so the absolute veto cannot mask it
    p_plan = copied(V2.p_future_collision)
    p_plan[0, 1] = 0.005
    p_native = copied(V2.native_p_future_collision)
    p_native[0, 1] = 0.001
    pred = rep(V2, p_future_collision=p_plan, native_p_future_collision=p_native)
    veto = safety_veto_v2(pred)  # tol = 0: 0.005 > 0.001 + 0 -> cell vetoed
    assert bool(veto[0, 1]) and not veto[0, 0] and not veto[0, 2]
    veto_tol = safety_veto_v2(pred, tol_future_collision=0.05)
    assert not veto_tol.any()
    # the absolute veto still fires independently of the tolerance
    p_plan2 = copied(V2.p_future_collision)
    p_plan2[0, 0] = 0.5
    assert safety_veto_v2(
        rep(V2, p_future_collision=p_plan2), tol_future_collision=1.0
    ).all()


def test_v2_nan_future_probability_fails_closed():
    nan_p = np.full((M, K), np.nan)
    assert safety_veto_v2(rep(V2, p_future_collision=nan_p)).all()
    assert safety_veto_v2(rep(V2, native_p_future_collision=nan_p)).all()


# --- rule 4: relative safety, SUMMED (window + future) delta per metric ------------

def test_v2_relative_future_exposure_mean_rule_vetoes_call():
    # amended (PL-01 2026-09-27): the mean gate applies to the SUMMED
    # (window + future) delta — mean(delta) <= rel_safety_tol_s, default 0
    plan = np.full((M, K), 2.0)
    native = np.full((M, K), 1.0)
    pred = rep(V2, future_ttc_lt3_exposure_s=plan, native_future_ttc_lt3_exposure_s=native)
    assert safety_veto_v2(pred).all()


def test_v2_rule4_delta_sums_window_and_future_parts():
    # PL-01 rule 4: delta = (plan_m - native_m) + (plan_future_m -
    # native_future_m); each part alone is within rel_safety_tol_s=0.4,
    # only the SUM vetoes
    w = np.full((M, K), 0.2)
    f = np.full((M, K), 0.3)
    win_only = rep(V2, hard_brake_s=w)  # native and future all 0
    fut_only = rep(V2, future_hard_brake_s=f)  # plan-window and native all 0
    assert not safety_veto_v2(win_only, rel_safety_tol_s=0.4).any()
    assert not safety_veto_v2(fut_only, rel_safety_tol_s=0.4).any()
    summed = rep(win_only, future_hard_brake_s=f)
    assert safety_veto_v2(summed, rel_safety_tol_s=0.4).all()


def test_v2_relative_future_exposure_fraction_rule_vetoes_call():
    # amended (PL-01 2026-09-27): the <= fraction gate applies to the SUMMED
    # delta; one increasing cell out of three keeps vetoing the whole call
    plan = np.full((M, K), 1.0)
    native = np.full((M, K), 2.0)
    native[0, 2] = 0.5  # only this cell increases: delta = +0.5 > 0
    pred = rep(V2, future_hard_brake_s=plan, native_future_hard_brake_s=native)
    assert safety_veto_v2(pred).all()


def test_v2_rule4_quantile_gate_vetoes_when_mean_is_ok():
    # PL-01 rule 4 quantile clause: mean(delta <= tol) >= rel_safety_quantile
    # (default 0.90); the mean gate alone would pass this call
    delta = np.array([-10.0, 0.0, 0.5])  # mean -3.17 <= 0 ...
    pred = rep(V2, hard_brake_s=delta)  # native + future all 0
    assert safety_veto_v2(pred).all()  # ... but 2/3 <= 0 < 0.90 vetoes
    assert not safety_veto_v2(pred, rel_safety_quantile=0.5).any()


def test_v2_rule4_rel_safety_tol_keyword():
    # rel_safety_tol_s loosens BOTH gates: mean(delta) <= tol and per-cell
    # delta <= tol (boundary inclusive)
    pred = rep(V2, future_hard_brake_s=np.full((M, K), 0.5))
    assert safety_veto_v2(pred).all()  # default tol 0.0: veto
    assert not safety_veto_v2(pred, rel_safety_tol_s=0.5).any()  # at the bound: ok


def test_v2_rule4_window_must_not_increase_flag_both_ways():
    # amended (owner decision 2026-09-27, contract §7): the window-only
    # clause is now ON by default (True); False restores the pre-decision
    # summed-only behaviour
    pred = rep(
        V2,
        hard_brake_s=np.full((M, K), 0.5),  # window delta +0.5
        native_future_hard_brake_s=np.full((M, K), 1.0),  # future delta 0 - 1 = -1
    )
    assert safety_veto_v2(pred).all()  # default True: window part fails
    assert not safety_veto_v2(pred, window_must_not_increase=False).any()


def test_v2_rule4_window_must_not_increase_quantile_gate():
    # amended (owner decision 2026-09-27): the window clause is the default;
    # the window mean passes but its quantile gate fails (2/3 <= 0 < 0.90)
    # while the summed delta passes everywhere
    win_delta = np.array([-10.0, 0.0, 0.5])
    pred = rep(
        V2,
        hard_brake_s=win_delta,
        native_future_hard_brake_s=np.array([0.0, 0.0, 0.6]),  # fut delta (0, 0, -0.6)
    )
    assert safety_veto_v2(pred).all()  # default True vetoes via the window gate
    assert not safety_veto_v2(pred, window_must_not_increase=False).any()


def test_v2_owner_decision_2026_09_27_defaults():
    # owner decisions 2026-09-27 (PREDICT_CONTRACT_v2.md §7): the REFERENCE
    # defaults are native_collision_vetoes=False (a native collision is
    # logged, not a veto) and window_must_not_increase=True (the commit
    # window must be non-increasing on its own)
    params = inspect.signature(safety_veto_v2).parameters
    assert params["native_collision_vetoes"].default is False
    assert params["window_must_not_increase"].default is True
    # behaviour at the defaults: a native-collision-only call passes ...
    native_collision = copied(V2.native_collision)
    native_collision[0, 1] = True
    assert not safety_veto_v2(rep(V2, native_collision=native_collision)).any()
    # ... and a summed-pass / window-increase call is vetoed by the window clause
    pred = rep(
        V2,
        hard_brake_s=np.full((M, K), 0.5),  # window delta +0.5
        native_future_hard_brake_s=np.full((M, K), 1.0),  # future delta 0 - 1 = -1
    )
    assert safety_veto_v2(pred).all()


def test_v2_rule4_nan_delta_anywhere_vetoes_all():
    # PL-01 rule 4: ANY non-finite delta anywhere fails (veto the whole call)
    nan_native = copied(V2.native_future_ttc_lt3_exposure_s)
    nan_native[0, 1] = np.nan
    assert safety_veto_v2(rep(V2, native_future_ttc_lt3_exposure_s=nan_native)).all()
    inf_native = copied(V2.native_hard_brake_s)
    inf_native[0, 0] = np.inf
    assert safety_veto_v2(rep(V2, native_hard_brake_s=inf_native)).all()


def test_v2_relative_window_exposure_rules_fire_too():
    # amended (PL-01 2026-09-27): the window part enters the summed delta,
    # so a window-only increase still vetoes on its own
    plan = np.full((M, K), 2.0)
    zeros = np.zeros((M, K))
    pred = rep(V2, hard_brake_s=plan, native_hard_brake_s=zeros)
    assert safety_veto_v2(pred).all()
    pred = rep(V2, ttc_lt3_exposure_s=plan, native_ttc_lt3_exposure_s=zeros)
    assert safety_veto_v2(pred).all()


def test_v2_no_relative_increase_when_native_dominates():
    plan = np.full((M, K), 1.0)
    native = np.full((M, K), 2.0)
    pred = rep(V2, future_ttc_lt3_exposure_s=plan, native_future_ttc_lt3_exposure_s=native)
    assert not safety_veto_v2(pred).any()


# --- rule 5 / G-1: the future min-TTC floor needs validation ------------------------

def test_v2_g1_future_ttc_floor_only_when_validated():
    risky_q05 = np.full((M, K), 0.5)  # below 1.5 s beyond Δ, pair present
    pred = rep(V2, future_min_ttc_q05_s=risky_q05)
    # unvalidated head (the stub's state): the floor does NOT apply beyond Δ
    assert not safety_veto_v2(pred).any()
    # validated head: the 1.5 s floor extends to the future
    validated = repmeta(
        pred,
        future_min_ttc_validated=True,
        future_min_ttc_validation_id="PREDICT_LEDGER/2026-09-26/min_ttc_v1",
    )
    assert safety_veto_v2(validated).all()


def test_v2_g1_nan_future_ttc_with_pair_fails_closed_when_validated():
    nan_q05 = np.full((M, K), np.nan)
    validated = repmeta(
        rep(V2, future_min_ttc_q05_s=nan_q05),
        future_min_ttc_validated=True,
        future_min_ttc_validation_id="PREDICT_LEDGER/2026-09-26/min_ttc_v1",
    )
    assert safety_veto_v2(validated).all()
    # with no closing pair the NaN is legitimate and passes the floor
    no_pair = rep(V2, future_has_closing_pair=np.zeros((M, K), dtype=bool))
    validated = repmeta(
        rep(no_pair, future_min_ttc_q05_s=nan_q05),
        future_min_ttc_validated=True,
        future_min_ttc_validation_id="PREDICT_LEDGER/2026-09-26/min_ttc_v1",
    )
    assert not safety_veto_v2(validated).any()


def test_v2_g1_native_future_q05_nan_with_pair_vetoes_whether_or_not_validated():
    # PL-01 rule 5: a native future_min_ttc_q05_s NaN with a native future
    # closing pair vetoes — independent of meta.future_min_ttc_validated
    nan_q05 = copied(V2.native_future_min_ttc_q05_s)
    nan_q05[0, 0] = np.nan
    pred = rep(V2, native_future_min_ttc_q05_s=nan_q05)  # stub ships unvalidated
    veto = safety_veto_v2(pred)
    assert bool(veto[0, 0]) and not veto[0, 1:].any()
    validated = repmeta(
        pred,
        future_min_ttc_validated=True,
        future_min_ttc_validation_id="PREDICT_LEDGER/2026-09-26/min_ttc_v1",
    )
    veto = safety_veto_v2(validated)
    assert bool(veto[0, 0]) and not veto[0, 1:].any()


def test_v2_live_mode_allowed_requires_validated_head_with_id():
    # the reference stub ships unvalidated: live mode REFUSED (§5 G-1)
    assert live_mode_allowed(V2.meta) is False
    assert (
        live_mode_allowed(repmeta(V2, future_min_ttc_validated=True).meta) is False
    )  # flag without id
    assert (
        live_mode_allowed(repmeta(V2, future_min_ttc_validation_id="ledger/x").meta)
        is False
    )  # id without flag
    assert (
        live_mode_allowed(
            repmeta(
                V2,
                future_min_ttc_validated=True,
                future_min_ttc_validation_id="PREDICT_LEDGER/2026-09-26/min_ttc_v1",
            ).meta
        )
        is True
    )
    # v1/v1.1 outputs have no validated future head: refused
    assert live_mode_allowed(V1.meta) is False
