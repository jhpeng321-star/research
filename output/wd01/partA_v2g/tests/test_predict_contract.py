"""Metrics conventions (hand-computed) + predictor contract v2.1 (M=1)
validation, including the shared validator's new optional fields."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

WD01 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WD01))

from partA_v2g import metrics as MX
from partA_v2g.engine import load_params
from partA_v2g.predict import PartAV2GeneralPredictor

from predict.contract import (ControllerStateEntry, PlanCandidate,
                              Prediction, SampleMeta, StateSample, Arrival,
                              WallSpec)
from predict.validate import (PredictionViolation, validate_prediction)

P = load_params()
DT = P["dt_s"]


# ------------------------------------------------------------- metrics

def test_delay_step_hand_computed():
    # lane 1, one 0.5 s step at half the limit: delay = dt - ds/lim
    d = MX.delay_step(1, 2000.0, 1, 2000.0 + 0.5 * 31.29 / 2, P)
    assert d == pytest.approx(DT * 0.5)
    # at the limit: zero delay
    d0 = MX.delay_step(1, 2000.0, 1, 2000.0 + 0.5 * 31.29, P)
    assert d0 == pytest.approx(0.0, abs=1e-12)


def test_window_delay_single_vehicle():
    snaps = [
        {"x": (1, 2000.0, 20.0, 0.0)},
        {"x": (1, 2007.8225, 20.0, 0.0)},     # half the limit for 0.5 s
    ]
    assert MX.window_delay(snaps, P) == pytest.approx(DT * 0.5)


def test_ttc_tick_and_exposure():
    # one closing pair: gap 30, closing 10 -> ttc 3.0 (not < 3)
    lt15, lt3, m = MX.ttc_tick(np.array([1, 1]),
                               np.array([2000.0, 2030.0]),
                               np.array([20.0, 10.0]))
    assert (lt15, lt3) == (0, 0)
    assert m == pytest.approx(3.0)
    # closing 11 -> ttc 30/11 < 3
    lt15, lt3, m = MX.ttc_tick(np.array([1, 1]),
                               np.array([2000.0, 2030.0]),
                               np.array([20.0, 9.0]))
    assert lt3 == 1 and lt15 == 0
    assert m == pytest.approx(30.0 / 11.0)
    expo, min_ttc, closing = MX.ttc_exposure(
        [snaps := {"x": (1, 2000.0, 20.0, 0.0), "y": (1, 2030.0, 9.0, 0.0)}]
        * 3, P)
    assert closing is True
    assert expo == pytest.approx(2 * DT)       # two post-t0 ticks
    assert min_ttc == pytest.approx(30.0 / 11.0)


def test_hard_brake_and_lane_change_counts():
    snaps = [
        {"x": (1, 2000.0, 20.0, 0.0)},
        {"x": (1, 1998.0, 18.2, -3.6)},        # a = -3.6 < -3
        {"x": (1, 1997.0, 18.0, -0.4)},
    ]
    assert MX.hard_brake_s(snaps, P) == pytest.approx(DT)
    rows = [
        {"x": (2, 400.0, 18.0, 0.0), "y": (1, 2300.0, 20.0, 0.0)},
        {"x": (1, 2265.56, 18.0, 0.0), "y": (1, 2300.0, 20.0, 0.0)},
        {"x": (0, 1000.0, 18.0, 0.0), "y": (1, 2305.0, 20.0, 0.0)},
    ]
    lc = MX.lane_change_counts(rows)
    assert lc == {"merge": 1, "disc_left": 1, "disc_right": 0}


def test_traj_rmse_matched_ids_only():
    pred = [
        {"x": (1, 2000.0, 20.0, 0.0)},
        {"x": (1, 2010.0, 20.0, 0.0)},
    ]
    truth = [
        {"x": (1, 2000.0, 20.0, 0.0)},
        {"x": (1, 2013.0, 21.0, 0.0)},
    ]
    out = MX.traj_rmse(pred, truth, ["x"])
    assert out["s_rmse"] == pytest.approx(3.0)
    assert out["v_rmse"] == pytest.approx(1.0)
    assert out["n_matched"] == 1


def test_window_series_consistent_with_reference_functions():
    """The eval-side one-pass WindowSeries must reproduce window_delay /
    ttc_exposure / hard_brake_s / lane_change_counts exactly."""
    snaps = [
        {"x": (1, 2000.0, 20.0, 0.0), "y": (1, 2030.0, 9.0, 0.0)},
        {"x": (1, 2007.8225, 20.0, -3.6), "y": (1, 2034.0, 9.0, 0.0)},
        {"x": (2, 400.0, 18.0, 0.2), "y": (1, 2038.0, 9.0, 0.0)},
        {"x": (1, 2665.56, 18.0, 0.0), "y": (0, 999.0, 9.0, 0.0)},
    ]
    ids = ["x", "y"]
    ser = MX.WindowSeries(snaps, ids)
    for j_hi in (1, 2, 3):
        st = ser.stats(j_hi, P)
        assert st["delay"] == pytest.approx(
            MX.window_delay(snaps[:j_hi + 1], P, ids))
        expo, min_ttc, closing = MX.ttc_exposure(snaps[:j_hi + 1], P, ids)
        assert st["expo3"] == pytest.approx(expo)
        assert st["min_ttc"] == (pytest.approx(min_ttc)
                                 if min_ttc == min_ttc else True)
        assert st["closing"] == closing
        assert st["hard"] == pytest.approx(
            MX.hard_brake_s(snaps[:j_hi + 1], P, ids))
        assert st["lc"] == MX.lane_change_counts(snaps[:j_hi + 1])
    # ids=None: the all-vehicles view includes ids absent from step 0
    snaps2 = snaps + [{"x": (1, 2670.0, 18.0, 0.0), "y": (0, 1005.0, 9.0, 0.0),
                       "z": (1, 2100.0, 10.0, 0.0)}]
    ser_all = MX.WindowSeries(snaps2, None)
    st_all = ser_all.stats(4, P)
    assert st_all["delay"] == pytest.approx(MX.window_delay(snaps2, P, None))
    # rmse parity with traj_rmse (x present at every step)
    ser2 = MX.WindowSeries(snaps, ["x"])
    r = ser2.rmse(MX.WindowSeries(snaps, ["x"]), 3)
    assert r["n_matched"] == 3
    assert r["s_rmse"] == pytest.approx(
        MX.traj_rmse(snaps[:4], snaps[:4], ["x"])["s_rmse"])


# ----------------------------------------------------- predictor contract

def _sample(seed_variant=0, with_arrivals=False):
    vs = {
        "m1": {"s_down_m": -700.0, "lane_phys": 1,
               "speed_mps": 28.0 - 0.3 * seed_variant},
        "m2": {"s_down_m": -650.0, "lane_phys": 1, "speed_mps": 27.0},
        "r1": {"s_down_m": -300.0, "lane_phys": 2, "speed_mps": 18.0},
    }
    hybrid = {
        "t_s": 480.0,
        "header": {"t_s": 480.0, "k": seed_variant, "K": 2,
                   "sample_weight": 0.5},
        "zone_extent": {"0": [-800.0, 225.44], "1": [-800.0, 225.44],
                        "2": [-378.66, 75.44]},
        "vehicles": vs,
        "cav_controller_block": {},
    }
    arrivals = (Arrival(500.0, "mainline", 28.0, False, "car"),) \
        if with_arrivals else ()
    return StateSample(hybrid=hybrid, arrivals=arrivals,
                       meta=SampleMeta(f"s{seed_variant}", 480.0, True, 0.9))


CS = {"m1": ControllerStateEntry("native")}
PLAN = PlanCandidate(plan_version=1, per_cav={"m1": "native"}, walls=())


def test_contract_v2_1_validator_passes_with_m_equals_1():
    pred = PartAV2GeneralPredictor()
    p = pred.predict_many([_sample(0), _sample(1)], [PLAN], CS,
                          horizon_s=120.0, commit_s=30.0,
                          continuation="native")[0]
    validate_prediction(p, 1, 2, 1, 30.0)      # M = 1 explicitly
    assert p.meta.part_a_scope == "tracked"
    assert p.meta.part_a_model == "v2_general"
    assert p.meta.param_set_id == pred.weights_sha256()
    assert p.n_tracked.shape == (1, 2)
    assert float(p.n_tracked[0, 0]) == 3.0
    assert bool(np.isfinite(p.delay_part_a_s).all())
    assert p.delay_part_b_s[0, 0] == pytest.approx(0.0)
    # v2 identities
    assert np.allclose(p.delay_vs_native_s,
                       p.plan_delay_s - p.native_delay_s)
    assert np.allclose(p.delay_part_a_s + p.delay_part_b_s, p.plan_delay_s)


def test_planner_continuation_and_long_trajectory():
    pred = PartAV2GeneralPredictor()
    p = pred.predict_many([_sample(0)], [PLAN], CS, horizon_s=120.0,
                          commit_s=30.0, continuation="planner",
                          trajectory_s=120.0)[0]
    validate_prediction(p, 1, 1, 1, 120.0)
    assert p.meta.continuation_policy_id == "parta_v2g/hold-commit-actions"
    assert p.trajectories_tracked["m1"].shape == (1, 1, 121, 4)


def test_horizon_beyond_rollout_rejected():
    pred = PartAV2GeneralPredictor()
    with pytest.raises(ValueError):
        pred.predict_many([_sample(0)], [PLAN], CS, horizon_s=600.0)


def test_arrivals_ignored_never_refused():
    pred = PartAV2GeneralPredictor()
    with_arr = pred.predict_many([_sample(0, with_arrivals=True)], [PLAN],
                                 CS, horizon_s=120.0)[0]
    without = pred.predict_many([_sample(0, with_arrivals=False)], [PLAN],
                                CS, horizon_s=120.0)[0]
    assert float(with_arr.delay_part_a_s[0, 0]) == \
        pytest.approx(float(without.delay_part_a_s[0, 0]))
    assert with_arr.trajectories_tracked.keys() == \
        without.trajectories_tracked.keys()
    assert float(with_arr.n_tracked[0, 0]) == \
        float(without.n_tracked[0, 0])      # no insertion from arrivals


def test_predict_identity_native_shared_determinism():
    """Mini conformance: predict == predict_many[0]; native arrays shared
    across plans of one call; a repeat call is identical; a damping plan
    with a close slow leader genuinely differs from native (the command
    binds below the IDM safe speed)."""
    pred = PartAV2GeneralPredictor()

    def mk(dv=0.0):
        return {
            "t_s": 480.0,
            "header": {"t_s": 480.0, "k": 0, "K": 2, "sample_weight": 0.5},
            "vehicles": {
                "m1": {"s_down_m": -700.0, "lane_phys": 1,
                       "speed_mps": 28.0 - dv},
                "m2": {"s_down_m": -675.0, "lane_phys": 1,
                       "speed_mps": 15.0},
                "r1": {"s_down_m": -300.0, "lane_phys": 2,
                       "speed_mps": 18.0},
            },
            "cav_controller_block": {},
        }

    samples = [StateSample(hybrid=mk(), arrivals=(),
                           meta=SampleMeta("s0", 480.0, True, 0.9)),
               StateSample(hybrid=mk(0.5), arrivals=(),
                           meta=SampleMeta("s1", 480.0, True, 0.9))]
    plan2 = PlanCandidate(plan_version=1,
                          per_cav={"m1": "damping"},
                          walls=())
    preds = pred.predict_many(samples, [PLAN, plan2], CS,
                              horizon_s=120.0, commit_s=30.0)
    single = pred.predict(samples, PLAN, CS, horizon_s=120.0,
                          commit_s=30.0)
    assert float(single.delay_part_a_s[0, 0]) == \
        pytest.approx(float(preds[0].delay_part_a_s[0, 0]))
    assert np.array_equal(preds[0].native_delay_s, preds[1].native_delay_s)
    assert np.array_equal(preds[0].native_collision, preds[1].native_collision)
    again = pred.predict_many(samples, [PLAN, plan2], CS,
                              horizon_s=120.0, commit_s=30.0)
    assert np.array_equal(again[0].delay_part_a_s, preds[0].delay_part_a_s)
    assert again[0].meta.sample_ids == preds[0].meta.sample_ids
    # the damping plan actually differs from native
    assert float(preds[1].delay_part_a_s[0, 0]) != \
        pytest.approx(float(preds[1].native_delay_s[0, 0]))


# -------------------------------------------- shared validator (v2.1)

def _base_pred():
    pred = PartAV2GeneralPredictor()
    return pred.predict_many([_sample(0)], [PLAN], CS, horizon_s=120.0,
                             commit_s=30.0)[0]


def test_v2_1_optional_fields_absent_is_valid():
    """Every v2.1 field is OPTIONAL: a Prediction stripped of them (the
    perfect_sim shape) stays valid."""
    p = _base_pred()
    stripped = dataclasses.replace(
        p, n_tracked=None, n_exited_tracked=None, n_free_road_assumed=None,
        untracked_arrivals_expected=None,
        meta=dataclasses.replace(p.meta, part_a_model=None,
                                 param_set_id=None,
                                 recon_schema_version=None,
                                 part_a_scope=None))
    validate_prediction(stripped, 1, 1, 1, 30.0)


def test_v2_1_bad_scope_and_negative_counts_rejected():
    p = _base_pred()
    bad_scope = dataclasses.replace(
        p, meta=dataclasses.replace(p.meta, part_a_scope="everyone"))
    with pytest.raises(PredictionViolation):
        validate_prediction(bad_scope, 1, 1, 1, 30.0)
    neg = dataclasses.replace(p, n_tracked=np.array([[-1.0]]))
    with pytest.raises(PredictionViolation):
        validate_prediction(neg, 1, 1, 1, 30.0)
    wrong_shape = dataclasses.replace(p, n_free_road_assumed=np.zeros((2, 1)))
    with pytest.raises(PredictionViolation):
        validate_prediction(wrong_shape, 1, 1, 1, 30.0)


def test_validator_accepts_m_one_on_reference_stub(monkeypatch=None):
    """The kit validator accepts M = 1 explicitly (v2.1 section 1) --
    checked directly on the reference const-speed v2 stub."""
    from predict.reference_const_speed import ConstSpeedPredictorV2
    from predict import conformance as K
    samples = K.build_state_samples(k=3)
    plans = K.build_plans(K.build_controller_state())
    stub = ConstSpeedPredictorV2()
    preds = stub.predict_many(samples, [plans[0]], K.build_controller_state(),
                              horizon_s=600.0)
    validate_prediction(preds[0], 1, 3, 1, None)   # M = 1, K = 3
