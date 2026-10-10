"""§8 (v1 clarifications) and §9 (v1.1) — one test per §8 item plus the §9
validator/kit/consumer rules. Old names are kept working elsewhere; tests here
reference their clause number explicitly."""
import dataclasses
import inspect
import math

import numpy as np
import pytest

from predict import (
    CONTRACT_VERSION,
    CONTRACT_VERSION_V1_1,
    SUPPORTED_CONTRACT_VERSIONS,
    Arrival,
    ControllerStateEntry,
    PHASE_ASSEMBLE,
    PHASE_HOLD,
    PHASE_RELEASE,
    PLAN_VERSION,
    PlanCandidate,
    PlanViolation,
    PredictionViolation,
    StateSamplesViolation,
    WallSpec,
    build_controller_state,
    build_plans,
    build_state_samples,
    native_reference,
    run_conformance,
    run_conformance_both_versions,
    validate_plan,
    validate_prediction,
    validate_state_samples,
)
from predict.conformance import ConformanceError, _pred_equal, broken_predictors
from predict.reference_const_speed import ConstSpeedPredictor

CS = build_controller_state()
IDS = tuple(sorted(CS))
SAMPLES = build_state_samples(k=3)
PLANS = build_plans(CS)
V1 = ConstSpeedPredictor().predict_many(SAMPLES, PLANS[:2], CS)[0]
V11 = ConstSpeedPredictor(contract_version=CONTRACT_VERSION_V1_1).predict_many(
    SAMPLES, PLANS[:2], CS
)[0]
M, K = V1.collision.shape


def rep(pred, **kw):
    return dataclasses.replace(pred, **kw)


def repmeta(pred, **kw):
    return rep(pred, meta=dataclasses.replace(pred.meta, **kw))


# --- §8: one test per clarification -------------------------------------------

def test_s8_1_native_kept_assemble_is_hold_inf():
    native, kept = native_reference(CS, IDS)
    w1 = next(w for w in native.walls if w.wall_id == "W1")
    assert kept == ("W1",)
    assert w1.phase == PHASE_ASSEMBLE and w1.hold_s == math.inf
    validate_plan(native, CS)
    # +inf is allowed ONLY in hold_s
    with pytest.raises(PlanViolation, match="release_at_s"):
        validate_plan(
            PlanCandidate(1, {}, (WallSpec("Wx", ("cav_n0",), 5.0, PHASE_RELEASE, math.inf, None),)),
            CS,
        )


def test_s8_2_dissolved_walls_are_release_now_keeping_members():
    native, kept = native_reference(CS, IDS)
    assert kept == ("W1",)  # meta.native_kept_walls carries keep/dissolve
    w2 = next(w for w in native.walls if w.wall_id == "W2")
    w3 = next(w for w in native.walls if w.wall_id == "W3")
    for w in (w2, w3):
        assert w.phase == PHASE_RELEASE and w.release_at_s == 0.0 and w.hold_s is None
        assert len(w.members) > 0  # members kept: locked CAVs can't enter per_cav


def test_s8_3_v1_member_order_is_cav_id_and_lane_check_skips():
    native, _ = native_reference(CS, IDS)  # fixture entries have no lane_id
    w1 = next(w for w in native.walls if w.wall_id == "W1")
    assert w1.members == ("cav_a0", "cav_a1")  # sorted(cav_id)
    # descending lane ids would raise, but with lane_id missing the check is
    # skipped (see §9 ordering tests for the lane-present cases)


def test_s8_4_release_offset_and_hold_from_assembly_xor_in_every_phase():
    # start from the native plan so locked CAVs keep their existing walls;
    # add one new wall over the unlocked cav_n0 with each time-reference form
    native_plan, _ = native_reference(CS, IDS)
    others = {c: a for c, a in native_plan.per_cav.items() if c != "cav_n0"}
    for phase in (PHASE_ASSEMBLE, PHASE_HOLD, PHASE_RELEASE):
        ok_release = WallSpec("Wx", ("cav_n0",), 5.0, phase, 0.0, None)
        ok_hold = WallSpec("Wy", ("cav_n0",), 5.0, phase, None, 0.0)
        validate_plan(PlanCandidate(1, dict(others), native_plan.walls + (ok_release,)), CS)
        validate_plan(PlanCandidate(1, dict(others), native_plan.walls + (ok_hold,)), CS)
    with pytest.raises(PlanViolation, match="relative to t0"):
        validate_plan(
            PlanCandidate(1, dict(others), native_plan.walls + (
                WallSpec("Wx", ("cav_n0",), 5.0, PHASE_RELEASE, -1.0, None),)),
            CS,
        )


def test_s8_5_wall_speed_finite_positive():
    with pytest.raises(PlanViolation, match="finite > 0"):
        validate_plan(
            PlanCandidate(1, {c: "native" for c in IDS if c != "cav_n0"},
                          (WallSpec("Wx", ("cav_n0",), 0.0, PHASE_ASSEMBLE, None, 5.0),)),
            CS,
        )


def test_s8_6_trajectory_nan_is_trailing_whole_row_suffix():
    traj = dict(V1.trajectories_tracked)
    arr = np.array(traj["cav_a0"], copy=True)
    arr[0, 0, 5, :] = np.nan  # hole in the middle
    traj["cav_a0"] = arr
    with pytest.raises(PredictionViolation, match="suffix"):
        validate_prediction(rep(V1, trajectories_tracked=traj), M, K, 2)
    arr2 = np.array(V1.trajectories_tracked["cav_a0"], copy=True)
    arr2[0, 0, 40, 3] = np.nan  # single column
    traj["cav_a0"] = arr2
    with pytest.raises(PredictionViolation, match="whole"):
        validate_prediction(rep(V1, trajectories_tracked=traj), M, K, 2)


def test_s8_7_v1_nan_only_in_min_ttc_other_floats_finite():
    nan_ttc = np.full((M, K), np.nan)
    validate_prediction(rep(V1, min_ttc_s=nan_ttc, native_min_ttc_s=nan_ttc), M, K, 2)
    bad = np.array(V1.plan_delay_s, copy=True)
    bad[0, 0] = np.nan
    with pytest.raises(PredictionViolation, match="§8.7|finite"):
        validate_prediction(rep(V1, plan_delay_s=bad), M, K, 2)


def test_s8_8_identity_tolerances_rtol_1e_6_atol_1e_9():
    scale = np.abs(V1.plan_delay_s - V1.native_delay_s)
    tol = 1e-6 * scale + 1e-9
    within = np.array(V1.delay_vs_native_s, copy=True)
    within[0, 0] += 0.5 * tol[0, 0]  # inside the §8.8 envelope
    validate_prediction(rep(V1, delay_vs_native_s=within), M, K, 2)
    beyond = np.array(V1.delay_vs_native_s, copy=True)
    beyond[0, 0] += 10.0 * tol[0, 0] + 1e-6
    with pytest.raises(PredictionViolation, match="delay_vs_native_s"):
        validate_prediction(rep(V1, delay_vs_native_s=beyond), M, K, 2)


def test_s8_9_entrance_labels_and_coverage_bounds():
    validate_state_samples(SAMPLES)  # mainline/ramp labels, coverage 0.75
    bad_entrance = rep(
        SAMPLES[0],
        arrivals=(Arrival(t_s=1.0, entrance="side", speed_mps=20.0,
                          is_cav=False, vehicle_type="car"),),
    )
    with pytest.raises(StateSamplesViolation, match="entrance"):
        validate_state_samples([bad_entrance])
    bad_cov = rep(SAMPLES[0], meta=dataclasses.replace(SAMPLES[0].meta, coverage=1.5))
    with pytest.raises(StateSamplesViolation, match="coverage"):
        validate_state_samples([bad_cov])


def test_s8_10_trajectory_beyond_part_a_may_be_rejected():
    with pytest.raises(ValueError, match="part_a_horizon_s"):
        ConstSpeedPredictor().predict_many(
            SAMPLES, PLANS[:1], CS, trajectory_s=120.0
        )


def test_s8_11_wall_clock_excluded_from_determinism():
    pred2 = ConstSpeedPredictor().predict_many(SAMPLES, PLANS[:2], CS)[0]
    assert _pred_equal(V1, pred2)
    assert _pred_equal(
        rep(V1, meta=dataclasses.replace(V1.meta, wall_clock_s=V1.meta.wall_clock_s + 4.0)),
        pred2,
    )


def test_s8_12_validate_prediction_takes_n_plans():
    params = list(inspect.signature(validate_prediction).parameters)
    assert params[:4] == ["pred", "M", "K", "n_plans"]
    with pytest.raises(PredictionViolation, match="n_plans"):
        validate_prediction(V1, M, K, 0)


def test_s8_13_action_vocabulary_native_damping_pi_external_u():
    per_cav = dict(PLANS[2].per_cav)
    per_cav["cav_n0"] = "wall"  # wall members never sit in per_cav actions
    with pytest.raises(PlanViolation, match="per_cav"):
        validate_plan(rep(PLANS[2], per_cav=per_cav), CS)


# --- §9 (v1.1) -----------------------------------------------------------------

def test_s9_controller_lane_id_domain():
    e = ControllerStateEntry("native")
    assert e.lane_id is None  # optional, backward compatible
    for bad in (3, -1, "1", True, 1.0):
        cs = dict(CS)
        cs["cav_n0"] = ControllerStateEntry("native", lane_id=bad)
        with pytest.raises(PlanViolation, match="lane_id"):
            validate_plan(PLANS[0], cs)


@pytest.mark.parametrize(
    "lanes,ok",
    [
        ((0, 1), True),   # ascending: valid
        ((1, 0), False),  # descending: rejected
        ((0, 0), False),  # equal lanes: rejected (strictly ascending)
        ((2, None), True),  # partially missing: check skipped (§9)
        ((None, 0), True),
    ],
)
def test_s9_wall_member_lane_ordering(lanes, ok):
    cs = {c: e for c, e in CS.items() if e.wall_id is None}  # no locked walls
    for cav, ln in zip(("cav_p0", "cav_p1"), lanes):
        cs[cav] = ControllerStateEntry("wall", "W9", PHASE_ASSEMBLE, 0.0, 5.0, False, lane_id=ln)
    wall = WallSpec("W9", ("cav_p0", "cav_p1"), 5.0, PHASE_ASSEMBLE, None, 5.0)
    per_cav = {c: "native" for c in cs if c not in ("cav_p0", "cav_p1")}
    plan = PlanCandidate(1, per_cav, (wall,))
    if ok:
        validate_plan(plan, cs)
    else:
        with pytest.raises(PlanViolation, match="ascending lane_id"):
            validate_plan(plan, cs)


def test_s9_native_reference_orders_members_by_lane_id():
    cs = dict(CS)
    # same wall W1; give cav_a1 lane 0 and cav_a0 lane 1 (out of id order)
    cs["cav_a0"] = ControllerStateEntry("wall", "W1", PHASE_ASSEMBLE, 12.0, 5.0, True, lane_id=1)
    cs["cav_a1"] = ControllerStateEntry("wall", "W1", PHASE_ASSEMBLE, 12.0, 5.0, True, lane_id=0)
    native, kept = native_reference(cs, IDS)
    assert kept == ("W1",)
    w1 = next(w for w in native.walls if w.wall_id == "W1")
    assert w1.members == ("cav_a1", "cav_a0")  # lane 0 first
    validate_plan(native, cs)


def test_s9_cell_valid_required_in_1_1_never_defaulted():
    for f in ("cell_valid", "native_cell_valid", "has_closing_pair", "native_has_closing_pair"):
        arr = getattr(V11, f)
        assert isinstance(arr, np.ndarray) and arr.dtype == np.bool_ and arr.shape == (M, K)
    with pytest.raises(PredictionViolation, match="never defaulted to True"):
        validate_prediction(rep(V11, cell_valid=None), M, K, 2)
    with pytest.raises(PredictionViolation, match="REQUIRED"):
        validate_prediction(rep(V11, has_closing_pair=None), M, K, 2)


def test_s9_nan_min_ttc_iff_no_closing_pair_in_valid_cell():
    # NaN min_ttc with has_closing_pair True -> rejected (§9 kit case)
    min_ttc = np.array(V11.min_ttc_s, copy=True)
    min_ttc[0, 0] = np.nan
    with pytest.raises(PredictionViolation, match="if and only if"):
        validate_prediction(rep(V11, min_ttc_s=min_ttc), M, K, 2)
    # finite min_ttc with has_closing_pair False -> also rejected (iff)
    closing = np.array(V11.has_closing_pair, copy=True)
    closing[0, 0] = False
    with pytest.raises(PredictionViolation, match="if and only if"):
        validate_prediction(rep(V11, has_closing_pair=closing), M, K, 2)
    # valid cell, no closing pair, NaN min_ttc -> correct and accepted
    ok = rep(V11, min_ttc_s=np.full((M, K), np.nan), has_closing_pair=np.zeros((M, K), dtype=bool))
    validate_prediction(ok, M, K, 2)


def test_s9_invalid_cell_may_carry_nan_anywhere():
    cell_valid = np.array(V11.cell_valid, copy=True)
    cell_valid[0, 1] = False
    delay = np.array(V11.plan_delay_s, copy=True)
    delay[0, 1] = np.nan
    ttc = np.array(V11.min_ttc_s, copy=True)
    ttc[0, 1] = np.nan
    closing = np.array(V11.has_closing_pair, copy=True)
    closing[0, 1] = True  # NaN ttc with closing pair: fine in an INVALID cell
    validate_prediction(rep(V11, cell_valid=cell_valid, plan_delay_s=delay,
                            min_ttc_s=ttc, has_closing_pair=closing), M, K, 2)


def test_s9_native_cell_rules_mirror_plan_rules():
    native_valid = np.array(V11.native_cell_valid, copy=True)
    native_valid[0, 2] = False
    delay = np.array(V11.native_delay_s, copy=True)
    delay[0, 2] = np.nan
    validate_prediction(rep(V11, native_cell_valid=native_valid, native_delay_s=delay), M, K, 2)
    delay[0, 0] = np.nan  # NaN in a VALID native cell
    with pytest.raises(PredictionViolation, match="native_delay_s"):
        validate_prediction(rep(V11, native_cell_valid=native_valid, native_delay_s=delay), M, K, 2)


def test_s9_other_floats_finite_in_valid_cells():
    bad = np.array(V11.hard_brake_s, copy=True)
    bad[0, 0] = np.inf
    with pytest.raises(PredictionViolation, match="hard_brake_s"):
        validate_prediction(rep(V11, hard_brake_s=bad), M, K, 2)


def test_s9_v1_output_must_not_carry_cell_fields():
    with pytest.raises(PredictionViolation, match="belong to 1.1"):
        validate_prediction(rep(V1, cell_valid=np.ones((M, K), dtype=bool)), M, K, 2)


def test_s9_versions_accepted_and_reported():
    # v2 ('wd01_predict/2') joined the supported set; the v1 family stays
    # frozen (PREDICT_CONTRACT_v2.md preamble: v1.1 remains as it is).
    assert SUPPORTED_CONTRACT_VERSIONS == (
        "wd01_predict/1",
        "wd01_predict/1.1",
        "wd01_predict/2",
    )
    assert V1.meta.contract_version == CONTRACT_VERSION
    assert V11.meta.contract_version == CONTRACT_VERSION_V1_1
    validate_prediction(V1, M, K, 2)
    validate_prediction(V11, M, K, 2)
    with pytest.raises(PredictionViolation, match="SUPPORTED_CONTRACT_VERSIONS|must be one of"):
        validate_prediction(repmeta(V1, contract_version="wd01_predict/9"), M, K, 2)


@pytest.mark.parametrize("name,_", broken_predictors(), ids=[n for n, _ in broken_predictors()])
def test_s9_kit_rejects_broken_predictors(name, _):
    with pytest.raises(ConformanceError):
        run_conformance(dict(broken_predictors())[name])


def test_s9_kit_exercises_both_versions():
    reports = run_conformance_both_versions(lambda v: ConstSpeedPredictor(contract_version=v))
    # the frozen v1 family; the v2 kit is run_conformance_v2 (per continuation)
    assert set(reports) == {"wd01_predict/1", "wd01_predict/1.1"}
    assert all(r.predictor_id == "reference_const_speed/1" for r in reports.values())
