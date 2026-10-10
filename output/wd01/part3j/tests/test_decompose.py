"""Part 3j synthetic attribution tests: no engine rollouts, no data, no
SUMO -- trajectories are synthesized by hand with KNOWN per-step delays
and a KNOWN deficit in one cell, which the decompose.py accounting must
attribute exactly."""
import pytest

import decompose as D

L1 = 1
S0 = 2644.22 - 400.0            # lane 1, d = 400 (rsu_window)
LIM = 31.29


def seq_to_snaps(vid_states):
    """{j: {vid: (lane, s, v)}} passthrough helper for readability."""
    return vid_states


def lane1_s(d):
    return 2644.22 - d


def lane2_s(d):
    return 378.66 - d


# ------------------------------------------------------------- geometry ---

def test_seg_bin_boundaries():
    assert D.seg_bin(-0.1) == "downstream"
    assert D.seg_bin(0.0) == "merge_edge"
    assert D.seg_bin(74.9) == "merge_edge"
    assert D.seg_bin(75.0) == "rsu_window"
    assert D.seg_bin(799.9) == "rsu_window"
    assert D.seg_bin(800.0) == "upstream"


def test_limit_of_matches_engine_and_contract():
    import numpy as np
    import engine as E
    for li in (0, 1, 2):
        for s in (0.0, 100.0, 300.0, 380.0, 440.0, 2000.0):
            assert D.limit_of(li, s) == float(E.lane_speed_limits(
                np.array([li]), np.array([s]))[0])
    assert D.limit_of(0, 100.0) == pytest.approx(31.29)   # main
    assert D.limit_of(2, 0.0) == pytest.approx(22.22)     # accel ramp
    assert D.limit_of(2, 440.0) == pytest.approx(31.29)   # merge


def test_step_delay_sign_convention():
    # stationary: full dt of delay; moving downstream (d decreases):
    # ds > 0 and delay < dt
    d0, d1 = 400.0, 390.0
    delay, ds = D.step_delay(1, lane1_s(d0), 1, lane1_s(d1))
    assert ds == pytest.approx(10.0)
    assert delay == pytest.approx(0.5 - 10.0 / LIM)
    delay0, ds0 = D.step_delay(1, lane1_s(d0), 1, lane1_s(d0))
    assert ds0 == 0.0 and delay0 == pytest.approx(0.5)
    # d INCREASING (wrong-way/jitter) yields negative ds, recorded as-is
    _d, dsneg = D.step_delay(1, lane1_s(400.0), 1, lane1_s(401.0))
    assert dsneg == pytest.approx(-1.0)


def test_lc_hit_window_inclusive():
    assert D.lc_hit([2], 2) is True          # the change step itself
    assert D.lc_hit([2], 12) is True         # 10 steps = 5.0 s later
    assert D.lc_hit([2], 13) is False        # 11 steps = 5.5 s later
    assert D.lc_hit((), 5) is False


# ----------------------------------------------------------- pred side ----

def test_pred_records_constant_speed_cell_exact():
    snaps = {j: {1: (1, S0 + 5.0 * j, 10.0)} for j in range(5)}
    recs, pres, first = D.pred_records(snaps, {}, K=4)
    per = 0.5 - 5.0 / LIM
    assert len(recs) == 4
    for r in recs:
        assert r["vid"] == 1 and r["lane"] == 1
        assert r["seg"] == "rsu_window"
        assert r["cls"] == "t0" and r["lc"] is False
        assert r["ds"] == pytest.approx(5.0)
        assert r["delay"] == pytest.approx(per)
    assert pres == {1: [1, 2, 3, 4]} and first == {1: 0}
    cells = D.cells_from(recs, [])
    assert list(cells) == [(1, "rsu_window", "t0", False)]
    c = cells[(1, "rsu_window", "t0", False)]
    assert c[0] == pytest.approx(4 * per) and c[2] == 4 and c[1] == 0.0
    # exposure identity (pred-only accounting here: n_true = 0)
    assert 0.5 * (c[2] - c[3]) == pytest.approx(2.0)


def test_pred_records_lane_change_d_continuity_and_lc_flags():
    # lane 1 -> lane 2 AT step 2, d continuous (engine keeps d constant)
    d = {0: 300.0}
    d[1] = 295.0                                   # ds 5, lane 1
    d[2] = 292.0                                   # ds 3, change to lane 2
    for j in range(3, 14):
        d[j] = d[j - 1] - 5.0                      # ds 5 on lane 2
    snaps = {0: {1: (1, lane1_s(d[0]), 10.0)}}
    for j in range(1, 14):
        lane = 1 if j == 1 else 2
        snaps[j] = {1: (lane, lane2_s(d[j]) if lane == 2 else lane1_s(d[j]),
                        10.0)}
    recs, _pres, _first = D.pred_records(snaps, {}, K=13)
    by_j = {r["j"]: r for r in recs}
    assert by_j[1]["ds"] == pytest.approx(5.0)
    assert by_j[2]["ds"] == pytest.approx(3.0)     # from d, not from s
    assert by_j[3]["ds"] == pytest.approx(5.0)
    assert by_j[1]["lane"] == 1 and by_j[2]["lane"] == 2
    assert by_j[1]["delay"] == pytest.approx(0.5 - 5.0 / LIM)
    assert by_j[2]["delay"] == pytest.approx(0.5 - 3.0 / LIM)
    assert by_j[3]["delay"] == pytest.approx(0.5 - 5.0 / 22.22)  # accel lim
    flags = {j: by_j[j]["lc"] for j in range(1, 14)}
    assert flags[1] is False
    assert all(flags[j] is True for j in range(2, 13))
    assert flags[13] is False
    cells = D.cells_from(recs, [])
    assert set(cells) == {(1, "rsu_window", "t0", False),
                          (2, "rsu_window", "t0", True),
                          (2, "rsu_window", "t0", False)}
    assert cells[(1, "rsu_window", "t0", False)][2] == 1
    assert cells[(2, "rsu_window", "t0", True)][2] == 11
    assert cells[(2, "rsu_window", "t0", False)][2] == 1


def test_pred_birth_step_uses_insertion_state():
    s_ins = lane1_s(500.0)
    snaps = {0: {1: (1, S0, 10.0)}, 1: {1: (1, S0 + 5.0, 10.0)},
             2: {1: (1, S0 + 10.0, 10.0), 7: (1, s_ins + 5.0, 10.0)},
             3: {1: (1, S0 + 15.0, 10.0), 7: (1, s_ins + 10.0, 10.0)}}
    births = {2: {7: (1, s_ins)}}
    recs, pres, first = D.pred_records(snaps, births, K=3)
    born = [r for r in recs if r["vid"] == 7]
    assert len(born) == 2 and all(r["cls"] == "born" for r in born)
    assert born[0]["ds"] == pytest.approx(5.0)
    assert born[0]["delay"] == pytest.approx(0.5 - 5.0 / LIM)
    assert pres[7] == [2, 3] and first[7] == 2
    cells = D.cells_from(recs, [])
    assert (1, "rsu_window", "born", False) in cells
    assert cells[(1, "rsu_window", "born", False)][2] == 2
    assert cells[(1, "rsu_window", "t0", False)][2] == 3


# ---------------------------------------------------------- truth side ----

def test_truth_records_mirror_and_lc_ticks():
    d = {0: 300.0, 1: 295.0, 2: 292.0, 3: 287.0, 4: 282.0}
    k0 = 120
    rows = {0: {1: (1, lane1_s(d[0]), 10.0)}}
    for j in range(1, 5):
        lane = 1 if j == 1 else 2
        rows[j] = {1: (lane, lane2_s(d[j]) if lane == 2 else lane1_s(d[j]),
                       10.0)}
    lc = {1: [k0 + 2]}                             # first tick on new lane
    recs, pres, first = D.truth_records(rows, lc, {}, k0, K=4)
    by_j = {r["j"]: r for r in recs}
    assert by_j[2]["ds"] == pytest.approx(3.0)
    assert by_j[2]["delay"] == pytest.approx(0.5 - 3.0 / LIM)
    assert by_j[3]["delay"] == pytest.approx(0.5 - 5.0 / 22.22)
    assert [by_j[j]["lc"] for j in range(1, 5)] == [False, True, True, True]
    assert all(r["cls"] == "t0" for r in recs)
    assert pres == {1: [1, 2, 3, 4]} and first == {1: 0}


def test_truth_birth_step_insertion_and_fallback():
    s_ins = lane1_s(500.0)
    rows = {0: {1: (1, S0, 10.0)},
            2: {1: (1, S0 + 5.0, 10.0), 7: (1, s_ins + 3.0, 6.0)},
            3: {1: (1, S0 + 10.0, 10.0), 7: (1, s_ins + 9.0, 12.0)}}
    k0 = 120
    # with the logged insertion state: ds = 500 -> 497 = 3 m
    recs, pres, first = D.truth_records(
        rows, {}, {7: (1, s_ins)}, k0, K=3)
    b = [r for r in recs if r["vid"] == 7 and r["j"] == 2][0]
    assert b["cls"] == "born" and b["ds"] == pytest.approx(3.0)
    assert b["delay"] == pytest.approx(0.5 - 3.0 / LIM)
    assert first[7] == 2 and pres[7] == [2, 3]
    # fallback without a logged insertion state: ds = v*dt (integrator)
    recs2, _p2, _f2 = D.truth_records(rows, {}, {}, k0, K=3)
    b2 = [r for r in recs2 if r["vid"] == 7 and r["j"] == 2][0]
    assert b2["ds"] == pytest.approx(3.0)          # v=6 m/s * 0.5 s
    assert b2["delay"] == pytest.approx(0.5 - 3.0 / LIM)


# ---------------------------------------- acceptance: exact attribution ---

def test_known_deficit_attributed_exactly_one_cell():
    """ACCEPTANCE: a known deficit (pred faster than truth) is attributed
    exactly to the one cell it was planted in."""
    K = 4
    # truth: v = 10 (ds 5/step), pred: v = 20 (ds 10/step), same cell
    snaps = {j: {1: (1, S0 + 10.0 * j, 20.0)} for j in range(K + 1)}
    rows = {j: {1: (1, S0 + 5.0 * j, 10.0)} for j in range(K + 1)}
    recs_p, _pp, fp = D.pred_records(snaps, {}, K)
    recs_t, _pt, ft = D.truth_records(rows, {}, {}, 0, K)
    cells = D.cells_from(recs_p, recs_t)
    key = (1, "rsu_window", "t0", False)
    assert set(cells) == {key}
    dp, dt_, npp, ntt = cells[key]
    assert npp == 4 and ntt == 4
    assert dp == pytest.approx(4 * (0.5 - 10.0 / LIM))
    assert dt_ == pytest.approx(4 * (0.5 - 5.0 / LIM))
    deficit = dp - dt_
    assert deficit == pytest.approx(-20.0 / LIM)
    # nothing leaked anywhere else; exposure deficit is 0 by construction
    assert 0.5 * (npp - ntt) == pytest.approx(0.0)
    assert fp == ft == {1: 0}
    # the same deficit read off the component identity:
    ff_p = 0.5 * npp - dp
    ff_t = 0.5 * ntt - dt_
    assert 0.5 * (npp - ntt) + (ff_t - ff_p) == pytest.approx(deficit)


def test_exit_mismatch_counts_other_side_remaining_delay_exactly():
    """Pred exits after step 2; truth stays for all 4 steps.  The cell
    deficit must equal -1.0 veh s (equal free-flow usage, two missing
    0.5 s steps) and the exit bucket must count the remaining truth
    steps and their delay exactly."""
    K = 4
    snaps = {0: {1: (1, S0, 20.0)}, 1: {1: (1, S0 + 10.0, 20.0)},
             2: {1: (1, S0 + 20.0, 20.0)}}
    rows = {j: {1: (1, S0 + 5.0 * j, 10.0)} for j in range(K + 1)}
    recs_p, pres_p, first_p = D.pred_records(snaps, {}, K)
    recs_t, pres_t, first_t = D.truth_records(rows, {}, {}, 0, K)
    cells = D.cells_from(recs_p, recs_t)
    dp, dt_, npp, ntt = cells[(1, "rsu_window", "t0", False)]
    assert npp == 2 and ntt == 4
    assert dt_ - dp == pytest.approx(1.0)          # exact cell deficit
    assert 0.5 * (npp - ntt) == pytest.approx(-1.0)
    xm = D.exit_mismatch(first_p, pres_p, first_t, pres_t,
                         recs_p, recs_t, K)
    assert xm["pred_early"][0] == 1
    assert xm["pred_early"][1] == 2
    assert xm["pred_early"][2] == pytest.approx(2 * (0.5 - 5.0 / LIM))
    assert xm["true_early"][0] == 0
    wait, n_m, only_p, only_t, t0m = D.entry_check({}, first_p, first_t)
    assert (wait, n_m, only_p, only_t, t0m) == (0, 0, 0, 0, 0)


def test_exit_mismatch_same_entry_exit_is_silent():
    # both sides born at step 1, both exit after step 2: no bucket
    snaps = {1: {9: (1, S0, 10.0)}, 2: {9: (1, S0 + 5.0, 10.0)}}
    rows = {1: {9: (1, S0, 10.0)}, 2: {9: (1, S0 + 5.0, 10.0)}}
    recs_p, pres_p, first_p = D.pred_records(snaps, {1: {9: (1, S0)}}, 2)
    recs_t, pres_t, first_t = D.truth_records(rows, {}, {9: (1, S0)}, 0, 2)
    xm = D.exit_mismatch(first_p, pres_p, first_t, pres_t,
                         recs_p, recs_t, 2)
    assert xm["pred_early"][0] == 0 and xm["true_early"][0] == 0


def test_entry_waiting_and_t0_missing():
    first_t = {1: 0, 7: 2}                          # 1 at t0, 7 born at 2
    births = {7: 2}                                 # flat {vid: insertion j}
    first_p = {1: 0, 7: 2}
    assert D.entry_check(births, first_p, first_t) == (0, 1, 0, 0, 0)
    # a birth inserted one step late (at 3, truth first at 2): 1 waiting
    births_late = {7: 3}
    assert D.entry_check(births_late, first_p, first_t)[0] == 1
    # a t0 vehicle missing from pred entirely is counted
    assert D.entry_check({}, {}, first_t)[4] == 1


def test_speed_profile_matched_pairs_truth_bin():
    seq_p = {1: {1: (1, S0, 20.0)}, 2: {1: (1, S0, 20.0)}}
    seq_t = {1: {1: (1, S0, 10.0)}}                 # vid 2 unmatched
    prof = D.speed_profile_from(seq_p, seq_t)
    assert set(prof) == {(1, 400)}                  # d=400 -> bin 400
    c = prof[(1, 400)]                              # only vid 1 is matched
    assert c[0] == pytest.approx(20.0) and c[1] == 1
    assert c[2] == pytest.approx(10.0) and c[3] == 1


def test_aggregate_profiles_keeps_variants_separate():
    # profile keys are (lane, d_bin); the variant lives on the window
    w0 = dict(variant="L0", profile={(1, 400): [20.0, 1, 10.0, 1]})
    w3 = dict(variant="L3", profile={(1, 400): [15.0, 1, 10.0, 1],
                                     (2, -50): [5.0, 2, 8.0, 2]})
    agg = D.aggregate_profiles([w0, w3, w0])
    assert agg["L0"][(1, 400)] == [pytest.approx(40.0), 2,
                                   pytest.approx(20.0), 2]
    assert agg["L3"][(1, 400)][1] == 1
    assert agg["L3"][(2, -50)] == [pytest.approx(5.0), 2,
                                   pytest.approx(8.0), 2]
    assert (1, 400) not in {} and len(agg) == 2


def test_sum_exit_buckets_adds_elementwise():
    # regression: sum(list_of_lists, start=[0,0,0.0]) concatenates and
    # always read 0 at index 0 -- must be elementwise per bucket
    w0 = dict(variant="L0", exit={"pred_early": [3, 10, 5.0],
                                  "true_early": [1, 2, 0.5]})
    w1 = dict(variant="L0", exit={"pred_early": [2, 6, 3.0],
                                  "true_early": [0, 0, 0.0]})
    w3 = dict(variant="L3", exit={"pred_early": [9, 9, 9.0],
                                  "true_early": [9, 9, 9.0]})
    b = D.sum_exit_buckets([w0, w1, w3], "L0")
    assert b["pred_early"] == [5, 16, pytest.approx(8.0)]
    assert b["true_early"] == [1, 2, pytest.approx(0.5)]


def test_truth_stationary_and_negative_ds_recorded():
    # measurement jitter: truth advances 0 then -1 (d increases): ds=-1
    rows = {0: {1: (1, lane1_s(400.0), 10.0)},
            1: {1: (1, lane1_s(400.0), 0.0)},
            2: {1: (1, lane1_s(401.0), 10.0)}}
    recs, pres, _f = D.truth_records(rows, {}, {}, 0, K=2)
    by_j = {r["j"]: r for r in recs}
    assert by_j[1]["ds"] == pytest.approx(0.0)
    assert by_j[2]["ds"] == pytest.approx(-1.0)
    assert by_j[2]["delay"] == pytest.approx(0.5 + 1.0 / LIM)
    assert pres[1] == [1, 2]
