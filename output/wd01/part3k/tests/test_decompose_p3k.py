"""Part 3k: faithfulness of the copied p3j attribution machinery.

decompose_p3k.py is a copy of output/wd01/part3j/decompose.py (commit
c618aac) adapted for L4; these tests re-run three of the p3j synthetic
attribution tests verbatim in spirit (same hand-built trajectories with
KNOWN delays) against THIS copy, so the p3j guarantees (exact per-cell
attribution, d-continuity across lane changes, exit-mismatch accounting)
transfer to the part 3k module.
"""
import numpy as np
import pytest

import decompose_p3k as D

LIM = 31.29
S0 = 2644.22 - 400.0            # lane 1, d = 400 (rsu_window)


def lane1_s(d):
    return 2644.22 - d


def lane2_s(d):
    return 378.66 - d


def test_seg_bins_and_limits_match_engine():
    import engine as E
    assert D.seg_bin(-0.1) == "downstream"
    assert D.seg_bin(0.0) == "merge_edge"
    assert D.seg_bin(74.9) == "merge_edge"
    assert D.seg_bin(75.0) == "rsu_window"
    assert D.seg_bin(800.0) == "upstream"
    for li in (0, 1, 2):
        for s in (0.0, 100.0, 300.0, 380.0, 440.0, 2000.0):
            assert D.limit_of(li, s) == float(E.lane_speed_limits(
                np.array([li]), np.array([s]))[0])
    assert D.limit_of(0, 100.0) == pytest.approx(31.29)
    assert D.limit_of(2, 0.0) == pytest.approx(22.22)
    assert D.limit_of(2, 440.0) == pytest.approx(31.29)


def test_known_deficit_attributed_exactly_one_cell():
    """Pred v=20 vs truth v=10, same cell: the deficit is attributed
    exactly to that one cell with the exact per-step delays."""
    K = 4
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
    assert 0.5 * (npp - ntt) == pytest.approx(0.0)
    assert fp == ft == {1: 0}
    # component identity: expo + ff = delay deficit
    ff_p, ff_t = 0.5 * npp - dp, 0.5 * ntt - dt_
    assert 0.5 * (npp - ntt) + (ff_t - ff_p) == \
        pytest.approx(dp - dt_)


def test_lane_change_d_continuity_and_lc_flag():
    """lane 1 -> 2 at step 2 with d continuous: ds/delay read from d
    (not s), and lc_affected flags steps 2..12 inclusive."""
    d = {0: 300.0, 1: 295.0, 2: 292.0}
    for j in range(3, 14):
        d[j] = d[j - 1] - 5.0
    snaps = {0: {1: (1, lane1_s(d[0]), 10.0)}}
    for j in range(1, 14):
        lane = 1 if j == 1 else 2
        snaps[j] = {1: (lane, lane2_s(d[j]) if lane == 2
                        else lane1_s(d[j]), 10.0)}
    recs, _pres, _first = D.pred_records(snaps, {}, K=13)
    by_j = {r["j"]: r for r in recs}
    assert by_j[1]["ds"] == pytest.approx(5.0)
    assert by_j[2]["ds"] == pytest.approx(3.0)      # from d, not s
    assert by_j[3]["ds"] == pytest.approx(5.0)
    assert by_j[2]["delay"] == pytest.approx(0.5 - 3.0 / LIM)
    assert by_j[3]["delay"] == pytest.approx(0.5 - 5.0 / 22.22)
    flags = {j: by_j[j]["lc"] for j in range(1, 14)}
    assert not flags[1] and all(flags[j] for j in range(2, 13))
    assert not flags[13]


def test_exit_mismatch_counts_remaining_delay_exactly():
    """Pred exits after step 2 while truth runs all 4 steps: cell deficit
    = -1.0 veh s and the pred_early bucket carries the exact remaining
    truth delay."""
    K = 4
    snaps = {0: {1: (1, S0, 20.0)}, 1: {1: (1, S0 + 10.0, 20.0)},
             2: {1: (1, S0 + 20.0, 20.0)}}
    rows = {j: {1: (1, S0 + 5.0 * j, 10.0)} for j in range(K + 1)}
    recs_p, pres_p, first_p = D.pred_records(snaps, {}, K)
    recs_t, pres_t, first_t = D.truth_records(rows, {}, {}, 0, K)
    cells = D.cells_from(recs_p, recs_t)
    dp, dt_, npp, ntt = cells[(1, "rsu_window", "t0", False)]
    assert npp == 2 and ntt == 4
    assert dt_ - dp == pytest.approx(1.0)
    xm = D.exit_mismatch(first_p, pres_p, first_t, pres_t,
                         recs_p, recs_t, K)
    assert xm["pred_early"][0] == 1
    assert xm["pred_early"][1] == 2
    assert xm["pred_early"][2] == pytest.approx(2 * (0.5 - 5.0 / LIM))
    assert xm["true_early"][0] == 0


def test_phys_delay_sums_match_records():
    """phys_delay_sums (used by the ladder) equals the record-based sums
    on the same synthetic window."""
    K = 6
    snaps = {j: {1: (1, S0 + 7.0 * j, 14.0)} for j in range(K + 1)}
    births = {2: {9: (1, lane1_s(500.0))}}
    for j in range(2, K + 1):
        snaps[j][9] = (1, lane1_s(500.0) + 6.0 * (j - 1), 12.0)
    rows = {j: dict(snaps[j]) for j in range(K + 1)}
    recs_p, _pp, _fp = D.pred_records(snaps, births, K)
    recs_t, _pt, _ft = D.truth_records(
        rows, {}, {9: (1, lane1_s(500.0))}, 0, K)
    dp_rec = sum(r["delay"] for r in recs_p)
    dt_rec = sum(r["delay"] for r in recs_t)
    dp_f, dt_f, np_f, nt_f = D.phys_delay_sums(
        snaps, births, rows, {}, {9: (1, lane1_s(500.0))}, 0, K)
    assert dp_f == pytest.approx(dp_rec)
    assert dt_f == pytest.approx(dt_rec)
    assert np_f == len(recs_p) and nt_f == len(recs_t)


def test_window_truth_stats_merge_d_uses_new_lane_offset():
    """The truth merge d of a 2->1 lane change is OFFSETS[1] - lc_s
    (lc_s is recorded on the NEW lane's coordinate)."""
    class FakeRA:
        def __init__(self):
            self.lane_m = np.array([1, 2, 2, 1])
            self.s = np.array([2594.22, 40.0, 240.0, 2694.22])
            self.v = np.array([30.0, 10.0, 20.0, 10.0])
            self.lc_tick = np.array([5, 6, 999])
            self.lc_from = np.array([2, 0, 2])
            self.lc_to = np.array([1, 1, 1])
            # lc_s on the NEW (main_right) coordinate: d = 2644.22 - s
            self.lc_s = np.array([2644.22 - 50.0, 2644.22 - 60.0, 0.0])

        def tick_rows(self, k):
            return np.arange(4) if 0 <= k < 10 else \
                np.array([], dtype=np.int64)

    st = D.window_truth_stats(FakeRA(), 0, K=10)
    assert st["merges"] == pytest.approx([50.0])   # 2->1 only
    assert st["occ"] == 0                          # no vehicle in-zone
    # lane-1 rows 0 and 3 sit at d = +50 and -50: both in [-100, 100]
    assert st["mr"][1] == 18
    assert st["mr"][0] == pytest.approx(9 * (30.0 + 10.0))
