"""Part 3m library tests: TTC pair extraction identical to the frozen
window_metrics definition, event attribution, segment bins, cell
accounting and the synthetic window_cells scenario (no data, no
rollouts, no weights)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine as E  # noqa: E402
import p3m_lib as M  # noqa: E402
import p3lib as P3  # noqa: E402


# ------------------------------------------------------------ ttc_pairs --

def test_ttc_pairs_leader_is_next_larger_s():
    # lane 1, s increasing forward: vid row0 is frontmost
    lane = np.array([1, 1, 1])
    s = np.array([110.0, 100.0, 92.0])   # row0 front, row2 rear
    v = np.array([20.0, 25.0, 30.0])     # both closing
    pairs = M.ttc_pairs(lane, s, v)
    assert len(pairs) == 2
    # (row1 -> row0): gap 10, closing 5 -> ttc 2.0; (row2 -> row1):
    # gap 8, closing 5 -> ttc 1.6; CONSECUTIVE pairs only
    by_f = {f: (l_, g, t) for f, l_, g, t in pairs}
    assert by_f[1] == (0, 10.0, 2.0)
    assert by_f[2] == (1, 8.0, pytest.approx(1.6))


def test_pair_exposure_matches_ttc_stats():
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(0, 12))
        lane = rng.integers(0, 3, n)
        s = np.sort(rng.uniform(0, 500, n))[::-1] if n else np.array([])
        v = rng.uniform(0, 30, n)
        lt15, lt3, _m = E.ttc_stats(lane, s, v)
        assert M.pair_exposure_s(lane, s, v) == lt3 * 0.5
        assert M.pair_exposure_s(lane, s, v, thr=1.5) == lt15 * 0.5


def test_free_and_touching_excluded():
    # gap 0 (touching) and negative closing excluded, like ttc_stats
    lane = np.array([0, 0, 0])
    s = np.array([100.0, 95.0, 95.0])
    v = np.array([10.0, 20.0, 20.0])   # row1->row0 ttc 1.0; row2 closes 0
    pairs = M.ttc_pairs(lane, s, v)
    assert len(pairs) == 1 and pairs[0][0] == 1


# --------------------------------------------------- event attribution --

def _snaps_from_rows(rows_by_j):
    """rows_by_j: j -> list of (vid, lane, s, v) -> snapshots dict."""
    return {j: dict(vid=np.array([r[0] for r in rows]),
                    lane=np.array([r[1] for r in rows]),
                    s=np.array([r[2] for r in rows]),
                    v=np.array([r[3] for r in rows]))
            for j, rows in rows_by_j.items()}


def test_pred_event_ticks_merge_vs_lc():
    # vid 10: lane 2->1 at j=3 (merge); vid 11: lane 1->0 at j=5 (lc)
    snaps = _snaps_from_rows({
        0: [(10, 2, 100.0, 20.0), (11, 1, 90.0, 20.0)],
        1: [(10, 2, 110.0, 20.0), (11, 1, 100.0, 20.0)],
        2: [(10, 2, 120.0, 20.0), (11, 1, 110.0, 20.0)],
        3: [(10, 1, 100.35, 20.0), (11, 1, 120.0, 20.0)],   # merge (2644.22-2645.67 = -1.45)
        4: [(10, 1, 110.0, 20.0), (11, 1, 130.0, 20.0)],
        5: [(10, 1, 120.0, 20.0), (11, 0, 131.45, 20.0)],   # disc lc
    })
    ev = M.pred_event_ticks(snaps, 5)
    assert ev[10] == ((3,), ())
    assert ev[11] == ((), (5,))


def test_truth_event_ticks_split_and_window():
    k0 = 100
    ra = SimpleNamespace(
        lc_vid=np.array([5, 5, 6, 7]),
        lc_tick=np.array([k0 - 5, k0 + 3, k0 + 7, k0 + 65]),
        lc_from=np.array([2, 2, 1, 1]),
        lc_to=np.array([1, 1, 0, 1]))
    m, l = M.truth_event_ticks(ra, k0, 60)
    assert m == {5: (k0 + 3,)}
    assert l == {6: (k0 + 7,)}          # tick k0+65 outside the window
    assert 7 not in m and 7 not in l


def test_pair_event_class_priority_and_who():
    tick = 100
    assert M.pair_event_class((95,), (), (98,), (), tick) == \
        (M.EV_MERGE, "both")
    assert M.pair_event_class((95,), (), (), (), tick) == \
        (M.EV_MERGE, "foll")
    assert M.pair_event_class((), (), (99,), (), tick) == \
        (M.EV_MERGE, "lead")
    # merge beats lc even when the lc is more recent
    assert M.pair_event_class((90,), (98,), (), (), tick) == \
        (M.EV_MERGE, "foll")
    assert M.pair_event_class((), (95,), (), (99,), tick) == \
        (M.EV_LC, "-")
    # 5 s window inclusive at both ends, strictly beyond excluded
    assert M.pair_event_class((90,), (), (), (), tick) == (M.EV_MERGE,
                                                           "foll")
    assert M.pair_event_class((89,), (), (), (), tick) == (M.EV_CF, "-")
    assert M.pair_event_class((), (), (), (), tick) == (M.EV_CF, "-")


# -------------------------------------------------------- bins / bands --

def test_band_idx():
    assert M.band_idx(0.5) == 0
    assert M.band_idx(1.0) == 1
    assert M.band_idx(1.9) == 2
    assert M.band_idx(2.9) == 3


def test_seg_of_d_bins():
    assert M.seg_of_d(-5.0) == "downstream"
    assert M.seg_of_d(0.0) == "merge_edge"
    assert M.seg_of_d(74.9) == "merge_edge"
    assert M.seg_of_d(75.0) == "rsu_window"
    assert M.seg_of_d(799.9) == "rsu_window"
    assert M.seg_of_d(800.0) == "upstream"


# -------------------------------------------------- window_cells synth --

class _FakeRA:
    """Minimal ra stand-in: tick_rows + per-row arrays + lc events."""

    def __init__(self, k0, rows_by_tick, lc_events):
        # rows_by_tick: abs tick -> list of (vid, lane, s, v, typ)
        self.rows = rows_by_tick
        self.lc_vid = np.array([e[0] for e in lc_events], dtype=np.int64)
        self.lc_tick = np.array([e[1] for e in lc_events], dtype=np.int64)
        self.lc_from = np.array([e[2] for e in lc_events], dtype=np.int8)
        self.lc_to = np.array([e[3] for e in lc_events], dtype=np.int8)

    def tick_rows(self, abs_tick):
        return np.arange(len(self.rows.get(abs_tick, ())))


def _ra_view(ra, abs_tick):
    rows = ra.rows.get(abs_tick, ())
    return (np.array([r[0] for r in rows], dtype=np.int64),
            np.array([r[1] for r in rows], dtype=np.int64),
            np.array([r[2] for r in rows], dtype=np.float64),
            np.array([r[3] for r in rows], dtype=np.float64),
            np.array([r[4] for r in rows], dtype=np.int64))


class _FakeRoll:
    """Minimal rollout stand-in for window_cells."""

    def __init__(self, snaps, vtype_map, k0, dawdle=1.0):
        self.snapshots = snaps
        self.vtype_map = vtype_map
        self.k0 = k0
        self.dawdle_scale = dawdle


def _patch_ra_view(monkeypatch):
    monkeypatch.setattr(M, "truth_arrays_for_tick",
                        lambda ra, t, vids: _ra_view_ra(ra, t, vids))


def _ra_view_ra(ra, abs_tick, vids):
    """truth_arrays_for_tick over _FakeRA, honouring the vid order."""
    tv = {int(v): (int(l), float(ss), float(vv), int(tp))
          for v, l, ss, vv, tp in ra.rows.get(abs_tick, ())}
    lane, s, v, typ = [], [], [], []
    for vid in vids:
        row = tv.get(int(vid))
        if row is None:
            lane.append(-1)
            s.append(np.nan)
            v.append(np.nan)
            typ.append(-1)
            continue
        lane.append(row[0])
        s.append(row[1])
        v.append(row[2])
        typ.append(row[3])
    return (np.asarray(lane), np.asarray(s, dtype=np.float64),
            np.asarray(v, dtype=np.float64), np.asarray(typ))


def test_window_cells_synthetic_behaviour_vs_composition(monkeypatch):
    _patch_ra_view(monkeypatch)
    k0 = 100
    # truth: leader vid1 (v=20), follower vid2 (v=20): gap 8, no closing
    truth = {
        k0 + 1: [(1, 1, 110.0, 20.0, 1), (2, 1, 102.0, 20.0, 0)],
        k0 + 2: [(1, 1, 120.0, 20.0, 1), (2, 1, 112.0, 20.0, 0)],
    }
    # pred: same vids, follower FASTER (ttc 8/5 = 1.6 < 3 both ticks);
    # plus vid3 PRED-ONLY ahead of vid1 closing slowly (ttc 4/5 = 0.8)
    pred = {
        0: [(1, 1, 110.0, 20.0), (2, 1, 102.0, 25.0), (3, 1, 114.0, 15.0)],
        1: [(1, 1, 110.0, 20.0), (2, 1, 102.0, 25.0), (3, 1, 114.0, 15.0)],
        2: [(1, 1, 120.0, 20.0), (2, 1, 112.0, 25.0), (3, 1, 124.0, 15.0)],
    }
    snaps = dict()
    snaps[0] = dict(vid=np.array([r[0] for r in pred[0]]),
                    lane=np.array([r[1] for r in pred[0]]),
                    s=np.array([r[2] for r in pred[0]]),
                    v=np.array([r[3] for r in pred[0]]))
    for j in (1, 2):
        rows = pred[j]
        snaps[j] = dict(vid=np.array([r[0] for r in rows]),
                        lane=np.array([r[1] for r in rows]),
                        s=np.array([r[2] for r in rows]),
                        v=np.array([r[3] for r in rows]))
    ra = _FakeRA(k0, truth, [])
    r = _FakeRoll(snaps, {1: 1, 2: 0, 3: 2}, k0)
    cells, tot = M.window_cells(r, ra, 2)
    # pa: vid2->vid1 (1.6 s) + vid1->vid3 (0.8 s) = 2 pairs x 2 ticks
    assert tot["pred_n"] == 4 and tot["pred_s"] == pytest.approx(2.0)
    # pm: matched (vid1, vid2) only -> 1 pair x 2 ticks
    assert tot["pred_matched_n"] == 2
    assert tot["pred_matched_s"] == pytest.approx(1.0)
    # tr: truth has no closing pair
    assert tot["truth_n"] == 0 and tot["truth_s"] == 0.0
    # behaviour cell (pm - tr) is on lane 1; composition (pa - pm) too
    assert cells["lane"][(1,)]["pm"] == pytest.approx(1.0)
    assert cells["lane"][(1,)]["pa"] - cells["lane"][(1,)]["pm"] == \
        pytest.approx(1.0)
    # event class: no events -> pure_cf; band [1.5,2) for the 1.6 s pair
    assert cells["ev"][(M.EV_CF,)]["pa"] == pytest.approx(2.0)
    assert cells["ttc"][("[1.5,2)",)]["pa"] == pytest.approx(1.0)
    assert cells["ttc"][("[0,1)",)]["pa"] == pytest.approx(1.0)
    # vtype of the followers: vid2 (0) twice (pm view) and vid1 (1)
    assert cells["vt"][(0,)]["pm"] == pytest.approx(1.0)
    # truth vtype label never sampled (tr zero everywhere here)
    assert all(c["tr"] == 0.0 for c in cells["vt"].values())


def test_window_cells_post_merge_event_and_parity(monkeypatch):
    _patch_ra_view(monkeypatch)
    k0 = 50
    # truth: leader vid1 / follower vid2 on lane 1, gap 40, follower
    # closes at 20 -> ttc 2.0 both ticks; vid2 MERGED at k0+1 (2->1)
    truth = {
        k0 + 1: [(1, 1, 200.0, 20.0, 1), (2, 1, 160.0, 40.0, 0)],
        k0 + 2: [(1, 1, 220.0, 20.0, 1), (2, 1, 180.0, 40.0, 0)],
    }
    pred = {
        0: [(1, 1, 200.0, 20.0), (2, 2, 161.45, 40.0)],
        1: [(1, 1, 200.0, 20.0), (2, 1, 160.0, 40.0)],   # merged (2->1)
        2: [(1, 1, 220.0, 20.0), (2, 1, 180.0, 40.0)],
    }
    snaps = {}
    for j, rows in pred.items():
        snaps[j] = dict(vid=np.array([r[0] for r in rows]),
                        lane=np.array([r[1] for r in rows]),
                        s=np.array([r[2] for r in rows]),
                        v=np.array([r[3] for r in rows]))
    ra = _FakeRA(k0, truth, [(2, k0 + 1, 2, 1)])
    r = _FakeRoll(snaps, {1: 1, 2: 0}, k0)
    cells, tot = M.window_cells(r, ra, 2)
    # j0: different lanes, no pair; j1 and j2: gap 40, closing 20,
    # ttc 2.0 -> 2 pair-ticks = 1.0 veh s on BOTH sides (parity)
    assert tot["pred_s"] == pytest.approx(tot["truth_s"]) == \
        pytest.approx(1.0)
    ev = cells["ev"][(M.EV_MERGE,)]
    assert ev["pa"] == pytest.approx(ev["tr"]) == pytest.approx(1.0)
    # `who`: the FOLLOWER is the merged vehicle
    assert cells["who"][(M.EV_MERGE, "foll")]["pa"] == \
        pytest.approx(1.0)
    # follower d = 2644.22 - 160 = 2484.22 m -> upstream
    assert cells["seg"][("upstream",)]["pa"] == pytest.approx(1.0)


def test_merge_cells_and_excess_table():
    a = {"lane": {(1,): dict(pa=4.0, pm=3.0, tr=1.0, na=8, nm=6, nt=2)}}
    b = {"lane": {(1,): dict(pa=1.0, pm=1.0, tr=1.0, na=2, nm=2, nt=2),
                  (2,): dict(pa=0.0, pm=0.0, tr=2.0, na=0, nm=0, nt=4)}}
    M.merge_cells(a, b, weight=0.5)
    lane = a["lane"][(1,)]
    assert lane["pa"] == pytest.approx(4.5)
    assert lane["pm"] == pytest.approx(3.5)
    assert lane["tr"] == pytest.approx(1.5)
    # counts are ADDITIVE (never scaled by weight)
    assert lane["na"] == 10 and lane["nm"] == 8 and lane["nt"] == 4
    rows = M.excess_table(a["lane"])
    assert rows[0][7] == (1,) and rows[0][0] == pytest.approx(3.0)
    rows_beh = M.excess_table(a["lane"], key_sort="pm-tr")
    assert rows_beh[0][7] == (1,) and rows_beh[0][0] == \
        pytest.approx(2.0)
