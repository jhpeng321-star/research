"""Part 3l fit + split tests (synthetic; no rollouts, no data writes).
The split test needs the populated data root (WD01_DATA_ROOT) and skips
if it is absent, like the p3 weight tests."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "part3k"))

import fit_p3l as F  # noqa: E402
import p3l_lib as L  # noqa: E402


def stat(occ=100, delay=500.0, lc30=10, lc60=20, mr=(500.0, 20),
         merges=(10.0,), prof=None):
    return dict(occ=occ, delay=delay, lc30=lc30, lc60=lc60,
                mr=list(mr), merges=list(merges),
                prof=prof or {})


# ------------------------------------------------------ fit losses ----

def test_losses_zero_at_exact_match():
    p = {"w": stat()}
    t = {"w": stat()}
    l1, d1 = F.loss_stage1(p, t)
    l2, d2 = F.loss_stage2(p, t)
    assert l1 == 0.0 and l2 == 0.0
    assert d1["occ"] == 0.0 and d1["merge"] == 0.0
    assert d2["lc30"] == 0.0 and d2["lc60"] == 0.0


def test_stage2_lc30_terms_and_half_weight_lc60():
    t = {"w": stat()}
    # lc30 off by 100 % (10 vs 20 -> (20-10)/max(10,10) = 1 -> term 1.0)
    p = {"w": stat(lc30=20)}
    _l2, d2 = F.loss_stage2(p, t)
    assert d2["lc30"] == pytest.approx(1.0)
    assert d2["lc60"] == 0.0
    # lc60 off by 100 % (20 vs 40 -> term (20/20)^2=1.0, weighted 0.5)
    p = {"w": stat(lc60=40)}
    l2, d2 = F.loss_stage2(p, t)
    assert d2["lc60"] == pytest.approx(1.0)
    assert l2 == pytest.approx(0.5 * 1.0, rel=1e-9)
    # delay off by 200 veh s -> (200/200)^2 = 1.0
    p = {"w": stat(delay=700.0)}
    _l2, d2 = F.loss_stage2(p, t)
    assert d2["delay"] == pytest.approx(1.0)


def test_stage1_merge_side_penalty_and_occ():
    t = {"w": stat(occ=100)}
    p = {"w": stat(occ=100, merges=[])}
    _l1, d1 = F.loss_stage1(p, t)          # one side merged, other not
    assert d1["merge"] == pytest.approx(4.0)
    p = {"w": stat(occ=150)}
    _l1, d1 = F.loss_stage1(p, t)          # occ +50 % on truth 100
    assert d1["occ"] == pytest.approx(0.25)


# ------------------------------------------------------ member draws --

def test_softmax_draw_deterministic_and_peaked():
    rng = np.random.default_rng(7)
    for _ in range(20):
        assert F.softmax_draw([0.0, 5.0], rng) == 0    # p(1) ~ 6e-15
    draws = [F.softmax_draw([0.0, 0.3], np.random.default_rng(s))
             for s in range(200)]
    assert 0 < sum(draws) < 200                        # both reachable
    assert F.softmax_draw([1.0, 1.0], np.random.default_rng(3)) in (0, 1)


def test_softmax_draw_matches_argmin_when_losses_separated():
    rng = np.random.default_rng(11)
    assert F.softmax_draw([0.0, 1.0, 2.0], rng) == 0   # T=0.15


# --------------------------------------------------- lc count helpers --

class _RA:
    lc_tick = np.array([50, 60, 61, 130, 200])


def test_lc_truth_between_boundaries():
    ra = _RA()
    assert L.lc_truth_between(ra, 0, 60) == 2      # (0, 60]
    assert L.lc_truth_between(ra, 0, 120) == 3
    assert L.lc_truth_between(ra, 60, 120) == 1    # exclusive at k0
    assert L.lc_truth_between(ra, 0, 240) == 5


def test_lc_count_between_snapshots():
    snaps = {0: {"lc": 0}, 30: {"lc": 3}, 60: {"lc": 7}, 120: {"lc": 9}}
    assert L.lc_count_between(snaps, 0, 60) == 7
    assert L.lc_count_between(snaps, 0, 120) == 9
    assert L.lc_count_between(snaps, 60, 120) == 2


def test_widen_singleton_and_monotone_coverage():
    x = np.array([5.0])
    assert L.widen_ensemble(x, 3.0)[0] == 5.0
    rng = np.random.default_rng(1)
    xs = [rng.normal(100.0, 20.0, 5) for _ in range(30)]
    ys = [float(x.mean() + rng.normal(0, 40.0)) for x in xs]
    covs = [L.coverage_frac([L.widen_ensemble(x, s) for x in xs], ys)
            for s in (1.0, 2.0, 4.0, 8.0)]
    assert covs == sorted(covs)                    # monotone non-decreasing
    assert covs[-1] >= covs[0]


# ------------------------------------------------------- split guard --

def test_splits_3l_disjoint_and_holdout_free():
    if not os.environ.get("WD01_DATA_ROOT", "").strip():
        pytest.skip("WD01_DATA_ROOT not set (data root absent)")
    sp = L.splits_3l()
    fit, cal, val, hold = (set(sp["fit"]), set(sp["calibration"]),
                           set(sp["val"]), set(sp["holdout"]))
    assert len(fit) == 4 and len(cal) == 4
    assert not (fit & cal) and not (fit & val) and not (cal & val)
    assert not (fit | cal | val) & hold
    train = set(sorted(fit | cal))                 # both are TRAIN roots
    all_train = set()
    import p3lib as P3
    sp0 = P3.roots_by_split()
    all_train = set(sp0["train"])
    assert (fit | cal) <= all_train
