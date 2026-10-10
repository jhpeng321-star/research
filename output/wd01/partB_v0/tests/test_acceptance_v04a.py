"""Unit tests of the WD-01-PB-V04A acceptance metric code on synthetic
data (registration section 6)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from partB_v0 import acceptance_v04a as AC


# ----------------------------------------------------------- CRPS core ---

def test_crps_samples_matches_bruteforce_and_point_mass():
    rng = np.random.default_rng(0)
    S = rng.normal(0, 2, (5, 30))
    y = rng.normal(0, 2, 5)
    got = AC.crps_samples(S, y)
    for i in range(5):
        t1 = np.abs(S[i] - y[i]).mean()
        t2 = np.mean([abs(a - b) for a, b in itertools.product(S[i], S[i])])
        assert got[i] == pytest.approx(t1 - 0.5 * t2, abs=1e-9)
    # a point mass at c: CRPS = |c - y|
    pm = np.full((3, 7), 4.0)
    assert AC.crps_samples(pm, np.array([1.0, 4.0, 9.0])).tolist() == \
        pytest.approx([3.0, 0.0, 5.0])


def test_quantile_samples_interpolate_with_flat_tails():
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    preds = {q: np.array([10.0 * q]) for q in qs}
    u = AC.level_grid(10)
    s = AC.quantile_samples(qs, preds, u)[0]
    assert s[0] == pytest.approx(0.5)          # u=0.05 -> q05 (flat tail)
    assert s[-1] == pytest.approx(9.5)         # u=0.95 -> q95
    assert (np.diff(s) >= 0).all()
    # unsorted (non-monotone) inputs are made monotone
    bad = {0.1: np.array([5.0]), 0.5: np.array([1.0]), 0.9: np.array([9.0])}
    sb = AC.quantile_samples([0.1, 0.5, 0.9], bad, AC.level_grid(9))[0]
    assert (np.diff(sb) >= 0).all()


def test_independent_vs_comonotone_pairing():
    qs = [0.1, 0.5, 0.9]
    a = {0.1: np.array([0.0]), 0.5: np.array([10.0]), 0.9: np.array([20.0])}
    n = {0.1: np.array([0.0]), 0.5: np.array([10.0]), 0.9: np.array([20.0])}
    u = AC.level_grid(20)
    sa, sn = AC.quantile_samples(qs, a, u), AC.quantile_samples(qs, n, u)
    ind = AC.delta_samples(sa, sn, "independent")
    com = AC.delta_samples(sa, sn, "comonotone")
    assert ind.shape == (1, 400) and com.shape == (1, 20)
    # identical arms: the comonotone dV is a point mass at 0 while the
    # independent pairing keeps the spread of both outcomes
    assert np.allclose(com, 0.0)
    assert ind.std() > 3.0
    assert AC.delta_samples(sa, sn, "independent").mean() == \
        pytest.approx(0.0, abs=1e-9)
    # a mean shift is recovered by both pairings
    sa2 = sa + 7.0
    assert AC.delta_samples(sa2, sn, "independent").mean() == \
        pytest.approx(7.0)
    assert AC.delta_samples(sa2, sn, "comonotone").mean() == \
        pytest.approx(7.0)
    with pytest.raises(ValueError):
        AC.delta_samples(sa, sn, "nope")


# ------------------------------------------------------------ bootstrap ---

def test_world_bootstrap_is_grouped_reproducible_and_bounded():
    worlds = np.repeat([f"w{i}" for i in range(10)], 3)
    correct = np.ones(30)
    r1 = AC.world_bootstrap_ci(correct, worlds, n_boot=200, seed=1)
    r2 = AC.world_bootstrap_ci(correct, worlds, n_boot=200, seed=1)
    assert r1 == r2                                   # reproducible
    assert r1["acc"] == 1.0 and r1["lo"] == 1.0 and r1["n_worlds"] == 10
    # a wrong world drags the lower bound below 1 but inside [0, 1]
    c2 = correct.copy()
    c2[:3] = 0.0                                     # one whole world wrong
    r3 = AC.world_bootstrap_ci(c2, worlds, n_boot=500, seed=2)
    assert 0.0 <= r3["lo"] < r3["acc"] < 1.0 + 1e-12
    assert r3["acc"] == pytest.approx(27 / 30)
    # resampling is by WORLD: correlated pairs widen the interval vs
    # treating the 30 pairs as independent
    rng = np.random.default_rng(5)
    ind_worlds = np.arange(30).astype(str)
    cw = np.repeat(rng.integers(0, 2, 10), 3).astype(float)
    wide = AC.world_bootstrap_ci(cw, worlds, n_boot=800, seed=3)
    narrow = AC.world_bootstrap_ci(cw, ind_worlds, n_boot=800, seed=3)
    assert (wide["hi"] - wide["lo"]) > (narrow["hi"] - narrow["lo"])
    assert np.isnan(AC.world_bootstrap_ci([], [])["acc"])


# ---------------------------------------------------------------- pairs ---

def _rows():
    """t0 label rows for 3 worlds (2 TRAIN, 1 CAL) and the sources of
    every contrast, plus auxiliary and off-t0 distractors."""
    recs = []
    idx = 0
    for chunk, world, split, dem, pen in (
            ("chunk4", "wA", "TRAIN", 2600, 0.02),
            ("chunk4", "wB", "TRAIN", 2700, 0.05),
            ("chunk4", "wC", "CALIBRATION", 2600, 0.02)):
        for src, y in (("native", 100.0), ("base", 90.0), ("pi", 110.0),
                       ("fixed1", 95.0), ("fixed2", 99.0),
                       ("switching1", 101.0)):
            for tau in (60.0, 120.0):
                recs.append(dict(
                    chunk_id=chunk, world=world, split=split, source=src,
                    tau=tau, t=240.0, t0_s=240.0,
                    total_delay_ref_s=y * tau / 60, demand_veh_h=dem,
                    penetration=pen, aux=False))
        # distractors: off-t0 rows, an aux native, a HOLDOUT-like row
        recs.append(dict(chunk_id=chunk, world=world, split=split,
                         source="native", tau=60.0, t=250.0, t0_s=240.0,
                         total_delay_ref_s=1e9, demand_veh_h=dem,
                         penetration=pen, aux=False))
        recs.append(dict(chunk_id=chunk, world=world, split=split,
                         source="native", tau=60.0, t=240.0, t0_s=240.0,
                         total_delay_ref_s=-1e9, demand_veh_h=dem,
                         penetration=pen, aux=True))
    return pd.DataFrame(recs)


def test_build_pairs_uses_t0_native_reference_and_never_aux():
    rows = _rows()
    cal = AC.build_pairs(rows, ("CALIBRATION",))
    assert set(cal["split"]) == {"CALIBRATION"}
    assert set(cal["contrast"]) == set(AC.REPORT_CONTRASTS)
    dn = cal[(cal["contrast"] == "damping-native") & (cal["tau"] == 60.0)]
    assert len(dn) == 1 and dn["dY"].iloc[0] == pytest.approx(-10.0)
    pn = cal[(cal["contrast"] == "pi-native") & (cal["tau"] == 120.0)]
    assert pn["dY"].iloc[0] == pytest.approx(20.0)      # (110-100)*2
    assert (cal[cal["contrast"] == "fixed-native"].groupby("tau").size()
            == 2).all()
    assert (cal[cal["contrast"] == "switching-native"].groupby("tau").size()
            == 1).all()
    assert (cal["y_n"].abs() < 1e6).all()           # no distractor used
    tr = AC.build_pairs(rows, ("TRAIN",))
    assert set(tr["world"]) == {"wA", "wB"}


def test_train_references_and_cell_mean_baseline_with_fallback():
    rows = _rows()
    tr = AC.build_pairs(rows, ("TRAIN",)).reset_index(drop=True)
    ref = AC.train_references(tr)
    r = ref[("damping-native", 60.0)]
    assert r["median_abs"] == pytest.approx(10.0)
    assert r["cell_mean"][(2600, 0.02)] == pytest.approx(-10.0)
    cal = AC.build_pairs(rows, ("CALIBRATION",)).reset_index(drop=True)
    cm = AC.cell_mean_baseline(cal, ref)
    assert cm[(cal["contrast"] == "damping-native")
              & (cal["tau"] == 60.0)].tolist() == pytest.approx([-10.0])
    # unseen cell -> pooled TRAIN mean of that (contrast, tau)
    cal2 = cal.copy()
    cal2["demand"] = 9999
    assert AC.cell_mean_baseline(cal2, ref)[0] == pytest.approx(
        ref[(cal2["contrast"][0], float(cal2["tau"][0]))]["pooled_mean"])


# ----------------------------------------------------------- A3 metrics ---

def _pairs_for(dY, dv_mean, sd=1.0, contrast="damping-native", tau=60.0,
               n_world=None):
    n = len(dY)
    pairs = pd.DataFrame(dict(
        contrast=contrast, tau=tau, dY=np.asarray(dY, float),
        key=[f"c/w{i}" for i in range(n)] if n_world is None else
        [f"c/w{i % n_world}" for i in range(n)],
        demand=2600, penetration=0.02))
    u = AC.level_grid(20)
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    z = np.array([-1.64, -0.67, 0.0, 0.67, 1.64])
    preds_a = {q: np.asarray(dv_mean, float) + sd * zz
               for q, zz in zip(qs, z)}
    preds_n = {q: np.zeros(n) + 0.3 * sd * zz for q, zz in zip(qs, z)}
    sa = AC.quantile_samples(qs, preds_a, u)
    sn = AC.quantile_samples(qs, preds_n, u)
    ind = AC.delta_samples(sa, sn, "independent")
    com = AC.delta_samples(sa, sn, "comonotone")
    point = np.asarray(dv_mean, float)
    return pairs, ind, com, point


def _ref(median_abs, mean=0.0):
    return {("damping-native", 60.0): dict(
        median_abs=median_abs, cell_mean={(2600, 0.02): mean},
        pooled_mean=mean, n=10)}


def test_a3_sign_filter_uses_train_median_and_ci_lower_bound():
    rng = np.random.default_rng(0)
    dY = rng.choice([-1, 1], 40) * rng.uniform(5, 50, 40)
    small = np.abs(dY) < 10                          # near-ties
    # perfect sign predictions on the eligible pairs, WRONG on near-ties
    dv = np.where(small, -np.sign(dY) * 3.0, np.sign(dY) * 20.0)
    pairs, ind, com, point = _pairs_for(dY, dv, sd=2.0)
    rec = AC.contrast_metrics(pairs, ind, com, point, _ref(10.0),
                              n_boot=300)[0]
    assert rec["n_eligible"] == int((~small).sum()) and rec["n_pairs"] == 40
    assert rec["sign_acc"] == 1.0 and rec["sign_ci_lo"] > 0.5
    assert rec["pass_i"] is True
    # the same predictions with a huge threshold: nothing eligible -> no
    # evidence -> (i) cannot pass
    rec2 = AC.contrast_metrics(pairs, ind, com, point, _ref(1e9),
                               n_boot=50)[0]
    assert rec2["n_eligible"] == 0 and rec2["pass_i"] is False
    # coin-flip predictions: lower bound cannot exceed 0.5
    flip = rng.choice([-1, 1], 40) * 20.0
    p3, i3, c3, pt3 = _pairs_for(dY, flip, sd=2.0)
    rec3 = AC.contrast_metrics(p3, i3, c3, pt3, _ref(0.0), n_boot=300)[0]
    assert rec3["sign_ci_lo"] <= 0.5 + 1e-9 or rec3["sign_acc"] < 0.75
    assert rec3["pass_i"] is False or rec3["sign_ci_lo"] > 0.5


def test_a3_crps_must_beat_zero_and_train_cell_mean_baselines():
    dY = np.array([-30.0, -28.0, -33.0, -31.0, -29.0, -32.0])
    good, ind, com, pt = _pairs_for(dY, dY, sd=1.0)       # sharp and right
    rec = AC.contrast_metrics(good, ind, com, pt, _ref(0.0, mean=0.0),
                              n_boot=50)[0]
    assert rec["crps_dv"] < rec["crps_zero"]
    assert rec["crps_dv"] < rec["crps_cell_mean"] or rec["pass_ii"] is False
    # cell-mean baseline that is already exact -> the model must beat it
    exact = AC.contrast_metrics(good, ind, com, pt,
                                _ref(0.0, mean=float(dY.mean())),
                                n_boot=50)[0]
    assert exact["crps_cell_mean"] == pytest.approx(
        np.abs(dY - dY.mean()).mean())
    assert exact["pass_ii"] is (exact["crps_dv"] < exact["crps_cell_mean"])
    # a model predicting the WRONG sign with the same spread loses to the
    # zero baseline
    bad, ind2, com2, pt2 = _pairs_for(dY, -dY, sd=1.0)
    rb = AC.contrast_metrics(bad, ind2, com2, pt2, _ref(0.0), n_boot=50)[0]
    assert rb["crps_dv"] > rb["crps_zero"] and rb["pass_ii"] is False
    assert rb["pass_tau"] is False


def test_a3_verdict_needs_two_of_three_taus_per_contrast():
    def rec(c, tau, ok):
        return dict(contrast=c, tau=tau, pass_i=ok, pass_ii=ok,
                    pass_tau=ok)
    recs = [rec("damping-native", 60.0, True),
            rec("damping-native", 120.0, True),
            rec("damping-native", 300.0, False),
            rec("pi-native", 60.0, True),
            rec("pi-native", 120.0, False),
            rec("pi-native", 300.0, False),
            rec("fixed-native", 60.0, True)]
    a3 = AC.a3_verdict(recs)
    assert a3["damping-native"]["passes"] is True
    assert a3["pi-native"]["passes"] is False
    assert set(a3) == set(AC.GATING_CONTRASTS)   # fixed/switching not gating


# ------------------------------------------------------------- A1 / A2 ---

def _a1_rows(model_lt_clim=True):
    return [dict(tau=t, crps_model=(90.0 if model_lt_clim else 130.0),
                 crps_climatology=100.0) for t in AC.TAUS_A2_GATING
            + (300.0, 570.0)]


def test_a1_requires_every_tau_and_coverage_band():
    assert AC.a1_verdict(_a1_rows(), 0.80)["passes"] is True
    assert AC.a1_verdict(_a1_rows(), 0.75)["passes"] is True      # edges in
    assert AC.a1_verdict(_a1_rows(), 0.85)["passes"] is True
    assert AC.a1_verdict(_a1_rows(), 0.7499)["passes"] is False
    assert AC.a1_verdict(_a1_rows(), 0.8501)["passes"] is False
    rows = _a1_rows()
    rows[-1]["crps_model"] = 100.0                # tie with climatology
    assert AC.a1_verdict(rows, 0.80)["passes"] is False
    assert AC.a1_verdict(_a1_rows(False), 0.80)["passes"] is False
    assert AC.a1_verdict([], 0.80)["passes"] is False


def test_a2_gating_taus_and_ties():
    def acc(res):
        return [dict(tau=t, result=res.get(t, "PASS"))
                for t in (30.0, 60.0, 120.0, 300.0, 570.0)]
    assert AC.a2_verdict(acc({}))["gating_pass"] is True
    assert AC.a2_verdict(acc({300.0: "FAIL", 570.0: "FAIL"}))[
        "gating_pass"] is True                      # long tau: reported
    assert AC.a2_verdict(acc({120.0: "TIE"}))["gating_pass"] is False
    v = AC.a2_verdict(acc({60.0: "FAIL"}))
    assert v["gating_pass"] is False and v["gating"]["60"] == "FAIL"
    assert set(v["nongating"]) == {"300", "570"}
    # a missing gating tau cannot pass
    assert AC.a2_verdict([dict(tau=30.0, result="PASS")])[
        "gating_pass"] is False


# --------------------------------------------------------------- verdict --

def _a(passes):
    return dict(passes=passes)


@pytest.mark.parametrize("a1,a2,d,p,want", [
    (True, True, True, True, "ACCEPT"),
    (True, True, True, False, "PARTIAL"),
    (True, True, False, True, "PARTIAL"),
    (True, True, False, False, "FAIL"),
    (False, True, True, True, "FAIL"),
    (True, False, True, True, "FAIL"),
    (False, False, False, False, "FAIL")])
def test_overall_verdict_table(a1, a2, d, p, want):
    v = AC.overall_verdict(_a(a1), dict(gating_pass=a2), {
        "damping-native": _a(d), "pi-native": _a(p)})
    assert v["verdict"] == want
    assert v["a3_contrasts_passing"] == int(d) + int(p)


def test_contrast_constants_are_the_registered_ones():
    assert AC.TAUS_A2_GATING == (30.0, 60.0, 120.0)
    assert AC.TAUS_A3 == (60.0, 120.0, 300.0)
    assert AC.GATING_CONTRASTS == ("damping-native", "pi-native")
    assert AC.COVERAGE_BAND == (0.75, 0.85)
    assert AC.BOOT_CI == 0.90
    assert set(AC.REPORT_CONTRASTS) == {"damping-native", "pi-native",
                                        "switching-native", "fixed-native"}
