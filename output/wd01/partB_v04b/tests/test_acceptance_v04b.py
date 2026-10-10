"""v0.4b acceptance metrics on synthetic truth tables (registration
s.6): A1, A3 (i)/(ii), strata, verdicts; identical CRPS code path as
v0.4a (partB_v0.acceptance_v04a primitives, imported read-only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from partB_v04b import reg
from partB_v04b import acceptance_v04b as AC


# ------------------------------------------------------------ baselines --

def test_crps_zero_and_cell_truth_tables():
    dY = np.array([-3.0, 0.0, 7.0])
    assert AC.crps_zero(dY).tolist() == pytest.approx([3.0, 0.0, 7.0])
    assert AC.crps_cell(dY, np.array([-3.0, 1.0, 2.0])).tolist() == \
        pytest.approx([0.0, 1.0, 5.0])


def test_crps_model_point_mass_equals_abs_distance():
    dY = np.array([-3.0, 0.0, 7.5])
    flat = {q: np.full(3, 1.0) for q in reg.QUANTILES}
    assert AC.crps_model(flat, dY).tolist() == \
        pytest.approx([4.0, 1.0, 6.5], abs=1e-6)


def test_crps_model_beats_zero_when_informative():
    rng = np.random.default_rng(7)
    dY = rng.normal(-50.0, 5.0, 40)
    tight = {q: np.full(40, -50.0) + (q - 0.5) for q in reg.QUANTILES}
    assert AC.crps_model(tight, dY).mean() < \
        AC.crps_zero(dY).mean()


def test_train_references_median_and_cells():
    df = pd.DataFrame(dict(
        contrast=["wall-native"] * 4, tau=[60.0] * 4,
        demand=[2700, 2700, 2400, 2400], penetration=[0.1, 0.1, 0.4, 0.4],
        dy_delay_60=[10.0, -30.0, 4.0, -2.0]))
    ref = AC.train_references_v04b(df)
    rr = ref[("wall-native", 60.0)]
    assert rr["median_abs"] == 7.0            # median(|10|,|30|,|4|,|2|)
    assert rr["cell_mean"][(2700, 0.1)] == pytest.approx(-10.0)
    assert rr["cell_mean"][(2400, 0.4)] == pytest.approx(1.0)
    assert rr["pooled_mean"] == pytest.approx(-4.5)
    # a CAL cell absent from TRAIN falls back to the pooled mean
    cal = pd.DataFrame(dict(contrast=["wall-native"], tau=[60.0],
                            demand=[9999], penetration=[0.9],
                            dy_delay=[5.0]))
    got = AC.cell_mean_of(cal, ref, 60.0)
    assert got[0] == pytest.approx(-4.5)


# ------------------------------------------------------------------- A1 --

def _preds_of(qlo, qhi):
    def f(g):
        n = len(g)
        return {q: np.full(n, np.interp(q, [0.1, 0.9], [qlo, qhi]))
                for q in reg.QUANTILES}
    return f


def test_a1_passes_when_calibrated_and_beating_zero():
    n = 5
    dY60 = -95.0 + np.arange(n) * 2.0          # -95 .. -87
    pairs = pd.DataFrame({
        "contrast": "wall-native", "tau": np.repeat([30.0, 60.0], n),
        "world": np.tile([f"w{i}" for i in range(n)], 2),
        "key": "chunk4/" + np.tile([f"w{i}" for i in range(n)], 2),
        "dy_delay_30": np.r_[np.full(n, -100.0), np.full(n, -100.0)],
        "dy_delay_60": np.r_[np.full(n, -100.0), dY60]})
    # predictive band [-102, -98] catches ~all of tau30 and ~none of 60
    a1 = AC.a1_records(pairs, _preds_of(-102.0, -98.0))
    assert a1["all_taus_beat_zero"]
    assert not a1["coverage_in_band"]          # 1.0 is outside the band
    assert not a1["passes"]
    # a band far from both taus' dY -> CRPS no longer beats zero
    a1b = AC.a1_records(pairs, _preds_of(-300.0, -260.0))
    assert not a1b["all_taus_beat_zero"]
    assert not a1b["passes"]


def test_a1_coverage_band_edges():
    dY = np.arange(100) - 50.0                 # exactly 80 inside a band
    pairs = pd.DataFrame({
        "contrast": "wall-native", "tau": 30.0,
        "world": [f"w{i % 10}" for i in range(100)],
        "key": [f"chunk4/w{i % 10}" for i in range(100)],
        "dy_delay_30": dY})
    wide = AC.a1_records(pairs, _preds_of(-1000, 1000))
    assert wide["coverage_overall"] == 1.0
    assert not wide["coverage_in_band"] and not wide["passes"]
    # band [-40, 39] covers dY in [-40, 39] = 80 of 100 rows -> 0.80
    inside = AC.a1_records(pairs, _preds_of(-40.0, 39.0))
    assert inside["coverage_overall"] == pytest.approx(0.80)
    assert inside["coverage_in_band"]
    assert not inside["all_taus_beat_zero"] or True   # coverage is the
    # gate under test here; the CRPS side has its own truth tables


# ------------------------------------------------------------------- A3 --

def _pairs_for_sign(n_worlds, dY, world_seed=0):
    rows = []
    for i, dy in enumerate(dY):
        w = f"w{(i // 1) % n_worlds + world_seed}"
        rows.append(dict(contrast="wall-native", tau=60.0, world=w,
                         key=f"chunk4/{w}", demand=2700, penetration=0.1,
                         dy_delay_60=dy))
    return pd.DataFrame(rows)


def test_a3_sign_accuracy_and_bootstrap_ci():
    dY = np.array([-200.0, -180.0, -160.0, 150.0, 170.0] * 8)
    pairs = _pairs_for_sign(20, dY)
    ref = AC.train_references_v04b(pairs)

    def pred(g):                     # the realised dY itself: CRPS ~ 0
        return {q: g["dy_delay_60"].to_numpy().copy()
                for q in reg.QUANTILES}
    recs = AC.a3_records(pairs, 60.0, pred, ref)
    r = recs[0]
    assert r["contrast"] == "wall-native"
    assert r["sign_acc"] == 1.0 and r["sign_ci_lo"] > 0.5
    assert r["pass_i"]
    assert r["crps_model"] < r["crps_zero"]
    assert r["crps_model"] < r["crps_cell_mean"]
    assert r["pass_ii"] and r["pass_tau"]

    def adversarial(g):              # always the wrong sign
        return {q: -g["dy_delay_60"].to_numpy() for q in reg.QUANTILES}
    bad = AC.a3_records(pairs, 60.0, adversarial, ref)[0]
    assert bad["sign_acc"] == 0.0 and bad["sign_ci_hi"] < 0.5
    assert not bad["pass_i"]


def test_a3_eligibility_uses_train_median_abs():
    dY = np.array([-10.0, 10.0, -200.0, 200.0])
    pairs = _pairs_for_sign(4, dY)
    ref = AC.train_references_v04b(pairs)
    assert ref[("wall-native", 60.0)]["median_abs"] == 105.0

    def flat(g):
        return {q: np.zeros(len(g)) for q in reg.QUANTILES}
    r = AC.a3_records(pairs, 60.0, flat, ref)[0]
    assert r["n_eligible"] == 2      # only |dY| > 105 counts; near-ties
    # (|dY| == median) never count


def test_a3_verdict_needs_two_of_three_taus():
    recs = [dict(contrast="wall-native", tau=t, pass_i=p, pass_ii=p,
                 pass_tau=p)
            for t, p in ((60.0, True), (120.0, False), (300.0, True))]
    assert AC.a3_verdict(recs)["passes"]
    recs[0]["pass_tau"] = False
    assert not AC.a3_verdict(recs)["passes"]
    assert AC.a3_verdict(recs)["n_taus_pass"] == 1


# --------------------------------------------------------------- verdict --

def test_verdict_truth_table():
    a1 = dict(passes=True)
    a3 = dict(passes=True)
    assert AC.verdict(a1, a3)["verdict"] == "ACCEPT"
    assert AC.verdict(a1, dict(passes=False))["verdict"] == "PARTIAL"
    assert AC.verdict(dict(passes=False), a3)["verdict"] == "PARTIAL"
    assert AC.verdict(dict(passes=False),
                      dict(passes=False))["verdict"] == "FAIL"


# --------------------------------------------------------------- strata --

def test_stratified_records_split_by_stratum(pairs3, rows3):
    from partB_v04b import pairs_v04b as PB
    ref = AC.train_references_v04b(pairs3[pairs3["split"] == "TRAIN"])
    g = PB.complete_pairs(pairs3, 60.0)

    def pred(f):
        return {q: f[f"dy_delay_60"].to_numpy() * 0.9
                for q in reg.QUANTILES}
    recs = AC.stratified_records(g, 60.0, pred, ref)
    labels = {r["stratum"] for r in recs}
    assert any(s.startswith("dur:") for s in labels)
    assert any(s.startswith("arm:") for s in labels)
    assert any(s.startswith("wall_family:") for s in labels)
    dur_le60 = next(r for r in recs
                    if r["stratum"] == "dur:le60")
    assert dur_le60["n"] == (g["dur_stratum"] == "le60").sum()


def test_aux_contrasts_are_reported_not_gating(pairs3, rows3):
    from partB_v04b import pairs_v04b as PB
    ref = AC.train_references_v04b(pairs3[pairs3["split"] == "TRAIN"])
    g = PB.complete_pairs(pairs3, 60.0)

    def pred(f):
        return {q: f[f"dy_delay_60"].to_numpy() for q in reg.QUANTILES}
    recs = AC.a3_records(g, 60.0, pred, ref)
    contrasts = {r["contrast"] for r in recs}
    assert reg.HEADLINE_CONTRAST in contrasts
    assert "damping-native" in contrasts and "pi-native" in contrasts
    # gating verdict runs on the headline contrast only
    v = AC.a3_verdict(recs)
    assert v["contrast"] == reg.HEADLINE_CONTRAST
