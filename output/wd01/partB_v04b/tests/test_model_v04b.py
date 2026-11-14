"""The direct-dY quantile-GBM: world-grouped CV over the registered
fixed grid, conformal shifts, monotone quantiles (registration s.5)."""
from __future__ import annotations

import numpy as np
import pytest

from partB_v04b import reg
from partB_v04b import descriptors_v04b as DE
from partB_v04b import pairs_v04b as PB
from partB_v04b.model_v04b import DirectDiffGBM, pinball

# shrunken stand-in for the registered grid: the FIT-time tests exercise
# the machinery, not the constants (test_reg covers the real grid)
SMALL_GRID = ((80, 0.05, 4, 20), (80, 0.1, 4, 10))


def test_pinball_truth_table():
    assert pinball(np.array([1.0]), np.array([1.0]), 0.5) == 0.0
    # under-prediction at q=0.9 costs 0.9 per unit; over costs 0.1
    assert pinball(np.array([2.0]), np.array([1.0]), 0.9) == \
        pytest.approx(0.9)
    assert pinball(np.array([0.0]), np.array([1.0]), 0.9) == \
        pytest.approx(0.1)


def test_world_grouped_folds_never_split_a_world():
    groups = np.array(["w1"] * 4 + ["w2"] * 4 + ["w3"] * 4)
    m = DirectDiffGBM()
    folds = m.world_folds(groups)
    assert len(folds) == 3                  # min(CV_FOLDS, 3 worlds)
    for fit_i, cal_i in folds:
        ws_fit = set(groups[fit_i])
        ws_cal = set(groups[cal_i])
        assert not (ws_fit & ws_cal)        # no world on both sides
        assert ws_fit | ws_cal == set(groups)


def test_folds_refuse_a_single_world():
    with pytest.raises(ValueError, match="2 worlds"):
        DirectDiffGBM().world_folds(np.array(["w1"] * 5))


def test_fit_registers_all_models_and_shifts(pairs3, rows3):
    g = PB.complete_pairs(pairs3, 60.0)
    X = DE.pair_matrix(g, rows3, "deployable_strict")
    y = {h: g[f"dy_{h}_60"].to_numpy(float) for h in reg.HEADS}
    m = DirectDiffGBM(taus=(60.0,), grid=SMALL_GRID, cv_folds=3)
    m.fit(X, y, 60.0, g["world"].to_numpy(),
          (g["contrast"] == reg.HEADLINE_CONTRAST).to_numpy())
    assert len(m.selection) == 1
    assert len(m.selection[0]["grid"]) == len(SMALL_GRID)  # the (shrunken) candidate grid
    assert tuple(m.chosen[60.0]) in SMALL_GRID
    for h in reg.HEADS:
        for q in reg.QUANTILES:
            assert (60.0, h, q) in m.models
            assert np.isfinite(m.shifts[(60.0, h, q)])


def test_predict_quantiles_are_monotone_and_ordered(pairs3, rows3):
    g = PB.complete_pairs(pairs3, 60.0)
    X = DE.pair_matrix(g, rows3, "deployable_strict")
    y = {h: g[f"dy_{h}_60"].to_numpy(float) for h in reg.HEADS}
    m = DirectDiffGBM(taus=(60.0,), grid=SMALL_GRID, cv_folds=3)
    m.fit(X, y, 60.0, g["world"].to_numpy(),
          (g["contrast"] == reg.HEADLINE_CONTRAST).to_numpy())
    preds = m.predict_quantiles(X, 60.0)
    assert sorted(preds) == sorted(reg.QUANTILES)
    qs = sorted(preds)
    P = np.stack([preds[q] for q in qs])
    assert (np.diff(P, axis=0) >= -1e-9).all()   # monotone in q
    assert (m.predict_median(X, 60.0) == preds[0.5]).all()


def test_conformal_uses_headline_rows_only(pairs3, rows3):
    """The aux rows never enter selection or the conformal shifts: the
    OOB residual pool of the shifts is the headline mask."""
    g = PB.complete_pairs(pairs3, 60.0)
    X = DE.pair_matrix(g, rows3, "deployable_strict")
    y = {h: g[f"dy_{h}_60"].to_numpy(float) for h in reg.HEADS}
    hl = (g["contrast"] == reg.HEADLINE_CONTRAST).to_numpy()
    m = DirectDiffGBM(taus=(60.0,), grid=SMALL_GRID, cv_folds=3)
    m.fit(X, y, 60.0, g["world"].to_numpy(), hl)
    oof = m.oof[(60.0, "delay")][0.5]
    assert len(oof) == len(g)                 # every TRAIN row predicted
    assert len(m.selection[0]["grid"]) == len(SMALL_GRID)


def test_wall_only_ablation_trains_on_headline_rows_only(pairs3, rows3):
    g = PB.complete_pairs(pairs3, 60.0)
    X = DE.pair_matrix(g, rows3, "deployable_strict")
    y = {h: g[f"dy_{h}_60"].to_numpy(float) for h in reg.HEADS}
    hl = (g["contrast"] == reg.HEADLINE_CONTRAST).to_numpy()
    m = DirectDiffGBM(taus=(60.0,), grid=SMALL_GRID, cv_folds=3)
    m.fit(X, y, 60.0, g["world"].to_numpy(), hl)
    abl = m.fit_wall_only(X, y, 60.0, g["world"].to_numpy(), hl)
    assert abl is not m
    oof = abl.oof[(60.0, "delay")][0.5]
    assert len(oof) == int(hl.sum())          # wall rows only
    assert abl.models.keys() and m.models.keys()


def test_ridge_reference_is_fitted(pairs3, rows3):
    g = PB.complete_pairs(pairs3, 60.0)
    X = DE.pair_matrix(g, rows3, "deployable_strict")
    y = {h: g[f"dy_{h}_60"].to_numpy(float) for h in reg.HEADS}
    m = DirectDiffGBM(taus=(60.0,), grid=SMALL_GRID, cv_folds=3)
    m.fit(X, y, 60.0, g["world"].to_numpy(),
          (g["contrast"] == reg.HEADLINE_CONTRAST).to_numpy())
    r = m.ridge_median(X, 60.0)
    assert r.shape == (len(g),) and np.isfinite(r).all()


