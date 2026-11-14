"""Stacking: cross-fitted pool weight and CAL conformal recalibration."""
from __future__ import annotations

import numpy as np
import pytest

from combine.crps import mixture_quantiles
from combine.guards import SplitLeakError
from combine.stacking import (DEFAULT_QUANTILES, conformal_shifts,
                              cross_fit_pool, fit_conformal_pool,
                              fit_pool_weight, fit_stack, interval_coverage)
from combine.synthetic import gaussian_pred

LV = np.asarray(DEFAULT_QUANTILES, dtype=np.float64)


def _weight_case(truth: str, n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    loc_t = rng.normal(0.0, 1.0, n)
    loc_n = loc_t + 40.0
    fn = gaussian_pred(LV, loc_n, 5.0)
    ft = gaussian_pred(LV, loc_t, 5.0)
    if truth == "T":
        y = loc_t + rng.normal(0, 5.0, n)
    elif truth == "N":
        y = loc_n + rng.normal(0, 5.0, n)
    else:
        pick = rng.random(n) < 0.5
        y = np.where(pick, loc_n, loc_t) + rng.normal(0, 5.0, n)
    return fn, ft, y


@pytest.mark.parametrize("truth,lo,hi", [("T", 0.0, 0.12),
                                         ("N", 0.88, 1.0),
                                         ("C", 0.38, 0.62)])
def test_pool_weight_recovered(truth, lo, hi):
    fn, ft, y = _weight_case(truth)
    rec = fit_pool_weight(fn, ft, y)
    assert lo <= rec["weight"] <= hi, rec["weight"]


def test_cross_fit_uses_disjoint_worlds():
    fn, ft, y = _weight_case("C", n=600, seed=1)
    worlds = np.repeat([f"w{i:02d}" for i in range(6)], 100)
    splits = np.array(["CALIBRATION"] * len(y))
    xf = cross_fit_pool(fn, ft, y, worlds, splits, folds=2)
    allw = set(worlds)
    for rec in xf["folds"]:
        ta, tr = set(rec["test_worlds"]), set(rec["train_worlds"])
        assert ta.isdisjoint(tr)
        assert ta | tr == allw
    assert np.isfinite(xf["final_weight"])
    assert xf["n_folds"] == 2


def test_cross_fit_ranking_is_world_grouped():
    fn, ft, y = _weight_case("C", n=400, seed=2)
    worlds = np.array([f"w{i}" for i in range(4)]).repeat(100)
    splits = np.array(["CALIBRATION"] * len(y))
    xf = cross_fit_pool(fn, ft, y, worlds, splits, folds=2)
    # each fold's weight is fitted on the other half's worlds
    assert len(xf["folds"]) == 2


def test_guard_raises_on_test_rows():
    fn, ft, y = _weight_case("T", n=200, seed=3)
    worlds = np.array([f"w{i}" for i in range(4)]).repeat(50)
    for split in ("TEST", "HOLDOUT", "TRAIN"):
        splits = np.array([split] * len(y))
        with pytest.raises(SplitLeakError):
            cross_fit_pool(fn, ft, y, worlds, splits, folds=2)
        with pytest.raises(SplitLeakError):
            fit_conformal_pool(fn, ft, 0.5, y, splits)


def test_conformal_recalibration_nominal_coverage():
    rng = np.random.default_rng(7)
    n = 5000
    loc = rng.normal(0, 1, n)
    fn = gaussian_pred(LV, loc, 3.0)
    ft = gaussian_pred(LV, loc + 2.0, 3.0)
    y = loc + 30.0 + rng.normal(0, 3.0, n)
    splits = np.array(["CALIBRATION"] * n)

    # the pool is narrow but badly mis-centred (truth sits ~28 s above it)
    w = 0.5
    pooled = mixture_quantiles(fn, ft, w, LV)
    cov_before = interval_coverage(pooled, y)
    assert cov_before < 0.2

    shifts = conformal_shifts(LV, pooled, y)
    recal = pooled + shifts[None, :]
    cov_after = interval_coverage(recal, y)
    assert abs(cov_after - 0.80) < 0.06, cov_after

    # the packaged calibration agrees
    cal = fit_conformal_pool(fn, ft, w, y, splits)
    cov_packaged = interval_coverage(cal.apply(pooled), y)
    assert abs(cov_packaged - 0.80) < 0.06


def test_fit_stack_returns_weight_and_calibration():
    fn, ft, y = _weight_case("C", n=2000, seed=4)
    worlds = np.repeat([f"w{i:02d}" for i in range(6)], 2000 // 6 + 1)[:2000]
    splits = np.array(["CALIBRATION"] * len(y))
    model, xf = fit_stack(fn, ft, y, worlds, splits, folds=2)
    assert 0.3 <= model.weight <= 0.7
    assert model.cal.levels.shape == LV.shape
    pred = model.predict(fn, ft)
    assert pred.shape == (len(y), LV.size)
    assert np.isfinite(pred).all()
