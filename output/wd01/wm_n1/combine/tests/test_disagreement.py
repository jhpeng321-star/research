"""Disagreement signal: distance, monotonicity, reporting helpers."""
from __future__ import annotations

import numpy as np

from combine.crps import build_cdf_grid
from combine.disagreement import (abstain_curve, disagreement,
                                  spearman_abs_error, tau_scale)
from combine.stacking import DEFAULT_QUANTILES
from combine.synthetic import gaussian_pred

LV = np.asarray(DEFAULT_QUANTILES, dtype=np.float64)


def test_identical_distributions_give_zero():
    rng = np.random.default_rng(0)
    loc = rng.normal(0, 5, 200)
    fn = gaussian_pred(LV, loc, 2.0)
    ft = gaussian_pred(LV, loc, 2.0)
    d = disagreement(fn, ft)
    assert np.allclose(d, 0.0, atol=1e-9)


def test_shifted_distributions_increase_monotonically():
    base = np.linspace(0.0, 50.0, 300)
    fn = gaussian_pred(LV, base, 2.0)
    ft_far = gaussian_pred(LV, base - 12.0, 2.0)
    xg = build_cdf_grid([fn, ft_far], np.array([base.min(), base.max()]))
    means = []
    for delta in (0.5, 1.0, 2.0, 4.0, 8.0):
        ft = gaussian_pred(LV, base - delta, 2.0)
        means.append(float(np.mean(disagreement(fn, ft, x_grid=xg))))
    assert all(b > a for a, b in zip(means[:-1], means[1:])), means


def test_scale_normalises_to_unit_median():
    rng = np.random.default_rng(1)
    loc = rng.normal(0, 5, 100)
    fn = gaussian_pred(LV, loc, 2.0)
    ft = gaussian_pred(LV, loc - 3.0, 2.0)
    raw = disagreement(fn, ft)
    s = tau_scale(raw)
    assert s > 0
    norm = disagreement(fn, ft, scale=s)
    assert np.isclose(np.median(norm), 1.0)
    assert tau_scale(np.array([0.0, 0.0])) > 0


def test_spearman_and_abstain_curve():
    d = np.linspace(0.0, 10.0, 200)
    err = 3.0 * d + 0.01 * np.random.default_rng(2).normal(0, 1, 200)
    rec = spearman_abs_error(d, err)
    assert rec["rho"] > 0.99
    curve = abstain_curve(d, err)
    assert {"threshold", "frac_kept", "mean_abs_error",
            "n_kept"} <= set(curve.columns)
    assert curve["frac_kept"].is_monotonic_increasing
    assert curve["mean_abs_error"].is_monotonic_increasing
