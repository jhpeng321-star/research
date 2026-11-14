"""D14 calibration unit tests: per-tau shift/isotonic maps on synthetic
biased quantiles."""
from __future__ import annotations

import numpy as np
import pytest

from partB_v0.calibration import (CAL_LEVELS, FIT_LEVELS, CalMap,
                                  cal_metric, coverage, fit_per_tau,
                                  pit_from_quantiles, quantile_matrix)
from partB_v0.model import PmfGrid


def _synthetic(n=400, bias=0.6, seed=0):
    """y ~ N(2, 1) per tau; predicted quantiles = true quantiles + bias."""
    rng = np.random.default_rng(seed)
    taus = np.array([30.0] * n + [300.0] * n)
    y = rng.normal(2.0, 1.0, size=2 * n)
    lv = np.asarray(FIT_LEVELS)
    true_q = np.quantile(rng.normal(2.0, 1.0, size=20000), lv)
    qmat = true_q[None, :] + bias + rng.normal(0, 0.02, size=(2 * n, 1))
    return qmat, y, taus


def test_shift_calibration_reduces_cal_metric():
    qmat, y, taus = _synthetic()
    cal0 = cal_metric(y, qmat)
    m = fit_per_tau(qmat, y, taus, method="shift")
    qcal = m.apply(qmat, taus)
    cal1 = cal_metric(y, qcal)
    print(f"CAL {cal0:.3f} -> {cal1:.3f}")
    assert cal1 < cal0
    assert cal1 < 0.05
    # the fitted shift for a +0.6 bias is about -0.6 at the median level
    # (tail levels are noisier with n=400, so only the centre is pinned)
    for mp in m.per_tau.values():
        assert abs(mp["shift"][len(mp["shift"]) // 2] + 0.6) < 0.15


def test_isotonic_map_monotone_and_calibrating():
    qmat, y, taus = _synthetic()
    m = fit_per_tau(qmat, y, taus, method="isotonic")
    qcal = m.apply(qmat, taus)
    assert cal_metric(y, qcal) < cal_metric(y, qmat)
    for mp in m.per_tau.values():
        for iso in mp["iso"]:
            x = np.asarray(iso["knots_x"])
            v = np.asarray(iso["knots_y"])
            assert (np.diff(v) >= -1e-9).all()    # monotone
            assert len(x) == len(v) and len(x) >= 5


def test_coverage_and_pit_helpers():
    rng = np.random.default_rng(1)
    n = 500
    grid = PmfGrid()
    # spread pmfs centred on a known normalised value: (1/4, 1/2, 1/4)
    # over bins (k-1, k, k+1) -> q10 < value < q90 exactly
    vals = rng.uniform(0.5, 2.0, size=n)
    probs = np.zeros((n, grid.n_bins))
    for i, v in enumerate(vals):
        k = int(np.searchsorted(grid.centers, v))
        k = min(max(k, 1), grid.n_bins - 2)
        probs[i, k - 1:k + 2] = (0.25, 0.5, 0.25)
    qmat = quantile_matrix(grid, probs, CAL_LEVELS)
    y = np.array([grid.centers[min(max(int(np.searchsorted(
        grid.centers, v)), 1), grid.n_bins - 2)] for v in vals])
    cov = coverage(y, qmat)
    assert cov[-1] == pytest.approx(1.0)          # everything <= own q90
    assert cov[0] == pytest.approx(0.0)           # nothing <= own q10
    y_below = 0.5 * y
    cov_b = coverage(y_below, qmat)
    # below the own q10 at every row -> covered at EVERY level
    assert np.allclose(cov_b, 1.0)
    pit = pit_from_quantiles(y, quantile_matrix(grid, probs, FIT_LEVELS),
                             FIT_LEVELS)
    # y = own bin center: F(c_k) = 0.25 + 0.50 = 0.75 under the
    # interpolated quantile function, so the PIT sits at ~0.75 (the
    # q50 of this pmf is the c[k-1]-c[k] midpoint, not the center)
    assert ((pit > 0.65) & (pit < 0.85)).mean() > 0.95


def test_cal_map_json_round_trip(tmp_path):
    qmat, y, taus = _synthetic(n=60)
    m = fit_per_tau(qmat, y, taus, method="shift")
    m.save(tmp_path / "calibration.json")
    m2 = CalMap.load(tmp_path / "calibration.json")
    assert m2.method == m.method
    assert np.allclose(m2.apply(qmat, taus), m.apply(qmat, taus))


def test_thin_tau_left_raw():
    qmat, y, taus = _synthetic(n=5)      # 5 rows per tau < min_rows
    m = fit_per_tau(qmat, y, taus, method="shift", min_rows=30)
    assert m.per_tau == {}
