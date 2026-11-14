"""Twohot / symlog / distribution-metric contract tests (synthetic)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from partB_v0.distributions import (DistMetrics, bin_centers_np,
                                    crps_empirical, spearman)
from partB_v0.model import (N_BINS, SYM_MAX, bin_centers, inv_symlog,
                            nll_twohot, symlog, tau_sinusoidal,
                            twohot_encode, decode_mean)


def test_symlog_round_trip():
    x = torch.tensor([0.0, 1e-3, 1.0, -30.0, 2000.0, 1e5, -1e6])
    y = inv_symlog(symlog(x))
    assert torch.allclose(y, x, rtol=1e-5, atol=1e-6)


def test_twohot_rows_sum_to_one_and_use_two_bins():
    x = torch.tensor([0.0, 90.0, -1500.0, 40000.0])
    p = twohot_encode(x)
    assert p.shape == (4, N_BINS)
    assert torch.allclose(p.sum(-1), torch.ones(4))
    for row in p:
        assert int((row > 1e-8).sum()) <= 2


def test_twohot_decode_within_bin_resolution():
    """decode(pmf) within ~1% of the value for x >= 100 veh·s."""
    centers = torch.as_tensor(bin_centers_np())
    for x in (30.0, 100.0, 500.0, 2000.0, 17000.0, 57000.0):
        p = twohot_encode(torch.tensor([x]))
        dec = float(p.double() @ centers.double())
        assert abs(dec - x) <= 0.012 * max(abs(x), 30.0), (x, dec)


def test_twohot_clamps_out_of_range():
    p = twohot_encode(torch.tensor([-1e12, 1e12]))
    assert p[0, 0] == 1.0
    assert p[1, -1] == 1.0


def test_crps_point_mass_equals_abs_distance():
    met = DistMetrics()
    k = 137
    pm = np.zeros((1, N_BINS))
    pm[0, k] = 1.0
    for y in (met.centers[k], met.centers[k] - 250.0,
              met.centers[k] + 90.0):
        got = met.crps(pm, np.array([y]))[0]
        assert got == pytest.approx(abs(met.centers[k] - y), rel=1e-6)


def test_crps_uniform_pmf_beats_point_at_far_tail():
    met = DistMetrics()
    uni = np.full((1, N_BINS), 1.0 / N_BINS)
    pt = np.zeros((1, N_BINS))
    pt[0, 20] = 1.0
    y = np.array([5000.0])
    assert met.crps(uni, y)[0] < met.crps(pt, y)[0]


def test_pit_and_quantile_and_coverage():
    met = DistMetrics()
    uni = np.full((200, N_BINS), 1.0 / N_BINS)
    pit = met.pit(uni, met.centers[128] + np.zeros(200))
    assert ((pit > 0) & (pit < 1)).all()
    q10, q90 = met.quantile(uni, 0.1), met.quantile(uni, 0.9)
    assert (q10 < q90).all()
    cov = met.central_coverage(uni, np.array([met.centers[128]]))
    assert cov[0]


def test_quantile_monotone_in_q():
    met = DistMetrics()
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(N_BINS), size=3)
    qs = [met.quantile(p, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
    for a, b in zip(qs, qs[1:]):
        assert (a <= b + 1e-9).all()


def test_crps_empirical_known_value():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.5])
    # mean|x - y| = 0.625; mean pairwise |x-x'| = (sum)/n^2:
    # pairs dist mean = 1.25 -> crps = 0.625 - 0.625 = 0 ... check directly
    got = crps_empirical(s, y)[0]
    d1 = np.abs(s - y[0]).mean()
    d2 = np.abs(s[None, :] - s[:, None]).mean() * 0.5
    assert got == pytest.approx(d1 - d2)


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 1, 1], [2, 3, 4]))
    assert spearman([1, 2, 3, 4], [2, 4, 9, 8]) == pytest.approx(0.8)


def test_nll_twohot_reaches_entropy_floor_for_perfect_logits():
    """A soft twohot target has an entropy floor; matching logits
    (log-proportional) attain exactly that floor."""
    x = torch.tensor([123.0, -40.0])
    t = twohot_encode(x)
    logits = torch.log(t.clamp(min=1e-12))
    floor = -(t * torch.log(t.clamp(min=1e-12))).sum(-1)
    got = nll_twohot(logits, t)
    assert got.item() == pytest.approx(floor.mean().item(), abs=1e-5)
    # and a one-hot target is driven to 0 exactly
    one = torch.zeros(1, N_BINS)
    one[0, 77] = 1.0
    perfect = torch.log(one.clamp(min=1e-12) * 1e12)   # big logit at 77
    assert nll_twohot(perfect, one).item() == pytest.approx(0.0, abs=1e-4)


def test_tau_embedding_bounded_and_differs():
    a = tau_sinusoidal(torch.tensor([30.0]), 64)
    b = tau_sinusoidal(torch.tensor([570.0]), 64)
    assert a.shape == (1, 64)
    assert (a.abs() <= 1.0 + 1e-6).all()
    assert not torch.allclose(a, b)
