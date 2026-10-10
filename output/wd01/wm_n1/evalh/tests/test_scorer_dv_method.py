"""Tests for the paired-dV construction flag (comonotone vs independent).

Generated numbers only: no CAL, HOLDOUT or trajectory data is opened.  The
synthetic coverage experiment (``rho=0``) is the reference for the claim that
comonotone subtraction under-covers a genuinely independent difference.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pytest

from evalh.protocol import N_QUANTILES, QUANTILE_LEVELS, OutcomeRow
from evalh.scorer.dv_sampling_coverage import coverage_experiment
from evalh.scorer.metrics_loop import (DV_METHOD_COMONOTONE,
                                       DV_METHOD_INDEPENDENT,
                                       independent_difference_quantiles, score)

_NORM = NormalDist()


def _const(base: float) -> np.ndarray:
    return np.full(N_QUANTILES, float(base))


def _marg(mu: float, sigma: float) -> np.ndarray:
    return np.asarray([mu + sigma * _NORM.inv_cdf(float(v))
                       for v in QUANTILE_LEVELS])


def test_constant_opponent_recovers_shifted_marginal():
    qa = _marg(100.0, 12.0)
    q_b = _const(40.0)
    dv = independent_difference_quantiles(
        qa, q_b, levels=QUANTILE_LEVELS, n_samples=20000, seed=1, key="k")
    assert dv.shape == (N_QUANTILES,)
    assert np.all(np.diff(dv) >= -1e-9)
    assert np.allclose(dv, qa - 40.0, atol=1.0)


def test_deterministic_with_key_and_seed():
    qa, qb = _marg(0.0, 10.0), _marg(5.0, 20.0)
    a = independent_difference_quantiles(qa, qb, n_samples=2048, seed=3, key=7)
    b = independent_difference_quantiles(qa, qb, n_samples=2048, seed=3, key=7)
    c = independent_difference_quantiles(qa, qb, n_samples=2048, seed=4, key=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def _independent_toy(n_worlds: int = 300, seed: int = 11):
    rng = np.random.default_rng(seed)
    rows, qa = [], []
    for i in range(n_worlds):
        dy = float(np.sqrt(2.0) * rng.normal())
        rows.append(OutcomeRow(world=f"w{i}", t0_s=0.0, decision_t_s=0.0,
                               tau_s=60.0, policy="native", y_delay_s=0.0))
        rows.append(OutcomeRow(world=f"w{i}", t0_s=0.0, decision_t_s=0.0,
                               tau_s=60.0, policy="pi", y_delay_s=dy))
        # two IDENTICAL standard-normal marginals on both arms: the true
        # difference is N(0, sqrt(2)), so comonotone dV is identically 0.
        qa.append(_marg(0.0, 1.0))
        qa.append(_marg(0.0, 1.0))
    return rows, np.stack(qa)


def test_independent_method_widens_and_covers_where_comonotone_fails():
    rows, qa = _independent_toy()
    com = score(rows, {"A": qa, "v2g": qa}, taus=(60,), n_boot=50, seed=1,
                dv_method=DV_METHOD_COMONOTONE)
    ind = score(rows, {"A": qa, "v2g": qa}, taus=(60,), n_boot=50, seed=1,
                dv_method=DV_METHOD_INDEPENDENT)
    c = com["cells"]["pi-native"]["60"]
    i = ind["cells"]["pi-native"]["60"]
    assert com["dv_method"] == DV_METHOD_COMONOTONE
    assert ind["dv_method"] == DV_METHOD_INDEPENDENT
    assert c["coverage_10_90_A"] <= 0.05
    assert i["coverage_10_90_A"] >= 0.70
    assert i["coverage_10_90_A"] > c["coverage_10_90_A"]


def test_unknown_method_refused():
    rows, qa = _independent_toy(n_worlds=4)
    with pytest.raises(ValueError):
        score(rows, {"A": qa, "v2g": qa}, taus=(60,), dv_method="bogus")


def test_coverage_experiment_matches_the_claim():
    doc = coverage_experiment(n_pairs=3000, seed=5, rho=0.0, n_samples=4096)
    cov = doc["coverage"]
    assert cov[DV_METHOD_COMONOTONE] < 0.70
    assert 0.72 <= cov[DV_METHOD_INDEPENDENT] <= 0.88
    assert doc["crps"][DV_METHOD_INDEPENDENT] < doc["crps"][DV_METHOD_COMONOTONE]
    corr = coverage_experiment(n_pairs=3000, seed=5, rho=0.5, n_samples=4096)
    assert corr["coverage"][DV_METHOD_INDEPENDENT] >= 0.85
