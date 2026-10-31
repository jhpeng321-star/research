import numpy as np
import pytest

pytest.importorskip('scipy', reason='optional offline count diagnostic; cpuQ has the existing SciPy environment')

from scipy.integrate import quad
from scipy.stats import gamma, poisson

from tools.rbunc.arrival_prefix import discrete_crps, forecast, scores


def test_gamma_poisson_matches_independent_rate_quadrature_and_total_variance():
    distribution = forecast(4, 10., 7., family='gamma_poisson')
    for count in (0, 2, 6, 12):
        integral = quad(lambda rate: poisson.pmf(count, rate*7.)
                        * gamma.pdf(rate, a=4.5, scale=.1), 0., np.inf)[0]
        assert distribution.pmf(count) == pytest.approx(integral, abs=1e-10)
    assert distribution.mean() == pytest.approx(4.5 * .7)
    assert distribution.var() == pytest.approx(4.5 * .7 * 1.7)
    same_mean = forecast(4, 10., 7., family='posterior_mean_poisson')
    assert same_mean.mean() == pytest.approx(distribution.mean())
    assert same_mean.var() < distribution.var()


def test_integer_crps_matches_pairwise_expectations_and_discrete_coverage_mass():
    distribution = forecast(3, 10., 10., family='gamma_poisson')
    support = np.arange(100)
    weights = distribution.pmf(support)
    target = 5
    oracle = np.dot(weights, abs(support-target)) - .5 * (
        weights[:, None]*weights[None, :]*abs(support[:, None]-support[None, :])).sum()
    assert discrete_crps(distribution, target) == pytest.approx(oracle, abs=1e-10)
    result = scores(distribution, target)
    lo, hi = result['interval90']
    assert result['discrete_nominal90'] == pytest.approx(distribution.pmf(np.arange(lo, hi+1)).sum())
    assert result['discrete_nominal90'] >= .9


def test_zero_prefix_preserves_rate_uncertainty_and_reports_impossible_mle():
    distribution = forecast(0, 150., 150., family='gamma_poisson')
    assert distribution.mean() == pytest.approx(.5)
    assert distribution.var() == pytest.approx(1.)
    assert np.isfinite(scores(distribution, 2)['nll'])
    mle = scores(forecast(0, 150., 150., family='mle_poisson'), 2)
    assert mle['impossible'] and mle['nll'] is None
