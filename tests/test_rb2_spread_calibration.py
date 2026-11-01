import copy
import json

import numpy as np
import pytest
import torch
from scipy.stats import nbinom, poisson

from tests.unit.test_fitted_hidden import fitted
from tests.unit.test_hidden_state_futures import fixture, plan
from tools.rbcal.fit import count_nll, quadratic_fit, weighted_crps_terms
from tools.rbcal.outcomes import development_truth, snapshot_dev
from tools.rbeval.core import ARMS
from wms.control.arrival_fit import LearnedArrivalFit
from wms.control.spread_calibration import SpreadCalibration
from wms.model.entries import EntranceQueue, build_entrance_history


def calibration(retention=1., variance=1., alpha=None):
    artifact = dict(schema='wm_phys.spread_calibration.v1', version='test',
        fit_worlds=list(range(969400, 969432)), conditioning_end_s=300,
        learned_corrections=False, idm_population_sha256='population', arrivals_sha256='arrivals',
        parameters=dict(posterior_retention=retention, arrival_variance_multiplier=variance,
                        entrant_concentration=alpha))
    return SpreadCalibration(artifact, sha256='calibration')


def source():
    base = fitted()
    base.population['weights'] = [.4, .6]
    base.weights[:] = torch.tensor([.4, .6], dtype=torch.float64)
    base.provenance['idm_population_sha256'] = 'population'
    return base


def arrivals():
    base = LearnedArrivalFit(dict(schema='wm_phys.observed_arrivals.v1',
        generator_family='piecewise_constant_poisson', conditioning_end_s=300,
        observed_phase_end_s=1200, worlds={'969400':dict(total=[123, 40], exposure_s=300.)}))
    base.sha256 = 'arrivals'
    return base


def test_crps_quadratic_against_independent_pairwise_identity():
    rng = np.random.default_rng(891)
    x, y = rng.normal(size=(13, 7)), rng.normal(size=13)
    p, q = rng.dirichlet(np.ones(7), 2)
    cc = weighted_crps_terms(x, y, p, q)
    for fraction in (0., .13, .5, .87, 1.):
        w = p+fraction*(q-p)
        direct = (np.abs(x-y[:, None])@w - .5*np.sum(
            np.abs(x[:, :, None]-x[:, None, :])*w[None, :, None]*w[None, None, :], axis=(1, 2))).sum()
        assert cc@[1, fraction, fraction*fraction] == pytest.approx(direct, abs=1e-12)
    assert quadratic_fit([[0., -.6, 1.]]) == pytest.approx(.3)


def test_gamma_poisson_log_score_against_scipy_and_point_endpoint():
    count, target = np.array([0., 11., 100.]), np.array([3, 8, 120])
    for dispersion in (.3, 1., 7.):
        expected = -nbinom((count+.5)/dispersion, 1/(1+dispersion)).logpmf(target)
        np.testing.assert_allclose(count_nll(count, target, dispersion), expected, rtol=1e-12)
    np.testing.assert_allclose(count_nll(count, target, 0), -poisson(count+.5).logpmf(target))


def test_retention_endpoints_and_causal_rebinding_clear_saved_ids():
    base = source()
    for r, expected in ((0., [.4, .6]), (.5, [.7, .3]), (1., [1., 0.])):
        adapted = calibration(retention=r).hidden(base)
        torch.testing.assert_close(adapted.dense_weights(969400, 101), torch.tensor(expected, dtype=torch.float64))
        torch.testing.assert_close(adapted.dense_weights(969400, 999), base.weights)
        rebound = adapted.with_motion_posteriors({}, t0_s=300)
        assert not rebound.posterior['drivers']
        torch.testing.assert_close(rebound.dense_weights(969400, 101), base.weights)


def test_disabled_equivalent_is_bit_identical_and_rng_local():
    model, state, h, _ = fixture(rows=2)
    base = source()
    ids = torch.tensor([[101, 102, 103, 104, -1]]).expand(2, -1)
    kwargs = dict(vehicle_ids=ids, t0_s=300, row_keys=[969400]*2, n_samples=8, seed=42)
    before = torch.get_rng_state().clone()
    a = plan(model, state, h, fitted_state=base, **kwargs)
    b = plan(model, state, h, fitted_state=calibration().hidden(base), **kwargs)
    for fa, fb in zip(a, b):
        for name in fa.kwargs['driver_params']:
            torch.testing.assert_close(fa.kwargs['driver_params'][name], fb.kwargs['driver_params'][name], rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())
    for sample in range(10):
        pa = arrivals().draw(969400, t0_s=300, seed=42, sample=sample)
        pb = calibration().arrivals(arrivals()).draw(969400, t0_s=300, seed=42, sample=sample)
        np.testing.assert_array_equal(pa.rates, pb.rates)


def test_future_mixture_crn_and_admission_newborn_coupling():
    model, state, h, _ = fixture(rows=2)
    fit = calibration(alpha=0., retention=.5).hidden(source())
    ids = torch.tensor([[101, 102, 103, 104, -1]]).expand(2, -1)
    history = build_entrance_history(counter=np.zeros((2, 30)))
    queue = EntranceQueue(np.array([1., 0.]), np.zeros(2), cav_share=np.zeros(2))
    before = torch.get_rng_state().clone()
    futures = plan(model, state, h, fitted_state=fit, vehicle_ids=ids,
        row_keys=[969400]*2, t0_s=300, n_samples=8, seed=42,
        entrance_histories=[history]*2, queues=[queue]*2)
    weights = []
    for future in futures:
        mix = future.kwargs['entrant_driver_sampler']
        torch.testing.assert_close(mix.weights[0], mix.weights[1], rtol=0, atol=0)
        weights.append(mix.weights[0])
        new = torch.zeros_like(state['present'])
        new[:, 4] = True
        params = mix(state, new, future.kwargs['driver_params'], {(0, 4):19, (1, 4):19})
        a = future.kwargs['entries'].cfg.learned_admission.candidate(0, 0, 19)
        assert a == mix.candidate(1, 0, 19)
        for name, value in a.items():
            assert params[name][0, 4].item() == pytest.approx(value)
    assert any(not torch.equal(weights[0], w) for w in weights[1:])
    assert torch.equal(before, torch.get_rng_state())


def test_rate_moments_and_no_extra_poisson_noise():
    base = arrivals()
    fit = calibration(variance=2.).arrivals(base)
    draws = np.array([fit.draw(969400, t0_s=300, seed=12, sample=k).rates[0]
                      for k in range(8000)])
    np.testing.assert_allclose(draws.mean(0), np.array([123.5, 40.5])/300, rtol=.015)
    np.testing.assert_allclose(draws.var(0), 2*np.array([123.5, 40.5])/300**2, rtol=.06)
    fixed = calibration(variance=0.).arrivals(base)
    np.testing.assert_array_equal(fixed.draw(969400, t0_s=300, seed=5, sample=0).rates,
                                  fixed.draw(969400, t0_s=300, seed=5, sample=999).rates)


def test_calibration_binding_and_invalid_parameters_rejected():
    c = calibration()
    for key, value in [('posterior_retention', 1.1), ('entrant_concentration', -1.),
                       ('arrival_variance_multiplier', float('nan'))]:
        artifact = copy.deepcopy(c.artifact)
        artifact['parameters'][key] = value
        with pytest.raises(ValueError):
            SpreadCalibration(artifact)
    artifact = copy.deepcopy(c.artifact)
    artifact['fit_worlds'].append(969432)
    with pytest.raises(ValueError):
        SpreadCalibration(artifact)
    with pytest.raises(ValueError, match='population mismatch'):
        c.hidden(fitted())


def test_factual_reader_stops_before_reserved_outcome_body(tmp_path):
    rows = [dict(seed=world, arm=arm) for world in range(969400, 969432) for arm in ARMS]
    raw = json.dumps(dict(episodes=rows), indent=1)
    # Match the registered pretty-printed episode contract. The reserved outcome
    # is intentionally not valid JSON: reaching it would make the test fail.
    prefix = raw[:raw.rfind(' ]')]
    prefix = prefix.rstrip()+',\n  {\n   "seed": 969432,\nINVALID RESERVED OUTCOME'
    path = tmp_path/'factual.json'
    path.write_text(prefix)
    assert len(development_truth(path)) == 160


def test_snapshot_admits_explicit_development_names_only(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    shard = source/'shard0'
    shard.mkdir()
    (shard/'IDENTITY.json').write_text('{}')
    for world in range(969420, 969432):
        for future in range(8):
            (shard/f'w{world}_k{future}.json').write_text('{}')
    (shard/'w969432_k0.json').write_text('RESERVED: MUST NOT COPY OR PARSE')
    destination = tmp_path/'snapshot'
    snapshot_dev(source, destination)
    assert len(list((destination/'shard0').glob('w*_k*.json'))) == 96
    assert not (destination/'shard0/w969432_k0.json').exists()
    with pytest.raises(ValueError, match='development12'):
        snapshot_dev(source, tmp_path/'bad', worlds=range(969432, 969440))
