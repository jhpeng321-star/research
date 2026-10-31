import numpy as np
import torch
import pytest

from tests.unit.test_hidden_state_futures import fixture, plan
from wms.control.arrival_fit import DemandPlan, LearnedArrivalFit, request_bins
from wms.model.entries import EntranceQueue, build_entrance_history


def artifact():
    return dict(schema='wm_phys.observed_arrivals.v1', conditioning_end_s=300,
        generator_family='piecewise_constant_poisson', observed_phase_end_s=1200,
        worlds={'969400': dict(total=[150, 30], exposure_s=300.)})


def test_requests_recovered_under_blocking_with_exact_causal_boundaries():
    events = [dict(time=2., lane=1), dict(time=4., lane=0), dict(time=6., lane=2)]
    queues = [[2., 1, 1], [4., 1, 1], [6., 1, 1], [8., 2, 1], [10., 2, 2]]
    result = request_bins(events, queues, t0_s=10., bin_s=2.)
    assert result['counts'] == [[2, 1], [1, 0], [0, 1], [1, 0], [0, 1]]
    assert result['total'] == [4, 3]  # Not the [2,1] insertion count.
    with pytest.raises(ValueError, match='noncausal'):
        request_bins(events+[dict(time=12., lane=0)], queues, t0_s=10., bin_s=2.)
    with pytest.raises(ValueError, match='conservation'):
        request_bins([], [[2., 2, 0], [4., 0, 0]], t0_s=4., bin_s=2.)


def test_exact_piecewise_hazard_crosses_zero_rate_and_rate_boundaries():
    p = DemandPlan([0., 10., 20., 30.], [[0., 1.], [2., 1.], [.5, 0.]])
    assert p.next_time(0., 0, 3.) == 11.5
    assert p.next_time(19., 0, 3.) == 22.
    assert p.next_time(19., 1, 2.) == np.inf
    assert p.next_time(5., 1, 7.) == 12.
    with pytest.raises(ValueError, match='piecewise'):
        DemandPlan([0, 2, 1], [[1, 1], [1, 1]])


def test_learned_rate_posterior_and_exact_paired_requests():
    model, state, h, _ = fixture(rows=2)
    fit = LearnedArrivalFit(artifact())
    histories = [build_entrance_history(counter=np.zeros((2, 30)))]*2
    queues = [EntranceQueue(np.zeros(2), np.zeros(2), cav_share=np.zeros(2))]*2
    futures = plan(model, state, h, arrival_fit=fit, t0_s=300, row_keys=[969400]*2,
                   n_samples=64, seed=123, entrance_histories=histories, queues=queues)
    totals, rates = [], []
    for future in futures:
        process = future.kwargs['entries']
        np.testing.assert_array_equal(process.demand_plans[0].rates, process.demand_plans[1].rates)
        for _ in range(1200):
            assert process._emit_due(0) == process._emit_due(1)
        totals.append(process.due_total[0].sum())
        rates.append(process.demand_plans[0].rates.sum())
    assert abs(np.mean(totals)-361.) < 20.
    # Posterior-predictive Var(N600) = E(lambda)*600 + Var(lambda)*600² =1083.
    assert 550. < np.var(totals, ddof=1) < 1700.
    assert np.std(rates) > 0


def test_fixed_arrival_source_and_fit_conditioning_fail_closed():
    model, state, h, _ = fixture()
    fit = LearnedArrivalFit(artifact())
    histories = [build_entrance_history(counter=np.zeros((2, 30)))]
    futures = plan(model, state, h, arrival_fit=fit, t0_s=300, row_keys=[969400],
                   n_samples=4, sample_arrivals=False, entrance_histories=histories)
    for f in futures[1:]:
        np.testing.assert_array_equal(f.kwargs['entries'].t_next, futures[0].kwargs['entries'].t_next)
    for world, t0, message in [(969400, 301, 'conditioning'), (969432, 300, 'no causal')]:
        with pytest.raises(ValueError, match=message):
            fit.draw(world, t0_s=t0, seed=0, sample=0)
    conditioned = fit.condition(969431, [], [[float(t), 0, 0] for t in range(10, 301, 10)])
    assert '969431' not in fit.artifact['worlds']
    assert conditioned.draw(969431, t0_s=300, seed=0, sample=0).rates.sum() > 0
    process = futures[0].kwargs['entries']
    process.now[:] = 900.
    with pytest.raises(ValueError, match='exceeds'):
        process._emit_due(0)


def test_causal_counter_receipts_distinguish_conditioning_with_same_source_file():
    raw = artifact()
    source = LearnedArrivalFit(raw)
    source.sha256 = 'frozen-source-family'
    original = source.observation_receipt(969400, t0_s=300)
    raw['worlds']['969400']['total'][0] = 999  # source owns its copy
    assert source.observation_receipt(969400, t0_s=300) == original
    queues = [[float(t), 0, 0] for t in range(10, 301, 10)]
    first = source.condition(969400, [], queues)
    second = source.condition(969400, [dict(time=5., lane=0)], queues)
    a = first.observation_receipt(969400, t0_s=300)
    b = second.observation_receipt(969400, t0_s=300)
    assert a['source_sha256'] == b['source_sha256'] == 'frozen-source-family'
    assert len({a['observation_sha256'], b['observation_sha256'], original['observation_sha256']}) == 3
    assert source.observation_receipt(969400, t0_s=300) == original
    replay = source.condition(969400, [], queues)
    assert replay.observation_receipt(969400, t0_s=300) == a


def test_future_receipts_bind_paired_arrival_observations_without_rng_consumption():
    model, state, h, _ = fixture(rows=2)
    fit = LearnedArrivalFit(artifact())
    before = torch.get_rng_state().clone()
    futures = plan(model, state, h, arrival_fit=fit, t0_s=300, row_keys=[969400]*2,
                   n_samples=2,
                   entrance_histories=[build_entrance_history(counter=np.zeros((2, 30)))]*2)
    assert torch.equal(before, torch.get_rng_state())
    observations = futures[0].receipt['learned_arrival_observations']
    assert observations[0] == observations[1] == fit.observation_receipt(969400, t0_s=300)
    assert futures[1].receipt['learned_arrival_observations'] == observations


def test_fractional_observed_request_counts_are_not_a_poisson_observation():
    raw = artifact()
    raw['worlds']['969400']['total'][0] = .5
    with pytest.raises(ValueError, match='counts/exposure'):
        LearnedArrivalFit(raw).draw(969400, t0_s=300, seed=0, sample=0)
