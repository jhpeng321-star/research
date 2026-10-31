"""Meaningful source isolation, inference likelihood, CRN and deterministic replay tests."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wms.control.hidden_state import (Population, _motion_weights, _prediction,
                                      plan_hidden_state_futures)
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.entries import EntranceQueue, build_entrance_history


def fixture(rows=1):
    vectors = torch.tensor([[30., 1.2, 2., 1.2, 2., .4, .2, 8., 1.],
                            [34., .8, 1., 2., 3., .2, .1, 5., 1.3],
                            [26., 1.8, 3., .8, 1.5, .7, .3, 10., .8],
                            [30., 1.2, 2., 1.2, 2., .4, .2, 8., 1.],
                            [30., 1.2, 2., 1.2, 2., .4, .2, 8., 1.]])
    params = {name: vectors[:, j][None].expand(rows, -1).clone()
              for j, name in enumerate(PARAMETER_NAMES)}
    state = dict(s_m=torch.tensor([[-1000., -900., -800., -700., 0.]]).expand(rows, -1).clone(),
                 v_mps=torch.full((rows, 5), 20.), a_mps2=torch.zeros(rows, 5),
                 length_m=torch.full((rows, 5), 5.),
                 lane=torch.zeros(rows, 5, dtype=torch.long),
                 present=torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool).expand(rows, -1).clone(),
                 is_cav=torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.bool).expand(rows, -1).clone())
    history = torch.zeros(rows, 5, 20, 9)
    history[..., 0], history[..., 2], history[..., 5] = 20 / 30, 1, 1
    mask = state['present'][..., None].expand(-1, -1, 20).clone()
    for b in range(rows):
        for k in range(4):
            history[b, k, :, 1] = _prediction(history[b, k], vectors[k:k+1])[0] / 3
    class Encoder:
        physics = SimpleNamespace(dt_s=.5, entry_learned=False)
        driver_encoder = SimpleNamespace(shrink_prior_steps=8.)
        calls = 0
        def encode_drivers(self, state, **kwargs):
            self.calls += 1
            # Deliberately absurd untrained variance: the new path must ignore it.
            return {'uncertainty': torch.full((rows, 5, 9), 1e9)}, torch.ones(rows, 5, 3), params
    return Encoder(), state, dict(driver_history=history, driver_history_mask=mask), params


def plan(model, state, history, **kw):
    return plan_hidden_state_futures(model, state, history, future_sampling='hidden_state', **kw)


def test_off_is_noop_and_rejects_runtime_event_noise():
    model, state, h, _ = fixture()
    rng = torch.get_rng_state().clone()
    assert plan_hidden_state_futures(model, state, h) is None
    assert model.calls == 0 and torch.equal(rng, torch.get_rng_state())
    for attr, value in (('_samples_lanes', True), ('merge_pivot', object())):
        setattr(model, attr, value)
        with pytest.raises(ValueError, match='require'):
            plan(model, state, h)
        delattr(model, attr)
    model.training = True
    with pytest.raises(ValueError, match='evaluation mode'):
        plan(model, state, h)
    model.training = False
    for key in ('accel_noise', 'present_seq', 'observe_states', 'lane_uniform'):
        with pytest.raises(ValueError, match='reject process noise'):
            plan(model, state, {**h, key: torch.ones(1)})


def test_point_likelihood_identifies_speed_and_preserves_unobserved_lane_prior():
    point = torch.tensor([30., 1.2, 2., 1.2, 2., .4, .2, 8., 1.])
    particles = point.repeat(3, 1)
    particles[1, 0] = 15.
    particles[2, 5] = .9
    history = torch.zeros(30, 9)
    history[:, 0] = torch.linspace(.4, .9, 30)
    history[:, 1] = _prediction(history, point[None])[0] / 3
    weights, diag = _motion_weights(history, torch.ones(30, dtype=torch.bool),
        particles, point, torch.ones(3) / 3)
    assert weights[1] < 1e-10
    torch.testing.assert_close(weights[0], weights[2])
    assert diag['particle_ess'] == pytest.approx(2.)


def test_joint_population_for_zero_history_and_no_embedding_or_global_noise():
    model, state, h, params = fixture()
    pop = Population.from_motion(params, state, h['driver_history_mask'])
    h['driver_history_mask'][:] = False
    before = torch.get_rng_state().clone()
    futures = plan(model, state, h, population=pop, n_samples=32)
    assert torch.equal(before, torch.get_rng_state())
    for future in futures:
        assert future.receipt['driver_branches'] == dict(population=3, posterior=0)
        assert set(future.kwargs) == {'driver_params', 'driver_embedding'}
        assert torch.equal(future.kwargs['driver_embedding'], torch.ones(1, 5, 3))
        for slot in range(3):
            vec = torch.stack([future.kwargs['driver_params'][n][0, slot] for n in PARAMETER_NAMES])
            assert any(torch.equal(vec, p) for p in pop.hdv[0])
        assert all(torch.equal(future.kwargs['driver_params'][n][0, 3:], params[n][0, 3:])
                   for n in PARAMETER_NAMES)  # CAV + padding untouched
    assert len({float(f.kwargs['driver_params']['v0'][0, 0]) for f in futures}) > 1


def test_rows_crn_birth_identity_persistence_and_replay():
    model, state, h, _ = fixture(rows=2)
    histories = [build_entrance_history(counter=np.ones((2, 30)))] * 2
    queues = [EntranceQueue(np.zeros(2), np.array([1., .2]), cav_share=np.zeros(2))] * 2
    kw = dict(n_samples=2, seed=13, row_keys=['world', 'world'],
              entrance_histories=histories, queues=queues)
    futures = plan(model, state, h, **kw)
    repeat = plan(model, state, h, **kw)
    for a, b in zip(futures, repeat):
        for name in PARAMETER_NAMES:
            assert torch.equal(a.kwargs['driver_params'][name], b.kwargs['driver_params'][name])
            assert torch.equal(a.kwargs['driver_params'][name][0], a.kwargs['driver_params'][name][1])
        entries, mix = a.kwargs['entries'], a.kwargs['entrant_driver_sampler']
        # Request streams agree between paired actions despite different insertion.
        for _ in range(80):
            assert entries._emit_due(0) == entries._emit_due(1)
        birth = torch.zeros_like(state['present'])
        birth[0, 4] = True
        expected = entries.cfg.learned_admission.candidate(0, 0, 7)
        assigned = mix(state, birth, a.kwargs['driver_params'], birth_ordinals={(0, 4): 7})
        assert all(float(assigned[n][0, 4]) == expected[n] for n in PARAMETER_NAMES)
        nothing = mix(state, torch.zeros_like(birth), assigned)
        assert all(torch.equal(assigned[n], nothing[n]) for n in PARAMETER_NAMES)
    assert not np.array_equal(futures[0].kwargs['entries'].t_next, futures[1].kwargs['entries'].t_next)


def test_exact_poisson_zero_rate_no_clock_shift_or_headway_floor():
    model, state, h, _ = fixture()
    histories = [build_entrance_history(counter=np.zeros((2, 30)))]
    queues = [EntranceQueue(np.zeros(2), np.array([2., 0.]), cav_share=np.zeros(2))]
    totals = []
    for future in plan(model, state, h, n_samples=100, seed=71,
                       entrance_histories=histories, queues=queues):
        process = future.kwargs['entries']
        assert np.isinf(process.t_next[0, 1])
        # First request is scheduled relative to t0, not t0+dt.
        first = process.t_next[0, 0]
        for _ in range(120):
            process._emit_due(0)
        assert process.due_total[0, 1] == 0
        totals.append(process.due_total[0, 0])
        assert first >= 0
    # Distribution-level check, fixed private RNG; Poisson(120) mean and variance.
    assert abs(np.mean(totals) - 120) < 4
    assert 70 < np.var(totals, ddof=1) < 175


def test_ablations_fix_exactly_the_requested_sources():
    model, state, h, _ = fixture()
    histories = [build_entrance_history(counter=np.ones((2, 30)))]
    a = plan(model, state, h, n_samples=8, sample_present=False,
             sample_arrivals=False, entrance_histories=histories)
    for f in a[1:]:
        assert all(torch.equal(a[0].kwargs['driver_params'][n], f.kwargs['driver_params'][n])
                   for n in PARAMETER_NAMES)
        assert np.array_equal(a[0].kwargs['entries'].t_next, f.kwargs['entries'].t_next)
        assert a[0].kwargs['entrant_driver_sampler'].candidate(0, 0, 11) == \
               f.kwargs['entrant_driver_sampler'].candidate(0, 0, 11)


def test_real_physics_repeated_future_is_bit_identical_and_sources_change_outcomes():
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    from wms.physics.config import PhysicsConfig
    encoder, state, h, params = fixture()
    model = WmSModel(ModelSpec(rollout_mode='driver_params'),
                     physics=PhysicsConfig()).eval()
    # Fixed observable-only inference fixture isolates sampling from random weights.
    model.encode_drivers = encoder.encode_drivers
    futures = plan(model, state, h, n_samples=8)
    outputs = []
    def controller(s):
        return torch.stack((s['s_m'] + .5 * s['v_mps'], s['lane'].float(),
                            s['v_mps'], torch.zeros_like(s['s_m'])), -1)
    with torch.inference_mode():
        for future in futures:
            r = model.rollout({k: v.clone() for k, v in state.items()}, n_steps=8,
                              cav_controller=controller, **future.kwargs)
            repeat = model.rollout({k: v.clone() for k, v in state.items()}, n_steps=8,
                                   cav_controller=controller, **future.kwargs)
            assert torch.equal(r['positions'], repeat['positions'])
            assert torch.isfinite(r['speeds']).all()
            outputs.append(r['positions'])
    assert any(not torch.equal(outputs[0], x) for x in outputs[1:])


def test_supplied_motion_posterior_is_used_without_untrained_variance_or_caps():
    model, state, h, params = fixture()
    particles = torch.stack([torch.stack([params[n] for n in PARAMETER_NAMES], -1)] * 2)
    particles[0, :, :, 0] = 12.
    particles[1, :, :, 0] = 42.
    def encode(state, **kwargs):
        return {'posterior_samples': particles}, None, params
    model.encode_drivers = encode
    values = [f.kwargs['driver_params']['v0'][0, 0].item()
              for f in plan(model, state, h, n_samples=64)]
    assert set(values) == {12., 42.}


def test_population_is_required_when_no_motion_donors_exist():
    model, state, h, _ = fixture()
    h['driver_history_mask'][:] = False
    with pytest.raises(ValueError, match='no observed HDV population'):
        plan(model, state, h)


def test_actual_rollout_admits_same_settings_it_later_moves_and_replays_exactly():
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    encoder, state, h, _ = fixture()
    model = WmSModel(ModelSpec(rollout_mode='driver_params')).eval()
    model.encode_drivers = encoder.encode_drivers
    histories = [build_entrance_history(counter=np.zeros((2, 30)))]
    queues = [EntranceQueue(np.array([1., 0.]), np.zeros(2), cav_share=np.zeros(2))]
    results = []
    def controller(s):
        return torch.stack((s['s_m'] + .5 * s['v_mps'], s['lane'].float(),
                            s['v_mps'], torch.zeros_like(s['s_m'])), -1)
    before = torch.get_rng_state().clone()
    for _ in range(2):
        f = plan(model, state, h, n_samples=1, seed=51,
                 entrance_histories=histories, queues=queues)[0]
        expected = f.kwargs['entries'].cfg.learned_admission.candidate(0, 0, 0)
        with torch.inference_mode():
            r = model.rollout({k: v.clone() for k, v in state.items()}, n_steps=5,
                              cav_controller=controller, **f.kwargs)
        assert r['entrant'][0, 4]
        assert all(float(r['driver_parameters'][0, 4, j]) == expected[n]
                   for j, n in enumerate(PARAMETER_NAMES))
        assert f.kwargs['entries'].stats[0].inserted == 1
        assert not f.kwargs['entries'].pending.any()
        results.append(r)
    assert torch.equal(before, torch.get_rng_state())
    for key in ('positions', 'speeds', 'lanes', 'masks', 'driver_parameters'):
        assert torch.equal(results[0][key], results[1][key])


def test_sumo_configured_after_planning_uses_exact_weighted_newborn_draw():
    model, state, h, _ = fixture()
    histories = [build_entrance_history(counter=np.zeros((2, 30)))]
    q = [EntranceQueue(np.zeros(2), np.zeros(2), cav_share=np.zeros(2))]
    f = plan(model, state, h, n_samples=1, entrance_histories=histories, queues=q)[0]
    process, mix = f.kwargs['entries'], f.kwargs['entrant_driver_sampler']
    # Emulates the merge adapter's configure_entries call inside rollout.
    insertion = SimpleNamespace(mix=mix, candidate=lambda *a: {'incorrect': True})
    process.cfg.sumo_insertion = insertion
    process.insert({k: v.clone() for k, v in state.items()})
    for ordinal in (0, 7, 19):
        assert insertion.candidate(0, 0, ordinal) == mix.candidate(0, 0, ordinal)
    process.cfg.sumo_insertion = SimpleNamespace(mix=object())
    with pytest.raises(ValueError, match='must share'):
        process.insert(state)


def test_likelihood_uses_executed_plant_bounds_at_compressed_gap():
    point = torch.tensor([30., 1.2, 2., 1.2, 2., .4, .2, 8., 1.])
    history = torch.zeros(10, 9)
    history[:, 0], history[:, 1], history[:, 5] = .8, -1., .003
    history[:, 7], history[:, 8] = -.5, 1
    raw = _prediction(history, point[None])
    assert raw.min() < -1000
    bounded = _prediction(history, point[None], (-9., 3.))
    assert (bounded == -9).all()
    _, diag = _motion_weights(history, torch.ones(10, dtype=torch.bool),
        point.repeat(2, 1), point, torch.ones(2)/2, (-9., 3.))
    assert diag['residual_mse'] == pytest.approx(36.)
