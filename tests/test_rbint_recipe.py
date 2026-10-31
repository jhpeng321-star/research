from types import SimpleNamespace
from pathlib import Path
import json
import tomllib
from dataclasses import replace

import numpy as np
import torch
import pytest

from tools.rbint.recipe import InstrumentedEntries, candidate_future, native_controller
from tools.rbeval.core import reference
from wms.control.hidden_state import BirthMix, Population, PoissonEntries
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.entries import EntranceQueue, EntryConfig, build_entrance_history
from wms.model.factory import build_model
from wms.model.spec import ModelSpec
from wms.model.sumo_insertion import SumoInsertion
from wms.physics.config import PhysicsConfig


def test_recipe_clears_incompatible_trained_hazard_modes():
    config = tomllib.loads((Path(__file__).resolve().parents[1]/'configs/wm_phys.toml').read_text())
    old = ModelSpec(rollout_mode='hybrid',history_s=60., lane_model='hazard_v5',
        lane_decision='gated',lane_v5_gates=(.2,.3,.4),lane_v5_head_ap=(.1,.2,.3),
        lane_v5_extra_features=True,lane_v5_mainline='sample',
        lane_hazard_features='v2',merge_hazard_zone='full',merge_decision='first_passage')
    rebuilt = replace(old,**config['model'])
    assert rebuilt.history_s == old.history_s and rebuilt.lane_model == 'lc2013'
    physics = dict(config['physics'])
    physics['emergency_decel_mps2'] = None
    legacy = PhysicsConfig(emergency_braking=True, emergency_onset='driver_b')
    assert replace(legacy,**physics).cf_model == 'sumo_idm'


def observer():
    namespace = dict(np=np, torch=torch, ACTORS=[])
    reference('costs', namespace)
    return SimpleNamespace(**namespace)


def test_weighted_hidden_admission_and_motion_use_exact_same_identity():
    vectors = torch.tensor([[30., 1., 2., 1., 2., .2, .1, 6., 1.],
                            [20., 2., 3., 2., 3., .5, .2, 8., .8]])
    population = Population([vectors], [vectors], [torch.tensor([1., 99.])])
    mix = BirthMix(population, 13, [969420])
    admission = SumoInsertion(PhysicsConfig(merge_model='sumo'), mix)
    for ordinal in range(50):
        assert admission.candidate(0, 0, ordinal) == mix.candidate(0, 0, ordinal)


def test_poisson_fractional_source_wait_and_wrapper_delegation():
    hist = build_entrance_history(counter=np.ones((2, 30)))
    process = PoissonEntries(histories=[hist], queues=[EntranceQueue(
        np.zeros(2), np.ones(2), cav_share=np.zeros(2))], batch=1, dt_s=.5,
        cfg=EntryConfig(conserve_queue=True), couple_draws=True, row_seeds=[19])
    entries = InstrumentedEntries(process, observer().Entries)
    process.t_next[:] = [.1, .4]
    entries.observer.due_area = np.zeros(1)
    process._request_rngs = [[SimpleNamespace(exponential=lambda scale: 10.)] * 2]
    assert process._emit_due(0) == [0, 1]
    np.testing.assert_allclose(entries.due_area, [.5])  # .5-.1 + .5-.4
    assert entries.cfg is process.cfg
    process.last_birth_ordinals = {(0, 3): 7}
    assert entries.last_birth_ordinals == {(0, 3): 7}


@pytest.mark.parametrize('variant', ['B', 'C'])
def test_fixed_population_for_present_and_newborns(variant):
    torch.manual_seed(1)
    model = build_model(ModelSpec(rollout_mode='driver_params', lane_model='lc2013'),
                        PhysicsConfig(merge_model='sumo')).eval()
    model._wmphys_variant = variant
    if variant == 'C':
        artifact = Path(__file__).resolve().parents[1]/'artifacts/rbfit/pooled_full_v5/idm_pooled.json'
        model._wmphys_fit = json.loads(artifact.read_text())['parameters']
    model.cav_exec_cap = 'idm'
    state = dict(s_m=torch.tensor([[-900., -800., 0., 0.]]),
        v_mps=torch.full((1, 4), 20.), a_mps2=torch.zeros(1, 4),
        lane=torch.zeros(1, 4, dtype=torch.long), length_m=torch.full((1, 4), 5.),
        present=torch.tensor([[True, True, False, False]]),
        is_cav=torch.tensor([[False, True, False, False]]))
    history = torch.zeros(1, 4, model.history_len, model.history_features)
    mask = state['present'][..., None].expand(1, 4, model.history_len)
    hist = build_entrance_history(counter=np.ones((2, 30)))
    q = EntranceQueue(np.zeros(2), np.ones(2), cav_share=np.zeros(2))
    batch = SimpleNamespace(rollout_kwargs=dict(history=history, history_mask=mask),
        entry_spec=dict(histories=[hist], queues=[q]))
    kw, entries, receipt = candidate_future(model=model, batch=batch, initial=state,
                                            future=2, seed=969420, frozen=observer())
    assert receipt['posterior_source'] == ('fixed_inherited_physics_population'
        if variant == 'B' else 'motion_fitted_pooled_point')
    expected = torch.stack([kw['driver_params'][n] for n in PARAMETER_NAMES], -1)
    birth = kw['entrant_driver_sampler'].candidate(0, 0, 4)
    assert all(float(expected[0, 0, i]) == birth[n] for i, n in enumerate(PARAMETER_NAMES))
    if variant == 'C':
        assert float(expected[0, 0, 0]) == pytest.approx(model._wmphys_fit['v0'])
        assert float(expected[0, 1, 0]) == pytest.approx(model.physics.v0_mps)
    assert entries.process.cfg.conserve_queue
    state['exit_s_m'] = torch.full_like(state['s_m'], 553.33)
    with torch.no_grad():
        rollout = model.rollout(state, n_steps=30, cav_controller=native_controller(), **kw)
    audit = entries.audit_population(state, rollout)
    assert audit['conserved'] and sum(audit['due'][0]) > 0


def test_full_recipe_observable_identity_fitted_posterior_and_learned_source(monkeypatch):
    from tools.rbint import observations
    from tests.unit.test_fitted_hidden import fitted
    from wms.control.arrival_fit import LearnedArrivalFit
    model = build_model(ModelSpec(rollout_mode='driver_params', lane_model='lc2013'),
                        PhysicsConfig(merge_model='sumo', cf_model='sumo_idm')).eval()
    model._wmphys_variant = 'FULL'
    model._wmphys_fitted_state = fitted()
    post = model._wmphys_fitted_state.posterior['drivers'].pop('969400:101')
    model._wmphys_fitted_state.posterior['drivers']['969420:42'] = post
    model._wmphys_arrival_fit = LearnedArrivalFit(dict(
        schema='wm_phys.observed_arrivals.v1', generator_family='piecewise_constant_poisson',
        conditioning_end_s=300., observed_phase_end_s=1200., worlds={}))
    state = dict(s_m=torch.tensor([[-900., -800., 0., 0.]]),
        v_mps=torch.full((1, 4), 20.), a_mps2=torch.zeros(1, 4),
        lane=torch.zeros(1, 4, dtype=torch.long), length_m=torch.full((1, 4), 5.),
        present=torch.tensor([[True, True, False, False]]),
        is_cav=torch.tensor([[False, True, False, False]]))
    # Original slots 1/3 pack into 0/1; recorded IDs42/103 are neither ordinal.
    original = {n:v[:, [2, 0, 3, 1]].clone() for n,v in state.items()}
    start = dict(state=original, initial_state_sha256='synthetic',
        past_entries=[dict(time=1., lane=0), dict(time=2., lane=2)],
        observed_queue_history=[[float(t), 0., 0.] for t in range(10, 301, 10)])
    monkeypatch.setattr(observations, 'load_observations',
                        lambda seed: (start, torch.tensor([42, 103]), {'synthetic': 'bound'}))
    hist = build_entrance_history(counter=np.ones((2, 30)))
    history = torch.zeros(1, 4, model.history_len, model.history_features)
    mask = state['present'][..., None].expand(1, 4, model.history_len)
    batch = SimpleNamespace(rollout_kwargs=dict(history=history, history_mask=mask),
        metadata=[dict(initial_state_sha256='synthetic')],
        entry_spec=dict(histories=[hist], queues=[EntranceQueue(
            np.zeros(2), np.ones(2), cav_share=np.zeros(2))]))
    kw, entries, receipt = candidate_future(model=model, batch=batch, initial=state,
        future=2, seed=969420, frozen=observer())
    assert kw['driver_params']['v0'][0, 0] == 43  # posterior ID42, not population17
    with torch.no_grad():
        _, _, original_params = model.encode_drivers(state, **batch.rollout_kwargs)
    assert kw['driver_params']['v0'][0, 1] == original_params['v0'][0, 1]  # CAV unchanged
    assert kw['entrant_driver_sampler'].candidate(0, 0, 4)['v0'] == 17
    assert receipt['rate_source'] == 'fitted_entrance_counters'
    assert receipt['present_identities'] == 2 and receipt['variant'] == 'FULL'
    replay, _, replay_receipt = candidate_future(model=model, batch=batch, initial=state,
        future=2, seed=969420, frozen=observer())
    assert replay_receipt == receipt
    assert torch.equal(replay['driver_params']['v0'], kw['driver_params']['v0'])
    np.testing.assert_array_equal(replay['entries'].t_next, entries.t_next)
    # Refuse a packed-slot permutation or changed start before drawing futures.
    with pytest.raises(ValueError, match='cohort differs'):
        observations.fitted_inputs({n:v[:, [1, 0, 2, 3]] for n,v in state.items()},
            969420, 'synthetic', model._wmphys_arrival_fit)
    with pytest.raises(ValueError, match='different registered start'):
        observations.fitted_inputs(state, 969420, 'other', model._wmphys_arrival_fit)
