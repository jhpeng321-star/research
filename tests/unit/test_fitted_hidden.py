import copy

import numpy as np
import pytest
import torch

from tests.unit.test_hidden_state_futures import fixture, plan
from wms.control.fitted_hidden import FittedHiddenState, IDM_ORDER, LC_ORDER
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.entries import EntranceQueue, build_entrance_history


def fitted(lane=False):
    pop = dict(schema='wm_phys.idm_population.v1', parameter_order=IDM_ORDER,
               distribution='finite_support_mixture', support=[[43., 2., 3., 4., 5.],
                   [17., .7, .8, 1.1, 1.2]], weights=[0., 1.])
    post = dict(schema='wm_phys.idm_posteriors.v1', parameter_order=IDM_ORDER,
                conditioning_end_s=300, drivers={'969400:101': dict(support_indices=[0], weights=[1.])})
    lp = dict(schema='wm_phys.lc2013_population.v1', parameter_order=LC_ORDER,
              support=[[2., 3., 4., 5., 6.], [.2, .3, .4, .5, .6]], weights=[0., 1.])
    ls = dict(schema='wm_phys.lc2013_posteriors.v1',
              conditioning_end_s=300, drivers={'969400:101': dict(weights=[1., 0.])})
    return FittedHiddenState(pop, post, lane_population=lp if lane else None,
                             lane_posterior=ls if lane else None)


def test_fitted_joint_order_no_clipping_prior_fallback_and_cav_preservation():
    model, state, h, params = fixture(rows=2)
    ids = torch.tensor([[101, 102, 103, 104, -1]]).expand(2, -1)
    rng = torch.get_rng_state().clone()
    futures = plan(model, state, h, fitted_state=fitted(), vehicle_ids=ids,
                   t0_s=300, row_keys=[969400]*2, n_samples=3)
    assert torch.equal(rng, torch.get_rng_state())
    for f in futures:
        p = f.kwargs['driver_params']
        assert [p[n][0, 0].item() for n in IDM_ORDER] == [43., 2., 3., 4., 5.]
        assert [p[n][0, 1].item() for n in IDM_ORDER] == pytest.approx([17., .7, .8, 1.1, 1.2])
        for n in PARAMETER_NAMES:
            assert torch.equal(p[n][0], p[n][1])
            assert torch.equal(p[n][:, 3:], params[n][:, 3:])
        assert f.receipt['driver_branches'] == dict(posterior=2, population=4)


def test_identity_permutation_and_fitted_ablation_are_exact_crn():
    model, state, h, params = fixture()
    fit = fitted()
    fit.population['weights'] = [1., 1.]
    fit.weights[:] = 1
    ids = torch.tensor([[101, 102, 103, 104, -1]])
    kw = dict(fitted_state=fit, t0_s=300, row_keys=[969400], n_samples=8, seed=77)
    original = plan(model, state, h, vehicle_ids=ids, **kw)
    permutation = torch.tensor([2, 0, 1, 3, 4])
    model.encode_drivers = lambda state, **h: (None, None, {n:p[:, permutation] for n,p in params.items()})
    moved = plan(model, {n:v[:, permutation] for n,v in state.items()},
                 {n:v[:, permutation] for n,v in h.items()}, vehicle_ids=ids[:, permutation], **kw)
    for a, b in zip(original, moved):
        for n in IDM_ORDER:
            assert torch.equal(a.kwargs['driver_params'][n][:, permutation], b.kwargs['driver_params'][n])
    fixed = plan(model, state, h, vehicle_ids=ids, sample_present=False, **kw)
    for f in fixed[1:]:
        for n in IDM_ORDER:
            assert torch.equal(f.kwargs['driver_params'][n], fixed[0].kwargs['driver_params'][n])


def test_lc_joint_persistence_and_newborn_exact_admission_identity():
    model, state, h, _ = fixture()
    ids = torch.tensor([[101, 102, 103, 104, -1]])
    histories = [build_entrance_history(counter=np.zeros((2, 30)))]
    queues = [EntranceQueue(np.array([1., 0.]), np.zeros(2), cav_share=np.zeros(2))]
    f = plan(model, state, h, fitted_state=fitted(lane=True), t0_s=300,
             row_keys=[969400], vehicle_ids=ids, entrance_histories=histories, queues=queues,
             n_samples=1)[0]
    params = f.kwargs['driver_params']
    assert [params[n][0, 0].item() for n in LC_ORDER] == [2., 3., 4., 5., 6.]
    mix = f.kwargs['entrant_driver_sampler']
    candidate = f.kwargs['entries'].cfg.learned_admission.candidate(0, 0, 9)
    born = torch.zeros_like(state['present'])
    born[0, 4] = True
    new = mix(state, born, params, birth_ordinals={(0, 4): 9})
    for n, value in candidate.items():
        assert new[n][0, 4].item() == pytest.approx(value)
        assert torch.equal(mix(state, torch.zeros_like(born), new)[n], new[n])
    assert [candidate[n] for n in LC_ORDER] == [.2, .3, .4, .5, .6]


def test_conditioning_identity_and_schema_fail_closed():
    model, state, h, _ = fixture()
    kw = dict(fitted_state=fitted(), t0_s=299, row_keys=[969400], vehicle_ids=torch.zeros(1, 5))
    with pytest.raises(ValueError, match='conditioning'):
        plan(model, state, h, **kw)
    kw['t0_s'] = 300
    kw['vehicle_ids'] = None
    with pytest.raises(ValueError, match='vehicle_ids'):
        plan(model, state, h, **kw)
    bad = copy.deepcopy(fitted().population)
    bad['parameter_order'] = tuple(reversed(IDM_ORDER))
    with pytest.raises(ValueError, match='schema/order'):
        FittedHiddenState(bad, fitted().posterior)
    bad = copy.deepcopy(fitted().posterior)
    bad['drivers']['969400:101']['support_indices'] = [-1]
    fit = FittedHiddenState(fitted().population, bad)
    with pytest.raises(ValueError, match='indices/weights'):
        fit.draw(969400, 101, generator=torch.Generator())
