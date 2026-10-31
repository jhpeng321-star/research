from types import SimpleNamespace

import numpy as np
import torch

from tools.rbint.recipe import InstrumentedEntries, candidate_future
from tools.rbeval.core import reference
from wms.control.hidden_state import BirthMix, Population, PoissonEntries
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.entries import EntranceQueue, EntryConfig, build_entrance_history
from wms.model.factory import build_model
from wms.model.spec import ModelSpec
from wms.model.sumo_insertion import SumoInsertion
from wms.physics.config import PhysicsConfig


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


def test_candidate_B_is_fixed_population_for_present_and_newborns():
    torch.manual_seed(1)
    model = build_model(ModelSpec(rollout_mode='driver_params', lane_model='lc2013'),
                        PhysicsConfig(merge_model='sumo'))
    model._wmphys_variant = 'B'
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
    assert receipt['posterior_source'] == 'fixed_inherited_physics_population'
    expected = torch.stack([kw['driver_params'][n] for n in PARAMETER_NAMES], -1)
    birth = kw['entrant_driver_sampler'].candidate(0, 0, 4)
    assert all(float(expected[0, 0, i]) == birth[n] for i, n in enumerate(PARAMETER_NAMES))
    assert entries.process.cfg.conserve_queue
