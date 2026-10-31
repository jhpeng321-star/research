from types import SimpleNamespace

import pytest
import torch

from tools.rbeval.encoded_fleet import EncodedPointBirths
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.entries import EntryConfig, EntranceHistory, TorchEntryProcessV2
from wms.model.sumo_insertion import SumoInsertion
from wms.physics.config import PhysicsConfig


def fixture():
    population = torch.tensor([32., 1.5, 3., 1.8, 2.5, .2, .1, 6., 1.])
    defaults = torch.tensor([31.29, 1.1, 2.5, 2.6, 4.5, .2, .1, 6., 1.])
    observed = population.clone()
    observed[0] = 27.
    def settings(ids, cav):
        values = torch.stack([defaults if c else observed if int(i) == 10 else population
                              for i, c in zip(ids.flatten(), cav.flatten())]).view(*ids.shape, -1)
        return {n: values[..., i] for i, n in enumerate(PARAMETER_NAMES)}
    encoded = SimpleNamespace(population=population, defaults=defaults, settings=settings)
    state = dict(s_m=torch.tensor([[100., 0., 0., 0.]]), v_mps=torch.zeros(1, 4),
        a_mps2=torch.zeros(1, 4), lane=torch.tensor([[1, 0, 0, 0]]),
        present=torch.tensor([[True, False, False, False]]),
        is_cav=torch.tensor([[True, False, False, False]]),
        length_m=torch.full((1, 4), 5.), vehicle_id=torch.tensor([[10, -1, -1, -1]]))
    return encoded, state


def test_initial_motion_settings_follow_identity_and_keep_cav_defaults():
    encoded, state = fixture()
    state['vehicle_id'][0, 1:3] = torch.tensor([10, 999])
    sampler = EncodedPointBirths(encoded)
    p = sampler.initial(state)
    torch.testing.assert_close(p['v0'], torch.tensor([[31.29, 27., 32., 32.]]))
    assert sampler.candidate(0, 1, 999)['v0'] == 32.
    assert sampler.candidate(0, 0, 999, True)['v0'] == pytest.approx(31.29)


def test_actual_admission_and_newborn_motion_use_identical_point_settings():
    encoded, state = fixture()
    sampler = EncodedPointBirths(encoded)
    old = sampler.initial(state)
    cfg = PhysicsConfig(merge_model='sumo')
    process = TorchEntryProcessV2(history=EntranceHistory.defaults(),
        cfg=EntryConfig(conserve_queue=True, insert_rule='sumo_like_v2',
                        sumo_insertion=SumoInsertion(cfg, sampler)), row_seeds=[17], couple_draws=True)
    process._emit_due = lambda b: [1]
    born = process.insert(state)
    assert born.sum() == 1
    new = sampler(state, born, old, birth_ordinals=process.last_birth_ordinals)
    expected = sampler.candidate(0, 1, 0)
    for name in PARAMETER_NAMES:
        assert new[name][born][0] == pytest.approx(expected[name])
        torch.testing.assert_close(new[name][~born], old[name][~born], rtol=0, atol=0)
    assert sampler.births_checked == 1
    assert process._source_ledgers[0, 1].queue == 0


def test_recycled_slot_cannot_keep_departed_driver_settings_or_guess_ordinal():
    encoded, state = fixture()
    sampler = EncodedPointBirths(encoded)
    old = sampler.initial(state)
    old['v0'][0, 1] = 12.
    born = torch.tensor([[False, True, False, False]])
    with pytest.raises(ValueError, match='birth ordinals'):
        sampler(state, born, old)
    new = sampler(state, born, old, birth_ordinals={(0, 1): 5})
    assert new['v0'][0, 1] == 32.
    assert new['v0'][0, 0] == old['v0'][0, 0]
