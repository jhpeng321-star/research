import numpy as np
import pytest
import torch

from wms.model.entries import ENTRY_S_M, EntryConfig, EntranceHistory, TorchEntryProcessV2
from wms.model.spec import ModelSpec
from wms.model.sumo_departure import step_after_movement
from wms.model.sumo_insertion import POSITION_EPS
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig


def setup(positions, lanes, speeds, present):
    n = len(positions)
    state = dict(s_m=torch.tensor([positions], dtype=torch.float32), lane=torch.tensor([lanes]),
        v_mps=torch.tensor([speeds], dtype=torch.float32), a_mps2=torch.zeros(1, n),
        length_m=torch.full((1, n), 5.), present=torch.tensor([present]),
        is_cav=torch.zeros(1, n, dtype=torch.bool), exit_s_m=553.33)
    model = WmSModel(ModelSpec(), physics=PhysicsConfig(merge_model='sumo')).eval()
    process = TorchEntryProcessV2(history=EntranceHistory.defaults(),
        cfg=EntryConfig(conserve_queue=True, insert_rule='sumo_like_v2'),
        row_seeds=[7], couple_draws=True)
    return state, model, process


def test_departure_uses_moved_leader_and_does_not_move_on_birth_tick():
    base = ENTRY_S_M[2]+5+POSITION_EPS
    state, model, entries = setup([base+6, 0.], [2, 0], [10., 0.], [True, False])
    entries._emit_due = lambda b: [1]
    with torch.no_grad():
        out = step_after_movement(model, state, entries)
    born = out['source_birth_mask']
    assert born.tolist() == [[False, True]]  # pre-movement gap was only1m
    assert out['positions'][0, 0, 1] == pytest.approx(base, abs=.0002)
    assert out['positions'][0, 0, 0] > state['s_m'][0, 0]
    assert out['accel'][born[:, None]].item() == 0
    assert out['source_before']['present'].tolist() == [[True, True]]
    assert out['history_mask'][0, 1].sum() == 1
    assert entries._source_ledgers[0, 1].queue == 0
    entries._emit_due = lambda b: []
    with torch.no_grad():
        later = step_after_movement(model, out['state'], entries, history=out['history'], history_mask=out['history_mask'])
    assert later['positions'][0, 0, 1] > out['positions'][0, 0, 1]


def test_exit_and_birth_cannot_overwrite_one_identity_in_same_frame():
    state, model, entries = setup([553., 0.], [1, 0], [20., 0.], [True, False])
    entries._emit_due = lambda b: [0]
    with torch.no_grad():
        out = step_after_movement(model, state, entries)
    assert out['source_exit_mask'].tolist() == [[True, False]]
    assert out['source_birth_mask'].tolist() == [[False, True]]
    assert out['masks'].tolist() == [[[False, True]]]
    assert entries.pool is None
    assert not entries.virgin[0, 0]  # observed before the initial vehicle exited
    assert out['source_before']['v_mps'][0, 0] == 20.


def test_original_padding_pool_is_preserved_and_demand_not_discarded():
    state, model, entries = setup([553., 0.], [1, 0], [20., 0.], [True, False])
    original = np.array([True, False])
    entries.pool = original
    entries._emit_due = lambda b: [0]
    with torch.no_grad():
        out = step_after_movement(model, state, entries)
    assert not out['source_birth_mask'].any()
    assert entries.pool is original
    assert entries.pending[0, 0] == 1 and entries.stats[0].dropped_pending == 0
    assert entries._source_ledgers[0, 0].queue == 1


def test_default_legacy_rollout_is_not_switched_implicitly():
    state, _, entries = setup([100., 0.], [1, 0], [20., 0.], [True, False])
    model = WmSModel(ModelSpec(), physics=PhysicsConfig()).eval()
    with pytest.raises(ValueError, match='merge_model=sumo'):
        step_after_movement(model, state, entries)
    assert entries.now[0] == 0.


def test_no_arrival_is_bit_exact_to_the_existing_physics_step():
    state, model, entries = setup([10., 100.], [2, 1], [5., 20.], [True, True])
    entries._emit_due = lambda b: []
    with torch.no_grad():
        ordinary = model.rollout(state, n_steps=1)
        phased = step_after_movement(model, state, entries)
    for key in ('positions', 'speeds', 'lanes', 'masks', 'accel', 'history',
                'history_mask', 'sections', 'entrant', 'entrant_seq'):
        torch.testing.assert_close(phased[key], ordinary[key], rtol=0, atol=0)


def test_post_movement_birth_keeps_admission_and_motion_settings_coupled():
    from types import SimpleNamespace
    from tools.rbeval.encoded_fleet import EncodedPointBirths
    from wms.model.driver_encoder import PARAMETER_NAMES
    state, model, entries = setup([100., 0.], [1, 0], [10., 0.], [True, False])
    state['vehicle_id'] = torch.tensor([[10, -1]])
    point = torch.tensor([32., 1.5, 3., 1.8, 2.5, .2, .1, 6., 1.])
    original = point.clone()
    original[0] = 27.
    encoded = SimpleNamespace(population=point, defaults=point,
        settings=lambda ids, cav: {name: original[i].expand_as(ids).float()
                                  for i, name in enumerate(PARAMETER_NAMES)})
    sampler = EncodedPointBirths(encoded)
    initial = sampler.initial(state)
    entries._emit_due = lambda b: [1]
    with torch.no_grad():
        out = step_after_movement(model, state, entries, driver_params=initial,
                                   entrant_driver_sampler=sampler)
    born = out['source_birth_mask']
    assert born.sum() == 1
    assert out['driver_params_after_insertion']['v0'][born].item() == 32.
    assert out['driver_params_after_insertion']['v0'][~born].item() == 27.
    assert sampler.births_checked == 1
    assert entries.last_birth_ordinals == {(0, 1): 0}


def test_new_cav_is_not_controlled_on_its_insertion_tick():
    state, model, entries = setup([0.], [0], [0.], [False])
    entries.histories[0].cav_share[:] = 1.
    entries._emit_due = lambda b: [1]
    seen = []
    def controller(current):
        seen.append((current['present'] & current['is_cav']).clone())
        return torch.stack((current['s_m'], current['lane'], torch.zeros_like(current['v_mps']),
                            torch.full_like(current['v_mps'], -5.)), -1)
    with torch.no_grad():
        out = step_after_movement(model, state, entries, cav_controller=controller)
    assert len(seen) == 1 and not seen[0].any()
    assert out['state']['is_cav'].item()
    assert out['speeds'].item() == pytest.approx(22.22, abs=1e-5)


@pytest.mark.parametrize('mode,params,message', [
    ('residual', {'v0': torch.full((1, 2), 30.)}, 'identity-coupled entrant sampler'),
    ('driver_params', None, 'pre-bound settings')])
def test_unbound_newborn_settings_cannot_silently_reuse_departed_driver(mode, params, message):
    state, _, entries = setup([100., 0.], [1, 0], [10., 0.], [True, False])
    model = WmSModel(ModelSpec(rollout_mode=mode), physics=PhysicsConfig(merge_model='sumo')).eval()
    with pytest.raises(ValueError, match=message):
        step_after_movement(model, state, entries, driver_params=params)
    assert entries.now[0] == 0.
