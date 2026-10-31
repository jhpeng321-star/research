"""Boundary clocks, recycled identity/origin and real entry/LC chunk contracts."""
import copy

import numpy as np
import pytest
import torch

from tools.rbeval.full_fleet import ObservedEntries, make_state, road_delay
from wms.model.entries import EntryConfig, EntranceQueue, TorchEntryProcessV2


def fixture():
    kin = np.zeros((4, 7), dtype=np.float32)
    kin[:, 6] = 5
    kin[0, [0, 2, 4]] = [50, 1, 10]  # ramp-origin already merged
    state = make_state(kin, np.array([1, 0, 0, 0], bool), np.arange(4),
                       np.zeros(4, bool), ['ramp.0', 'mainline.0', 'ramp.1', 'mainline.1'], 4)
    cfg = EntryConfig(conserve_queue=True, insert_rule='sumo_like_v2')
    process = TorchEntryProcessV2(np.zeros((2, 30)), cfg=cfg, row_seeds=[1], couple_draws=True,
        queues=[EntranceQueue(np.array([0, 1]), np.zeros(2), cav_share=np.zeros(2))])
    return state, process


def test_reported_boundary_excludes_birth_and_preserves_exit_speed():
    before = dict(present=np.array([1, 1, 1, 0], bool), v_mps=np.array([0., 12., 3., 0.]))
    after = dict(present=np.array([1, 1, 0, 0], bool), v_mps=np.array([5., 6., 20., 0.]))
    charge = road_delay(before, after, np.array([0, 1, 0, 0], bool), .5)
    np.testing.assert_allclose(charge, [.5*(1-5/31.29), 0, .5*(1-3/31.29), 0])


def test_origin_survives_merge_and_recycled_slot_gets_fresh_identity():
    state, process = fixture()
    entries = ObservedEntries(process, state)
    first = entries.insert(state)
    assert entries.origin[0]  # entrance identity, not current lane
    assert first.sum() == 1
    slot = int(torch.where(first[0])[0][0])
    old = int(entries.ids[slot])
    assert old < -1 and entries.origin[slot]
    state['present'][0, slot] = False
    process.pending[0, 0] = 1
    entries.ledgers[0].advance(1, 0)
    entries.insert(state)
    assert int(entries.ids[slot]) != old and not entries.origin[slot]
    assert entries.ledgers[1].queue == 0


def test_fractional_due_waiting_and_source_conservation():
    state, process = fixture()
    process.pending[:] = 0
    entries = ObservedEntries(process, state)
    process._started[:] = True
    process.t_next[:] = [.25, np.inf]
    entries.insert(state)
    np.testing.assert_allclose(entries.source, [.25, 0.])
    assert entries.ledgers[0].due == 1
    assert entries.ledgers[0].inserted == 1
    assert entries.ledgers[0].queue == 0


def test_observer_detects_external_queue_corruption():
    state, process = fixture()
    entries = ObservedEntries(process, state)
    process.pending[0, 1] += 1
    with pytest.raises((AssertionError, ValueError), match='source ledger|available demand'):
        entries.insert(state)


def test_integrated_chunk_memory_matches_long_rollout():
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    from wms.physics.config import PhysicsConfig
    from tools.rbeval.full_fleet import native_controller
    # This test requires the exact peer integration runtime selected by PYTHONPATH.
    model = WmSModel(ModelSpec(lane_model='lc2013', lane_head='off'),
                     physics=PhysicsConfig(merge_model='sumo', sumo_fidelity=True)).eval()
    model.cav_exec_cap = 'idm'
    state, _ = fixture()
    state['lane'][0, 0] = 2
    state['s_m'][0, 0] = 20
    state['is_cav'][0, 0] = True
    with torch.no_grad():
        full = model.rollout(copy.deepcopy(state), n_steps=4, cav_controller=native_controller())
        last = copy.deepcopy(state)
        history = history_mask = None
        for tick in range(4):
            one = model.rollout(last, n_steps=1, k0=tick, cav_controller=native_controller(),
                                history=history, history_mask=history_mask)
            last = one['state']
            history, history_mask = one['history'], one['history_mask']
    for key in ('s_m', 'v_mps', 'lane', 'present'):
        torch.testing.assert_close(last[key], full['state'][key], rtol=0, atol=0)
    for key in full['state']:
        if key.startswith('lc2013_'):
            torch.testing.assert_close(last[key], full['state'][key], rtol=0, atol=0)
