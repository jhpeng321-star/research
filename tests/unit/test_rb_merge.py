import math

import numpy as np
import pytest
import torch

from wms.model.entries import EntryConfig, EntranceHistory, TorchEntryProcessV2
from wms.model.spec import ModelSpec
from wms.model.sumo_insertion import SumoInsertion, insertion_speed
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.physics.junction_merge import apply_merge, merge_geometry, merge_speed_advice, secure_gap
from wms.physics.state import state_kin


def state(positions, lanes, speeds=None):
    n = len(positions)
    return dict(s_m=torch.tensor([positions], dtype=torch.float32),
                v_mps=torch.tensor([speeds or [0.] * n], dtype=torch.float32),
                a_mps2=torch.zeros(1, n), lane=torch.tensor([lanes]),
                length_m=torch.full((1, n), 5.), present=torch.ones(1, n, dtype=torch.bool),
                is_cav=torch.zeros(1, n, dtype=torch.bool))


def test_idm_secure_gap_is_not_squared_speed_braking_rule():
    v = torch.tensor(20.)
    assert secure_gap(v, torch.tensor(15.), torch.tensor(2.), torch.tensor(4.), torch.tensor(1.)).item() == pytest.approx(20 + 100 / (2 * math.sqrt(8)))
    assert secure_gap(torch.tensor(5.), torch.tensor(30.), torch.tensor(2.), torch.tensor(4.), torch.tensor(1.)) == 0


def test_two_sided_gaps_and_no_stopped_relaxation():
    cfg = PhysicsConfig(merge_model='sumo')
    s = state([73., 85., 65.], [2, 1, 1])
    kin = state_kin(s)
    assert merge_geometry(kin, s['present'], cfg)['safe'][0, 0]
    s['s_m'][0, 2] = 68.  # zero rear clearance
    assert not merge_geometry(state_kin(s), s['present'], cfg)['safe'][0, 0]
    s['s_m'][0, 2] = 73.  # equal fronts must count as overlap
    assert merge_geometry(state_kin(s), s['present'], cfg)['overlap'][0, 0]


def test_follower_uses_own_inferred_headway():
    cfg = PhysicsConfig(merge_model='sumo')
    s = state([60., 30.], [2, 1], [10., 10.])
    params = {'time_headway': torch.tensor([[1., 5.]])}
    assert merge_geometry(state_kin(s), s['present'], cfg)['safe'][0, 0]
    assert not merge_geometry(state_kin(s), s['present'], cfg, params)['safe'][0, 0]


def test_sequential_mergers_cannot_take_same_gap_and_teacher_is_preserved():
    cfg = PhysicsConfig(merge_model='sumo')
    s = state([74., 73.], [2, 2])
    deciders = s['present'].clone()
    chosen = apply_merge(state_kin(s), s['present'], s['lane'], deciders, cfg)
    assert chosen.tolist() == [[1, 2]]
    deciders[0, 0] = False
    proposed = torch.tensor([[1, 2]])
    chosen = apply_merge(state_kin(s), s['present'], proposed, deciders, cfg)
    assert chosen.tolist() == [[1, 2]]


def test_lane_end_stop_and_no_blanket_mainline_yield():
    cfg = PhysicsConfig(merge_model='sumo')
    s = state([75.44, 150., -100.], [2, 1, 1], [3., 20., 20.])
    accel, _ = merge_speed_advice(state_kin(s), s['present'], torch.zeros(1, 3), cfg)
    assert accel[0, 0] == -6.
    assert accel[0, 1] == 0 and accel[0, 2] == 0


def test_lane_end_does_not_get_a_fictitious_leader_min_gap():
    cfg = PhysicsConfig(merge_model='sumo')
    model = WmSModel(ModelSpec(), physics=cfg).eval()
    s = state([74.], [2])
    with torch.no_grad():
        out = model.forward(s)
    assert out['accel_mps2'][0, 0] > 0  # IDM stopSpeed fallback permits creeping to end


def test_sumo_forward_bypasses_every_learned_head_and_sampling(monkeypatch):
    cfg = PhysicsConfig(merge_model='sumo')
    model = WmSModel(ModelSpec(), physics=cfg).eval()
    def forbidden(*args, **kwargs):
        raise AssertionError('learned motion executed')
    monkeypatch.setattr(model.residual_net, 'forward', forbidden)
    monkeypatch.setattr(model, '_temporal_delta', forbidden)
    s = state([74., 90., 60.], [2, 1, 1])
    with torch.no_grad():
        out = model.step(s, accel_noise=torch.ones(1, 3))
    assert not model._samples_lanes
    assert out['next_lane'][0, 0] == 1
    assert torch.equal(out['raw_residual_mps2'], torch.zeros(1, 3))
    assert torch.isfinite(out['next_s_m']).all()


def test_insertion_idm_exact_far_speed_and_collision_rejection():
    p = dict(accel=2., decel=4., time_headway=1., min_gap=2., v0=30.)
    assert insertion_speed(20., 1000., 10., p, .5) == 20.
    assert insertion_speed(20., 1.99, 10., p, .5) is None
    assert insertion_speed(20., 8., 0., p, .5) < 20.


def test_existing_learned_lane_checkpoint_shape_can_roll_out_in_sumo_mode():
    model = WmSModel(ModelSpec(lane_model='hazard_v5', rollout_mode='driver_params'),
                    physics=PhysicsConfig(merge_model='sumo')).eval()
    s = state([70., 100., 40.], [2, 1, 1])
    with torch.no_grad():
        result = model.rollout(s, n_steps=120)
    assert torch.isfinite(result['positions']).all()
    assert torch.isfinite(result['speeds']).all()


def test_exact_source_conservation_and_no_same_lane_double_birth():
    cfg = PhysicsConfig(merge_model='sumo')
    process = TorchEntryProcessV2(history=EntranceHistory.defaults(),
        cfg=EntryConfig(conserve_queue=True, insert_rule='sumo_like_v2', sumo_insertion=SumoInsertion(cfg)),
        row_seeds=[17], couple_draws=True)
    s = state([0.] * 8, [0] * 8)
    s['present'].fill_(False)
    process._emit_due = lambda b: [1, 1, 1]
    born = process.insert(s)
    assert born.sum() == 1
    assert process.pending[0, 1] == 2
    assert process.stats[0].dropped_pending == 0
    assert process.last_birth_ordinals == {(0, 0): 0}
    ledger = process._source_ledgers[0, 1]
    assert (ledger.due, ledger.inserted, ledger.queue) == (3, 1, 2)
    assert s['v_mps'][0, 0] == pytest.approx(22.22)


def test_given_depart_speed_refuses_unsafe_instead_of_patching():
    cfg = PhysicsConfig(merge_model='sumo')
    process = TorchEntryProcessV2(history=EntranceHistory.defaults(),
        cfg=EntryConfig(conserve_queue=True, insert_rule='sumo_like_v2',
                        sumo_insertion=SumoInsertion(cfg, depart_speed=20.)),
        row_seeds=[17], couple_draws=True)
    s = state([0., float(process.entry_s[2]) + 5. + 5. + 3.], [0, 2])
    s['present'][0, 0] = False
    process._emit_due = lambda b: [1]
    assert process.insert(s).sum() == 0
    assert process.pending[0, 1] == 1
