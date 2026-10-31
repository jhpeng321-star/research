"""Cross-worker contracts for LC2013 + SUMO merge, rather than isolated kernels."""
import torch

from wms.model.factory import build_model
from wms.model.spec import ModelSpec
from wms.physics.config import PhysicsConfig
from tools.rbint.audit import forbid_learned_corrections


def model():
    return build_model(ModelSpec(lane_model='lc2013', lane_head='off',
                                 residual_accel_off=True, temporal_accel_off=True),
                       PhysicsConfig(merge_model='sumo')).eval()


def state():
    # Mainline ego is on the real upstream corridor, with ample continuation.
    # The old +1000m fixture was beyond the 553.33m road exit and relied on
    # the former infinite-route approximation to accumulate keep-right wishes.
    return dict(s_m=torch.tensor([[-1000., 74., 73.]]),
                v_mps=torch.tensor([[25., 0., 0.]]),
                a_mps2=torch.zeros(1, 3), lane=torch.tensor([[0, 2, 2]]),
                length_m=torch.full((1, 3), 5.),
                present=torch.ones(1, 3, dtype=torch.bool),
                is_cav=torch.zeros(1, 3, dtype=torch.bool),
                vehicle_id=torch.tensor([[11, 12, 13]]))


@torch.no_grad()
def test_mainline_lc2013_and_ramp_reservations_both_execute():
    m = model()
    with forbid_learned_corrections(m):
        first = m.step(state())
        full = m.rollout(state(), n_steps=120)
    assert first['next_lane'].tolist() == [[0, 1, 2]]
    # Finite continuation reduces the source keep-right accumulator increment;
    # allow a full minute rather than asserting the old infinite-road timing.
    assert (full['lanes'][0, :, 0] == 1).any()
    assert 'lc2013_keep_right' in full['state']
    assert torch.isfinite(full['positions']).all()
    assert torch.isfinite(full['speeds']).all()


@torch.no_grad()
def test_combined_rollout_chunk_state_is_exact():
    m = model()
    full = m.rollout(state(), n_steps=24)
    first = m.rollout(state(), n_steps=12)
    second = m.rollout(first['state'], n_steps=12,
                       history=first['history'], history_mask=first['history_mask'])
    for key in ('lanes', 'positions', 'speeds'):
        torch.testing.assert_close(full[key][:, 12:], second[key], rtol=0, atol=0)
    assert full['state']['present'].sum() == 3
