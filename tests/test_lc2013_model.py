"""LC2013 runtime switch, CAV contract and chunked persistent-state integration."""
from dataclasses import replace

import pytest
import torch

from wms.model.factory import build_model
from wms.model.spec import ModelSpec
from wms.physics.config import PhysicsConfig


def state(lane=0, s=0., speed=25.):
    x = torch.tensor([[s]])
    return dict(s_m=x, v_mps=torch.full_like(x, speed),
                a_mps2=torch.zeros_like(x), lane=torch.full_like(x, lane).long(),
                length_m=torch.full_like(x, 5.), present=torch.ones_like(x).bool(),
                is_cav=torch.zeros_like(x).bool(), vehicle_id=torch.tensor([[11]]))


def model():
    return build_model(ModelSpec(lane_model='lc2013', residual_accel_off=True,
                                  temporal_accel_off=True), PhysicsConfig()).eval()


def test_default_and_switch_validation():
    assert ModelSpec().lane_model == 'residual'
    with pytest.raises(ValueError, match='structured'):
        ModelSpec(lane_model='lc2013', black_box=True)
    with pytest.raises(ValueError, match='structured'):
        ModelSpec(lane_model='lc2013', lane_head='history')
    with pytest.raises(ValueError):
        ModelSpec(lane_model='lc2013', merge_decision='first_passage')


@torch.no_grad()
def test_lane_logits_override_residual_and_infeasible_baseline():
    m = model()
    s = state(lane=2, s=40., speed=5.)
    out = m.step(s, baseline_lane_logits=torch.tensor([[[-1e6, -1e6, 1e6]]]))
    assert out['next_lane'].item() == 1
    assert out['lane_logits'].argmax(-1).item() == 1
    assert out['lc2013_wish'].item()


@torch.no_grad()
def test_chunked_rollout_preserves_wishes_and_adds_no_weights():
    m = model()
    torch.manual_seed(12)
    default = build_model(replace(m.spec, lane_model='residual'), m.physics).eval()
    assert m.state_dict().keys() == default.state_dict().keys()
    full = m.rollout(state(), n_steps=12)
    first = m.rollout(state(), n_steps=6)
    second = m.rollout(first['state'], n_steps=6,
                       history=first['history'], history_mask=first['history_mask'])
    torch.testing.assert_close(full['lanes'][:, 6:], second['lanes'], rtol=0, atol=0)
    torch.testing.assert_close(full['positions'][:, 6:], second['positions'], rtol=0, atol=0)
    assert 'lc2013_keep_right' in second['state']


@torch.no_grad()
def test_teacher_cav_lane_is_preserved():
    m = model()
    s = state(lane=2, s=40., speed=5.)
    s['is_cav'][:] = True
    teacher = torch.tensor([[[42.5, 2., 5., 0.]]])
    out = m.step(s, cav_teacher=teacher)
    assert out['next_lane'].item() == 2


@torch.no_grad()
def test_right_overtake_advice_is_carried_into_next_longitudinal_update():
    m = model()
    s = dict(s_m=torch.tensor([[0., 100.]]), v_mps=torch.tensor([[30., 20.]]),
             a_mps2=torch.zeros(1, 2), lane=torch.tensor([[1, 0]]),
             length_m=torch.full((1, 2), 5.), present=torch.ones(1, 2).bool(),
             is_cav=torch.zeros(1, 2).bool(), vehicle_id=torch.tensor([[11, 12]]))
    first = m(s)
    assert first['lc2013_own_has_advice'][0, 0]
    from wms.physics.lc2013 import STATE_KEYS
    carried = {**s, **{key: first[key] for key in STATE_KEYS if key in first}}
    second = m(carried)
    assert second['accel_mps2'][0, 0] <= first['lc2013_own_advice_accel'][0, 0] + 1e-6


@torch.no_grad()
def test_upstream_lane_change_preserves_sumo_local_position_and_translates_model_s():
    from wms.physics.lc2013 import lane_change, STATE_KEYS
    m = model()
    s = state(lane=0, s=-1000., speed=5.)
    from wms.physics.state import state_kin
    memory = lane_change(state_kin(s), s['present'], m.physics, identity=s['vehicle_id'])
    memory['lc2013_keep_right'][:] = -40_000_000
    s.update({key: memory[key] for key in STATE_KEYS if key in memory})
    out = m.step(s)
    assert out['next_lane'].item() == 1
    # Longitudinal update plus the coordinate-origin translation only.
    assert out['lc2013_position_shift'].item() == pytest.approx(1.45, abs=1e-4)
    assert out['next_s_m'].item() == pytest.approx(
        -1000 + .5 * (5 + out['next_v_mps'].item()) * m.physics.dt_s + 1.45, abs=1e-4)


@torch.no_grad()
def test_fixed_upstream_teacher_position_is_not_translated_twice():
    m = model()
    s = state(lane=0, s=-1000., speed=5.)
    s['is_cav'][:] = True
    teacher = torch.tensor([[[-996.05, 1., 5., 0.]]])
    out = m.step(s, cav_teacher=teacher)
    assert out['next_lane'].item() == 1
    assert out['next_s_m'].item() == teacher[0, 0, 0].item()
    assert out['lc2013_position_shift'].item() == 0


def test_rejected_front_move_restores_its_origin_before_rear_move_is_admitted():
    from wms.model.lc2013_adapter import resolve_shifted_lanes
    # First pass: front mover is rejected by the fixed right-lane body. Its
    # hypothetical+1.45m origin shift appears to leave0.45m for the rear mover.
    # Restoring the front mover's true origin would create a1m overlap, so the
    # fixed-point pass must reject the rear move as well.
    s = torch.tensor([[-100., -102.55, -95.]], dtype=torch.float64)
    source = torch.tensor([[0, 1, 1]])
    proposed = torch.tensor([[1, 0, 1]])
    lanes, positions = resolve_shifted_lanes(s, torch.full_like(s, 5.), source,
        proposed, torch.ones(1, 3).bool(), torch.tensor([[True, True, False]]), PhysicsConfig())
    assert torch.equal(lanes, source)
    torch.testing.assert_close(positions, s, rtol=0, atol=0)
