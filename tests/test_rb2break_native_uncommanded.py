import copy

import pytest
import torch

from tests.unit.test_rb_merge import state
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.model.sumo_native import native_physics_controller
from wms.physics.config import PhysicsConfig


def model():
    return WmSModel(ModelSpec(rollout_mode='driver_params', lane_model='lc2013', lane_head='off'),
        physics=PhysicsConfig(cf_model='sumo_idm', merge_model='sumo')).eval()


def settings(s):
    values = dict(v0=31.29, time_headway=1.2, min_gap=2., accel=1.2, decel=2.,
                  politeness=.2, mobil_threshold=.3, merge_gap=7., merge_urgency=1.)
    return {name:torch.full_like(s['s_m'], values[name]) for name in PARAMETER_NAMES}


def callback(uncommanded):
    ticks = []
    def call(s):
        ticks.append(s['physics_step_index'])
        v = s['v_mps']+.5  # deliberately DIFFERENT from native CF: must be ignored only where delegated
        return torch.stack((s['s_m']+.5*v, s['lane'], v, torch.ones_like(v)), -1)
    call.native_lane_mask = lambda s: s['present'].bool() & s['is_cav'].bool()
    if uncommanded:
        def mask(s):
            result = torch.zeros_like(s['present'])
            result[0] = s['present'][0] & s['is_cav'][0]
            return result
        call.native_uncommanded_mask = mask
    return call, ticks


@pytest.mark.parametrize('positions,speeds,advice,lanes', [([10., 23.], [20., 0.], False, [1, 1]),
    ([-200., -100.], [20., 20.], True, [1, 1]), ([30., 65.], [10., 10.], False, [2, 1])])
def test_uncontrolled_cav_is_identical_to_hdv_clone_including_lc_speed_patch(positions, speeds, advice, lanes):
    m = model()
    original = state(positions, lanes, speeds)
    if advice:
        original.update(lc2013_present=original['present'].clone(), lc2013_lane=original['lane'].clone(),
            lc2013_advice_accel=torch.tensor([[-1., 0.]]),
            lc2013_has_advice=torch.tensor([[True, False]]), lc2013_advice_weight=torch.ones(1, 2))
    native = copy.deepcopy(original)
    native['is_cav'][0, 0] = True
    cb, ticks = callback(True)
    with torch.no_grad():
        hdv = m.step(original, driver_params=settings(original))
        cav = m.step(native, driver_params=settings(native), cav_controller=cb)
    assert ticks == [0]
    for name in ('next_s_m', 'next_v_mps', 'next_lane', 'next_present', 'accel_mps2'):
        torch.testing.assert_close(cav[name], hdv[name], rtol=0, atol=0)


def test_commanded_batch_row_is_bit_exact_and_dispatcher_still_runs_once():
    m = model()
    s = state([-100., 0.], [1, 1], [20., 20.])
    s = {name:value.expand(2, *value.shape[1:]).clone() for name,value in s.items()}
    s['is_cav'][:, 0] = True
    old_cb, old_ticks = callback(False)
    new_cb, new_ticks = callback(True)
    with torch.no_grad():
        old = m.step(copy.deepcopy(s), driver_params=settings(s), cav_controller=old_cb)
        new = m.step(copy.deepcopy(s), driver_params=settings(s), cav_controller=new_cb)
    assert old_ticks == [0] and new_ticks == [0]
    for name in ('next_s_m', 'next_v_mps', 'next_lane', 'next_present', 'accel_mps2'):
        torch.testing.assert_close(new[name][1], old[name][1], rtol=0, atol=0)
    assert not torch.equal(new['next_v_mps'][0], old['next_v_mps'][0])


def test_standalone_uncommanded_callback_needs_no_external_execution_cap():
    m = model()
    s = state([-200.], [1], [20.])
    s['is_cav'][:] = True
    with torch.no_grad():
        result = m.step(s, driver_params=settings(s),
                        cav_controller=native_physics_controller(uncommanded=True))
    assert result['next_v_mps'][0, 0] > 20.
