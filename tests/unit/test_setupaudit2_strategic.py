"""Mode-513 strategic ramp fallback: CPU only, no SUMO or data files."""
import copy
from unittest.mock import patch

import pytest
import torch

from tests.unit.test_loop14c import (ScriptedLanes, group, identity_safety, ramp_scene,
                                    pivot_model, fixed_forward, wm_scene)
from wms.control.bridge import make_cav_controller
from wms.control.lanes import LaneRules, LANE_LEFT
from wms.control.policy14 import make_controller_factory
from wms.model.cav import WM_LANE_KEY, WM_LANE_TARGET_KEY, WM_MERGE_KEY


def controller(choice=0, **extra):
    return make_cav_controller(ScriptedLanes([choice], accel=.3), group, identity_safety,
        lane_rules=LaneRules(), native=lambda s: torch.zeros_like(s['s_m']),
        native_lane_change=True, native_ramp_merge='wm_law', policy_ramp_merge=True, **extra)


def test_stay_delegates_strategic_ramp_but_keeps_absolute_longitudinal_authority():
    st = ramp_scene()
    st[WM_LANE_TARGET_KEY] = st['lane'].clone()
    cb = controller(policy_strategic_merge=True)
    assert not cb.native_lane_mask(st).any()  # request-dependent fallback is post-callback.
    out = cb(st)
    assert st[WM_MERGE_KEY].tolist() == [[True, False, False]]
    assert out[0, 0, 1] == 2 and float(out[0, 0, 3]) == pytest.approx(.3)


def test_explicit_ramp_crossing_and_mainline_policy_lanes_keep_ownership():
    st = ramp_scene()
    st[WM_LANE_TARGET_KEY] = st['lane'].clone()
    out = controller(LANE_LEFT, policy_strategic_merge=True)(st)
    assert out[0, 0, 1] == 1 and not st[WM_MERGE_KEY][0, 0]
    st = ramp_scene()
    st['lane'][:] = 1
    st[WM_LANE_TARGET_KEY] = st['lane'].clone()
    cb = controller(LANE_LEFT, policy_strategic_merge=True)
    assert not cb.native_lane_mask(st).any()
    out = cb(st)
    assert out[0, 0, 1] == 0


@pytest.mark.parametrize('hybrid', [False, True])
def test_fallback_reaches_wm_first_passage_but_default_stay_has_no_merge_path(hybrid):
    m = pivot_model(hybrid)
    outputs = []
    for flag in (False, True):
        st = wm_scene(2., True)
        torch.manual_seed(5)
        with torch.no_grad(), patch.object(m, 'forward', side_effect=fixed_forward):
            outputs.append(m.step(st, cav_controller=controller(policy_strategic_merge=flag),
                                  lane_uniform=torch.full_like(st['s_m'], .6)))
    assert not outputs[0][WM_MERGE_KEY][0, 0]
    assert outputs[1][WM_MERGE_KEY][0, 0]
    assert outputs[1]['merge_pivot_eligible'][0, 0]


def test_default_false_bitwise_motion_rng_and_factory_metadata():
    outputs, rngs = [], []
    for extra in ({}, {'policy_strategic_merge': False}):
        st = ramp_scene()
        st[WM_LANE_TARGET_KEY] = st['lane'].clone()
        torch.manual_seed(5)
        outputs.append(controller(**extra)(copy.deepcopy(st)))
        rngs.append(torch.get_rng_state())
    assert torch.equal(*outputs) and torch.equal(*rngs)
    legacy = make_controller_factory(True, 'wm_law', True)
    assert 'policy_strategic_merge' not in legacy.ramp_merge
    enabled = make_controller_factory(True, 'wm_law', True, policy_strategic_merge=True)
    assert enabled.ramp_merge['policy_strategic_merge'] is True
    with pytest.raises(ValueError, match='boolean'):
        controller(policy_strategic_merge=1)
    with pytest.raises(ValueError, match='requires'):
        make_controller_factory(policy_strategic_merge=True)


def test_full_wm_explicit_crossing_is_not_overwritten_by_strategic_pivot():
    m = pivot_model(False)
    st = wm_scene(2., True)

    def forward(state, **kw):
        out = fixed_forward(state, **kw)
        native = state.get(WM_LANE_KEY, torch.zeros_like(state['present']))
        assert not native[0, 0]  # policy request must be known before any ramp delegation.
        out['lane_decider_mask'] = state['present'] & (~state['is_cav'] | native)
        return out

    with torch.no_grad(), patch.object(m, 'forward', side_effect=forward):
        out = m.step(st, cav_controller=controller(LANE_LEFT, policy_strategic_merge=True),
                     lane_uniform=torch.full_like(st['s_m'], .6))
    assert not out[WM_MERGE_KEY][0, 0]
    assert out['next_lane'][0, 0] == 1
