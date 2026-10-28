import torch

from tests.unit.test_longmem import scene
from wms.control.bridge import make_cav_controller
from wms.control.contract import REPORT_FEATURES
from wms.control.counterfactual import native_fn
from wms.control.obs import CommandMemory, build_observations
from wms.control.policy import CavPolicy, PolicyExecutor, RsuGroupEncoder
from wms.control.safety import safe_accel
from wms.control.train_cli import GroupSignal
from wms.model.cav import WM_LANE_TARGET_KEY


def test_residual_training_and_deployment_suppress_command_intent_and_age():
    torch.manual_seed(191)
    state = scene(batch=1, slots=4)
    state['cmd_accel_mps2'] = torch.ones_like(state['s_m'])
    state['lane_intent'] = torch.ones_like(state['s_m'])
    state[WM_LANE_TARGET_KEY] = state['lane'].clone()
    policy = CavPolicy(lane_head=False, residual_on_native=True, accel_max=1.2)
    group = GroupSignal(RsuGroupEncoder())
    captured = []
    hook = policy.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    bridge = make_cav_controller(policy, group, safe_accel, native=native_fn('idm', a_max=1.2),
                                 native_lane_change=True, native_ramp_merge='wm_law')
    bridge({**state, 'physics_step_index': 0})
    memory = CommandMemory()
    memory.record(state, torch.full_like(state['s_m'], .7), torch.ones_like(state['s_m']))
    executor = PolicyExecutor(policy, group, memory=memory)
    executor(state)
    hook.remove()
    assert len(captured) == 2
    for obs in captured:
        assert obs.report_mask.any()
        for name in ('cmd_accel_mps2', 'lane_intent', 'age_s'):
            assert torch.count_nonzero(obs.reports[..., REPORT_FEATURES.index(name)]) == 0
    torch.testing.assert_close(captured[0].reports, captured[1].reports, rtol=0, atol=0)
    # Builder still supports explicit report features/ages for other controllers.
    other = build_observations(state, report_age_s=.5)
    for name in ('cmd_accel_mps2', 'lane_intent', 'age_s'):
        assert torch.count_nonzero(other.reports[..., REPORT_FEATURES.index(name)]) > 0
