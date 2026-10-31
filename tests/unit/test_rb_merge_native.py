"""Native means CF/LC delegation; regression for the FIT-observed ramp-head trap."""
import copy

import pytest
import torch

from wms.model.cav import EXEC_CAP_KEY, WM_MERGE_KEY
from wms.model.spec import ModelSpec
from wms.model.sumo_native import native_physics_controller
from wms.model.wms import WmSModel
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.physics.config import PhysicsConfig
from wms.physics.state import state_kin


def ramp_head():
    return dict(s_m=torch.tensor([[-1.46431577]]), v_mps=torch.zeros(1, 1),
                a_mps2=torch.zeros(1, 1), length_m=torch.full((1, 1), 5.),
                lane=torch.full((1, 1), 2), present=torch.ones(1, 1, dtype=torch.bool),
                is_cav=torch.ones(1, 1, dtype=torch.bool))


def test_physics_native_releases_unled_cav_before_merge_zone():
    # Position/speed are the observable front-ramp trace in FIT969400. No
    # per-driver labels or future tape feed-back enter this regression fixture.
    from tools.rbeval.full_fleet import native_controller
    model = WmSModel(ModelSpec(rollout_mode='driver_params'),
                     physics=PhysicsConfig(merge_model='sumo', sumo_fidelity=True)).eval()
    model.cav_exec_cap = 'idm'
    state = ramp_head()
    # The first packed slot is an UNRELATED mainline car. The frozen head's
    # no-leader feature gathers this speed despite has_leader=False; a singleton
    # fixture misses the resulting artificial stop (dv spuriously becomes-31).
    state = {k: v.repeat(1, 2) for k, v in state.items()}
    state['s_m'][0, 0] = -100.
    state['v_mps'][0, 0] = 31.
    state['lane'][0, 0] = 0
    state['is_cav'][0, 0] = False
    defaults = model._default_driver_parameters(state_kin(state), 1, 2)
    params = {n: defaults[..., i] for i, n in enumerate(PARAMETER_NAMES)}
    with torch.no_grad():
        physics = model.forward(copy.deepcopy(state), driver_params=params)
        old = model.rollout(copy.deepcopy(state), n_steps=20, cav_controller=native_controller(), driver_params=params)
        new = model.rollout(copy.deepcopy(state), n_steps=20, cav_controller=native_physics_controller(), driver_params=params)
    assert physics['physics_accel_mps2'][0, 1] > 0
    assert old['positions'][0, -1, 1] < 0 and old['speeds'][0, -1, 1] == 0
    assert new['positions'][0, -1, 1] > 0
    assert new['lanes'][0, -1, 1] == 1


def test_native_delegates_merge_and_preserves_safety_ceiling():
    state = ramp_head()
    state['v_mps'].fill_(1.)
    state[EXEC_CAP_KEY] = torch.full((1, 1), -10.)
    output = native_physics_controller()(state)
    assert output[0, 0, 2] == 0
    assert output[0, 0, 3] == -2.
    assert state[WM_MERGE_KEY].item()
    assert native_physics_controller().native_lane_mask(state).item()


def test_native_requires_physics_ceiling_and_refuses_nonfinite():
    state = ramp_head()
    with pytest.raises(KeyError):
        native_physics_controller()(state)
    state[EXEC_CAP_KEY] = torch.full((1, 1), float('nan'))
    with pytest.raises(ValueError, match='nonfinite'):
        native_physics_controller()(state)
