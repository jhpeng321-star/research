"""Closed-loop CAV callbacks run the frozen controllers through ``wms.sim_bridge``."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.train import cav_drive

pytest.importorskip("traffic_shepherd.damping")


def state(n=4):
    s = torch.tensor([[0.0, 30.0, 60.0, 90.0][:n]]) - 200.0
    return {"s_m": s, "v_mps": torch.full((1, n), 12.0), "a_mps2": torch.zeros(1, n),
            "lane": torch.zeros(1, n, dtype=torch.long), "length_m": torch.full((1, n), 5.0),
            "present": torch.ones(1, n, dtype=torch.bool),
            "is_cav": torch.tensor([[False, True, True, True][:n]])}


def timeline(controllers):
    ids = np.asarray(controllers, dtype=np.int64)[None, None, :].repeat(3, axis=1)
    return {"controller": ids, "acceleration_mps2": np.full(ids.shape, -1.0, np.float32),
            "target_speed_mps": np.full(ids.shape, 8.0, np.float32)}


def test_dependencies_come_from_the_bridge_and_pi_is_loaded_only_on_demand():
    damping_only = cav_drive.load_controller_dependencies(np.array([0, 1]))
    assert damping_only[5] is None and damping_only[1].__name__ == "CirclesController"
    with_pi = cav_drive.load_controller_dependencies(np.array([0, 2]))
    assert with_pi[5].__name__ == "PIExternalU" and with_pi[7].__name__ == "PISaturationParameters"


def test_native_damping_pi_and_direct_slots_are_driven():
    torch.manual_seed(0)
    model = WmSModel(ModelSpec(), physics=PhysicsConfig())
    controllers = cav_drive.TimelineControllers(model, timeline([0, 1, 2, 3]))
    s = state()
    native = model.physics_accel(s).detach()
    out = controllers(s)
    assert out.shape == (1, 4, 4) and torch.isfinite(out).all()
    accel = out[..., 3]
    assert torch.isclose(accel[0, 0], native[0, 0])                  # slot 0 stays native
    # direct: a request of -1 m/s2 (and speed target 8 m/s) can only lower the native acceleration
    assert accel[0, 3] <= native[0, 3] + 1e-6
    controllers(s)                                                   # second physics tick
    assert controllers.step_index == 2
