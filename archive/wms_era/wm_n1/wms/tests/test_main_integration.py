"""wms/main integration parity: action timeline x rollout modes x lane head.

The switch branch (action timeline, direct commands, closed-loop CAV
callbacks) was merged onto the hybrid/lane-head branch.  These tests pin that
the new timeline input is inert when it encodes a constant fixed controller in
every rollout mode, that direct commands still override CAVs in the hybrid +
lane-head path, and that structured checkpoints keep their weight layout.
"""
from __future__ import annotations

import pytest
import torch

from output.wd01.wm_n1.hdv.config import HdvConfig
from output.wd01.wm_n1.wms.action_timeline import CONTROLLERS, ActionTimeline
from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel

MODES = [("residual", "off"), ("driver_params", "off"), ("hybrid", "off"),
         ("driver_params", "history"), ("hybrid", "history")]
KEYS = ("positions", "speeds", "lanes", "masks", "accel", "lane_logits")


def _model(mode: str, lane_head: str) -> WmSModel:
    torch.manual_seed(11)
    return WmSModel(physics=PhysicsConfig(max_gap_steps=8),
                    hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16),
                    rollout_mode=mode, lane_head=lane_head).eval()


def _state(slots: int = 3):
    kin = torch.zeros(1, slots, 7)
    kin[..., 0] = torch.arange(slots, dtype=torch.float32) * 30.0
    kin[..., 1] = 1.0
    kin[..., 4] = torch.linspace(10.0, 14.0, slots)
    kin[..., 6] = 5.0
    is_cav = torch.zeros(1, slots, dtype=torch.bool)
    is_cav[0, 0] = True
    return {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
            "a_mps2": kin[..., 5], "length_m": kin[..., 6],
            "lane": kin[..., 1:4].argmax(-1),
            "present": torch.ones(1, slots, dtype=torch.bool), "is_cav": is_cav}


def _history(slots: int = 3):
    torch.manual_seed(5)
    return torch.randn(1, slots, 10, 6), torch.ones(1, slots, 10, dtype=torch.bool)


def _teacher(n_steps: int, slots: int = 3):
    teacher = torch.zeros(1, n_steps + 1, slots, 4)
    teacher[..., 0] = torch.arange(n_steps + 1, dtype=torch.float32).view(1, -1, 1) * 5.0
    teacher[..., 1] = 1.0
    teacher[..., 2] = 10.0
    return teacher


def _packed(timeline: ActionTimeline) -> dict:
    return {"controller": torch.as_tensor(timeline.controller[None]),
            "acceleration_mps2": torch.as_tensor(timeline.acceleration_mps2[None]),
            "target_speed_mps": torch.as_tensor(timeline.target_speed_mps[None]),
            "control_dt_s": timeline.control_dt_s}


@pytest.mark.parametrize("mode,lane_head", MODES)
@pytest.mark.parametrize("controller", ["native", "damping", "pi-external-u"])
def test_constant_timeline_is_bit_exact_fixed_controller_in_every_mode(
        mode, lane_head, controller):
    model = _model(mode, lane_head)
    hist, mask = _history()
    n_steps = 4
    kwargs = dict(n_steps=n_steps, cav_teacher=_teacher(n_steps),
                  driver_history=hist, driver_history_mask=mask)
    with torch.no_grad():
        legacy = model.rollout(_state(), **kwargs)
        timeline = _packed(ActionTimeline.constant(controller, 2, 3))
        got = model.rollout(_state(), action_timeline=timeline, **kwargs)
    for key in KEYS:
        torch.testing.assert_close(got[key], legacy[key], rtol=0, atol=0)


@pytest.mark.parametrize("mode,lane_head", MODES)
def test_constant_timeline_parity_with_closed_loop_callback(mode, lane_head):
    model = _model(mode, lane_head)
    hist, mask = _history()
    kwargs = dict(n_steps=4, cav_controller=model.hold_teacher,
                  driver_history=hist, driver_history_mask=mask)
    with torch.no_grad():
        legacy = model.rollout(_state(), **kwargs)
        got = model.rollout(_state(), action_timeline=_packed(
            ActionTimeline.constant("damping", 2, 3)), **kwargs)
    for key in KEYS:
        torch.testing.assert_close(got[key], legacy[key], rtol=0, atol=0)


def test_direct_command_overrides_cav_in_hybrid_lane_head_rollout():
    model = _model("hybrid", "history")
    hist, mask = _history()
    ids = torch.full((1, 2, 3), CONTROLLERS.index("native"), dtype=torch.long)
    ids[0, 1, 0] = CONTROLLERS.index("direct")
    accel = torch.zeros(1, 2, 3)
    accel[0, 1, 0] = -1.0
    target = torch.full((1, 2, 3), 30.0)
    timeline = {"controller": ids, "acceleration_mps2": accel,
                "target_speed_mps": target, "control_dt_s": 1.0}
    with torch.no_grad():
        out = model.rollout(_state(), n_steps=4, cav_teacher=_teacher(4),
                            driver_history=hist, driver_history_mask=mask,
                            action_timeline=timeline)
    torch.testing.assert_close(out["accel"][0, :2, 0], torch.zeros(2))
    torch.testing.assert_close(out["accel"][0, 2:, 0], torch.tensor([-1.0, -1.0]))
    torch.testing.assert_close(out["speeds"][0, 2:, 0], torch.tensor([9.5, 9.0]))
    assert out["lane_probabilities"].shape == (1, 4, 3, 3)
    assert "residual_magnitude" in out


def test_structured_weight_layout_unchanged_and_blackbox_gains_command_proj():
    for mode, lane_head in MODES:
        keys = set(_model(mode, lane_head).state_dict())
        assert not any(key.startswith("command_proj") for key in keys)
    black_box = WmSModel(physics=PhysicsConfig(max_gap_steps=8), black_box=True,
                         controller_embed_dim=4)
    assert "command_proj.weight" in black_box.state_dict()
