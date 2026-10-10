from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from output.wd01.wm_n1.wms.action_timeline import (
    ActionTimeline, CONTROLLERS, ControllerStateSemantics, load_timeline,
    sample_window_indices,
)
from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.train import TrajectoryDataset, collate
from output.wd01.wm_n1.wms.tests.test_wms import small_model, state


def test_switch_is_effective_at_the_exact_control_step_and_direct_is_projected():
    model = small_model()
    current = state(batch=1, slots=4)
    current["present"][:] = True
    current["is_cav"][:] = False
    current["is_cav"][0, 0] = True
    ids = np.full((2, 4), CONTROLLERS.index("native"), dtype=np.int64)
    ids[1, 0] = CONTROLLERS.index("direct")
    timeline = {"controller": torch.as_tensor(ids[None]),
                "acceleration_mps2": torch.tensor([[[0.0] * 4, [-1.0, 0.0, 0.0, 0.0]]]),
                "target_speed_mps": torch.tensor([[[30.0] * 4, [30.0, 0.0, 0.0, 0.0]]]),
                "control_dt_s": 1.0}
    teacher = torch.zeros(1, 5, 4, 4)
    teacher[..., 0] = 7.0
    teacher[..., 2] = 12.0
    teacher[..., 3] = 0.0
    out = model.rollout(current, n_steps=4, cav_teacher=teacher,
                        action_timeline=timeline)
    torch.testing.assert_close(out["accel"][0, :2, 0], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(out["accel"][0, 2:, 0], torch.tensor([-1.0, -1.0]))
    # The 1 s direct command is held for two 0.5 s physics integrations.
    torch.testing.assert_close(out["speeds"][0, 2:, 0], torch.tensor([11.5, 11.0]))


def test_constant_timeline_is_legacy_parity_and_blackbox_accepts_commands():
    model = small_model()
    current = state(batch=1, slots=4)
    current["is_cav"][0, 0] = True
    teacher = torch.zeros(1, 4, 4, 4)
    teacher[..., 0] = 5.0
    teacher[..., 2] = 10.0
    legacy = model.rollout(current, n_steps=3, cav_teacher=teacher)
    constant = ActionTimeline.constant("native", 2, 4)
    packed = {"controller": torch.as_tensor(constant.controller[None]),
              "acceleration_mps2": torch.as_tensor(constant.acceleration_mps2[None]),
              "target_speed_mps": torch.as_tensor(constant.target_speed_mps[None]),
              "control_dt_s": 1.0}
    got = model.rollout(current, n_steps=3, cav_teacher=teacher,
                        action_timeline=packed)
    for key in ("positions", "speeds", "lanes", "accel"):
        torch.testing.assert_close(got[key], legacy[key])

    black_box = WmSModel(physics=PhysicsConfig(max_gap_steps=8), black_box=True,
                         controller_embed_dim=4)
    result = black_box(current, controller_id=torch.ones(1, 4, dtype=torch.long),
                       action_command=torch.tensor([[[0.5, 20.0], [0.0, 0.0],
                                                     [0.0, 0.0], [0.0, 0.0]]]))
    assert torch.isfinite(result["accel_mps2"]).all()


def test_controller_lifecycle_retains_damping_and_resets_pi_on_reentry():
    semantics = ControllerStateSemantics()
    instances = []

    def factory(vehicle_id):
        instances.append(vehicle_id)
        return object()
    first = semantics.transition("cav-1", "damping", damping_factory=factory)
    semantics.transition("cav-1", "native", damping_factory=factory)
    resumed = semantics.transition("cav-1", "damping", damping_factory=factory)
    assert first["damping_resume"] is False
    assert resumed["damping_resume"] is True
    assert len(instances) == 1
    assert semantics.transition("cav-1", "pi-external-u")["pi_reset"] is True
    semantics.transition("cav-1", "native")
    assert semantics.transition("cav-1", "pi-external-u")["pi_reset"] is True


def test_schedule_loader_collate_and_switch_sampler(tmp_path):
    schedule = {"schema": "wms_switch_schedule/1", "control_steps": 3,
                "vehicle_ids": ["c1"], "initial": [{"vehicle_id": "c1",
                                                        "controller": "native"}],
                "switches": [{"step": 2, "vehicle_id": "c1",
                              "controller": "direct"}]}
    schedule_path = tmp_path / "switch_schedule.json"
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    command_path = tmp_path / "issued.jsonl"
    command_path.write_text(json.dumps({"step": 2, "vehicle_id": "c1",
                                        "issued_acceleration_mps2": -0.5,
                                        "issued_target_speed_mps": 20.0}) + "\n",
                            encoding="utf-8")
    timeline = load_timeline(schedule_path, command_path)
    assert timeline.controller[:, 0].tolist() == [0, 0, 3]
    assert timeline.acceleration_mps2[2, 0] == -0.5
    rng = np.random.default_rng(3)
    fixed = ActionTimeline.constant("native", 3, 1)
    selected = sample_window_indices([timeline, fixed], count=10,
                                     switch_fraction=0.6, rng=rng)
    assert sum(i == 0 for i in selected) == 6

    states = np.zeros((2, 4, 1, 7), dtype=np.float32)
    states[..., 1] = 1.0
    states[..., 4] = 10.0
    arrays = {"states": states, "schema": "wms_action_timeline/1",
              "controller_timeline": np.stack((timeline.controller,
                                                 fixed.controller)),
              "issued_acceleration_mps2": np.stack((timeline.acceleration_mps2,
                                                      fixed.acceleration_mps2)),
              "issued_target_speed_mps": np.stack((timeline.target_speed_mps,
                                                     fixed.target_speed_mps))}
    dataset = TrajectoryDataset.from_arrays(arrays)
    batch = collate([dataset[0], dataset[1]])
    assert batch["action_timeline"]["controller"].shape == (2, 3, 1)
    assert batch["action_timeline"]["control_dt_s"] == 1.0


def test_schedule_loader_rejects_duplicate_and_missing_direct_commands(tmp_path):
    schedule = {"schema": "wms_switch_schedule/1", "control_steps": 1,
                "vehicle_ids": ["c1"], "initial": [{"vehicle_id": "c1",
                                                       "controller": "direct"}]}
    schedule_path = tmp_path / "direct_schedule.json"
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    command_path = tmp_path / "direct_commands.jsonl"
    command_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="every direct timeline action"):
        load_timeline(schedule_path, command_path)
    row = {"step": 0, "vehicle_id": "c1", "issued_acceleration_mps2": 0.1,
           "issued_target_speed_mps": 20.0}
    command_path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n",
                            encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate direct command"):
        load_timeline(schedule_path, command_path)


def test_cli_cpu_train_smoke_on_synthetic_switch_fixture(tmp_path):
    from output.wd01.wm_n1.wms.cli import main

    states = np.zeros((1, 3, 2, 7), dtype=np.float32)
    states[..., 1] = 1.0
    states[..., 4] = 10.0
    states[..., 0] = np.arange(3, dtype=np.float32)[None, :, None] * 5.0
    cav = np.zeros((1, 3, 2), dtype=bool)
    cav[:, :, 0] = True
    teacher = np.zeros((1, 3, 2, 4), dtype=np.float32)
    teacher[..., 0] = states[..., 0]
    teacher[..., 2] = 10.0
    teacher[..., 3] = 0.0
    ids = np.array([[[1, 1], [6, 1], [6, 1]]], dtype=np.int64)
    pack = tmp_path / "synthetic_switch.npz"
    np.savez(pack, states=states, is_cav=cav, cav_teacher=teacher,
             controller_timeline=ids,
             issued_acceleration_mps2=np.zeros_like(ids, dtype=np.float32),
             issued_target_speed_mps=np.full_like(ids, 20.0, dtype=np.float32))
    result = main(["train", "--pack", str(pack), "--out", str(tmp_path / "run"),
                   "--device", "cpu", "--threads", "1", "--batch-size", "1",
                   "--unroll-s", "1", "--steps", "1"])
    assert result == 0
