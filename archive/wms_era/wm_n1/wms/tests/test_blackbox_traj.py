"""Contract tests for the WM-N1 strong black-box trajectory baseline.

Checks the four registered properties: output shapes, gradient flow, CAV slots
teacher-forced (HDV slots learned), and no leakage (kinematics-only input, no
IDM/MOBIL physics base).
"""
from __future__ import annotations

import numpy as np
import torch

from output.wd01.wm_n1.wms.blackbox_traj import BlackBoxTrajModel, _LearnedDynamics
from output.wd01.wm_n1.wms.model import HISTORY_F, HISTORY_H, PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.train import TrajectoryDataset, WmSTrainConfig, WmSTrainer, collate
from hdv.config import KIN_F, HdvConfig


def small_model(**kwargs) -> BlackBoxTrajModel:
    return BlackBoxTrajModel(
        physics=PhysicsConfig(max_gap_steps=8),
        hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16),
        **kwargs,
    )


def state(batch=2, slots=4):
    lane = torch.zeros(batch, slots, dtype=torch.long)
    lane[:, slots // 2:] = 1
    one_hot = torch.nn.functional.one_hot(lane, 3).float()
    kin = torch.cat((torch.arange(slots).float().view(1, slots, 1).expand(batch, -1, -1),
                     one_hot, torch.full((batch, slots, 1), 12.0),
                     torch.zeros(batch, slots, 2)), dim=-1)
    return {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
            "a_mps2": kin[..., 5], "length_m": kin[..., 6], "lane": lane,
            "present": torch.tensor([[1, 1, 1, 0]] * batch, dtype=torch.bool),
            "is_cav": torch.zeros(batch, slots, dtype=torch.bool)}


def test_subclasses_wms_interface_and_shapes():
    model = small_model()
    assert isinstance(model, WmSModel)
    out = model(state(), k_steps=0)
    assert out["accel_mps2"].shape == (2, 4)
    assert out["lane_logits"].shape == (2, 4, 3)
    assert out["learned_accel_mps2"].shape == (2, 4)
    assert torch.equal(out["gate"], torch.zeros_like(out["gate"]))


def test_k_zero_is_purely_learned_base_no_physics():
    model = small_model()
    current = state()
    out = model(state(), k_steps=0)
    base = model.residual_net(current["veh_kin"], current["present"])["accel_mean_mps2"]
    expected = base.clamp(model.physics.accel_min_mps2, model.physics.accel_max_mps2)
    torch.testing.assert_close(out["accel_mps2"], expected)
    torch.testing.assert_close(out["accel_delta_mps2"], torch.zeros_like(expected))


def test_gradients_flow_to_trunk_temporal_and_gate():
    model = small_model()
    current = state()
    history = torch.zeros(2, 4, HISTORY_H, HISTORY_F)
    history_mask = torch.zeros(2, 4, HISTORY_H, dtype=torch.bool)
    out = model(current, k_steps=torch.ones(2, 4, dtype=torch.long),
                history=history, history_mask=history_mask)
    loss = out["accel_mps2"].sum() + out["lane_logits"].sum()
    loss.backward()
    assert model.gate_increments.grad is not None
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in model.residual_net.parameters())
    assert model.temporal[0].weight.grad is not None
    assert any(p.grad is not None for p in model.history_proj.parameters())


def test_cav_slots_teacher_forced_hdv_slots_learned():
    model = small_model()
    current = state()
    current["is_cav"][:, 0] = True
    teacher = torch.zeros(2, 4, 4)
    teacher[:, 0, 0] = 8.0
    teacher[:, 0, 1] = 0.0
    teacher[:, 0, 2] = 7.0
    teacher[:, 0, 3] = -1.0
    teacher[:, 1, 0] = 1.0e6      # a value the learned base cannot reproduce
    teacher[:, 1, 1] = 0.0
    teacher[:, 1, 2] = 1.0e6
    out = model.step(current, cav_teacher=teacher)
    torch.testing.assert_close(out["next_s_m"][:, 0], torch.full((2,), 8.0))
    torch.testing.assert_close(out["next_v_mps"][:, 0], torch.full((2,), 7.0))
    assert torch.equal(out["next_lane"][:, 0], torch.zeros(2, dtype=torch.long))
    assert torch.all(out["next_s_m"][:, 1] < 1.0e5)   # HDV slot not forced
    assert torch.all(out["next_v_mps"][:, 1] < 1.0e5)


def test_rollout_teacher_forces_cav_only():
    model = small_model()
    current = state()
    n = 6
    present_seq = torch.ones(2, n + 1, 4, dtype=torch.bool)
    is_cav_seq = torch.zeros(2, n + 1, 4, dtype=torch.bool)
    is_cav_seq[:, :, 0] = True
    teacher = torch.zeros(2, n + 1, 4, 4)
    teacher[:, :, 0, 0] = torch.arange(n + 1).float().view(1, n + 1).expand(2, -1) * 10.0
    teacher[:, :, 0, 1] = 0.0
    teacher[:, :, 0, 2] = 21.0
    teacher[:, :, 0, 3] = 0.0
    out = model.rollout(current, n_steps=n, cav_teacher=teacher,
                        present_seq=present_seq, is_cav_seq=is_cav_seq)
    forced = teacher[:, 1:, 0, 0]
    torch.testing.assert_close(out["positions"][:, :, 0], forced)
    assert torch.all(out["positions"][:, :, 1] < 1.0e6)


def test_no_leakage_kinematics_only_input():
    net = _LearnedDynamics(HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16))
    kin = state()["veh_kin"]
    mask = state()["present"]
    net(kin, mask)                                   # KIN_F wide: accepted
    leaked = torch.cat((kin, torch.zeros(*kin.shape[:2], 1)), dim=-1)
    try:
        net(leaked, mask)
    except ValueError:
        pass
    else:
        raise AssertionError("a CAV-flag/action channel leaked into the HDV net")
    assert not HdvConfig().use_cav_flag
    assert KIN_F == kin.shape[-1]


def test_checkpoint_preserves_outputs():
    model = small_model()
    kin, mask = state()["veh_kin"], state()["present"]
    model.train()
    trained = model.residual_net(kin, mask)
    model.eval()
    plain = model.residual_net(kin, mask)
    torch.testing.assert_close(trained["accel_mean_mps2"], plain["accel_mean_mps2"])
    torch.testing.assert_close(trained["lane_logits"], plain["lane_logits"])


def test_no_physics_base_is_reachable():
    model = small_model()
    for method in (model.physics_accel, model.mobil_merge_logits):
        try:
            method(state())
        except NotImplementedError:
            pass
        else:
            raise AssertionError("black-box model exposed a physics base")
    try:
        BlackBoxTrajModel(physics_step=torch.nn.Identity())
    except ValueError:
        pass
    else:
        raise AssertionError("black-box model accepted a pluggable physics step")


def test_trainer_batch_is_check400_compatible():
    n, t1, slots = 2, 21, 4
    base = state(batch=n, slots=slots)["veh_kin"].numpy()
    states = np.repeat(base[:, None], t1, axis=1)
    states[..., 0] += np.arange(t1, dtype=np.float32)[None, :, None] * 6.0
    states[..., 4] = 12.0
    present = np.ones((n, t1, slots), dtype=bool)
    present[:, :, -1] = False
    dataset = TrajectoryDataset.from_arrays({"states": states, "present": present})
    cfg = WmSTrainConfig(device="cpu", threads=4, steps_per_epoch=2,
                         unroll_curriculum_s=(10.0,), batch_size=2)
    trainer = WmSTrainer(small_model(), cfg)
    batch = collate([dataset[0], dataset[1]])
    records = [trainer.train_batch(batch, unroll_steps=20) for _ in range(2)]
    assert len(records) == 2
    for rec in records:
        assert np.isfinite(rec["loss"])
        assert rec["finite"] is True
        assert rec["gate_g0"] == 0.0
        assert rec["gate_monotone"] is True
        assert rec["valid_targets"] >= 1
        assert rec["unroll_steps"] == 20
        assert all(x >= 0.0 for x in rec["gate_curve"])
