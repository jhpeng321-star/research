from __future__ import annotations

import json

import numpy as np
import torch

from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.plan_steps import plan
from output.wd01.wm_n1.wms.train import (TrajectoryDataset, WmSTrainConfig,
                                         WmSTrainer, collate)
from output.wd01.wm_n1.wms.traffloss import (TrafficLossConfig, entrant_steps,
                                             section_metrics, section_weights,
                                             soft_stopped, traffic_bin_loss)
from hdv.config import HdvConfig


def small_model(**kwargs):
    return WmSModel(physics=PhysicsConfig(max_gap_steps=8),
                    hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16),
                    **kwargs)


def state(batch=2, slots=4):
    lane = torch.zeros(batch, slots, dtype=torch.long)
    lane[:, slots // 2:] = 1
    one_hot = torch.nn.functional.one_hot(lane, 3).float()
    kin = torch.cat((torch.arange(slots).float().view(1, slots, 1).expand(batch, -1, -1),
                     one_hot, torch.full((batch, slots, 1), 12.0),
                     torch.zeros(batch, slots, 2)), dim=-1)
    return {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
            "a_mps2": kin[..., 5], "length_m": kin[..., 6], "lane": lane,
            "present": torch.ones(batch, slots, dtype=torch.bool),
            "is_cav": torch.zeros(batch, slots, dtype=torch.bool)}


def _states(n=1, t1=21, slots=4):
    base = state(batch=n, slots=slots)["veh_kin"].numpy()
    states = np.repeat(base[:, None], t1, axis=1)
    states[..., 0] += np.arange(t1, dtype=np.float32)[None, :, None] * 6.0
    states[..., 4] = 12.0
    present = np.ones((n, t1, slots), dtype=bool)
    return TrajectoryDataset.from_arrays({"states": states, "present": present})


def test_section_weights_hard_assignment_matches_eval_contract():
    s = torch.tensor([-10.0, 5.0, 40.0, 100.0])
    w = section_weights(s, bounds=(0.0, 75.44), temp_m=0.1)
    names = ["upstream", "merge", "merge", "downstream"]
    order = ["upstream", "merge", "downstream"]
    hard = [order[int(i)] for i in w.argmax(-1)]
    assert hard == names
    torch.testing.assert_close(w.sum(-1), torch.ones(4), atol=1e-5, rtol=1e-5)


def test_soft_stopped_agrees_with_hard_count_within_tolerance():
    v = torch.tensor([0.1, 0.4, 0.6, 20.0])
    cfg = TrafficLossConfig(stop_speed_mps=0.5, stop_temp_mps=0.02)
    soft = soft_stopped(v, cfg=cfg)
    hard = (v < 0.5).float()
    assert abs(float(soft.sum()) - float(hard.sum())) < 0.05
    # deterministic, finite, and monotone in speed
    assert torch.isfinite(soft).all()
    assert float(soft[0]) > float(soft[1]) > float(soft[2])


def test_section_metrics_soft_counts_match_hard_counts():
    s = torch.tensor([[-10.0, 40.0, 100.0, 40.0]])
    v = torch.tensor([[0.2, 0.1, 20.0, 12.0]])
    present = torch.ones_like(s, dtype=torch.bool)
    cfg = TrafficLossConfig(section_temp_m=0.1, stop_temp_mps=0.02)
    met = section_metrics(s, v, present, cfg=cfg)
    up = met["upstream"]
    assert abs(float(up["density"]) - 1.0 / 1.0) < 0.02
    assert abs(float(up["stopped_vehicles"]) - 1.0) < 0.05
    merge = met["merge"]
    assert abs(float(merge["density"]) - 2.0 / 0.07544) < 1.0
    assert abs(float(merge["stopped_vehicles"]) - 1.0) < 0.05
    assert abs(float(merge["mean_speed"]) - (0.1 + 12.0) / 2.0) < 0.2
    assert float(merge["merge_queue_length"]) >= 0.0


def test_traffic_loss_zero_when_prediction_equals_target():
    torch.manual_seed(0)
    cfg = TrafficLossConfig(section_temp_m=0.5)
    s = torch.rand(1, 8, 5) * 120.0
    v = torch.rand(1, 8, 5) * 20.0
    present = torch.ones(1, 8, 5, dtype=torch.bool)
    idx = torch.arange(8)
    loss, diag = traffic_bin_loss(s, v, present, s.clone(), v.clone(),
                                  present.clone(), step_indices=idx, cfg=cfg)
    assert float(loss) < 1e-8
    assert all(value < 1e-6 for value in diag.values())


def test_traffic_loss_gradients_finite_and_positive_when_different():
    torch.manual_seed(0)
    cfg = TrafficLossConfig()
    pred_s = (torch.rand(2, 70, 6) * 120.0).requires_grad_(True)
    pred_v = (torch.rand(2, 70, 6) * 20.0).requires_grad_(True)
    present = torch.ones(2, 70, 6, dtype=torch.bool)
    tgt_s = (torch.rand(2, 70, 6) * 120.0)
    tgt_v = (torch.rand(2, 70, 6) * 20.0)
    idx = torch.arange(70)
    loss, _ = traffic_bin_loss(pred_s, pred_v, present, tgt_s, tgt_v, present.clone(),
                               step_indices=idx, cfg=cfg)
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert torch.isfinite(pred_s.grad).all()
    assert torch.isfinite(pred_v.grad).all()
    assert float(pred_s.grad.abs().sum()) > 0.0
    # 70 steps at 0.5 s spans two 30 s bins (60 + 10 steps).
    assert cfg.bin_steps == 60


def test_traffic_schedule_ramps_with_horizon():
    cfg = TrafficLossConfig(ramp_start_s=30.0, ramp_end_s=120.0)
    # step index -> seconds = index * dt_s (0.5 s): 0/30/60/120/300 s.
    w = cfg.schedule(torch.tensor([0, 60, 120, 240, 600]))
    assert float(w[0]) == 0.0
    assert float(w[1]) == 0.0
    assert abs(float(w[2]) - 1.0 / 3.0) < 1e-6
    assert float(w[3]) == 1.0
    assert float(w[4]) == 1.0


def test_entrant_steps_slicing_and_validation():
    mask = torch.zeros(2, 9, 4, dtype=torch.bool)
    assert entrant_steps(None, 8, 8, "cpu") is None
    three = entrant_steps(mask, 8, 8, "cpu")
    assert three.shape == (2, 8, 4)
    four = entrant_steps(mask[:, :8], 8, 7, "cpu")
    assert four.shape == (2, 8, 4)
    try:
        entrant_steps(torch.zeros(2, 3, 4, dtype=torch.bool), 8, 8, "cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("bad entrant shape should raise")


def test_rollout_state_noise_default_off_is_identical_and_on_is_finite():
    model = small_model()
    current = state()
    torch.manual_seed(3)
    base = model.rollout(current, n_steps=6)
    torch.manual_seed(3)
    same = model.rollout(current, n_steps=6, state_noise_std_m=0.0, state_noise_std_mps=0.0)
    torch.testing.assert_close(base["positions"], same["positions"])
    torch.testing.assert_close(base["speeds"], same["speeds"])

    torch.manual_seed(4)
    noisy = model.rollout(current, n_steps=6, state_noise_std_m=1.0,
                          state_noise_std_mps=0.5)
    assert not torch.allclose(base["positions"], noisy["positions"])
    (noisy["positions"].sum() + noisy["speeds"].sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_trainer_traffic_loss_integration_and_entrant_exclusion():
    dataset = _states(n=2, t1=41, slots=4)
    cfg = WmSTrainConfig(device="cpu", threads=4, batch_size=2,
                         unroll_curriculum_s=(20.0,), traffic_loss=True,
                         traffic_weight=0.5, traffic_ramp_start_s=0.0,
                         traffic_ramp_end_s=1.0)
    trainer = WmSTrainer(small_model(), cfg)
    batch = collate([dataset[0], dataset[1]])
    rec = trainer.train_batch(batch, unroll_steps=40)
    assert rec["finite"] is True
    assert np.isfinite(rec["loss"])
    assert "traffic" in rec

    # An entrant mask removes that slot from the per-car supervised count but
    # the same batch still yields a finite loss (traffic side keeps counting).
    entrant = torch.zeros_like(batch["present"])
    entrant[:, 1:, 0] = True
    base_valid = trainer._forward_backward(batch, unroll_steps=40,
                                           backward=False)["valid_targets"]
    with_entrant = trainer._forward_backward({**batch, "entrant": entrant},
                                             unroll_steps=40,
                                             backward=False)["valid_targets"]
    assert with_entrant < base_valid


def test_traffic_loss_off_keeps_previous_terms():
    dataset = _states(n=1, t1=21, slots=4)
    cfg = WmSTrainConfig(device="cpu", threads=4, batch_size=1,
                         unroll_curriculum_s=(10.0,))
    assert cfg.traffic_loss is False
    trainer = WmSTrainer(small_model(), cfg)
    batch = collate([dataset[0]])
    rec = trainer._forward_backward(batch, unroll_steps=20, backward=False)
    assert "traffic" not in rec
    assert rec["valid_targets"] > 0


def test_plan_steps_supports_300s_stage(tmp_path):
    log = tmp_path / "train_log.jsonl"
    rows = [{"seconds": 1.0} for _ in range(20)]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    steps, info = plan(log, budget_s=3600.0)
    assert info["stage_steps"] == steps
    assert info["cost_per_unit_sim_steps"] == 290.0        # frozen default
    five, info5 = plan(log, budget_s=3600.0, stages_s="10,30,60,120,300")
    assert len(five) == 5
    assert info5["cost_per_unit_sim_steps"] == 20 + 60 + 90 + 120 + 150
    assert all(s >= 1 for s in five)
    try:
        plan(log, budget_s=3600.0, stages_s="10,30", ratios="1")
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched ratios should raise")
