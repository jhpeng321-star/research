from __future__ import annotations

import torch

from output.wd01.wm_n1.wms.lane_head import HistoryLaneHead, balanced_lane_loss
from output.wd01.wm_n1.wms.model import PhysicsConfig, WmSModel
from output.wd01.wm_n1.wms.train import WmSTrainConfig


def test_lane_head_shape_and_hard_feasibility_mask():
    head = HistoryLaneHead(history_dim=5)
    embedding = torch.zeros(2, 4, 5)
    context = torch.zeros(2, 4, 23)
    feasible = torch.tensor([[[1, 0, 0]] * 4, [[1, 1, 1]] * 4], dtype=torch.bool)
    logits = head(embedding, context, feasible)
    assert logits.shape == (2, 4, 3)
    assert torch.all(logits[0, :, 1:] < -1.0e30)
    assert torch.isfinite(logits[1]).all()


def test_history_fixture_learns_driver_specific_lane_event():
    torch.manual_seed(17)
    head = HistoryLaneHead(history_dim=4, hidden=12)
    history = torch.tensor([[-1.0, 0, 0, 0], [1.0, 0, 0, 0]]).repeat(32, 1)
    context = torch.zeros(64, 23)
    target = torch.where(history[:, 0] < 0, 1, 0)
    feasible = torch.ones(64, 3, dtype=torch.bool)
    opt = torch.optim.Adam(head.parameters(), lr=0.03)
    for _ in range(60):
        opt.zero_grad()
        loss = balanced_lane_loss(head(history, context, feasible), target,
                                  torch.ones(64, dtype=torch.bool))
        loss.backward()
        opt.step()
    prediction = head(history, context, feasible).argmax(-1)
    assert (prediction == target).float().mean() > 0.95


def test_balanced_loss_ignores_hard_masked_targets():
    fill = torch.finfo(torch.float32).min
    logits = torch.stack((torch.tensor([0.0, 1.0, fill]),      # left is feasible
                          torch.tensor([0.0, fill, fill])))     # left is masked
    mask = torch.ones(2, dtype=torch.bool)
    mixed = balanced_lane_loss(logits, torch.tensor([1, 1]), mask)
    assert torch.isfinite(mixed) and float(mixed) > 0.0
    all_masked = balanced_lane_loss(logits, torch.tensor([2, 2]), mask)
    assert torch.isfinite(all_masked) and float(all_masked) == 0.0


def test_lane_head_off_retains_default_physics_logits_bit_exactly():
    torch.manual_seed(23)
    implicit = WmSModel(physics=PhysicsConfig(max_gap_steps=8),
                        rollout_mode="driver_params")
    explicit = WmSModel(physics=PhysicsConfig(max_gap_steps=8),
                        rollout_mode="driver_params", lane_head="off")
    explicit.load_state_dict(implicit.state_dict())
    state = {
        "s_m": torch.tensor([[0.0, 20.0]]),
        "lane": torch.tensor([[0, 1]]),
        "v_mps": torch.tensor([[12.0, 11.0]]),
        "a_mps2": torch.zeros(1, 2),
        "length_m": torch.full((1, 2), 5.0),
        "present": torch.ones(1, 2, dtype=torch.bool),
        "is_cav": torch.zeros(1, 2, dtype=torch.bool),
    }
    ordinary = implicit.rollout(state, n_steps=3)
    explicit_off = explicit.rollout(state, n_steps=3)
    for key in ("lane_logits", "lanes", "positions", "speeds"):
        torch.testing.assert_close(ordinary[key], explicit_off[key], rtol=0, atol=0)


def test_history_lane_head_requires_driver_params_mode():
    try:
        WmSTrainConfig(lane_head="history")
    except ValueError as exc:
        assert "driver_params" in str(exc)
    else:
        raise AssertionError("history lane head must require driver_params mode")


def test_opt_in_model_produces_relative_action_logits():
    model = WmSModel(physics=PhysicsConfig(max_gap_steps=8),
                     rollout_mode="driver_params", lane_head="history")
    lane = torch.tensor([[1, 1, 0]])
    one_hot = torch.nn.functional.one_hot(lane, 3).float()
    kin = torch.cat((torch.tensor([[[0.0], [20.0], [40.0]]]), one_hot,
                     torch.full((1, 3, 1), 12.0), torch.zeros(1, 3, 2)), -1)
    state = {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
             "a_mps2": kin[..., 5], "length_m": kin[..., 6], "lane": lane,
             "present": torch.ones(1, 3, dtype=torch.bool),
             "is_cav": torch.zeros(1, 3, dtype=torch.bool)}
    history = torch.zeros(1, 3, 10, 6)
    history[..., 3] = 1.0
    mask = torch.ones(1, 3, 10, dtype=torch.bool)
    out = model.rollout(state, n_steps=2, driver_history=history,
                        driver_history_mask=mask)
    assert out["lane_logits"].shape == (1, 2, 3, 3)
    assert out["lanes"].shape == (1, 2, 3)
    assert out["lane_probabilities"].shape == (1, 2, 3, 3)
    torch.testing.assert_close(out["lane_probabilities"].sum(-1),
                               torch.ones(1, 2, 3))


def test_physics_context_masks_change_into_unsafe_gap():
    model = WmSModel(physics=PhysicsConfig(max_gap_steps=8),
                     rollout_mode="driver_params", lane_head="history")
    lane = torch.tensor([[0, 1]])
    one_hot = torch.nn.functional.one_hot(lane, 3).float()
    kin = torch.cat((torch.tensor([[[0.0], [3.0]]]), one_hot,
                     torch.full((1, 2, 1), 12.0), torch.zeros(1, 2, 2)), -1)
    state = {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
             "a_mps2": kin[..., 5], "length_m": kin[..., 6], "lane": lane,
             "present": torch.ones(1, 2, dtype=torch.bool)}
    baseline = model.mobil_merge_logits(state)
    _, safe = model._lane_context(kin, state["present"], baseline, None,
                                  torch.full((1, 2), 999.0))
    # Vehicle in lane 0 would change right into the lane-1 leader only 3 m ahead.
    assert not bool(safe[0, 0, 2])


def test_mandatory_merge_overrides_dwell_when_safe_before_lane_end():
    model = WmSModel(physics=PhysicsConfig(max_gap_steps=8),
                     rollout_mode="driver_params", lane_head="history")
    lane = torch.tensor([[2, 1, 1]])
    positions = torch.tensor([[60.0, 100.0, 20.0]])
    one_hot = torch.nn.functional.one_hot(lane, 3).float()
    kin = torch.cat((positions.unsqueeze(-1), one_hot,
                     torch.full((1, 3, 1), 12.0), torch.zeros(1, 3, 2)), -1)
    state = {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
             "a_mps2": kin[..., 5], "length_m": kin[..., 6], "lane": lane,
             "present": torch.ones(1, 3, dtype=torch.bool)}
    baseline = model.mobil_merge_logits(state)
    _, safe = model._lane_context(kin, state["present"], baseline, None,
                                  torch.zeros(1, 3))
    # Merge-lane actor's last-safe lane-1 move overrides the minimum dwell.
    assert bool(safe[0, 0, 1])
    assert not bool(safe[0, 0, 0])
