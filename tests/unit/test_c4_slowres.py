"""SLOWRES: learned slow temporal residual state (CPU, tiny random WM-S)."""
from __future__ import annotations

import torch

from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.temporal_slow import slow_residual_step
from wms.model.wms import WmSModel


def _model(**spec_kwargs):
    torch.manual_seed(11)
    return WmSModel(
        ModelSpec(rollout_mode="hybrid", history_hidden=16, k_hidden=8, **spec_kwargs),
        hdv=HdvConfig(d_model=16, n_heads=2, n_layers=2, ffn=32)).eval()


def _state(s, v, lane, cav):
    k = len(s)
    return {"s_m": torch.tensor([s], dtype=torch.float32),
            "v_mps": torch.tensor([v], dtype=torch.float32),
            "a_mps2": torch.zeros(1, k), "lane": torch.tensor([lane], dtype=torch.long),
            "present": torch.ones(1, k, dtype=torch.bool),
            "is_cav": torch.tensor([cav]), "length_m": torch.full((1, k), 5.0)}


def test_default_is_direct_and_has_no_slow_parameters():
    model = _model()
    assert model.temporal_residual_mode == "direct"
    assert model.temporal_tau_raw is None
    assert not any(name.startswith("temporal_tau") for name in model.state_dict())
    x = _state((-600.0, -560.0, -500.0), (15.0, 10.0, 20.0), (1, 1, 0),
               (False, False, False))
    first = model.rollout(dict(x), n_steps=4)
    again = model.rollout(dict(x), n_steps=4)
    for key in ("positions", "speeds", "accel", "lanes", "masks"):
        assert torch.equal(first[key], again[key]), key
    assert "temporal_residual_mps2" not in first["state"]


def test_slow_state_is_carried_across_truncated_chunks():
    model = _model(temporal_residual_mode="slow", temporal_tau_init_s=0.5)
    x = _state((-600.0, -560.0, -500.0), (15.0, 10.0, 20.0), (1, 1, 0),
               (False, False, False))
    whole = model.rollout(dict(x), n_steps=4)
    chunked = model.rollout(dict(x), n_steps=4, truncation_steps=2)
    assert "temporal_residual_mps2" in whole["state"]
    assert torch.equal(whole["state"]["temporal_residual_mps2"].detach(),
                       chunked["state"]["temporal_residual_mps2"].detach())
    assert torch.isfinite(chunked["state"]["temporal_residual_mps2"]).all()
    # Two steps alone end in a different state: the carry is not reset per chunk.
    short = model.rollout(dict(x), n_steps=2)
    assert not torch.equal(short["state"]["temporal_residual_mps2"].detach(),
                           whole["state"]["temporal_residual_mps2"].detach())


def test_tau_is_learnable_with_finite_gradients_over_two_steps():
    model = _model(temporal_residual_mode="slow", temporal_tau_init_s=0.2)
    model.train()
    x = _state((-600.0, -560.0, -500.0), (15.0, 10.0, 20.0), (1, 1, 0),
               (False, False, False))
    optimizer = torch.optim.Adam([model.temporal_tau_raw], lr=1.0e-2)
    before = float(model.temporal_tau_raw.detach())
    for _ in range(2):
        optimizer.zero_grad()
        roll = model.rollout(dict(x), n_steps=2)
        loss = (roll["accel"] ** 2).mean()
        loss.backward()
        grad = model.temporal_tau_raw.grad
        assert grad is not None and torch.isfinite(grad).all() and float(grad.abs()) > 0.0
        optimizer.step()
    after = float(model.temporal_tau_raw.detach())
    assert torch.isfinite(torch.tensor(after)) and after != before


def test_low_pass_removes_single_step_pulses_in_a_synthetic_sequence():
    target = torch.tensor([[0.0, -8.0, 8.0, -8.0, 0.0]])
    previous = torch.zeros_like(target[:, 0])
    tau = torch.tensor([2.0])
    values = []
    for step in range(target.shape[1]):
        previous = slow_residual_step(target[:, step], previous, tau, 0.5)
        values.append(previous)
    residual = torch.stack(values, dim=1)
    target_jumps = (target[:, 1:] - target[:, :-1]).abs().max()
    residual_jumps = (residual[:, 1:] - residual[:, :-1]).abs().max()
    assert residual_jumps < 0.5 * target_jumps
    # A one-step target pulse leaves a smaller one-step residual pulse.
    assert (residual[:, 1] - residual[:, 0]).abs() < 0.5 * (target[:, 1] - target[:, 0]).abs()
    # A tiny tau is close to the direct path (identity-at-load option).
    fast = slow_residual_step(target[:, 1], previous=torch.zeros(1), tau_s=0.05, dt_s=0.5)
    assert (fast - target[:, 1]).abs().max() < 0.15 * target[:, 1].abs().max()
