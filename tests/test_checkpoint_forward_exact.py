"""checkpoint_forward recomputes the pure per-step forward in backward: values and ALL gradients must be unchanged
(with the first-passage straight-through adjoint kept, merge_pivot_state_st=True)."""
from dataclasses import replace

import pytest
import torch

from wms.model import ModelSpec, WmSModel
from wms.physics.config import PhysicsConfig


def _model(st=True):
    torch.manual_seed(5)
    spec = ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5", lane_decision="sample",
                     merge_decision="first_passage", merge_passage_law="hybrid", merge_quadrature_order=8,
                     merge_pivot_state_st=st)
    return WmSModel(spec, physics=PhysicsConfig())


def _state():
    s = torch.tensor([[30.0, 49.0, 10.0, 60.0]])
    return {"s_m": s, "v_mps": torch.tensor([[6.0, 5.0, 7.0, 8.0]]), "a_mps2": torch.zeros_like(s),
            "length_m": torch.full_like(s, 5.0), "lane": torch.tensor([[2, 1, 1, 2]]),
            "present": torch.ones_like(s, dtype=torch.bool), "is_cav": torch.zeros_like(s, dtype=torch.bool)}


def _run(model, forward_ckpt):
    model.train()
    model.checkpoint_forward = forward_ckpt
    model.zero_grad(set_to_none=True)
    torch.manual_seed(11)
    # perturb the (zero-initialised) heads so every parameter has a nonzero adjoint
    with torch.no_grad():
        g = torch.Generator().manual_seed(3)
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    roll = model.rollout(_state(), n_steps=8, lane_uniform=torch.full((1, 8, 4), 0.3))
    loss = (roll["positions"].square().sum() * 1e-4 + roll["speeds"].sum() * 1e-2 + roll["accel"].sum()
            + roll["merge_pivot_c"].clamp(-50, 50).sum())
    loss.backward()
    return float(loss.detach()), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("st", [True, False])
def test_checkpoint_forward_is_exact(st):
    base, ckpt = _model(st), _model(st)
    ckpt.load_state_dict(base.state_dict())
    l0, g0 = _run(base, False)
    l1, g1 = _run(ckpt, True)
    assert l0 == l1
    assert g0.keys() == g1.keys() and len(g0) > 10
    worst = 0.0
    for n in g0:
        scale = g0[n].abs().max().clamp_min(1e-30)
        worst = max(worst, float((g0[n] - g1[n]).abs().max() / scale))
    assert worst <= 1e-8, worst
    assert any(g0[n].abs().sum() > 0 for n in g0 if n.startswith("merge_pivot."))
