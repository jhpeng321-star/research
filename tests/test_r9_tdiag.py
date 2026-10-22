"""R9TDIAG: training-path contract fixes found by the real-batch diagnosis."""
import pytest
import torch

from test_merge_pivot_training import _fixture
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner


def _runner(model, **extra):
    settings = TrainSettings(rollout_mode="hybrid", lane_model="hazard_v5", merge_decision="first_passage",
                             merge_core_tau_reg_weight=.1, truncation_steps=1, **extra)
    return UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                        projection=None, device=torch.device("cpu"))


def _batch(inp):
    return dict(states=inp.states, present=inp.present, is_cav=torch.zeros_like(inp.is_cav))


def test_differentiable_unroll_term_with_all_inputs_frozen_does_not_abort_the_backward():
    """Round 9 trains only merge_pivot but inherits differentiable unroll terms (lane_event / stability).

    Their value has no autograd graph (all inputs frozen), so ``value.backward()`` raised "element 0 of
    tensors does not require grad" at the end of EVERY batch, after the chunk gradients were already
    accumulated.  The empty backward is skipped; the merge_pivot gradient is untouched.
    """
    model, inp, _ = _fixture()
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("merge_pivot"))
    runner = _runner(model, stability_weight=.5)
    assert "stability" in runner.composer.names and runner.composer.differentiable("stability")
    result = runner.run(_batch(inp), unroll_steps=4, backward=True)
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert result["stability"] is not None
    grad = model.merge_pivot.tau_raw.grad
    assert grad is not None and grad > 0
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("merge_pivot"))


def test_differentiable_unroll_term_with_trainable_inputs_is_still_backpropagated():
    model, inp, _ = _fixture()
    runner = _runner(model, stability_weight=.5)
    runner.run(_batch(inp), unroll_steps=4, backward=True)
    stab = [p.grad for n, p in model.named_parameters() if n.startswith("residual_net") and p.grad is not None]
    assert stab and any(bool(g.abs().sum() > 0) for g in stab)
