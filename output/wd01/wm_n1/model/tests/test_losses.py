"""Loss contract tests (design section 3)."""
from __future__ import annotations

import torch

from model import constants as C
from model import losses as L
from model.net import WmN1Net
from synth_model import make_batch, make_step_targets, small_cfg


def test_gmm_nll_lower_when_target_matches_mean():
    target = torch.zeros(1, 2)
    mu = torch.zeros(1, 2, C.N_MIX)
    ls = torch.zeros(1, 2, C.N_MIX)
    ml = torch.zeros(1, 2, C.N_MIX)
    mask = torch.ones(1, 2, dtype=torch.bool)
    good = L.gmm_accel_nll(target, mu, ls, ml, mask)
    bad = L.gmm_accel_nll(target + 20.0, mu, ls, ml, mask)
    assert good < bad


def test_lane_change_nll_ignores_none_and_padding():
    logits = torch.randn(1, 3, 3)
    lane = torch.tensor([[C.LANE_STAY, C.LANE_NONE, C.LANE_LEFT_CLS]])
    mask = torch.tensor([[True, True, False]])
    loss = L.lane_change_nll(lane, logits, mask)
    assert torch.isfinite(loss)


def test_neg_binom_and_arrival_loss_finite():
    count = torch.tensor([[0.0, 2.0], [1.0, 0.0]])
    mu = torch.full((2, 2), 0.7)
    log_alpha = torch.zeros(2, 2)
    v = L.neg_binom_nll(count, mu, log_alpha)
    assert torch.isfinite(v) and v >= 0


def test_pinball_zero_at_exact_quantile():
    levels = torch.tensor([0.1, 0.5, 0.9])
    q = torch.tensor([[1.0, 1.0, 1.0]])
    y = torch.tensor([1.0])
    assert float(L.pinball(q, y, levels).mean()) == 0.0


def test_outcome_pinball_finite_and_masked():
    q = torch.randn(2, C.N_TAU, C.N_OUTCOME_TARGETS, C.N_QUANTILES)
    y = torch.randn(2, C.N_TAU, C.N_OUTCOME_TARGETS)
    mask = torch.ones(2, C.N_TAU, dtype=torch.bool)
    assert torch.isfinite(L.outcome_pinball(q, y, mask))
    assert torch.isfinite(L.outcome_crps(q, y, mask))


def test_contrast_loss_zero_for_identical_predictions_and_outcome():
    q = torch.randn(2, C.N_TAU, C.N_OUTCOME_TARGETS, C.N_QUANTILES)
    y = torch.randn(2, C.N_TAU, C.N_OUTCOME_TARGETS)
    loss = L.contrast_loss(q, q.clone(), y, y.clone())
    assert float(loss) == 0.0


def test_contrast_loss_prefers_correct_sign():
    # one target, all quantile levels; y_a - y_b = 1.0 > 0
    qa_correct = torch.full((1, C.N_TAU, C.N_OUTCOME_TARGETS,
                             C.N_QUANTILES), 1.0)
    qa_wrong = torch.full((1, C.N_TAU, C.N_OUTCOME_TARGETS,
                           C.N_QUANTILES), -1.0)
    qb = torch.zeros(1, C.N_TAU, C.N_OUTCOME_TARGETS, C.N_QUANTILES)
    ya = torch.full((1, C.N_TAU, C.N_OUTCOME_TARGETS), 1.0)
    yb = torch.zeros(1, C.N_TAU, C.N_OUTCOME_TARGETS)
    good = L.contrast_loss(qa_correct, qb, ya, yb)
    bad = L.contrast_loss(qa_wrong, qb, ya, yb)
    assert float(good) < float(bad)


def test_total_loss_backward_has_gradients():
    net = WmN1Net(small_cfg()).train()
    b = make_batch(B=2, K=8)
    out = net(**b)
    targets = make_step_targets(b)
    loss = L.total_loss(out, targets)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
