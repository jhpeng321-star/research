"""A11 synthetic contracts: full-horizon plan summary + within-decision rank loss.

No CAL/HOLDOUT/chunk-5/J4B/INC1 data is opened.  Pure functions only.
"""
from __future__ import annotations

import pytest
import torch

from model import constants as C
from model.paired_effect import (LONG_PLAN_FEATS_PER_BIN, LONG_PLAN_FIELDS,
                                 PLAN_RAW_F, PairedEffectHead,
                                 long_plan_n_bins, long_plan_raw_f,
                                 long_plan_summary, paired_effect_rank_loss)

_TAUS = C.TAUS
_NQ = C.N_QUANTILES


def test_long_plan_dims_and_refusals():
    assert long_plan_n_bins(120.0) == 12
    assert long_plan_raw_f(120.0) == 12 * LONG_PLAN_FEATS_PER_BIN == 60
    assert long_plan_raw_f(30.0) == 15
    assert long_plan_raw_f(125.0) == 65           # ceil(12.5) = 13 bins
    # the long width is never the historical 16-D plan -> unambiguous select
    assert long_plan_raw_f(120.0) != PLAN_RAW_F
    assert len(LONG_PLAN_FIELDS) == LONG_PLAN_FEATS_PER_BIN
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            long_plan_n_bins(bad)


def test_long_plan_summary_shape_determinism_and_documented_features():
    torch.manual_seed(0)
    t_steps, batch, k = 40, 3, 5
    sp = torch.rand(t_steps, batch, k) * 12.0
    ac = torch.rand(t_steps, batch, k) - 0.5
    m = (torch.rand(t_steps, batch, k) > 0.3).to(torch.float32)
    vec = long_plan_summary(sp, ac, m, horizon_s=20.0)
    assert vec.shape == (batch, long_plan_raw_f(20.0)) == (3, 10)
    # exact determinism (pure; same inputs -> identical bytes)
    assert torch.equal(vec, long_plan_summary(sp.clone(), ac.clone(),
                                              m.clone(), horizon_s=20.0))
    # bin 0 covers steps 0..19, bin 1 steps 20..39 (T = 40 = 2 bins)
    for b in range(2):
        sl = slice(b * 20, (b + 1) * 20)
        act = m[sl] > 0
        den = act.sum(dim=(0, 2)).clamp(min=1.0)             # [B]
        col = b * LONG_PLAN_FEATS_PER_BIN
        assert torch.allclose(vec[:, col + 0],
                              (sp[sl] * m[sl]).sum(dim=(0, 2)) / den,
                              atol=1e-5)
        big = torch.full_like(sp[sl], torch.finfo(sp.dtype).max)
        min_s = torch.where(act, sp[sl], big).amin(dim=(0, 2))
        min_s = torch.where(act.any(dim=0).any(dim=-1), min_s,
                            torch.zeros_like(min_s))
        assert torch.allclose(vec[:, col + 1], min_s, atol=1e-5)
        assert torch.allclose(vec[:, col + 2],
                              (ac[sl] * m[sl]).sum(dim=(0, 2)) / den,
                              atol=1e-5)
        assert torch.allclose(vec[:, col + 3],
                              ((ac[sl] < 0) & act).sum(dim=(0, 2)).float()
                              / den, atol=1e-5)
        assert torch.allclose(vec[:, col + 4],
                              act.sum(dim=(0, 2)).float() / 20.0, atol=1e-5)


def test_long_plan_summary_short_window_leaves_trailing_bins_zero():
    sp = torch.ones(5, 2, 3)
    ac = torch.zeros(5, 2, 3)
    m = torch.ones(5, 2, 3)
    vec = long_plan_summary(sp, ac, m, horizon_s=20.0)   # 2 bins, only bin 0
    assert vec.shape == (2, 10)
    assert torch.all(vec[:, LONG_PLAN_FEATS_PER_BIN:] == 0.0)
    assert torch.all(vec[:, 0] == 1.0)


def test_long_plan_summary_empty_bins_are_zero():
    sp = torch.ones(20, 1, 2)
    ac = torch.zeros(20, 1, 2)
    m = torch.zeros(20, 1, 2)
    vec = long_plan_summary(sp, ac, m, horizon_s=20.0)
    assert torch.all(vec == 0.0)


# --------------------------------------------------------------- rank loss ----
def test_rank_loss_zero_when_ordering_matches_and_large_when_reversed():
    scale = torch.ones(len(_TAUS))
    delta = torch.zeros(1, len(_TAUS))
    delta[0, 0] = -2.0                    # arm i has lower delay (better)
    valid = torch.ones_like(delta, dtype=torch.bool)
    good = torch.zeros(1, len(_TAUS))
    good[0, 0] = -10.0                    # predicted i below j -> correct
    bad = torch.zeros(1, len(_TAUS))
    bad[0, 0] = 10.0                      # predicted i above j -> wrong
    l_good = paired_effect_rank_loss(good, delta, valid, scale=scale)
    l_bad = paired_effect_rank_loss(bad, delta, valid, scale=scale)
    assert float(l_good) < 1e-3
    assert float(l_bad) > 1.0
    assert float(l_bad) > 100.0 * float(l_good)


def test_rank_loss_ties_and_invalid_are_exactly_zero_and_finite():
    scale = torch.full((len(_TAUS),), 50.0)
    ties = torch.zeros(4, len(_TAUS))
    valid = torch.ones(4, len(_TAUS), dtype=torch.bool)
    assert float(paired_effect_rank_loss(torch.randn(4, len(_TAUS)), ties,
                                         valid, scale=scale)) == 0.0
    invalid = torch.zeros(4, len(_TAUS), dtype=torch.bool)
    assert float(paired_effect_rank_loss(torch.randn(4, len(_TAUS)) * 7.0,
                                         torch.randn(4, len(_TAUS)) * 100.0,
                                         invalid, scale=scale)) == 0.0
    valid6 = torch.ones(6, len(_TAUS), dtype=torch.bool)
    mixed = paired_effect_rank_loss(torch.randn(6, len(_TAUS)) * 5.0,
                                    torch.randn(6, len(_TAUS)) * 100.0,
                                    valid6, scale=scale)
    assert torch.isfinite(mixed) and 0.0 <= float(mixed) < 5.0


def test_rank_loss_is_differentiable():
    score = torch.randn(3, len(_TAUS), requires_grad=True)
    delta = torch.randn(3, len(_TAUS))
    valid = torch.ones(3, len(_TAUS), dtype=torch.bool)
    loss = paired_effect_rank_loss(score, delta, valid,
                                   scale=torch.ones(len(_TAUS)))
    loss.backward()
    assert score.grad is not None and torch.isfinite(score.grad).all()


def test_long_plan_aware_head_returns_climatology_and_consumes_new_width():
    import numpy as np

    rng = np.random.default_rng(0)
    anchor = np.stack([np.sort(rng.normal(scale=5.0 + t, size=_NQ))
                       for t in _TAUS])
    head = PairedEffectHead(d_model=8, hidden=8, plan_aware=True, plan_dim=8,
                            plan_raw_dim=long_plan_raw_f(120.0))
    head.set_climatology(anchor)
    assert head.plan_raw_f == 60
    token = torch.randn(2, 8)
    for plan in (torch.zeros(2, 60), torch.randn(2, 60)):
        q = head(token, plan=plan)
        assert torch.equal(q, head.climatology.expand_as(q))
    with torch.no_grad():
        head.gate.fill_(1.0)
    q_a = head(token, plan=torch.zeros(2, 60))
    q_b = head(token, plan=torch.ones(2, 60))
    assert not torch.equal(q_a, q_b)
