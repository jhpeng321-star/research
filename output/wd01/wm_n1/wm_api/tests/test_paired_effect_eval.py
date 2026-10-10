"""Synthetic-only checks for the wm_api paired-effect evaluation surface."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from corr import StateGlobalToken
from model import constants as C
from model.paired_effect import PairedEffectHead
from train.idea_a import IdeaABundle, IdeaATrainConfig
from wm_api import (has_paired_effect_head, paired_effect_quantiles,
                    paired_effect_quantiles_for_plan,
                    paired_effect_quantiles_from_token)

_NQ = C.N_QUANTILES


def _head(d_model=8):
    rng = np.random.default_rng(0)
    anchor = np.stack([np.sort(rng.normal(scale=5.0 + t, size=_NQ))
                       for t in C.TAUS])
    h = PairedEffectHead(d_model=d_model)
    h.set_climatology(anchor)
    return h, anchor


def test_has_head_and_refusal_without_head():
    assert has_paired_effect_head(IdeaABundle()) is False
    with pytest.raises(ValueError):
        paired_effect_quantiles_from_token(IdeaABundle(), torch.zeros(1, 8))


def test_from_token_returns_untrained_climatology_exactly():
    h, _ = _head()
    bundle = IdeaABundle(paired_effect=h)
    q = paired_effect_quantiles_from_token(bundle, torch.zeros(3, 8))
    assert q.shape == (3, len(C.TAUS), _NQ)
    ref = h.climatology.detach().numpy()
    assert np.array_equal(q, np.broadcast_to(ref, q.shape))


def test_state_built_path_with_global_token_off():
    h, _ = _head()
    bundle = IdeaABundle(config=IdeaATrainConfig(state_global_token=False),
                         paired_effect=h)
    kin = np.zeros((2, 4, 7), dtype=np.float32)
    mask = np.ones((2, 4), dtype=bool)
    counter = np.zeros((2, 2, 30), dtype=np.float32)
    q = paired_effect_quantiles(bundle, t0_kin=kin, t0_mask=mask,
                                t0_counter=counter)
    assert q.shape == (2, len(C.TAUS), _NQ)
    ref = h.climatology.detach().numpy()
    assert np.array_equal(q, np.broadcast_to(ref, q.shape))


# ------------------------------------------------------ plan-aware A9 --------
def _plan_head(d_model=8):
    rng = np.random.default_rng(0)
    anchor = np.stack([np.sort(rng.normal(scale=5.0 + t, size=_NQ))
                       for t in C.TAUS])
    h = PairedEffectHead(d_model=d_model, plan_aware=True, plan_dim=d_model)
    h.set_climatology(anchor)
    return h, anchor


def _plan_bundle(h, *, on=False):
    return IdeaABundle(config=IdeaATrainConfig(state_global_token=on),
                       state_enc=StateGlobalToken(int(h.d_model)) if on else None,
                       paired_effect=h)


def _plan_inputs(d_model=8, seed=1):
    rng = np.random.default_rng(seed)
    return dict(t0_kin=rng.normal(size=(4, 7)).astype(np.float32),
                t0_mask=np.ones(4, dtype=bool),
                t0_counter=np.zeros((2, 30), dtype=np.float32),
                summary_raw=rng.normal(size=16).astype(np.float32))


def test_for_plan_untrained_returns_climatology_exactly():
    h, _ = _plan_head()
    bundle = _plan_bundle(h, on=False)
    q = paired_effect_quantiles_for_plan(bundle, **_plan_inputs())
    assert q.shape == (len(C.TAUS), _NQ)
    assert q.dtype == np.float64
    assert np.array_equal(q, h.climatology.detach().numpy())


def test_for_plan_requires_a_plan_aware_head():
    h, _ = _head()
    bundle = IdeaABundle(config=IdeaATrainConfig(state_global_token=False),
                         paired_effect=h)
    with pytest.raises(ValueError):
        paired_effect_quantiles_for_plan(bundle, **_plan_inputs())


def test_for_plan_uses_the_plan_vector():
    h, _ = _plan_head()
    with torch.no_grad():
        h.gate.fill_(1.0)
    bundle = _plan_bundle(h, on=False)
    inputs = _plan_inputs()
    q_zero = paired_effect_quantiles_for_plan(
        bundle, **{**inputs, "summary_raw": np.zeros(16, dtype=np.float32)})
    q_one = paired_effect_quantiles_for_plan(
        bundle, **{**inputs, "summary_raw": np.ones(16, dtype=np.float32)})
    assert not np.array_equal(q_zero, q_one)


def test_for_plan_builds_the_state_token_with_the_encoder():
    from corr.evaluation import state_token

    h, _ = _plan_head()
    bundle = _plan_bundle(h, on=True)
    inputs = _plan_inputs()
    q = paired_effect_quantiles_for_plan(bundle, **inputs)
    token = state_token(
        bundle, torch.zeros(1, int(h.d_model)),
        kin=torch.as_tensor(inputs["t0_kin"]).unsqueeze(0),
        mask=torch.as_tensor(inputs["t0_mask"]).unsqueeze(0),
        counter=torch.as_tensor(inputs["t0_counter"]).unsqueeze(0),
        d_model=int(h.d_model))
    expected = h.dv_quantiles(
        token, plan=torch.as_tensor(inputs["summary_raw"]).unsqueeze(0)
    ).detach().numpy()[0].astype(np.float64)
    assert np.array_equal(q, expected)


def test_for_plan_refuses_wrong_summary_width():
    h, _ = _plan_head()
    bundle = _plan_bundle(h, on=False)
    with pytest.raises(ValueError):
        paired_effect_quantiles_for_plan(
            bundle, **{**_plan_inputs(),
                       "summary_raw": np.zeros(8, dtype=np.float32)})


# -------------------------------------------------- full-horizon A11 --------
def _long_plan_head(d_model=8, horizon_s=120.0):
    from model.paired_effect import long_plan_raw_f

    rng = np.random.default_rng(0)
    anchor = np.stack([np.sort(rng.normal(scale=5.0 + t, size=_NQ))
                       for t in C.TAUS])
    h = PairedEffectHead(d_model=d_model, plan_aware=True, plan_dim=d_model,
                         plan_raw_dim=long_plan_raw_f(horizon_s))
    h.set_climatology(anchor)
    return h, anchor


def test_for_plan_long_head_requires_plan_long_and_uses_it():
    h, _ = _long_plan_head()
    bundle = _plan_bundle(h, on=False)
    inputs = _plan_inputs()
    with pytest.raises(ValueError):
        paired_effect_quantiles_for_plan(bundle, **inputs)   # no plan_long
    q = paired_effect_quantiles_for_plan(
        bundle, **inputs, plan_long=np.zeros(60, dtype=np.float32))
    assert q.shape == (len(C.TAUS), _NQ)
    assert q.dtype == np.float64
    with torch.no_grad():
        h.gate.fill_(1.0)
    q0 = paired_effect_quantiles_for_plan(
        bundle, **inputs, plan_long=np.zeros(60, dtype=np.float32))
    q1 = paired_effect_quantiles_for_plan(
        bundle, **inputs, plan_long=np.ones(60, dtype=np.float32))
    assert not np.array_equal(q0, q1)


def test_for_plan_long_head_refuses_wrong_width_and_short_head_refuses_long():
    h, _ = _long_plan_head()
    bundle = _plan_bundle(h, on=False)
    with pytest.raises(ValueError):
        paired_effect_quantiles_for_plan(
            bundle, **_plan_inputs(), plan_long=np.zeros(59, dtype=np.float32))
    h_short, _ = _plan_head()
    short = _plan_bundle(h_short, on=False)
    with pytest.raises(ValueError):
        paired_effect_quantiles_for_plan(
            short, **_plan_inputs(), plan_long=np.zeros(60, dtype=np.float32))
