"""Variant A predictor: protocol conformance and rollout shapes (fakes)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from corr import CorrConfig
from train.g4_real import RealCorrBundle

from evalh import idea_a as ia
from evalh.protocol import (QUANTILE_LEVELS, OutcomeRow, Predictor, RolloutBatch,
                            RolloutRow)


def _rollouts(n_batch, n_steps=4, K=6):
    T = n_steps + 1
    base = torch.linspace(0.0, 10.0, T).view(1, T, 1, 1)
    pos = (base + torch.arange(n_batch).view(1, 1, n_batch, 1)
           + torch.arange(K).view(1, 1, 1, K) * 0.1)
    spd = torch.full((1, T, n_batch, K), 15.0)
    lanes = torch.zeros(1, T, n_batch, K, dtype=torch.long)
    masks = torch.ones(1, T, n_batch, K)
    accel = torch.zeros(1, n_steps, n_batch, K)
    kin = torch.zeros(n_batch, K, 7)
    kin[..., 0], kin[..., 4] = pos[0, 0], spd[0, 0]
    kin[..., 1], kin[..., 5], kin[..., 6] = 1., 2., 5.
    ids = torch.arange(K).expand(n_batch, K).clone()
    cav = torch.zeros(n_batch, K, dtype=torch.bool)
    cav[:, ::2] = True
    return SimpleNamespace(positions=pos, speeds=spd, lanes=lanes, masks=masks,
                           accel=accel, t0_kin=kin, t0_mask=masks[0, 0].bool(),
                           t0_is_cav=cav, t0_slot_id=ids,
                           slot_ids=ids[None, None].expand(1, T, n_batch, K),
                           t0_counter=torch.zeros(n_batch, 2, 30))


def _rollout_fn(rows, *, n_steps, n_samples, seed):
    return _rollouts(len(rows), n_steps=n_steps)


def _outcome_fn(rows, rollouts):
    q = np.arange(QUANTILE_LEVELS.size, dtype=float)[None, :]
    return q + np.asarray([r.y_delay_s for r in rows], float)[:, None]


def _predictor():
    return ia.IdeaAVariantPredictor(bundle=object(), rollout_fn=_rollout_fn,
                                    outcome_fn=_outcome_fn)


def _outcome_rows(n=2, taus=(60.0, 120.0)):
    return [OutcomeRow(world=f"w{i}", t0_s=0.0, decision_t_s=0.0,
                       tau_s=float(t), policy="native", y_delay_s=1.0 + i)
            for i in range(n) for t in taus]


def test_variant_a_conforms_to_predictor_protocol():
    p = _predictor()
    assert isinstance(p, Predictor)
    assert p.name == "A"
    assert np.asarray(p.quantile_levels).size == 19


def test_predict_quantiles_shape_and_monotone():
    p = _predictor()
    rows = _outcome_rows()
    q = p.predict_quantiles(rows)
    assert q.shape == (len(rows), QUANTILE_LEVELS.size)
    assert np.all(np.diff(q, axis=1) >= -1e-12)
    assert p.predict_quantiles([]).shape == (0, QUANTILE_LEVELS.size)


def test_sample_rollouts_shape():
    p = _predictor()
    rows = [RolloutRow(world=f"w{i}", t0_s=0.0, policy="native",
                       true_pos=np.zeros((2, 6)), true_speed=np.zeros((2, 6)),
                       horizons_s=(10.0, 30.0)) for i in range(2)]
    b = p.sample_rollouts(rows, (10.0, 30.0))
    assert isinstance(b, RolloutBatch)
    assert b.horizons_s == (10.0, 30.0)
    assert b.pos.shape == (2, 2, 6)
    assert b.speed.shape == (2, 2, 6)
    assert b.mask.shape == (2, 2, 6)
    assert np.all(np.isfinite(b.pos))


def test_rollout_uses_row_horizons_when_none():
    p = _predictor()
    rows = [RolloutRow(world="w", t0_s=0.0, policy="native",
                       true_pos=np.zeros((1, 6)), true_speed=np.zeros((1, 6)),
                       horizons_s=(10.0,))]
    b = p.sample_rollouts(rows, None)
    assert b.pos.shape == (1, 1, 6)


def test_sample_rollouts_empty():
    p = _predictor()
    b = p.sample_rollouts([], (10.0,))
    assert b.pos.shape == (0, 1, 0)


def test_not_loaded_raises_with_rows():
    p = ia.IdeaAVariantPredictor()
    row = RolloutRow(world="w", t0_s=0.0, policy="native",
                     true_pos=np.zeros((1, 6)), true_speed=np.zeros((1, 6)))
    with pytest.raises(ia.IdeaANotLoadedError):
        p.sample_rollouts([row])


# ------------------------------------------------- A6 state global token ------
class _FakeCorrHead(RealCorrBundle):
    """DRAFT: record the token through the real synthetic P5 composition."""

    def __init__(self, d_model=8):
        super().__init__(CorrConfig(d_model=d_model))
        self.seen_token = None

    def forward(self, raw, token):
        self.seen_token = token
        return super().forward(raw, token)


class _Cfg:
    def __init__(self, on):
        self.state_global_token = on


def _a6_bundle(on=True):
    from corr import StateGlobalToken
    return SimpleNamespace(state_enc=(StateGlobalToken(d_model=8) if on
                                      else None),
                           corr=_FakeCorrHead(d_model=8),
                           config=_Cfg(on))


def _a6_rows(n=2):
    return [OutcomeRow(world=f"w{i}", t0_s=0.0, decision_t_s=0.0, tau_s=60.0,
                       policy="native", y_delay_s=0.0) for i in range(n)]


def test_idea_a_outcome_uses_state_token():
    bundle = _a6_bundle(on=True)
    p = ia.IdeaAVariantPredictor(bundle=bundle)
    q = p._outcome(_a6_rows(), _rollouts(2))
    assert q.shape == (2, QUANTILE_LEVELS.size)
    token = bundle.corr.seen_token
    assert token is not None
    # built from the t0 state with the encoder's weights, never zeros
    initial = _rollouts(2)
    kin, mask = ia._kin_from_rollout(initial)
    expect = bundle.state_enc(kin, mask, initial.t0_counter)
    assert torch.allclose(token, expect)
    assert bool(token.abs().sum() > 0.0)


def test_idea_a_outcome_refuses_state_token_checkpoint_with_no_encoder():
    bundle = _a6_bundle(on=True)
    bundle.state_enc = None
    p = ia.IdeaAVariantPredictor(bundle=bundle)
    with pytest.raises(ia.IdeaANotLoadedError):
        p._outcome(_a6_rows(), _rollouts(2))


def test_idea_a_outcome_refuses_engine_bundle_cfg_on_without_encoder():
    """The engine bundle names its config ``cfg``; an ON checkpoint with no
    encoder must still be refused (not silently scored with zeros)."""
    bundle = SimpleNamespace(state_enc=None, corr=_FakeCorrHead(d_model=8),
                             cfg=_Cfg(True))
    p = ia.IdeaAVariantPredictor(bundle=bundle)
    with pytest.raises(ia.IdeaANotLoadedError):
        p._outcome(_a6_rows(), _rollouts(2))


def test_idea_a_outcome_refuses_empty_t0_state():
    bundle = _a6_bundle(on=True)
    p = ia.IdeaAVariantPredictor(bundle=bundle)
    r = _rollouts(2)
    r.masks = torch.zeros_like(r.masks)
    r.t0_mask = torch.zeros_like(r.t0_mask)
    with pytest.raises(ia.IdeaANotLoadedError):
        p._outcome(_a6_rows(), r)


def test_idea_a_outcome_flag_off_uses_zeros():
    bundle = _a6_bundle(on=False)
    p = ia.IdeaAVariantPredictor(bundle=bundle)
    p._outcome(_a6_rows(), _rollouts(2))
    assert torch.equal(bundle.corr.seen_token,
                       torch.zeros_like(bundle.corr.seen_token))


def test_delay_quantiles_from_head_picks_row_tau():
    head_q = np.zeros((2, 2, 3, 19))
    head_q[0, 1, 0, :] = 5.0
    head_q[1, 0, 0, :] = 7.0
    rows = [OutcomeRow(world="a", t0_s=0.0, decision_t_s=0.0, tau_s=120.0,
                       policy="native", y_delay_s=0.0),
            OutcomeRow(world="b", t0_s=0.0, decision_t_s=0.0, tau_s=30.0,
                       policy="native", y_delay_s=0.0)]
    out = ia.delay_quantiles_from_head(head_q, rows, taus=(30.0, 120.0),
                                       delay_index=0)
    assert np.allclose(out[0], 5.0)
    assert np.allclose(out[1], 7.0)
