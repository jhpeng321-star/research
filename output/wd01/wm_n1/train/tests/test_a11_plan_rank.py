"""A11 synthetic contracts: long-plan config/bundle/loss + group-aware batching.

No real corpus is opened.  A schema-valid TRAIN climatology is written to
``tmp_path``.  The external-file test in ``test_paired_effect.py`` covers the
real climatology, so nothing here touches ranking data.
"""
from __future__ import annotations

import json

import pytest
import torch

from model import constants as C
from model.paired_effect import PairedEffectHead, long_plan_raw_f
from train import idea_a
from train.dataset import contrast_pairs, pair_dense_batches
from train.idea_a import (IdeaABundle, IdeaATrainConfig, IdeaATrainer)

_TAUS = C.TAUS
_NQ = C.N_QUANTILES


def _clim(path):
    rng = torch.Generator().manual_seed(3)
    doc = {
        "schema": "wm_n1_train_climatology/1",
        "contrasts": ["pi-native"],
        "taus": [float(t) for t in _TAUS],
        "quantile_levels": list(C.QUANTILE_LEVELS),
        "reference_policy": "native",
        "cells": {"pi-native": {
            f"{float(t):g}": {
                "quantiles": sorted(
                    (torch.randn(_NQ, generator=rng) * (5 + t)).tolist())}
            for t in _TAUS}},
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------- defaults ----
def test_a11_flags_default_off_and_unchanged_weights():
    args = idea_a.build_parser().parse_args(
        ["--member", "1", "--corpus", "x", "--out", "y"])
    assert args.paired_effect_plan_horizon_s is None
    assert args.paired_effect_rank_weight == 0.0
    assert args.paired_effect_group_batching is False
    cfg = IdeaATrainConfig()
    assert cfg.paired_effect_plan_horizon_s is None
    assert cfg.paired_effect_rank_weight == 0.0
    assert cfg.paired_effect_group_batching is False
    assert cfg.loss_weights()["paired_effect_rank"] == 0.0


def test_a11_dry_run_echoes_new_flags(capsys):
    rc = idea_a.main(["--member", "1", "--corpus", "x", "--out", "y",
                      "--dry-run", "--paired-effect-head",
                      "--paired-effect-plan-aware",
                      "--paired-effect-plan-horizon-s", "120",
                      "--paired-effect-rank-weight", "0.5",
                      "--paired-effect-group-batching"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["paired_effect_plan_horizon_s"] == 120.0
    assert out["paired_effect_rank_weight"] == 0.5
    assert out["paired_effect_group_batching"] is True


def test_a11_config_validation_fails_closed():
    with pytest.raises(idea_a.IdeaAConfigError):
        IdeaATrainConfig(paired_effect_plan_horizon_s=120.0)
    with pytest.raises(idea_a.IdeaAConfigError):
        IdeaATrainConfig(paired_effect_head=True,
                         paired_effect_climatology="c.json",
                         paired_effect_plan_horizon_s=120.0)
    with pytest.raises(idea_a.IdeaAConfigError):
        IdeaATrainConfig(paired_effect_rank_weight=-1.0)
    with pytest.raises(idea_a.IdeaAConfigError):
        IdeaATrainConfig(paired_effect_group_batching=True)
    cfg = IdeaATrainConfig(
        paired_effect_head=True, paired_effect_plan_aware=True,
        paired_effect_plan_horizon_s=120.0, pairs_per_batch=2,
        paired_effect_group_batching=True, paired_effect_rank_weight=0.5)
    assert cfg.paired_effect_plan_horizon_s == 120.0
    assert cfg.loss_weights()["paired_effect_rank"] == 0.5


def test_a11_bundle_builds_long_width_head(tmp_path):
    clim = _clim(tmp_path / "clim.json")
    on = idea_a.build_idea_a_bundle(IdeaATrainConfig(
        paired_effect_head=True, paired_effect_plan_aware=True,
        paired_effect_plan_horizon_s=120.0, paired_effect_climatology=clim))
    head = on.paired_effect
    assert head.plan_aware is True
    assert head.plan_raw_f == long_plan_raw_f(120.0) == 60
    assert head.plan_proj.in_features == 60


# ------------------------------------------------------------ plan vector ----
def _long_trainer(tmp_path, horizon=120.0, rank=0.0):
    clim = _clim(tmp_path / "clim.json")
    cfg = IdeaATrainConfig(
        paired_effect_head=True, paired_effect_weight=0.5,
        paired_effect_plan_aware=True,
        paired_effect_plan_horizon_s=horizon,
        paired_effect_rank_weight=rank,
        paired_effect_climatology=clim, state_global_token=False)
    head = PairedEffectHead(d_model=8, plan_aware=True, plan_dim=8,
                            plan_raw_dim=long_plan_raw_f(horizon))
    bundle = IdeaABundle(config=cfg, paired_effect=head)
    return IdeaATrainer(cfg, None, member=1, seed=1, bundle=bundle)


def test_plan_vector_selects_long_when_head_widened(tmp_path):
    tr = _long_trainer(tmp_path)
    v = tr._paired_plan_vector({"plan_long": torch.randn(2, 60)})
    assert v.shape == (2, 60)
    # the historical key is refused once the head consumes the long plan
    with pytest.raises(idea_a.IdeaAConfigError):
        tr._paired_plan_vector({"summary_raw": torch.randn(2, 16)})
    with pytest.raises(idea_a.IdeaAConfigError):
        tr._paired_plan_vector({"plan_long": torch.randn(2, 59)})


def test_plan_vector_still_selects_summary_for_short_head(tmp_path):
    clim = _clim(tmp_path / "clim.json")
    cfg = IdeaATrainConfig(paired_effect_head=True,
                           paired_effect_plan_aware=True,
                           paired_effect_climatology=clim,
                           state_global_token=False)
    head = PairedEffectHead(d_model=8, plan_aware=True, plan_dim=8)
    tr = IdeaATrainer(cfg, None, member=1, seed=1,
                      bundle=IdeaABundle(config=cfg, paired_effect=head))
    assert tr._paired_plan_vector(
        {"summary_raw": torch.randn(2, 16)}).shape == (2, 16)
    with pytest.raises(idea_a.IdeaAConfigError):
        tr._paired_plan_vector({"plan_long": torch.randn(2, 60)})


# ------------------------------------------------------------- rank loss -----
def _rank_batch():
    n_tau = len(_TAUS)
    meta = [{"pair_key": "w1", "policy": "native", "arm": "native"},
            {"pair_key": "w1", "policy": "pi", "arm": "pi"},
            {"pair_key": "w1", "policy": "damping", "arm": "damping"}]
    return {
        "meta": meta,
        "outcome_quantiles": torch.randn(3, n_tau, 3),
        "outcome_mask": torch.ones(3, n_tau, dtype=torch.bool),
        "global_token": torch.randn(3, 8),
    }


def _rank_trainer(tmp_path, weight=0.5):
    clim = _clim(tmp_path / "clim.json")
    cfg = IdeaATrainConfig(paired_effect_head=True, paired_effect_weight=0.5,
                           paired_effect_rank_weight=weight,
                           paired_effect_climatology=clim,
                           state_global_token=False)
    head = PairedEffectHead(d_model=8)
    bundle = IdeaABundle(config=cfg, paired_effect=head)
    return IdeaATrainer(cfg, None, member=1, seed=1, bundle=bundle)


def test_rank_loss_counts_and_gradient(tmp_path):
    tr = _rank_trainer(tmp_path)
    loss = tr.paired_effect_rank_loss(_rank_batch())
    assert torch.isfinite(loss)
    assert tr._rank_diag["rank_pairs"] == 1        # two treatments of w1
    assert tr._rank_diag["rank_cells"] > 0
    loss.backward()
    assert tr.bundle.paired_effect.gate.grad is not None
    assert torch.isfinite(tr.bundle.paired_effect.gate.grad).all()


def test_rank_weight_applied_once_in_weighted_total(tmp_path):
    tr = _rank_trainer(tmp_path, weight=0.5)
    total = tr._weighted_total({"paired_effect_rank": torch.tensor(2.0)})
    assert float(total) == pytest.approx(1.0)          # 0.5 * 2.0, once
    off = _rank_trainer(tmp_path, weight=0.0)
    assert off.config.loss_weights()["paired_effect_rank"] == 0.0


# --------------------------------------------------------- checkpoint -------
def test_a11_checkpoint_round_trip_preserves_long_head(tmp_path):
    tr = _long_trainer(tmp_path, rank=0.5)
    sd = tr.state_dict()
    assert sd["paired.plan_proj.weight"].shape[-1] == 60
    ckpt = tr.save_checkpoint(tmp_path / "m.pt")
    payload = torch.load(ckpt, weights_only=True)
    assert payload["config"]["paired_effect_plan_horizon_s"] == 120.0
    assert payload["config"]["paired_effect_rank_weight"] == 0.5
    assert payload["state_dict"]["paired.plan_proj.weight"].shape == \
        sd["paired.plan_proj.weight"].shape
    fresh = _long_trainer(tmp_path, rank=0.5)
    fresh.load_state_dict(payload["state_dict"])
    sd2 = fresh.state_dict()
    for key in sd:
        assert torch.equal(sd[key], sd2[key])


# ----------------------------------------------------- group-aware batch -----
_ITEMS = [
    {"pair_key": "w1", "policy": "native", "arm": "native"},
    {"pair_key": "w1", "policy": "pi", "arm": "pi"},
    {"pair_key": "w1", "policy": "damping", "arm": "damping"},
    {"pair_key": "w2", "policy": "native", "arm": "native"},
    {"pair_key": "w2", "policy": "pi", "arm": "pi"},
]


def test_group_aware_pairs_put_multiple_arms_of_one_decision_together():
    pairs = contrast_pairs(_ITEMS)
    got = pair_dense_batches(_ITEMS, 3, seed=5, epoch=0, pairs=pairs,
                             group_aware=True)
    # same-decision treatments (positions 1 and 2) must co-occur in a batch
    assert any(1 in b and 2 in b for b in got), got
    # deterministic for the same seed/epoch
    again = pair_dense_batches(_ITEMS, 3, seed=5, epoch=0, pairs=pairs,
                               group_aware=True)
    assert got == again
