"""CLI / config / loss-hook contracts for the optional paired-effect head.

Synthetic only; no CAL/HOLDOUT data.  The external TRAIN climatology is
written locally as a schema-valid fixture.
"""
from __future__ import annotations

import json

import pytest
import torch

from model import constants as C
from model.paired_effect import PairedEffectHead
from train import idea_a
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


# ------------------------------------------------------------------ parser ----
def test_parser_paired_effect_defaults_off():
    args = idea_a.build_parser().parse_args(
        ["--member", "1", "--corpus", "x", "--out", "y"])
    assert args.paired_effect_head is False
    assert args.paired_effect_weight == 1.0
    assert args.paired_effect_climatology is None


def test_parser_accepts_paired_effect_flags():
    args = idea_a.build_parser().parse_args(
        ["--member", "1", "--corpus", "x", "--out", "y",
         "--paired-effect-head", "--paired-effect-weight", "0.25",
         "--paired-effect-climatology", "c.json",
         "--paired-effect-climatology-sha256", "a" * 64])
    assert args.paired_effect_head is True
    assert args.paired_effect_weight == 0.25
    assert args.paired_effect_climatology == "c.json"
    assert args.paired_effect_climatology_sha256 == "a" * 64


def test_dry_run_echoes_paired_effect_flags(capsys):
    rc = idea_a.main(["--member", "1", "--corpus", "x", "--out", "y",
                      "--dry-run", "--paired-effect-head",
                      "--paired-effect-weight", "0.5",
                      "--paired-effect-climatology", "c.json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["paired_effect_head"] is True
    assert out["paired_effect_weight"] == 0.5
    assert out["paired_effect_climatology"] == "c.json"


# ------------------------------------------------------------------ config ----
def test_config_accepts_new_keys_and_rejects_negative_weight():
    cfg = IdeaATrainConfig.from_dict({"paired_effect_head": True,
                                      "paired_effect_weight": 0.0})
    assert cfg.paired_effect_head is True and cfg.paired_effect_weight == 0.0
    with pytest.raises(idea_a.IdeaAConfigError):
        IdeaATrainConfig(paired_effect_weight=-1.0)
    assert IdeaATrainConfig().loss_weights()["paired_effect"] == 1.0


def test_bundle_builds_head_only_when_enabled(tmp_path):
    clim = _clim(tmp_path / "clim.json")
    off = idea_a.build_idea_a_bundle(IdeaATrainConfig())
    assert off.paired_effect is None
    on = idea_a.build_idea_a_bundle(IdeaATrainConfig(
        paired_effect_head=True, paired_effect_climatology=clim))
    assert isinstance(on.paired_effect, PairedEffectHead)
    # untrained head reproduces the anchor exactly
    token = torch.zeros(2, int(on.paired_effect.d_model))
    q = on.paired_effect(token)
    assert torch.equal(q, on.paired_effect.climatology.expand_as(q))


# --------------------------------------------------------------- loss hook ----
def _batch(batch_size=2, d_model=8):
    meta = [{"pair_key": "w1", "policy": "pi"},
            {"pair_key": "w1", "policy": "native"}]
    return {
        "meta": meta,
        "outcome_quantiles": torch.randn(batch_size, len(_TAUS), 3),
        "outcome_mask": torch.ones(batch_size, len(_TAUS), dtype=torch.bool),
        "global_token": torch.randn(batch_size, d_model),
    }


def _trainer(tmp_path, *, enabled=True, weight=0.5, d_model=8):
    clim = _clim(tmp_path / "clim.json")
    cfg = IdeaATrainConfig(paired_effect_head=enabled,
                           paired_effect_weight=weight,
                           paired_effect_climatology=clim,
                           state_global_token=False)
    head = PairedEffectHead(d_model=d_model)
    bundle = IdeaABundle(config=cfg,
                         paired_effect=head if enabled else None)
    return IdeaATrainer(cfg, None, member=1, seed=1, bundle=bundle)


def test_paired_effect_loss_is_finite(tmp_path):
    tr = _trainer(tmp_path)
    loss = tr.paired_effect_loss(_batch())
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert torch.isfinite(tr.bundle.paired_effect.gate.grad).all()


def test_paired_effect_loss_requires_head(tmp_path):
    tr = _trainer(tmp_path, enabled=False)
    with pytest.raises(idea_a.IdeaAConfigError):
        tr.paired_effect_loss(_batch())


def test_head_is_included_in_parameters_and_state_dict(tmp_path):
    tr = _trainer(tmp_path)
    names = tr.state_dict()
    assert any(k.startswith("paired.") for k in names)
    assert "paired.gate" in names


# ------------------------------------------------------ plan-aware A9 --------
def test_parser_paired_effect_plan_aware_defaults_off():
    args = idea_a.build_parser().parse_args(
        ["--member", "1", "--corpus", "x", "--out", "y"])
    assert args.paired_effect_plan_aware is False


def test_parser_accepts_paired_effect_plan_aware_flag():
    args = idea_a.build_parser().parse_args(
        ["--member", "1", "--corpus", "x", "--out", "y",
         "--paired-effect-head", "--paired-effect-plan-aware"])
    assert args.paired_effect_plan_aware is True


def test_dry_run_echoes_plan_aware(capsys):
    rc = idea_a.main(["--member", "1", "--corpus", "x", "--out", "y",
                      "--dry-run", "--paired-effect-head",
                      "--paired-effect-plan-aware",
                      "--paired-effect-climatology", "c.json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["paired_effect_plan_aware"] is True


def test_config_plan_aware_requires_the_head():
    with pytest.raises(idea_a.IdeaAConfigError):
        IdeaATrainConfig(paired_effect_plan_aware=True)
    cfg = IdeaATrainConfig.from_dict({"paired_effect_head": True,
                                      "paired_effect_plan_aware": True})
    assert cfg.paired_effect_plan_aware is True


def test_bundle_builds_plan_aware_head(tmp_path):
    clim = _clim(tmp_path / "clim.json")
    on = idea_a.build_idea_a_bundle(IdeaATrainConfig(
        paired_effect_head=True, paired_effect_plan_aware=True,
        paired_effect_climatology=clim))
    head = on.paired_effect
    assert isinstance(head, PairedEffectHead)
    assert head.plan_aware is True and head.plan_proj is not None
    token = torch.zeros(2, int(head.d_model))
    q = head(token, plan=torch.randn(2, 16))
    assert torch.equal(q, head.climatology.expand_as(q))


def _plan_trainer(tmp_path, *, plan_aware=True, d_model=8):
    clim = _clim(tmp_path / "clim.json")
    cfg = IdeaATrainConfig(paired_effect_head=True,
                           paired_effect_weight=0.5,
                           paired_effect_plan_aware=plan_aware,
                           paired_effect_climatology=clim,
                           state_global_token=False)
    head = PairedEffectHead(d_model=d_model, plan_aware=plan_aware,
                            plan_dim=d_model)
    bundle = IdeaABundle(config=cfg, paired_effect=head)
    return IdeaATrainer(cfg, None, member=1, seed=1, bundle=bundle)


def _batch_plan(batch_size=2, d_model=8, *, summary=True):
    batch = _batch(batch_size, d_model)
    if summary:
        batch["summary_raw"] = torch.randn(batch_size, 16)
    return batch


def test_paired_effect_loss_plan_aware_is_finite_and_trains_plan_proj(tmp_path):
    tr = _plan_trainer(tmp_path)
    loss = tr.paired_effect_loss(_batch_plan())
    assert torch.isfinite(loss)
    assert 0.0 < float(loss.detach()) < 100.0
    loss.backward()
    head = tr.bundle.paired_effect
    assert torch.isfinite(head.gate.grad).all()
    assert head.plan_proj.weight.grad is not None
    assert torch.isfinite(head.plan_proj.weight.grad).all()
    assert any(k.startswith("paired.plan_proj.") for k in tr.state_dict())


def test_paired_effect_loss_plan_aware_requires_summary_raw(tmp_path):
    tr = _plan_trainer(tmp_path)
    with pytest.raises(idea_a.IdeaAConfigError):
        tr.paired_effect_loss(_batch_plan(summary=False))
