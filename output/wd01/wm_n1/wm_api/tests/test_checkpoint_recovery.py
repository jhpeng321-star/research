"""DRAFT, NOT RUN: synthetic checkpoint integration, pending reviewed guard.

All checkpoint payloads are freshly constructed from real module weights in
pytest's tmp_path. No existing checkpoint/corpus or trainer run is opened.
"""
import numpy as np
import pytest
import torch

from corr import CorrectionHead
from train import idea_a as training
from train.g4_real import RealCorrBundle
from wm_api.checkpoint import CHECKPOINT_FORMAT, load_checkpoint
from wm_api.engine import outcome_head
from wm_api.errors import CheckpointError
from wm_api.state import StateSnapshot


def _synthetic_bundle(on=True):
    torch.manual_seed(17)
    # Default correction architecture: external raw16, internal summary64.
    # Small HDV/arrival fixtures keep this loader test bounded.
    cfg = training.IdeaATrainConfig(
        hdv_d_model=16, hdv_n_heads=2, hdv_n_layers=1, hdv_ffn=32,
        arrival_counter_channels=8, state_global_token=on, num_workers=0)
    assert cfg.corr_summary_dim == 64
    return training.build_idea_a_bundle(cfg)


def _payload(bundle):
    flat = {}
    for prefix, name in (("hdv.", "hdv"), ("arrival.", "arrival"),
                         ("corr.", "corr"), ("state.", "state_enc")):
        module = getattr(bundle, name)
        if isinstance(module, torch.nn.Module):
            flat.update({prefix + key: value.detach().clone()
                         for key, value in module.state_dict().items()})
    return dict(format=CHECKPOINT_FORMAT, config=bundle.config.to_dict(),
                state_dict=flat, member=1, seed=17)


def _write(tmp_path, payload):
    torch.save(payload, tmp_path / "member_1.pt")


@pytest.mark.parametrize("on", [False, True])
def test_load_default_summary64_checkpoint_and_score_api(tmp_path, on):
    source = _synthetic_bundle(on)
    payload = _payload(source)
    _write(tmp_path, payload)
    loaded = load_checkpoint(tmp_path)
    bundle = loaded.members[0].bundle
    assert isinstance(bundle.corr, RealCorrBundle)
    assert bundle.corr.cfg.summary_dim == 64
    assert bundle.config.state_global_token is on
    assert (bundle.state_enc is not None) is on
    for prefix, name in (("hdv.", "hdv"), ("arrival.", "arrival"),
                         ("corr.", "corr"), ("state.", "state_enc")):
        module = getattr(bundle, name)
        if module is None:
            continue
        assert not module.training
        dtype = torch.float32 if name == "arrival" else torch.float64
        assert all(p.dtype == dtype and p.device.type == "cpu" for p in module.parameters())
        for key, value in module.state_dict().items():
            assert torch.equal(value, payload["state_dict"][prefix + key].to(value.dtype))
    state = StateSnapshot(
        ids=["cav", "hdv"], lane_phys=np.array([0, 1]),
        s_down_m=np.array([0., 100.]), v_mps=np.array([10., 20.]),
        a_mps2=np.array([-2., 1.]), length_m=np.array([4.5, 7.]),
        is_cav=np.array([True, False]),
        entrance_counters=np.zeros((2, 30)) if on else None)
    pos = np.array([[[0., 100.]], [[5., 110.]], [[11., 120.]]])
    spd = np.array([[[10., 20.]], [[10., 20.]], [[12., 20.]]])
    acc = np.array([[[0., 0.]], [[4., 0.]]])
    masks = np.ones((3, 1, 2), dtype=bool)
    ids = np.repeat(np.array([[[0, 1]]]), 3, axis=0)
    q, raw = outcome_head(bundle, pos, spd, acc, masks, state=state, slot_ids=ids)
    assert q.shape == (5, 3, 19) and raw.shape == (16,)
    assert np.isfinite(q).all() and raw[9] == -2. and raw[12] == 11.


@pytest.mark.parametrize("fault", ["shape", "missing_projection", "missing_corr",
                                   "missing_state", "unexpected_corr_key", "unknown_key"])
def test_checkpoint_strict_weight_mismatches_still_refused(tmp_path, fault):
    payload = _payload(_synthetic_bundle())
    flat = payload["state_dict"]
    if fault == "shape":
        key = "corr.proj.proj.0.weight"
        flat[key] = flat[key][:-1]
    elif fault == "missing_projection":
        payload["state_dict"] = {key: value for key, value in flat.items()
                                 if not key.startswith("corr.proj.")}
    elif fault == "missing_corr":
        payload["state_dict"] = {key: value for key, value in flat.items()
                                 if not key.startswith("corr.")}
    elif fault == "missing_state":
        payload["state_dict"] = {key: value for key, value in flat.items()
                                 if not key.startswith("state.")}
    elif fault == "unexpected_corr_key":
        flat["corr.unexpected_synthetic_weight"] = torch.zeros(1)
    else:
        flat["unknown.synthetic"] = torch.zeros(1)
    _write(tmp_path, payload)
    with pytest.raises(CheckpointError, match="mismatch"):
        load_checkpoint(tmp_path)


@pytest.mark.parametrize("fault", ["bare_head", "missing_projection"])
def test_loader_refuses_unsupported_composition_even_with_matching_weights(tmp_path, monkeypatch, fault):
    source = _synthetic_bundle()
    if fault == "bare_head":
        source.corr = CorrectionHead(source.corr.cfg)
    else:
        source.corr.proj = None
    _write(tmp_path, _payload(source))
    monkeypatch.setattr(training, "build_idea_a_bundle", lambda config: source)
    with pytest.raises(CheckpointError, match="unsupported correction composition"):
        load_checkpoint(tmp_path)


def test_explicit_off_checkpoint_with_stray_state_weights_refused(tmp_path):
    payload = _payload(_synthetic_bundle(on=False))
    payload["state_dict"]["state.synthetic_weight"] = torch.zeros(1)
    _write(tmp_path, payload)
    with pytest.raises(CheckpointError, match="no matching module"):
        load_checkpoint(tmp_path)


@pytest.mark.parametrize("name", ["hdv", "arrival", "corr", "state_enc"])
def test_checkpoint_weights_for_absent_module_refused(tmp_path, monkeypatch, name):
    source = _synthetic_bundle()
    _write(tmp_path, _payload(source))
    setattr(source, name, None)
    monkeypatch.setattr(training, "build_idea_a_bundle", lambda config: source)
    with pytest.raises(CheckpointError, match="no matching module"):
        load_checkpoint(tmp_path)


# --------------------------------------------------- plan-aware A9 head ------
def _climatology(path):
    import json

    from model import constants as C

    rng = np.random.default_rng(0)
    doc = {
        "schema": "wm_n1_train_climatology/1",
        "contrasts": ["pi-native"],
        "taus": [float(t) for t in C.TAUS],
        "quantile_levels": list(C.QUANTILE_LEVELS),
        "reference_policy": "native",
        "cells": {"pi-native": {
            f"{float(t):g}": {"quantiles": sorted(
                rng.normal(size=C.N_QUANTILES).tolist())}
            for t in C.TAUS}},
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _plan_aware_bundle(tmp_path):
    clim = _climatology(tmp_path / "clim.json")
    cfg = training.IdeaATrainConfig(
        hdv_d_model=16, hdv_n_heads=2, hdv_n_layers=1, hdv_ffn=32,
        arrival_counter_channels=8, state_global_token=True, num_workers=0,
        paired_effect_head=True, paired_effect_plan_aware=True,
        paired_effect_climatology=clim)
    return cfg, training.build_idea_a_bundle(cfg)


def test_paired_effect_plan_aware_checkpoint_roundtrip(tmp_path):
    from model.paired_effect import PairedEffectHead

    cfg, source = _plan_aware_bundle(tmp_path)
    head = source.paired_effect
    assert isinstance(head, PairedEffectHead) and head.plan_aware
    with torch.no_grad():                      # non-trivial saved weights
        head.plan_proj.weight.normal_()
        head.plan_proj.bias.normal_()
        head.gate.fill_(0.25)
    flat = {}
    for prefix, name in (("hdv.", "hdv"), ("arrival.", "arrival"),
                         ("corr.", "corr"), ("state.", "state_enc"),
                         ("paired.", "paired_effect")):
        module = getattr(source, name)
        if isinstance(module, torch.nn.Module):
            flat.update({prefix + key: value.detach().clone()
                         for key, value in module.state_dict().items()})
    _write(tmp_path, dict(format=CHECKPOINT_FORMAT, config=cfg.to_dict(),
                          state_dict=flat, member=1, seed=17))
    loaded = load_checkpoint(tmp_path)
    loaded_head = loaded.members[0].bundle.paired_effect
    assert isinstance(loaded_head, PairedEffectHead)
    assert loaded_head.plan_aware is True
    assert all(p.dtype == torch.float64 for p in loaded_head.parameters())
    for key, value in source.paired_effect.state_dict().items():
        assert torch.equal(loaded_head.state_dict()[key],
                           value.to(torch.float64)), key


def test_paired_effect_weights_refused_without_a_head(tmp_path):
    _, source = _plan_aware_bundle(tmp_path)
    flat = {("paired." + key): value.detach().clone()
            for key, value in source.paired_effect.state_dict().items()}
    payload = _payload(source)
    payload["state_dict"].update(flat)
    payload["config"] = training.IdeaATrainConfig(
        hdv_d_model=16, hdv_n_heads=2, hdv_n_layers=1, hdv_ffn=32,
        arrival_counter_channels=8, state_global_token=True,
        num_workers=0).to_dict()
    _write(tmp_path, payload)
    with pytest.raises(CheckpointError, match="no matching module"):
        load_checkpoint(tmp_path)
