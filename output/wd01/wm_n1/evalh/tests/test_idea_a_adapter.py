"""Composed Idea A bundle loader in adapter.predictor (synthetic, no data)."""
from __future__ import annotations

import json

import torch

from adapter import predictor as AP
from corr import CorrConfig, CorrectionHead
from evalh.idea_a import VARIANT_A, IdeaAVariantPredictor
from evalh.protocol import Predictor

_CORR_CFG = dict(summary_dim=8, d_model=8, n_tau=2, n_outcome_targets=3,
                 n_quantiles=19)


def test_load_bundle_from_mapping_with_factory_specs():
    bundle = AP.load_idea_a_bundle({
        "ctrl": None, "hdv": None, "arrival": None,
        "corr": {"factory": "corr:CorrectionHead", "config": {"class": "corr:CorrConfig", "kwargs": _CORR_CFG}}})
    assert isinstance(bundle.corr, CorrectionHead)
    assert bundle.ctrl is None and bundle.hdv is None
    p = AP.IdeaABundlePredictor(bundle)
    assert isinstance(p, Predictor)
    assert isinstance(p, IdeaAVariantPredictor)
    assert p.name == VARIANT_A == "A"


def test_load_bundle_roundtrips_existing_engine_bundle():
    from rollout2.engine import IdeaABundle
    b = IdeaABundle(ctrl=object(), hdv=object(), arrival=object(), corr=object())
    assert AP.load_idea_a_bundle(b) is b


def test_load_bundle_from_directory_manifest_restores_state(tmp_path):
    head = CorrectionHead(CorrConfig(**_CORR_CFG))
    torch.save(head.state_dict(), tmp_path / "corr.pt")
    (tmp_path / "idea_a.json").write_text(json.dumps({
        "corr": {"factory": "corr:CorrectionHead", "config": {"class": "corr:CorrConfig", "kwargs": _CORR_CFG},
                 "state": "corr.pt"}}), encoding="utf-8")
    bundle = AP.load_idea_a_bundle(tmp_path)
    want = head.state_dict()
    got = bundle.corr.state_dict()
    assert set(want) == set(got)
    assert all(torch.equal(want[k], got[k]) for k in want)


def test_load_bundle_mounts_a6_state_encoder(tmp_path):
    """A6: the composed bundle loader mounts ``state_enc`` with its weights."""
    from corr import StateGlobalToken

    enc = StateGlobalToken(d_model=8)
    torch.save(enc.state_dict(), tmp_path / "state.pt")
    (tmp_path / "idea_a.json").write_text(json.dumps({
        "state_enc": {"factory": "corr:StateGlobalToken",
                      "kwargs": {"d_model": 8},
                      "state": "state.pt"}}), encoding="utf-8")
    bundle = AP.load_idea_a_bundle(tmp_path)
    assert isinstance(bundle.state_enc, StateGlobalToken)
    want = enc.state_dict()
    got = bundle.state_enc.state_dict()
    assert set(want) == set(got)
    assert all(torch.equal(want[k], got[k]) for k in want)


def test_load_idea_a_predictor_wraps_bundle():
    bundle = AP.load_idea_a_bundle({
        "corr": {"factory": "corr:CorrectionHead", "config": {"class": "corr:CorrConfig", "kwargs": _CORR_CFG}}})
    p = AP.load_idea_a_predictor(bundle)
    assert isinstance(p, AP.IdeaABundlePredictor)
    assert p.name == "A"
    assert p.bundle is bundle
