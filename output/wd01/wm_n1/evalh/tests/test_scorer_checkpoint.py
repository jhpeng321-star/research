"""E4 scorer checkpoint binding tests (generated fixtures only).

Every checkpoint here is a tiny random-init bundle rebuilt through
``train.idea_a.build_idea_a_bundle`` and saved under pytest ``tmp_path``.
The real member-1 checkpoint is NEVER opened; the sha refusal test
monkeypatches ``torch.load`` to fail so a refused bind provably
deserializes nothing.
"""
from __future__ import annotations

import hashlib

import pytest
import torch

import train.idea_a as training
from evalh.scorer.checkpoint import (DYNAMICS_ROLLOUT_LABEL,
                                     EXPECTED_MEMBER_SHA256,
                                     CheckpointRefusal,
                                     load_scoring_checkpoint)
from evalh.seals import LF_SHA_AMENDMENT_6


def _small_config(on: bool) -> training.IdeaATrainConfig:
    return training.IdeaATrainConfig(
        hdv_d_model=16, hdv_n_heads=2, hdv_n_layers=1, hdv_ffn=32,
        arrival_counter_channels=8, state_global_token=on, num_workers=0)


def _payload(on: bool = True, *, a6_sha=LF_SHA_AMENDMENT_6,
             config_overrides=None) -> dict:
    torch.manual_seed(17)
    bundle = training.build_idea_a_bundle(_small_config(on))
    flat = {}
    for prefix, name in (("hdv.", "hdv"), ("arrival.", "arrival"),
                         ("corr.", "corr"), ("state.", "state_enc")):
        module = getattr(bundle, name)
        if isinstance(module, torch.nn.Module):
            flat.update({prefix + key: value.detach().clone()
                         for key, value in module.state_dict().items()})
    config = bundle.config.to_dict()
    if config_overrides:
        config.update(config_overrides)
    payload = dict(format=training.CHECKPOINT_FORMAT, config=config,
                   state_dict=flat, member=1, seed=17,
                   state_global_token=on)
    if a6_sha is not None:
        payload["amendment_6_lf_sha256"] = a6_sha
    return payload


def _write(tmp_path, payload, name="member_1.pt"):
    path = tmp_path / name
    torch.save(payload, path)
    return path


def _load(path):
    """Load with the synthetic file's own pin (the real pin is unreachable)."""
    return load_scoring_checkpoint(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _forbid_deserialization(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("torch.load must not be called")
    monkeypatch.setattr(torch, "load", _boom)


def test_contract3_window_constant_is_twenty():
    from model import constants as C

    assert int(C.CORRECTION_WINDOW_STEPS) == 20
    assert EXPECTED_MEMBER_SHA256.startswith("94984cc0")
    assert DYNAMICS_ROLLOUT_LABEL == "extrapolation"


def test_sha_refusal_deserializes_nothing(tmp_path, monkeypatch):
    path = _write(tmp_path, _payload())
    _forbid_deserialization(monkeypatch)
    with pytest.raises(CheckpointRefusal, match="sha256"):
        load_scoring_checkpoint(path, expected_sha256="0" * 64)


def test_missing_file_refusal_deserializes_nothing(tmp_path, monkeypatch):
    _forbid_deserialization(monkeypatch)
    with pytest.raises(CheckpointRefusal, match="missing"):
        load_scoring_checkpoint(tmp_path / "absent.pt")


def test_load_returns_bundle_for_predictor_and_engine(tmp_path):
    path = _write(tmp_path, _payload())
    bound = _load(path)
    assert bound.member == 1
    assert bound.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert bound.bundle.config.state_global_token is True
    assert bound.dynamics_rollout_label == "extrapolation"
    assert bound.amendment_6_lf_sha256 == LF_SHA_AMENDMENT_6
    from evalh.idea_a import IdeaAVariantPredictor

    predictor = IdeaAVariantPredictor(bound.bundle)
    assert predictor.bundle is bound.bundle
    from wm_api.engine import outcome_head          # noqa: F401 (usable)

    # a second bind without the recorded binding still loads
    path2 = _write(tmp_path, _payload(a6_sha=None), name="member_2.pt")
    bound2 = _load(path2)
    assert bound2.amendment_6_lf_sha256 is None


def test_a6_off_refused(tmp_path):
    path = _write(tmp_path, _payload(on=False))
    with pytest.raises(CheckpointRefusal, match="state_global_token"):
        _load(path)


def test_a6_flag_conflict_refused(tmp_path):
    payload = _payload(on=True)
    payload["config"]["state_global_token"] = False
    path = _write(tmp_path, payload)
    with pytest.raises(CheckpointRefusal, match="conflicting"):
        _load(path)


def test_amendment6_binding_mismatch_refused(tmp_path):
    path = _write(tmp_path, _payload(a6_sha="0" * 64))
    with pytest.raises(CheckpointRefusal, match="amendment-6"):
        _load(path)


def test_recorded_window_mismatch_refused(tmp_path):
    path = _write(tmp_path, _payload(config_overrides={
        "rollout_horizon_steps": 19}))
    with pytest.raises(CheckpointRefusal, match="rollout_horizon_steps"):
        _load(path)


def test_window_constant_drift_refused(tmp_path, monkeypatch):
    path = _write(tmp_path, _payload())
    import model.constants as C

    monkeypatch.setattr(C, "CORRECTION_WINDOW_STEPS", 19)
    with pytest.raises(CheckpointRefusal, match="contract 3"):
        _load(path)


def test_unbuildable_payload_refused(tmp_path):
    payload = _payload()
    payload["state_dict"] = dict(payload["state_dict"])
    payload["state_dict"]["corr.not_a_real_weight"] = torch.zeros(1)
    path = _write(tmp_path, payload)
    with pytest.raises(CheckpointRefusal, match="member builder refused"):
        _load(path)
