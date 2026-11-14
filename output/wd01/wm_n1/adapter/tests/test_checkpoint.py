"""Ensemble checkpoint save/load roundtrip (plain state_dict + JSON config)."""
from __future__ import annotations

import json

import numpy as np
import torch

from adapter import checkpoint as CK
from adapter.predictor import WmN1Predictor

import synth_adapter as synth


def test_directory_roundtrip_predictions_identical(tmp_path):
    ensemble = synth.tiny_ensemble()
    ckpt = tmp_path / "ens"
    CK.save_ensemble(ckpt, ensemble,
                     member_train_worlds={"1": ["w0"], "2": ["w1"]})
    assert (ckpt / "config.json").is_file()
    assert sorted(p.name for p in ckpt.glob("member_*.pt")) == [
        f"member_{k}.pt" for k in range(1, 6)]

    corpus = synth.fake_corpus(["w0", "w1"])
    rows = synth.outcome_rows(["w0", "w1"])
    direct = WmN1Predictor(ensemble, corpus, device="cpu")
    loaded = WmN1Predictor.from_checkpoint(ckpt, corpus, device="cpu")
    qa = direct.predict_quantiles(rows)
    qb = loaded.predict_quantiles(rows)
    assert np.allclose(qa, qb, atol=1e-6)
    # OOF bookkeeping survives the roundtrip (0-based member keys).
    assert loaded.member_train_worlds[0] == {"w0"}
    assert loaded.member_train_worlds[1] == {"w1"}


def test_single_file_layout_roundtrip(tmp_path):
    ensemble = synth.tiny_ensemble()
    path = tmp_path / "ens.pt"
    CK.save_ensemble_file(path, ensemble)
    loaded, conf = CK.load_ensemble(path, device="cpu")
    assert tuple(loaded.seeds) == tuple(ensemble.seeds)
    for a, b in zip(loaded.members, ensemble.members):
        for ka, va in a.state_dict().items():
            assert torch.equal(va, b.state_dict()[ka])
    assert conf["n_members"] == 5


def test_load_is_weights_only(tmp_path):
    ensemble = synth.tiny_ensemble()
    ckpt = tmp_path / "ens"
    CK.save_ensemble(ckpt, ensemble)
    conf = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    assert conf["arch"]["d_model"] == 16
    # loading a plain state_dict with weights_only=True must succeed
    state = torch.load(ckpt / "member_1.pt", map_location="cpu",
                       weights_only=True)
    assert all(isinstance(v, torch.Tensor) for v in state.values())


def test_missing_checkpoint_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        CK.load_ensemble(tmp_path / "nope", device="cpu")
