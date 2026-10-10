"""A5 K=192 train-side contract: MemmapCorpus K_max + pack/model refusal.

The slot width is a FORMAT parameter (amendment 5): a v3 pack carries
k_max=192; a K mismatch between the pack and the model/config refuses.
Synthetic packs only (``packsynth``); zero SUMO, no real data.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import train.idea_a as ia
from model import constants as C
from packsynth import make_runs, write_pack
from train.dataset import CorpusRefusal, MemmapCorpus

K192 = 192


@pytest.fixture
def pack192(tmp_path):
    root = tmp_path / "pack_v3"
    write_pack(root, make_runs(), k=K192)
    return root


def _set_format(pack_dir, format_value):
    import json

    p = pack_dir / "manifest.json"
    man = json.loads(p.read_text(encoding="utf-8"))
    if format_value is None:
        man.pop("format", None)
    else:
        man["format"] = format_value
    p.write_text(json.dumps(man, indent=1), encoding="utf-8")


def test_v3_pack_opens_and_carries_k_max(pack192):
    corpus = MemmapCorpus(pack192)
    assert corpus.k_max == K192
    assert corpus.format is None            # synthetic fixture: unclaimed
    item = corpus[0]
    assert item["veh_step"].shape == (K192, C.VEH_F)
    assert item["plan"].shape == (K192, C.N_SEGMENTS, C.PLAN_F)
    assert item["accel"].shape == (K192,)
    assert item["veh_mask"].sum() >= 3


def test_pack_and_model_k_mismatch_refuses(pack192):
    with pytest.raises(CorpusRefusal) as ei:
        MemmapCorpus(pack192, k_max=128)
    assert "K_max mismatch" in str(ei.value)
    assert "128" in str(ei.value) and "192" in str(ei.value)
    # and the reverse pairing refuses too
    pack128 = pack192.parent / "pack_v2"
    write_pack(pack128, make_runs(), k=128)
    with pytest.raises(CorpusRefusal):
        MemmapCorpus(pack128, k_max=192)
    corpus = MemmapCorpus(pack128, k_max=128)      # matching pair is fine
    assert corpus.k_max == 128


def test_manifest_slot_dim_disagreement_refuses(pack192):
    import json

    p = pack192 / "manifest.json"
    man = json.loads(p.read_text(encoding="utf-8"))
    man["k_max"] = 128
    p.write_text(json.dumps(man, indent=1), encoding="utf-8")
    with pytest.raises(CorpusRefusal) as ei:
        MemmapCorpus(pack192)
    assert "slot dim" in str(ei.value)


def test_format_claim_must_agree_with_k_max(pack192):
    _set_format(pack192, "v2")
    with pytest.raises(CorpusRefusal):
        MemmapCorpus(pack192)
    _set_format(pack192, "v3")
    corpus = MemmapCorpus(pack192)
    assert corpus.format == "v3" and corpus.k_max == K192
    _set_format(pack192, "v9")
    with pytest.raises(CorpusRefusal):
        MemmapCorpus(pack192)
    # the declared v3 format also refuses an explicit v2 request
    _set_format(pack192, "v3")
    with pytest.raises(CorpusRefusal):
        MemmapCorpus(pack192, k_max=128)


def test_idea_a_config_k_max_parameterisation(pack192):
    cfg = ia.IdeaATrainConfig(k_max=192)
    assert cfg.k_max == 192
    with pytest.raises(ia.IdeaAConfigError):
        ia.IdeaATrainConfig(k_max=300)
    # pack vs model(config) mismatch refuses before any batch is built
    cfg128 = ia.IdeaATrainConfig(k_max=128)
    with pytest.raises(CorpusRefusal):
        MemmapCorpus(pack192, k_max=cfg128.k_max)
    corpus = MemmapCorpus(pack192, k_max=cfg.k_max)
    assert corpus.k_max == 192


def test_k192_pack_end_to_end_through_fit(tmp_path, pack192):
    """One tiny WmN1 member trains on a K=192 pack and checkpoints."""
    torch.set_num_threads(1)
    corpus = MemmapCorpus(pack192, pair_sampling=True, k_max=192)
    from train.config import TrainConfig
    from train.trainer import Trainer

    cfg = TrainConfig(d_model=16, n_heads=2, n_layers=1, ffn=32,
                      bias_hidden=8, counter_channels=8, n_mix=3,
                      epochs=1, steps_per_epoch=2, batch_size=2, lr=1e-2,
                      device="cpu", out_dir=str(tmp_path / "members"))
    receipt = Trainer(cfg, corpus, member=1, seed=cfg.seed_for_member(1),
                      device="cpu").fit()
    assert receipt["steps"] == 2
    out = tmp_path / "members"
    assert (out / "member_1.pt").is_file()
    assert np.isfinite(receipt["losses"]).all()
