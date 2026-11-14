"""Amendment 5 K_max on the Idea A ``main()`` path: v3 (K=192) and v2 (K=128).

A tiny TRAIN pack is built the real way (``corpus.tests.synth`` shards ->
``corpus.idea_a.write_idea_a_channels`` -> ``corpus.pack.pack_shards``) and
trained through ``train.idea_a.main`` - the exact GPU-box argv - so the
``TRAIN_RECEIPT.json`` on disk records every registered A3.2 loss term with the
FORMAT slot width threaded from ``--k-max`` into ``IdeaACorpus`` ->
``MemmapCorpus``.  The cumulative rollout mask and the run-boundary window are
re-checked at K=192.  tmp dirs only; no data reads, no SUMO, no D:.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import train.idea_a as ia
from model import constants as C
from train.dataset import CorpusRefusal
from train.tests import packsynth

K128 = int(C.K_MAX_V2)
K192 = int(C.K_MAX_V3)


def _build_idea_a_pack(tmp_path, k: int):
    from corpus import idea_a as CIA
    from corpus import pack as PK
    from corpus.tests import synth as CS

    root = tmp_path / f"shards_k{k}"
    CS.write_shards(root, k=k)
    index = json.loads((root / "INDEX.json").read_text(encoding="utf-8"))
    for key, rec in index["runs"].items():
        shard_dir = root / rec["shard_dir"]
        arrs = {name: np.load(shard_dir / f"{name}.npy")
                for name in CS.NPY_FILES}
        meta = json.loads((shard_dir / "meta.json").read_text(encoding="utf-8"))
        meta["n_ticks"] = int(arrs["veh_mask"].shape[0])
        meta.setdefault("split", "TRAIN")
        shard = dict(arrs, slot_vid=[], outcomes=None, meta=meta)
        out = CIA.write_idea_a_channels(shard, shard_dir)
        assert out["status"] == "written", key
    pack = tmp_path / f"pack_k{k}"
    PK.pack_shards(root, pack, workers=1)
    return pack


# ------------------------------------------------------ IdeaACorpus k_max -----
def test_idea_a_corpus_threads_k_max(tmp_path):
    pack = _build_idea_a_pack(tmp_path, K192)
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN", k_max=K192)
    assert corpus.k_max == K192
    assert corpus._v1.k_max == K192
    with pytest.raises(CorpusRefusal) as ei:
        ia.IdeaACorpus(str(pack), split="TRAIN", k_max=K128)
    assert "K_max mismatch" in str(ei.value)
    # K is taken from the pack when None
    assert ia.IdeaACorpus(str(pack), split="TRAIN").k_max == K192


# --------------------------------------------- main() exact path (v3 + v2) ----
@pytest.mark.parametrize("k", [K192, K128])
def test_main_path_fit_active_five_losses(tmp_path, k):
    pack = _build_idea_a_pack(tmp_path, k)
    out = tmp_path / f"out_k{k}"
    rc = ia.main(["--member", "1", "--seed", "101", "--corpus", str(pack),
                  "--out", str(out), "--device", "cpu", "--epochs", "1",
                  "--steps-per-epoch", "2", "--k-max", str(k)])
    assert rc == 0
    receipt = json.loads((out / "TRAIN_RECEIPT.json")
                         .read_text(encoding="utf-8"))
    assert set(receipt["active_loss_terms"]) == set(ia.REQUIRED_LOSS_TERMS), \
        receipt["active_loss_terms"]
    assert receipt["allow_incomplete_losses"] is False
    assert receipt["required_loss_terms"] == list(ia.REQUIRED_LOSS_TERMS)
    assert receipt["steps"] == 2
    assert (out / "member_1.pt").is_file()
    assert np.isfinite(receipt["losses"]).all()


def test_main_path_k_mismatch_refuses(tmp_path):
    pack = _build_idea_a_pack(tmp_path, K192)
    with pytest.raises(CorpusRefusal):
        ia.main(["--member", "1", "--seed", "101", "--corpus", str(pack),
                 "--out", str(tmp_path / "out"), "--device", "cpu",
                 "--epochs", "1", "--steps-per-epoch", "1", "--k-max",
                 str(K128)])


# ------------------------------------------- cumulative mask at K=192 --------
def _reuse_runs(n_ticks: int = 10):
    return [dict(chunk="chunk4", world="w1", arm="base", policy="damping",
                 split="TRAIN", t0_s=540.0, n_ticks=n_ticks, n_dec=2)]


def _patch_slot_reuse(shard, shard_dir):
    veh = np.asarray(shard["veh_step"])
    mask = np.asarray(shard["veh_mask"])
    tm = np.asarray(shard["target_mask"])
    acc = np.asarray(shard["accel_next"])
    n, k = mask.shape
    for t in range(n):
        present3 = (t <= 4) or (t >= 6)
        mask[t, 3] = 1 if present3 else 0
        veh[t, 3, C.PRESENT_IDX] = 1.0 if present3 else 0.0
        veh[t, 3, C.IS_CAV_IDX] = 0.0
        veh[t, 3, C.ACT_NATIVE] = 0.0
        veh[t, 3, C.ACT_UNKNOWN] = 1.0
        present1 = t <= 1
        mask[t, 1] = 1 if present1 else 0
        veh[t, 1, C.PRESENT_IDX] = 1.0 if present1 else 0.0
        veh[t, 1, C.IS_CAV_IDX] = 1.0
        veh[t, 1, C.ACT_NATIVE] = 1.0
        veh[t, 1, C.ACT_UNKNOWN] = 0.0
        for s in range(k):
            acc[t, s] = 0.05 * (t + 1) + 0.01 * s
    tm[:, 3] = 0
    tm[0:4, 3] = 1
    tm[4, 3] = 0
    tm[5, 3] = 0
    tm[6:9, 3] = 1
    tm[:, 1] = 0
    tm[0:2, 1] = 1


class _RecordingHdv(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.masks = []

    @property
    def cfg(self):
        return self.inner.cfg

    def forward(self, kin, kin_mask, **kw):
        self.masks.append(kin_mask.detach().clone())
        return self.inner(kin, kin_mask, **kw)


def test_slot_reuse_mask_cumulative_at_k192(tmp_path):
    pack = packsynth.write_idea_a_pack(tmp_path, _reuse_runs(10), k=K192,
                                       patch=_patch_slot_reuse)
    cfg = ia.IdeaATrainConfig(out_dir=str(tmp_path / "out"),
                              rollout_horizon_steps=8, ensemble_k=1)
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN", horizon_steps=8,
                            k_max=K192)
    real = ia.build_idea_a_bundle(cfg)
    rec = _RecordingHdv(real.hdv)
    rec.eval()
    trainer = ia.IdeaATrainer(cfg, corpus=None, member=1, seed=101,
                              bundle=ia.IdeaABundle(hdv=rec, config=cfg))
    win = corpus[0]
    rm = win["rollout_mask"].squeeze(1)                  # [horizon, K]
    assert bool(rm[:4, 3].all())
    assert not bool(rm[4:, 3].any())                     # never re-alives
    rec.masks.clear()
    base = trainer.rollout_loss(win)
    assert torch.isfinite(base)
    assert all(bool(rec.masks[t][0, 3]) for t in range(4))
    assert all(not bool(rec.masks[t][0, 3])
               for t in range(4, len(rec.masks)))
    assert all(not bool(rec.masks[t][0, 1]) for t in range(2, len(rec.masks)))


def test_window_never_crosses_run_boundary_at_k192(tmp_path):
    runs = [
        dict(chunk="chunk4", world="w1", arm="base", policy="damping",
             split="TRAIN", t0_s=540.0, n_ticks=6, n_dec=2),
        dict(chunk="chunk4", world="w2", arm="native", policy="native",
             split="TRAIN", t0_s=600.0, n_ticks=6, n_dec=2),
    ]
    pack = packsynth.write_idea_a_pack(tmp_path, runs, k=K192)
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN", horizon_steps=20,
                            k_max=K192)
    assert sorted(corpus._runs) == [(0, 6), (6, 12)]
    boundary = dict(corpus._v1[0], tick=5)
    assert "teacher" not in corpus.window(boundary)      # horizon capped to 0
    start = corpus.window(dict(corpus._v1[0], tick=0))
    assert int(start["rollout_mask"].squeeze(1).any(dim=-1).sum()) == 5
