"""Real-corpus Idea A batch builder: full canonical batch, fail-closed fit.

A tiny TRAIN pack is built the real way - ``corpus.tests.synth`` shards get the
P9 Idea A channels (:func:`corpus.idea_a.write_idea_a_channels`) and are packed
with :func:`corpus.pack.pack_shards` - then trained through the exact
``train.idea_a`` corpus path (``IdeaACorpus`` + ``build_idea_a_bundle`` +
``IdeaATrainer.fit``) that ``train/idea_a.py:main`` uses.  tmp dirs only; no
data reads, no SUMO, no D:.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import train.idea_a as ia
from model import constants as C
from train.dataset import MemmapCorpus
from train.tests import packsynth


def _load_shard(shard_dir) -> dict:
    from corpus.tests import synth as CS

    arrs = {name: np.load(shard_dir / f"{name}.npy") for name in CS.NPY_FILES}
    meta = json.loads((shard_dir / "meta.json").read_text(encoding="utf-8"))
    meta["n_ticks"] = int(arrs["veh_mask"].shape[0])
    meta.setdefault("split", "TRAIN")
    return dict(arrs, slot_vid=[], outcomes=None, meta=meta)


def _build_idea_a_pack(tmp_path):
    """Shards -> P9 channels -> corpus.pack; a real-format TRAIN pack."""
    from corpus import idea_a as CIA
    from corpus import pack as PK
    from corpus.tests import synth as CS

    root = tmp_path / "shards"
    CS.write_shards(root)
    index = json.loads((root / "INDEX.json").read_text(encoding="utf-8"))
    for key, rec in index["runs"].items():
        shard_dir = root / rec["shard_dir"]
        out = CIA.write_idea_a_channels(_load_shard(shard_dir), shard_dir)
        assert out["status"] == "written", key
    pack_dir = tmp_path / "pack"
    PK.pack_shards(root, pack_dir, workers=1)
    return pack_dir


def _trainer(pack_dir, out_dir, **overrides):
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=2, batch_size=4,
                              out_dir=str(out_dir), **overrides)
    corpus = ia.IdeaACorpus(str(pack_dir), split="TRAIN", pair_sampling=True,
                            horizon_steps=cfg.rollout_horizon_steps)
    trainer = ia.IdeaATrainer(cfg, corpus, member=1, seed=101,
                              bundle=ia.build_idea_a_bundle(cfg))
    return cfg, corpus, trainer


def test_main_path_fit_activates_every_registered_loss(tmp_path):
    pack = _build_idea_a_pack(tmp_path)
    cfg, corpus, trainer = _trainer(pack, tmp_path / "out")
    receipt = trainer.fit()
    active = set(receipt["active_loss_terms"])
    assert active == set(ia.REQUIRED_LOSS_TERMS), receipt["active_loss_terms"]
    assert receipt["allow_incomplete_losses"] is False
    assert receipt["required_loss_terms"] == list(ia.REQUIRED_LOSS_TERMS)

    weights = cfg.loss_weights()
    items = [corpus[i] for i in range(len(corpus))]
    batch = ia.collate_canonical(items)
    terms = trainer.loss_terms(batch, require_all=True)
    for name in ia.REQUIRED_LOSS_TERMS:
        assert name in terms
        assert weights[name] != 0.0
        assert torch.isfinite(terms[name]).all(), (name, terms[name])
    # the receipt on disk carries the active-term list (GPU-box evidence)
    disk = json.loads((tmp_path / "out" / "TRAIN_RECEIPT.json")
                      .read_text(encoding="utf-8"))
    assert set(disk["active_loss_terms"]) == set(ia.REQUIRED_LOSS_TERMS)


def test_baseline_idm_is_the_hdv_base_idm_column(tmp_path):
    pack = _build_idea_a_pack(tmp_path)
    _, corpus, _ = _trainer(pack, tmp_path / "out")
    for i in range(len(corpus)):
        window = corpus[i]
        assert torch.allclose(window["baseline_idm"][0],
                              window["hdv_base"][0][:, ia.HDV_BASE_IDM_IDX],
                              atol=1e-4)


def test_teacher_cav_slots_equal_recorded_kinematics(tmp_path):
    pack = _build_idea_a_pack(tmp_path)
    _, corpus, _ = _trainer(pack, tmp_path / "out")
    item = corpus[0]
    tick = int(corpus._items[corpus._order[0]]["tick"])
    kin_all = np.asarray(corpus._array("veh_kin"))
    cav_slots = torch.nonzero(item["is_cav"][0]).flatten().tolist()
    assert cav_slots, "the fixture must carry CAV slots"
    assert item["summary_raw"].shape[-1] == 16
    real_steps = int(item["rollout_mask"].squeeze(1).any(dim=-1).sum())
    assert real_steps >= 1
    for t in range(real_steps + 1):
        row = kin_all[tick + t]
        for c in cav_slots:
            assert item["teacher"][t, 0, c, ia.TEACHER_S] == pytest.approx(
                float(row[c, 0]), abs=1e-3)
            assert item["teacher"][t, 0, c, ia.TEACHER_V] == pytest.approx(
                float(row[c, 4]), abs=1e-3)
            lane = int(np.argmax(row[c, 1:4])) if row[c, 1:4].sum() > 0 else -1
            assert int(item["teacher"][t, 0, c, ia.TEACHER_LANE]) == lane


def test_g4_overfit_runs_through_the_corpus_builder(tmp_path):
    from train.g4_real import real_bundle
    from train.g4_standins import make_windows

    pack = ia.write_g4_tiny_pack(make_windows(2, seed=0), tmp_path / "g4pack")
    cfg = ia.IdeaATrainConfig(out_dir="unused", grad_clip=0.0,
                              rollout_horizon_steps=3)
    probe = ia.corpus_overfit_probe(pack, cfg, bundle=real_bundle(101),
                                    max_steps=25, lr=0.03)
    assert set(probe["terms_seen"]) == set(ia.REQUIRED_LOSS_TERMS)
    assert all(torch.isfinite(torch.tensor(v))
               for v in probe["terms"].values())
    # the default G4 report now goes through this corpus path
    report = ia.overfit_report(max_steps=5, lr=0.03)
    assert report["corpus_path"] is True
    assert set(report["terms_seen"]) == set(ia.REQUIRED_LOSS_TERMS)


def test_pack_without_idea_a_channels_is_refused(tmp_path):
    pack = tmp_path / "v1pack"
    packsynth.write_pack(pack, packsynth.make_runs())
    with pytest.raises(ia.IdeaAConfigError):
        ia.IdeaACorpus(str(pack), split="TRAIN")


def test_fit_fails_closed_on_v1_corpus_and_flag_relaxes_it(tmp_path):
    pack = tmp_path / "v1pack"
    packsynth.write_pack(pack, packsynth.make_runs())

    def _mk(flag, out):
        cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=2, batch_size=4,
                                  out_dir=str(out),
                                  allow_incomplete_losses=flag)
        corpus = MemmapCorpus(str(pack), split="TRAIN", pair_sampling=True)
        return ia.IdeaATrainer(cfg, corpus, member=1, seed=101,
                               bundle=ia.build_idea_a_bundle(cfg))

    with pytest.raises(ia.IdeaAConfigError):
        _mk(False, tmp_path / "refuse").fit()

    receipt = _mk(True, tmp_path / "allowed").fit()
    assert receipt["allow_incomplete_losses"] is True
    assert set(receipt["active_loss_terms"]) < set(ia.REQUIRED_LOSS_TERMS)


# ------------------------------- pair-aware sampler on the real corpus path --
def _many_group_runs(groups: int, arms_per_group: int):
    """``groups`` worlds each with one policy-native reference and
    ``arms_per_group - 1`` treatments (switching/damping), so a random batch of
    ``arms_per_group`` items would rarely complete a sibling pair."""
    runs = []
    for g in range(groups):
        world = f"w{g}"
        t0 = 100.0 + g
        runs.append(dict(chunk="chunk4", world=world, arm="native",
                         policy="native", split="TRAIN", t0_s=t0))
        for a in range(arms_per_group - 1):
            policy = "switching" if a % 2 else "damping"
            arm = f"switching{a + 1}" if policy == "switching" else f"base{a + 1}"
            runs.append(dict(chunk="chunk4", world=world, arm=arm,
                             policy=policy, split="TRAIN", t0_s=t0))
    return runs


def test_main_path_fit_activates_every_loss_on_a_many_group_pack(tmp_path):
    """The real ``main``-path fit on 40 groups x 8 arms (320 items): random
    batches of 8 would almost never share a sibling pair, so the pair-aware
    sampler is what keeps the CONTRAST term computable with the fit failing
    closed."""
    from bundle.e2e import write_synthetic_pack

    pack = tmp_path / "pack"
    write_synthetic_pack(pack, k=16, n_ticks=3, seed=0,
                         runs=_many_group_runs(40, 8))
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=6, batch_size=8,
                              out_dir=str(tmp_path / "out"))
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN", pair_sampling=True,
                            horizon_steps=cfg.rollout_horizon_steps)
    trainer = ia.IdeaATrainer(cfg, corpus, member=1, seed=101,
                              bundle=ia.build_idea_a_bundle(cfg))
    receipt = trainer.fit()
    assert set(receipt["active_loss_terms"]) == set(ia.REQUIRED_LOSS_TERMS)
    assert receipt["allow_incomplete_losses"] is False


def test_iter_batches_pair_complete_coverage_and_deterministic(tmp_path):
    """Every sampled batch contains a complete pair, one epoch covers every
    item, and the same seed reproduces the same batches."""
    import math

    from train.dataset import contrast_batch_indices

    root = tmp_path / "pack"
    packsynth.write_pack(root, _many_group_runs(3, 4))
    corpus = MemmapCorpus(root, split="TRAIN", pair_sampling=True)
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=5,
                              out_dir=str(tmp_path / "out"))

    def epoch(seed):
        trainer = ia.IdeaATrainer(cfg, corpus, member=1, seed=seed,
                                  bundle=ia.IdeaABundle(config=cfg))
        batches = trainer._iter_batches(corpus)
        return [next(batches)["meta"] for _ in
                range(math.ceil(len(corpus) / 5))]

    first, second = epoch(101), epoch(101)
    assert first == second
    for meta in first:
        assert int(contrast_batch_indices(meta)["a"].numel()) >= 1, meta
    seen = {(it["world"], it["arm"]) for meta in first for it in meta}
    all_ids = {(corpus[i]["world"], corpus[i]["arm"])
               for i in range(len(corpus))}
    assert seen == all_ids


# ------------------------------------------------ cumulative rollout mask ----
def _reuse_runs(n_ticks: int = 10):
    return [dict(chunk="chunk4", world="w1", arm="base", policy="damping",
                 split="TRAIN", t0_s=540.0, n_ticks=n_ticks, n_dec=2)]


def _patch_slot_reuse(shard, shard_dir):
    """Craft: slot 3 vacated after tick 4, reused by a new vehicle at tick 6;
    slot 1 is a CAV that departs after tick 1.  ``target_mask`` is the
    per-step SAME-vehicle transition (non-cumulative), as in the real pack."""
    veh = np.asarray(shard["veh_step"])
    mask = np.asarray(shard["veh_mask"])
    tm = np.asarray(shard["target_mask"])
    acc = np.asarray(shard["accel_next"])
    n, k = mask.shape
    for t in range(n):
        present3 = (t <= 4) or (t >= 6)
        mask[t, 3] = 1 if present3 else 0
        veh[t, 3, C.PRESENT_IDX] = 1.0 if present3 else 0.0
        veh[t, 3, C.IS_CAV_IDX] = 0.0            # slot 3 is HDV
        veh[t, 3, C.ACT_NATIVE] = 0.0
        veh[t, 3, C.ACT_UNKNOWN] = 1.0
        present1 = t <= 1
        mask[t, 1] = 1 if present1 else 0
        veh[t, 1, C.PRESENT_IDX] = 1.0 if present1 else 0.0
        veh[t, 1, C.IS_CAV_IDX] = 1.0            # slot 1 is CAV
        veh[t, 1, C.ACT_NATIVE] = 1.0
        veh[t, 1, C.ACT_UNKNOWN] = 0.0
        for s in range(k):
            acc[t, s] = 0.05 * (t + 1) + 0.01 * s
    tm[:, 3] = 0
    tm[0:4, 3] = 1                               # same vehicle through tick 4
    tm[4, 3] = 0                                 # vacated after tick 4
    tm[5, 3] = 0                                 # empty tick 5 -> new tick 6
    tm[6:9, 3] = 1                               # new vehicle continues
    tm[:, 1] = 0
    tm[0:2, 1] = 1                               # CAV alive steps 0, 1


class _RecordingHdv(torch.nn.Module):
    """Wrap the real HDV net and record the attention mask it receives."""

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


def _slot_reuse_trainer(tmp_path, pack):
    cfg = ia.IdeaATrainConfig(out_dir=str(tmp_path / "out"),
                              rollout_horizon_steps=8, ensemble_k=1)
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN", horizon_steps=8)
    real = ia.build_idea_a_bundle(cfg)
    rec = _RecordingHdv(real.hdv)
    rec.eval()
    trainer = ia.IdeaATrainer(cfg, corpus=None, member=1, seed=101,
                              bundle=ia.IdeaABundle(hdv=rec, config=cfg))
    return corpus, trainer, rec


def test_slot_reuse_rollout_mask_is_cumulative_and_masks_attention(tmp_path):
    pack = packsynth.write_idea_a_pack(tmp_path, _reuse_runs(10), k=8,
                                       patch=_patch_slot_reuse)
    corpus, trainer, rec = _slot_reuse_trainer(tmp_path, pack)
    win = corpus[0]
    rm = win["rollout_mask"].squeeze(1)                  # [horizon, K]
    assert bool(rm[:4, 3].all())
    assert not bool(rm[4:, 3].any())                     # never re-alives

    rec.masks.clear()
    base = trainer.rollout_loss(win)
    assert torch.isfinite(base)
    assert all(bool(rec.masks[t][0, 3]) for t in range(4))
    assert all(not bool(rec.masks[t][0, 3]) for t in range(4, len(rec.masks)))
    assert all(not bool(rec.masks[t][0, 1]) for t in range(2, len(rec.masks)))

    # zero loss contribution for the vacated/reused slot after step 4
    pert = dict(win)
    pert["rollout_accel"] = win["rollout_accel"].clone()
    pert["rollout_accel"][4:, 0, 3] = 1e3
    assert torch.allclose(trainer.rollout_loss(pert), base, atol=1e-6)

    # ...but an ALIVE step still contributes (the test is not vacuous)
    pert2 = dict(win)
    pert2["rollout_accel"] = win["rollout_accel"].clone()
    pert2["rollout_accel"][0, 0, 3] += 5.0
    assert not torch.allclose(trainer.rollout_loss(pert2), base)

    # a departed CAV's teacher columns are not consumed (attention excludes it)
    pert3 = dict(win)
    pert3["teacher"] = win["teacher"].clone()
    pert3["teacher"][3:, 0, 1, ia.TEACHER_S] += 50.0
    pert3["teacher"][3:, 0, 1, ia.TEACHER_A] += 5.0
    assert torch.allclose(trainer.rollout_loss(pert3), base, atol=1e-6)


def test_summary_raw_uses_only_alive_steps(tmp_path):
    pack = packsynth.write_idea_a_pack(tmp_path, _reuse_runs(10), k=8,
                                       patch=_patch_slot_reuse)
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN", horizon_steps=8)
    base = corpus[0]["summary_raw"]
    plan = np.load(pack / "cav_plan_ch.npy", mmap_mode="r+")
    kin = np.load(pack / "veh_kin.npy", mmap_mode="r+")
    # perturb slot 1 (the departing CAV) on its DEAD steps only
    plan[3:9, 1, 0] = (plan[3:9, 1, 0].astype(np.float32) + 0.5).astype(
        np.float16)
    plan[3:9, 1, 1] = (plan[3:9, 1, 1].astype(np.float32) - 0.5).astype(
        np.float16)
    kin[3:9, 1, 0] = (kin[3:9, 1, 0].astype(np.float32) + 7.0).astype(
        np.float16)
    plan.flush()
    kin.flush()
    corpus_b = ia.IdeaACorpus(str(pack), split="TRAIN", horizon_steps=8)
    assert torch.allclose(base, corpus_b[0]["summary_raw"], atol=1e-5)
    # perturbing an ALIVE step does change the summary
    plan[0, 1, 0] = np.float16(plan[0, 1, 0] + 0.5)
    plan[0, 1, 1] = np.float16(plan[0, 1, 1] - 0.5)
    plan.flush()
    corpus_c = ia.IdeaACorpus(str(pack), split="TRAIN", horizon_steps=8)
    assert not torch.allclose(base, corpus_c[0]["summary_raw"], atol=1e-5)


# ------------------------------------------------------- run boundaries ----
def test_window_never_crosses_a_run_boundary(tmp_path):
    runs = [
        dict(chunk="chunk4", world="w1", arm="base", policy="damping",
             split="TRAIN", t0_s=540.0, n_ticks=6, n_dec=2),
        dict(chunk="chunk4", world="w2", arm="native", policy="native",
             split="TRAIN", t0_s=600.0, n_ticks=6, n_dec=2),
    ]
    pack = packsynth.write_idea_a_pack(tmp_path, runs, k=8)
    corpus = ia.IdeaACorpus(str(pack), split="TRAIN", horizon_steps=20)
    assert sorted(corpus._runs) == [(0, 6), (6, 12)]
    assert corpus._run_end(5) == 6                     # last tick of run 1
    assert corpus._run_end(6) == 12                    # first tick of run 2
    boundary = dict(corpus._v1[0], tick=5)
    assert "teacher" not in corpus.window(boundary)    # horizon capped to 0
    start = dict(corpus._v1[0], tick=0)
    win = corpus.window(start)
    assert int(win["rollout_mask"].squeeze(1).any(dim=-1).sum()) == 5


def test_index_json_fails_closed_and_test_only_fallback(tmp_path):
    pack = packsynth.write_idea_a_pack(tmp_path, _reuse_runs(6), k=8)
    idx = pack / "INDEX.json"
    backup = idx.read_bytes()
    idx.rename(pack / "INDEX.json.off")                # no file deletion
    with pytest.raises(ia.IdeaAConfigError):
        ia.IdeaACorpus(str(pack), split="TRAIN")
    fallback = ia.IdeaACorpus(str(pack), split="TRAIN",
                              allow_single_run_fallback=True)
    assert fallback._runs == [(0, 6)]
    (pack / "INDEX.json.off").rename(idx)
    idx.write_text("{not json", encoding="utf-8")
    with pytest.raises(ia.IdeaAConfigError):
        ia.IdeaACorpus(str(pack), split="TRAIN")
    idx.write_text(json.dumps({"runs": [{"world": "w1", "arm": "base"}]}),
                   encoding="utf-8")
    with pytest.raises(ia.IdeaAConfigError):
        ia.IdeaACorpus(str(pack), split="TRAIN")
    idx.write_bytes(backup)
    ia.IdeaACorpus(str(pack), split="TRAIN")           # valid again
