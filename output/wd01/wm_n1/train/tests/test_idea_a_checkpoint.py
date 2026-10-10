"""P7 Idea A checkpoint/resume + partial-ensemble flags (no training data).

Plain state_dict + JSON sidecar, explicit ``weights_only`` on load, ensemble
``k`` in 1..5 recorded with ``partial``/``final_test_eligible`` flags, and a
resume that reproduces the identical next-step loss.
"""
from __future__ import annotations

import json

import pytest
import torch

import train.idea_a as ia
from test_idea_a_fakes import make_windows, seeded_bundle


def _config(**cfg) -> ia.IdeaATrainConfig:
    kwargs = dict(epochs=1, steps_per_epoch=2, batch_size=2, lr=0.01,
                  out_dir="unused")
    kwargs.update(cfg)
    return ia.IdeaATrainConfig(**kwargs)


def _trained_trainer(tmp_path, k: int):
    windows = make_windows(2)
    cfg = _config(ensemble_k=k, out_dir=str(tmp_path / "out"))
    tr = ia.IdeaATrainer(cfg, corpus=None, bundle=seeded_bundle(0))
    tr.fit(windows=windows)
    ckpt = tmp_path / "out" / "member_1.pt"
    assert ckpt.is_file()
    return tr, ckpt, windows


@pytest.mark.parametrize("k", [1, 5])
def test_partial_ensemble_checkpoint_roundtrip(tmp_path, k):
    tr, ckpt, _ = _trained_trainer(tmp_path, k)
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert payload["format"] == ia.CHECKPOINT_FORMAT
    assert payload["ensemble_k"] == k
    assert payload["members"] == list(ia.ensemble_members(k))
    assert payload["partial"] is (k < 5)
    assert payload["final_test_eligible"] is (k == 5)
    sidecar = json.loads(ckpt.with_suffix(".json").read_text("utf-8"))
    assert sidecar["ensemble_k"] == k
    assert sidecar["partial"] is (k < 5)
    assert "state_dict" not in sidecar
    receipt = json.loads(
        (tmp_path / "out" / "TRAIN_RECEIPT.json").read_text("utf-8"))
    assert receipt["ensemble_k"] == k and receipt["partial"] is (k < 5)


def test_require_full_ensemble_refuses_partial():
    for k in (1, 2, 3, 4):
        with pytest.raises(ia.IdeaAConfigError):
            ia.require_full_ensemble(k)
        with pytest.raises(ia.IdeaAConfigError):
            ia.require_full_ensemble({"ensemble_k": k})
    assert ia.require_full_ensemble(5) == (1, 2, 3, 4, 5)
    assert ia.require_full_ensemble({"ensemble_k": 5}) == (1, 2, 3, 4, 5)


def test_resume_gives_identical_next_step_loss(tmp_path):
    tr, ckpt, windows = _trained_trainer(tmp_path, k=1)
    fixed = ia.collate_canonical(windows)
    # one more step on the original trainer (after the checkpoint)
    opt = torch.optim.Adam(tr.parameters(), lr=0.01)
    tr.resume(ckpt, optimizer=opt)
    loss_a = float(tr._optim_step(fixed, opt))
    # a fresh trainer with the same seeded bundle resumes from the file
    cfg = _config(ensemble_k=1, out_dir=str(tmp_path / "out"))
    tr2 = ia.IdeaATrainer(cfg, corpus=None, bundle=seeded_bundle(0))
    opt2 = torch.optim.Adam(tr2.parameters(), lr=0.01)
    tr2.resume(ckpt, optimizer=opt2)
    loss_b = float(tr2._optim_step(fixed, opt2))
    assert loss_a == pytest.approx(loss_b, abs=1e-6)
    flat_a, flat_b = tr.state_dict(), tr2.state_dict()
    assert set(flat_a) == set(flat_b)
    for key in flat_a:
        assert torch.allclose(flat_a[key], flat_b[key]), key


def _advance(trainer, windows, steps: int, total_steps: int) -> list:
    """Run ``steps`` fit-identical optimiser steps on one fixed batch."""
    # fit() fits the TRAIN-only outcome normalization before stepping; these
    # steps must use the same stats to stay fit-identical (M1FIX2 item 1).
    trainer.outcome_norm = ia.fit_outcome_norm(windows=windows)
    opt, sched = ia._make_optimizer(trainer, trainer.config.lr, total_steps)
    trainer._optimizer, trainer._scheduler = opt, sched
    batch = ia.collate_canonical(windows)
    losses = []
    for _ in range(steps):
        losses.append(trainer._optim_step(batch, opt))
        if sched is not None:
            sched.step()
    return losses


def test_resume_fit_continues_optimizer_and_schedule(tmp_path):
    """``resume(); fit()`` must continue the recorded Adam/cosine state.

    A fresh trainer (the crash-restart flow: no optimizer frame yet) that
    resumes a mid-run checkpoint (3 of 6 steps) and fits the remaining epoch
    must reproduce the uninterrupted run; silently restarting the optimizer
    would break resume determinism and waste GPU hours.
    """
    windows = make_windows(1)      # n=1 -> the identical batch every step
    total = 6
    cfg = _config(epochs=2, steps_per_epoch=3, batch_size=2, lr=0.02,
                  ensemble_k=5, out_dir=str(tmp_path / "out"))
    # phase 1: run 3 steps and checkpoint (the interrupted process)
    tr_mid = ia.IdeaATrainer(cfg, corpus=None, bundle=seeded_bundle(0))
    first3 = _advance(tr_mid, windows, 3, total)
    tr_mid.epoch, tr_mid.step = 1, 3
    mid = tmp_path / "mid.pt"
    tr_mid.save_checkpoint(mid)
    # reference: one uninterrupted run of all 6 steps
    tr_b = ia.IdeaATrainer(cfg, corpus=None, bundle=seeded_bundle(0))
    tr_b.fit(windows=windows)
    # phase 2: a FRESH trainer resumes with no optimizer frame and continues
    tr_a = ia.IdeaATrainer(cfg, corpus=None, bundle=seeded_bundle(0))
    assert getattr(tr_a, "_optimizer", None) is None
    tr_a.resume(mid)
    assert getattr(tr_a, "_pending_optimizer_state", None) is not None
    tr_a.fit(windows=windows)
    resumed = list(tr_a.losses)        # fit records only its own steps
    assert len(resumed) == 3 and len(tr_b.losses) == 6
    assert getattr(tr_a, "_pending_optimizer_state", None) is None  # consumed
    for i, (a, b) in enumerate(zip(first3, tr_b.losses[:3])):
        assert a == pytest.approx(b, abs=1e-6), f"step {i} diverged"
    for i, (a, b) in enumerate(zip(resumed, tr_b.losses[3:])):
        assert a == pytest.approx(b, abs=1e-6), f"step {3 + i} diverged"


def test_resume_rejects_wrong_format_and_ensemble_mismatch(tmp_path):
    _, ckpt, _ = _trained_trainer(tmp_path, k=1)
    tr = ia.IdeaATrainer(_config(ensemble_k=5, out_dir="unused"),
                         corpus=None, bundle=seeded_bundle(7))
    with pytest.raises(ia.IdeaAConfigError):
        tr.resume(ckpt)              # ensemble_k 1 (file) != 5 (config)
    bad = tmp_path / "bad.pt"
    torch.save({"format": "other/9"}, bad)
    with pytest.raises(ia.IdeaAConfigError):
        tr.resume(bad)


def test_state_dict_namespacing_and_missing_keys(tmp_path):
    tr = ia.IdeaATrainer(_config(out_dir="unused"), corpus=None,
                         bundle=seeded_bundle(3))
    flat = tr.state_dict()
    prefixes = {key.split(".", 1)[0] for key in flat}
    assert prefixes == {"hdv", "arrival", "corr"}
    tr.load_state_dict(flat)                      # round-trips
    with pytest.raises(ia.IdeaAConfigError):
        tr.load_state_dict({"nobody.weight": torch.zeros(1)})
