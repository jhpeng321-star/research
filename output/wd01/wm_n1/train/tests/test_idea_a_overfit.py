"""P7 Idea A G4 overfit gate: one member drives the TOTAL loss to ~0.

Two tiny synthetic TRAIN windows (complete sibling pairs) on CPU with the
P2/P4/P5 fakes; the total includes the contrast and rollout terms.  This is
the function the owner pre-flight package calls (:func:`overfit_probe`).
"""
from __future__ import annotations

import time

import train.idea_a as ia
from test_idea_a_fakes import make_windows, seeded_bundle

THRESHOLD = 0.05
MAX_STEPS = 600
OVERFIT_LR = 0.04


def test_g4_overfit_total_loss_near_zero():
    windows = make_windows(2, seed=0)
    bundle = seeded_bundle(101)
    started = time.time()
    probe = ia.overfit_probe(windows, ia.IdeaATrainConfig(out_dir="unused"),
                             bundle=bundle, max_steps=MAX_STEPS,
                             lr=OVERFIT_LR, threshold=THRESHOLD)
    elapsed = time.time() - started
    assert {"response_contrast", "rollout"} <= set(probe["terms_seen"])
    assert probe["terms"]["rollout"] < THRESHOLD
    assert probe["terms"]["response_contrast"] < THRESHOLD
    assert probe["final_total"] < THRESHOLD, probe["terms"]
    assert probe["passed"] is True
    assert probe["history"][0] > probe["final_total"]   # it really learned
    assert elapsed < 120.0, f"overfit probe took {elapsed:.1f}s"


def test_g4_overfit_via_trainer_fit(tmp_path):
    windows = make_windows(2, seed=1)
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=MAX_STEPS,
                              batch_size=2, lr=OVERFIT_LR,
                              ensemble_k=1, out_dir=str(tmp_path / "out"))
    tr = ia.IdeaATrainer(cfg, corpus=None, bundle=seeded_bundle(202))
    receipt = tr.fit(windows=windows)
    final = receipt["losses"][-1]
    assert final < THRESHOLD, receipt["history"]
    assert receipt["partial"] is True and receipt["ensemble_k"] == 1
    assert receipt["losses"][0] > final


def test_g4_real_bundle_uses_real_modules_and_backpropagates():
    import torch

    from arrival import IdeaAArrivalHead
    from corr import CorrectionHead
    from hdv import HdvResidualNet
    from train.g4_real import RealCorrBundle, real_bundle

    bundle = real_bundle(101)
    assert isinstance(bundle.hdv, HdvResidualNet)
    assert isinstance(bundle.arrival, IdeaAArrivalHead)
    assert isinstance(bundle.corr, RealCorrBundle)
    assert isinstance(bundle.corr.head, CorrectionHead)
    batch = ia.collate_canonical(make_windows(2, seed=0))
    tr = ia.IdeaATrainer(ia.IdeaATrainConfig(out_dir="unused"), corpus=None,
                         bundle=bundle)
    terms = tr.loss_terms(batch)
    assert {"residual_nll", "rollout", "arrival", "correction_crps",
            "response_contrast"} <= set(terms)
    sum(terms.values()).backward()
    grads = [p.grad for p in tr.parameters()]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_g4_standin_fallback_fails_the_gate():
    report = ia.overfit_report(max_steps=30, lr=0.04, standins=True)
    assert report["standins"] is True
    assert report["real_modules"] is False
    assert report["ok"] is False
    assert report.get("warnings")

