"""A3.5 CAV-motion OOD calibration: CAL-only guard, threshold, score."""
from __future__ import annotations

import numpy as np
import pytest

from evalh import idea_a as ia
from evalh.seals import HoldoutRefusal, SealError


def _draw(n, f=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n, f))


def test_calibration_is_cal_only_guarded():
    train, cal = _draw(32), _draw(16, seed=1)
    with pytest.raises(SealError):
        ia.calibrate_ood(train, cal, split="final_test")
    with pytest.raises(SealError):
        ia.calibrate_ood(train, cal, split="train_oof")
    with pytest.raises(HoldoutRefusal):
        ia.calibrate_ood(train, cal, worlds=["HOLDOUT__w000"])


def test_threshold_is_cal_quantile_and_unfamiliar_scores_high():
    train, cal = _draw(64, seed=2), _draw(32, seed=3)
    cfg = ia.OodConfig(cal_quantile=0.90)
    calib = ia.calibrate_ood(train, cal, cfg=cfg)
    assert calib["split"] == "cal"
    assert calib["n_train"] == 64 and calib["n_cal"] == 32
    assert np.isfinite(calib["threshold"])
    familiar = ia.ood_score(cal, calib)
    far = ia.ood_score(np.full((1, 4), 50.0), calib)
    assert far[0] > calib["threshold"]
    assert float(np.mean(ia.hold_back(familiar, calib))) <= 0.15
    assert bool(ia.hold_back(far, calib)[0])


def test_calibration_uses_injected_corr_ood_primitives():
    train, cal = _draw(20, seed=4), _draw(10, seed=5)
    seen = {}

    def fit_fn(summaries, *, min_summary_std):
        seen["fit"] = np.asarray(summaries, float).shape
        return {"mean": np.zeros(summaries.shape[1]),
                "std": np.ones(summaries.shape[1]),
                "min_summary_std": min_summary_std}

    def score_fn(summaries, model):
        x = np.asarray(summaries, float)
        return np.sum(x * x, axis=1)

    calib = ia.calibrate_ood(train, cal, cfg=ia.OodConfig(cal_quantile=0.8),
                             fit_fn=fit_fn, score_fn=score_fn)
    assert seen["fit"] == (20, 4)
    assert np.isclose(calib["threshold"],
                      np.quantile(np.sum(cal * cal, axis=1), 0.8))
    out = ia.ood_score(cal, calib, score_fn=score_fn)
    assert np.allclose(out, np.sum(cal * cal, axis=1))


def test_summary_shape_validation():
    with pytest.raises(ValueError):
        ia.calibrate_ood(np.zeros(4), np.zeros((2, 4)))
    with pytest.raises(ValueError):
        ia.calibrate_ood(np.zeros((4, 3)), np.zeros((2, 4)))
