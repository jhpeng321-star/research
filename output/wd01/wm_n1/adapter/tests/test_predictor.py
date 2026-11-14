"""WmN1Predictor: protocol conformance and rollout-shape collapse."""
from __future__ import annotations

import numpy as np

from adapter.predictor import WmN1Predictor
from evalh.protocol import (Predictor, QUANTILE_LEVELS, RolloutBatch,
                            OutcomeRow)

import synth_adapter as synth


def _predictor(worlds=("w0", "w1")):
    ensemble = synth.tiny_ensemble()
    corpus = synth.fake_corpus(list(worlds))
    return WmN1Predictor(ensemble, corpus, device="cpu")


def test_is_a_predictor_protocol():
    p = _predictor()
    assert isinstance(p, Predictor)
    assert p.name == "wm_n1"
    assert np.asarray(p.quantile_levels).size == 19


def test_predict_quantiles_shape_and_monotone():
    p = _predictor()
    rows = synth.outcome_rows(["w0", "w1"], taus=(60.0, 120.0))
    q = p.predict_quantiles(rows)
    assert q.shape == (len(rows), QUANTILE_LEVELS.size)
    assert np.all(np.diff(q, axis=1) >= -1e-12)
    # the harness's own shape check (PredictorBase._validate) accepts it
    assert p._validate(q, len(rows)).shape == q.shape
    assert p.predict_quantiles([]).shape == (0, QUANTILE_LEVELS.size)


def test_rollout_shape_collapse():
    p = _predictor()
    horizons = (10.0, 30.0)
    rows = synth.rollout_rows(["w0", "w1"], horizons=horizons)
    batch = p.sample_rollouts(rows, horizons)
    assert isinstance(batch, RolloutBatch)
    assert batch.horizons_s == horizons
    assert batch.pos.shape == (len(rows), 2, 6)
    assert batch.speed.shape == (len(rows), 2, 6)
    assert batch.mask.shape == (len(rows), 2, 6)
    assert np.all(np.isfinite(batch.pos))


def test_rollout_uses_row_horizons_when_none():
    p = _predictor()
    rows = synth.rollout_rows(["w0"], horizons=(10.0,))
    batch = p.sample_rollouts(rows, None)
    assert batch.pos.shape == (1, 1, 6)


def test_window_from_features_tensors():
    win = synth.make_window(K=6, seed=9)
    rows = [OutcomeRow(world="wF", t0_s=0.0, decision_t_s=0.0, tau_s=60.0,
                       policy="native", y_delay_s=1.0,
                       features={"tensors": win})]
    p = WmN1Predictor(synth.tiny_ensemble(), corpus=None, device="cpu")
    q = p.predict_quantiles(rows)
    assert q.shape == (1, 19)


def test_action_override_honoured():
    win = synth.make_window(K=6, seed=1)
    rows = [OutcomeRow(world="w0", t0_s=0.0, decision_t_s=0.0, tau_s=60.0,
                       policy="damping", y_delay_s=1.0,
                       features={"tensors": win,
                                 "actions": {0: "damping"}})]
    p = WmN1Predictor(synth.tiny_ensemble(), corpus=None, device="cpu")
    assert p.predict_quantiles(rows).shape == (1, 19)


def test_not_loaded_raises():
    import pytest
    from adapter.predictor import PredictorNotLoadedError
    p = WmN1Predictor(corpus=synth.fake_corpus(["w0"]), device="cpu")
    with pytest.raises(PredictorNotLoadedError):
        p.predict_quantiles(synth.outcome_rows(["w0"]))
