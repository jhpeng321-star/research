"""Scorer comparator v2g tests: generated fixtures only (no real packs)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from evalh.protocol import N_QUANTILES, QUANTILE_LEVELS, OutcomeRow
from evalh.scorer import comparator_v2g as C

PARAMS = {
    "dt_s": 0.5,
    "geometry": {
        "lane_segment_starts_m": [[0.0], [0.0], [0.0]],
        "lane_segment_limits_mps": [[25.0], [30.0], [20.0]],
    },
}

ENGINE_FIXTURE = '''\
"""Generated fixture engine (tests only)."""
import numpy as np


def segment_limits(lane, s, params):
    lane = np.asarray(lane)
    s = np.asarray(s, dtype=float)
    starts = params["geometry"]["lane_segment_starts_m"]
    lims = params["geometry"]["lane_segment_limits_mps"]
    out = np.full(len(lane), np.nan, dtype=float)
    for li in (0, 1, 2):
        m = lane == li
        if not m.any():
            continue
        idx = np.searchsorted(np.asarray(starts[li]), s[m],
                              side="right") - 1
        idx = np.clip(idx, 0, len(lims[li]) - 1)
        out[m] = np.asarray(lims[li])[idx]
    return out
'''

K = 4


class FakeRollouts:
    """[S=1, T+1, B=1, K] constant-speed numpy stand-in for Rollouts."""

    def __init__(self, n_steps, speed=10.0, present=True, samples=1):
        shape = (samples, n_steps + 1, 1, K)
        self.positions = np.full(shape, 100.0)
        self.speeds = np.full(shape, speed)
        self.masks = np.full(shape, present, dtype=bool)
        self.lanes = np.full(shape, 1, dtype=np.int64)


def _write_fixtures(tmp_path, params_text=None):
    root = tmp_path / "partA_v2g"
    root.mkdir()
    params = root / "params_nominal.json"
    params.write_bytes(
        (params_text if params_text is not None
         else json.dumps(PARAMS, indent=2) + "\n").encode("utf-8"))
    engine = root / "engine.py"
    engine.write_text(ENGINE_FIXTURE, encoding="utf-8")
    return root, params, engine


def _lf(path):
    raw = Path(path).read_bytes()
    return hashlib.sha256(
        raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _bound(tmp_path):
    root, params, engine = _write_fixtures(tmp_path)
    record = C.bind_v2g_artifacts(
        root, expected_params_lf_sha256=_lf(params),
        expected_engine_sha256=hashlib.sha256(
            engine.read_bytes()).hexdigest())
    return record, params, engine


def _resolver(record_items):
    def resolve(item):
        return record_items
    return resolve


def _fake_rollout(recorder, *, n_steps=None, speed=10.0, present=True,
                  samples=1):
    def rollout(bundle, world, schedule, *, n_steps, integrator,
                residual_enabled, n_samples, seed, t0_s):
        recorder.append(dict(n_steps=n_steps, integrator=integrator,
                             residual_enabled=residual_enabled,
                             n_samples=n_samples, seed=seed, t0_s=t0_s))
        return FakeRollouts(n_steps, speed=speed, present=present,
                            samples=samples)
    return rollout


def _rows(taus=(30.0, 60.0)):
    return [OutcomeRow(world="w", t0_s=0.0, decision_t_s=0.0, tau_s=tau,
                       policy="native", y_delay_s=1.0) for tau in taus]


def test_bind_accepts_matching_fixture_and_records_hashes(tmp_path):
    record, params, engine = _bound(tmp_path)
    assert record["params_lf_sha256"] == _lf(params)
    assert record["engine_sha256"] == hashlib.sha256(
        engine.read_bytes()).hexdigest()
    assert record["params"]["dt_s"] == 0.5
    assert record["params_path"].name == "params_nominal.json"


def test_bind_refuses_registered_pin_on_synthetic_params(tmp_path):
    root, _, _ = _write_fixtures(tmp_path)
    with pytest.raises(C.V2GArtifactRefusal, match="mismatch"):
        C.bind_v2g_artifacts(root)


def test_bind_refuses_wrong_explicit_pin(tmp_path):
    root, _, _ = _write_fixtures(tmp_path)
    with pytest.raises(C.V2GArtifactRefusal, match="mismatch"):
        C.bind_v2g_artifacts(root, expected_params_lf_sha256="0" * 64)


def test_bind_refuses_tampered_params(tmp_path):
    record, params, _ = _bound(tmp_path)
    good = params.read_bytes()
    params.write_bytes(good + b" ")
    with pytest.raises(C.V2GArtifactRefusal, match="mismatch"):
        C.bind_v2g_artifacts(
            record["root"], expected_params_lf_sha256=record["params_lf_sha256"])


def test_bind_refuses_missing_files(tmp_path):
    record, params, engine = _bound(tmp_path)
    (record["root"] / "engine.py").unlink()
    with pytest.raises(C.V2GArtifactRefusal, match="absent"):
        C.bind_v2g_artifacts(record["root"],
                             expected_params_lf_sha256=_lf(params))


def test_predictor_fails_closed_without_option(tmp_path):
    record, _, _ = _bound(tmp_path)
    with pytest.raises(C.ComparatorUnavailable,
                       match=r"total_delay_ref_s.*trajectory_window_delay"
                             r".*hybrid_correction_head"):
        C.V2GBaselinePredictor(record)


def test_predictor_refuses_unknown_and_hybrid_options(tmp_path):
    record, _, _ = _bound(tmp_path)
    with pytest.raises(C.ComparatorUnavailable, match="unknown v2g delay"):
        C.V2GBaselinePredictor(record, option="zero_substitute")
    with pytest.raises(C.ComparatorUnavailable, match="not implemented"):
        C.V2GBaselinePredictor(record, option=C.OPTION_HYBRID_HEAD)


def test_trajectory_option_requires_window_resolver(tmp_path):
    record, _, _ = _bound(tmp_path)
    with pytest.raises(C.ComparatorUnavailable, match="resolve_window"):
        C.V2GBaselinePredictor(record, option=C.OPTION_TRAJECTORY_WINDOW)


def test_artifacts_tampered_after_bind_refused(tmp_path):
    record, params, engine = _bound(tmp_path)
    params.write_bytes(params.read_bytes() + b" ")
    with pytest.raises(C.V2GArtifactRefusal, match="changed after bind"):
        C.V2GBaselinePredictor(
            record, option=C.OPTION_TRAJECTORY_WINDOW,
            resolve_window=_resolver((None, None, None)),
            rollout_fn=_fake_rollout([]))


def test_shape_monotone_degenerate_and_deterministic(tmp_path):
    record, _, _ = _bound(tmp_path)
    calls = []
    predictor = C.V2GBaselinePredictor(
        record, option=C.OPTION_TRAJECTORY_WINDOW,
        resolve_window=_resolver((object(), object(), object())),
        rollout_fn=_fake_rollout(calls))
    first = predictor.predict_quantiles(_rows())
    second = predictor.predict_quantiles(_rows())
    assert first.shape == (2, N_QUANTILES)
    assert (np.diff(first, axis=1) >= 0.0).all()
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    # degenerate: every level of a row equals its point delay
    assert np.array_equal(first, np.repeat(first[:, :1], N_QUANTILES, axis=1))
    assert predictor.quantile_levels is QUANTILE_LEVELS
    # the registered v2g baseline switches, per item tau
    assert len(calls) == 4
    assert all(c["integrator"] == "v2g" for c in calls)
    assert all(c["residual_enabled"] is False for c in calls)
    assert all(c["n_samples"] == 1 for c in calls)
    assert [c["n_steps"] for c in calls] == [60, 120, 60, 120]


def test_trajectory_delay_value_is_exact_not_zeros(tmp_path):
    record, _, _ = _bound(tmp_path)
    calls = []
    predictor = C.V2GBaselinePredictor(
        record, option=C.OPTION_TRAJECTORY_WINDOW,
        resolve_window=_resolver((None, None, None)),
        rollout_fn=_fake_rollout(calls))
    q = predictor.predict_quantiles(_rows(taus=(30.0,)))
    # 60 steps x 0.5 s x (1 - 10/30) x 4 tracked vehicles = 80 veh-s
    expected = 60 * 0.5 * (1.0 - 10.0 / 30.0) * K
    assert q[0, 0] == pytest.approx(expected)
    assert (q != 0.0).all()


def test_empty_t0_roster_refused_not_zero(tmp_path):
    record, _, _ = _bound(tmp_path)
    predictor = C.V2GBaselinePredictor(
        record, option=C.OPTION_TRAJECTORY_WINDOW,
        resolve_window=_resolver((None, None, None)),
        rollout_fn=_fake_rollout([], present=False))
    with pytest.raises(C.ComparatorUnavailable, match="empty t0 roster"):
        predictor.predict_quantiles(_rows(taus=(30.0,)))


def test_short_rollout_and_multisample_refused(tmp_path):
    record, _, _ = _bound(tmp_path)

    def short(bundle, world, schedule, *, n_steps, integrator,
              residual_enabled, n_samples, seed, t0_s):
        return FakeRollouts(10)

    predictor = C.V2GBaselinePredictor(
        record, option=C.OPTION_TRAJECTORY_WINDOW,
        resolve_window=_resolver((None, None, None)),
        rollout_fn=short)
    with pytest.raises(C.ComparatorUnavailable, match="shorter"):
        predictor.predict_quantiles(_rows(taus=(30.0,)))

    def multi(bundle, world, schedule, *, n_steps, integrator,
              residual_enabled, n_samples, seed, t0_s):
        return FakeRollouts(n_steps, samples=3)

    predictor = C.V2GBaselinePredictor(
        record, option=C.OPTION_TRAJECTORY_WINDOW,
        resolve_window=_resolver((None, None, None)),
        rollout_fn=multi)
    with pytest.raises(C.ComparatorUnavailable, match="samples"):
        predictor.predict_quantiles(_rows(taus=(30.0,)))


def test_real_lazy_actions_path_used_without_injection(tmp_path, monkeypatch):
    record, _, _ = _bound(tmp_path)
    predictor = C.V2GBaselinePredictor(
        record, option=C.OPTION_TRAJECTORY_WINDOW,
        resolve_window=_resolver((None, None, None)))
    seen = {}

    def fake_with_schedule(bundle, world, schedule, *, n_steps,
                           integrator, residual_enabled, n_samples, seed,
                           t0_s):
        seen["kwargs"] = dict(integrator=integrator,
                              residual_enabled=residual_enabled,
                              n_steps=n_steps)
        return FakeRollouts(n_steps)

    import types
    actions_mod = types.ModuleType("evalh.scorer.actions")
    actions_mod.rollout_with_schedule = fake_with_schedule
    monkeypatch.setitem(sys.modules, "evalh.scorer.actions", actions_mod)
    q = predictor.predict_quantiles(_rows(taus=(30.0,)))
    assert q.shape == (1, N_QUANTILES)
    assert seen["kwargs"]["integrator"] == "v2g"
    assert seen["kwargs"]["residual_enabled"] is False
    assert seen["kwargs"]["n_steps"] == 60
