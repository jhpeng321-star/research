"""Metric unit tests on synthetic arrays (registration section 4/6)."""
from __future__ import annotations

import numpy as np
import pytest

from evalh import metrics as M
from evalh.protocol import (HORIZONS_S, QUANTILE_LEVELS, OutcomeRow,
                            RolloutBatch, RolloutRow)


def test_crps_point_mass_equals_abs_error():
    q = np.full((1, QUANTILE_LEVELS.size), 5.0)
    assert M.crps_of_quantiles(q, [8.0])[0] == pytest.approx(3.0)
    assert M.crps_of_quantiles(q, [2.0])[0] == pytest.approx(3.0)


def test_crps_sharper_forecast_is_better():
    y = np.array([0.0])
    wide = np.tile(np.array([-3.0, 3.0]), (1, 1))
    sharp = np.tile(np.array([-0.1, 0.1]), (1, 1))
    cw = M.crps_of_quantiles(wide, y, levels=[0.5, 0.5])[0]
    cs = M.crps_of_quantiles(sharp, y, levels=[0.5, 0.5])[0]
    assert cs < cw


def test_sign_accuracy_threshold_filters_pairs():
    dy = np.array([10.0, -10.0, 0.2, -0.1])
    pred = np.array([1.0, -1.0, -1.0, -1.0])
    acc, n = M.sign_accuracy(pred, dy, threshold=1.0)
    assert n == 2
    assert acc == pytest.approx(1.0)
    acc0, n0 = M.sign_accuracy(pred, dy, threshold=1000.0)
    assert n0 == 0 and np.isnan(acc0)


def test_sign_accuracy_ci_brackets_point_and_uses_worlds():
    pred = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    dy = np.array([5.0, -5.0, 5.0, 2.0, 5.0, -5.0])
    worlds = ["a", "a", "b", "b", "c", "c"]
    acc, n, lo, hi = M.sign_accuracy_ci(pred, dy, worlds, threshold=3.0,
                                        n_boot=200, seed=0)
    assert n == 5                                # |dy| == 2.0 excluded
    assert acc == pytest.approx(1.0)
    assert lo <= acc <= hi
    acc0, n0, lo0, hi0 = M.sign_accuracy_ci(pred, dy, worlds, threshold=999.0)
    assert n0 == 0 and np.isnan(acc0) and np.isnan(lo0) and np.isnan(hi0)


def test_delta_metrics_reports_sign_ci():
    rows = []
    variant_q = []
    hybrid_q = []
    for w in range(6):
        rows.append(OutcomeRow(f"w{w}", 300.0, 120.0, 60.0, "damping", 10.0 + w))
        variant_q.append(np.full(QUANTILE_LEVELS.size, 10.0 + w))
        hybrid_q.append(np.full(QUANTILE_LEVELS.size, 10.0 + w))
        rows.append(OutcomeRow(f"w{w}", 300.0, 120.0, 60.0, "native", 4.0))
        variant_q.append(np.full(QUANTILE_LEVELS.size, 4.0))
        hybrid_q.append(np.full(QUANTILE_LEVELS.size, 8.0))
    delta = M.delta_metrics(
        M.build_delta_pairs(rows, np.array(variant_q), np.array(hybrid_q)),
        threshold=1.0, n_boot=200, seed=0)
    cell = delta["damping-native"][60.0]
    assert cell["variant_sign_ci"][0] <= cell["variant_sign_accuracy"] \
        <= cell["variant_sign_ci"][1]
    assert cell["sign_n_used"] == 6


def test_world_bootstrap_ci_zero_difference_contains_zero():
    values = np.zeros(10)
    worlds = ["a"] * 5 + ["b"] * 5
    lo, hi, point = M.world_bootstrap_ci(values, worlds, n_boot=300, seed=1)
    assert lo == 0.0 and hi == 0.0 and point == 0.0


def test_world_bootstrap_ci_separation_excludes_zero():
    values = np.concatenate([np.full(5, 1.0), np.full(5, 1.5)])
    worlds = ["a"] * 5 + ["b"] * 5
    lo, hi, point = M.world_bootstrap_ci(values, worlds, n_boot=300, seed=1)
    assert lo > 0.0 and hi > 0.0
    assert lo <= point <= hi


def test_coverage_bounds():
    q = np.tile(np.array([-1.0, 1.0]), (4, 1))
    y = np.array([0.0, 0.5, 2.0, -2.0])
    cov = M.coverage(q[:, 0], q[:, 1], y)
    assert cov == pytest.approx(0.5)
    assert 0.0 <= cov <= 1.0


def _pair_rows():
    return [
        OutcomeRow("w0", 300.0, 120.0, 60.0, "damping", 10.0),
        OutcomeRow("w0", 300.0, 120.0, 60.0, "native", 4.0),
    ]


def test_build_delta_pairs_comonotone_comonotone_pairing():
    rows = _pair_rows()
    variant_q = np.vstack([np.full(QUANTILE_LEVELS.size, 8.0),
                           np.full(QUANTILE_LEVELS.size, 3.0)])
    hybrid_q = np.vstack([np.full(QUANTILE_LEVELS.size, 9.0),
                          np.full(QUANTILE_LEVELS.size, 1.0)])
    pairs = M.build_delta_pairs(rows, variant_q, hybrid_q)
    damp = [p for p in pairs if p["contrast"] == "damping-native"]
    assert len(damp) == 1
    p = damp[0]
    assert p["dy"] == pytest.approx(6.0)
    assert p["variant_dv"] == pytest.approx(5.0)   # 8 - 3
    assert p["hybrid_dv"] == pytest.approx(8.0)    # 9 - 1
    assert p["variant_crps"] == pytest.approx(1.0)  # |5 - 6|
    assert p["hybrid_crps"] == pytest.approx(2.0)   # |8 - 6|


def test_delta_metrics_reports_ci_and_threshold():
    rows = []
    variant_q = []
    hybrid_q = []
    for w in range(6):
        rows.append(OutcomeRow(f"w{w}", 300.0, 120.0, 60.0, "damping", 10.0 + w))
        variant_q.append(np.full(QUANTILE_LEVELS.size, 10.0 + w))
        hybrid_q.append(np.full(QUANTILE_LEVELS.size, 10.0 + w))
        rows.append(OutcomeRow(f"w{w}", 300.0, 120.0, 60.0, "native", 4.0))
        variant_q.append(np.full(QUANTILE_LEVELS.size, 4.0))
        hybrid_q.append(np.full(QUANTILE_LEVELS.size, 8.0))   # +4 bias
    delta = M.delta_metrics(
        M.build_delta_pairs(rows, np.array(variant_q), np.array(hybrid_q)),
        threshold=1.0, n_boot=200, seed=0)
    cell = delta["damping-native"][60.0]
    assert cell["n_pairs"] == 6
    assert cell["variant_crps"] < cell["hybrid_crps"]
    assert cell["improves"] is True
    assert cell["ci_hi"] < 0.0
    assert cell["variant_sign_accuracy"] == pytest.approx(1.0)


def test_level_metrics_fields():
    rows = [OutcomeRow("w0", 300.0, 120.0, 60.0, "native", 5.0),
            OutcomeRow("w1", 300.0, 120.0, 60.0, "native", 7.0)]
    q = np.tile(QUANTILE_LEVELS, (2, 1)) * 4.0 + 3.0
    m = M.level_metrics(rows, q, name="v")
    cell = m[60.0]
    for key in ("crps", "mae", "coverage_10_90", "pit_mean", "mean_delay"):
        assert key in cell
    assert 0.0 <= cell["coverage_10_90"] <= 1.0


def _rollout(size=4):
    pos = np.arange(size * len(HORIZONS_S) * 3, dtype=float).reshape(
        size, len(HORIZONS_S), 3)
    speed = np.ones_like(pos)
    rows = [RolloutRow(f"w{i}", 300.0, "native", true_pos=pos[i],
                       true_speed=speed[i], true_dist=10.0,
                       true_speed_scale=1.0, veh_steps=100.0,
                       collision=float(i % 2))
            for i in range(size)]
    return pos, speed, rows


def test_rollout_metrics_exact_prediction_zero_rmse():
    pos, speed, rows = _rollout()
    batch = RolloutBatch(HORIZONS_S, pos, speed, mask=np.ones_like(pos),
                         collision=np.array([0.0, 1.0, 0.0, 1.0]),
                         ttc_lt3_s=np.zeros(4), hard_brake_s=np.zeros(4),
                         intervention=np.zeros(4))
    m = M.rollout_metrics(batch, rows, name="exact")
    assert m["s_rmse_pct"][10.0]["s_rmse_pct"] == pytest.approx(0.0)
    assert m["s_rmse_pct"][10.0]["v_rmse_pct"] == pytest.approx(0.0)
    assert m["collision_per_1000"] == pytest.approx(1000.0 * 2 / 400.0)


def test_per_1000_helper():
    assert M.per_1000(2, 0.0) != M.per_1000(2, 0.0)  # nan != nan
    assert M.per_1000(2, 1000.0) == pytest.approx(2.0)
