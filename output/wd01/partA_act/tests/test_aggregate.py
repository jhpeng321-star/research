"""Aggregation-math tests for aggregate_act.py (synthetic rows, no
data, no rollouts)."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import aggregate_act as A  # noqa: E402


def _row(world, arm, variant, dp, dt_, pp, pt, prefix=False, coll=0.0):
    return dict(world=world, arm=arm, variant=variant,
                delay_phys_pred=dp, delay_phys_true=dt_,
                pred_ttc3_exposure=pp, true_ttc3_exposure=pt,
                speed_rmse_cav=1.0, speed_rmse_human=1.0,
                collisions_per_1000=coll, pred_collisions=1 if coll else 0,
                truth_collisions_in_window=0, pred_vehicle_steps=1000,
                n_commands=5, penetration=0.02, prefix=prefix)


def test_paired_delta_matches_by_hand():
    rows = [
        # world w1: arm +20 pred, +10 true; native roll 100/110
        _row("w1", "arm", "act", 120.0, 120.0, 4.0, 2.0),
        _row("w1", "native", "act", 100.0, 110.0, 2.0, 2.0),
        # world w2: arm -30 pred, -20 true; native 200/200
        _row("w2", "arm", "act", 170.0, 180.0, 1.0, 3.0),
        _row("w2", "native", "act", 200.0, 200.0, 2.0, 2.0),
    ]
    d = A.paired_delta(rows, "arm", "delay_phys_pred", "delay_phys_true")
    assert d["n_pairs"] == 2
    assert d["d_pred_mean"] == pytest.approx((20.0 + -30.0) / 2)
    assert d["d_true_mean"] == pytest.approx((10.0 + -20.0) / 2)
    assert d["delta_bias"] == pytest.approx(0.0)
    assert d["delta_mae"] == pytest.approx(10.0)
    assert d["sign_agreement"] == pytest.approx(1.0)
    # correlations need >= 3 pairs (act_lib.corr guard)
    assert d["pearson"] is None and d["spearman"] is None


def test_paired_delta_skips_incomplete_worlds():
    rows = [_row("w1", "arm", "act", 120.0, 120.0, 1.0, 1.0)]
    assert A.paired_delta(rows, "arm", "delay_phys_pred",
                          "delay_phys_true") is None


def test_prefix_tagging_rule():
    rows = [_row("w1", "native", "act", 1, 1, 0, 0, prefix=False),
            _row("w2", "native", "act", 1, 1, 0, 0, prefix=True)]
    pre = [r for r in rows if r["prefix"]]
    post = [r for r in rows if not r["prefix"]]
    assert len(pre) == 1 and len(post) == 1


def test_agg_group_counters():
    rows = [_row("w1", "native", "act", 100.0, 90.0, 3.0, 2.0,
                 coll=2.0),
            _row("w2", "native", "act", 60.0, 60.0, 1.0, 1.0)]
    a = A.agg_group(rows)
    assert a["n_windows"] == 2
    assert a["delay_bias"] == pytest.approx(5.0)
    assert a["ttc3_bias"] == pytest.approx(0.5)
    assert a["collisions_per_1000_mean"] == pytest.approx(1.0)
    assert a["windows_with_pred_collision"] == 1
    assert a["truth_collisions_total"] == 0
