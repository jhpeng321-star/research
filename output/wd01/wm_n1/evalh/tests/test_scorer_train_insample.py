"""Unit tests for the in-sample TRAIN scoring helpers (no data opened).

Only pure helpers are exercised with generated dicts/rows; the pack and the
checkpoint are never read.
"""
from __future__ import annotations

from evalh.protocol import OutcomeRow
from evalh.scorer.train_insample import (_canon, absolute_bias,
                                         in_sample_climatology, row_to_item)


def test_canon_from_arm_kind():
    assert _canon("damping_fleet") == "damping"
    assert _canon("pi_fleet") == "pi"
    assert _canon("native_fleet") == "native"
    assert _canon("switching") == "switching"
    assert _canon("fixed") == "fixed"


def test_row_to_item_disambiguates_same_world_arm_t0():
    items = [
        {"world": "w1", "arm": "pi", "t0_s": 30.0, "split": "TRAIN",
         "outcome_rows": [0, 1]},
        {"world": "w1", "arm": "pi", "t0_s": 30.0, "split": "TRAIN",
         "outcome_rows": [2, 3]},
        {"world": "w2", "arm": "pi", "t0_s": 30.0, "split": "CALIBRATION",
         "outcome_rows": [4]},
    ]
    mapping = row_to_item(items)
    assert mapping[0][0] == 0 and mapping[3][0] == 1
    assert 4 not in mapping                     # CAL is never mapped
    assert set(mapping) == {0, 1, 2, 3}


def _row(world, tau, policy, y):
    return OutcomeRow(world=world, t0_s=0.0, decision_t_s=0.0, tau_s=tau,
                      policy=policy, y_delay_s=y)


def test_in_sample_climatology_pairs_and_quantiles():
    rows = []
    for i, dy in enumerate((1.0, 3.0, 5.0)):
        rows.append(_row(f"w{i}", 60.0, "native", 10.0))
        rows.append(_row(f"w{i}", 60.0, "pi", 10.0 + dy))
    doc = in_sample_climatology(rows)
    cell = doc["cells"]["pi-native"]["60"]
    assert cell["n_pairs"] == 3 and cell["n_worlds"] == 3
    assert cell["quantiles"][9] == 3.0          # median dV


def test_absolute_bias_mean_and_median():
    abs_rows = [dict(world="w", arm="pi", tau=30.0, q50=101.0, y=100.0),
                dict(world="w", arm="pi", tau=30.0, q50=10.0, y=0.0)]
    out = absolute_bias(abs_rows)
    cell = out["per_arm_tau"]["pi@30"]
    assert cell["n"] == 2
    assert cell["mean_bias"] == 5.5
    assert cell["median_bias"] == 5.5
    assert out["per_arm"]["pi"]["n"] == 2
