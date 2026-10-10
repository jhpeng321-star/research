"""E4 scorer metric-loop tests on a hand-computable toy (generated only).

Toy: two worlds (``w1`` dy=5, ``w2`` dy=7), taus 60/120, policies
native/damping/pi, constant per-row quantile vectors.  For constant
quantiles ``q = c`` and a constant outcome ``y`` the 19-level CRPS reduces
to ``|y - c|``, so every cell below is exact by hand.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from evalh.metrics import world_bootstrap_ci
from evalh.protocol import N_QUANTILES, OutcomeRow
from evalh.scorer.metrics_loop import score

_SEED = 20261001
_BOOT = 200          # tests use a small draw count; production default 2000


def _const(base: float) -> np.ndarray:
    return np.full(N_QUANTILES, float(base))


def _toy_rows():
    dy = {"w1": 5.0, "w2": 7.0}
    rows = []
    for w in ("w1", "w2"):
        for tau in (60.0, 120.0):
            y_nat, y_dmp = 2.0, 2.0 + dy[w]
            for pol, y in (("native", y_nat), ("damping", y_dmp),
                           ("pi", y_dmp)):
                rows.append(OutcomeRow(world=w, t0_s=0.0, decision_t_s=0.0,
                                       tau_s=tau, policy=pol, y_delay_s=y))
    # one extra (native, damping) pair at tau 45: excluded by the tau filter
    # (45 s is not one of the registered reporting taus)
    rows.append(OutcomeRow(world="w1", t0_s=0.0, decision_t_s=0.0,
                           tau_s=45.0, policy="native", y_delay_s=2.0))
    rows.append(OutcomeRow(world="w1", t0_s=0.0, decision_t_s=0.0,
                           tau_s=45.0, policy="damping", y_delay_s=7.0))
    return rows, dy


def _toy_quantiles(rows, dy):
    q_a, q_v2g = [], []
    for r in rows:
        if r.tau_s == 45.0:                      # excluded rows: any values
            q_a.append(_const(1.0))
            q_v2g.append(_const(1.0))
            continue
        d = dy[r.world]
        off_a = 1.0 if r.world == "w2" else 0.0  # A imperfect only on w2
        off_v = 3.0 if r.tau_s == 60.0 else -10.0
        if r.policy == "native":
            q_a.append(_const(2.0))
            q_v2g.append(_const(2.0))
        else:                                    # damping and pi alike
            q_a.append(_const(2.0 + d + off_a))
            q_v2g.append(_const(2.0 + d + off_v))
    return {"A": np.stack(q_a), "v2g": np.stack(q_v2g)}


def test_toy_cells_exact():
    rows, dy = _toy_rows()
    q = _toy_quantiles(rows, dy)
    out = score(rows, q, n_boot=_BOOT, seed=_SEED)
    counts = out["counts"]
    assert counts["n_rows"] == 14
    assert counts["n_rows_used"] == 12
    assert counts["n_excluded_by_tau"] == 2
    assert counts["n_excluded_by_mask"] == 0
    assert counts["n_excluded_unknown_policy"] == 0
    assert counts["n_pairs"] == 8             # 2 taus x 2 contrasts x 2 worlds
    assert counts["n_rows_by_policy"] == {"native": 4, "damping": 4, "pi": 4}
    assert "wall-native" not in out["cells"]       # present only when present
    assert set(out["cells"]) == {"damping-native", "pi-native"}

    d60 = out["cells"]["damping-native"]["60"]
    assert d60["n_pairs"] == 2 and d60["n_worlds"] == 2
    assert d60["crps_A"] == pytest.approx(0.5)     # (0 + 1) / 2
    assert d60["crps_v2g"] == pytest.approx(3.0)   # |offset 3| both worlds
    assert d60["diff_mean"] == pytest.approx(-2.5)
    assert d60["sign_accuracy_A"] == pytest.approx(1.0)
    assert d60["sign_accuracy_v2g"] == pytest.approx(1.0)
    assert d60["sign_n_used"] == 2
    assert d60["coverage_10_90_A"] == pytest.approx(0.5)
    assert d60["coverage_10_90_v2g"] == pytest.approx(0.0)

    d120 = out["cells"]["damping-native"]["120"]
    assert d120["crps_A"] == pytest.approx(0.5)
    assert d120["crps_v2g"] == pytest.approx(10.0)
    assert d120["diff_mean"] == pytest.approx(-9.5)
    assert d120["sign_accuracy_A"] == pytest.approx(1.0)
    assert d120["sign_accuracy_v2g"] == pytest.approx(0.0)
    assert d120["coverage_10_90_A"] == pytest.approx(0.5)
    assert d120["coverage_10_90_v2g"] == pytest.approx(0.0)

    # pi mirrors damping on this toy
    assert out["cells"]["pi-native"]["60"]["diff_mean"] == \
        pytest.approx(d60["diff_mean"])
    assert out["cells"]["pi-native"]["120"]["sign_accuracy_v2g"] == \
        pytest.approx(0.0)


def test_bootstrap_ci_uses_the_registered_seed():
    from evalh.metrics import crps_of_quantiles
    from evalh.protocol import QUANTILE_LEVELS

    rows, dy = _toy_rows()
    q = _toy_quantiles(rows, dy)
    out = score(rows, q, n_boot=_BOOT, seed=_SEED)
    cell = out["cells"]["damping-native"]["60"]

    def crps_const(base, y):
        return float(crps_of_quantiles(_const(base), float(y),
                                       QUANTILE_LEVELS))

    # per-pair values exactly as the loop computes them (constant quantiles
    # reduce the CRPS to |y - c|): w1 -> 0 - 3, w2 -> 1 - 3
    diff_vals = [crps_const(5.0, 5.0) - crps_const(8.0, 5.0),
                 crps_const(8.0, 7.0) - crps_const(10.0, 7.0)]
    lo, hi, point = world_bootstrap_ci(diff_vals, ["w1", "w2"],
                                       n_boot=_BOOT, alpha=0.10,
                                       seed=_SEED)
    assert cell["diff_ci"] == pytest.approx([float(lo), float(hi)], abs=1e-9)
    assert cell["diff_mean"] == pytest.approx(point, abs=1e-9)
    # per-pair crps_A: 0 and 1 -> exact mean
    crps_vals = [crps_const(5.0, 5.0), crps_const(8.0, 7.0)]
    lo_a, hi_a, point_a = world_bootstrap_ci(crps_vals, ["w1", "w2"],
                                             n_boot=_BOOT, alpha=0.10,
                                             seed=_SEED)
    assert cell["crps_A_ci"] == pytest.approx([float(lo_a), float(hi_a)],
                                              abs=1e-9)
    assert cell["crps_A"] == pytest.approx(point_a, abs=1e-9)


def test_bootstrap_deterministic_under_fixed_seed():
    rows, dy = _toy_rows()
    q = _toy_quantiles(rows, dy)
    one = score(rows, q, n_boot=_BOOT, seed=_SEED)
    two = score(rows, q, n_boot=_BOOT, seed=_SEED)
    assert one == two


def test_masks_exclude_cells_with_counts():
    rows, dy = _toy_rows()
    q = _toy_quantiles(rows, dy)
    masks = [r.world == "w1" for r in rows]
    out = score(rows, q, masks=masks, n_boot=_BOOT, seed=_SEED)
    assert out["counts"]["n_excluded_by_mask"] == 6
    assert out["counts"]["n_rows_used"] == 6
    cell = out["cells"]["damping-native"]["60"]
    assert cell["n_pairs"] == 1 and cell["n_worlds"] == 1
    assert cell["crps_A"] == pytest.approx(0.0)
    assert cell["crps_v2g"] == pytest.approx(3.0)
    assert cell["diff_mean"] == pytest.approx(-3.0)
    assert cell["coverage_10_90_A"] == pytest.approx(1.0)
    # identical cells to scoring the kept rows on their own
    kept = [i for i, r in enumerate(rows) if masks[i]]
    rows1 = [rows[i] for i in kept]
    q1 = {"A": q["A"][kept], "v2g": q["v2g"][kept]}
    subset = score(rows1, q1, n_boot=_BOOT, seed=_SEED)
    assert subset["cells"] == out["cells"]
    assert subset["counts"]["n_pairs"] == out["counts"]["n_pairs"]


def test_unknown_policy_excluded_and_counted():
    rows, dy = _toy_rows()
    q = _toy_quantiles(rows, dy)
    rows = list(rows)
    rows.append(OutcomeRow(world="w1", t0_s=0.0, decision_t_s=0.0,
                           tau_s=60.0, policy="mystery_arm",
                           y_delay_s=4.0))
    q["A"] = np.concatenate([q["A"], _const(1.0)[None]])
    q["v2g"] = np.concatenate([q["v2g"], _const(1.0)[None]])
    out = score(rows, q, n_boot=_BOOT, seed=_SEED)
    assert out["counts"]["n_excluded_unknown_policy"] == 1
    assert out["counts"]["n_rows_used"] == 12
    assert out["counts"]["n_pairs"] == 8


def test_zero_dy_sign_pairs_report_nan_not_zero():
    rows = [OutcomeRow(world="w9", t0_s=0.0, decision_t_s=0.0, tau_s=60.0,
                       policy="native", y_delay_s=2.0),
            OutcomeRow(world="w9", t0_s=0.0, decision_t_s=0.0, tau_s=60.0,
                       policy="damping", y_delay_s=2.0)]
    q = {"A": np.stack([_const(2.0), _const(7.0)]),
         "v2g": np.stack([_const(2.0), _const(9.0)])}
    out = score(rows, q, n_boot=_BOOT, seed=_SEED)
    cell = out["cells"]["damping-native"]["60"]
    assert cell["sign_n_used"] == 0
    assert math.isnan(cell["sign_accuracy_A"])
    assert math.isnan(cell["sign_accuracy_v2g"])


def test_unequal_widths_never_stacked():
    rows, dy = _toy_rows()
    q = _toy_quantiles(rows, dy)
    bad = {"A": q["A"][:, :-1], "v2g": q["v2g"]}
    with pytest.raises(ValueError, match="unequal widths"):
        score(rows, bad)
    short = {"A": q["A"][:-1], "v2g": q["v2g"][:-1]}
    with pytest.raises(ValueError, match="unequal widths"):
        score(rows, short)
    with pytest.raises(ValueError, match="masks"):
        score(rows, q, masks=[True] * (len(rows) + 1))
    with pytest.raises(ValueError, match="exactly"):
        score(rows, {"A": q["A"], "x": q["v2g"]})


def test_no_pairs_refused():
    rows = [OutcomeRow(world="w1", t0_s=0.0, decision_t_s=0.0, tau_s=60.0,
                       policy="wall", y_delay_s=3.0),
            OutcomeRow(world="w1", t0_s=0.0, decision_t_s=0.0, tau_s=60.0,
                       policy="damping", y_delay_s=5.0)]
    q = {"A": np.stack([_const(1.0), _const(2.0)]),
         "v2g": np.stack([_const(1.0), _const(2.0)])}
    with pytest.raises(ValueError, match="no native-referenced"):
        score(rows, q)
