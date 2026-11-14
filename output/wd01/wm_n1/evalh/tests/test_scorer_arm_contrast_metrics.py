"""E2 metric-loop tests: arm contrasts are scored, not skipped (synthetic)."""
from __future__ import annotations

import numpy as np

from evalh.protocol import N_QUANTILES, OutcomeRow
from evalh.scorer.metrics_loop import score


def _const(base: float) -> np.ndarray:
    return np.full(N_QUANTILES, float(base))


def _rows():
    base = {"native": 100.0, "fixed1": 130.0, "fixed2": 150.0,
            "switching1": 170.0, "w-lo-early": 190.0, "arm0": 210.0}
    rows = []
    for w in ("w1", "w2"):
        for tau in (60.0, 120.0):
            for pol, y in base.items():
                rows.append(OutcomeRow(world=w, t0_s=0.0, decision_t_s=0.0,
                                       tau_s=tau, policy=pol, y_delay_s=y))
    return rows, base


def _quantiles(rows, base):
    q_a, q_v = [], []
    for r in rows:
        q_a.append(_const(base[r.policy]))
        q_v.append(_const(base[r.policy] + 5.0))
    return {"A": np.stack(q_a), "v2g": np.stack(q_v)}


def test_e2_arm_contrasts_scored_and_pooled():
    rows, base = _rows()
    out = score(rows, _quantiles(rows, base), taus=(60, 120), n_boot=50,
                seed=7)
    for name in ("fixed1-native", "fixed2-native", "fixed_mix-native",
                 "switching1-native", "switching_mix-native",
                 "w-lo-early-native", "wall-native", "arm0-native",
                 "trajectory-native"):
        assert name in out["cells"], name
        assert out["cells"][name]["60"]["n_pairs"] >= 2
    # fixed_mix pools fixed1+fixed2 across two worlds at each tau
    assert out["cells"]["fixed_mix-native"]["60"]["n_pairs"] == 4
    assert out["cells"]["fixed_mix-native"]["60"]["n_worlds"] == 2
    # no E2 policy is dropped as unknown
    assert out["counts"]["n_excluded_unknown_policy"] == 0
    assert out["counts"]["n_rows_by_policy"]["fixed1"] == 4
    assert "fixed_mix-native" in out["contrasts_present"]


def test_pooled_dv_is_union_of_members():
    rows, base = _rows()
    out = score(rows, _quantiles(rows, base), taus=(60, 120), n_boot=50,
                seed=7)
    member = out["cells"]["fixed1-native"]["60"]
    pooled = out["cells"]["fixed_mix-native"]["60"]
    assert pooled["n_pairs"] == member["n_pairs"] + out["cells"][
        "fixed2-native"]["60"]["n_pairs"]
    # member dV is exact (30 / 50) so each member cell is prefect; the pooled
    # cell is the union and still covers all of its observations
    assert member["crps_A"] == 0.0 and pooled["crps_A"] == 0.0
    assert pooled["coverage_10_90_A"] == 1.0


def test_no_e2_rows_leaves_legacy_cells_unchanged():
    rows = [OutcomeRow(world="w1", t0_s=0.0, decision_t_s=0.0, tau_s=60.0,
                       policy="native", y_delay_s=2.0),
            OutcomeRow(world="w1", t0_s=0.0, decision_t_s=0.0, tau_s=60.0,
                       policy="pi", y_delay_s=5.0)]
    q = {"A": np.stack([_const(2.0), _const(5.0)]),
         "v2g": np.stack([_const(2.0), _const(9.0)])}
    out = score(rows, q, taus=(60,), n_boot=50, seed=7)
    assert "pi-native" in out["cells"]
    assert out["contrasts_present"] == ["pi-native"]
