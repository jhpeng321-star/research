"""Part 3l aggregator tests: pure-function checks on synthetic ladder
rows (no data, no rollouts, no weights).  Covers the gate v2 (a)-(d)
logic INCLUDING the binding per-horizon (d) form, the L5c widening
derivation (mean exactly invariant), coverage/CRPS behaviour, and the
old gate."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "part3k"))

import aggregate_p3l as AG  # noqa: E402
import p3l_lib as L  # noqa: E402


def mkrow(variant, member, root, t0, H, speed=4.0, coll=0.0, lc_p=10,
          lc_t=10, dp=100.0, dpt=100.0, t3=5.0, t3t=5.0, dexp=100.0,
          dexp_t=100.0):
    return dict(variant=variant, member=member, root=root, t0=t0, H=H,
                speed_rmse=speed, pos_rmse=1.0,
                collisions_per_1000=coll,
                pred_lc=lc_p, true_lc=lc_t,
                delay_pred=dexp, delay_true=dexp_t,
                pred_vehicle_steps=1000, matched_vehicle_steps=1000,
                delay_phys_pred=dp, delay_phys_true=dpt,
                phys_steps_pred=60, phys_steps_true=60,
                pred_ttc3_exposure=t3, true_ttc3_exposure=t3t,
                pred_hard_brake_s=0.0, true_hard_brake_s=0.0,
                any_ttc3_pred=t3 > 0, any_ttc3_true=t3t > 0,
                any_hb_pred=False, any_hb_true=False, lane_frac=1.0,
                dawdle_scale=1.0, engine="synthetic")


def mk_ensemble(variant, spread, truth, H=30, t0=120, root="r1",
                speed=4.0, centre=None):
    """5 synthetic members whose delay/ttc3 values sit around `centre`
    (default: the truth itself) while the truth is `truth`."""
    c = truth if centre is None else centre
    rows = []
    for mi, off in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0)):
        rows.append(mkrow(variant, mi, root, t0, H,
                          speed=speed, dp=c + off * spread,
                          dpt=truth, t3=max(0.0, 5.0 + off * spread),
                          t3t=5.0, dexp=c + off * spread,
                          dexp_t=truth))
    return rows


# ------------------------------------------------------------ helpers --

def test_load_rows_dedup_keeps_last():
    p = Path(AG.__file__).parent / "_tmp_rows.jsonl"
    try:
        a = mkrow("L5", 0, "r1", 120, 30, dp=1.0)
        b = mkrow("L5", 0, "r1", 120, 30, dp=2.0)
        p.write_text("\n".join(json.dumps(r) for r in (a, b)),
                     encoding="utf-8")
        rows = AG.load_rows(p)
        assert len(rows) == 1 and rows[0]["delay_phys_pred"] == 2.0
    finally:
        p.unlink(missing_ok=True)


def test_summarise_pools_rmse_and_means():
    rows = (mk_ensemble("L5", 1.0, 100.0, root="r1", speed=3.0)
            + mk_ensemble("L5", 1.0, 100.0, root="r2", t0=180,
                          speed=5.0))
    s = AG.summarise(rows)[("L5", 30)]
    assert s["n_members"] == 5 and s["n_windows"] == 2
    assert s["speed_rmse"] == pytest.approx(
        np.sqrt((9.0 + 25.0) / 2), abs=1e-3)      # pooled RMSE
    assert s["delay_phys_pred"] == pytest.approx(100.0, abs=1e-3)
    assert s["delay_phys_bias"] == pytest.approx(0.0, abs=1e-3)


# ----------------------------------------------------------- coverage --

def test_coverage_and_crps_and_widening():
    # window r1: ensemble centred on its truth 100 -> covers;
    # window r2: ensemble centred on 100 but truth 103.5 -> the raw
    # 10-90 band [99, 101] misses it; widening by s=4 -> band [96, 104]
    rows = (mk_ensemble("L5", 1.0, 100.0, root="r1")
            + mk_ensemble("L5", 1.0, 103.5, root="r2", t0=180,
                          centre=100.0))
    cov = AG.coverage_and_crps(rows, horizons=(30,))
    c = cov[("L5", 30)]
    assert c["n_windows"] == 2
    assert c["delay_phys_cov_frac"] == pytest.approx(0.5)
    assert c["ttc3_cov_frac"] == pytest.approx(1.0)
    assert c["crps_phys_mean"] >= 0.0
    # widening by s=4 around the MEAN must leave each window mean
    # unchanged and cover the far window with the widened band
    wmap = dict(delay_phys={30: 4.0}, ttc3={30: 4.0})
    cov_w = AG.coverage_and_crps(rows, horizons=(30,), widen=wmap)
    cw = cov_w[("L5", 30)]
    assert cw["delay_phys_cov_frac"] == pytest.approx(1.0)
    assert cw["s_out_delay_phys"] == 4.0
    # band grew around an unchanged mean: min-max now spans the truth
    assert cw["delay_phys_minmax_frac"] > c["delay_phys_minmax_frac"]


def test_widen_mean_centre_invariance_vs_median():
    x = np.array([10.0, 11.0, 12.0, 13.0, 30.0])   # mean 15.2, median 12
    w_mean = L.widen_ensemble(x, 4.0, "mean")
    assert w_mean.mean() == pytest.approx(x.mean(), abs=1e-12)
    w_med = L.widen_ensemble(x, 4.0, "median")
    assert np.median(w_med) == pytest.approx(np.median(x), abs=1e-12)
    assert abs(w_med.mean() - x.mean()) > 1.0      # mean DOES shift


def test_s_out_search_reaches_target():
    rng = np.random.default_rng(0)
    samples, truths = [], []
    for _ in range(20):                    # truth beyond the raw band
        base = float(rng.uniform(50, 150))
        samples.append(base + np.array([-2, -1, 0, 1, 2]) * 1.0)
        truths.append(base + float(rng.choice([-6.0, 6.0])))
    s = L.s_out_for(samples, truths, target=0.80)
    assert s is not None
    cov = L.coverage_frac([L.widen_ensemble(x, s) for x in samples],
                          truths)
    assert cov >= 0.80


# ----------------------------------------------------------- gate v2 --

def gate_inputs(speed, coll=0.0, lc_dev=0.0, dp_cov=0.9, t3_cov=0.9,
                crps=10.0):
    summary = dict(collisions_per_1000=coll, pred_lc=10, true_lc=10,
                   speed_rmse=speed, delay_phys_bias=0.0,
                   delay_exposure_bias=0.0)
    cov = dict(ttc3_cov_frac=t3_cov, delay_phys_cov_frac=dp_cov,
               crps_phys_mean=crps, delay_phys_minmax_frac=0.5)
    l0_cov = dict(delay_phys_mae=50.0)
    return summary, cov, l0_cov


def test_gate_v2_d_per_horizon_is_binding():
    # L0@H30 speed 4.0 -> binding threshold 4.2; fixed is 5.2185
    s, c, l0 = gate_inputs(speed=4.1)
    g = AG.gate_v2_verdict(s, c, l0, l0_speed_same_H=4.0)
    assert g["d_fixed_pass"] is True
    assert g["d_per_horizon_pass"] is True
    assert g["v2_PASS"] is True
    s, c, l0 = gate_inputs(speed=4.3)
    g = AG.gate_v2_verdict(s, c, l0, l0_speed_same_H=4.0)
    assert g["d_fixed_pass"] is True          # fixed 5.2185 would pass
    assert g["d_per_horizon_pass"] is False   # binding form fails
    assert g["v2_PASS"] is False
    assert "d_speed_per_horizon" in g["v2_failing"]


def test_gate_v2_a_b_c_failures():
    s, c, l0 = gate_inputs(speed=4.0, coll=2.0)
    g = AG.gate_v2_verdict(s, c, l0, l0_speed_same_H=4.0)
    assert g["a_collisions_pass"] is False and g["v2_PASS"] is False
    s, c, l0 = gate_inputs(speed=4.0, lc_dev=0.5)
    s["pred_lc"], s["true_lc"] = 15, 10
    g = AG.gate_v2_verdict(s, c, l0, l0_speed_same_H=4.0)
    assert g["b_lc_pass"] is False and "b_lc" in g["v2_failing"]
    s, c, l0 = gate_inputs(speed=4.0, dp_cov=0.5)
    g = AG.gate_v2_verdict(s, c, l0, l0_speed_same_H=4.0)
    assert g["c_coverage_crps_pass"] is False
    s, c, l0 = gate_inputs(speed=4.0, crps=60.0)   # crps >= mae 50
    g = AG.gate_v2_verdict(s, c, l0, l0_speed_same_H=4.0)
    assert g["c_crps_pass"] is False


def test_gate_v2_c_not_applicable_for_deterministic():
    s, c, l0 = gate_inputs(speed=4.0)
    g = AG.gate_v2_verdict(s, dict(ttc3_cov_frac=None,
                                   delay_phys_cov_frac=None,
                                   crps_phys_mean=None,
                                   delay_phys_minmax_frac=None),
                           l0, l0_speed_same_H=4.0, deterministic=True)
    assert g["c_not_applicable"] is True and g["v2_PASS"] is True


def test_old_gate_fixed_thresholds():
    s, c, _ = gate_inputs(speed=4.9)
    g = AG.old_gate_verdict(s, c)
    assert g["old_pass"] is True
    s, c, _ = gate_inputs(speed=5.0)
    assert AG.old_gate_verdict(s, c)["old_pass"] is False
