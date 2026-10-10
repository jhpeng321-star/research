"""Part 3k synthetic tests for aggregate_p3k.py: pure functions on
hand-built raw rows -- no rollouts, no data, no SUMO."""
import numpy as np
import pytest

import aggregate_p3k as A


def make_rows(n_windows=4, spread=5.0, truth_off=0.0):
    """Synthetic raw rows for one root, both horizons, all rungs.

    L3/L4 members are centered on `centre` with half-widths
    sp in {-1, -0.5, 0, +0.5, +1} * spread (delay_phys; half that for
    exposure/ttc3/lc).  Truth sits truth_off ABOVE the member centre, so
    the 10-90 % band covers iff |truth_off| <= ~0.75 * spread and the
    min-max band iff |truth_off| <= spread.  L0 is deterministic with a
    -30 s (exposure) / -20 s (physics) delay bias."""
    rows = []
    for H in (30, 60):
        for w in range(n_windows):
            base = 100.0 + w
            for variant in ("L3", "L4"):
                for m in range(5):
                    sp = spread * (m - 2) / 2.0
                    rows.append(dict(
                        variant=variant, member=m, root="r1",
                        t0=120 + 30 * w, H=H, speed_rmse=4.5,
                        pos_rmse=60.0, collisions_per_1000=0.0,
                        pred_lc=24.0 + sp / 8.0, true_lc=24.0,
                        pred_vehicle_steps=5000,
                        matched_vehicle_steps=4900,
                        pred_ttc3_exposure=4.0 + sp / 50.0,
                        true_ttc3_exposure=4.0, pred_hard_brake_s=10.0,
                        true_hard_brake_s=11.0,
                        delay_pred=base + sp / 2.0,
                        delay_true=base + truth_off / 2.0,
                        delay_phys_pred=base * 0.6 + sp,
                        delay_phys_true=base * 0.6 + truth_off,
                        lane_frac=0.95))
            rows.append(dict(
                variant="L0", member=0, root="r1", t0=120 + 30 * w, H=H,
                speed_rmse=4.9, pos_rmse=70.0, collisions_per_1000=0.0,
                pred_lc=0, true_lc=24.0, pred_vehicle_steps=4800,
                matched_vehicle_steps=4700, pred_ttc3_exposure=5.0,
                true_ttc3_exposure=4.0, pred_hard_brake_s=9.0,
                true_hard_brake_s=11.0, delay_pred=base - 30.0,
                delay_true=base + truth_off / 2.0,
                delay_phys_pred=base * 0.6 - 20.0,
                delay_phys_true=base * 0.6 + truth_off, lane_frac=0.93))
    return rows


# ------------------------------------------------------------- pieces ----

def test_coverage_10_90_and_minmax():
    assert A.coverage_10_90(list(range(10)), 4.5)
    assert not A.coverage_10_90(list(range(10)), 9.5)   # above p90
    assert not A.coverage_10_90(list(range(10)), -0.5)  # below p10
    assert A.coverage_minmax([1.0, 5.0], 3.0)
    assert not A.coverage_minmax([1.0, 5.0], 7.0)


def test_crps_ensemble_known_values():
    assert A.crps_ensemble([3.0], 1.0) == pytest.approx(2.0)  # n < 2
    assert A.crps_ensemble([1.0, 1.0], 2.0) == pytest.approx(1.0)
    # 2-point ensemble symmetric around truth: E|X-y| = 1, E|X-X'| = 1
    # (diagonal zeros included) -> CRPS = 1 - 0.5
    assert A.crps_ensemble([0.0, 2.0], 1.0) == pytest.approx(0.5)
    # a tighter ensemble scores better on the same truth
    assert A.crps_ensemble([0.9, 1.1], 1.0) < \
        A.crps_ensemble([0.0, 2.0], 1.0)


def test_load_rows_dedups_keeping_last(tmp_path):
    import json
    row = dict(variant="L4", member=0, root="r1", t0=120, H=30,
               speed_rmse=1.0)
    row2 = dict(row, speed_rmse=2.0)
    p = tmp_path / "raw.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n\n")
        fh.write(json.dumps(row2) + "\n")
    out = A.load_rows(p)
    assert len(out) == 1 and out[0]["speed_rmse"] == 2.0


# -------------------------------------------------- summaries + gates ----

def test_summarise_means_and_biases():
    rows = make_rows()
    per = A.summarise(rows)
    s = per[("L4", 30)]
    assert s["n_members"] == 5 and s["n_windows"] == 4
    assert s["pred_lc"] == pytest.approx(24.0)
    assert s["delay_phys_pred"] == pytest.approx(
        np.mean([0.6 * (100.0 + w) for w in range(4)]))
    assert s["delay_phys_bias"] == pytest.approx(0.0)
    assert s["delay_exposure_bias"] == pytest.approx(0.0)
    s0 = per[("L0", 30)]
    assert s0["delay_phys_bias"] == pytest.approx(-20.0)
    assert s0["delay_exposure_bias"] == pytest.approx(-30.0)


def test_coverage_and_crps_centre_covers_bias_does_not():
    # truth on the member centre: everything covered
    cov = A.coverage_and_crps(make_rows(truth_off=0.0), horizons=(30,))
    c = cov[("L4", 30)]
    assert c["n_windows"] == 4
    assert c["delay_phys_cov_frac"] == 1.0
    assert c["delay_exposure_cov_frac"] == 1.0
    assert c["ttc3_cov_frac"] == 1.0
    assert c["delay_phys_minmax_frac"] == 1.0
    assert c["crps_phys_mean"] < cov[("L0", 30)]["delay_phys_mae"]
    # biased truth (off = 5 > 0.75 * 6 but <= 6): 10-90 % fails while
    # min-max still covers
    cov_b = A.coverage_and_crps(make_rows(spread=6.0, truth_off=5.0),
                                horizons=(30,))
    cb = cov_b[("L4", 30)]
    assert cb["delay_phys_cov_frac"] == 0.0
    assert cb["delay_exposure_cov_frac"] == 0.0
    assert cb["delay_phys_minmax_frac"] == 1.0


def test_gate_v2_verdict_pass_and_fail_modes():
    rows = make_rows(truth_off=3.0, spread=5.0)   # |3| <= 3.75: covered
    per = A.summarise(rows)
    cov = A.coverage_and_crps(rows, horizons=(30,))
    g = A.gate_v2_verdict(per[("L4", 30)], cov[("L4", 30)],
                          cov[("L0", 30)])
    assert g["v2_PASS"] and g["v2_failing"] == []
    assert g["c_coverage_crps_pass"]
    # L0: deterministic -> (c) NA; (b) fails on 0 vs 24 LC
    c0 = dict(ttc3_cov_frac=None, delay_phys_cov_frac=None,
              crps_phys_mean=None, delay_phys_minmax_frac=None)
    g0 = A.gate_v2_verdict(per[("L0", 30)], c0, cov[("L0", 30)],
                           deterministic=True)
    assert g0["c_not_applicable"]
    assert g0["c_coverage_crps_pass"] is None
    assert not g0["v2_PASS"] and g0["v2_failing"] == ["b_lc"]
    # too many collisions fails only (a)
    s_bad = dict(per[("L4", 30)])
    s_bad["collisions_per_1000"] = 2.0
    g_bad = A.gate_v2_verdict(s_bad, cov[("L4", 30)], cov[("L0", 30)])
    assert g_bad["v2_failing"] == ["a_collisions"]
    assert not g_bad["v2_PASS"]
    # speed above 1.05 x B2 fails (d)
    s_spd = dict(per[("L4", 30)])
    s_spd["speed_rmse"] = 5.3
    g_spd = A.gate_v2_verdict(s_spd, cov[("L4", 30)], cov[("L0", 30)])
    assert g_spd["v2_failing"] == ["d_speed"]
    # LC 25 % off fails (b)
    s_lc = dict(per[("L4", 30)])
    s_lc["pred_lc"] = 30.0
    g_lc = A.gate_v2_verdict(s_lc, cov[("L4", 30)], cov[("L0", 30)])
    assert g_lc["v2_failing"] == ["b_lc"]
    # biased members (0 % band coverage) fail (c) even with CRPS < MAE
    rows_b = make_rows(spread=6.0, truth_off=5.0)
    per_b = A.summarise(rows_b)
    cov_b = A.coverage_and_crps(rows_b, horizons=(30,))
    g_b = A.gate_v2_verdict(per_b[("L4", 30)], cov_b[("L4", 30)],
                            cov_b[("L0", 30)])
    assert g_b["v2_failing"] == ["c_coverage_crps"]


def test_crps_beats_mae_required_for_c():
    rows = make_rows(truth_off=3.0)
    per = A.summarise(rows)
    cov = A.coverage_and_crps(rows, horizons=(30,))
    assert cov[("L4", 30)]["crps_phys_mean"] < \
        cov[("L0", 30)]["delay_phys_mae"]
    # if CRPS were worse than the L0 MAE, (c) fails even with full bands
    cov_bad = dict(cov[("L4", 30)],
                   crps_phys_mean=cov[("L0", 30)]["delay_phys_mae"] + 1.0)
    g = A.gate_v2_verdict(per[("L4", 30)], cov_bad, cov[("L0", 30)])
    assert g["v2_failing"] == ["c_coverage_crps"]


def test_old_gate_verdict():
    rows = make_rows(truth_off=3.0)
    per = A.summarise(rows)
    cov = A.coverage_and_crps(rows, horizons=(30,))
    g = A.old_gate_verdict(per[("L4", 30)], cov[("L4", 30)])
    assert g["old_pass"]
    # L0 has no member band -> minmax None -> fails (and LC 0 vs 24)
    c0 = dict(ttc3_cov_frac=None, delay_phys_cov_frac=None,
              crps_phys_mean=None, delay_phys_minmax_frac=None)
    g0 = A.old_gate_verdict(per[("L0", 30)], c0)
    assert not g0["old_pass"]
