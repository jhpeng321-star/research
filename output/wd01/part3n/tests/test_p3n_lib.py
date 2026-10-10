"""Part 3n library tests (pure, no data, no rollouts, no weights):
LC-by-type counting on both sides, the by-type loss, the 2-fold gate
machinery (no fold scores itself), climatology/sharpness math, widened
row derivation, and the freeze gate."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import p3n_lib as N


# ------------------------------------------------------- LC by type --

def _snaps_lane_seq(seq):
    """Fake snapshots: ONE vehicle, lane per step 0..K."""
    return [dict(vid=np.array([7]), lane=np.array([l]),
                 s=np.array([100.0]), v=np.array([20.0]),
                 lc=0)
            for l in seq]


def test_pred_lc_by_type_merge_vs_disc():
    # steps (0.5 s each): 2->1 at step 3 (merge), 1->0 at step 70 and
    # 0->1 at step 90 (discretionary); K = 100.  H=30 -> prefix
    # 1..60 steps, H=60 -> 1..120 steps.
    seq = [2] * 4 + [1] * 67 + [0] * 20 + [1] * 11   # idx 0..101
    assert len(seq) == 102
    snaps = _snaps_lane_seq(seq)
    out = N.pred_lc_by_type(snaps, 100)
    # H=30 prefix covers steps 1..60 -> only the merge
    assert out[30] == dict(pred_merge=1, pred_disc=0)
    # H=60 prefix covers steps 1..120 -> all three changes
    assert out[60] == dict(pred_merge=1, pred_disc=2)


def test_truth_lc_by_type_windows_and_types():
    ra = SimpleNamespace(
        lc_vid=np.array([1, 2, 3]),
        lc_tick=np.array([5, 70, 30]),
        lc_from=np.array([2, 0, 1]),
        lc_to=np.array([1, 1, 0]))
    out = N.truth_lc_by_type(ra, k0=0, K=120)
    # H=30 prefix is steps 1..60: merge@5 yes, disc 1->0@30 yes,
    # 0->1@70 no
    assert out[30] == dict(true_merge=1, true_disc=1)
    assert out[60] == dict(true_merge=1, true_disc=2)
    # shifted window k0=40: only tick 70 inside
    out2 = N.truth_lc_by_type(ra, k0=40, K=120)
    assert out2[60] == dict(true_merge=0, true_disc=1)


# -------------------------------------------------------------- loss --

def _win_stats(occ, lcM30, lcD30, lcM60, lcD60, ttc3, delay):
    return dict(occ=occ, prof={}, merges=[1.0], mr=[20.0, 10],
                delay=delay, lc30=lcM30 + lcD30, lc60=lcM60 + lcD60,
                ttc3=ttc3, lcM30=lcM30, lcD30=lcD30, lcM60=lcM60,
                lcD60=lcD60)


def test_loss_p3n_zero_at_perfect_and_grows_off():
    t = {("r", 120): _win_stats(10, 3, 1, 5, 2, 8.0, 500.0)}
    p = {("r", 120): _win_stats(10, 3, 1, 5, 2, 8.0, 500.0)}
    l2, det = N.loss_p3n(p, t)
    assert l2 == pytest.approx(0.0)
    assert det["lcM30"] == 0.0 and det["ttc3"] == 0.0
    # an overshoot in the MERGE type only moves the merge terms
    p2 = {("r", 120): _win_stats(10, 6, 1, 5, 2, 8.0, 500.0)}
    l2m, detm = N.loss_p3n(p2, t)
    assert detm["lcM30"] > 0 and detm["lcD30"] == 0.0
    assert l2m > l2


def test_softmax_draw_extremes():
    rng = np.random.default_rng(0)
    assert N.softmax_draw([5.0, 1.0, 3.0], rng) == 1
    ties = N.softmax_draw([1.0, 1.0, 1.0], np.random.default_rng(1))
    assert ties in (0, 1, 2)


# -------------------------------------------------- 2-fold gate math --

def _mk_rows(n_win_per_root=3, bias_merge=0.0, spread=1.0,
             l6_speed=3.0):
    """Synthetic rows over both folds: TWO L6 members at truth +/- spread
    (so the raw band covers) + an L0 row (perfect delay, slow speed).
    H=30 truths: delay 10 / ttc3 5; H=60: 20 / 8.  pred_lc 10+merge
    bias for both L6 members."""
    rows = []
    truths = {30: dict(dp=10.0, t3=5.0), 60: dict(dp=20.0, t3=8.0)}

    def add(variant, mi, root, t0, H, dp_pred, t3_pred):
        t = truths[H]
        rows.append(dict(
            variant=variant, member=mi, root=root, t0=t0, H=H,
            speed_rmse=2.0 if variant == "L0" else l6_speed,
            pos_rmse=30.0, collisions_per_1000=0.0,
            pred_lc=(10.0 + bias_merge) if variant == "L6" else 0.0,
            true_lc=10.0,
            delay_pred=100.0, delay_true=100.0,
            pred_vehicle_steps=1000, matched_vehicle_steps=1000,
            delay_phys_pred=dp_pred, delay_phys_true=t["dp"],
            phys_steps_pred=100, phys_steps_true=100,
            pred_ttc3_exposure=t3_pred, true_ttc3_exposure=t["t3"],
            pred_hard_brake_s=0.0, true_hard_brake_s=0.0,
            any_ttc3_pred=0, any_ttc3_true=0,
            any_hb_pred=0, any_hb_true=0,
            lane_frac=1.0, dawdle_scale=1.0, engine="synthetic"))

    for roots in (N.FOLD_A, N.FOLD_B):
        for root in roots:
            for w in range(n_win_per_root):
                t0 = 120 + 60 * w
                for H in (30, 60):
                    t = truths[H]
                    add("L0", 0, root, t0, H, t["dp"], t["t3"])
                    add("L6", 0, root, t0, H,
                        t["dp"] - 2.0 * spread,
                        t["t3"] - 0.5 * spread)
                    add("L6", 1, root, t0, H,
                        t["dp"] + 2.0 * spread,
                        t["t3"] + 0.5 * spread)
    return rows


def test_fit_s_out_uses_only_the_given_fold():
    rows = _mk_rows()
    s_a = N.fit_s_out(rows, set(N.FOLD_A))
    s_b = N.fit_s_out(rows, set(N.FOLD_B))
    assert set(s_a) == {"delay_phys", "ttc3"}
    for s_map in (s_a, s_b):
        for name in s_map:
            for H in (30, 60):
                # the raw +/- band already covers the constant truth,
                # so the smallest factor (1.0) suffices
                assert s_map[name][H] == pytest.approx(1.0)


def test_gate_rows_pools_folds_and_uses_other_fold_factors():
    rows = _mk_rows()
    out = N.gate_rows(rows)
    for name in ("delay_phys", "ttc3"):
        for H in (30, 60):
            d = out[name][H]
            assert d["n_windows"] == 12       # 6 windows x 2 folds
            assert set(d["per_fold"]) == {"A", "B"}
            assert d["per_fold"]["A"]["n_windows"] == 6
            assert d["per_fold"]["A"]["cov"] == 1.0


def test_gate_verdicts_b_fail_and_pass():
    # (b): pred 10 vs true 10 -> pass; pred 14 vs true 10 -> fail
    rows_ok = _mk_rows()
    v_ok = N.gate_verdicts(rows_ok, horizons=(30,))
    assert v_ok[30]["b_lc_pass"] is True
    rows_bad = _mk_rows(bias_merge=4.0)
    v_bad = N.gate_verdicts(rows_bad, horizons=(30,))
    assert v_bad[30]["b_lc_pass"] is False
    assert "b_lc" in v_bad[30]["v2_failing"]
    assert v_bad[30]["v2_PASS"] is False
    assert "old_pass" in v_bad[30]["old"]
    # the (d) reference is the SAME-split L0 speed
    assert v_bad[30]["d_per_horizon_threshold"] == \
        pytest.approx(1.05 * 2.0)


def test_minmax_cov_widened_with_other_fold():
    rows = _mk_rows(spread=1.0)
    mm = N.minmax_cov(rows)
    assert 0.0 <= mm <= 1.0
    assert mm == 1.0                    # raw extremes already cover


# ------------------------------------------------ climatology/sharpness

def test_climatology_from_rows_dedups_members():
    rows = _mk_rows()
    clima = N.climatology_from_rows(rows)
    # truth side is member-independent: one value per (root, t0, H)
    assert len(clima[("delay_phys", 30)]) == 12
    assert set(np.unique(clima[("delay_phys", 30)])) == {10.0}


def test_climatology_table_crps_values():
    rows = _mk_rows()
    clima = N.climatology_from_rows(rows)
    tab = N.climatology_table(rows, clima)
    c = tab[("delay_phys", 30)]
    # the empirical TRAIN distribution is degenerate here (all 10.0),
    # so empirical and mean-point climatology CRPS coincide
    assert c["clima_crps_empirical"] == pytest.approx(0.0, abs=1e-6)
    assert c["clima_crps_meanpoint"] == pytest.approx(0.0, abs=1e-6)
    assert c["clima_samples"] == 12


def test_sharpness_fields_present():
    rows = _mk_rows(spread=2.0)
    out = N.gate_rows(rows)
    d = out["ttc3"][30]
    assert d["median_band_width"] > 0
    assert d["truth_iqr_width"] == pytest.approx(0.0)  # constant truth
    # zero truth spread -> callers must guard the ratio (aggregate does)


# ------------------------------------------------------ widened rows --

def test_widened_l6_rows_mean_invariant():
    rows = [r for r in _mk_rows() if r["H"] == 30 and
            r["variant"] == "L6"]
    s_map = {"delay_phys": {30: 3.0}, "ttc3": {}}
    out = N.widened_l6_rows(rows, s_map)
    by_win = {}
    for r in out:
        by_win.setdefault((r["root"], r["t0"]), {})[r["member"]] = r
    for memb in by_win.values():
        xs = np.array([m["delay_phys_pred"] for m in memb.values()])
        # input pair: 8/12 around the mean 10; widened x3 -> 6/14
        assert xs.mean() == pytest.approx(10.0, abs=1e-9)
        assert xs.max() - xs.min() == pytest.approx(12.0)
        # ttc3 untouched (no factor in the map)
        ys = np.array([m["pred_ttc3_exposure"] for m in memb.values()])
        assert ys.max() - ys.min() == pytest.approx(1.0)


# -------------------------------------------------------- freeze gate --

def test_write_and_check_frozen_roundtrip(tmp_path):
    f = tmp_path / "artifact.json"
    f.write_text(json.dumps({"a": 1}), encoding="utf-8")
    side = N.write_frozen(f, "note")
    assert N.check_frozen(
        f, tmp_path / "artifact.json.FROZEN.sha256")["sha256"] == \
        side["sha256"]
    # any modification must trip the gate
    f.write_text(json.dumps({"a": 2}), encoding="utf-8")
    with pytest.raises(SystemExit):
        N.check_frozen(f, tmp_path / "artifact.json.FROZEN.sha256")
    # missing files must trip the gate
    with pytest.raises(SystemExit):
        N.check_frozen(tmp_path / "nope.json",
                       tmp_path / "nope.json.FROZEN.sha256")


def test_other_fold():
    assert N.other_fold("A") == "B"
    assert N.other_fold("B") == "A"


def test_engine_end_force_zone_sizes():
    """The brief's force grid {60,100,150,200} is plain merge_params
    values: the relaxation zone grows with the value, never tightens,
    and 0.0 stays exactly the shipped behaviour."""
    import engine as E
    for fs in (60.0, 100.0, 150.0, 200.0):
        # outside the zone: untouched
        assert E.end_force_params(100.0, 2.4, 3.0, fs) == (2.4, 3.0)
        # inside the zone: tau capped, b floored
        assert E.end_force_params(E.ACCEL_END_S - fs + 1.0,
                                  2.4, 3.0, fs) == (0.5, 6.0)
    assert E.end_force_params(454.0, 2.4, 3.0, 0.0) == (2.4, 3.0)
