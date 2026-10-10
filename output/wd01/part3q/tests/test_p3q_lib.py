"""Part 3q library unit tests (no data root needed): fold declaration,
k-fold cross-fitting semantics, the bias/width breakdown algebra, the
merge-event builders, and the goal-package semantics."""
from __future__ import annotations

import numpy as np
import pytest

import p3q_lib as Q
import p3n_lib as N


# ----------------------------------------------------- fold declaration

def test_train17_and_folds():
    roots = Q.train17()
    assert len(roots) == 17
    sp = Q.O.splits()
    assert not (set(roots) & set(sp["val"]))
    assert not (set(roots) & set(sp["holdout"]))
    # fit + calibration roots of the earlier parts are included
    assert set(sp["fit"]) <= set(roots)
    assert set(sp["calibration"]) <= set(roots)
    folds = Q.folds_by_root(roots)
    assert sorted(folds) == sorted(Q.FOLD_NAMES)
    union = [r for rs in folds.values() for r in rs]
    assert sorted(union) == roots
    assert len(union) == len(set(union))          # disjoint
    sizes = sorted(len(v) for v in folds.values())
    assert sizes == [3, 3, 3, 4, 4]
    # determinism
    assert folds == Q.folds_by_root(roots)


def test_fold_of_and_other_folds():
    assert Q.fold_of("cv17") == "f0"
    for r in Q.FOLDS["f0"]:
        assert Q.fold_of(r) == "f0"
    assert Q.other_folds("f0") == ("f1", "f2", "f3", "f4")
    assert Q.DEV_FOLD == "f0"
    assert "f0" not in Q.FIT_FOLDS
    assert len(Q.FIT_ROOTS) == 13


# ---------------------------------------------------- synthetic gate rows

def _row(root, t0, H, member, delay_pred, delay_true, t3p, t3t,
         lc=(4, 8, 4, 8), speed=2.7, coll=0.0):
    return dict(
        variant="L6", member=member, root=root, t0=t0, H=H,
        speed_rmse=speed, pos_rmse=30.0,
        collisions_per_1000=coll, pred_lc=lc[0] + lc[1],
        true_lc=lc[2] + lc[3], delay_pred=1.0, delay_true=1.0,
        pred_vehicle_steps=100, matched_vehicle_steps=100,
        delay_phys_pred=delay_pred, delay_phys_true=delay_true,
        phys_steps_pred=100, phys_steps_true=100,
        pred_ttc3_exposure=t3p, true_ttc3_exposure=t3t,
        pred_hard_brake_s=0.0, true_hard_brake_s=0.0,
        any_ttc3_pred=False, any_ttc3_true=False, any_hb_pred=False,
        any_hb_true=False, lane_frac=0.9, dawdle_scale=1.0,
        engine="synthetic",
        pred_lc_merge=lc[0], pred_lc_disc=lc[1],
        true_lc_merge=lc[2], true_lc_disc=lc[3])


def _synth_rows(n_mem=5, spread=10.0, truth_off=0.0):
    """Every fold: delay ensemble centred on the window truth (grid
    spread), truths varying with t0 (non-degenerate climatology)."""
    rows = []
    for f, rs in Q.FOLDS.items():
        for root in rs:
            for t0 in Q.GATE_T0S:
                y = 100.0 + 40.0 * (t0 - 120) / 360.0 + truth_off
                for H in Q.HORIZONS:
                    for mi in range(n_mem):
                        dp = y + spread * (mi / (n_mem - 1) - 0.5)
                        rows.append(_row(root, t0, H, mi, dp, y,
                                         5.0 + 0.1 * mi, 5.0,
                                         lc=(4 + mi, 8, 4, 8)))
    return rows


def _l0_rows():
    rows = []
    for rs in Q.FOLDS.values():
        for root in rs:
            for t0 in Q.GATE_T0S:
                for H in Q.HORIZONS:
                    rows.append(dict(
                        variant="L0", member=0, root=root, t0=t0, H=H,
                        speed_rmse=2.8, pos_rmse=29.0,
                        collisions_per_1000=0.0, pred_lc=0, true_lc=12,
                        delay_pred=1.0, delay_true=1.0,
                        pred_vehicle_steps=100,
                        matched_vehicle_steps=100,
                        delay_phys_pred=110.0, delay_phys_true=100.0,
                        phys_steps_pred=100, phys_steps_true=100,
                        pred_ttc3_exposure=0.0,
                        true_ttc3_exposure=5.0,
                        pred_hard_brake_s=0.0, true_hard_brake_s=0.0,
                        any_ttc3_pred=False, any_ttc3_true=False,
                        any_hb_pred=False, any_hb_true=False,
                        lane_frac=0.9, dawdle_scale=1.0,
                        engine="synthetic"))
    return rows


def test_gate_rows_kf_cross_fits_by_construction():
    """Each fold's s_out must equal a factor fitted on the OTHER folds'
    rows only (the brief's 'never on the fold scored')."""
    rows = _synth_rows()
    cov = Q.gate_rows_kf(rows)
    for name in ("delay_phys", "ttc3"):
        for H in Q.HORIZONS:
            for fname in Q.FOLD_NAMES:
                want = N.fit_s_out(
                    rows, Q._fit_roots_of(fname, Q.FOLDS))[name][H]
                got = cov[name][H]["per_fold"][fname]["s_out"]
                assert got == want


def test_gate_rows_kf_bias_width_algebra():
    """below + in + above == n; a spread big enough to cover the fixed
    truth gives in_frac == 1; the residuals (truth minus centre) are
    widening-invariant."""
    rows = _synth_rows(spread=10.0)
    cov = Q.gate_rows_kf(rows)
    for name in ("delay_phys", "ttc3"):
        for H in Q.HORIZONS:
            c = cov[name][H]
            assert c["below_frac"] + c["in_frac"] + c["above_frac"] == \
                pytest.approx(1.0)
            assert c["n_windows"] == 17 * 7
            # delay: centre = truth -> residuals exactly 0;
            # ttc3: truth 5.0 vs ensemble mean 5.2 -> bias exactly -0.2
            if name == "delay_phys":
                assert c["bias_mean"] == pytest.approx(0.0, abs=1e-9)
                assert c["resid_med"] == pytest.approx(0.0, abs=1e-9)
            else:
                assert c["bias_mean"] == pytest.approx(-0.2, abs=1e-9)
    # a huge grid factor must reach the target on this easy synthetic
    s = cov["delay_phys"][30]["per_fold"]["f1"]["s_out"]
    assert s is not None and s <= 16.0
    assert cov["delay_phys"][30]["cov"] >= 0.8


def test_gate_rows_kf_records_miss_direction():
    """Ensembles shifted far BELOW their truths -> misses must be
    'above' (band too low), with positive bias_mean = truth - centre."""
    rows = _synth_rows(spread=10.0)
    for r in rows:
        r["delay_phys_pred"] -= 500.0
    cov = Q.gate_rows_kf(rows)
    c = cov["delay_phys"][30]
    assert c["above_frac"] > 0.5
    assert c["below_frac"] < 0.5
    assert c["bias_mean"] == pytest.approx(500.0, abs=1e-9)


def test_gate_verdicts_kf_shape_and_pooling():
    rows = _synth_rows() + _l0_rows()
    gate = Q.gate_verdicts_kf(rows)
    assert set(gate["per_fold"]) == set(Q.FOLD_NAMES)
    for H in Q.HORIZONS:
        g = gate["pooled"][H]
        assert g["v2_PASS"] is True          # easy synthetic
        assert g["lc_pred"] > 0 and g["lc_true"] > 0
        for fname in Q.FOLD_NAMES:
            gf = gate["per_fold"][fname][H]
            assert gf["v2_PASS"] is True
            assert "delay_phys" in gf["fold_s_out"]
        bw = g["bias_width"]["delay_phys"]
        assert bw["below_frac"] + bw["in_frac"] + bw["above_frac"] == \
            pytest.approx(1.0)


def test_climatology_kf_out_of_fold():
    rows = _synth_rows()
    cl = Q.climatology_kf(rows)
    for H in Q.HORIZONS:
        c = cl["delay_phys"][H]
        assert c["n_windows"] == 17 * 7
        assert c["clima_crps_empirical"] is not None
        # a per-window clima built from 13-14 roots has more samples
        # than the 5-member ensemble -> its CRPS against a fixed truth
        # stays finite and positive
        assert c["clima_crps_empirical"] > 0
        assert set(c["per_fold"]) == set(Q.FOLD_NAMES)


# --------------------------------------------------- merge event helpers

def test_truth_merge_rows_filters_2to1():
    class RA:
        lc_vid = np.array([1, 1, 2, 3])
        lc_tick = np.array([10, 20, 15, 30])
        lc_from = np.array([2, 1, 2, 2])
        lc_to = np.array([1, 0, 1, 1])
        lc_s = np.array([400.0, 410.0, 420.0, 430.0])

    rows = Q.truth_merge_rows(RA(), k0=10, K=12)
    # (10, 22] window, 2->1 only: vid 2 at tick 15 (vid 1's tick-10 event
    # is at k0 itself -> excluded, its tick-20 event is 1->0; vid 3 at
    # tick 30 is outside k0+K=22)
    assert len(rows) == 1
    assert rows[0]["rel_s"] == 2.5 and rows[0]["s"] == 420.0


def test_pred_merge_rows_horizon_filter():
    class R:
        merge_events = [dict(tick=10, vid=1, s=400.0, d=-21.3, v=20.0),
                        dict(tick=70, vid=2, s=410.0, d=-31.3, v=19.0)]

    out = Q.pred_merge_rows(R(), 30)
    assert len(out) == 1
    assert out[0]["rel_s"] == 5.0
    out60 = Q.pred_merge_rows(R(), 60)
    assert len(out60) == 2


def test_dist_summary_and_pool_merge_stats():
    ev = [dict(rel_s=5.0, s=400.0), dict(rel_s=20.0, s=430.0)]
    s = Q.dist_summary(ev, horizons=(30,))
    assert s[30]["n"] == 2
    assert s[30]["rel_p50"] == 12.5
    assert s[30]["dist_p50"] == pytest.approx(454.10 - 415.0)
    st = Q.pool_merge_stats([dict(checks=2, accepted=1, rej_prob=1),
                             dict(checks=3, accepted=2, rej_prob=1,
                                  cd_set=1)])
    assert st["checks"] == 5 and st["accepted"] == 3
    assert st["rej_prob"] == 2 and st["cd_set"] == 1
    assert st["cd_blocked"] == 0


def test_by_type_block_folds():
    rows = []
    for H in Q.HORIZONS:
        for f, rs in Q.FOLDS.items():
            for root in rs:
                rows.append(_row(root, 120, H, 0, 100.0, 100.0, 5, 5,
                                 lc=(4, 8, 4, 8)))
    blk = Q.by_type_block(rows, 30)
    assert blk["pooled"]["pred_lc_merge"] == 4.0
    for f in Q.FOLD_NAMES:
        assert blk[f]["dev_merge"] == 0.0
        # the synthetic rows carry one t0 per root only
        assert blk[f]["n_windows"] == len(Q.FOLDS[f])


# ------------------------------------------------------ fit field helpers

def test_set_get_field_roundtrip():
    cfg = dict(seed=1, lc_scale=0.002,
               merge_params=dict(tau_accept=2.4, b_accept=1.0,
                                 p_coop_scale=0.0, end_force_s=60.0,
                                 v_merge_blend=0.0),
               dawdle_scale=1.0, merge_accept_prob=0.55,
               lc_scale_disc=0.0016)
    c2 = Q.set_field(cfg, "merge_cooldown_ticks", 8)
    assert Q.get_field(c2, "merge_cooldown_ticks") == 8
    assert Q.get_field(cfg, "merge_cooldown_ticks") == 0
    c3 = Q.set_field(c2, "merge_early_prob", 0.3)
    assert Q.get_field(c3, "merge_early_prob") == 0.3
    c4 = Q.set_field(c3, "b_accept", 2.0)
    assert Q.get_field(c4, "b_accept") == 2.0
    assert Q.get_field(cfg, "b_accept") == 1.0
    assert Q.get_field(cfg, "merge_early_d") == 0.0
    assert Q.get_field(cfg, "merge_early_prob") is None
