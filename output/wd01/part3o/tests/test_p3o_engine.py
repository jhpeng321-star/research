"""Part 3o engine-flag tests.

The part-3o engine change is TWO new Rollout kwargs (lc_scale_disc,
merge_accept_prob) whose DEFAULTS must be exact no-ops so every
earlier kind and prior-part config stays byte-reproducible.  The
data-root integration tests re-roll one real window both ways; the
pure unit tests exercise the hazard algebra without any data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import p3o_lib as O

HERE = Path(__file__).resolve().parent


def _data_root():
    return os.environ.get("WD01_DATA_ROOT", "").strip()


def _base_cfg():
    return dict(seed=1, lc_scale=0.002,
                merge_params=dict(tau_accept=1.9, b_accept=2.0,
                                  p_coop_scale=0.0, end_force_s=60.0,
                                  v_merge_blend=0.0),
                dawdle_scale=1.0)


# ------------------------------------------------------- pure unit tests

def test_lc_by_type_totals_derive_from_types():
    rows = [dict(variant="L6", H=30, root="r",
                 pred_lc_merge=4.0, pred_lc_disc=9.0,
                 true_lc_merge=3.0, true_lc_disc=8.0),
            dict(variant="L6", H=30, root="r",
                 pred_lc_merge=2.0, pred_lc_disc=5.0,
                 true_lc_merge=3.0, true_lc_disc=5.0)]
    t = O.lc_by_type(rows, 30)
    # pooled: pred 3.0 merge + 7.0 disc; true 3.0 + 6.5
    assert t["pred_lc"] == pytest.approx(10.0)
    assert t["true_lc"] == pytest.approx(9.5)
    assert t["dev_merge"] == pytest.approx(0.0)
    assert t["dev_disc"] == pytest.approx(round((7.0 - 6.5) / 6.5, 4))
    assert t["dev_total"] == pytest.approx(round((10.0 - 9.5) / 9.5, 4))


def test_lc_by_type_root_filter():
    rows = [dict(variant="L6", H=60, root="a",
                 pred_lc_merge=1, pred_lc_disc=1,
                 true_lc_merge=1, true_lc_disc=1),
            dict(variant="L6", H=60, root="b",
                 pred_lc_merge=9, pred_lc_disc=9,
                 true_lc_merge=1, true_lc_disc=1),
            dict(variant="L0", H=60, root="a",
                 pred_lc_merge=99, pred_lc_disc=99,
                 true_lc_merge=1, true_lc_disc=1)]
    t = O.lc_by_type(rows, 60, roots={"a"})
    assert t["n_windows"] == 1
    assert t["pred_lc"] == 2.0


def test_goal_package_checks_all_criteria():
    class G:
        pass

    def verdicts(lc_dev, coll, speed, d_thr, crps):
        return {30: dict(b_lc_pass=lc_dev <= 0.20,
                         a_collisions_pass=coll <= 1.0,
                         d_per_horizon_pass=speed <= d_thr,
                         crps_phys=crps, collisions_per_1000=coll,
                         lc_deviation=lc_dev, lc_pred=1, lc_true=1,
                         speed_rmse=speed,
                         d_per_horizon_threshold=d_thr,
                         ttc3_cov_frac=0.5, delay_phys_cov_frac=0.5),
                60: dict(b_lc_pass=lc_dev <= 0.20,
                         a_collisions_pass=coll <= 1.0,
                         d_per_horizon_pass=speed <= d_thr,
                         crps_phys=crps, collisions_per_1000=coll,
                         lc_deviation=lc_dev, lc_pred=1, lc_true=1,
                         speed_rmse=speed,
                         d_per_horizon_threshold=d_thr,
                         ttc3_cov_frac=0.5, delay_phys_cov_frac=0.5)}

    # on-target by type (using the REAL fold roots so every table has
    # windows), (a)/(d) pass, crps better -> PASS
    rows = [dict(variant="L6", H=H, root=root,
                 pred_lc_merge=3.0, pred_lc_disc=8.0,
                 true_lc_merge=3.0, true_lc_disc=8.0)
            for H in (30, 60) for root in O.FOLDS["A"] + O.FOLDS["B"]]
    ref = {30: dict(crps=35.0), 60: dict(crps=82.0)}
    goal = O.goal_package(rows, verdicts(0.0, 0.01, 3.0, 4.0, 34.0),
                          ref)
    assert all(goal[H]["checks"]["PASS"] for H in (30, 60))
    # crps worse than the p3n iter1 reference -> crps_not_worse fails
    goal2 = O.goal_package(rows, verdicts(0.0, 0.01, 3.0, 4.0, 36.0),
                           ref)
    assert not goal2[30]["checks"]["crps_not_worse"]
    assert not goal2[30]["checks"]["PASS"]
    # by-type band breach (disc) -> b_type fails even when the gate
    # total criterion passes
    rows3 = [dict(variant="L6", H=H, root=root,
                  pred_lc_merge=3.0, pred_lc_disc=11.0,
                  true_lc_merge=3.0, true_lc_disc=8.0)
             for H in (30, 60) for root in O.FOLDS["A"] + O.FOLDS["B"]]
    goal3 = O.goal_package(rows3,
                           verdicts(0.18, 0.01, 3.0, 4.0, 34.0), ref)
    assert goal3[30]["checks"]["b_total_gate"] is True
    assert goal3[30]["checks"]["b_type_pooled"] is False


def test_goal_package_by_type_uses_folds():
    """Per-FOLD tables are reported as diagnostics; the gating
    b_type check is POOLED (the same pooling as the formal gate-v2
    (b)): fold A merges -15 %, fold B +25 % -> pooled +5 % passes."""
    def verdicts(_):
        return {H: dict(b_lc_pass=True, a_collisions_pass=True,
                        d_per_horizon_pass=True, crps_phys=30.0,
                        collisions_per_1000=0.0, lc_deviation=0.0,
                        lc_pred=1, lc_true=1, speed_rmse=3.0,
                        d_per_horizon_threshold=4.0, ttc3_cov_frac=0.9,
                        delay_phys_cov_frac=0.9)
                for H in (30, 60)}
    rows = []
    for H in (30, 60):
        for root in O.FOLDS["A"]:       # fold A merges -15 %
            rows.append(dict(variant="L6", H=H, root=root,
                             pred_lc_merge=2.55, pred_lc_disc=8.0,
                             true_lc_merge=3.0, true_lc_disc=8.0))
        for root in O.FOLDS["B"]:       # fold B merges +25 %
            rows.append(dict(variant="L6", H=H, root=root,
                             pred_lc_merge=3.75, pred_lc_disc=8.0,
                             true_lc_merge=3.0, true_lc_disc=8.0))
    ref = {30: dict(crps=35.0), 60: dict(crps=82.0)}
    goal = O.goal_package(rows, verdicts(0), ref)
    assert goal[30]["checks"]["b_type_pooled"] is True
    assert goal[30]["by_type"]["fold_B"]["dev_merge"] == \
        pytest.approx(0.25)
    assert goal[30]["by_type"]["fold_A"]["dev_merge"] == \
        pytest.approx(-0.15)
    assert goal[30]["checks"]["PASS"] is True


def test_member_file_complete_resumability(tmp_path):
    p = tmp_path / "rows.jsonl"
    assert not O.member_file_complete(p, 2)
    O.write_rows(p, [dict(a=1)])
    assert not O.member_file_complete(p, 2)
    O.write_rows(p, [dict(a=1), dict(a=2)], mode="a")
    assert O.member_file_complete(p, 2)
    p.write_text("not json\n", encoding="utf-8")
    assert not O.member_file_complete(p, 1)


def test_p3n_iter1_reference_shape():
    ref = O.p3n_iter1_reference()
    for H in O.HORIZONS:
        assert ref[H]["crps"] is not None
        assert ref[H]["crps"] > 0


# ------------------------------------------- data-root integration tests

@pytest.mark.skipif(not _data_root(), reason="WD01_DATA_ROOT not set")
def test_engine_default_is_noop_and_flagged_path_changes_counts():
    """Same window, same seed: (1) the default config must reproduce
    byte-identical rollouts (determinism), (2) the flagged config must
    keep the canonical merge_stats keys and strictly thin merges,
    (3) the canonical key SET stays canonical at defaults (the part 3l
    transcription test's invariant)."""
    import p3lib as P3
    dev = P3.device()
    cfg = _base_cfg()
    r1 = O.roll_member("cv20_7002m2000", 120.0, cfg, dev)
    r2 = O.roll_member("cv20_7002m2000", 120.0, cfg, dev)
    assert r1.lc_count == r2.lc_count
    assert r1.merge_d == r2.merge_d
    assert "rej_prob" not in r1.merge_stats
    canon = {"checks", "accepted", "rej_lead", "rej_foll", "rej_both",
             "with_foll", "breq_hist", "breq_foll_rej"}
    assert set(r1.merge_stats.keys()) == canon

    flagged = dict(cfg, lc_scale_disc=0.0005, merge_accept_prob=0.5)
    r3 = O.roll_member("cv20_7002m2000", 120.0, flagged, dev)
    r4 = O.roll_member("cv20_7002m2000", 120.0, flagged, dev)
    # the flagged config is itself deterministic, thins merges (the
    # rej_prob counter is present and positive), and never merges MORE
    # than the deterministic acceptance at the moment of acceptance
    assert r3.lc_count == r4.lc_count
    assert r3.merge_d == r4.merge_d
    assert r3.merge_stats["rej_prob"] > 0
    assert (r3.merge_stats["accepted"] +
            r3.merge_stats["rej_prob"]) >= r1.merge_stats["accepted"]
    assert len(r3.merge_d) <= len(r1.merge_d)
    assert r3.lc_count <= r1.lc_count


@pytest.mark.skipif(not _data_root(), reason="WD01_DATA_ROOT not set")
def test_typed_rows_carry_by_type_consistent_with_totals():
    """The typed builder's by-type counts must sum to the row's total
    LC on BOTH sides (the by-type gate table's internal consistency)."""
    import p3lib as P3
    dev = P3.device()
    cfg = _base_cfg()
    r = O.roll_member("cv20_7002m2000", 120.0, cfg, dev)
    rows = O.window_raw_rows_typed(r, "L6", 0, "cv20_7002m2000", 120,
                                   1.0, "test")
    assert len(rows) == 2
    for row in rows:
        assert (row["pred_lc_merge"] + row["pred_lc_disc"] ==
                row["pred_lc"])
        assert (row["true_lc_merge"] + row["true_lc_disc"] ==
                row["true_lc"])
        for k in ("pred_lc_merge", "pred_lc_disc", "true_lc_merge",
                  "true_lc_disc"):
            assert isinstance(row[k], int)


@pytest.mark.skipif(not _data_root(), reason="WD01_DATA_ROOT not set")
def test_holdout_and_validation_roots_refused_or_unused():
    """The physical holdout block is authoritative (inherited); the
    p3o scripts' declared roots never intersect val/holdout."""
    import p3lib as P3
    sp = O.splits()
    for root in sp["holdout"]:
        with pytest.raises(Exception):
            P3.root_arrays(root)
    used = set(sp["fit"]) | set(sp["calibration"])
    assert not (used & set(sp["val"]))
    assert not (used & set(sp["holdout"]))
