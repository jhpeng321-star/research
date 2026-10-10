"""Part 3q engine-flag tests.

The part-3q engine change is THREE new Rollout kwargs
(merge_cooldown_ticks, merge_early_d, merge_early_prob) whose DEFAULTS
must be exact no-ops so every earlier kind and prior-part config stays
byte-reproducible, plus Rollout.merge_events bookkeeping (pure
records).  The data-root integration tests re-roll one real window;
the committed-rows regression pins the no-op against the p3o iter2
gate rolls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import p3q_lib as Q

HERE = Path(__file__).resolve().parent


def _data_root():
    return os.environ.get("WD01_DATA_ROOT", "").strip()


def _p3o_cfg():
    """The p3o iter2 winner (member 0), i.e. p3q fields at defaults."""
    cand = Q.N.load_json(Q.PART3O / "results" /
                         "candidate_p3o_iter2.json")
    return dict(cand["members"][0])


# ------------------------------------------------------- pure unit tests

def _stub(lane, s, v, mp=None, **attrs):
    """Hand-built idm_merge Rollout stub with only what
    _gap_accept_merge touches (the part 3l test pattern, extended by
    the p3q attributes via getattr defaults)."""
    import engine as E
    import torch
    n = len(lane)
    r = E.Rollout.__new__(E.Rollout)
    r.dev = torch.device("cpu")
    r.kind = "idm_merge"
    r.vid = np.arange(n)
    r.lane = torch.tensor(lane, dtype=torch.long)
    r.s = torch.tensor(s, dtype=torch.float64)
    r.v = torch.tensor(v, dtype=torch.float64)
    r.typ = np.ones(n, dtype=np.int64)
    r.vtype = np.ones(n, dtype=np.int64)
    r.mp = dict(E.MERGE_PARAMS_DEFAULT)
    if mp:
        r.mp.update(mp)
    for k, val in attrs.items():
        setattr(r, k, val)
    return r


def test_stub_defaults_reproduce_gap_acceptance():
    """With no p3q attributes at all (plain part 3l stub), the merge
    decision is the shipped deterministic one and the stats keys stay
    canonical."""
    r = _stub([2, 1], [400.0, 370.0], [20.0, 20.0])
    out = r._gap_accept_merge()
    assert bool(out[0]) is True
    assert set(r.merge_stats.keys()) == {
        "checks", "accepted", "rej_lead", "rej_foll", "rej_both",
        "with_foll", "breq_hist", "breq_foll_rej"}


def test_cooldown_blocks_rethinning_for_cd_ticks():
    """cooldown=50 with mprob=0.0 (always thinned): the candidate is
    thinned at tick 10 (cd_set=1) and BLOCKED at tick 11 (cd_blocked),
    then allowed again at tick 60."""
    r = _stub([2, 1], [400.0, 370.0], [20.0, 20.0],
              merge_accept_prob=0.0, merge_cooldown_ticks=50,
              _merge_rng=None, member_seed=1, t0=120.0)
    r._m2_tick = 10
    out10 = r._gap_accept_merge()
    assert bool(out10[0]) is False
    assert r.merge_stats["rej_prob"] == 1
    assert r.merge_stats["cd_set"] == 1
    r._m2_tick = 11
    out11 = r._gap_accept_merge()
    assert bool(out11[0]) is False
    assert r.merge_stats["cd_blocked"] == 1
    r._m2_tick = 60
    out60 = r._gap_accept_merge()
    assert bool(out60[0]) is False
    assert r.merge_stats["cd_blocked"] == 1     # not blocked any more
    assert r.merge_stats["rej_prob"] == 2       # thinned again
    assert r.merge_stats["cd_set"] == 2


def test_cooldown_zero_never_blocks():
    r = _stub([2, 1], [400.0, 370.0], [20.0, 20.0], merge_accept_prob=0.0,
              member_seed=1, t0=120.0)
    for j in (10, 11, 12):
        r._m2_tick = j
        r._gap_accept_merge()
    assert "cd_blocked" not in r.merge_stats
    assert "cd_set" not in r.merge_stats
    assert getattr(r, "_merge_cd", None) is None


def test_early_prob_sets_early_counter_and_uses_d():
    """early_d = -15.44: the candidate at s=400 (d = -21.3) is NOT in
    the early zone; one at s=385 (d = -6.3) IS.  mprob = 0.0 (deep zone
    always thins), early_prob = 1.0 (early never thins) -> the early
    candidate merges, the deep one does not, and the counters split."""
    import torch
    # deep candidate only: s = 400 -> d = -21.3 <= early_d
    r1 = _stub([2, 1], [400.0, 370.0], [20.0, 20.0],
               merge_accept_prob=0.0, merge_early_d=-15.44,
               merge_early_prob=1.0, member_seed=1, t0=120.0)
    r1._m2_tick = 5
    out1 = r1._gap_accept_merge()
    assert bool(out1[0]) is False
    assert r1.merge_stats["rej_prob"] == 1
    assert "rej_prob_early" not in r1.merge_stats
    # early candidate only: s = 385 -> d = -6.3 > early_d
    r2 = _stub([2, 1], [385.0, 370.0], [20.0, 20.0],
               merge_accept_prob=0.0, merge_early_d=-15.44,
               merge_early_prob=1.0, member_seed=1, t0=120.0)
    r2._m2_tick = 5
    out2 = r2._gap_accept_merge()
    assert bool(out2[0]) is True
    assert r2.merge_stats.get("rej_prob_early", 0) == 0
    assert "rej_prob" not in r2.merge_stats
    assert torch.equal(out2, torch.tensor([True, False]))


def test_early_path_rng_stream_matches_p3o_semantics():
    """With early_prob = None the RNG consumption must be IDENTICAL to
    the p3o thinning (same seed, same draws for gap-accepted
    candidates).  Two stubs, same gaps, mprob 0.5: same accept/reject
    pattern with and without the (inert) early fields."""
    lanes, ss, vv = [2, 2, 1], [395.0, 380.0, 370.0], [20.0, 20.0, 20.0]
    import torch
    r1 = _stub(lanes, ss, vv, merge_accept_prob=0.5,
               member_seed=7, t0=120.0)
    r1._m2_tick = 3
    o1 = r1._gap_accept_merge()
    r2 = _stub(lanes, ss, vv, merge_accept_prob=0.5,
               merge_early_d=0.0, merge_early_prob=None,
               member_seed=7, t0=120.0)
    r2._m2_tick = 3
    o2 = r2._gap_accept_merge()
    assert torch.equal(o1, o2)
    assert r1.merge_stats == r2.merge_stats


# ------------------------------------------- data-root integration tests

@pytest.mark.skipif(not _data_root(), reason="WD01_DATA_ROOT not set")
def test_engine_defaults_noop_and_events_bookkeeping():
    """Defaults: same-config re-rolls are deterministic, the canonical
    merge_stats key set is unchanged, and merge_events mirror merge_d
    (one record per executed merge, same length/order)."""
    import p3lib as P3
    dev = P3.device()
    cfg = _p3o_cfg()
    r1 = Q.roll_member_q("cv20_7002m2000", 120.0, cfg, dev)
    r2 = Q.roll_member_q("cv20_7002m2000", 120.0, cfg, dev)
    assert r1.lc_count == r2.lc_count
    assert r1.merge_d == r2.merge_d
    assert len(r1.merge_events) == len(r1.merge_d)
    for e, d in zip(r1.merge_events, r1.merge_d):
        assert e["d"] == d
    assert r1.merge_events[0]["tick"] >= 1
    assert set(r1.merge_stats.keys()) == {
        "checks", "accepted", "rej_lead", "rej_foll", "rej_both",
        "with_foll", "breq_hist", "breq_foll_rej", "rej_prob"}


@pytest.mark.skipif(not _data_root(), reason="WD01_DATA_ROOT not set")
def test_engine_cooldown_reduces_or_keeps_merges_and_is_deterministic():
    """cooldown=8 at mprob 0.55 must never merge MORE than the p3o
    config on the same window, must log cd_set/cd_blocked, and must be
    deterministic on the flagged path."""
    import p3lib as P3
    dev = P3.device()
    base = _p3o_cfg()
    r_base = Q.roll_member_q("cv20_7002m2000", 120.0, base, dev)
    cd = dict(base, merge_cooldown_ticks=8)
    r1 = Q.roll_member_q("cv20_7002m2000", 120.0, cd, dev)
    r2 = Q.roll_member_q("cv20_7002m2000", 120.0, cd, dev)
    assert r1.lc_count == r2.lc_count
    assert r1.merge_d == r2.merge_d
    assert r1.merge_stats["cd_set"] > 0
    assert len(r1.merge_d) <= len(r_base.merge_d)
    assert r1.merge_stats["accepted"] + \
        r1.merge_stats.get("rej_prob", 0) >= 1


@pytest.mark.skipif(not _data_root(), reason="WD01_DATA_ROOT not set")
def test_engine_early_path_active_and_deterministic():
    """early_prob 0.1 at early_d -15.44 shifts thinnings into the early
    counter; the flagged config is deterministic and never merges MORE
    in the early zone at the moment of acceptance."""
    import p3lib as P3
    dev = P3.device()
    base = _p3o_cfg()
    early = dict(base, merge_early_d=-15.44, merge_early_prob=0.1)
    r1 = Q.roll_member_q("cv20_7002m2000", 120.0, early, dev)
    r2 = Q.roll_member_q("cv20_7002m2000", 120.0, early, dev)
    assert r1.lc_count == r2.lc_count and r1.merge_d == r2.merge_d
    st = r1.merge_stats
    assert st.get("rej_prob", 0) + st.get("rej_prob_early", 0) > 0
    assert len(r1.merge_d) <= len(
        Q.roll_member_q("cv20_7002m2000", 120.0, base, dev).merge_d)


@pytest.mark.skipif(not _data_root(), reason="WD01_DATA_ROOT not set")
def test_noop_regression_vs_committed_p3o_rows():
    """The p3q engine (new flags at defaults) must reproduce the
    COMMITTED p3o iter2 gate rolls on the calibration roots exactly
    (same box/env; the p3o raw rows were kept).  Any systematic gap
    would mean a non-no-op default."""
    import p3lib as P3
    dev = P3.device()
    cand = Q.N.load_json(Q.PART3O / "results" /
                         "candidate_p3o_iter2.json")
    fields = ("speed_rmse", "pos_rmse", "collisions_per_1000",
              "pred_lc", "true_lc", "delay_phys_pred", "delay_phys_true",
              "pred_ttc3_exposure", "true_ttc3_exposure",
              "pred_lc_merge", "pred_lc_disc", "true_lc_merge",
              "true_lc_disc")
    worst = {f: 0.0 for f in fields}
    n_cmp = 0
    for mi, mem in enumerate(cand["members"][:2]):   # 2 members ~ 4 min
        committed = Q.read_rows(
            Q.PART3O / "results" / "raw" /
            f"raw_train_internal_iter2_m{mi}.jsonl")
        by_key = {(r["root"], int(r["t0"]), int(r["H"])): r
                  for r in committed}
        for root in Q.N.FOLD_A + Q.N.FOLD_B:
            for t0 in Q.GATE_T0S:
                _r, rows = Q.roll_member_rows_q(
                    root, float(t0), dict(mem), mi, "L6", dev,
                    "p3q noop regression vs p3o iter2")
                for row in rows:
                    ref = by_key[(root, int(t0), int(row["H"]))]
                    for f in fields:
                        d = abs(float(row[f]) - float(ref[f]))
                        worst[f] = max(worst[f], d)
                    n_cmp += 1
    print(f"\ncompared {n_cmp} rows; worst abs diffs: "
          f"{ {k: round(v, 9) for k, v in worst.items()} }")
    for f in fields:
        assert worst[f] < 1e-6, f
