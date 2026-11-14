"""WD-01 part 3q TASK A: the shared raw basis + MERGE DIAGNOSIS on the
17 TRAIN roots (resumable per member file).

  results/raw/raw_p3q_L0.jsonl          L0 (idm_normal, deterministic)
                                        on 17 roots x 7 t0 -- the (d)
                                        speed reference + (c) L0 MAE;
  results/raw/raw_p3q_base_m{0..4}.jsonl
                                        the FROZEN p3o iter2 candidate
                                        (read-only) re-rolled on the
                                        17 TRAIN roots with the typed
                                        row builder -- the p3q baseline
                                        (iter 0) AND the merge
                                        diagnosis roll set (engine
                                        flags at no-op defaults);
  results/raw/raw_p3q_base_mevents_m{0..4}.json
                                        per-roll executed-merge events
                                        (tick/s/d/v) + rejection-cause
                                        counters (part 3q bookkeeping).

Diagnosis output (results/diag_p3q.json + stdout): LC by type pooled /
per k-fold / per root (the brief's first-class per-type metric), the
merge-position (s, distance to lane end) and merge-timing distributions
pred vs truth, and the pooled M2 rejection-cause counters -- the basis
for the merge-lever fit.

The 4 validation roots are NEVER loaded or scored; holdout blocked;
ZERO simulation.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("WD_DEVICE", "cpu")    # single process, shared box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p3q_lib as Q  # noqa: E402
import decompose_p3k as D  # noqa: E402

T_START = time.time()


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def merge_diag(rows_l6, mevents, roots):
    """The merge diagnosis block: LC by type, merge positions/timing
    pred vs truth, rejection causes."""
    out = {"lc_by_type": {}, "merge_pos_timing": {}, "merge_stats": None,
           "lc_by_root": {}}
    for H in Q.HORIZONS:
        out["lc_by_type"][f"H{H}"] = Q.by_type_block(rows_l6, H)
    # positions/timing: pred events pooled over members; truth computed
    # once per window (member-independent)
    pred_ev = []
    truth_ev = []
    for rec in mevents:
        pred_ev.extend(rec["events"])
    seen = set()
    for rec in mevents:
        key = (rec["root"], rec["t0"])
        if key in seen:
            continue
        seen.add(key)
        ra = Q.P3.root_arrays(rec["root"])
        truth_ev.extend(Q.truth_merge_rows(
            ra, int(round((rec["t0"] - 60.0) / 0.5)), Q.K_TICKS))
    for H in Q.HORIZONS:
        out["merge_pos_timing"][f"H{H}"] = dict(
            pred=Q.dist_summary(
                [dict(rel_s=e["tick"] * Q.DT_S, s=e["s"])
                 for e in pred_ev if e["tick"] * Q.DT_S <= H + 1e-9],
                horizons=(H,))[H],
            truth=Q.dist_summary(truth_ev, horizons=(H,))[H])
    out["merge_stats"] = Q.pool_merge_stats(
        [rec["stats"] for rec in mevents])
    # per-root LC by type (the caveat table; members pooled)
    for H in Q.HORIZONS:
        out["lc_by_root"][f"H{H}"] = {
            r: Q.lc_by_type(rows_l6, H, {r}) for r in sorted(roots)}
    return out


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    Q.RAW_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = D.Tee(Q.RES / "stdout_diag.txt")
    print("priority below-normal:", D.below_normal_priority())
    print("=== WD-01 part 3q TASK A: base rows + merge diagnosis on the "
          "17 TRAIN roots ===")
    print("ZERO simulation; holdout blocked; NO validation look (the "
          "validation roots are never loaded here).")
    print(f"device={Q.P3.device()} torch_threads={torch.get_num_threads()}")
    roots = Q.train17()
    print(f"TRAIN roots (17): {roots}")
    print(f"k={Q.K_FOLD} folds: {Q.FOLDS}")
    print(f"dev fit folds {Q.FIT_FOLDS} ({len(Q.FIT_ROOTS)} roots); "
          f"fold {Q.DEV_FOLD} held out of ALL parameter fitting")
    val = set(Q.O.splits()["val"])
    assert not (set(roots) & val)
    dev = Q.P3.device()

    cand = Q.N.load_json(Q.PART3O / "results" /
                         "candidate_p3o_iter2.json")
    members = cand["members"]
    print(f"base = the FROZEN p3o iter2 candidate "
          f"(sha256 {cand.get('sha256', 'see sidecar')}), "
          f"{len(members)} members")

    # ---- L0 on the 17 roots (the (d) reference + (c) L0 MAE)
    l0_path = Q.RAW_DIR / "raw_p3q_L0.jsonl"
    if Q.member_file_complete(l0_path, 2 * len(Q.GATE_T0S) * len(roots)):
        log(f"L0: {l0_path.name} complete -- skipped")
    else:
        n = 0
        with open(l0_path, "w", encoding="utf-8") as f:
            for root in roots:
                for t0 in Q.GATE_T0S:
                    r = Q.roll_l0(root, float(t0), dev)
                    for row in Q.N.window_raw_rows(
                            r, "L0", 0, root, t0, 1.0,
                            "part3q base: L0 idm_normal (deterministic)"):
                        f.write(json.dumps(row, default=str) + "\n")
                        n += 1
                    f.flush()
        log(f"L0: wrote {n} rows -> {l0_path.name}")

    # ---- base (p3o candidate) rolls + merge-event bookkeeping
    expect = 2 * len(Q.GATE_T0S) * len(roots)
    for mi, mem in enumerate(members):
        path = Q.RAW_DIR / f"raw_p3q_base_m{mi}.jsonl"
        ev_path = Q.RAW_DIR / f"raw_p3q_base_mevents_m{mi}.json"
        if (Q.member_file_complete(path, expect) and ev_path.exists()):
            log(f"base member {mi}: complete -- skipped")
            continue
        n = 0
        evs = []
        with open(path, "w", encoding="utf-8") as f:
            for root in roots:
                for t0 in Q.GATE_T0S:
                    r, rows = Q.roll_member_rows_q(
                        root, float(t0), dict(mem), mi, "L6", dev,
                        "part3q base (p3o iter2 candidate, engine at "
                        "defaults): " + Q.cfg_note(mem))
                    evs.append(Q.events_and_stats(r, root, t0, mi))
                    for row in rows:
                        f.write(json.dumps(row, default=str) + "\n")
                        n += 1
                    f.flush()
        ev_path.write_text(json.dumps(evs, default=str), encoding="utf-8")
        log(f"base member {mi}: wrote {n} rows + {len(evs)} event "
            f"records -> {path.name}")

    # ---- the merge diagnosis
    rows_l6 = []
    mevents = []
    for mi in range(len(members)):
        rows_l6 += Q.read_rows(Q.RAW_DIR / f"raw_p3q_base_m{mi}.jsonl")
        mevents += json.loads(
            (Q.RAW_DIR /
             f"raw_p3q_base_mevents_m{mi}.json").read_text())
    diag = merge_diag(rows_l6, mevents, roots)
    diag.update(dict(
        part="3q", task="diagnosis",
        basis="the frozen p3o iter2 candidate re-rolled on the 17 TRAIN "
              "roots x 7 t0 (5 members), typed rows + merge events",
        train_roots=list(roots),
        folds={k: list(v) for k, v in Q.FOLDS.items()},
        fit_folds=list(Q.FIT_FOLDS), dev_fold=Q.DEV_FOLD,
        holdout_accessed=False, simulation=False,
        validation_used=False, validation_loaded=False))
    (Q.RES / "diag_p3q.json").write_text(
        json.dumps(diag, indent=1, default=str), encoding="utf-8")

    print("\n== MERGE DIAGNOSIS (base = p3o iter2 candidate; 17 TRAIN "
          "roots, members pooled) ==")
    for H in Q.HORIZONS:
        print(f"  H={H}:")
        for name in ("pooled",) + Q.FOLD_NAMES:
            t = diag["lc_by_type"][f"H{H}"][name]
            print(f"    {name:<8} predM {t['pred_lc_merge']:5.2f} "
                  f"trueM {t['true_lc_merge']:5.2f} "
                  f"({100*t['dev_merge']:+6.1f}%) | predD "
                  f"{t['pred_lc_disc']:5.2f} trueD "
                  f"{t['true_lc_disc']:5.2f} "
                  f"({100*t['dev_disc']:+6.1f}%) | total "
                  f"{t['pred_lc']:5.2f}/{t['true_lc']:5.2f} "
                  f"({100*t['dev_total']:+6.1f}%)")
    print("\n== merge position/timing (per merge event; dist = m to the "
          "accel lane end) ==")
    for H in Q.HORIZONS:
        pt = diag["merge_pos_timing"][f"H{H}"]
        for side in ("pred", "truth"):
            s_ = pt[side]
            print(f"  H={H} {side:<5} n={s_['n']:4d} rel_s mean "
                  f"{s_['rel_mean']} p10/50/90 {s_['rel_p10']}/"
                  f"{s_['rel_p50']}/{s_['rel_p90']} | dist mean "
                  f"{s_['dist_mean']} p10/50/90 {s_['dist_p10']}/"
                  f"{s_['dist_p50']}/{s_['dist_p90']}")
    st = diag["merge_stats"]
    print(f"\n== M2 rejection causes (pooled over all rolls) ==")
    print(f"  checks {st['checks']} accepted {st['accepted']} "
          f"rej_lead {st['rej_lead']} rej_foll {st['rej_foll']} "
          f"rej_both {st['rej_both']} with_foll {st['with_foll']} "
          f"rej_prob {st['rej_prob']}")
    print("\n== per-root LC by type (H=30; the val8-style caveat table) "
          "==")
    for root in sorted(roots):
        t = diag["lc_by_root"]["H30"][root]
        print(f"  {root:<18} predM {t['pred_lc_merge']:5.2f} trueM "
              f"{t['true_lc_merge']:5.2f} ({100*t['dev_merge']:+6.1f}%) "
              f"| predD {t['pred_lc_disc']:5.2f} trueD "
              f"{t['true_lc_disc']:5.2f} ({100*t['dev_disc']:+6.1f}%) "
              f"| total ({100*t['dev_total']:+6.1f}%)")
    log("wrote results/diag_p3q.json")
    print("\nDONE part 3q diagnosis.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
