"""WD-01 part 3q TASK 0: no-op regression of the p3q engine flags
against the COMMITTED p3o iter2 gate rolls.

The part-3q engine change adds three Rollout kwargs
(merge_cooldown_ticks, merge_early_d, merge_early_prob) plus the
merge_events bookkeeping.  At their defaults the merge decision must be
byte-identical to the p3o engine, so re-rolling the p3o iter2 members
on the 4 calibration roots x 7 t0 must reproduce the committed raw rows
(raw_train_internal_iter2_m{mi}.jsonl) EXACTLY (same box/env).  A
systematic gap would mean a non-no-op default (a real leak moves LC
totals by 10-40 %, not 1e-9).

ZERO simulation; validation roots never loaded; holdout blocked.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("WD_DEVICE", "cpu")    # single process, shared box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p3q_lib as Q  # noqa: E402
import decompose_p3k as D  # noqa: E402

T_START = time.time()
FIELDS = ("speed_rmse", "pos_rmse", "collisions_per_1000",
          "pred_lc", "true_lc", "delay_phys_pred", "delay_phys_true",
          "pred_ttc3_exposure", "true_ttc3_exposure",
          "pred_lc_merge", "pred_lc_disc", "true_lc_merge",
          "true_lc_disc")


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    Q.RES.mkdir(parents=True, exist_ok=True)
    sys.stdout = D.Tee(Q.RES / "stdout_check_noop.txt")
    print("priority below-normal:", D.below_normal_priority())
    print("=== WD-01 part 3q TASK 0: no-op regression vs the committed "
          "p3o iter2 gate rolls ===")
    print("ZERO simulation; holdout blocked; validation roots NEVER "
          "loaded here.")
    dev = Q.P3.device()
    cand = Q.N.load_json(Q.PART3O / "results" /
                         "candidate_p3o_iter2.json")
    roots = tuple(Q.N.FOLD_A) + tuple(Q.N.FOLD_B)
    print(f"p3o iter2 members {len(cand['members'])}, calibration roots "
          f"{roots}, t0 {Q.GATE_T0S}")
    worst = {f: 0.0 for f in FIELDS}
    n_cmp = 0
    for mi, mem in enumerate(cand["members"]):
        committed = Q.read_rows(
            Q.PART3O / "results" / "raw" /
            f"raw_train_internal_iter2_m{mi}.jsonl")
        by_key = {(r["root"], int(r["t0"]), int(r["H"])): r
                  for r in committed}
        for root in roots:
            for t0 in Q.GATE_T0S:
                _r, rows = Q.roll_member_rows_q(
                    root, float(t0), dict(mem), mi, "L6", dev,
                    "p3q check-noop re-roll of the p3o iter2 member")
                for row in rows:
                    ref = by_key[(root, int(t0), int(row["H"]))]
                    for f in FIELDS:
                        d = abs(float(row[f]) - float(ref[f]))
                        worst[f] = max(worst[f], d)
                    n_cmp += 1
        log(f"member {mi}: compared {n_cmp} rows so far; worst "
            f"{max(worst.values()):.3g}")
    print("\n== worst abs diff per field vs the committed p3o rows ==")
    ok = True
    for f in FIELDS:
        flag = "OK" if worst[f] < 1e-6 else "MISMATCH"
        ok &= worst[f] < 1e-6
        print(f"  {f:<24} {worst[f]:.3g}  {flag}")
    print(f"\ncompared {n_cmp} (window, H) rows over "
          f"{len(cand['members'])} members")
    print("NO-OP REGRESSION " + ("OK -- the p3q engine defaults are "
                                 "byte-identical to the p3o engine"
                                 if ok else "FAILED -- investigate"))
    if not ok:
        raise SystemExit(1)
    print("\nDONE part 3q check-noop.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
