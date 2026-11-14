"""WD-01 part 3o TASK A': the shared raw basis, rolled ONCE and reused
by every iteration (resumable per member file).

  results/raw/raw_base_L0.jsonl            L0 (idm_normal, deterministic)
                                           on the 4 calibration roots x
                                           7 t0 -- the (d) speed
                                           reference for the gate;
  results/raw/raw_train_internal_iter0_m{0..4}.jsonl
                                           the p3n iteration-1 L6 members
                                           (part3n fit_p3n.json,
                                           read-only) re-rolled on the 4
                                           calibration roots with the
                                           TYPED row builder -- this IS
                                           the p3o iteration 0 (baseline
                                           = p3n iter1); the engine's
                                           part-3o flags are at their
                                           no-op defaults here, so the
                                           totals must reproduce the
                                           committed p3n iter1 gate;
  results/raw/raw_fit_truth_m0.jsonl       member 0 on the 4 FIT roots x
                                           7 t0 -- ONLY its
                                           member-independent truth
                                           sides are read (the
                                           climatology source, the p3n
                                           gate protocol).

All part-3o knobs at no-op defaults; ZERO simulation; the validation
roots are never passed to root_arrays; holdout blocked.
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
import p3o_lib as O  # noqa: E402
import p3n_lib as N  # noqa: E402
import decompose_p3k as D  # noqa: E402
import p3lib as P3  # noqa: E402
import engine as E  # noqa: E402

T_START = time.time()
ROWS_PER_MEMBER = 2 * len(O.GATE_T0S)      # one row per (window, H)


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    O.RAW_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = D.Tee(HERE / "results" / "stdout_base.txt")
    print("priority below-normal:", D.below_normal_priority())
    print("=== WD-01 part 3o TASK A': base rows (L0 + iter0 p3n "
          "members TYPED + fit-truth) ===")
    print("ZERO simulation; holdout blocked; validation roots NEVER "
          "loaded here.")
    print(f"device={P3.device()} torch_threads={torch.get_num_threads()}")
    sp = O.splits()
    cal, fit = sp["calibration"], sp["fit"]
    assert not (set(cal) & set(fit))
    val = set(sp["val"])
    assert not (set(cal) & val) and not (set(fit) & val)
    print(f"calibration roots (gate split): {cal}")
    print(f"fit roots (climatology truth sides): {fit}")
    print(f"validation {sorted(val)} NOT touched; holdout blocked")

    dev = P3.device()
    p3n_fit = N.load_json(O.PART3N / "results" / "fit_p3n.json")
    members = p3n_fit["members"]
    print(f"iter 0 members = p3n iter1 members "
          f"(part3n/results/fit_p3n.json, read-only): {len(members)}")

    # ---- L0 on the calibration roots (the (d) reference)
    l0_path = O.RAW_DIR / "raw_base_L0.jsonl"
    if O.member_file_complete(l0_path, ROWS_PER_MEMBER):
        log(f"L0: {l0_path.name} complete -- skipped")
    else:
        n = 0
        with open(l0_path, "w", encoding="utf-8") as f:
            for root in cal:
                for t0 in O.GATE_T0S:
                    r = O.roll_l0(root, float(t0), dev)
                    for row in N.window_raw_rows(
                            r, "L0", 0, root, t0, 1.0,
                            "part3o base: L0 idm_normal (deterministic; "
                            "engine lane-change paths untouched for "
                            "idm_normal)"):
                        f.write(json.dumps(row, default=str) + "\n")
                        n += 1
                    f.flush()
        log(f"L0: wrote {n} rows -> {l0_path.name}")

    # ---- iter 0 = the p3n members, TYPED rows, calibration roots
    for mi, mem in enumerate(members):
        path = O.RAW_DIR / f"raw_train_internal_iter0_m{mi}.jsonl"
        if O.member_file_complete(path, ROWS_PER_MEMBER):
            log(f"iter0 member {mi}: {path.name} complete -- skipped")
            continue
        cfg = dict(mem)
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for root in cal:
                for t0 in O.GATE_T0S:
                    rows = O.roll_member_rows(
                        root, float(t0), cfg, mi, "L6", dev,
                        "part3o iter0 (p3n iter1 members; part-3o "
                        "flags at no-op defaults): " + O.cfg_note(cfg))
                    for row in rows:
                        f.write(json.dumps(row, default=str) + "\n")
                        n += 1
                    f.flush()
        log(f"iter0 member {mi}: wrote {n} rows -> {path.name}")

    # ---- fit-root truth sides (climatology source; member 0 only)
    ft_path = O.RAW_DIR / "raw_fit_truth_m0.jsonl"
    if O.member_file_complete(ft_path, ROWS_PER_MEMBER):
        log(f"fit truth: {ft_path.name} complete -- skipped")
    else:
        cfg = dict(members[0])
        n = 0
        with open(ft_path, "w", encoding="utf-8") as f:
            for root in fit:
                for t0 in O.GATE_T0S:
                    rows = O.roll_member_rows(
                        root, float(t0), cfg, 0, "L6", dev,
                        "part3o base: fit-root roll; ONLY the truth "
                        "sides are read (climatology): " +
                        O.cfg_note(cfg))
                    for row in rows:
                        f.write(json.dumps(row, default=str) + "\n")
                        n += 1
                    f.flush()
        log(f"fit truth: wrote {n} rows -> {ft_path.name}")

    # ---- quick sanity: by-type pooled table for iter 0
    rows = []
    for mi in range(len(members)):
        rows += O.read_rows(O.RAW_DIR /
                            f"raw_train_internal_iter0_m{mi}.jsonl")
    print("\n== iter0 (p3n iter1 members) LC by type, calibration "
          "roots, all members pooled ==")
    for H in O.HORIZONS:
        t = O.lc_by_type(rows, H)
        print(f"  H={H}: predM {t['pred_lc_merge']:.2f} "
              f"trueM {t['true_lc_merge']:.2f} "
              f"({100*t['dev_merge']:+.1f}%) | predD "
              f"{t['pred_lc_disc']:.2f} trueD {t['true_lc_disc']:.2f} "
              f"({100*t['dev_disc']:+.1f}%) | total "
              f"{t['pred_lc']:.2f}/{t['true_lc']:.2f} "
              f"({100*t['dev_total']:+.1f}%)")

    # ---- regression check: iter0 totals must reproduce the committed
    # p3n iter1 gate.  The p3n session's raw rolls were not kept and ran
    # on a different box, so exact equality is not attainable; the
    # tolerance guards against REAL flag leakage (a non-no-op default
    # would move totals by 10-40 %, not sub-percent).  Our own rolls
    # are deterministic given a fixed process config (unit-tested).
    ref = O.p3n_iter1_reference()
    print("\n== no-op regression vs the committed p3n iter1 gate "
          "(tolerance 0.25 lc/window ~ 2 %) ==")
    ok = True
    for H in O.HORIZONS:
        t = O.lc_by_type(rows, H)
        want_pred = (N.load_json(
            O.PART3N / "results" / "gate_train_internal_iter1.json")
            ["gate_v2"][f"H{H}"]["lc_pred"])
        diff = abs(t["pred_lc"] - want_pred)
        flag = "OK" if diff <= 0.25 else "MISMATCH"
        ok &= diff <= 0.25
        print(f"  H={H}: pooled pred_lc {t['pred_lc']:.3f} vs committed "
              f"{want_pred} (diff {diff:.3f}) {flag}")
    print("REGRESSION " + ("OK -- defaults are no-ops within "
                           "cross-platform float jitter" if ok else
                           "FAILED -- investigate the flag defaults"))
    print("\nDONE part 3o base rows.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
