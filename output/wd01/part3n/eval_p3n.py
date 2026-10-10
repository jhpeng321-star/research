"""WD-01 part 3n TASK D: the SINGLE VALIDATION LOOK (VAL look #8 per
the WD-01-DEV-2 ledger -- seven looks were spent before this part).

Runs ONLY after a TRAIN-internal gate v2 pass and the FREEZE:
refuses to run unless BOTH results/calibration_p3n.json and
results/fit_p3n.json match their FROZEN sha256 sidecars (the freeze
never precedes the pass; the look never precedes the freeze).

Rolls the five part 3n members (fit_p3n.json) over the 4 validation
roots x t0 every 30 s (52 windows) x H in {30, 60}; raw rows are
written to results/raw/raw_metrics_p3n.jsonl BEFORE any aggregation.
L0/L4/L5 rows are REUSED verbatim from the part 3l raw rows
(results/raw/raw_p3l_metrics.jsonl, committed) -- no re-roll; L6c is
derived at aggregation time with the FROZEN factors.

ONE look: if the gate fails here, the part stops -- diagnosis
continues on TRAIN only, with no second look in this task.

ZERO simulation; holdout blocked in root_arrays (never passed).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("WD_DEVICE", "cpu")    # single process, shared box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p3n_lib as N  # noqa: E402
import decompose_p3k as D  # noqa: E402
import p3lib as P3  # noqa: E402

HORIZONS = N.HORIZONS
VAL_T0S = tuple(range(120, 481, 30))
RAW_DIR = HERE / "results" / "raw"
RES = HERE / "results"
T_START = time.time()


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def check_frozen_both():
    """FREEZE GATE: fit + calibration must both match their sidecars."""
    f1 = N.check_frozen(RES / "fit_p3n.json",
                        RES / "fit_p3n.json.FROZEN.sha256")
    f2 = N.check_frozen(RES / "calibration_p3n.json",
                        RES / "calibration_p3n.json.FROZEN.sha256")
    return f1, f2


def member_configs():
    fit = N.load_json(RES / "fit_p3n.json")
    out = []
    for mi, mem in enumerate(fit["members"]):
        out.append((int(mem["seed"]), float(mem["lc_scale"]),
                    dict(mem["merge_params"]),
                    float(mem.get("dawdle_scale", 1.0))))
    return out


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    sys.stdout = D.Tee(RES / "stdout_eval.txt")

    frozen_fit, frozen_cal = check_frozen_both()
    print("priority below-normal:", D.below_normal_priority())
    print("=== WD-01 part 3n TASK D: the SINGLE VALIDATION LOOK "
          "(VAL look #8) ===")
    print("ZERO simulation; holdout blocked in root_arrays (never "
          "passed).")
    dev = P3.device()
    print(f"device={dev} torch_threads={torch.get_num_threads()}")
    print(f"FREEZE GATE ok: fit_p3n.json sha256 "
          f"{frozen_fit['sha256'][:16]}... frozen_utc="
          f"{frozen_fit['frozen_utc']}")
    print(f"FREEZE GATE ok: calibration_p3n.json sha256 "
          f"{frozen_cal['sha256'][:16]}... frozen_utc="
          f"{frozen_cal['frozen_utc']}")
    sp = N.L3.splits_3l()
    val_roots = sp["val"]
    print(f"validation roots {val_roots}; t0 every 30 s {VAL_T0S} "
          f"({len(val_roots) * len(VAL_T0S)} windows); H {HORIZONS}; "
          f"L0/L4/L5 rows reused from part 3l (no re-roll)")
    print("THIS IS VALIDATION LOOK #8 (WD-01-DEV-2 ledger: 7 previous "
          "looks); one look only in part 3n.")
    configs = member_configs()
    print("L6 (part 3n) member configs (from the FROZEN fit_p3n.json):")
    for mi, (seed, lc, mp, dw) in enumerate(configs):
        print(f"  L6 member {mi}: seed={seed} lc_scale={lc} "
              f"merge_params={mp} dawdle_scale={dw}")

    raw_path = RAW_DIR / "raw_metrics_p3n.jsonl"
    n_rows = 0
    with open(raw_path, "w", encoding="utf-8") as raw_f:
        for mi, (seed, lc, mp, dw) in enumerate(configs):
            log(f"rolling L6 member {mi} (seed {seed}, dawdle {dw}) "
                f"...")
            for root in val_roots:
                for t0 in VAL_T0S:
                    r = N.P3_roll(root, float(t0), mp, lc, dev, dw,
                                  seed=seed)
                    note = ("part3n VAL look #8: engine unchanged; "
                            f"end_force_s {mp.get('end_force_s', 0.0)}, "
                            f"blend {mp.get('v_merge_blend', 0.0)})")
                    for row in N.window_raw_rows(r, "L6", mi, root, t0,
                                                 dw, note,
                                                 horizons=HORIZONS):
                        raw_f.write(json.dumps(row, default=str) +
                                    "\n")
                        raw_f.flush()
                        n_rows += 1
            log(f"L6 member {mi}: done ({n_rows} rows total)")
    log(f"raw rows written to {raw_path} (total this run: {n_rows})")
    print("\nDONE part 3n TASK D (the one validation look; next: "
          "aggregate_p3n.py -- no second look regardless of verdict).")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
