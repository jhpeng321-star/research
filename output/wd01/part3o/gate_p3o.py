"""WD-01 part 3o TASK C': TRAIN-internal gate v2 + the BY-TYPE LC gate
table, per iteration (resumable).

  --iter 0   scores the base files (the p3n iter1 members re-rolled
             with the typed builder -- no re-roll);
  --iter k>0 rolls the fit_p3o_iter{k}.json members onto the 4
             calibration roots x 7 t0 first, one file per member
             (results/raw/raw_train_internal_iter{k}_m{mi}.jsonl;
             complete files are skipped -- --fresh re-rolls).

Per iteration output: results/gate_train_internal_iter{k}.json with
  - the full gate v2 (a)-(d) + old gate at H {30, 60} on the 2-fold
    protocol (N.gate_verdicts, inherited verbatim from p3n);
  - the BY-TYPE LC table (pooled + per fold: merge, disc, total
    deviations) and the p3o goal package (LC by type AND total within
    +-20 % at BOTH horizons; (a); (d); delay-phys 2-fold CRPS no worse
    than the committed p3n iter1 gate -- read at runtime);
  - the sharpness/climatology diagnostics (TTC coverage + climatology
    CRPS reported, NOT optimised -- owner decision pending).

ZERO simulation; validation roots never touched; holdout blocked.
"""
from __future__ import annotations

import argparse
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

T_START = time.time()
ROWS_PER_MEMBER = 2 * len(O.GATE_T0S)


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    k = args.iter
    sys.stdout = D.Tee(HERE / "results" / f"stdout_gate_iter{k}.txt")
    print("priority below-normal:", D.below_normal_priority())
    print(f"=== WD-01 part 3o: TRAIN-internal gate v2 + LC by type, "
          f"iteration {k} ===")
    print("ZERO simulation; holdout blocked; NO validation look (the "
          "validation roots are never loaded here).")
    print(f"device={P3.device()} torch_threads={torch.get_num_threads()}")
    sp = O.splits()
    cal = set(sp["calibration"])
    fit_roots = tuple(sp["fit"])
    assert not (cal & set(fit_roots)) and not (cal & set(sp["val"]))

    dev = P3.device()
    if k == 0:
        member_files = [O.RAW_DIR / f"raw_train_internal_iter0_m{mi}.jsonl"
                        for mi in range(5)]
        missing = [p.name for p in member_files
                   if not O.member_file_complete(p, ROWS_PER_MEMBER)]
        if missing:
            raise SystemExit(f"run base_p3o.py first; missing/short "
                             f"member files: {missing}")
    else:
        fit_path = RES_FIT(k)
        if not fit_path.exists():
            raise SystemExit(f"run fit_p3o.py --iter {k} first")
        members = N.load_json(fit_path)["members"]
        member_files = []
        for mi, mem in enumerate(members):
            path = O.RAW_DIR / f"raw_train_internal_iter{k}_m{mi}.jsonl"
            if args.fresh or not O.member_file_complete(
                    path, ROWS_PER_MEMBER):
                n = 0
                with open(path, "w", encoding="utf-8") as f:
                    for root in sorted(cal):
                        for t0 in O.GATE_T0S:
                            rows = O.roll_member_rows(
                                root, float(t0), dict(mem), mi, "L6",
                                dev,
                                f"part3o iter{k} gate roll: " +
                                O.cfg_note(mem))
                            for row in rows:
                                f.write(json.dumps(
                                    row, default=str) + "\n")
                                n += 1
                            f.flush()
                log(f"iter{k} member {mi}: wrote {n} rows -> "
                    f"{path.name}")
            else:
                log(f"iter{k} member {mi}: {path.name} complete -- "
                    f"skipped")
            member_files.append(path)

    rows_l6 = []
    for p in member_files:
        rows_l6 += [r for r in O.read_rows(p) if r["root"] in cal]
    l0_rows = [r for r in O.read_rows(O.RAW_DIR / "raw_base_L0.jsonl")
               if r["root"] in cal]
    fit_truth_rows = O.read_rows(O.RAW_DIR / "raw_fit_truth_m0.jsonl")
    rows_all = rows_l6 + l0_rows + fit_truth_rows
    n_win = len({(r["root"], int(r["t0"])) for r in rows_l6})
    print(f"TRAIN-internal gate split: fold A {O.FOLD_A}, fold B "
          f"{O.FOLD_B}; t0 {O.GATE_T0S}; scored windows {n_win}; "
          f"members {len(member_files)}; 2-fold spread (scored fold "
          f"never fits its own factors)")

    # ---- gate v2 + old gate (2-fold), inherited verbatim
    rows_cal = rows_l6 + l0_rows
    verdicts = N.gate_verdicts(rows_cal, horizons=O.HORIZONS)
    for H in O.HORIZONS:
        g = verdicts[H]
        verdict = ("PASS" if g["v2_PASS"]
                   else "FAIL: " + ",".join(g["v2_failing"]))
        print(f"\n== TRAIN-internal GATE v2 at H={H} "
              f"({'PRIMARY' if H == 30 else 'secondary'}) ==")
        print(f"  L6:  (a) coll/1000={g['collisions_per_1000']} "
              f"{'PASS' if g['a_collisions_pass'] else 'FAIL'} | "
              f"(b) lc {g['lc_pred']}/{g['lc_true']} "
              f"({100 * g['lc_deviation']:.1f}%) "
              f"{'PASS' if g['b_lc_pass'] else 'FAIL'} | "
              f"(c) ttc3 {g['ttc3_cov_frac']} delay-phys "
              f"{g['delay_phys_cov_frac']} crps {g['crps_phys']} vs "
              f"L0 mae {g['l0_delay_phys_mae']} -> "
              f"{'PASS' if g['c_coverage_crps_pass'] else 'FAIL'} | "
              f"(d) speed {g['speed_rmse']} vs 1.05xL0="
              f"{g['d_per_horizon_threshold']} "
              f"({'PASS' if g['d_per_horizon_pass'] else 'FAIL'}, "
              f"binding) ==> v2 {verdict}")

    # ---- the p3o BY-TYPE LC gate table + goal package
    bt = O.by_type_gate_block(rows_l6)
    ref = O.p3n_iter1_reference()
    goal = O.goal_package(rows_l6, verdicts, ref)
    print("\n== LC BY TYPE (per window; calibration roots, members "
          "pooled) ==")
    for H in O.HORIZONS:
        print(f"  H={H}:")
        for name in ("pooled", "fold_A", "fold_B"):
            t = bt[H][name]
            print(f"    {name:<8} predM {t['pred_lc_merge']:5.2f} "
                  f"trueM {t['true_lc_merge']:5.2f} "
                  f"({100*t['dev_merge']:+6.1f}%) | predD "
                  f"{t['pred_lc_disc']:5.2f} trueD "
                  f"{t['true_lc_disc']:5.2f} "
                  f"({100*t['dev_disc']:+6.1f}%) | total "
                  f"{t['pred_lc']:5.2f}/{t['true_lc']:5.2f} "
                  f"({100*t['dev_total']:+6.1f}%)")
    print("\n== p3o GOAL PACKAGE (LC by type+total +-20% at BOTH H; "
          "(a); (d); delay CRPS <= p3n iter1) ==")
    for H in O.HORIZONS:
        gp = goal[H]
        c = gp["checks"]
        print(f"  H={H}: type-dev max pooled "
              f"{gp['max_type_dev']['pooled']:.3f} foldA "
              f"{gp['max_type_dev']['fold_A']:.3f} foldB "
              f"{gp['max_type_dev']['fold_B']:.3f} | crps "
              f"{gp['crps_phys']} <= ref {gp['crps_p3n_iter1']} | "
              f"speed {gp['speed_rmse']} <= {gp['d_threshold']} | "
              f"{', '.join(f'{a}:{b}' for a, b in c.items())}")
    all_pass = all(goal[H]["checks"]["PASS"] for H in O.HORIZONS)
    print(f"  ==> GOAL PACKAGE {'PASS' if all_pass else 'FAIL'}")

    # ---- sharpness + climatology diagnostics (NOT optimised)
    clima = N.climatology_from_rows(rows_all, roots=set(fit_roots))
    cl = N.climatology_table(rows_l6, clima)
    print("\n== SHARPNESS + CLIMATOLOGY (diagnostics; TTC criterion "
          "under owner decision) ==")
    print("   output      H   cov    med_band  truth_iqr | crps_L6  "
          "clima_crps(empirical) crps(TRAIN-mean) n_train")
    summary_c = {}
    gate_rows_cached = N.gate_rows(rows_l6, ("A", "B"))
    for (name, H), c in sorted(cl.items()):
        gN = gate_rows_cached[name][H]
        beats = gN["crps"] < c["clima_crps_empirical"]
        summary_c[f"{name}@H{H}"] = dict(
            cov=gN["cov"], median_band=gN["median_band_width"],
            truth_iqr=gN["truth_iqr_width"], crps_l6=gN["crps"],
            clima_crps_empirical=c["clima_crps_empirical"],
            clima_crps_meanpoint=c["clima_crps_meanpoint"],
            beats_climatology=bool(beats),
            informative=bool(gN["cov"] >= 0.8 and beats))
        print(f"   {name:<11} {H}  {gN['cov']:.4f}  "
              f"{gN['median_band_width']:8.2f}  "
              f"{gN['truth_iqr_width']:9.2f} | {gN['crps']:7.2f}  "
              f"{c['clima_crps_empirical']:8.2f} "
              f"{c['clima_crps_meanpoint']:8.2f} {c['clima_samples']:7d}"
              f"  {'beats' if beats else 'DOES NOT beat'} climatology")

    (HERE / "results" /
     f"gate_train_internal_iter{k}.json").write_text(
        json.dumps(dict(
            iteration=k, part="3o",
            split=dict(fold_A=list(O.FOLD_A), fold_B=list(O.FOLD_B),
                       t0s=list(O.GATE_T0S), windows=n_win,
                       protocol="2-fold: scored fold's factors fitted "
                                "on the other fold only"),
            gate_v2={f"H{H}": verdicts[H] for H in O.HORIZONS},
            lc_by_type={f"H{H}": bt[H] for H in O.HORIZONS},
            goal_package={f"H{H}": goal[H] for H in O.HORIZONS},
            goal_package_pass=all_pass,
            p3n_iter1_reference=ref,
            sharpness_climatology=summary_c,
            climatology_source="fit-root truth sides (raw_fit_truth_m0,"
                               " disjoint from the scored folds)",
            holdout_accessed=False, simulation=False,
            validation_used=False, validation_loaded=False),
            indent=1, default=str),
        encoding="utf-8")
    log(f"wrote results/gate_train_internal_iter{k}.json")
    print(f"\nDONE part 3o gate iteration {k}.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


def RES_FIT(k):
    return HERE / "results" / f"fit_p3o_iter{k}.json"


if __name__ == "__main__":
    main()
