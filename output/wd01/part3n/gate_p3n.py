"""WD-01 part 3n TASK C: TRAIN-internal gate v2 -- the full gate v2
(a)-(d) + old gate on HELD-OUT TRAIN roots, with 2-fold spread
calibration.

Split (declared): the 4 calibration roots, 2 folds
  fold A = cv20_7002m2000, cv20_7003m2000
  fold B = cv20_7004m2000, cv20_7007m2700
7 t0 each (120..480 step 60) -> 28 windows, M=5, H in {30, 60}.
When scoring fold X, the spread factors s_out per (output, horizon)
are fitted on fold Y's rows ONLY (the brief's "spread factors fitted
on a DIFFERENT subset than the one scored"); coverage/CRPS are pooled
over the 28 windows.  The model itself is never fitted here: the fit
roots are disjoint and the gate scores whatever members the given fit
JSON declares.

  --iter 0   scores the part 3m L6 members (baseline) from the TASK A
             raw rows (no re-roll);
  --iter k>0 rolls the part 3n members (results/fit_p3n.json) onto the
             28 windows first (--fresh forces a re-roll), writing
             results/raw/raw_train_internal_iter{k}.jsonl.

Outputs per iteration: results/gate_train_internal_iter{k}.json with
the gate table, the sharpness table (median widened 10-90 % band width
vs the across-window 10-90 % range of the truth) and the climatology
CRPS table (TRAIN fit-window truth distribution + TRAIN-mean point
forecast, built from the FIT-root truth sides only -- disjoint from
the scored folds).

ZERO simulation; holdout blocked; validation never touched here.
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
import p3n_lib as N  # noqa: E402
import decompose_p3k as D  # noqa: E402
import p3lib as P3  # noqa: E402

RAW_DIR = HERE / "results" / "raw"
T_START = time.time()


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def member_configs(fit_path):
    fit = N.load_json(fit_path)
    out = []
    for mi, mem in enumerate(fit["members"]):
        out.append((mi, int(mem["seed"]), float(mem["lc_scale"]),
                    dict(mem["merge_params"]),
                    float(mem.get("dawdle_scale", 1.0))))
    return out


def roll_iter(configs, out_path):
    """Roll the given members over the 28 TRAIN-internal windows and
    write raw rows (no L0 here -- the TASK A file has L0 already)."""
    dev = P3.device()
    n = 0
    with open(out_path, "w", encoding="utf-8") as rf:
        for mi, seed, lc, mp, dw in configs:
            log(f"rolling L6(p3n) member {mi} (seed {seed}, dawdle "
                f"{dw}) over the 4 calibration roots ...")
            for root in N.L3.splits_3l()["calibration"]:
                for t0 in N.GATE_T0S:
                    r = N.P3_roll(root, float(t0), mp, lc, dev, dw,
                                  seed=seed)
                    note = ("part3n TRAIN-internal gate roll (engine "
                            "unchanged; end_force_s "
                            f"{mp.get('end_force_s', 0.0)}, blend "
                            f"{mp.get('v_merge_blend', 0.0)})")
                    for row in N.window_raw_rows(r, "L6", mi, root, t0,
                                                 dw, note):
                        rf.write(json.dumps(row, default=str) + "\n")
                        n += 1
                    rf.flush()
            log(f"  member {mi}: done ({n} rows)")
    return n


def read_rows(path):
    return [json.loads(ln) for ln in
            Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--fresh", action="store_true",
                    help="re-roll even if the iteration raw file exists")
    args = ap.parse_args()
    k = args.iter
    sys.stdout = D.Tee(HERE / "results" /
                       f"stdout_gate_iter{k}.txt")

    print("priority below-normal:", D.below_normal_priority())
    print(f"=== WD-01 part 3n TASK C: TRAIN-internal gate v2, "
          f"iteration {k} ===")
    print("ZERO simulation; holdout blocked in root_arrays (never "
          "passed); validation NOT touched here.")
    print(f"device={P3.device()} torch_threads={torch.get_num_threads()}")

    sp = N.L3.splits_3l()
    cal = set(sp["calibration"])
    fit_roots = tuple(sp["fit"])
    iter_path = RAW_DIR / f"raw_train_internal_iter{k}.jsonl"
    if k == 0:
        base_path = RAW_DIR / "raw_train_internal_p3n.jsonl"
        if not base_path.exists():
            raise SystemExit("run diag_p3n.py first (TASK A raw rows)")
        if args.fresh:
            raise SystemExit("--iter 0 reuses the TASK A rows (the p3m "
                             "members); nothing to re-roll")
        rows_all = read_rows(base_path)
        n_l6 = len([r for r in rows_all if r["variant"] == "L6"])
        print(f"iter 0: scoring the p3m L6 members from TASK A rows "
              f"({base_path.name}; {n_l6} L6 rows total, calibration-"
              f"root subset scored)")
    else:
        fit_path = HERE / "results" / "fit_p3n.json"
        if not fit_path.exists():
            raise SystemExit("run fit_p3n.py first (TASK B)")
        configs = member_configs(fit_path)
        if args.fresh or not iter_path.exists():
            n = roll_iter(configs, iter_path)
            log(f"wrote {n} rows -> {iter_path.name}")
        else:
            log(f"reusing {iter_path.name} (exists; --fresh to re-roll)")
        rows_all = read_rows(iter_path)
        # iteration rolls carry no L0: splice the TASK A calibration
        # L0 rows in (the (d) reference), plus the TASK A FIT-root rows
        # (their member-independent TRUTH sides feed the climatology)
        base_rows = read_rows(RAW_DIR / "raw_train_internal_p3n.jsonl")
        l0_rows = [r for r in base_rows
                   if r["variant"] == "L0" and r["root"] in cal]
        fit_rows = [r for r in base_rows if r["root"] not in cal]
        rows_all = rows_all + l0_rows + fit_rows
        print(f"iter {k}: scoring the part 3n members "
              f"({iter_path.name}) + TASK A L0 rows ({len(l0_rows)}) + "
              f"TASK A fit-root truth rows ({len(fit_rows)}, climatology "
              f"only)")

    rows_cal = [r for r in rows_all if r["root"] in cal]
    for r in rows_cal:
        assert r["root"] in cal, r["root"]
    n_win = len({(r["root"], r["t0"]) for r in rows_cal
                 if r["variant"] == "L6"})
    print(f"TRAIN-internal gate split: fold A {N.FOLD_A}, fold B "
          f"{N.FOLD_B}; t0 {N.GATE_T0S}; scored windows {n_win}; "
          f"2-fold spread (scored fold never fits its own factors)")

    # ---- the gate (a)-(d) + old gate, 2-fold spread
    verdicts = N.gate_verdicts(rows_cal, horizons=N.HORIZONS)
    for H in N.HORIZONS:
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
        pf = g["per_fold"]
        print(f"       2-fold detail: ttc3 A(s={pf['ttc3']['A']['s_out']}"
              f", cov {pf['ttc3']['A']['cov']}) B(s="
              f"{pf['ttc3']['B']['s_out']}, cov {pf['ttc3']['B']['cov']})"
              f" | delay-phys A(s={pf['delay_phys']['A']['s_out']}, cov "
              f"{pf['delay_phys']['A']['cov']}) B(s="
              f"{pf['delay_phys']['B']['s_out']}, cov "
              f"{pf['delay_phys']['B']['cov']})")
        print(f"       minmax (old-gate item, widened): "
              f"{g['old']['delay_phys_minmax_frac']}; OLD gate -> "
              f"{'PASS' if g['old']['old_pass'] else 'FAIL'}")

    # ---- sharpness + climatology (brief: required per output/horizon)
    l6_cal = [r for r in rows_cal if r["variant"] == "L6"]
    # climatology: FIT-root truth sides only (disjoint from the scored
    # calibration folds; the truth side is member-independent)
    clima = N.climatology_from_rows(rows_all, roots=set(fit_roots))
    cl = N.climatology_table(l6_cal, clima)
    print("\n== SHARPNESS + CLIMATOLOGY (TRAIN-internal, 2-fold "
          "widened) ==")
    print("   output      H   cov    med_band  truth_iqr | crps_L6  "
          "clima_crps(empirical) crps(TRAIN-mean) n_train")
    summary_c = {}
    for (name, H), c in sorted(cl.items()):
        gN = N.gate_rows(l6_cal, ("A", "B"))[name][H]
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
            iteration=k, split=dict(fold_A=list(N.FOLD_A),
                                    fold_B=list(N.FOLD_B),
                                    t0s=list(N.GATE_T0S),
                                    windows=n_win,
                                    protocol="2-fold: scored fold's "
                                             "factors fitted on the "
                                             "other fold only"),
            gate_v2={f"H{H}": verdicts[H] for H in N.HORIZONS},
            sharpness_climatology=summary_c,
            climatology_source="FIT-root truth sides (disjoint from "
                               "the scored folds)",
            holdout_accessed=False, simulation=False,
            validation_used=False), indent=1, default=str),
        encoding="utf-8")
    log(f"wrote results/gate_train_internal_iter{k}.json")
    print(f"\nDONE part 3n TASK C iteration {k}.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
