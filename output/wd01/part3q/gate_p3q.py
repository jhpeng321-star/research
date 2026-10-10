"""WD-01 part 3q TASK C: the k-fold TRAIN gate -- gate v2 (a)-(d) at
H=30 and H=60 PER FOLD + pooled, with per-type LC, merge-point/merge-
timing diagnostics, the delay/TTC coverage bias-vs-width breakdown, and
CRPS vs the out-of-fold TRAIN climatology (resumable per member).

  --source base    scores the diag rolls (the frozen p3o iter2
                   candidate on the 17 TRAIN roots; no re-roll);
  --source iter{k} rolls the fit_p3q_iter{k}.json members onto the 17
                   roots x 7 t0 first (one file per member,
                   raw_p3q_iter{k}_m{mi}.jsonl; complete files are
                   skipped -- --fresh re-rolls).

Protocol (the brief): k=5 folds BY ROOT; when scoring fold f, the
spread factors per (output, horizon) are fitted on the OTHER folds'
rows only (cross-fitting); the pooled numbers accumulate windows across
folds (each scored out-of-fold).  Fold f0 additionally never entered
the parameter fit (fully out-of-sample); folds f1..f4 are
param-in-sample for the count/speed criteria and are marked as such in
the JSON.  Climatology: each window's empirical TRAIN truth
distribution is built from the OTHER folds (out-of-fold by
construction).

ZERO simulation; validation roots never loaded; holdout blocked.
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
import p3q_lib as Q  # noqa: E402
import decompose_p3k as D  # noqa: E402

T_START = time.time()


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def merge_diag_block(rows_l6, mevents, roots):
    out = {"merge_pos_timing": {}, "lc_by_root": {}, "merge_stats":
           Q.pool_merge_stats([rec["stats"] for rec in mevents])}
    pred_ev = [e for rec in mevents for e in rec["events"]]
    truth_ev = []
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="base | iter<K>")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if args.source == "base":
        tag = "base"
        fit_path = None
    elif args.source.startswith("iter"):
        k = int(args.source[4:])
        tag = f"iter{k}"
        fit_path = Q.RES / f"fit_p3q_iter{k}.json"
        if not fit_path.exists():
            raise SystemExit(f"run fit_p3q.py --iter {k} first")
    else:
        raise SystemExit(f"unknown source {args.source}")
    sys.stdout = D.Tee(Q.RES / f"stdout_gate_{tag}.txt")
    print("priority below-normal:", D.below_normal_priority())
    print(f"=== WD-01 part 3q TASK C: k-fold TRAIN gate v2, source "
          f"{args.source} ===")
    print("ZERO simulation; holdout blocked; NO validation look (the "
          "validation roots are never loaded here).")
    print(f"device={Q.P3.device()} torch_threads={torch.get_num_threads()}")
    roots = Q.train17()
    assert not (set(roots) & set(Q.O.splits()["val"]))
    assert not (set(roots) & set(Q.O.splits()["holdout"]))
    dev = Q.P3.device()

    if tag == "base":
        member_files = [Q.RAW_DIR / f"raw_p3q_base_m{mi}.jsonl"
                        for mi in range(5)]
        ev_files = [Q.RAW_DIR / f"raw_p3q_base_mevents_m{mi}.json"
                    for mi in range(5)]
        for p in member_files + ev_files:
            if not p.exists():
                raise SystemExit(f"run diag_p3q.py first; missing {p.name}")
    else:
        members = Q.N.load_json(fit_path)["members"]
        member_files, ev_files = [], []
        for mi, mem in enumerate(members):
            path = Q.RAW_DIR / f"raw_p3q_{tag}_m{mi}.jsonl"
            ev_path = Q.RAW_DIR / f"raw_p3q_{tag}_mevents_m{mi}.json"
            expect = 2 * len(Q.GATE_T0S) * len(roots)
            if args.fresh or not Q.member_file_complete(path, expect) \
                    or not ev_path.exists():
                n = 0
                evs = []
                with open(path, "w", encoding="utf-8") as f:
                    for root in roots:
                        for t0 in Q.GATE_T0S:
                            r, rows = Q.roll_member_rows_q(
                                root, float(t0), dict(mem), mi, "L6",
                                dev, f"part3q {tag} gate roll: " +
                                Q.cfg_note(mem))
                            evs.append(Q.events_and_stats(
                                r, root, t0, mi))
                            for row in rows:
                                f.write(json.dumps(
                                    row, default=str) + "\n")
                                n += 1
                            f.flush()
                ev_path.write_text(json.dumps(evs, default=str),
                                   encoding="utf-8")
                log(f"{tag} member {mi}: wrote {n} rows -> {path.name}")
            else:
                log(f"{tag} member {mi}: {path.name} complete -- "
                    f"skipped")
            member_files.append(path)
            ev_files.append(ev_path)

    rows_l6 = []
    for p in member_files:
        rows_l6 += [r for r in Q.read_rows(p) if r["root"] in set(roots)]
    mevents = []
    for p in ev_files:
        mevents += json.loads(p.read_text())
    l0_rows = [r for r in Q.read_rows(Q.RAW_DIR / "raw_p3q_L0.jsonl")
               if r["root"] in set(roots)]
    rows_all = rows_l6 + l0_rows
    n_win = len({(r["root"], int(r["t0"])) for r in rows_l6})
    print(f"k-fold gate split: {Q.FOLDS}; t0 {Q.GATE_T0S}; scored "
          f"windows {n_win}; members {len(member_files)}; k=5 "
          f"cross-fitted spread (scored fold never fits its own "
          f"factors); fold {Q.DEV_FOLD} never entered the fit")

    gate = Q.gate_verdicts_kf(rows_all)
    pooled = gate["pooled"]
    for H in Q.HORIZONS:
        g = pooled[H]
        verdict = ("PASS" if g["v2_PASS"]
                   else "FAIL: " + ",".join(g["v2_failing"]))
        print(f"\n== k-fold TRAIN GATE v2 at H={H} "
              f"({'PRIMARY' if H == 30 else 'secondary'}, pooled, "
              f"cross-fitted) ==")
        print(f"  (a) coll/1000={g['collisions_per_1000']} "
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
        bw = g["bias_width"]["delay_phys"]
        print(f"      delay-phys misses: below {bw['below_frac']} / in "
              f"{bw['in_frac']} / above {bw['above_frac']} | bias mean "
              f"{bw['bias_mean']} (truth-centre) | band med "
              f"{bw['median_band_width']} vs truth IQR "
              f"{bw['truth_iqr_width']}")
        bw = g["bias_width"]["ttc3"]
        print(f"      ttc3        misses: below {bw['below_frac']} / in "
              f"{bw['in_frac']} / above {bw['above_frac']} | bias mean "
              f"{bw['bias_mean']} | band med {bw['median_band_width']} "
              f"vs truth IQR {bw['truth_iqr_width']}")
    print("\n== per-fold gate v2 (fold f0 = out-of-sample wrt params; "
          "f1..f4 = param-in-sample) ==")
    for fname in Q.FOLD_NAMES:
        for H in Q.HORIZONS:
            gf = gate["per_fold"][fname][H]
            verdict = ("PASS" if gf["v2_PASS"]
                       else "FAIL: " + ",".join(gf["v2_failing"]))
            print(f"  {fname} H={H}: coll {gf['collisions_per_1000']} "
                  f"| lc {gf['lc_pred']}/{gf['lc_true']} "
                  f"({100*gf['lc_deviation']:+.1f}%) | ttc3 "
                  f"{gf['ttc3_cov_frac']} dphys "
                  f"{gf['delay_phys_cov_frac']} crps "
                  f"{gf['crps_phys']} | speed {gf['speed_rmse']} vs "
                  f"{gf['d_per_horizon_threshold']} | s_out dp/t3 "
                  f"{gf['fold_s_out']['delay_phys']}/"
                  f"{gf['fold_s_out']['ttc3']} ==> {verdict}")

    # ---- LC by type: pooled + per fold (+ per-root caveat table)
    for H in Q.HORIZONS:
        print(f"\n== LC BY TYPE (per window; 17 TRAIN roots, members "
              f"pooled) H={H} ==")
        for name in ("pooled",) + Q.FOLD_NAMES:
            t = (Q.by_type_block(rows_l6, H)[name])
            print(f"    {name:<8} predM {t['pred_lc_merge']:5.2f} trueM "
                  f"{t['true_lc_merge']:5.2f} "
                  f"({100*t['dev_merge']:+6.1f}%) | predD "
                  f"{t['pred_lc_disc']:5.2f} trueD "
                  f"{t['true_lc_disc']:5.2f} ({100*t['dev_disc']:+6.1f}"
                  f"%) | total {t['pred_lc']:5.2f}/{t['true_lc']:5.2f} "
                  f"({100*t['dev_total']:+6.1f}%)")

    # ---- merge position/timing + rejection causes
    md = merge_diag_block(rows_l6, mevents, roots)
    print("\n== merge position/timing (per merge event; dist = m to the "
          "accel lane end) ==")
    for H in Q.HORIZONS:
        pt = md["merge_pos_timing"][f"H{H}"]
        for side in ("pred", "truth"):
            s_ = pt[side]
            print(f"  H={H} {side:<5} n={s_['n']:4d} rel_s mean "
                  f"{s_['rel_mean']} p10/50/90 {s_['rel_p10']}/"
                  f"{s_['rel_p50']}/{s_['rel_p90']} | dist mean "
                  f"{s_['dist_mean']} p10/50/90 {s_['dist_p10']}/"
                  f"{s_['dist_p50']}/{s_['dist_p90']}")
    st = md["merge_stats"]
    print(f"\n== M2 rejection causes (pooled) ==\n  checks "
          f"{st['checks']} accepted {st['accepted']} rej_lead "
          f"{st['rej_lead']} rej_foll {st['rej_foll']} rej_both "
          f"{st['rej_both']} rej_prob {st['rej_prob']} "
          f"rej_prob_early {st['rej_prob_early']} cd_set {st['cd_set']} "
          f"cd_blocked {st['cd_blocked']}")

    # ---- goal package
    goal = Q.goal_package_q(pooled, rows_l6)
    print("\n== p3q GOAL PACKAGE (LC by type+total +-20% pooled; "
          "formal (c); (a); (d)) ==")
    for H in Q.HORIZONS:
        c = goal[H]["checks"]
        print(f"  H={H}: type-dev pooled "
              f"{goal[H]['max_type_dev']['pooled']:.3f} (folds "
              f"{ {f: round(goal[H]['max_type_dev'][f], 3) for f in Q.FOLD_NAMES} }) "
              f"| crps {goal[H]['crps_phys']} | speed "
              f"{goal[H]['speed_rmse']} <= {goal[H]['d_threshold']} | "
              f"{', '.join(f'{a}:{b}' for a, b in c.items())}")
    all_pass = all(goal[H]["checks"]["PASS"] for H in Q.HORIZONS)
    print(f"  ==> GOAL PACKAGE {'PASS' if all_pass else 'FAIL'}")

    # ---- climatology (out-of-fold) + spread table
    cl = Q.climatology_kf(rows_l6)
    print("\n== SHARPNESS + OUT-OF-FOLD CLIMATOLOGY ==")
    print("   output      H   cov    below/in/above      med_band  "
          "truth_iqr | crps_L6  clima_crps(emp) crps(TRAIN-mean)")
    for name in ("delay_phys", "ttc3"):
        for H in Q.HORIZONS:
            gN = gate["cov"][name][H]
            c = cl[name][H]
            beats = gN["crps"] < c["clima_crps_empirical"]
            print(f"   {name:<11} {H}  {gN['cov']:.4f}  "
                  f"{gN['below_frac']:.3f}/{gN['in_frac']:.3f}/"
                  f"{gN['above_frac']:.3f}  "
                  f"{gN['median_band_width']:9.2f}  "
                  f"{gN['truth_iqr_width']:9.2f} | "
                  f"{gN['crps']:7.2f}  "
                  f"{c['clima_crps_empirical']:8.2f} "
                  f"{c['clima_crps_meanpoint']:8.2f}  "
                  f"{'beats' if beats else 'DOES NOT beat'}")

    (Q.RES / f"gate_kfold_{tag}.json").write_text(
        json.dumps(dict(
            part="3q", source=args.source,
            split=dict(folds={k_: list(v) for k_, v in
                              Q.FOLDS.items()},
                       t0s=list(Q.GATE_T0S), windows=n_win,
                       train_roots=list(roots),
                       protocol="k=5 by root; fold spread factors "
                                "fitted on the other folds only "
                                "(cross-fitting); fold f0 additionally "
                                "out-of-sample wrt the parameter fit",
                       param_in_sample_folds=list(Q.FIT_FOLDS),
                       param_out_of_sample_folds=[Q.DEV_FOLD]),
            gate_v2={f"H{H}": pooled[H] for H in Q.HORIZONS},
            per_fold={f: {f"H{H}": gate["per_fold"][f][H]
                          for H in Q.HORIZONS}
                      for f in Q.FOLD_NAMES},
            lc_by_type={f"H{H}": Q.by_type_block(rows_l6, H)
                        for H in Q.HORIZONS},
            lc_by_root=md["lc_by_root"],
            merge_pos_timing=md["merge_pos_timing"],
            merge_stats=md["merge_stats"],
            goal_package={f"H{H}": goal[H] for H in Q.HORIZONS},
            goal_package_pass=all_pass,
            climatology=cl,
            holdout_accessed=False, simulation=False,
            validation_used=False, validation_loaded=False),
            indent=1, default=str), encoding="utf-8")
    log(f"wrote results/gate_kfold_{tag}.json")
    print(f"\nDONE part 3q gate ({tag}).")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
