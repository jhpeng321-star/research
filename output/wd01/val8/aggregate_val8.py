"""WD-01 VALIDATION LOOK #8 -- aggregation + gate verdicts.

Reads the val8 raw rows SAVED BY eval_val8.py (no re-roll, no second
look) and produces, exactly as registered:
  - gate v2 (a)-(d) per criterion at H = 30 (PRIMARY, AMENDMENT_1) and
    H = 60, with the (d) per-horizon binding form (1.05 x L0@same-H,
    L0 rolled in the same run) and the fixed 5.2185 alongside, and ONE
    overall verdict at H = 30 for the frozen candidate (L6c = L6 + the
    FROZEN calibration_p3o.json spread factors, mean-centred widening);
  - the OLD gate alongside (speed < 4.97, collisions <= 1/1000, LC
    +-20 %, delay min-max coverage >= 50 %);
  - caveat tables: LC BY TYPE (accel merges vs discretionary) pred vs
    true per horizon; TTC<3 and physics-delay CRPS vs CLIMATOLOGY
    (TRAIN per-window mean point forecast + TRAIN empirical
    distribution, built from the 8 TRAIN roots' truth sides ONLY --
    part3o raw_fit_truth_m0 + raw_train_internal_iter2_m0); TTC
    coverage breakdown (below / in / above, truth == 0 ties counted
    inside below/above);
  - the DEV-2 provenance note (the candidate came from 3 TRAIN-internal
    selection iterations on the calibration roots, p3o iter0-2, after
    7 prior validation looks).

The freeze gate is RE-CHECKED here (candidate + calibration + both
gate files against their registered sha256).  ZERO simulation;
holdout never accessed; validation rows only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("WD_DEVICE", "cpu")

HERE = Path(__file__).resolve().parent
P3O = HERE.parent / "part3o"
PART3N = HERE.parent / "part3n"
PART3M = HERE.parent / "part3m"
PART3L = HERE.parent / "part3l"
for _p in (str(P3O), str(PART3N), str(PART3M), str(PART3L)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import p3o_lib as O  # noqa: E402
import p3n_lib as N  # noqa: E402
import decompose_p3k as D  # noqa: E402  (Tee)
import aggregate_p3l as AG  # noqa: E402  (verdict functions, reused)
import aggregate_p3m as AGM  # noqa: E402  (coverage_and_crps_m, reused)

REGISTERED = {
    "candidate_p3o_iter2.json":
        "458890d9c4b0ee6ee03de145be50498d9bd0873505a4e86f2bf8a1e224ebe45d",
    "calibration_p3o.json":
        "c7ffa7540768fa67e75e99881eec6f801201990dea4ace90af93c202f789c22a",
    "VALIDATION_GATE_V2.md":
        "84b8a23df1ff654f8d602e1fd75119517c4bd7cb458dfb2191d2d088452da47f",
    "VALIDATION_GATE_V2_AMENDMENT_1.md":
        "27aeddab51a899dd1e7a7e24f073fbfe4d29efbe6c2cd3db7e992663d39d79d4",
}
RAW_DIR = HERE / "results" / "raw"
STDOUT_DIR = HERE / "results" / "stdout"
OUT = HERE / "results"
VAL_T0S = tuple(range(120, 481, 30))
N_WINDOWS = 52
N_MEMBERS = 5
HORIZONS = (30, 60)


def verify_registered():
    ok = True
    for name, want in REGISTERED.items():
        p = (P3O / "results" / name) if name.endswith(".json") else \
            (HERE.parent / "part3f" / name)
        got = N.sha256_text(p.read_text(encoding="utf-8"))
        ok &= got == want
        print(f"HASH {'OK' if got == want else 'MISMATCH'}: {name} "
              f"{got}")
    if not ok:
        raise SystemExit("HASH GATE FAILED at aggregation -- refusing")
    print("HASH GATE PASSED (candidate, calibration, gate v2, "
          "amendment 1).")


def main():
    sys.stdout = D.Tee(STDOUT_DIR / "stdout_aggregate_val8.txt")
    print("=== WD-01 VALIDATION LOOK #8: aggregation + gate v2 "
          "verdicts ===")
    print("ZERO simulation; holdout not accessed; raw rows from "
          "eval_val8.py (no re-roll).")
    verify_registered()

    # ---- raw rows (saved BEFORE aggregation by eval_val8.py)
    rows_l6, rows_l0 = [], []
    for mi in range(N_MEMBERS):
        rows_l6 += O.read_rows(RAW_DIR / f"raw_val8_m{mi}.jsonl")
    rows_l0 = O.read_rows(RAW_DIR / "raw_val8_L0.jsonl")
    wins = {(r["root"], int(r["t0"])) for r in rows_l6}
    assert len(wins) == N_WINDOWS, wins
    per_win_members = {k: len({r["member"] for r in rows_l6
                               if (r["root"], int(r["t0"])) == k})
                       for k in wins}
    assert set(per_win_members.values()) == {N_MEMBERS}
    print(f"raw rows: L6 {len(rows_l6)} ({N_MEMBERS} members x "
          f"{N_WINDOWS} windows x {len(HORIZONS)} H), L0 "
          f"{len(rows_l0)} -- saved before aggregation")

    # ---- the FROZEN spread factors (mean-centred widening)
    calib = N.load_json(P3O / "results" / "calibration_p3o.json")
    frozen_utc = N.load_json(
        P3O / "results" / "calibration_p3o.json.FROZEN.sha256"
    )["frozen_utc"]
    wmap = {}
    for name in ("delay_phys", "ttc3"):
        for H in HORIZONS:
            s = calib["s_out"][name].get(str(H))
            if s is not None:
                wmap.setdefault(name, {})[H] = float(s)
    print(f"frozen spread map (calibration_p3o.json, frozen_utc="
          f"{frozen_utc}): {wmap}")

    # ---- summaries + coverage/CRPS (part 3m functions, reused)
    per_vh = AG.summarise(rows_l6 + rows_l0)
    cov = AGM.coverage_and_crps_m(rows_l6 + rows_l0, horizons=HORIZONS,
                                  widen=None, variants=("L6",))
    cov_l6c = AGM.coverage_and_crps_m(rows_l6, horizons=HORIZONS,
                                      widen=wmap, variants=("L6c",))
    cov.update({k: v for k, v in cov_l6c.items() if k[0] == "L6c"})

    for H in HORIZONS:
        for variant in ("L6", "L6c"):
            c = cov[(variant, H)]
            print(f"  {variant}@H{H}: n_windows={c['n_windows']} "
                  f"ttc3_10-90_cov={c['ttc3_cov_frac']} "
                  f"(below {c['ttc3_below_frac']}, in "
                  f"{round(1 - c['ttc3_below_frac'] - c['ttc3_above_frac'], 4)}, "
                  f"above {c['ttc3_above_frac']}; truth=0 ties below "
                  f"{c['ttc3_below_truth0']}, above "
                  f"{c['ttc3_above_truth0']}) "
                  f"delay_phys_10-90_cov={c['delay_phys_cov_frac']} "
                  f"minmax={c['delay_phys_minmax_frac']} "
                  f"CRPS_phys={c['crps_phys_mean']} "
                  f"(s_out dp/t3 = {c['s_out_delay_phys']}/"
                  f"{c['s_out_ttc3']})")
        l0 = cov[("L0", H)]
        print(f"  L0@H{H}: deterministic delay_phys MAE="
              f"{l0['delay_phys_mae']} over {l0['n_windows']} windows "
              f"(rolled in this run)")

    # ---- gate v2 + old gate (L6c = the frozen candidate; L6 alongside)
    gates_v2, old_gates = {}, {}
    for H in HORIZONS:
        l0_speed = per_vh[("L0", H)]["speed_rmse"]
        s6 = per_vh[("L6", H)]
        gates_v2[("L6c", H)] = AG.gate_v2_verdict(
            s6, cov[("L6c", H)], cov[("L0", H)], l0_speed)
        gates_v2[("L6", H)] = AG.gate_v2_verdict(
            s6, cov[("L6", H)], cov[("L0", H)], l0_speed)
        old_gates[("L6c", H)] = AG.old_gate_verdict(s6, cov[("L6c", H)])
        old_gates[("L6", H)] = AG.old_gate_verdict(s6, cov[("L6", H)])

    for H in HORIZONS:
        tag = "PRIMARY (AMENDMENT_1)" if H == 30 else "secondary"
        print(f"\n== GATE v2 at H={H} ({tag}) -- (d) BINDING "
              f"per-horizon form ==")
        for variant in ("L6c", "L6"):
            g = gates_v2[(variant, H)]
            verdict = ("PASS" if g["v2_PASS"]
                       else "FAIL: " + ",".join(g["v2_failing"]))
            c_txt = (f"ttc3 {g['ttc3_cov_frac']} delay-phys "
                     f"{g['delay_phys_cov_frac']} "
                     f"crps {g['crps_phys']} vs mae "
                     f"{g['l0_delay_phys_mae']}")
            c_verdict = ("PASS" if g["c_coverage_crps_pass"] else "FAIL")
            print(f"  {variant}: (a) coll/1000="
                  f"{g['collisions_per_1000']} "
                  f"{'PASS' if g['a_collisions_pass'] else 'FAIL'} | "
                  f"(b) lc {g['lc_pred']}/{g['lc_true']} "
                  f"({100 * g['lc_deviation']:.1f}%) "
                  f"{'PASS' if g['b_lc_pass'] else 'FAIL'} | "
                  f"(c) {c_txt} -> {c_verdict} | "
                  f"(d) speed {g['speed_rmse']} vs "
                  f"1.05xL0@H={g['d_per_horizon_threshold']} "
                  f"({'PASS' if g['d_per_horizon_pass'] else 'FAIL'}, "
                  f"binding) / fixed 5.2185 "
                  f"({'PASS' if g['d_fixed_pass'] else 'FAIL'}) "
                  f"==> v2 {verdict}")
        if H == 30:
            g = gates_v2[("L6c", H)]
            overall = ("PASS" if g["v2_PASS"]
                       else "FAIL: " + ",".join(g["v2_failing"]))
            print(f"  ==> OVERALL GATE v2 H=30 (PRIMARY, frozen "
                  f"candidate L6c): {overall}")

    for H in HORIZONS:
        print(f"\n== OLD gate at H={H} ==")
        for variant in ("L6c", "L6"):
            g = old_gates[(variant, H)]
            print(f"  {variant}: speed {g['speed_rmse']} < 4.97 "
                  f"{g['speed_lt_B2']} | coll "
                  f"{g['collisions_per_1000']} {g['collisions_ok']} | "
                  f"lc {100 * g['lc_deviation']:.1f}% "
                  f"{g['lc_within_20pct']} | delay minmax "
                  f"{g['delay_phys_minmax_frac']} "
                  f"{g['delay_minmax_ge_50pct']} ==> "
                  f"{'PASS' if g['old_pass'] else 'FAIL'}")

    # ---- caveat table 1: LC BY TYPE (accel merges vs discretionary)
    sp = O.splits()
    print("\n== LC BY TYPE (per window, validation roots, members "
          "pooled) -- caveat table ==")
    bt = {}
    for H in HORIZONS:
        bt[H] = dict(pooled=O.lc_by_type(rows_l6, H),
                     per_root={root: O.lc_by_type(rows_l6, H, {root})
                               for root in sp["val"]})
        tab = bt[H]["pooled"]
        print(f"  H={H} pooled   predM {tab['pred_lc_merge']:5.2f} "
              f"trueM {tab['true_lc_merge']:5.2f} "
              f"({100*tab['dev_merge']:+6.1f}%) | predD "
              f"{tab['pred_lc_disc']:5.2f} trueD "
              f"{tab['true_lc_disc']:5.2f} "
              f"({100*tab['dev_disc']:+6.1f}%) | total "
              f"{tab['pred_lc']:5.2f}/{tab['true_lc']:5.2f} "
              f"({100*tab['dev_total']:+6.1f}%)")
        for root in sp["val"]:
            t = bt[H]["per_root"][root]
            print(f"  H={H} {root:<16} predM {t['pred_lc_merge']:5.2f} "
                  f"trueM {t['true_lc_merge']:5.2f} "
                  f"({100*t['dev_merge']:+6.1f}%) | predD "
                  f"{t['pred_lc_disc']:5.2f} trueD "
                  f"{t['true_lc_disc']:5.2f} "
                  f"({100*t['dev_disc']:+6.1f}%) | total "
                  f"{t['pred_lc']:5.2f}/{t['true_lc']:5.2f} "
                  f"({100*t['dev_total']:+6.1f}%)")

    # ---- caveat table 2: TTC<3 + physics-delay CRPS vs CLIMATOLOGY
    # (TRAIN roots ONLY: the 4 fit roots' truth sides (raw_fit_truth_m0)
    # + the 4 calibration roots' truth sides (raw_train_internal_iter2_
    # m0); the truth sides are member-independent)
    train_roots = set(sp["fit"]) | set(sp["calibration"])
    assert not (train_roots & set(sp["val"])) and \
        not (train_roots & set(sp["holdout"]))
    train_rows = O.read_rows(P3O / "results" / "raw" /
                             "raw_fit_truth_m0.jsonl")
    train_rows += O.read_rows(P3O / "results" / "raw" /
                              "raw_train_internal_iter2_m0.jsonl")
    roots_used = {r["root"] for r in train_rows}
    assert roots_used <= train_roots, roots_used - train_roots
    clima = N.climatology_from_rows(train_rows)
    n_train = {k: len(v) for k, v in sorted(clima.items())}
    print(f"\n== CLIMATOLOGY (TRAIN roots only: {sorted(roots_used)}; "
          f"samples per (output, H): {n_train}) ==")

    sharp_clim = {}
    print("   rung  output       H   cov    med_band  truth_iqr | "
          "crps     clima(emp) clima(mean) n_train | verdict")
    for rung, wm in (("L6", None), ("L6c", wmap)):
        for name, pk, tk in N.OUTPUTS:
            for H in HORIZONS:
                wins_h = N.member_arrays(rows_l6, "L6", H)
                widths, crps_v, truths = [], [], []
                s = None if wm is None else wm.get(name, {}).get(H)
                for _k, ws in wins_h:
                    x = np.array([float(m[pk]) for m in ws])
                    if s is not None:
                        x = N.L3.widen_ensemble(x, s, "mean")
                    lo, hi = N.L3.band_10_90(x)
                    widths.append(hi - lo)
                    crps_v.append(N.L3.crps_ensemble(x, float(ws[0][tk])))
                    truths.append(float(ws[0][tk]))
                samp = clima.get((name, H))
                clima_crps = (float(np.mean(
                    [N.L3.crps_ensemble(samp, y) for y in truths]))
                    if samp is not None and len(samp) >= 2 else None)
                clima_mean_crps = (float(np.mean(
                    [abs(float(samp.mean()) - y) for y in truths]))
                    if samp is not None and len(samp) >= 2 else None)
                crps_mean = float(np.mean(crps_v))
                cov_frac = cov[(rung, H)][
                    "ttc3_cov_frac" if name == "ttc3" else
                    "delay_phys_cov_frac"]
                beats = (crps_mean < clima_crps
                         if clima_crps is not None else None)
                truth_iqr = float(np.percentile(truths, 90) -
                                  np.percentile(truths, 10))
                key = f"{rung}@H{H}:{name}"
                sharp_clim[key] = dict(
                    median_band_width=round(float(np.median(widths)), 3),
                    truth_iqr_width=round(truth_iqr, 3),
                    band_over_truth_spread=round(
                        float(np.median(widths)) / truth_iqr, 3)
                    if truth_iqr > 0 else None,
                    crps=round(crps_mean, 3),
                    clima_crps_empirical=round(clima_crps, 3)
                    if clima_crps is not None else None,
                    clima_crps_meanpoint=round(clima_mean_crps, 3)
                    if clima_mean_crps is not None else None,
                    coverage=cov_frac,
                    beats_climatology=bool(beats)
                    if beats is not None else None,
                    informative=(bool(cov_frac >= 0.8 and beats)
                                 if beats is not None else None),
                    clima_samples=int(len(samp))
                    if samp is not None else 0)
                sc = sharp_clim[key]
                beats_txt = ("BEATS" if sc["beats_climatology"]
                             else "does NOT beat")
                print(f"   {rung:<5} {name:<12} {H}  {cov_frac:.4f}  "
                      f"{sc['median_band_width']:8.2f}  "
                      f"{sc['truth_iqr_width']:9.2f} | "
                      f"{sc['crps']:8.2f} {sc['clima_crps_empirical']:8.2f} "
                      f"{sc['clima_crps_meanpoint']:8.2f} "
                      f"{sc['clima_samples']:7d} | {beats_txt}; "
                      f"informative={sc['informative']}")

    result = dict(
        part="val8",
        what="WD-01 VALIDATION LOOK #8 (single, final for the frozen "
             "p3o candidate)",
        primary_H=30,
        registered_hashes=REGISTERED,
        raw_rows=dict(L6=len(rows_l6), L0=len(rows_l0),
                      windows=N_WINDOWS, members=N_MEMBERS,
                      val_roots=sorted(sp["val"]),
                      t0s=list(VAL_T0S), horizons=list(HORIZONS),
                      saved_before_aggregation=True),
        frozen_spread_map={k: {str(h): s for h, s in v.items()}
                           for k, v in wmap.items()},
        per_vh={f"{v}@H{h}": d for (v, h), d in per_vh.items()},
        coverage={f"{v}@H{h}": d for (v, h), d in cov.items()},
        gate_v2={f"{v}@H{h}": d for (v, h), d in gates_v2.items()},
        old_gate={f"{v}@H{h}": d for (v, h), d in old_gates.items()},
        overall_H30_primary=(
            "PASS" if gates_v2[("L6c", 30)]["v2_PASS"]
            else "FAIL: " + ",".join(
                gates_v2[("L6c", 30)]["v2_failing"])),
        lc_by_type={f"H{H}": bt[H] for H in HORIZONS},
        sharpness_climatology=sharp_clim,
        climatology_source=dict(
            roots=sorted(roots_used),
            files=["part3o/results/raw/raw_fit_truth_m0.jsonl",
                   "part3o/results/raw/raw_train_internal_iter2_m0.jsonl"],
            truth_sides_only=True,
            samples={f"{k[0]}@H{k[1]}": v for k, v in n_train.items()}),
        provenance=dict(
            candidate="p3o iter2 (3 TRAIN-internal selection iterations "
                      "on the calibration roots, iter0-2), frozen "
                      "2026-09-27T05:09:19Z",
            dev2_note="the candidate came from 3 TRAIN-internal "
                      "selection iterations on the calibration roots "
                      "(p3o iter0-2) after 7 prior validation looks "
                      "(WD-01-DEV-2 ledger); this is look #8, the "
                      "single final look for this candidate",
            holdout_accessed=False, holdout_loaded=False,
            simulation=False, validation_looks_used=8,
            one_look_policy="no re-run with changed settings; "
                            "aggregate reads saved raw rows only"),
    )
    (OUT / "gate_v2_val8.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {OUT / 'gate_v2_val8.json'}")
    print("DONE validation look #8 aggregation.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
