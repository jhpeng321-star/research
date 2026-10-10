"""WD-01 part 3k: aggregation of the ladder raw rows + gate verdicts.

Reads results/raw/raw_metrics.jsonl (written FIRST by eval_p3k.py) and
computes:
  - per-(variant, H) summaries (means; pooled RMSEs) with BOTH delay
    metrics: exposure-based (delay_pred/delay_true, the p3h metric, for
    continuity) and the CONTRACT PHYSICS DELAY (delay_phys_pred /
    delay_phys_true, dt - ds/limit, the p3j/p3k decompose model);
  - gate v2 criterion (c) member-band coverage + ensemble CRPS at H=30
    (PRIMARY per part3f/VALIDATION_GATE_V2_AMENDMENT_1.md) and H=60,
    computed on the CONTRACT PHYSICS DELAY (10-90 % member band);
    exposure-based coverage reported alongside for continuity;
  - gate v2 (a)-(d) verdicts at H=30 (PRIMARY) and H=60 for every rung,
    plus the old gate at both horizons.

Gate v2 (frozen part3f/VALIDATION_GATE_V2.md; primary H per amendment 1):
  (a) collisions <= 1 / 1000 vehicle-steps
  (b) |LC_pred - LC_true| / max(LC_true, 1) <= 0.20
  (c) member 10-90 % band covers truth in >= 80 % of windows for the
      window delay AND TTC<3 s exposure, AND ensemble CRPS(delay) <
      MAE of the deterministic L0 delay  [n/a for deterministic L0]
  (d) member-averaged speed RMSE <= 1.05 x B2 = 5.2185 m/s
Old gate: speed RMSE < 4.97 AND collisions <= 1/1000 AND LC +-20 %
AND delay min-max coverage >= 50 % (physics delay min-max).

Pure functions (coverage_10_90, coverage_minmax, crps_ensemble,
gate_v2_verdict, old_gate_verdict, summarise, load_rows,
coverage_and_crps) carry a synthetic unit test in
tests/test_aggregate_p3k.py.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("WD_DEVICE", "cpu")

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw" / "raw_metrics.jsonl"
OUT = HERE / "results"
B2_SPEED = 4.97                     # p3c B2 reference speed RMSE (m/s)


def coverage_10_90(vals, truth):
    """10th-90th percentile band (numpy linear) covers truth?"""
    p10 = float(np.percentile(vals, 10))
    p90 = float(np.percentile(vals, 90))
    return bool(p10 <= truth <= p90)


def coverage_minmax(vals, truth):
    return bool(min(vals) <= truth <= max(vals))


def crps_ensemble(vals, truth):
    """Empirical CRPS: mean|x_i - y| - 0.5 * mean_ij |x_i - x_j|."""
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) < 2:
        return float(abs(vals[0] - truth)) if len(vals) else float("nan")
    term1 = float(np.mean(np.abs(vals - truth)))
    diff = np.abs(vals[:, None] - vals[None, :])
    term2 = 0.5 * float(np.mean(diff))
    return term1 - term2


def load_rows(path=RAW):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    # deduplicate: keep the LAST occurrence of each key
    dedup = {}
    for r in rows:
        dedup[(r["variant"], int(r["member"]), r["root"], int(r["t0"]),
               int(r["H"]))] = r
    return list(dedup.values())


def summarise(rows):
    """(variant, H) -> summary dict (means; pooled RMSEs)."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["variant"], int(r["H"]))].append(r)
    out = {}
    for (variant, H), rr in sorted(grouped.items()):
        d = dict(n_rolls=len(rr),
                 n_members=len(set(r["member"] for r in rr)),
                 n_windows=len(set((r["root"], r["t0"]) for r in rr)))
        for k in ("pred_vehicle_steps", "matched_vehicle_steps",
                  "pred_lc", "true_lc",
                  "pred_ttc3_exposure", "true_ttc3_exposure",
                  "pred_hard_brake_s", "true_hard_brake_s",
                  "delay_pred", "delay_true",
                  "delay_phys_pred", "delay_phys_true"):
            d[k] = round(float(np.mean([r[k] for r in rr])), 3)
        d["speed_rmse"] = round(float(np.sqrt(np.mean(
            np.square([r["speed_rmse"] for r in rr])))), 4)
        d["pos_rmse"] = round(float(np.sqrt(np.mean(
            np.square([r["pos_rmse"] for r in rr])))), 2)
        d["lane_frac"] = round(float(np.mean(
            [r["lane_frac"] for r in rr])), 4)
        d["collisions_per_1000"] = round(float(np.mean(
            [r["collisions_per_1000"] for r in rr])), 3)
        d["delay_exposure_bias"] = round(
            d["delay_pred"] - d["delay_true"], 2)
        d["delay_phys_bias"] = round(
            d["delay_phys_pred"] - d["delay_phys_true"], 2)
        out[(variant, H)] = d
    return out


def coverage_and_crps(rows, horizons=(30, 60)):
    """Per variant & horizon: member-band coverage fractions and ensemble
    CRPS of the CONTRACT PHYSICS delay, plus the L0 deterministic MAE
    (physics and exposure)."""
    by_win = defaultdict(dict)      # (variant, H, root, t0) -> member->row
    for r in rows:
        by_win[(r["variant"], int(r["H"]), r["root"], int(r["t0"]))][
            int(r["member"])] = r
    out = {}
    for H in horizons:
        for variant in ("L3", "L4"):
            t3c = dphys_c = dexp_c = dphys_mm = 0
            crps_v = []
            n_win = 0
            for (v, h, _root, _t0), members in sorted(by_win.items()):
                if v != variant or h != H or len(members) < 2:
                    continue
                n_win += 1
                rows_m = list(members.values())
                t3 = [float(m["pred_ttc3_exposure"]) for m in rows_m]
                dphys = [float(m["delay_phys_pred"]) for m in rows_m]
                dexp = [float(m["delay_pred"]) for m in rows_m]
                truth_t3 = float(rows_m[0]["true_ttc3_exposure"])
                truth_dp = float(rows_m[0]["delay_phys_true"])
                truth_de = float(rows_m[0]["delay_true"])
                t3c += coverage_10_90(t3, truth_t3)
                dphys_c += coverage_10_90(dphys, truth_dp)
                dexp_c += coverage_10_90(dexp, truth_de)
                dphys_mm += coverage_minmax(dphys, truth_dp)
                crps_v.append(crps_ensemble(dphys, truth_dp))
            out[(variant, H)] = dict(
                n_windows=n_win,
                ttc3_cov_frac=round(t3c / n_win, 4) if n_win else None,
                delay_phys_cov_frac=(round(dphys_c / n_win, 4)
                                     if n_win else None),
                delay_exposure_cov_frac=(round(dexp_c / n_win, 4)
                                         if n_win else None),
                delay_phys_minmax_frac=(round(dphys_mm / n_win, 4)
                                        if n_win else None),
                crps_phys_mean=(round(float(np.mean(crps_v)), 2)
                                if crps_v else None))
        l0 = [r for r in rows if r["variant"] == "L0" and int(r["H"]) == H]
        out[("L0", H)] = dict(
            delay_phys_mae=round(float(np.mean(
                [abs(float(r["delay_phys_pred"])
                     - float(r["delay_phys_true"])) for r in l0])), 2)
            if l0 else None,
            delay_exposure_mae=round(float(np.mean(
                [abs(float(r["delay_pred"]) - float(r["delay_true"]))
                 for r in l0])), 2) if l0 else None,
            n_windows=len(l0))
    return out


def gate_v2_verdict(summary, cov, l0_cov, deterministic=False):
    """Gate v2 (a)-(d) for one rung from its (variant, H) summary; cov is
    the variant's coverage/CRPS block, l0_cov the L0 block (CRPS
    reference).  For the deterministic rung (L0) criterion (c) is n/a
    (no member band exists) and does not fail the gate."""
    a_ok = summary["collisions_per_1000"] <= 1.0
    lc_dev = abs(summary["pred_lc"] - summary["true_lc"]) / \
        max(summary["true_lc"], 1)
    b_ok = lc_dev <= 0.20
    t3 = (cov.get("ttc3_cov_frac") or 0) >= 0.80
    dc = (cov.get("delay_phys_cov_frac") or 0) >= 0.80
    crps = cov.get("crps_phys_mean")
    mae = l0_cov.get("delay_phys_mae")
    crps_ok = True if crps is None or mae is None else crps < mae
    c_ok = None if deterministic else (t3 and dc and crps_ok)
    d_ok = summary["speed_rmse"] <= 1.05 * B2_SPEED
    checks = (("a_collisions", a_ok), ("b_lc", b_ok),
              ("c_coverage_crps", c_ok), ("d_speed", d_ok))
    failing = [nm for nm, ok in checks if ok is False]
    passed = a_ok and b_ok and d_ok and (c_ok is not False)
    return dict(
        collisions_per_1000=summary["collisions_per_1000"],
        a_collisions_pass=a_ok,
        lc_pred=summary["pred_lc"], lc_true=summary["true_lc"],
        lc_deviation=round(lc_dev, 4), b_lc_pass=b_ok,
        ttc3_cov_frac=cov.get("ttc3_cov_frac"), c_ttc3_pass=t3,
        delay_phys_cov_frac=cov.get("delay_phys_cov_frac"),
        c_delay_phys_pass=dc,
        crps_phys=crps, l0_delay_phys_mae=mae, c_crps_pass=crps_ok,
        c_coverage_crps_pass=c_ok,
        c_not_applicable=deterministic,
        speed_rmse=summary["speed_rmse"], d_speed_pass=d_ok,
        delay_phys_bias=summary["delay_phys_bias"],
        delay_exposure_bias=summary["delay_exposure_bias"],
        v2_PASS=passed,
        v2_failing=failing)


def old_gate_verdict(summary, cov):
    """Old gate: speed RMSE < B2, collisions <= 1/1000, LC +-20 %, delay
    min-max coverage >= 50 % (physics delay min-max)."""
    sr = summary["speed_rmse"]
    coll = summary["collisions_per_1000"]
    lc_dev = abs(summary["pred_lc"] - summary["true_lc"]) / \
        max(summary["true_lc"], 1)
    mm = cov.get("delay_phys_minmax_frac") or 0
    return dict(
        speed_rmse=sr, speed_lt_B2=sr < B2_SPEED,
        collisions_per_1000=coll, collisions_ok=coll <= 1.0,
        lc_pred=summary["pred_lc"], lc_true=summary["true_lc"],
        lc_deviation=round(lc_dev, 4), lc_within_20pct=lc_dev <= 0.20,
        delay_phys_minmax_frac=cov.get("delay_phys_minmax_frac"),
        delay_minmax_ge_50pct=mm >= 0.50,
        old_pass=sr < B2_SPEED and coll <= 1.0 and lc_dev <= 0.20
        and mm >= 0.50)


def print_tables(per_vh, cov, gates_v2, old_gates, horizons=(30, 60)):
    print("\n== per-(variant, H) summary (4 validation roots x 52 "
          "windows x members) ==")
    for (variant, H), d in sorted(per_vh.items()):
        print(f"  {variant}@H{H}: n_rolls={d['n_rolls']} "
              f"n_members={d['n_members']} n_windows={d['n_windows']} "
              f"speed_rmse={d['speed_rmse']} "
              f"coll/1000={d['collisions_per_1000']} "
              f"lc {d['pred_lc']}/{d['true_lc']} "
              f"| exposure delay p/t {d['delay_pred']}/"
              f"{d['delay_true']} (bias {d['delay_exposure_bias']}) "
              f"| physics delay p/t {d['delay_phys_pred']}/"
              f"{d['delay_phys_true']} (bias {d['delay_phys_bias']})")
    print("\n== gate v2 criterion (c): member bands + CRPS "
          "(contract physics delay; exposure coverage alongside) ==")
    for H in horizons:
        for variant in ("L3", "L4"):
            c = cov[(variant, H)]
            print(f"  {variant}@H{H}: n_windows={c['n_windows']} "
                  f"delay_phys_10-90_cov={c['delay_phys_cov_frac']} "
                  f"delay_exposure_10-90_cov="
                  f"{c['delay_exposure_cov_frac']} "
                  f"ttc3_10-90_cov={c['ttc3_cov_frac']} "
                  f"delay_phys_minmax={c['delay_phys_minmax_frac']} "
                  f"CRPS_phys={c['crps_phys_mean']} "
                  f"(L0 MAE_phys={cov[('L0', H)]['delay_phys_mae']})")
        l0 = cov[("L0", H)]
        print(f"  L0@H{H}: deterministic delay_phys MAE="
              f"{l0['delay_phys_mae']} over {l0['n_windows']} windows")
    for H in horizons:
        tag = "PRIMARY" if H == 30 else "secondary"
        print(f"\n== GATE v2 at H={H} ({tag}) ==")
        for variant in ("L0", "L3", "L4"):
            g = gates_v2[(variant, H)]
            verdict = ("PASS" if g["v2_PASS"]
                       else "FAIL: " + ",".join(g["v2_failing"]))
            c_pass = g["c_coverage_crps_pass"]
            if g["c_not_applicable"]:
                c_verdict = "NA"
            else:
                c_verdict = "PASS" if c_pass else "FAIL"
            if g["c_not_applicable"]:
                c_txt = "n/a (deterministic)"
            else:
                c_txt = (f"ttc3 {g['ttc3_cov_frac']} delay-phys "
                         f"{g['delay_phys_cov_frac']} "
                         f"crps {g['crps_phys']} vs mae "
                         f"{g['l0_delay_phys_mae']}")
            print(f"  {variant}: (a) coll/1000="
                  f"{g['collisions_per_1000']} "
                  f"{'PASS' if g['a_collisions_pass'] else 'FAIL'} | "
                  f"(b) lc {g['lc_pred']}/{g['lc_true']} "
                  f"({100 * g['lc_deviation']:.1f}%) "
                  f"{'PASS' if g['b_lc_pass'] else 'FAIL'} | "
                  f"(c) {c_txt} -> {c_verdict} | "
                  f"(d) speed {g['speed_rmse']} "
                  f"{'PASS' if g['d_speed_pass'] else 'FAIL'} "
                  f"==> v2 {verdict}")
    for H in horizons:
        print(f"\n== OLD gate at H={H} ==")
        for variant in ("L0", "L3", "L4"):
            g = old_gates[(variant, H)]
            print(f"  {variant}: speed {g['speed_rmse']} < 4.97 "
                  f"{g['speed_lt_B2']} | coll "
                  f"{g['collisions_per_1000']} {g['collisions_ok']} | "
                  f"lc {100 * g['lc_deviation']:.1f}% "
                  f"{g['lc_within_20pct']} | delay minmax "
                  f"{g['delay_phys_minmax_frac']} "
                  f"{g['delay_minmax_ge_50pct']} ==> "
                  f"{'PASS' if g['old_pass'] else 'FAIL'}")


def main():
    import decompose_p3k as DT_

    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = DT_.Tee(OUT / "stdout_aggregate.txt")
    print("=== WD-01 part 3k: ladder aggregation + gate verdicts ===")
    print("ZERO simulation; holdout blocked in root_arrays (never "
          "passed); validation only.")
    rows = load_rows()
    print(f"loaded {len(rows)} deduped raw rows")
    per_vh = summarise(rows)
    cov = coverage_and_crps(rows, horizons=(30, 60))
    gates_v2, old_gates = {}, {}
    for H in (30, 60):
        for variant in ("L0", "L3", "L4"):
            s = per_vh[(variant, H)]
            if variant == "L0":
                c = dict(ttc3_cov_frac=None, delay_phys_cov_frac=None,
                         crps_phys_mean=None,
                         delay_phys_minmax_frac=None)
                gates_v2[(variant, H)] = gate_v2_verdict(
                    s, c, cov[("L0", H)], deterministic=True)
                old_gates[(variant, H)] = old_gate_verdict(s, c)
            else:
                gates_v2[(variant, H)] = gate_v2_verdict(
                    s, cov[(variant, H)], cov[("L0", H)])
                old_gates[(variant, H)] = old_gate_verdict(
                    s, cov[(variant, H)])
    print_tables(per_vh, cov, gates_v2, old_gates)
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(
        per_vh={f"{v}@H{h}": d for (v, h), d in per_vh.items()},
        coverage={f"{v}@H{h}": d for (v, h), d in cov.items()},
        gate_v2={f"{v}@H{h}": d for (v, h), d in gates_v2.items()},
        old_gate={f"{v}@H{h}": d for (v, h), d in old_gates.items()},
        note=dict(primary_H=30,
                  delay_metric="contract physics delay dt - ds/limit "
                               "(p3j/p3k decompose model); exposure "
                               "metrics saved alongside for continuity",
                  amendment="part3f/VALIDATION_GATE_V2_AMENDMENT_1.md"),
    )
    (OUT / "gate_v2_p3k.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {OUT / 'gate_v2_p3k.json'}")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
