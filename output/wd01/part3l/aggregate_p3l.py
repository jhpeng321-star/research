"""WD-01 part 3l TASK D: aggregation of the ladder raw rows + gate
verdicts.  Reads results/raw/raw_metrics.jsonl (written FIRST by
eval_p3l.py) and computes:

  - per-(variant, H) summaries (means; pooled RMSEs) with BOTH delay
    metrics: exposure-based (p3h, continuity) and the CONTRACT PHYSICS
    DELAY (dt - ds/limit, the p3j/p3k decompose model);
  - gate v2 criterion (c) member-band coverage + ensemble CRPS at
    H=30 (PRIMARY per part3f amendment 1) and H=60, on the contract
    physics delay, for L4, L5 and the DERIVED rung L5c = L5 + the
    TASK C spread calibration (per-window member ensembles widened by
    the calibration s_out factors around their mean -- mean exactly
    invariant); exposure coverage reported alongside;
  - gate v2 (a)-(d) verdicts at H=30 (PRIMARY) and H=60 for L0, L4,
    L5, L5c, plus the old gate at both horizons.

Criterion (d) is reported BOTH ways per the reviewer clarification:
  d_fixed     speed RMSE <= 1.05 x 4.97 = 5.2185 m/s (the frozen text);
  d_horizon   speed RMSE <= 1.05 x L0's speed RMSE at the SAME horizon
              in the same run (BINDING per the clarification; at H=30
              that is 1.05 x L0@H30 of this run).
The per-horizon check decides the gate verdict; the fixed one is
reported alongside.

Pure functions (load_rows, summarise, coverage_and_crps, widen_map,
gate_v2_verdict, old_gate_verdict) carry a synthetic unit test in
tests/test_aggregate_p3l.py.  ZERO simulation; validation rows only.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "part3k"))
import decompose_p3k as DT_  # noqa: E402  (Tee)
import p3l_lib as L  # noqa: E402

RAW = HERE / "results" / "raw" / "raw_metrics.jsonl"
OUT = HERE / "results"
B2_SPEED = 4.97                     # p3c B2 reference speed RMSE (m/s)
D_FIXED = 1.05 * B2_SPEED           # 5.2185 m/s (frozen gate text)
VARIANTS = ("L0", "L4", "L5", "L5c")
MEMBERED = ("L4", "L5", "L5c")


def load_rows(path=RAW):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
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


def coverage_and_crps(rows, horizons=(30, 60), widen=None):
    """Per (variant, H): member-band coverage fractions + ensemble CRPS
    of the contract physics delay (exposure alongside), plus the L0
    deterministic MAEs.  widen: optional {output: {H: s}} map; when
    given, each window's member delay/ttc3 ensembles are widened by
    s_out around their MEAN (mean exactly invariant) before coverage /
    CRPS -- this derives the L5c rung from the L5 rows."""
    by_win = defaultdict(dict)   # (variant, H, root, t0) -> member -> row
    for r in rows:
        by_win[(r["variant"], int(r["H"]), r["root"],
                int(r["t0"]))][int(r["member"])] = r

    def wid(output, H):
        return None if not widen else widen.get(output, {}).get(H)

    out = {}
    for H in horizons:
        for variant in MEMBERED:
            t3c = dphys_c = dexp_c = dphys_mm = 0
            crps_v = []
            n_win = 0
            s_dp = s_t3 = None
            for (v, h, _root, _t0), members in sorted(by_win.items()):
                if v != ("L5" if variant == "L5c" else variant) \
                        or h != H or len(members) < 2:
                    continue
                n_win += 1
                rows_m = list(members.values())
                t3 = np.array([float(m["pred_ttc3_exposure"])
                               for m in rows_m])
                dphys = np.array([float(m["delay_phys_pred"])
                                  for m in rows_m])
                dexp = np.array([float(m["delay_pred"]) for m in rows_m])
                truth_t3 = float(rows_m[0]["true_ttc3_exposure"])
                truth_dp = float(rows_m[0]["delay_phys_true"])
                truth_de = float(rows_m[0]["delay_true"])
                s_dp, s_t3 = wid("delay_phys", H), wid("ttc3", H)
                if s_dp:
                    dphys = L.widen_ensemble(dphys, s_dp, "mean")
                if s_t3:
                    t3 = L.widen_ensemble(t3, s_t3, "mean")
                t3c += L.band_10_90(t3)[0] <= truth_t3 <= \
                    L.band_10_90(t3)[1]
                dphys_c += L.band_10_90(dphys)[0] <= truth_dp <= \
                    L.band_10_90(dphys)[1]
                dexp_c += L.band_10_90(dexp)[0] <= truth_de <= \
                    L.band_10_90(dexp)[1]
                dphys_mm += bool(min(dphys) <= truth_dp <= max(dphys))
                crps_v.append(L.crps_ensemble(dphys, truth_dp))
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
                                if crps_v else None),
                s_out_delay_phys=s_dp, s_out_ttc3=s_t3)
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


def widen_map_from_calibration(path, horizons=(30, 60)):
    """{output: {H: s_out}} from results/calibration_p3l.json (only
    horizons needed at gate time; None entries dropped)."""
    calib = L.load_fit(path)
    out = {}
    for name in ("delay_phys", "ttc3"):
        for H in horizons:
            s = calib.get("s_out", {}).get(name, {}).get(str(H))
            if s is not None:
                out.setdefault(name, {})[H] = float(s)
    return out


def gate_v2_verdict(summary, cov, l0_cov, l0_speed_same_H,
                    deterministic=False):
    """Gate v2 (a)-(d) for one rung.  (d) is evaluated BOTH ways: the
    per-horizon form (1.05 x L0 speed RMSE at the same H of this run)
    is BINDING per the reviewer clarification; the fixed 5.2185 form is
    reported alongside.  For deterministic L0 criterion (c) is n/a."""
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
    speed = summary["speed_rmse"]
    d_fixed_ok = speed <= D_FIXED
    d_h_thr = 1.05 * float(l0_speed_same_H)
    d_horizon_ok = speed <= d_h_thr          # BINDING
    checks = (("a_collisions", a_ok), ("b_lc", b_ok),
              ("c_coverage_crps", c_ok), ("d_speed_per_horizon",
                                          d_horizon_ok))
    failing = [nm for nm, ok in checks if ok is False]
    passed = a_ok and b_ok and d_horizon_ok and (c_ok is not False)
    return dict(
        collisions_per_1000=summary["collisions_per_1000"],
        a_collisions_pass=a_ok,
        lc_pred=summary["pred_lc"], lc_true=summary["true_lc"],
        lc_deviation=round(lc_dev, 4), b_lc_pass=b_ok,
        ttc3_cov_frac=cov.get("ttc3_cov_frac"), c_ttc3_pass=t3,
        delay_phys_cov_frac=cov.get("delay_phys_cov_frac"),
        c_delay_phys_pass=dc,
        crps_phys=crps, l0_delay_phys_mae=mae, c_crps_pass=crps_ok,
        c_coverage_crps_pass=c_ok, c_not_applicable=deterministic,
        speed_rmse=speed,
        d_fixed_threshold=round(D_FIXED, 4), d_fixed_pass=d_fixed_ok,
        d_per_horizon_threshold=round(d_h_thr, 4),
        d_per_horizon_pass=d_horizon_ok, d_per_horizon_is_binding=True,
        speed_rmse_pass=d_horizon_ok,
        delay_phys_bias=summary["delay_phys_bias"],
        delay_exposure_bias=summary["delay_exposure_bias"],
        v2_PASS=passed, v2_failing=failing)


def old_gate_verdict(summary, cov):
    """Old gate: speed RMSE < 4.97, collisions <= 1/1000, LC +-20 %,
    delay min-max coverage >= 50 % (physics delay)."""
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
    print("\n== gate v2 criterion (c): member bands + CRPS (contract "
          "physics delay; exposure coverage alongside; L5c = L5 + "
          "TRAIN-calibration spread factors) ==")
    for H in horizons:
        for variant in MEMBERED:
            c = cov[(variant, H)]
            print(f"  {variant}@H{H}: n_windows={c['n_windows']} "
                  f"delay_phys_10-90_cov={c['delay_phys_cov_frac']} "
                  f"delay_exposure_10-90_cov="
                  f"{c['delay_exposure_cov_frac']} "
                  f"ttc3_10-90_cov={c['ttc3_cov_frac']} "
                  f"delay_phys_minmax={c['delay_phys_minmax_frac']} "
                  f"CRPS_phys={c['crps_phys_mean']} "
                  f"(s_out dp/t3 = {c['s_out_delay_phys']}/"
                  f"{c['s_out_ttc3']})")
        l0 = cov[("L0", H)]
        print(f"  L0@H{H}: deterministic delay_phys MAE="
              f"{l0['delay_phys_mae']} over {l0['n_windows']} windows")
    for H in horizons:
        tag = "PRIMARY" if H == 30 else "secondary"
        print(f"\n== GATE v2 at H={H} ({tag}) -- (d) BINDING per-horizon "
              f"form ==")
        for variant in VARIANTS:
            g = gates_v2[(variant, H)]
            verdict = ("PASS" if g["v2_PASS"]
                       else "FAIL: " + ",".join(g["v2_failing"]))
            if g["c_not_applicable"]:
                c_txt = "n/a (deterministic)"
                c_verdict = "NA"
            else:
                c_txt = (f"ttc3 {g['ttc3_cov_frac']} delay-phys "
                         f"{g['delay_phys_cov_frac']} "
                         f"crps {g['crps_phys']} vs mae "
                         f"{g['l0_delay_phys_mae']}")
                c_verdict = ("PASS" if g["c_coverage_crps_pass"]
                             else "FAIL")
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
    for H in horizons:
        print(f"\n== OLD gate at H={H} ==")
        for variant in VARIANTS:
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
    (OUT).mkdir(parents=True, exist_ok=True)
    sys.stdout = DT_.Tee(OUT / "stdout_aggregate.txt")
    print("=== WD-01 part 3l: ladder aggregation + gate verdicts ===")
    print("ZERO simulation; holdout blocked in root_arrays (never "
          "passed); validation rows only.")
    rows = load_rows()
    print(f"loaded {len(rows)} deduped raw rows")
    per_vh = summarise(rows)
    cov = coverage_and_crps(rows, horizons=(30, 60))
    wmap = widen_map_from_calibration(
        HERE / "results" / "calibration_p3l.json", horizons=(30, 60))
    print(f"L5c spread map applied from calibration_p3l.json: {wmap}")
    cov_c = coverage_and_crps([r for r in rows if r["variant"] == "L5"],
                              horizons=(30, 60), widen=wmap)
    # adopt ONLY the derived L5c blocks: cov_c was built from the L5
    # row subset, so its L4/L0 entries are empty and must not clobber
    # the real ones
    for k, v in cov_c.items():
        if k[0] == "L5c":
            cov[k] = v
    # per-horizon L0 speed reference for the BINDING (d) form
    gates_v2, old_gates = {}, {}
    for H in (30, 60):
        l0_speed = per_vh[("L0", H)]["speed_rmse"]
        for variant in VARIANTS:
            # L5c shares L5's trajectory-level metrics (the spread
            # calibration only widens the delay/ttc3 member bands)
            s = per_vh[("L5" if variant == "L5c" else variant, H)]
            if variant == "L0":
                c = dict(ttc3_cov_frac=None, delay_phys_cov_frac=None,
                         crps_phys_mean=None,
                         delay_phys_minmax_frac=None)
                gates_v2[(variant, H)] = gate_v2_verdict(
                    s, c, cov[("L0", H)], l0_speed, deterministic=True)
                old_gates[(variant, H)] = old_gate_verdict(s, c)
            else:
                gates_v2[(variant, H)] = gate_v2_verdict(
                    s, cov[(variant, H)], cov[("L0", H)], l0_speed)
                old_gates[(variant, H)] = old_gate_verdict(
                    s, cov[(variant, H)])
    print_tables(per_vh, cov, gates_v2, old_gates)
    result = dict(
        per_vh={f"{v}@H{h}": d for (v, h), d in per_vh.items()},
        coverage={f"{v}@H{h}": d for (v, h), d in cov.items()},
        gate_v2={f"{v}@H{h}": d for (v, h), d in gates_v2.items()},
        old_gate={f"{v}@H{h}": d for (v, h), d in old_gates.items()},
        l5c_widen_map={k: {str(h): s for h, s in v.items()}
                       for k, v in wmap.items()},
        note=dict(primary_H=30,
                  d_binding="per-horizon 1.05 x L0@same-H (reviewer "
                            "clarification); fixed 5.2185 reported "
                            "alongside",
                  delay_metric="contract physics delay dt - ds/limit "
                               "(p3j/p3k decompose model); exposure "
                               "saved alongside",
                  amendment="part3f/VALIDATION_GATE_V2_AMENDMENT_1.md",
                  l4_reference="p3k params incl. the p3k follower-slot "
                               "defect (part 3l TASK A); L5 has the fix"),
    )
    (OUT / "gate_v2_p3l.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {OUT / 'gate_v2_p3l.json'}")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
