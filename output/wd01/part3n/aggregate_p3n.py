"""WD-01 part 3n: aggregation + gate verdicts on VALIDATION (the one
look), plus the brief's SHARPNESS and CLIMATOLOGY tables.

Rungs (part 3m conventions; verdict functions reused from
aggregate_p3l / aggregate_p3m):
  L0   deterministic (part 3l rows, reused verbatim);
  L4   p3k params on the fixed engine (part 3l rows, reused);
  L5   part 3l fit (part 3l rows, reused); L5c + the part 3l factors;
  L6   the part 3n refit members (fresh rows, raw_metrics_p3n.jsonl);
  L6c  L6 + the FROZEN part 3n spread factors (calibration_p3n.json;
       the freeze gate is re-checked here as well).

Gate v2 (a)-(d) at H=30 (PRIMARY) and H=60 + the old gate, for every
rung.  Sharpness/climatology (brief, required): per output (contract
physics delay, TTC<3 exposure) and horizon -- median width of the
widened member 10-90 % band vs the across-window 10-90 % range of the
truth itself, and CRPS of the ensemble vs a CLIMATOLOGY baseline
(per-window TRAIN mean + the TRAIN empirical distribution, both built
from the 8 TRAIN roots x 7 t0 truth sides of the TASK A rows --
TRAIN-only by construction).  The TTC bands beating climatology is
stated explicitly; a band that passes coverage but not climatology
counts as UNINFORMATIVE.

ZERO simulation; validation rows only; holdout never passed.
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
import p3n_lib as N  # noqa: E402
import decompose_p3k as D  # noqa: E402  (Tee)
import aggregate_p3l as AG  # noqa: E402  (verdict functions, reused)
import aggregate_p3m as AGM  # noqa: E402  (coverage_and_crps_m)

RAW_P3L = HERE / "results" / "raw" / "raw_p3l_metrics.jsonl"
RAW_P3N = HERE / "results" / "raw" / "raw_metrics_p3n.jsonl"
RAW_TRAIN = HERE / "results" / "raw" / "raw_train_internal_p3n.jsonl"
CAL_P3L = HERE / "results" / "calibration_p3l_reference.json"
CAL_P3N = HERE / "results" / "calibration_p3n.json"
CAL_P3N_FROZEN = HERE / "results" / "calibration_p3n.json.FROZEN.sha256"
FIT_P3N_FROZEN = HERE / "results" / "fit_p3n.json.FROZEN.sha256"
OUT = HERE / "results"
VARIANTS = ("L0", "L4", "L5", "L5c", "L6", "L6c")
MEMBERED = ("L4", "L5", "L5c", "L6", "L6c")


def sha256_of(path):
    return N.sha256_text(Path(path).read_text(encoding="utf-8"))


def widen_map_from(path, horizons=(30, 60)):
    calib = N.load_json(path)
    out = {}
    for name in ("delay_phys", "ttc3"):
        for H in horizons:
            s = calib.get("s_out", {}).get(name, {}).get(str(H))
            if s is not None:
                out.setdefault(name, {})[H] = float(s)
    return out


def main():
    sys.stdout = D.Tee(OUT / "stdout_aggregate.txt")
    print("=== WD-01 part 3n: ladder aggregation + gate verdicts "
          "(VALIDATION look #8) ===")
    print("ZERO simulation; holdout blocked in root_arrays (never "
          "passed); validation rows only.")
    # freeze gate re-check (the calibration + model must pre-date this)
    frozen_cal = json.loads(
        CAL_P3N_FROZEN.read_text(encoding="utf-8"))
    frozen_fit = json.loads(
        FIT_P3N_FROZEN.read_text(encoding="utf-8"))
    sha_cal = sha256_of(CAL_P3N)
    sha_fit = sha256_of(HERE / "results" / "fit_p3n.json")
    assert sha_cal == frozen_cal["sha256"], \
        "calibration_p3n.json changed after the FREEZE"
    assert sha_fit == frozen_fit["sha256"], \
        "fit_p3n.json changed after the FREEZE"
    print(f"FREEZE GATE ok: calibration_p3n.json sha256 "
          f"{sha_cal[:16]}... frozen_utc={frozen_cal['frozen_utc']}")
    print(f"FREEZE GATE ok: fit_p3n.json sha256 "
          f"{sha_fit[:16]}... frozen_utc={frozen_fit['frozen_utc']}")

    rows_l = AG.load_rows(RAW_P3L)      # L0/L4/L5 (verbatim reuse)
    rows_m = AGM.load_rows_p3m(RAW_P3N)  # L6 (the one fresh look)
    rows = rows_l + rows_m
    print(f"loaded {len(rows_l)} part 3l rows (L0/L4/L5, reused) + "
          f"{len(rows_m)} part 3n rows (L6, VAL look #8)")

    wmap_l = widen_map_from(CAL_P3L)
    wmap_m = widen_map_from(CAL_P3N)
    print(f"L5c spread map (part 3l calibration, unchanged): {wmap_l}")
    print(f"L6c spread map (part 3n freeze, frozen "
          f"{frozen_cal['frozen_utc']}): {wmap_m}")

    per_vh = AG.summarise(rows)
    cov = AGM.coverage_and_crps_m(
        [r for r in rows if r["variant"] in ("L0", "L4", "L5")],
        horizons=(30, 60), widen=None, variants=("L4", "L5"))
    cov_l5c = AGM.coverage_and_crps_m(
        [r for r in rows if r["variant"] == "L5"], horizons=(30, 60),
        widen=wmap_l, variants=("L5c",))
    cov_l6 = AGM.coverage_and_crps_m(
        rows_m, horizons=(30, 60), widen=None, variants=("L6",))
    cov_l6c = AGM.coverage_and_crps_m(
        rows_m, horizons=(30, 60), widen=wmap_m, variants=("L6c",))
    for extra in (cov_l5c, cov_l6, cov_l6c):
        cov.update({k: v for k, v in extra.items()
                    if k[0] in ("L5c", "L6", "L6c")})

    for H in (30, 60):
        for variant in MEMBERED:
            c = cov[(variant, H)]
            print(f"  {variant}@H{H}: n_windows={c['n_windows']} "
                  f"ttc3_10-90_cov={c['ttc3_cov_frac']} "
                  f"(below {c['ttc3_below_frac']}, above "
                  f"{c['ttc3_above_frac']}; truth=0 ties below "
                  f"{c['ttc3_below_truth0']}, above "
                  f"{c['ttc3_above_truth0']}) "
                  f"delay_phys_10-90_cov={c['delay_phys_cov_frac']} "
                  f"delay_exposure_cov={c['delay_exposure_cov_frac']} "
                  f"minmax={c['delay_phys_minmax_frac']} "
                  f"CRPS_phys={c['crps_phys_mean']} "
                  f"(s_out dp/t3 = {c['s_out_delay_phys']}/"
                  f"{c['s_out_ttc3']})")
        l0 = cov[("L0", H)]
        print(f"  L0@H{H}: deterministic delay_phys MAE="
              f"{l0['delay_phys_mae']} over {l0['n_windows']} windows")

    gates_v2, old_gates = {}, {}
    for H in (30, 60):
        l0_speed = per_vh[("L0", H)]["speed_rmse"]
        for variant in VARIANTS:
            base = {"L5c": "L5", "L6c": "L6"}.get(variant, variant)
            s = per_vh[(base, H)]
            if variant == "L0":
                c = dict(ttc3_cov_frac=None, delay_phys_cov_frac=None,
                         crps_phys_mean=None,
                         delay_phys_minmax_frac=None)
                gates_v2[(variant, H)] = AG.gate_v2_verdict(
                    s, c, cov[("L0", H)], l0_speed, deterministic=True)
                old_gates[(variant, H)] = AG.old_gate_verdict(s, c)
            else:
                gates_v2[(variant, H)] = AG.gate_v2_verdict(
                    s, cov[(variant, H)], cov[("L0", H)], l0_speed)
                old_gates[(variant, H)] = AG.old_gate_verdict(
                    s, cov[(variant, H)])
        tag = "PRIMARY" if H == 30 else "secondary"
        print(f"\n== GATE v2 at H={H} ({tag}) -- (d) BINDING "
              f"per-horizon form ==")
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

    # ---- SHARPNESS + CLIMATOLOGY (brief) for L6 (raw) and L6c
    print("\n== SHARPNESS + CLIMATOLOGY on VALIDATION (L6 raw / L6c "
          "widened) ==")
    train_rows = [json.loads(ln) for ln in
                  RAW_TRAIN.read_text(encoding="utf-8").splitlines()
                  if ln.strip()]
    clima = N.climatology_from_rows(train_rows)   # ALL TRAIN windows
    sharp_clim = {}
    for rung, wm in (("L6", None), ("L6c", wmap_m)):
        for name, pk, tk in N.OUTPUTS:
            for H in (30, 60):
                wins = N.member_arrays(rows_m, "L6", H)
                widths, crps_v, truths = [], [], []
                s = None if wm is None else wm.get(name, {}).get(H)
                for _k, ws in wins:
                    x = np.array([float(m[pk]) for m in ws])
                    if s is not None:
                        x = N.L3.widen_ensemble(x, s, "mean")
                    lo, hi = N.L3.band_10_90(x)
                    y = float(ws[0][tk])
                    widths.append(hi - lo)
                    crps_v.append(N.L3.crps_ensemble(x, y))
                    truths.append(y)
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
                    beats_climatology=bool(beats) if beats is not None
                    else None,
                    informative=(bool(cov_frac >= 0.8 and beats)
                                 if beats is not None else None),
                    clima_samples=int(len(samp)) if samp is not None
                    else 0)
                sc = sharp_clim[key]
                beats_txt = ("BEATS" if sc["beats_climatology"]
                             else "does NOT beat")
                print(f"  {key:<24} band {sc['median_band_width']:8.2f} "
                      f"vs truth {sc['truth_iqr_width']:8.2f} "
                      f"(x{sc['band_over_truth_spread']}) | crps "
                      f"{sc['crps']:8.2f} vs clima(emp) "
                      f"{sc['clima_crps_empirical']} / clima(mean) "
                      f"{sc['clima_crps_meanpoint']} -> {beats_txt}; "
                      f"informative={sc['informative']}")

    result = dict(
        per_vh={f"{v}@H{h}": d for (v, h), d in per_vh.items()},
        coverage={f"{v}@H{h}": d for (v, h), d in cov.items()},
        gate_v2={f"{v}@H{h}": d for (v, h), d in gates_v2.items()},
        old_gate={f"{v}@H{h}": d for (v, h), d in old_gates.items()},
        sharpness_climatology=sharp_clim,
        widen_maps=dict(
            l5c={k: {str(h): s for h, s in v.items()}
                 for k, v in wmap_l.items()},
            l6c={k: {str(h): s for h, s in v.items()}
                 for k, v in wmap_m.items()}),
        freeze=dict(sha256_calibration=sha_cal,
                    frozen_utc=frozen_cal["frozen_utc"],
                    sha256_fit=sha_fit,
                    frozen_utc_fit=frozen_fit["frozen_utc"]),
        note=dict(
            primary_H=30,
            validation_look="look #8 (WD-01-DEV-2 ledger: 7 previous "
                            "looks); ONE look in part 3n",
            d_binding="per-horizon 1.05 x L0@same-H; fixed 5.2185 "
                      "alongside (part 3l convention)",
            l6="part 3n joint lc_scale x end_force_s refit with the "
               "LC-BY-TYPE stage-2 loss (TRAIN fit roots only)",
            l6c="L6 + the part 3n freeze (TRAIN-internal calibration "
                "split, frozen after the 2-fold TRAIN-internal gate "
                "pass, before the look)",
            climatology="TRAIN truth distribution: 8 TRAIN roots x 7 "
                        "t0 (part 3n TASK A truth sides); zero "
                        "simulation"),
    )
    (OUT / "gate_v2_p3n.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {OUT / 'gate_v2_p3n.json'}")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
