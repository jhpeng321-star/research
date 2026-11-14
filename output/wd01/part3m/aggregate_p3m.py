"""WD-01 part 3m: ladder aggregation + gate v2 verdicts on VALIDATION.

Rungs:
  L0   deterministic (part 3l rows, reused verbatim);
  L4   p3k params on the fixed engine (part 3l rows, reused);
  L5   part 3l fit (part 3l rows, reused);
  L5c  L5 + the part 3l TRAIN-calibration spread factors (derived, no
       re-roll; factors from calibration_p3l_reference.json, a verbatim
       copy of part 3l results/calibration_p3l.json);
  L6   part 3m refit (fit_p3m.json members; fresh rows, H in {30, 60});
  L6c  L6 + the part 3m recalibrated spread factors (calibration_p3m.json,
       frozen before the validation run -- the freeze gate is re-checked
       here as well).

Gate v2 (a)-(d) at H=30 (PRIMARY, per part3f amendment 1) and H=60,
plus the old gate alongside, using the SAME verdict functions as part
3l (imported from aggregate_p3l).  Criterion (c) = ttc3 coverage >= 0.8
AND delay-phys coverage >= 0.8 AND ensemble CRPS < L0 delay MAE.  Also
the TTC<3 coverage BREAKDOWN per rung and horizon: windows with the
truth BELOW / IN / ABOVE the widened 10-90 % band, with the truth == 0
ties counted inside below/above.

ZERO simulation; validation rows only; holdout never passed.
"""
from __future__ import annotations

import hashlib
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
sys.path.insert(0, str(HERE.parent / "part3l"))
import decompose_p3k as D  # noqa: E402  (Tee)
import aggregate_p3l as AG  # noqa: E402  (VERDICT functions, reused)
import p3m_lib as M  # noqa: E402

RAW_P3L = HERE / "results" / "raw" / "raw_p3l_metrics.jsonl"
RAW_P3M = HERE / "results" / "raw" / "raw_metrics_p3m.jsonl"
CAL_P3L = HERE / "results" / "calibration_p3l_reference.json"
CAL_P3M = HERE / "results" / "calibration_p3m.json"
CAL_P3M_FROZEN = HERE / "results" / "calibration_p3m.FROZEN.sha256"
OUT = HERE / "results"
VARIANTS = ("L0", "L4", "L5", "L5c", "L6", "L6c")
MEMBERED = ("L4", "L5", "L5c", "L6", "L6c")


def sha256_of(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8")
                          .encode("utf-8")).hexdigest()


def load_rows_p3m(path=RAW_P3M):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    dedup = {}
    for r in rows:
        dedup[(r["variant"], int(r["member"]), r["root"], int(r["t0"]),
               int(r["H"]))] = r
    return list(dedup.values())


def coverage_and_crps_m(rows, horizons=(30, 60), widen=None,
                        variants=MEMBERED):
    """Member-band coverage + CRPS for arbitrary rungs (the part 3l
    logic generalised to a variants argument; identical widening,
    coverage and CRPS formulas via p3l_lib/p3m_lib shared helpers)."""
    by_win = defaultdict(dict)
    for r in rows:
        by_win[(r["variant"], int(r["H"]), r["root"],
                int(r["t0"]))][int(r["member"])] = r

    def wid(output, H):
        return None if not widen else widen.get(output, {}).get(H)

    out = {}
    for H in horizons:
        for variant in variants:
            base = "L5" if variant == "L5c" else \
                "L6" if variant == "L6c" else variant
            t3c = dphys_c = dexp_c = dphys_mm = 0
            below = above = below_t0 = above_t0 = 0
            crps_v = []
            n_win = 0
            s_dp = s_t3 = None
            for (v, h, _root, _t0), members in sorted(by_win.items()):
                if v != base or h != H or len(members) < 2:
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
                    dphys = M.L3.widen_ensemble(dphys, s_dp, "mean")
                if s_t3:
                    t3 = M.L3.widen_ensemble(t3, s_t3, "mean")
                lo3, hi3 = M.L3.band_10_90(t3)
                hit3 = lo3 <= truth_t3 <= hi3
                t3c += hit3
                below += truth_t3 < lo3
                above += truth_t3 > hi3
                below_t0 += (truth_t3 < lo3) and truth_t3 == 0.0
                above_t0 += (truth_t3 > hi3) and truth_t3 == 0.0
                dphys_c += M.L3.band_10_90(dphys)[0] <= truth_dp <= \
                    M.L3.band_10_90(dphys)[1]
                dexp_c += M.L3.band_10_90(dexp)[0] <= truth_de <= \
                    M.L3.band_10_90(dexp)[1]
                dphys_mm += bool(min(dphys) <= truth_dp <= max(dphys))
                crps_v.append(M.L3.crps_ensemble(dphys, truth_dp))
            out[(variant, H)] = dict(
                n_windows=n_win,
                ttc3_cov_frac=round(t3c / n_win, 4) if n_win else None,
                ttc3_below_frac=round(below / n_win, 4) if n_win else None,
                ttc3_above_frac=round(above / n_win, 4) if n_win else None,
                ttc3_below_truth0=below_t0 if n_win else None,
                ttc3_above_truth0=above_t0 if n_win else None,
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


def widen_map_from(path, horizons=(30, 60)):
    """{output: {H: s_out}} from a calibration JSON (part 3l format)."""
    calib = M.L3.load_fit(path)
    out = {}
    for name in ("delay_phys", "ttc3"):
        for H in horizons:
            s = calib.get("s_out", {}).get(name, {}).get(str(H))
            if s is not None:
                out.setdefault(name, {})[H] = float(s)
    return out


def main():
    sys.stdout = D.Tee(OUT / "stdout_aggregate.txt")
    print("=== WD-01 part 3m: ladder aggregation + gate verdicts ===")
    print("ZERO simulation; holdout blocked in root_arrays (never "
          "passed); validation rows only.")
    # freeze gate re-check (the calibration must pre-date this report)
    frozen = json.loads(CAL_P3M_FROZEN.read_text(encoding="utf-8"))
    sha = sha256_of(CAL_P3M)
    assert sha == frozen["sha256"], \
        "calibration_p3m.json changed after the FREEZE"
    print(f"FREEZE GATE ok: calibration_p3m.json sha256 "
          f"{sha[:16]}... frozen_utc={frozen['frozen_utc']}")

    rows_l = AG.load_rows(RAW_P3L)          # L0/L4/L5 (verbatim reuse)
    rows_m = load_rows_p3m(RAW_P3M)         # L6 (fresh)
    rows = rows_l + rows_m
    print(f"loaded {len(rows_l)} part 3l rows (L0/L4/L5, reused) + "
          f"{len(rows_m)} part 3m rows (L6)")

    wmap_l = widen_map_from(CAL_P3L)
    wmap_m = widen_map_from(CAL_P3M)
    print(f"L5c spread map (part 3l calibration, unchanged): {wmap_l}")
    print(f"L6c spread map (part 3m recalibration, frozen "
          f"{frozen['frozen_utc']}): {wmap_m}")

    per_vh = AG.summarise(rows)
    cov = coverage_and_crps_m(
        [r for r in rows if r["variant"] in ("L0", "L4", "L5")],
        horizons=(30, 60), widen=None, variants=("L4", "L5"))
    cov_l5c = coverage_and_crps_m(
        [r for r in rows if r["variant"] == "L5"], horizons=(30, 60),
        widen=wmap_l, variants=("L5c",))
    cov_l6 = coverage_and_crps_m(rows_m, horizons=(30, 60), widen=None,
                                 variants=("L6",))
    cov_l6c = coverage_and_crps_m(rows_m, horizons=(30, 60), widen=wmap_m,
                                  variants=("L6c",))
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

    result = dict(
        per_vh={f"{v}@H{h}": d for (v, h), d in per_vh.items()},
        coverage={f"{v}@H{h}": d for (v, h), d in cov.items()},
        gate_v2={f"{v}@H{h}": d for (v, h), d in gates_v2.items()},
        old_gate={f"{v}@H{h}": d for (v, h), d in old_gates.items()},
        widen_maps=dict(
            l5c={k: {str(h): s for h, s in v.items()}
                 for k, v in wmap_l.items()},
            l6c={k: {str(h): s for h, s in v.items()}
                 for k, v in wmap_m.items()}),
        freeze=dict(sha256=sha, frozen_utc=frozen["frozen_utc"]),
        note=dict(primary_H=30,
                  d_binding="per-horizon 1.05 x L0@same-H; fixed 5.2185 "
                            "alongside (part 3l convention, functions "
                            "reused from aggregate_p3l)",
                  l6="part 3m refit: TTC<3 exposure added to the stage-2 "
                     "loss, b_accept extended to 4.0/5.0, end_force_s "
                     "and v_merge_blend knobs fitted on TRAIN",
                  l6c="L6 + part 3m recalibration (TRAIN-internal "
                      "split, frozen before validation)"),
    )
    (OUT / "gate_v2_p3m.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {OUT / 'gate_v2_p3m.json'}")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
