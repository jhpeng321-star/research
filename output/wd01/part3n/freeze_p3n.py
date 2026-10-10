"""WD-01 part 3n: FREEZE the model + spread factors after a
TRAIN-internal gate pass, BEFORE the single validation look.

Order (brief): TRAIN-internal gate v2 passes at H=30 (all four
criteria) on the 2-fold protocol -> the spread factors are refitted on
the FULL TRAIN-internal calibration split (all 4 calibration roots, 28
windows -- the part 3m protocol for the frozen factors; the 2-fold
fitting was the gate's anti-leakage device, not the production
calibration) -> results/calibration_p3n.json is written and FROZEN
(sha256 over the LF text + UTC) -> results/fit_p3n.json (the member
parameters) is FROZEN the same way -> eval_p3n.py refuses to run the
validation look unless BOTH sidecars match.

This script REFUSES to freeze unless the declared iteration's
TRAIN-internal gate JSON shows v2_PASS at H=30 (the gate must have
passed first -- the freeze never precedes the pass).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("WD_DEVICE", "cpu")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p3n_lib as N  # noqa: E402
import decompose_p3k as D  # noqa: E402  (Tee)

RES = HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True,
                    help="the PASSING TRAIN-internal gate iteration")
    args = ap.parse_args()
    sys.stdout = D.Tee(RES / "stdout_freeze.txt")
    print("=== WD-01 part 3n: FREEZE (model + spread) after "
          f"TRAIN-internal pass, iteration {args.iter} ===")
    print("ZERO simulation; holdout blocked; validation NOT touched "
          "here (the freeze precedes the single validation look).")

    gate_path = RES / f"gate_train_internal_iter{args.iter}.json"
    if not gate_path.exists():
        raise SystemExit(f"FREEZE REFUSED: {gate_path.name} missing -- "
                         f"the TRAIN-internal gate must run first")
    gate = N.load_json(gate_path)
    g30 = gate["gate_v2"]["H30"]
    if not g30["v2_PASS"]:
        raise SystemExit(f"FREEZE REFUSED: iteration {args.iter} "
                         f"TRAIN-internal gate v2 at H=30 is FAIL: "
                         f"{g30['v2_failing']} -- iterate on TRAIN "
                         f"first; the freeze never precedes the pass")
    print(f"TRAIN-internal gate v2 H=30 PASS confirmed "
          f"(failing={g30['v2_failing']}); lc "
          f"{g30['lc_pred']}/{g30['lc_true']} "
          f"({100 * g30['lc_deviation']:.1f}%), ttc3 "
          f"{g30['ttc3_cov_frac']}, delay-phys "
          f"{g30['delay_phys_cov_frac']}, crps {g30['crps_phys']} < "
          f"{g30['l0_delay_phys_mae']}, speed {g30['speed_rmse']} <= "
          f"{g30['d_per_horizon_threshold']}")

    # ---- which model?  iteration 0 == the p3m members; k >= 1 == the
    # part 3n refit members (results/fit_p3n.json, mandatory TASK B
    # artifact)
    fit_path = RES / "fit_p3n.json"
    if args.iter == 0:
        raise SystemExit("FREEZE REFUSED for iteration 0: the brief's "
                         "TASK B joint refit is mandatory; freeze a "
                         "refit iteration (the p3m baseline cannot be "
                         "the p3n deliverable)")
    if not fit_path.exists():
        raise SystemExit("run fit_p3n.py first")
    fit = N.load_json(fit_path)
    configs = fit["members"]

    # ---- spread on the FULL calibration split from the passing raw
    raw_path = RES / "raw" / (("raw_train_internal_p3n.jsonl")
                              if args.iter == 0 else
                              f"raw_train_internal_iter{args.iter}.jsonl")
    rows = [json.loads(ln) for ln in
            raw_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    cal = set(N.L3.splits_3l()["calibration"])
    rows = [r for r in rows if r["variant"] == "L6" and
            r["root"] in cal]
    wins = len({(r["root"], r["t0"]) for r in rows})
    print(f"\nFULL-split spread calibration on the TRAIN-internal "
          f"calibration roots: {wins} windows x {len(configs)} members "
          f"({raw_path.name}); L3 spread math; target 0.80")
    calib = dict(s_out={}, coverage={}, meta=dict(
        part="3n", task="freeze",
        calibration_roots=sorted(cal),
        windows=wins, members=len(configs),
        source_raw=raw_path.name,
        gate_iteration=args.iter,
        gate_json=gate_path.name,
        holdout_accessed=False, simulation=False,
        validation_used=False,
        note=("frozen factors for the SINGLE validation look, fitted "
              "on the FULL TRAIN-internal calibration split (the 2-fold "
              "fitting was the gate's anti-leakage device); mechanism A "
              "= mean-centred widening, mean exactly invariant; s_out "
              "= smallest grid factor reaching 80 % coverage")))
    for name, pk, tk in N.OUTPUTS:
        for H in N.HORIZONS:
            wa = N.member_arrays(rows, "L6", H)
            samples = [np.array([float(m[pk]) for m in ws])
                       for _k, ws in wa]
            truths = [float(ws[0][tk]) for _k, ws in wa]
            raw_cov = N.L3.coverage_frac(samples, truths)
            s_out = N.L3.s_out_for(samples, truths)
            entry = dict(raw_coverage=round(raw_cov, 4), s_out=s_out,
                         target=N.L3.COVERAGE_TARGET)
            if s_out is not None:
                wid = [N.L3.widen_ensemble(x, s_out, "mean")
                       for x in samples]
                entry["coverage_mean_centred"] = round(
                    N.L3.coverage_frac(wid, truths), 4)
                entry["crps_raw_mean"] = round(float(np.mean(
                    [N.L3.crps_ensemble(x, y) for x, y in
                     zip(samples, truths)])), 3)
                entry["crps_widened_mean"] = round(float(np.mean(
                    [N.L3.crps_ensemble(x, y) for x, y in
                     zip(wid, truths)])), 3)
                entry["mean_shift_max"] = round(float(max(
                    abs(float(np.mean(w)) - float(np.mean(x)))
                    for w, x in zip(wid, samples))), 6)
            calib["s_out"].setdefault(name, {})[str(H)] = s_out
            calib["coverage"][f"{name}@H{H}"] = entry
            print(f"  {name}@H{H}: raw_cov={entry['raw_coverage']} "
                  f"s_out={s_out} "
                  f"cov_mean={entry.get('coverage_mean_centred')} "
                  f"crps {entry.get('crps_raw_mean')}->"
                  f"{entry.get('crps_widened_mean')}")

    # ---- FREEZE (order: calibration file, then both sidecars)
    cal_text = json.dumps(calib, indent=1, default=str)
    cal_json = RES / "calibration_p3n.json"
    cal_json.write_text(cal_text, encoding="utf-8")
    side_cal = N.write_frozen(
        cal_json, "eval_p3n.py + aggregate_p3n.py refuse the validation "
                  "look unless this hash matches; frozen AFTER the "
                  "TRAIN-internal gate pass, BEFORE the validation look")
    side_fit = N.write_frozen(
        fit_path, "the part 3n member parameters (model) frozen "
                  "together with the spread factors")
    print(f"\nFROZEN {cal_json.name}: sha256={side_cal['sha256']}")
    print(f"        frozen_utc={side_cal['frozen_utc']}")
    print(f"FROZEN {fit_path.name}: sha256={side_fit['sha256']}")
    print(f"        frozen_utc={side_fit['frozen_utc']}")
    print("\nDONE part 3n freeze.  Next: ONE validation look "
          "(eval_p3n.py, VAL look #8).")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
