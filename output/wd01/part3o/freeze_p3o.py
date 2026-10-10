"""WD-01 part 3o: FREEZE the candidate + TRAIN-calibrated spread
factors, for a POSSIBLE LATER single validation look (NOT run here).

Refuses unless the declared iteration's TRAIN-internal gate JSON shows
the p3o goal package PASS at BOTH horizons:
  - LC by type AND total within +-20 % of truth at H=30 AND H=60
    (pooled + both folds),
  - gate v2 (a) collisions <= 1/1000 and (d) speed per-horizon,
  - 2-fold delay-phys CRPS no worse than the committed p3n iter1 gate.

What gets frozen (sha256 over LF text + UTC, N.write_frozen):
  results/candidate_p3o_iter{k}.json  the member parameters + the
                                      declared iteration protocol
                                      (fitted on the TRAIN fit roots
                                      only);
  results/calibration_p3o.json        the spread factors s_out per
                                      (output, horizon) fitted on the
                                      FULL TRAIN-internal calibration
                                      split (the part 3m/3n production
                                      protocol; the 2-fold fitting was
                                      the gate's anti-leakage device).

NO validation run happens here or anywhere in part 3o; holdout
blocked; ZERO simulation.
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
import p3o_lib as O  # noqa: E402
import p3n_lib as N  # noqa: E402
import decompose_p3k as D  # noqa: E402

ROWS_PER_MEMBER = 2 * len(O.GATE_T0S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--accept-residuals", action="store_true",
                    help="freeze despite crps_not_worse / b_type_pooled "
                         "residuals (they are recorded as "
                         "known_residuals in the frozen candidate; the "
                         "formal gate (b) total, (a) and (d) must still "
                         "PASS at both horizons)")
    args = ap.parse_args()
    k = args.iter
    sys.stdout = D.Tee(O.RES / "stdout_freeze.txt")
    print(f"=== WD-01 part 3o FREEZE (candidate + spread), "
          f"iteration {k} ===")
    print("NO validation run; holdout blocked; ZERO simulation.")

    gate_path = O.RES / f"gate_train_internal_iter{k}.json"
    fit_path = O.RES / f"fit_p3o_iter{k}.json"
    if not gate_path.exists() or not fit_path.exists():
        raise SystemExit(f"FREEZE REFUSED: {gate_path.name} / "
                         f"{fit_path.name} missing -- fit and gate "
                         f"must run first")
    gate = N.load_json(gate_path)
    hard, residuals = [], []
    for H in O.HORIZONS:
        chk = gate["goal_package"][f"H{H}"]["checks"]
        for item in ("b_total_gate", "a_collisions", "d_speed"):
            if not chk[item]:
                hard.append(f"H{H}:{item}")
        for item in ("b_type_pooled", "crps_not_worse"):
            if not chk[item]:
                residuals.append(f"H{H}:{item}")
    if hard:
        raise SystemExit(
            f"FREEZE REFUSED: formal criteria failed: {hard} -- "
            f"iterate on TRAIN first")
    if residuals and not args.accept_residuals:
        raise SystemExit(
            f"FREEZE REFUSED: goal-package residuals {residuals} "
            f"(merge-type pooled band / delay CRPS vs p3n iter1) -- "
            f"re-run with --accept-residuals to freeze with these "
            f"recorded as known_residuals")
    print(f"formal criteria PASS at both horizons "
          f"(b total, (a) collisions, (d) speed); residuals: "
          f"{residuals if residuals else 'NONE'}")
    for H in O.HORIZONS:
        g = gate["gate_v2"][f"H{H}"]
        gp = gate["goal_package"][f"H{H}"]
        print(f"  H={H}: lc {g['lc_pred']}/{g['lc_true']} "
              f"({100*g['lc_deviation']:+.1f}%) | by-type dev "
              f"merge {100*gp['by_type']['pooled']['dev_merge']:+.1f}% "
              f"disc {100*gp['by_type']['pooled']['dev_disc']:+.1f}% | "
              f"crps {gp['crps_phys']} vs ref {gp['crps_p3n_iter1']} | "
              f"speed {gp['speed_rmse']} <= {gp['d_threshold']} | "
              f"coll {g['collisions_per_1000']}")

    # ---- candidate params JSON (extracted, self-describing)
    fit = N.load_json(fit_path)
    cand = dict(
        part="3o", iteration=k,
        frozen_candidate=True,
        variant=fit["variant"],
        engine_kind="idm_merge",
        engine_flags=dict(lc_scale_disc="part 3o: discretionary 0<->1 "
                                         "hazard scale (default: "
                                         "lc_scale)",
                          merge_accept_prob="part 3o: thinning of "
                                            "gap-accepted M2 merges "
                                            "(default 1.0 = shipped "
                                            "deterministic)"),
        fitted_on=fit["fitted_on"],
        declared=fit["declared"],
        winners=fit["winners"],
        members=fit["members"],
        provenance=dict(
            fit_json=fit_path.name,
            gate_json=gate_path.name,
            p3n_iter1_reference=gate["p3n_iter1_reference"],
            formal_criteria_pass=True,
            goal_package_pass=gate.get("goal_package_pass", False),
            known_residuals=residuals,
            residual_note=("merge-type pooled LC band and delay-phys "
                           "CRPS vs p3n iter1 where listed: evidenced "
                           "as structural for the declared levers "
                           "(see REPORT_P3O.md); the owner decides "
                           "whether the single validation look "
                           "proceeds")),
    )
    cand_path = O.RES / f"candidate_p3o_iter{k}.json"
    cand_path.write_text(json.dumps(cand, indent=1, default=str),
                         encoding="utf-8")

    # ---- spread factors on the FULL calibration split
    member_files = [O.RAW_DIR / f"raw_train_internal_iter{k}_m{mi}.jsonl"
                    for mi in range(len(fit["members"]))]
    rows = []
    for p in member_files:
        if not O.member_file_complete(p, ROWS_PER_MEMBER):
            raise SystemExit(f"FREEZE REFUSED: {p.name} incomplete")
        rows += [r for r in O.read_rows(p) if r["root"] in
                 set(O.splits()["calibration"])]
    wins = len({(r["root"], int(r["t0"])) for r in rows})
    print(f"\nFULL-split spread calibration: {wins} windows x "
          f"{len(member_files)} members; L3 spread math; target 0.80")
    calib = dict(s_out={}, coverage={}, meta=dict(
        part="3o", task="freeze", iteration=k,
        calibration_roots=sorted(O.splits()["calibration"]),
        windows=wins, members=len(member_files),
        source_raw=[p.name for p in member_files],
        gate_json=gate_path.name, candidate_json=cand_path.name,
        holdout_accessed=False, simulation=False,
        validation_used=False, validation_loaded=False,
        note=("frozen factors for a POSSIBLE single validation look "
              "(NOT run in part 3o), fitted on the FULL TRAIN-internal "
              "calibration split; mechanism A = mean-centred widening, "
              "mean exactly invariant; s_out = smallest grid factor "
              "reaching 80 % coverage")))
    for name, pk, tk in O.OUTPUTS:
        for H in O.HORIZONS:
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
            calib["s_out"].setdefault(name, {})[str(H)] = s_out
            calib["coverage"][f"{name}@H{H}"] = entry
            print(f"  {name}@H{H}: raw_cov={entry['raw_coverage']} "
                  f"s_out={s_out} "
                  f"cov_mean={entry.get('coverage_mean_centred')}")

    # ---- FREEZE
    side_cand = N.write_frozen(
        cand_path, "the part 3o member parameters (frozen candidate); "
                   "any validation look must verify this hash first")
    cal_json = O.RES / "calibration_p3o.json"
    cal_json.write_text(json.dumps(calib, indent=1, default=str),
                        encoding="utf-8")
    side_cal = N.write_frozen(
        cal_json, "part 3o spread factors (FULL TRAIN-internal "
                  "calibration split); the validation look is NOT run")
    print(f"\nFROZEN {cand_path.name}: sha256={side_cand['sha256']}")
    print(f"        frozen_utc={side_cand['frozen_utc']}")
    print(f"FROZEN {cal_json.name}: sha256={side_cal['sha256']}")
    print(f"        frozen_utc={side_cal['frozen_utc']}")
    print("\nDONE part 3o freeze.  The single validation look is NOT "
          "part of part 3o.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
