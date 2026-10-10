"""WD-01 part 3q: FREEZE the candidate + the FULL-SPLIT k-fold TRAIN
spread factors, for a POSSIBLE LATER single validation look (NOT run).

Refuses unless the declared iteration's k-fold gate JSON shows, POOLED
over the 17 TRAIN roots at BOTH horizons: gate v2 (b) total, (a)
collisions and (d) per-horizon speed PASS (the p3o hard set), plus the
p3q goal package's b_type pooled band.  The cross-fitted (c) coverage
and the delay-CRPS-vs-p3n-iter1 item are REPORT-AND-RECORD: when they
fail they become known_residuals (with the bias/width breakdown) --
the k-fold cross-fitted (c) is the honest TRAIN estimate of what a
validation look would see, and is the headline result either way.

What gets frozen (sha256 over LF text + UTC, N.write_frozen):
  results/candidate_p3q_iter{k}.json  the member parameters + the
                                      declared iteration protocol
                                      (fitted on folds f1..f4 only);
  results/calibration_p3q.json        the spread factors s_out per
                                      (output, horizon) fitted on the
                                      FULL 17-root TRAIN split (119
                                      windows -- the larger sample the
                                      brief asks for, vs the 28-window
                                      p3o calibration) + the per-output
                                      bias/width decomposition
                                      (below/in/above pre/post
                                      widening).

NO validation run happens here or anywhere in part 3q; holdout
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
import p3q_lib as Q  # noqa: E402
import p3n_lib as N  # noqa: E402
import p3l_lib as L3  # noqa: E402
import decompose_p3k as D  # noqa: E402


def spread_block(rows, name, pk, tk, H):
    """Full-split spread entry for one (output, H): raw coverage, s_out,
    widened coverage + the below/in/above and width breakdown pre/post
    widening (the bias-vs-width decomposition, brief item 2)."""
    wa = N.member_arrays(rows, "L6", H)
    samples = [np.array([float(m[pk]) for m in ws]) for _k, ws in wa]
    truths = [float(ws[0][tk]) for _k, ws in wa]
    raw_cov = L3.coverage_frac(samples, truths)
    s_out = L3.s_out_for(samples, truths)
    entry = dict(raw_coverage=round(raw_cov, 4), s_out=s_out,
                 target=L3.COVERAGE_TARGET, n_windows=len(samples))
    centred = [float(np.mean(x)) for x in samples]
    below_raw = sum(y < np.percentile(x, 10)
                    for x, y in zip(samples, truths))
    above_raw = sum(y > np.percentile(x, 90)
                    for x, y in zip(samples, truths))
    entry["raw_breakdown"] = dict(
        below=below_raw,
        **{"in": len(samples) - below_raw - above_raw},
        above=above_raw,
        bias_mean=round(float(np.mean(
            [y - c for y, c in zip(truths, centred)])), 2),
        width_med=round(float(np.median(
            [float(np.percentile(x, 90) - np.percentile(x, 10))
             for x in samples])), 2),
        truth_iqr=round(float(
            np.percentile(truths, 90) - np.percentile(truths, 10)), 2))
    if s_out is not None:
        wid = [L3.widen_ensemble(x, s_out, "mean") for x in samples]
        wid_cov = L3.coverage_frac(wid, truths)
        below_w = sum(y < np.percentile(x, 10)
                      for x, y in zip(wid, truths))
        above_w = sum(y > np.percentile(x, 90)
                      for x, y in zip(wid, truths))
        entry["coverage_mean_centred"] = round(wid_cov, 4)
        entry["widened_breakdown"] = dict(
            below=below_w,
            **{"in": len(wid) - below_w - above_w},
            above=above_w,
            width_med=round(float(np.median(
                [float(np.percentile(x, 90) - np.percentile(x, 10))
                 for x in wid])), 2))
        entry["crps_raw_mean"] = round(float(np.mean(
            [L3.crps_ensemble(x, y) for x, y in
             zip(samples, truths)])), 3)
        entry["crps_widened_mean"] = round(float(np.mean(
            [L3.crps_ensemble(x, y) for x, y in
             zip(wid, truths)])), 3)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--accept-residuals", action="store_true",
                    help="freeze despite b_type_pooled / c_gate / "
                         "crps_not_worse residuals (recorded as "
                         "known_residuals; the formal (b) total, (a) "
                         "and (d) must still PASS pooled at both "
                         "horizons)")
    args = ap.parse_args()
    k = args.iter
    sys.stdout = D.Tee(Q.RES / "stdout_freeze.txt")
    print(f"=== WD-01 part 3q FREEZE (candidate + full-split spread), "
          f"iteration {k} ===")
    print("NO validation run; holdout blocked; ZERO simulation.")

    gate_path = Q.RES / f"gate_kfold_iter{k}.json"
    fit_path = Q.RES / f"fit_p3q_iter{k}.json"
    if not gate_path.exists() or not fit_path.exists():
        raise SystemExit(f"FREEZE REFUSED: {gate_path.name} / "
                         f"{fit_path.name} missing -- fit and gate "
                         f"must run first")
    gate = Q.N.load_json(gate_path)
    hard, residuals = [], []
    for H in Q.HORIZONS:
        chk = gate["goal_package"][f"H{H}"]["checks"]
        for item in ("b_total_gate", "a_collisions", "d_speed"):
            if not chk[item]:
                hard.append(f"H{H}:{item}")
        for item in ("b_type_pooled", "c_gate"):
            if not chk[item]:
                residuals.append(f"H{H}:{item}")
    if hard:
        raise SystemExit(
            f"FREEZE REFUSED: formal criteria failed: {hard} -- "
            f"iterate on TRAIN first")
    if residuals and not args.accept_residuals:
        raise SystemExit(
            f"FREEZE REFUSED: goal-package residuals {residuals} "
            f"(merge-type pooled band / cross-fitted (c)) -- re-run "
            f"with --accept-residuals to freeze with these recorded "
            f"as known_residuals")
    print(f"formal criteria PASS pooled at both horizons ((b) total, "
          f"(a) collisions, (d) speed); residuals: "
          f"{residuals if residuals else 'NONE'}")
    for H in Q.HORIZONS:
        g = gate["gate_v2"][f"H{H}"]
        gp = gate["goal_package"][f"H{H}"]
        print(f"  H={H}: lc {g['lc_pred']}/{g['lc_true']} "
              f"({100*g['lc_deviation']:+.1f}%) | by-type dev merge "
              f"{100*gp['by_type']['pooled']['dev_merge']:+.1f}% disc "
              f"{100*gp['by_type']['pooled']['dev_disc']:+.1f}% | "
              f"c: ttc3 {g['ttc3_cov_frac']} dphys "
              f"{g['delay_phys_cov_frac']} crps {g['crps_phys']} | "
              f"speed {g['speed_rmse']} <= "
              f"{g['d_per_horizon_threshold']} | coll "
              f"{g['collisions_per_1000']}")

    # ---- candidate params JSON (extracted, self-describing)
    fit = Q.N.load_json(fit_path)
    cand = dict(
        part="3q", iteration=k,
        frozen_candidate=True,
        variant=fit["variant"],
        engine_kind="idm_merge",
        engine_flags=dict(
            lc_scale_disc="part 3o: discretionary 0<->1 hazard scale "
                          "(default: lc_scale)",
            merge_accept_prob="part 3o: thinning of gap-accepted M2 "
                              "merges (default 1.0 = shipped "
                              "deterministic)",
            merge_cooldown_ticks="part 3q: hysteresis after a "
                                 "prob-thinned M2 attempt (default 0 = "
                                 "shipped)",
            merge_early_d="part 3q: d boundary of the EARLY merge zone "
                          "(default 0.0 = empty early zone)",
            merge_early_prob="part 3q: acceptance prob for EARLY zone "
                             "candidates (default None = merge_accept_"
                             "prob everywhere)"),
        base_from="p3o_iter2_member0 (the frozen p3o candidate)",
        fitted_on=fit["fitted_on"],
        declared=fit["declared"],
        winners=fit["winners"],
        members=fit["members"],
        provenance=dict(
            fit_json=fit_path.name,
            gate_json=gate_path.name,
            diag_json="diag_p3q.json",
            split="17 TRAIN roots, k=5 folds by root; parameters "
                  "fitted on folds f1..f4 (13 roots) ONLY; fold f0 "
                  "out-of-sample; spread cross-fitted per fold in the "
                  "gate; NO validation look; holdout blocked",
            goal_package_pass=gate.get("goal_package_pass", False),
            known_residuals=residuals,
            residual_note=("merge-type pooled band and the cross-fitted "
                           "(c) coverage/CRPS where listed: recorded "
                           "with the per-fold and bias/width breakdown "
                           "in the gate JSON; the owner decides whether "
                           "the single validation look proceeds")),
    )
    cand_path = Q.RES / f"candidate_p3q_iter{k}.json"
    cand_path.write_text(json.dumps(cand, indent=1, default=str),
                         encoding="utf-8")

    # ---- full-split spread factors (17 roots x 7 t0, the larger
    # TRAIN sample the brief asks for) + bias/width decomposition
    members = fit["members"]
    rows = []
    for mi in range(len(members)):
        p = Q.RAW_DIR / f"raw_p3q_iter{k}_m{mi}.jsonl"
        expect = 2 * len(Q.GATE_T0S) * 17
        if not Q.member_file_complete(p, expect):
            raise SystemExit(f"FREEZE REFUSED: {p.name} incomplete")
        rows += [r for r in Q.read_rows(p) if r["root"] in
                 set(Q.train17())]
    wins = len({(r["root"], int(r["t0"])) for r in rows})
    print(f"\nFULL-split spread calibration: {wins} windows x "
          f"{len(members)} members (17 TRAIN roots; the larger sample "
          f"vs the 28-window p3o calibration); L3 spread math; target "
          f"0.80")
    calib = dict(s_out={}, breakdown={}, meta=dict(
        part="3q", task="freeze", iteration=k,
        calibration_roots=list(Q.train17()), windows=wins,
        members=len(members),
        source_raw=[f"raw_p3q_iter{k}_m{mi}.jsonl"
                    for mi in range(len(members))],
        gate_json=gate_path.name, candidate_json=cand_path.name,
        holdout_accessed=False, simulation=False,
        validation_used=False, validation_loaded=False,
        note=("frozen factors for a POSSIBLE single validation look "
              "(NOT run in part 3q), fitted on the FULL 17-root TRAIN "
              "split; mechanism A = mean-centred widening, mean exactly "
              "invariant; s_out = smallest grid factor reaching 80 % "
              "coverage")))
    for name, pk, tk in Q.OUTPUTS:
        for H in Q.HORIZONS:
            entry = spread_block(rows, name, pk, tk, H)
            calib["s_out"].setdefault(name, {})[str(H)] = entry["s_out"]
            calib["breakdown"][f"{name}@H{H}"] = entry
            print(f"  {name}@H{H}: raw_cov={entry['raw_coverage']} "
                  f"({entry['raw_breakdown']['below']}/"
                  f"{entry['raw_breakdown']['in']}/"
                  f"{entry['raw_breakdown']['above']}) s_out="
                  f"{entry['s_out']} cov_widened="
                  f"{entry.get('coverage_mean_centred')} "
                  f"(bias_mean {entry['raw_breakdown']['bias_mean']}, "
                  f"width_med {entry['raw_breakdown']['width_med']} vs "
                  f"truth_iqr {entry['raw_breakdown']['truth_iqr']})")

    # ---- FREEZE
    side_cand = N.write_frozen(
        cand_path, "the part 3q member parameters (frozen candidate); "
                   "any validation look must verify this hash first")
    cal_json = Q.RES / "calibration_p3q.json"
    cal_json.write_text(json.dumps(calib, indent=1, default=str),
                        encoding="utf-8")
    side_cal = N.write_frozen(
        cal_json, "part 3q spread factors (FULL 17-root TRAIN split) "
                  "+ bias/width breakdown; the validation look is NOT "
                  "run")
    print(f"\nFROZEN {cand_path.name}: sha256={side_cand['sha256']}")
    print(f"        frozen_utc={side_cand['frozen_utc']}")
    print(f"FROZEN {cal_json.name}: sha256={side_cal['sha256']}")
    print(f"        frozen_utc={side_cal['frozen_utc']}")
    print("\nDONE part 3q freeze.  The single validation look is NOT "
          "part of part 3q.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
