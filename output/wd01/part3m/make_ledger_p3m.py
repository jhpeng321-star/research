"""WD-01 part 3m: build LEDGER_DRAFT_P3M.json from the results JSONs
(diagnosis, fit, calibration + freeze sidecar, gate).  Draft record for
the owner; holdout never accessed; ZERO simulation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def load(p):
    return json.loads((RES / p).read_text(encoding="utf-8"))


def top_cells(cells, n=4):
    """Top-n cells by excess (pa - tr) from a serialised cells dict."""
    rows = []
    for key, c in cells.items():
        rows.append((round(c["pa"] - c["tr"], 2), key,
                     dict(pa=round(c["pa"], 2), pm=round(c["pm"], 2),
                          tr=round(c["tr"], 2), na=c["na"], nt=c["nt"])))
    rows.sort(key=lambda r: -r[0])
    return [dict(excess=e, key=k, **vals) for e, k, vals in rows[:n]]


def main():
    diag = load("diag_p3m.json")
    fit = load("fit_p3m.json")
    cal = load("calibration_p3m.json")
    frozen = load("calibration_p3m.FROZEN.sha256")
    gate = load("gate_v2_p3m.json")

    def gate_row(H, variant):
        g = gate["gate_v2"][f"{variant}@H{H}"]
        return dict(pass_=bool(g["v2_PASS"]), failing=list(g["v2_failing"]),
                    ttc3_cov=g.get("ttc3_cov_frac"),
                    delay_phys_cov=g.get("delay_phys_cov_frac"),
                    crps=g.get("crps_phys"),
                    l0_delay_mae=g.get("l0_delay_phys_mae"),
                    speed=g.get("speed_rmse"),
                    lc_dev=g.get("lc_deviation"),
                    coll_1000=g.get("collisions_per_1000"))

    ledger = dict(
        part="3m",
        date="2026-09-27",
        title=("TTC<3 exposure bias fix: TASK A per-side attribution "
               "diagnosis, TRAIN refit with an added TTC<3 target and "
               "end-force/post-merge knobs, recalibrated spread frozen "
               "before validation, gate v2 re-run (L6c)"),
        holdout_accessed=False,
        holdout_note=("no holdout root loaded, no registration hash "
                      "created or passed; holdout_block authoritative"),
        simulation=False,
        engine_change=(
            "part3/engine.py, all default-neutral (shipped behaviour "
            "byte-identical, pinned by "
            "tests/test_p3m_engine.py::test_default_params_reproduce_"
            "p3l_raw_row): (1) run(n_ticks=None) early-stop (metrics at "
            "H <= n_ticks*DT identical to the full roll); (2) "
            "vtype_map instrumentation (vid -> sampled vType; the live "
            "array is re-indexed by exits); (3) merge_params "
            "end_force_s=0.0 (near-lane-end M2 acceptance relaxation, "
            "end_force_params pure helper); (4) merge_params "
            "v_merge_blend=0.0 (post-merge speed blend toward the "
            "target-lane local mean, local_target_speed pure helper). "
            "idm_sampled untouched"),
        data_root=("runs used WD01_DATA_ROOT=E:/rd-wt/wd01/output/wd01 "
                   "read-only; never written"),
        diagnosis=dict(
            K=diag["meta"]["K"], t0s=diag["meta"]["t0s"],
            windows_per_split=140,
            views="pa = pred all (frozen metric), pm = matched vehicles "
                  "with predicted states, tr = matched vehicles with "
                  "true states; composition = pa-pm, behaviour = pm-tr",
            train=diag["train"]["totals"],
            validation=diag["validation"]["totals"],
            parity="pa/tr == window_metrics pred_/true_ttc3_exposure on "
                   "every window (asserted in diag_p3m.py)",
            train_top={k: top_cells(diag["train"]["cells"][k])
                       for k in ("ev", "seg", "lane", "ttc", "vt",
                                 "dawdle")},
            validation_top={k: top_cells(diag["validation"]["cells"][k])
                            for k in ("ev", "seg", "lane", "ttc", "vt",
                                      "dawdle")},
            verdict=("the excess is BEHAVIOUR, not vehicle-set "
                     "composition (composition exactly 0.0): (1) "
                     "stop-and-go queueing of unmerged accel vehicles "
                     "DOWNSTREAM (d<0) at the lane-end obstacle "
                     "(pure_cf, lane 2); (2) merge-created transients "
                     "the truth never shows (post_merge: TRAIN pred "
                     "122.5 vs truth 0.0); (3) mild closings, TTC in "
                     "[1.5,3) (truth has almost none below 1.5 s); "
                     "(4) sampled cautious vType 10-50x over the truth "
                     "exposure; (5) truth exposure concentrates at the "
                     "merge_edge/rsu_window approach, where the "
                     "emulator UNDER-predicts"),
        ),
        fit=dict(
            fitted_on=fit["fitted_on"],
            added_target=fit["added_target"],
            selected=fit["selected"],
            grids=fit["grids"],
            members=fit["members"],
        ),
        calibration=dict(
            roots=cal["meta"]["calibration_roots"],
            windows=cal["meta"]["windows"],
            members=cal["meta"]["members"],
            s_out=cal["s_out"],
            coverage=cal["coverage"],
            frozen_sha256=frozen["sha256"],
            frozen_utc=frozen["frozen_utc"],
            freeze_order="calibrated and frozen on the TRAIN-internal "
                         "split BEFORE any validation row was written "
                         "(eval_p3m.py freeze gate; re-checked by "
                         "aggregate_p3m.py)",
        ),
        ladder=dict(
            validation_roots=4, windows_per_member=52,
            t0_grid="120:30:480", H=[30, 60], members=5,
            rungs=("L0/L4/L5/L5c reused verbatim from part 3l rows "
                   "(results/raw/raw_p3l_metrics.jsonl); L6 = part 3m "
                   "refit members (fresh rows, "
                   "results/raw/raw_metrics_p3m.jsonl); L6c = L6 + the "
                   "frozen part 3m recalibration"),
        ),
        gate_v2=dict(
            primary_H=30,
            d_binding="per-horizon 1.05 x L0@same-H; fixed 5.2185 "
                      "alongside (part 3l convention, verdict functions "
                      "reused from aggregate_p3l)",
            H30={v: gate_row(30, v) for v in ("L0", "L4", "L5", "L5c",
                                              "L6", "L6c")},
            H60={v: gate_row(60, v) for v in ("L0", "L4", "L5", "L5c",
                                              "L6", "L6c")},
            ttc3_breakdown={f"{v}@H{h}": {
                k: gate["coverage"][f"{v}@H{h}"][k]
                for k in ("ttc3_cov_frac", "ttc3_below_frac",
                          "ttc3_above_frac", "ttc3_below_truth0",
                          "ttc3_above_truth0")}
                for v in ("L5c", "L6c") for h in (30, 60)},
        ),
        old_gate={f"H{h}": {v: bool(gate["old_gate"][f"{v}@H{h}"]["old_pass"])
                            for v in ("L0", "L4", "L5", "L5c", "L6",
                                      "L6c")}
                  for h in (30, 60)},
        deviations=[
            "b_accept grid low end 1.5 dropped (p3l-measured cliff: "
            "stage-1 loss 8.1-8.8 vs 0.92-1.57 at b=3.0; the fix "
            "direction needs the ramp to discharge more, not less); "
            "extended up to 4.0/5.0 per the brief",
            "member ensemble keeps the p3l wider-draw mechanism "
            "(bootstrap resample x softmax T=0.15 lines x dawdle "
            "width), now over the extended coordinate grids",
            "end_force_s relaxation thresholds (tau 0.5 / b 6.0 inside "
            "the zone) are fixed knob constants, not fitted per-vType",
        ],
        not_done=[
            "gate v2 (c) ttc3 coverage may still fall short on "
            "validation (see gate_v2_p3m.json; the calibration split "
            "predicted 82-100 %)",
            "sampled-vType posterior re-fit (diagnosis shows the "
            "cautious draw over-exposes; the vType mixture is the "
            "part 3d posterior, untouched here)",
            "idm_sampled stale-llen defect (p3k item) recorded, not "
            "fixed",
            "holdout evaluation (blocked until the owner opens it)",
        ],
    )
    out = HERE / "LEDGER_DRAFT_P3M.json"
    out.write_text(json.dumps(ledger, indent=1, default=str),
                   encoding="utf-8")
    print(f"wrote {out}")
    if __name__ == "__main__":
        sys.stdout.flush()


if __name__ == "__main__":
    main()
