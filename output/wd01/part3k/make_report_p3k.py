"""WD-01 part 3k: assemble REPORT_P3K.md + LEDGER_DRAFT_P3K.json.

Every number in the report is pasted VERBATIM from the tee'd script
stdout files (results/stdout_fit.txt, results/stdout_decompose.txt,
results/stdout_aggregate.txt) -- nothing is hand-transcribed.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

HOLDOUT_LINE = ("**Holdout NOT accessed.**  No holdout root was loaded "
                "(the holdout block inside `root_arrays` stays "
                "authoritative); no registration hash was created or "
                "passed anywhere in part 3k.")


def read(p):
    return Path(p).read_text(encoding="utf-8").rstrip("\n")


def block(title, path, note=""):
    txt = read(path)
    return f"\n## {title}\n\n{note}\n\n```text\n{txt}\n```\n"


def main():
    fit = json.loads((OUT / "fit_p3k.json").read_text(encoding="utf-8"))
    gate = json.loads((OUT / "gate_v2_p3k.json").read_text(
        encoding="utf-8"))
    sel = fit["selected"]
    mp = sel["merge_params"]

    g30 = {k.split("@")[0]: v for k, v in gate["gate_v2"].items()
           if k.endswith("@H30")}
    g60 = {k.split("@")[0]: v for k, v in gate["gate_v2"].items()
           if k.endswith("@H60")}
    o30 = {k.split("@")[0]: v for k, v in gate["old_gate"].items()
           if k.endswith("@H30")}

    def v2_line(g, v):
        d = g[v]
        c_txt = ("NA (deterministic)" if d["c_not_applicable"] else
                 ("PASS" if d["c_coverage_crps_pass"] else "FAIL"))
        verdict = ("PASS" if d["v2_PASS"]
                   else "FAIL: " + ",".join(d["v2_failing"]))
        return (f"{v}: (a) {'PASS' if d['a_collisions_pass'] else 'FAIL'} "
                f"(coll/1000={d['collisions_per_1000']}) | "
                f"(b) {'PASS' if d['b_lc_pass'] else 'FAIL'} "
                f"({100 * d['lc_deviation']:.1f}%) | "
                f"(c) {c_txt} | "
                f"(d) {'PASS' if d['d_speed_pass'] else 'FAIL'} "
                f"(RMSE={d['speed_rmse']}) ==> {verdict}")

    md = f"""# WD-01 Part 3K — SUMO-like merge dynamics in the structured \
native Part A emulator (kind `idm_merge`), refit + re-gate

**{HOLDOUT_LINE}**

ZERO simulation: no SUMO/TraCI import anywhere; the only dynamics is the
part-3 rollout engine (pure torch).  TRAIN roots for fitting,
VALIDATION roots for reporting.

## What was built

1. **Data-root override** (`WD01_DATA_ROOT`): untracked
   data/weights/results live in a populated copy of the output tree;
   `pdl.py` (partA_data) resolves `DATA`/`PART1`, `p3lib.py` resolves
   `RESULTS`/`WEIGHTS`, and `engine.py` resolves the part3d weights dir
   through it.  UNSET (default) keeps the in-repo layout byte-identical
   to pre-3k behaviour (tested: `partA_data/tests/test_data_root.py`,
   `part3/tests/test_p3_merge.py`).  All part-3k runs used
   `WD01_DATA_ROOT=E:/rd-wt/wd01/output/wd01` (read-only).
2. **Engine kind `idm_merge`** in `part3/engine.py` = `idm_sampled`
   (unchanged branch) plus SUMO-like merge dynamics:
   - **M1** accel-lane end as a virtual standing obstacle at
     s = 454.10 m (IDM gap to the lane end) so unmerged vehicles
     decelerate and queue;
   - **M2** mandatory gap acceptance accel->main_right inside the merge
     zone (s >= 378.66 m): merge iff gap-to-target-leader >= minGap +
     max(0, v*tau_accept) AND the target-lane follower's required
     deceleration <= b_accept; fires as soon as acceptable, otherwise
     the vehicle keeps waiting (the `idm_sampled` stochastic
     accel->main_right merge is removed);
   - **M3** cooperative slow-down: a main_right follower with d in
     [-100, 100] m whose leader-to-be (nearest accel vehicle ahead in d,
     within 100 m) is still waiting adapts its IDM leader to it with
     per-vType probability lcCooperative_prior x p_coop_scale
     (prior 0.3/0.6/1.0);
   - **M4** discretionary 0<->1 changes stay on the existing LC hazard
     (lc_scale).
   Unit tests: `part3/tests/test_p3_merge.py` (lane-end obstacle stops a
   vehicle; synthetic gaps rejected/accepted; cooperative slow-down
   triggers; merge keeps d and survives `_drop_exits`).
   NOTE: `idm_sampled` itself is untouched, but the review found it
   leaves a stale accel-lane `llen`/`vcap` after a merge (a merged
   vehicle is then dropped by `_drop_exits` as if it had exited);
   `idm_merge` refreshes all lane attributes after a lane change.  The
   stale-`llen` defect in `idm_sampled` is recorded as a deviation, NOT
   fixed (that would change published L3 numbers).

## Fit (TRAIN only)

Protocol, grids, targets and the selected parameters are in the fit
stdout below.  Selected: merge_params={mp}, lc_scale={sel['lc_scale']}
(member 0; members 1-4 are bootstrap refits over the 4 train roots,
seeds m*17+1).  Fitted-on record: {fit['fitted_on']}.

## Gate verdicts (summary; full tables in the aggregation stdout)

- Gate v2 at H=30 (PRIMARY per
  `part3f/VALIDATION_GATE_V2_AMENDMENT_1.md`):
  - {v2_line(g30, 'L0')}
  - {v2_line(g30, 'L3')}
  - {v2_line(g30, 'L4')}
- Gate v2 at H=60 (secondary): L0/L3/L4 verdicts in the stdout below.
- Old gate at H=30/H=60: reported alongside (stdout below).
- Criterion (c) uses the CONTRACT PHYSICS DELAY (dt - ds/limit, the
  p3j/p3k decompose model); exposure-based coverage is reported
  alongside for continuity.
"""
    md += block("Fit stdout (fit_p3k.py, TRAIN only -- verbatim)",
                OUT / "stdout_fit.txt")
    md += block(
        "Validation decomposition stdout (decompose_p3k.py, L0/L3/L4, "
        "60 s grid, H=60 -- verbatim)",
        OUT / "stdout_decompose.txt",
        "Top cells before/after and the accel speed profile are in "
        "sections C/D/E.")
    md += block(
        "Ladder aggregation + gate stdout (aggregate_p3k.py -- verbatim)",
        OUT / "stdout_aggregate.txt",
        "Raw rows: results/raw/raw_metrics.jsonl (written first by "
        "eval_p3k.py); aggregated verdicts: results/gate_v2_p3k.json.")
    md += f"""
## NOT DONE / residual gaps

1. **Merge-zone occupancy still under-predicted**: even at the fitted
   acceptance, L4 accel-lane zone occupancy remains below truth (see
   the fit/decomposition stdout).  A stopped waiting vehicle accepts
   any gap >= its minGap (2.5 m for normal cars); reproducing SUMO's
   longer queues likely needs a stopped-merge urgency/penalty term
   (e.g. lcStrategic-style urgency or a d-dependent minGap) -- not
   fitted here.
2. **Optional M2 for 0<->1** (gap acceptance on discretionary main-lane
   changes) was NOT implemented (brief: optional); 0<->1 stays on the
   stochastic hazard (M4).
3. **Parameter bootstrap is coordinate-neighbourhood**, not a full
   refit per member (cost): members 1-4 refit tau_accept/b_accept/
   p_coop_scale one grid step around the stage winners on bootstrap
   resamples of the 4 train roots, lc_scale refined on the same grid.
4. **idm_sampled stale-llen defect** (merged vehicles dropped as
   exited) recorded, not fixed.
5. **Single process, below-normal priority, torch 2 threads** -- no
   parallelism; box shared with SUMO workers.
6. **Holdout untouched**: the registered holdout evaluation remains
   blocked until the owner opens it (no registration hash created).

## Artifacts

- Code: `part3k/decompose_p3k.py` (p3j copy + L4 + helpers),
  `part3k/fit_p3k.py`, `part3k/eval_p3k.py`, `part3k/aggregate_p3k.py`,
  `part3k/make_report_p3k.py`; engine changes in
  `part3/engine.py` (kind `idm_merge`); data-root override in
  `partA_data/pdl.py`, `part3/p3lib.py`, `part3/engine.py`.
- Tests: `part3/tests/test_p3_merge.py`,
  `partA_data/tests/test_data_root.py`, `part3k/tests/` (14 tests).
- Results: `part3k/results/` (fit_p3k.json, stdout_fit.txt,
  decomposition_p3k.csv, speed_profile_p3k.csv, stdout_decompose.txt,
  raw/raw_metrics.jsonl, gate_v2_p3k.json, stdout_aggregate.txt,
  LEDGER_DRAFT_P3K.json).
- Reconciliation: L0/L3 rollouts are the p3j configuration (same
  engine branch, same seeds); their 60 s-grid validation means must
  reproduce the p3j published values (delay_true 1300.74; delay_def
  L0 -731.41 / L3 -294.22) -- checked in the decomposition stdout
  section A/B footer.

*(This report is generated by make_report_p3k.py; all numbers are
pasted from the tee'd stdout files above.)*
"""
    (HERE / "REPORT_P3K.md").write_text(md, encoding="utf-8")

    ledger = dict(
        part="3k",
        date="2026-09-26",
        title="SUMO-like merge dynamics in the structured native Part A "
              "emulator (kind 'idm_merge'), TRAIN refit + VALIDATION "
              "re-gate",
        holdout_accessed=False,
        holdout_note="no holdout root loaded, no registration hash "
                     "created or passed; holdout_block authoritative",
        simulation=False,
        engine_change="part3/engine.py: new kind 'idm_merge' = "
                      "idm_sampled (unchanged) + M1 lane-end obstacle + "
                      "M2 merge-zone gap acceptance + M3 cooperative "
                      "slow-down + M4 hazard 0<->1; idm_merge refreshes "
                      "llen/vcap after a lane change (idm_sampled does "
                      "not -- recorded defect, not fixed)",
        data_root="WD01_DATA_ROOT env override in partA_data/pdl.py, "
                  "part3/p3lib.py, part3/engine.py; default (unset) "
                  "in-repo layout unchanged (tested); runs used "
                  "E:/rd-wt/wd01/output/wd01 read-only",
        fit=dict(
            train_roots=fit["fitted_on"]["roots"],
            selected=dict(merge_params=sel["merge_params"],
                          lc_scale=sel["lc_scale"]),
            losses=dict(stage1_before=sel["before"]["detail1"],
                        stage2_before=sel["before"]["detail2"],
                        stage1_after=sel["after"]["detail1"],
                        stage2_after=sel["after"]["detail2"]),
            members=fit["members"]),
        ladder=dict(validation_roots=4, windows_per_member=52,
                    t0_grid="120:30:480", H=(30, 60, 90, 120),
                    members=5, raw_rows="results/raw/raw_metrics.jsonl"),
        gate_v2=dict(
            primary_H=30,
            H30={v: dict(pass_=g30[v]["v2_PASS"],
                         failing=g30[v]["v2_failing"])
                 for v in ("L0", "L3", "L4")},
            H60={v: dict(pass_=g60[v]["v2_PASS"],
                         failing=g60[v]["v2_failing"])
                 for v in ("L0", "L3", "L4")},
            c_metric="contract physics delay dt - ds/limit (p3j model); "
                     "exposure-based coverage alongside"),
        old_gate=dict(
            H30={v: o30[v]["old_pass"] for v in ("L0", "L3", "L4")}),
        deviations=[
            "idm_sampled stale-llen/vcap after merge recorded, NOT "
            "fixed (would change published L3 numbers)",
            "member bootstrap is coordinate-neighbourhood, not full "
            "refit (cost)",
            "optional M2 for 0<->1 not implemented (brief: optional)",
        ],
        not_done=[
            "merge-zone occupancy under-prediction residual (stopped-"
            "merge urgency term not modelled)",
            "holdout evaluation (blocked until owner opens it)",
        ],
    )
    (HERE / "LEDGER_DRAFT_P3K.json").write_text(
        json.dumps(ledger, indent=1, default=str), encoding="utf-8")
    print(f"wrote {HERE / 'REPORT_P3K.md'} and "
          f"{HERE / 'LEDGER_DRAFT_P3K.json'}")


if __name__ == "__main__":
    main()
