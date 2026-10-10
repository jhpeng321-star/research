# REGISTRATION (analysis, zero simulation): Part A v2 general, evaluation (a) A2 — model error from the TRUE fleet-native initial state (chunk-4), after the LC-churn fix

- **ID:** WD-01-PA2G-A2.
- **Status:** registered 2026-09-30 before any run. Authority: the
  WD-01 Part A v2 general task brief (owner-delegated); reuses
  `REGISTRATION_PA2G_EVAL_A.md` with the truth source changed, as that
  report's "Next step" specified.
- **Type:** ANALYSIS on existing logs. No SUMO/TraCI. TRAIN only.

## Question

After the correctness fix of the discretionary lane-change churn
(`_neighbors` leader/follower swap + decision-acceleration clipping;
nominal stability guards retired to OFF), how large is the Part A v2
general MODEL error against TRUE FLEET-NATIVE campaign truth, with the
CAV path wired (is_cav flags + the native command path)? And does the
churn collapse hold on the wider panel?

## Data

- Truth: `E:/research-dense/runs/campaign_stage1_chunk4/<world>__native_base/run/physics_steps.jsonl`
  (chunk-4 native_base branch = ALL CAVs native from t = 0, speedMode 27).
- Plan: `output/campaign_stage1_v3/results_plan_chunk4/WORLD_PLAN.json`
  (63 TRAIN worlds; splits from the plan, read BEFORE any run file).
- Only runs with `run/summary.json` + `run/physics_steps.jsonl` whose
  `RUN_RECEIPT.json.exit_status == 'complete'` and whose physics log
  reaches the run duration are usable (61 of 63; w8703m2600p40t540 and
  w8706m2400p40t360 lack run/summary.json).
- CAV identity: the physics rows' own `cav_ids` field. insertions.jsonl
  and dispatcher_rows.jsonl are NEVER read.
- CALIBRATION/HOLDOUT worlds are never opened (asserted in code,
  `assert_train_only` at close-out; every opened file recorded).

## Protocol

- **Worlds:** a deterministic round-robin over the (demand, penetration)
  cells of the usable TRAIN worlds, 2 per cell where available
  (`--stratified 32`), workers <= 4.
- **Windows:** start every 30 s over (t0, t0+540], t0 from the plan;
  H ∈ {10, 30, 60, 120} s kept when the log covers w+H.
- **Initial state:** the logged vehicles inside the default RSU recon
  zone at the window start.
- **CAVs:** zone vehicles present in the run's `cav_ids` carry
  `is_cav=True`; the engine runs with the native command path
  (`cavctrl.CavControllerLoop`, all-native actions → no commands; the
  partA_ctrl native path engaged every control tick).
- **Engine:** the FIXED v2g engine (post churn fix). Nominal parameter
  set `params_nominal.json`: the textbook values are unchanged from
  PA2G-A (IDM T 1.5, a 1.0, b 1.5, s0 2, δ 4; MOBIL p 0.5, b_safe 4,
  threshold 0.1, keep-right bias 0.3); the declared non-textbook
  stability guards are retired to OFF (cooldown_s 0, lc_min_speed 0,
  ego_gain_floor 0) per the owner's design-deviation rule, flagged in
  DIAGNOSIS.md and the report.
- **Metrics per H** over the tracked vehicles: trajectory RMSE (s on the
  shared d axis, v), lane-change counts by type vs truth, window delay
  error (mean SIGNED and mean ABSOLUTE), TTC<3 exposure error,
  hard-brake error, predicted collisions per 1000 vehicle-steps.
- **Missing-arrivals curve** and `n_free_road_assumed` stats: as
  registered in PA2G-A.
- **Comparators:** the v1 p3o replica (frozen iter-2 knobs, same
  declared deviations) and constant-speed persistence.
- **Descriptive T/a sweep** (T ∈ {1.0, 1.5, 2.0} × a ∈ {0.7, 1.0, 1.5},
  H = 60, every 4th world): labelled, NO parameter selected.
- **Labelled variant (design-deviation comparison, never headline):**
  `v2g_guards_on` = the fixed engine with the PA2G-A-era guards
  re-enabled (cooldown_s 2.0, lc_min_speed_mps 1.0), every 4th world.
- **Labelled fix-attribution run (cheap):** the fixed engine on the
  PA2G-A chunk-2/3 truth (4 PA2G-A worlds, v2g only) so the A-vs-A2
  side-by-side can separate the fix effect from the truth change.

## Rules

- **No tuning.** Textbook parameters fixed; errors are reported as model
  error. The T/a sweep is descriptive only.
- **Records.** Raw rows saved BEFORE aggregation
  (`results/eval_a2/raw_rows.jsonl`); the outcome appended to
  `LEDGER_DRAFT_PA2G_A2.json`; no shared records edited by this task.
