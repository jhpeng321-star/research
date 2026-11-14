# REPORT — WD-01-PA2G-A: Part A v2 general, evaluation (a) — model error from the TRUE initial state

- **Analysis ID:** WD-01-PA2G-A (registered before any run:
  `REGISTRATION_PA2G_EVAL_A.md`; design `PARTA_V2_GENERAL_DESIGN.md`;
  contract `PREDICT_CONTRACT_v2_1.md`).
- **Status:** eval COMPLETED 2026-09-29 (wall 5967 s). This report and the
  committed `aggregate_eval_a.py` are ZERO-SIMULATION close-out over the
  saved rows. The eval was NOT re-run.
- **Executor branch:** `zcode/wd01pa3`, eval run from HEAD 3065ef2.

## TRUTH CAVEAT (read before quoting any number)

**The truth runs used here are NOT fleet-native.** The truth is the
chunk-2/chunk-3 campaign `<world>__native` branch
(`runs/campaign_stage1_chunk2|chunk3/<world>__native/run/physics_steps.jsonl`).
In those branches the CAVs followed **random per-CAV switching schedules**
(a damping/native mix decided per CAV); the authoritative per-CAV actions
live in each run's `switch_schedule.json` / `suffix_commands.jsonl`
(branch CAV internals: `dispatcher_rows.jsonl`, `suffix_commands.jsonl`,
`switch_schedule.json`). "Native" names the branch, not a fleet-native
policy.

**How the eval treated CAVs:** the v2g engine was initialised from the
zone state WITHOUT `is_cav` flags (`tracked_state()` copies only
lane/s/v/a), so `engine.cav` was all-False and the entire CAV command
path (`cavctrl.py`: damping / PI external-U / wall-v2 emulators) never
engaged — **every vehicle, CAV or not, was simulated by the generic
IDM+MOBIL rules.** The v1 p3o replica likewise did not replay the
switching schedules (humans got prior-drawn vTypes, CAVs their logged
types); const-speed persistence is behaviour-agnostic.

**Consequence:** part of every reported error is CAV-behaviour mismatch
(model runs generic rules; truth CAVs executed per-CAV damping/native
commands), not human-driver model error. This **confounds the error
attribution**. A re-run on the chunk-4 `__native_base` truth (ALL CAVs
native from t = 0, speedMode 27) is **pending and supersedes this
analysis for the headline**. Until then, treat the v2g numbers as an
upper bound on model error under a CAV-behaviour mismatch.

## Data and raw-row counts

- Source: `output/wd01/partA_v2g/results/eval_a/raw_rows.jsonl`,
  sha256 `F06FBEDE9B5BEF4CDFC3B3B15505D4B022300BADCEED71EDDD5714D26A476EA2`.
- **6402 raw lines** = 5826 main rows + 576 sweep-pass duplicates:
  - main rows: 25 TRAIN worlds (of 98; `world_step=4`, chunks 2+3) × 18
    windows × {v2g, const_speed, v1_p3o_replica} × H ∈ {10,30,60,120},
    with H120 present only when the log covers w+120 → 400 instead of
    450 per model;
  - the T/a sweep passes (every 8th world → 4 worlds × 18 windows × 8
    combos, H=60) re-emitted 576 const-speed rows that are exact
    recomputations of the same window's main-pass row. The committed
    aggregator **verifies value-identity (assertion, 0 mismatches) and
    drops them**; `v2g_sweep` rows (576) share their (world, w, H,
    model) key across the eight T/a combos and are distinguished by
    `T_s`/`a_max`.
- **Correction vs `summary.json`:** the in-run aggregator mixed those
  576 duplicates into `const_speed|H60` (n 1026, s_RMSE 129.0,
  delay_err 123.4). Deduplicated values are **n 450, s_RMSE 136.4,
  delay_err mean +93.3 / MAE 162.4**; the committed-script numbers stand.
- Run provenance: `summary.json` (sha256
  `A997C4BBC49B686855BC1248AA2C0503DCAFC59BAD3CE856F55A4C15DF7C5338`);
  param_set_id `6359f5…4b9`; `assert_train_only` passed; every opened run
  file is listed in `summary.json` (`opened_run_files`); CAL/VALIDATION/
  HOLDOUT were never opened.

All tables below are computed from the deduplicated 5826 rows by the
committed `output/wd01/partA_v2g/aggregate_eval_a.py`
(outputs: `results/eval_a/tables.json`, `tables.md`).

## Table 1 — per model × H (errors are pred − truth; mean SIGNED and mean ABSOLUTE both shown)

delay/TTC/hard are per-window aggregates over tracked vehicles (delay in
vehicle-seconds, TTC<3 exposure in pair-seconds, hard-brake in
vehicle-seconds, a < −3 m/s²); s_RMSE is on the shared downstream-d axis.

| model\|H | n | s_RMSE | v_RMSE | delay_err(mean) | delay_err(MAE) | ttc3_err(mean) | hard_err(mean) | LC merge p/t | LC disc p/t | coll/1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| const_speed\|H10 | 450 | 16.1 | 3.84 | −9.4 | 11.3 | 30.0 | −8.6 | 0/485 | 0/243 | 0.00 |
| const_speed\|H30 | 450 | 87.7 | 7.44 | −26.4 | 59.0 | 161.5 | −16.1 | 0/1428 | 0/616 | 0.00 |
| const_speed\|H60 | 450 | 136.4 | 8.43 | 93.3 | 162.4 | 312.6 | −16.5 | 0/1866 | 0/855 | 0.00 |
| const_speed\|H120 | 400 | 156.1 | 9.02 | 734.3 | 752.0 | 489.3 | −17.3 | 0/1714 | 0/949 | 0.00 |
| v1_p3o_replica\|H10 | 450 | 11.6 | 2.88 | 12.3 | 12.6 | 1.3 | 7.3 | 1160/485 | 856/243 | 0.30 |
| v1_p3o_replica\|H30 | 450 | 62.4 | 5.58 | 30.9 | 40.4 | 1.9 | 15.3 | 1905/1428 | 2362/616 | 1.21 |
| v1_p3o_replica\|H60 | 450 | 103.6 | 6.81 | −69.1 | 117.4 | 2.9 | 21.0 | 1908/1866 | 4119/855 | 6.29 |
| v1_p3o_replica\|H120 | 400 | 137.5 | 8.28 | −600.2 | 628.3 | 3.6 | 27.0 | 1714/1714 | 4928/949 | 19.53 |
| v2g\|H10 | 450 | 28.2 | 6.31 | 48.9 | 48.9 | 7.8 | 30.8 | 1050/485 | 68614/243 | 13.92 |
| v2g\|H30 | 450 | 119.2 | 9.26 | 217.9 | 217.9 | 8.9 | 61.1 | 1881/1428 | 232079/616 | 5.83 |
| v2g\|H60 | 450 | 151.3 | 9.19 | 396.6 | 396.6 | 12.1 | 92.8 | 1908/1866 | 407009/855 | 3.26 |
| v2g\|H120 | 400 | 183.6 | 10.11 | 49.8 | 368.0 | 13.1 | 105.8 | 1714/1714 | 486428/949 | 2.21 |

(LC p/t = predicted/true counts summed over windows; disc = disc_left +
disc_right.)

## Table 2 — missing-arrivals curve (truth delay, v2g windows)

| H | n | delay_true tracked | delay_true all | arrivals/window | untracked delay share |
|---|---|---|---|---|---|
| H10 | 450 | 426 | 537 | 8.2 | 0.207 |
| H30 | 450 | 1249 | 1632 | 24.7 | 0.234 |
| H60 | 450 | 2256 | 3331 | 49.4 | 0.323 |
| H120 | 400 | 3462 | 6639 | 98.7 | 0.479 |

Untracked arrivals contribute 21 % of truth delay at H10 and **48 % at
H120**: at the long horizon nearly half the true delay belongs to
vehicles the zone state does not contain. This is the registered
"horizon at which untracked arrivals start to dominate" answer: the
tracked-scope Part A degrades sharply beyond H = 60.

## Table 3 — `n_free_road_assumed` stats (v2g windows)

| H | n | assumed mean | median | max | share windows>0 | truth-flag mean | truth-flag max |
|---|---|---|---|---|---|---|---|
| H10 | 450 | 2.00 | 2 | 2 | 1.00 | 2.06 | 3 |
| H30 | 450 | 2.00 | 2 | 2 | 1.00 | 2.06 | 3 |
| H60 | 450 | 2.00 | 2 | 2 | 1.00 | 2.06 | 3 |
| H120 | 400 | 2.00 | 2 | 2 | 1.00 | 2.07 | 3 |

The predictor-side zone-truncation rule flags exactly 2 tracked vehicles
per window at every H (their leaders lie outside the recon zone; the
model assumes free road for them). The truth-exact rule flags 2.06 on
average (max 3). The optimistic bias is small and constant across H —
it does not explain the H-growth of the errors.

## Table 4 — descriptive T/a sweep at H = 60 (NO parameter selected)

| cell | kind | T | a_max | n | s_RMSE | v_RMSE | delay_err mean |
|---|---|---|---|---|---|---|---|
| T1.0_a0.7 | sweep | 1.0 | 0.7 | 72 | 142.5 | 8.70 | 477.8 |
| T1.0_a1.0 | sweep | 1.0 | 1.0 | 72 | 147.5 | 8.74 | 229.6 |
| T1.0_a1.5 | sweep | 1.0 | 1.5 | 72 | 158.7 | 9.32 | −65.6 |
| T1.5_a0.7 | sweep | 1.5 | 0.7 | 72 | 144.8 | 8.87 | 605.0 |
| T1.5_a1.5 | sweep | 1.5 | 1.5 | 72 | 147.0 | 8.69 | 189.9 |
| T2.0_a0.7 | sweep | 2.0 | 0.7 | 72 | 149.4 | 9.07 | 712.6 |
| T2.0_a1.0 | sweep | 2.0 | 1.0 | 72 | 147.2 | 8.73 | 544.2 |
| T2.0_a1.5 | sweep | 2.0 | 1.5 | 72 | 149.0 | 8.65 | 361.8 |
| T1.5_a1.0 | nominal (registered v2g; NOT selected from the sweep) | 1.5 | 1.0 | 450 | 151.3 | 9.19 | 396.6 |

Descriptive only: trajectory RMSE is insensitive (142–159 m) while the
signed delay error swings from −65.6 (a = 1.5) to +712.6 (a = 0.7,
T = 2.0) — lower desired headway/legacy acceleration mainly rescales
congestion build-up, not position tracking. Nothing was chosen from this
sweep (registration rule).

## Table 5 — breakdowns (v2g)

### by_penetration

| cell | n | s_RMSE | delay_err mean | delay_err MAE |
|---|---|---|---|---|
| 0.02\|H10 | 90 | 26.2 | 49.1 | 49.1 |
| 0.02\|H30 | 90 | 111.5 | 226.6 | 226.6 |
| 0.02\|H60 | 90 | 140.7 | 427.5 | 427.5 |
| 0.02\|H120 | 80 | 174.6 | 46.2 | 346.6 |
| 0.05\|H10 | 90 | 28.4 | 49.4 | 49.4 |
| 0.05\|H30 | 90 | 120.6 | 221.6 | 221.6 |
| 0.05\|H60 | 90 | 152.2 | 400.4 | 400.4 |
| 0.05\|H120 | 80 | 177.3 | 135.3 | 288.2 |
| 0.1\|H10 | 90 | 28.4 | 48.2 | 48.2 |
| 0.1\|H30 | 90 | 123.7 | 225.4 | 225.4 |
| 0.1\|H60 | 90 | 156.1 | 417.6 | 417.6 |
| 0.1\|H120 | 80 | 184.1 | 162.3 | 362.6 |
| 0.2\|H10 | 72 | 27.9 | 45.0 | 45.0 |
| 0.2\|H30 | 72 | 117.6 | 200.1 | 200.1 |
| 0.2\|H60 | 72 | 150.4 | 354.9 | 354.9 |
| 0.2\|H120 | 64 | 184.5 | −42.8 | 400.4 |
| 0.4\|H10 | 108 | 29.9 | 51.6 | 51.6 |
| 0.4\|H30 | 108 | 121.9 | 213.0 | 213.0 |
| 0.4\|H60 | 108 | 156.0 | 377.9 | 377.9 |
| 0.4\|H120 | 96 | 195.3 | −50.6 | 435.0 |

### by_demand

| cell | n | s_RMSE | delay_err mean | delay_err MAE |
|---|---|---|---|---|
| 2400\|H10 | 54 | 28.1 | 44.6 | 44.6 |
| 2400\|H30 | 54 | 118.0 | 197.7 | 197.7 |
| 2400\|H60 | 54 | 155.2 | 366.7 | 366.7 |
| 2400\|H120 | 48 | 187.1 | 71.0 | 454.6 |
| 2600\|H10 | 126 | 28.9 | 46.8 | 46.8 |
| 2600\|H30 | 126 | 122.1 | 208.6 | 208.6 |
| 2600\|H60 | 126 | 155.6 | 377.6 | 377.6 |
| 2600\|H120 | 112 | 184.4 | 58.0 | 430.9 |
| 2700\|H10 | 270 | 27.9 | 50.8 | 50.8 |
| 2700\|H30 | 270 | 118.2 | 226.2 | 226.2 |
| 2700\|H60 | 270 | 148.5 | 411.4 | 411.4 |
| 2700\|H120 | 240 | 182.5 | 41.7 | 321.2 |

## Table 6 — H120 vs H60 sign diagnostics (v2g delay_err)

| H | n | ≥0 | <0 | mean signed | mean abs |
|---|---|---|---|---|---|
| H60 | 450 | 450 | 0 | 396.6 | 396.6 |
| H120 | 400 | 228 | 172 | 49.8 | 368.0 |

## Why the H120 mean delay error (49.8) is far below H60 (396.6)

This is **sign cancellation, not an improvement**:

1. **Sign cancellation.** At H60 all 450 window errors are positive
   (mean = MAE = 396.6). At H120, 172 of 400 errors are negative: the
   per-penetration split flips — penetration ≥ 0.2 has mean −42.8 (p0.2)
   and −50.6 (p0.4) (the model *under*-predicts delay there), while
   penetration ≤ 0.1 stays positive (+46.2 … +162.3). The signed average
   of a bimodal error is 49.8 while the typical magnitude (MAE) is 368.0
   — comparable to H60's 396.6. Mechanistically, at high penetration the
   truth CAVs execute damping commands (slower than any modelled
   behaviour), pushing truth delay above the prediction; at low
   penetration the model's own LC churn over-produces delay. The two
   regimes cancel in the mean.
2. **Sample change.** n = 400 at H120 vs 450 at H60: 50 windows lack
   120 s of log coverage and drop out of the H120 aggregate (same 400
   for every model), so the two rows are not exactly the same window
   set.

Reported separately per the acceptance criteria: mean signed H60
+396.6 / H120 +49.8; mean absolute H60 396.6 / H120 368.0.

## Honest model-error statement

- The generic v2g rules (one nominal Treiber IDM+MOBIL set, no vType, no
  tuning) are **far from the SUMO truth on every registered metric** at
  every H: d-axis s_RMSE grows 28.2 → 183.6 m and v_RMSE 6.3 → 10.1 m/s
  with H; delay error grows to MAE ≈ 396.6 vehicle-s at H60 before the
  H120 sign cancellation; predicted collisions reach 2.2–13.9 per 1000
  vehicle-steps (truth tolerance in this project is zero teleport, and
  the model never teleports — it collides instead).
- The dominant visible defect is **discretionary lane-change churn**:
  e.g. at H10 the model makes 68,614 disc LCs vs 243 true (and 486,428
  vs 949 at H120), i.e. MOBIL as configured fires far too often at 0.5 s
  ballistic integration, manufacturing congestion the truth does not
  have. This alone inflates delay/TTC/hard-brake errors.
- The comparators bracket it: const-speed persistence wins s_RMSE up to
  H60 (16–136 m) but carries the largest positive delay bias at H120
  (+734.3) because it cannot react; the v1 p3o replica tracks best at
  short H (s_RMSE 11.6 at H10) but drifts to large negative delay error
  (−600.2 at H120) with rising predicted collisions (19.5/1000).
- **Attribution caveat:** per the TRUTH CAVEAT, none of these numbers is
  yet a clean "human-driver model error" — the truth CAVs ran per-CAV
  damping/native switching schedules while the model ran generic rules
  for all vehicles. The pending chunk-4 `__native_base` re-run is
  required before any headline claim.

## Artifacts

| file | sha256 |
|---|---|
| `partA_v2g/results/eval_a/raw_rows.jsonl` | F06FBEDE9B5BEF4CDFC3B3B15505D4B022300BADCEED71EDDD5714D26A476EA2 |
| `partA_v2g/results/eval_a/summary.json` | A997C4BBC49B686855BC1248AA2C0503DCAFC59BAD3CE856F55A4C15DF7C5338 |
| `partA_v2g/results/eval_a/tables.json` | 416B5C75883896C2225AC9DD60E8BE44E45BF111021A2D0024977101078634A7 |
| `partA_v2g/eval_run.out` | see commit (run log: 25/25 worlds, per-model-H print, `DONE WD-01-PA2G-A wall 5967s`) |

Ledger draft: `output/wd01/LEDGER_DRAFT_PA2G_A.json` (to be appended to
the master ledger by the integrating agent; shared records untouched
here).

## Next step

Registered re-run of this analysis on chunk-4 `__native_base` truth
(true fleet-native) after its registration; it supersedes the headline
numbers above.
