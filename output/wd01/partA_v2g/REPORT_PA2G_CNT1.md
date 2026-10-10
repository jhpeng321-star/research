# REPORT: WD-01-PA2G-CNT1 — Part A v2g + entrance-counter arrivals (v2g-CNT)

**Type:** TRAIN-only, zero-SUMO SENSITIVITY analysis (SIDE RESULT).
**Label (mandatory):** sensitivity; requires counter hardware; does not
change the CAV-strict headline.
**Registration:** `output/wd01/REGISTRATION_PA2G_CNT1.md`
(LF-sha256 `1bfbce27c7ada42a680fc807344c21ab8303fa8b03e469c8d8705c4e1c213116`,
registered 2026-09-30 BEFORE execution; owner decision ~11:30 via main
discussion). `params_nominal.json` UNCHANGED
(LF-sha256 `079a55768f803b23c92af188680c1df91dc63f04acbbbac948870b46e75935f7`,
re-verified post-run). Injector values are the registered a-priori ones;
nothing was tuned.

## Verdict up front

**NOT WORTH IT** under the registration section-5 rule, applied verbatim:
the all-vehicle delay MAE improves (v2g-CNT − v2g, world-bootstrap 90 % CI
excluding 0 at every H) but clause (1) requires ≥ 25 % relative MAE
reduction at H60 AND H120, and the measured reductions are −19.0 % (H60,
CI [−19.9, −18.3]) and −23.7 % (H120, CI [−24.9, −22.8]). Clause (1) holds
for neither horizon, so even the PARTIAL branch does not apply. Clauses
(2), (3) and (4) hold everywhere.

The direction of the registered motivation is confirmed — the tracked-only
v2g H120 all-vehicle delay error is −3045 s mean (MAE 71.2 % of the true
all-vehicle delay) and v2g-CNT cuts it to −2320 s (54.3 %) with near-exact
arrival-count reproduction — but the counter-based rate alone does not
reach the pre-registered benefit bar.

## 1. Question and mechanism (registered, a priori)

Does injecting arrivals at the network entrances at the rate an entrance
loop counter observed over the last W = 60 s (carried forward over
(w, w + H]) substantially reduce Part A's long-horizon ALL-vehicle delay
error versus the tracked-only v2g, against the TRUE all-vehicle delay
(tracked + arriving vehicles)?

- Counter channel: truth `departed_ids` classified by the vehicle's first
  physical row edge — mainline entry edge `1191554636` ("main"), ramp
  entry edge `11553048#1` ("ramp"); counts ONLY over (w − 60, w]; no count
  after w is read (sentinel test `tests/test_cnt1_injector.py`).
- Injector: deterministic evenly spaced insertion schedule (k-th vehicle
  due at w + (k − 0.5)/λ_e); mainline lanes alternate 0/1 round-robin
  starting lane 1; ramp uses its single lane; v_ins = min(lane limit,
  nearest leader speed); IDM safe-gap s0 + v_ins·T (s0 = 2 m, T = 1.5 s)
   else FIFO virtual entrance queue, retried each step. Queue-delay
   convention (registration §3, "use the same definition on both sides
   and state which"): the truth physics rows contain a vehicle only
   from SUMO insertion onward, so pre-insertion queue waiting accrues
   NO delay on either side; delay is counted from the first physical
   step at the free-flow reference (`test_queue_delay_accrues_only_
   after_insertion` enforces this).
  Injected vehicles are generic nominal HDVs, labelled `cntM*`/`cntR*`.
- Comparators, paired on the SAME windows: (i) **v2g-CNT** (this variant),
  (ii) **v2g** (tracked-only, the A2 nominal), (iii) **const-speed**,
  (iv) the **v1 p3o replica — oracle-arrival reference, not deployable**
  (it receives the LOGGED true arrivals).
- Zero SUMO; the engine, truth, worlds, windows and metrics code are the
  same as WD-01-PA2G-A2 / LC1 (post-LC1 counting convention).

## 2. Data, guard and close-out assertions

- Panel: the 32 PA2G-A2 TRAIN worlds of chunk 4
  (`results/lc1_a2_worlds32.json`), 18 windows/world (w = t0 + 30k,
  k = 1..18), H ∈ {10, 30, 60, 120}; 8960 raw rows = 280/world (16
  model-rows per full window = 4 models × 4 H, 12 at the 64 edge
  windows where the H exceeding the run end is dropped).
- Split checked from the chunk-4 `WORLD_PLAN.json` BEFORE any run file
  open (`eval_cnt1.load_panel_worlds` aborts on any non-TRAIN panel
  world). NO chunk-4 CALIBRATION or HOLDOUT file was opened.
- Close-out assertion `cnt1_merge_worlds.py`: every panel world has a
  280-row file with identical per-(model, H) structure; the union of the
  recorded opened-file lists (`worlds/<w>.opened.json`, 24/32 worlds)
  contains only TRAIN panel run files plus the plan — **PASSED**. The 8
  worlds salvaged from attempt-5 shards (w8623–w8634) have no
  `opened.json`; they were produced by the identical runner code path on
  the same TRAIN panel worlds before the crash-isolated per-world runner
  started recording the list.
- Preflight byte-identity gate (registration §7): with the injector OFF,
  the runner's v2g / const_speed rows equal the UNTOUCHED
  `eval_a2.window_rows` output on the first 2 panel worlds (all shared
  fields exactly), and an empty-schedule injector reproduces the v2g
  rollout bit-for-bit — **PASSED** (`results/eval_cnt1/preflight.json`,
  commit c03979d).
- post-LC1 comparator cross-check: the merged v2g / const_speed / p3o
  aggregates equal the frozen `eval_a2_post_lc1/summary.json` (all n
  match; max |float diff| 5.7e-14) — the comparators are bit-identical
  re-rolls in the registered single pass, no comparator re-run.

## 3. Execution history (full disclosure)

- **Attempt 1** (in-process panel): aborted at the tool timeout; also
  invalid for two code bugs (v2g rows not emitted; a RAMP_EDGE typo).
  Fixed, preflight re-passed (`results/eval_cnt1_aborted_tooltimeout/`).
- **Attempts 2–4** (detached panel runs): hard-killed externally with
  empty stderr within seconds of the spawn phase, while other sessions
  launched wall-campaign processes (`RUN2/3/4_KILLED_NOTE.txt`); the code
  was exonerated by two clean foreground smoke runs (exit 0, all four
  models).
- **Attempt 5**: relaunch via plain Start-Process detach; the session
  ended mid-panel and 8 world shards were salvaged unchanged
  (w8623–w8634).
- **Crash-isolated per-world runner** (`eval_cnt1_world.py`, one world
  per invocation, idempotent resume, rows written atomically): completed
  the remaining 24 worlds across the previous session (17 in commit
  5cfcbf1, then 10 more in commit 67556d7).
- **Runtime-limit resume (this session):** the previous task hit its
  240-min limit after commit 67556d7. The brief said "25/32 committed, 7
  missing"; the actual diff against `lc1_a2_worlds32.json` at 67556d7 was
  **27 committed / 5 missing** (w8694, w8695, w8697, w8699, w8706 — all
  p40). No completed world was re-run and nothing was re-decided; the 5
  missing worlds were run fresh with the identical runner, detached, at
  most 2 concurrent workers, BelowNormal priority, logs in
  `results/eval_cnt1/worlds/_logs/` (wall s: 382.9, 255.4, 261.9, 310.5,
  196.9). The registered diagnostic `cnt1_inj_diag.py` re-runs the
  deterministic v2g_cnt rollouts on a disclosed 2-world labelled subset
  (engine has no RNG; identical trajectories).
- No result-informed tuning or re-roll occurred at any point; all
  injector values are the registered a-priori ones.

## 4. Main table (per model × H, from the saved raw rows)

`delay_all_err` = predicted − true ALL-vehicle window delay [s]
(**primary**); MAE % = MAE / mean true ALL-vehicle delay;
`delay_err` = tracked-only error (continuity with A2); s_RMSE / v_RMSE on
tracked vehicles; TTC<3 / hard-brake errors are ALL-vehicle signed means
[steps] (per-window MAE in `results/eval_cnt1/tables.json`);
coll/1000 = predicted collisions per 1000 predicted veh-steps.

| model|H | n | all-veh err mean | MAE | MAE % of true | tracked err mean/MAE | s_RMSE | v_RMSE | TTC<3 err mean | hard-brake err mean | coll/1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2g_cnt|H10 | 576 | −81.8 | 81.8 | 23.9 % | +3.4/5.4 | 10.0 | 2.23 | −2.3 | −5.4 | 0.074 |
| v2g_cnt|H30 | 576 | −263.9 | 263.9 | 25.2 % | −2.8/29.3 | 51.6 | 5.02 | −6.8 | −21.9 | 0.023 |
| v2g_cnt|H60 | 576 | −780.1 | 780.1 | 36.4 % | −120.3/153.8 | 107.6 | 7.44 | −13.5 | −46.9 | 0.011 |
| v2g_cnt|H120 | 512 | −2320.2 | 2322.2 | 54.3 % | −543.0/575.3 | 137.5 | 8.33 | −27.2 | −97.1 | 0.005 |
| v2g|H10 | 576 | −85.4 | 85.4 | 24.9 % | +3.4/5.4 | 10.0 | 2.23 | −2.3 | −5.4 | 0.080 |
| v2g|H30 | 576 | −298.6 | 298.6 | 28.5 % | −2.8/29.3 | 51.6 | 5.02 | −6.8 | −21.9 | 0.029 |
| v2g|H60 | 576 | −963.4 | 963.4 | 44.9 % | −127.2/158.0 | 108.4 | 7.50 | −13.5 | −46.9 | 0.018 |
| v2g|H120 | 512 | −3045.1 | 3045.1 | 71.2 % | −568.8/595.6 | 137.9 | 8.36 | −27.2 | −97.2 | 0.018 |
| const_speed|H10 | 576 | −98.9 | 98.9 | 28.9 % | −10.1/11.8 | 16.3 | 3.63 | +22.5 | −8.1 | 0.000 |
| const_speed|H30 | 576 | −341.3 | 341.5 | 32.6 % | −45.6/63.2 | 84.5 | 7.35 | +110.7 | −24.8 | 0.000 |
| const_speed|H60 | 576 | −826.7 | 827.3 | 38.6 % | +9.5/130.4 | 122.4 | 8.45 | +174.4 | −49.8 | 0.000 |
| const_speed|H120 | 512 | −2052.9 | 2053.1 | 48.0 % | +423.4/474.8 | 130.7 | 8.93 | +218.2 | −100.0 | 0.000 |
| v1_p3o_replica (oracle-arrival reference, not deployable)|H10 | 576 | +25.8 | 26.1 | 7.6 % | +11.5/12.1 | 12.9 | 2.97 | +0.9 | +6.4 | 0.234 |
| v1_p3o_replica (oracle-arrival reference, not deployable)|H30 | 576 | +123.5 | 123.5 | 11.8 % | +33.9/39.5 | 65.1 | 5.91 | +0.4 | +15.5 | 1.241 |
| v1_p3o_replica (oracle-arrival reference, not deployable)|H60 | 576 | +261.2 | 262.5 | 12.2 % | −19.8/86.0 | 100.3 | 7.05 | −1.1 | +25.8 | 5.872 |
| v1_p3o_replica (oracle-arrival reference, not deployable)|H120 | 512 | +408.8 | 483.0 | 11.3 % | −269.9/314.3 | 118.2 | 7.88 | −4.7 | +36.0 | 17.021 |

Mean true ALL-vehicle window delay (same windows for v2g-CNT and v2g):
342.7 / 1047.1 / 2145.5 / 4275.0 s at H10/30/60/120.

The p3o replica's small all-vehicle delay error is bought with LOGGED true
arrivals and a large safety cost (coll/1000 rising to 17.0 at H120) — it
bounds what perfect arrival knowledge would give a v1-style model; it is
**not deployable** and not a candidate.

## 5. Injected vs true arrival counts (per window, v2g-CNT rows)

| H | windows | true arrivals mean | injected mean | scheduled mean | true sum | injected sum |
|---|---|---|---|---|---|---|
| 10 | 576 | 7.99 | 8.11 | 95.67 | 4600 | 4673 |
| 30 | 576 | 23.94 | 24.32 | 95.67 | 13792 | 14008 |
| 60 | 576 | 47.93 | 47.66 | 95.67 | 27608 | 27454 |
| 120 | 512 | 95.93 | 95.43 | 95.54 | 49114 | 48858 |

The carried-forward counter rate reproduces the realized arrival counts
within ±1 % at every H (counts within the shorter H window; "scheduled"
is the full 120 s schedule size, of which only the first H is scored).

## 6. v2g-CNT − v2g, world-bootstrap 90 % CIs (B = 2000, seed 20260930, worlds resampled)

| H | MAE diff [s] | MAE rel change | signed err diff [s] | s_RMSE diff [m] | coll/1000 diff |
|---|---|---|---|---|---|
| H10 | −3.6 [−4.0, −3.3] | −4.2 % [−4.6, −3.9] | +3.6 [+3.3, +4.0] | −0.0 [−0.0, +0.0] | −0.0 [−0.0, −0.0] |
| H30 | −34.7 [−37.5, −32.1] | −11.6 % [−12.3, −11.0] | +34.7 [+32.1, +37.5] | +0.0 [+0.0, +0.0] | −0.0 [−0.0, −0.0] |
| H60 | −183.3 [−196.5, −169.6] | −19.0 % [−19.9, −18.3] | +183.3 [+169.6, +196.5] | −0.8 [−1.0, −0.6] | −0.0 [−0.0, −0.0] |
| H120 | −722.9 [−791.3, −654.9] | −23.7 % [−24.9, −22.8] | +724.9 [+657.8, +791.8] | −0.4 [−0.7, −0.2] | −0.0 [−0.0, −0.0] |

Every CI excludes 0: the counter-arrival injector improves the all-vehicle
delay error at ALL horizons (sign convention: negative = improvement; the
signed column is positive only because v2g under-predicts and injecting
arrivals reduces that under-prediction). Tracked-vehicle accuracy is
unchanged (s_RMSE diff ≤ 0.8 m), and the collision rate does not increase
(clause 4 margin everywhere).

## 7. Injected-vehicle s_RMSE vs matched truth arrivals (descriptive, registered §4)

Subset: first 2 panel worlds, all 18 windows, H = 120; i-th injected
vehicle per entrance matched to the i-th true arrival per entrance;
shared-d-axis s_RMSE over co-present steps. See
`results/eval_cnt1/inj_diag.json` (generated by `cnt1_inj_diag.py`).

- Pooled by entrance: **main 223.7 m** (336,455 co-present steps),
  **ramp 206.3 m** (29,412 steps).
- By order of entry (first 10 mainline entries, per-entrance pooled):
  main#0 286 m, #1 324 m, #2 283 m, #3 305 m, #4 266 m, #5 256 m, #6 246 m,
  #7 257 m, #8 252 m, #9 244 m; ramp order values scatter more thinly
  (order 0–3 ≈ 250–330 m, thinning to ~120–200 m by order ≥ 13 as the
  co-present-step counts shrink).
- Reading (descriptive only): an injected vehicle tracks its matched true
  arrival to roughly 200–330 m over a 120 s horizon, with the earliest
  entries slightly worse (they inherit the tracked-scope initial error and
  the busiest insertion phase); no order-of-entry trend indicates a
  schedule-phase artefact. These magnitudes are small relative to the
  all-vehicle delay MAE gains at H120, consistent with the arrival-count
  reproduction in §5.

## 8. Decision rule (registration §5, applied verbatim)

- **(1)** All-vehicle delay MAE at H60 AND H120 each ≥ 25 % lower than
  v2g with the world-bootstrap 90 % CI of the relative change excluding 0:
  H60 −19.0 % (CI [−19.9, −18.3]) → **False**; H120 −23.7 %
  (CI [−24.9, −22.8]) → **False**. (CIs exclude 0; the ≥ 25 % magnitude
  is missed.)
- **(2)** H10/H30 MAE not worse than v2g by more than 5 %: −4.2 % and
  −11.6 % → **True / True**.
- **(3)** Tracked s_RMSE at H10/H30 not worse than v2g by more than 5 %:
  0.0 % and 0.0 % → **True / True**.
- **(4)** collisions ≤ v2g + 0.1 per 1000 veh-steps at every H: margins
  −0.006/−0.006/−0.007/−0.012 → **True / True / True / True**.

**VERDICT: NOT WORTH IT** (clause (1) holds for neither H60 nor H120, so
the PARTIAL branch is not reachable). The improvement is real and
consistent, but a 60 s carried-forward entrance-counter rate recovers
only part of the registered ≥ 25 % bar at long horizons.

Label: **sensitivity; requires counter hardware; does not change the
CAV-strict headline.**

## 9. Limitations

- TRAIN-only, 32 worlds, single chunk; the chunk-4 CALIBRATION split is
  consumed by C4L1, so any adopted variant would need fresh evaluation
  worlds (registered caveat).
- The counter gives a carried-forward RATE, not per-vehicle schedules;
  the injector cannot know which specific vehicle arrives, so part of the
  remaining all-vehicle error is arrival-composition noise, not rate
  error (the oracle-arrival p3o reference shows the remaining gap to
  perfect arrival knowledge).
- Injected vehicles are generic nominal HDVs with a-priori insertion
  rules; no calibration was permitted (registration §3).
- Raw rows carry per-window aggregates; the per-vehicle injected s_RMSE
  diagnostic is descriptive on a disclosed 2-world subset (§7).
- Collisions in the predictor are counted from the rollout engine, not a
  SUMO safety report; vehicle-steps normalization differs slightly across
  models with different insertion counts.
- Short mechanics-style windows are not efficacy evidence for control;
  this analysis changes only the prediction scope (no controller change).

## 10. Artifacts and provenance

- Raw rows (saved before aggregation): `results/eval_cnt1/worlds/<w>.jsonl`
  (32 × 280 rows) and the merged `results/eval_cnt1/raw_rows.jsonl`
  (8960 rows).
- Machine-readable results: `results/eval_cnt1/tables.json`
  (aggregates + bootstrap + decision rule + arrivals + cross-check),
  `results/eval_cnt1/inj_diag.json`, `results/eval_cnt1/preflight.json`,
  `results/eval_cnt1/merge_worlds.out`, `results/eval_cnt1/tables_run.out`,
  per-world logs `results/eval_cnt1/worlds/_logs/`.
- Code: `eval_cnt1.py` (registered harness), `eval_cnt1_world.py`
  (crash-isolated runner), `arrivals.py` (counter reader + injector),
  `cnt1_tables.py` (bootstrap + decision rule), `cnt1_merge_worlds.py`
  (close-out assertions), `cnt1_inj_diag.py` (descriptive diagnostic);
  tests `tests/test_cnt1_injector.py` (14 registered tests).
- Truth: `E:/research-dense/runs/campaign_stage1_chunk4/<world>__native_base/run/physics_steps.jsonl`;
  plan `E:/research-dense/output/campaign_stage1_v3/results_plan_chunk4/WORLD_PLAN.json`
  (raw sha256 `65510b27f651f477c3de9acc663b9138916a07dfb46468718dc9932147e1ffc4`);
  param_set_id `BA1FD93F0A21B1FFD2CDF20693E74FB8E3FBB8E4A1C57E7F88D9A3929E3455CE`.
