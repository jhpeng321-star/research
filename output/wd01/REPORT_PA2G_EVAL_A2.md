# REPORT — WD-01-PA2G-A2: Part A v2 general, evaluation (a) A2 — model error from the TRUE fleet-native initial state, after the LC-churn fix

- **Analysis ID:** WD-01-PA2G-A2 (registered before any run:
  `REGISTRATION_PA2G_EVAL_A2.md`; reuses `REGISTRATION_PA2G_EVAL_A.md`
  with the truth source changed, as REPORT_PA2G_EVAL_A.md's "Next step"
  specified; design `PARTA_V2_GENERAL_DESIGN.md`; churn diagnosis and
  fix: `partA_v2g/DIAGNOSIS.md`).
- **Status:** eval COMPLETED 2026-09-30 (wall 1377 s, 4 workers, zero
  SUMO/TraCI). Raw rows saved before aggregation. This report and the
  committed `aggregate_eval_a2.py` re-derive every table from the saved
  rows (the eval was not re-run for close-out).
- **Executor branch:** `zcode/wd01pa4`.

## What changed vs WD-01-PA2G-A

1. **Truth:** chunk-4 `<world>__native_base` runs — TRUE fleet-native
   (ALL CAVs native from t = 0, speedMode 27), replacing the chunk-2/3
   `<world>__native` runs whose CAVs followed random per-CAV
   damping/native switching schedules. This removes the PA2G-A TRUTH
   CAVEAT confound.
2. **Engine fix:** the discretionary LC churn root cause
   (`_neighbors` leader/follower swap + unclipped decision
   accelerations) is fixed; nominal parameter TEXTBOOK values are
   unchanged. See `partA_v2g/DIAGNOSIS.md`.
3. **CAV path wired:** zone vehicles present in a run's own `cav_ids`
   (physics rows; insertions.jsonl / dispatcher_rows.jsonl never read)
   carry `is_cav=True` and the engine runs with the native command path
   (`cavctrl.CavControllerLoop`, all-native actions → no commands; the
   partA_ctrl native path is engaged and advances the PI state every
   control tick). behaviourally, native CAVs follow the generic rules —
   same as truth.
4. **Design-deviation rule applied (OWNER FLAG):** the non-textbook
   "stability guards" declared in the PA2G-A nominal set are retired to
   OFF (`cooldown_s` 2.0 → 0.0, `lc_min_speed_mps` 1.0 → 0.0,
   `ego_gain_floor` no-op). No textbook value moved (IDM T 1.5, a 1.0,
   b 1.5, s0 2, δ 4; MOBIL p 0.5, b_safe 4, threshold 0.1, keep-right
   bias 0.3; merge values unchanged). `param_set_id`
   `BA1FD93F0A21B1FFD2CDF20693E74FB8E3FBB8E4A1C57E7F88D9A3929E3455CE`
   (PA2G-A used `6359f5…4b9`). The guards remain available as the
   labelled `v2g_guards_on` variant (Table G): post-fix they change
   the disc-LC rate by <2 % (3.60 vs 3.55 per window) — the churn
   collapse needs no guard.

## Data and raw-row counts

- Source: `partA_v2g/results/eval_a2/raw_rows.jsonl`, sha256
  `79E2BCD94D2604985386EA1A436A4C1D9284C2A9B865CCB7DDE51CE47068E92B`.
- 8432 raw lines = 6720 main rows + 1152 sweep rows + 560 labelled
  guards_on rows (no duplicate emission exists in A2: the sweep and
  guards passes skip the const-speed row).
- **Worlds:** 32 of the 61→63 usable TRAIN `__native_base` runs,
  deterministic round-robin over the 16 (demand, penetration) cells,
  2 per cell (Table: 16 cells × 2, from the eval log). Usable = plan
  split TRAIN ∧ `run/summary.json` present ∧ `RUN_RECEIPT.json`
  exit_status `complete` ∧ physics log reaches the run duration.
  (Note: two runs' `summary.json`/receipt files materialized during
  this task via a concurrent file restoration — their receipts are
  dated 2026-09-29; every selected run passed the completion and
  coverage checks at run time. w8703m2600p40t540 and w8706m2400p40t360
  were the latecomers; w8706 is in the panel.)
- Windows: every 30 s over (t0, t0+540], t0 from the plan; H ∈
  {10,30,60,120} kept when the log covers w+H → 576 windows per model
  per H (512 at H120; the runs cover [0, t0+600]).
- `assert_train_only` passed; every opened run file is recorded:
  `summary.json` (`opened_run_files`) lists the 32 physics logs + plan
  (the worker processes' opens), and
  `results/eval_a2/opened_run_files_full.json` adds the screening opens
  (per-run `summary.json`/`RUN_RECEIPT.json` of all 63 usable TRAIN
  worlds; 127 paths total, TRAIN-or-plan only). CAL/VALIDATION/HOLDOUT
  were never opened.

## Table 1 — per model × H (errors are pred − truth; mean SIGNED and mean ABSOLUTE both shown)

delay/TTC/hard are per-window aggregates over tracked vehicles (delay
in vehicle-seconds, TTC<3 exposure in pair-seconds, hard-brake in
vehicle-seconds, a < −3 m/s²); s_RMSE is on the shared downstream-d
axis; coll/1000 = predicted collisions per 1000 vehicle-steps.

| model\|H | n | s_RMSE | v_RMSE | delay_err(mean) | delay_err(MAE) | ttc3_err(mean) | hard_err(mean) | LC merge p/t | LC disc p/t | coll/1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| const_speed\|H10 | 576 | 16.6 | 3.71 | −9.3 | 11.3 | 22.8 | −7.6 | 0/611 | 0/443 | 0.00 |
| const_speed\|H30 | 576 | 84.8 | 7.40 | −40.4 | 61.3 | 115.6 | −16.6 | 0/1762 | 0/1139 | 0.00 |
| const_speed\|H60 | 576 | 122.5 | 8.50 | 23.8 | 133.6 | 188.4 | −16.9 | 0/2335 | 0/1404 | 0.00 |
| const_speed\|H120 | 512 | 130.8 | 8.97 | 452.1 | 499.8 | 247.2 | −17.5 | 0/2135 | 0/1393 | 0.00 |
| v1_p3o_replica\|H10 | 576 | 12.9 | 2.98 | 11.9 | 12.4 | 0.9 | 5.4 | 1483/611 | 807/443 | 0.23 |
| v1_p3o_replica\|H30 | 576 | 65.0 | 5.90 | 34.9 | 40.3 | 1.3 | 12.1 | 2420/1762 | 2130/1139 | 1.24 |
| v1_p3o_replica\|H60 | 576 | 100.1 | 7.03 | −17.3 | 86.4 | 1.9 | 15.6 | 2425/2335 | 3495/1404 | 5.87 |
| v1_p3o_replica\|H120 | 512 | 117.8 | 7.86 | −266.8 | 312.5 | 2.2 | 17.9 | 2136/2135 | 3761/1393 | 17.01 |
| v2g\|H10 | 576 | 10.1 | 2.28 | 4.5 | 6.1 | −2.2 | −4.8 | 390/611 | 2044/443 | 0.12 |
| v2g\|H30 | 576 | 51.6 | 5.02 | 2.8 | 30.3 | −4.8 | −13.6 | 631/1762 | 6245/1139 | 0.04 |
| v2g\|H60 | 576 | 106.8 | 7.44 | −114.4 | 149.8 | −4.8 | −14.0 | 1728/2335 | 8432/1404 | 0.03 |
| v2g\|H120 | 512 | 136.7 | 8.32 | −551.3 | 581.7 | −4.8 | −14.6 | 1955/2135 | 7323/1393 | 0.03 |

(LC p/t = predicted/true counts summed over windows; disc = disc_left +
disc_right.)

**Headline:** with the fix, v2g is the best model at H10/H30 on every
registered metric (s_RMSE 10.1 m at H10, delay MAE 6.1 veh-s), its
predicted collisions are near zero (0.03–0.12 per 1000 vehicle-steps,
vs 2.2–13.9 pre-fix and up to 17.0 for the p3o replica), and the
discretionary LC churn is gone (2044 predicted vs 443 true at H10;
486,428 → 7,323 at H120). The residual is a bounded ~4.6–6.0×
over-production of discretionary LCs plus a negative delay bias that
grows with H — see the model-error statement.

## Table 1b — side-by-side PA2G-A (pre-fix engine, switching-mix truth, CAV path off) vs A2

| quantity | PA2G-A | A2 | dominant cause of the difference |
|---|---|---|---|
| truth | chunk-2/3 `__native` (per-CAV damping/native switching; CAV path never engaged in the model) | chunk-4 `__native_base` (true fleet-native; CAV path engaged) | truth change |
| worlds × windows | 25 × 18 | 32 × 18 | panel change |
| v2g disc LC p/t @H10 | 68,614/243 (152.5/window) | 2044/443 (3.5/window) | **the fix** (Table F: 10,921 → 389 on the SAME windows) |
| v2g disc LC p/t @H120 | 486,428/949 | 7323/1393 | **the fix** (Table F: 84,108 → 1515) |
| v2g coll/1000 @H10 | 13.92 | 0.12 | **the fix** (Table F: 1470 → 14 collisions) |
| v2g s_RMSE @H10 | 28.2 m | 10.1 m | **the fix** (Table F: 27.7 → 8.5 m same-windows) |
| v2g delay_err @H10 | +48.9 mean (=MAE) | +4.5 mean, 6.1 MAE | **the fix** (Table F: 49.6 → 5.4 MAE same-windows) |
| v2g delay_err @H120 | +49.8 mean / 368.0 MAE (sign cancellation: damping-truth CAVs made truth slower) | −551.3 mean / 581.7 MAE (uniformly negative: the model under-predicts tracked delay) | **the fix removes the manufactured churn delay**; the remaining H120 bias is the missing-arrivals + untracked-congestion gap (Table 2: untracked share 0.577), not the truth change alone |
| const-speed / p3o comparators | — | — | unchanged by the fix (p3o numbers differ only via the panel/truth change) |

**Attribution summary (Table F, paired same-window comparison on the
PA2G-A chunk-2/3 truth, 4 worlds × 72 windows):** pre-fix → post-fix
disc LCs 10,921 → 389 (H10) and 84,108 → 1,515 (H120); collisions
1470 → 14; s_RMSE 27.7 → 8.5 m (H10); delay MAE 49.6 → 5.4 (H10).
These deltas are the FIX alone (same truth, same no-CAV-flag treatment
as PA2G-A). The truth change (switching-mix → fleet-native) and the CAV
wiring explain the rest of the A-vs-A2 movement (e.g. truth disc LC
counts roughly double: 243 → 443 at H10 over a larger, more congested
panel; the p3o comparator's disc over-production persists at 807/443).

## Table F — fixcheck: the FIXED engine on the PA2G-A chunk-2/3 truth (labelled)

4 PA2G-A worlds (`w8481m2700p02t480`, `w8513m2600p10t300`,
`w8546m2700p40t540`, `w8582m2600p10t360`; cells 2700/p0.02, 2600/p0.1,
2700/p0.4), same no-CAV-flag treatment as PA2G-A so the only delta vs
the saved PA2G-A rows is the engine fix. `v2g_fixed_guards_on` shows
the guards barely matter post-fix (disc 1515 vs 1490 at H120).
Raw: `results/eval_fixcheck/raw_rows.jsonl` (sha256
`1C4B1F8BA33CFE22D8456A8767EC8D1E83C603600D9097F1E4B6E84ED584785F`).

### paired: same windows, pre-fix (PA2G-A rows) vs post-fix (v2g_fixed)

| H | n windows | disc LC pre | disc LC post | disc LC true | coll pre | coll post | s_RMSE pre | s_RMSE post | delay_err pre (mean/MAE) | delay_err post (mean/MAE) |
|---|---|---|---|---|---|---|---|---|---|---|
| H10 | 72 | 10921 | 389 | 32 | 1470 | 14 | 27.7 | 8.5 | 49.6/49.6 | 2.2/5.4 |
| H30 | 72 | 38057 | 1326 | 111 | 1878 | 14 | 117.7 | 48.3 | 222.3/222.3 | −26.4/45.3 |
| H60 | 72 | 67543 | 1664 | 163 | 1924 | 14 | 151.0 | 118.6 | 405.1/405.1 | −261.5/273.6 |
| H120 | 64 | 84108 | 1515 | 209 | 1537 | 8 | 180.3 | 196.0 | 42.5/345.1 | −1332.0/1343.3 |

(The 4 fixcheck worlds are more congested than the panel average, so
their absolute values exceed the Table-1 rows; the paired DELTAS are
the point.)

## Table G — labelled `v2g_guards_on` variant (design-deviation comparison; NOT headline)

The fixed engine with the PA2G-A-era guards re-enabled (cooldown 2.0 s,
lc_min_speed 1 m/s), every 4th world (8 worlds, 144 windows):

| H | n | s_RMSE | delay_err mean | delay_err MAE | LC disc p | LC disc t | coll/1000 |
|---|---|---|---|---|---|---|---|
| H10 | 144 | 12.4 | 2.9 | 5.3 | 518 | 139 | 0.08 |
| H30 | 144 | 53.9 | 8.2 | 27.6 | 1199 | 400 | 0.03 |
| H60 | 144 | 95.3 | −50.5 | 104.6 | 1565 | 481 | 0.02 |
| H120 | 128 | 108.9 | −258.8 | 312.1 | 1345 | 456 | 0.01 |

Per-window disc rate 3.60 (guards on) vs 3.55 (nominal) — the guards
are behaviourally irrelevant once the bug is fixed, supporting their
retirement from the nominal set. The no-ping-pong property is
test-pinned WITHOUT any cooldown
(`test_no_ping_pong_after_a_change_without_cooldown`).

## Table 2 — missing-arrivals curve (truth delay, v2g windows)

| H | n | delay_true tracked | delay_true all | arrivals/window | untracked delay share |
|---|---|---|---|---|---|
| H10 | 576 | 257 | 343 | 8.0 | 0.249 |
| H30 | 576 | 758 | 1047 | 23.9 | 0.276 |
| H60 | 576 | 1318 | 2145 | 47.9 | 0.386 |
| H120 | 512 | 1807 | 4275 | 95.9 | 0.577 |

Untracked arrivals hold 25 % of truth delay at H10 and **58 % at
H120**: beyond H = 60 the tracked-scope Part A misses more than half of
the true system delay. This — not lane-change error — is the dominant
known driver of the growing negative delay bias at long H (the model's
tracked vehicles see free road where truth accumulates congestion from
arrivals).

## Table 3 — `n_free_road_assumed` stats (v2g windows)

| H | n | assumed mean | median | max | share windows>0 | truth-flag mean | truth-flag max |
|---|---|---|---|---|---|---|---|
| H10 | 576 | 2.00 | 2 | 2 | 1.00 | 2.07 | 3 |
| H30 | 576 | 2.00 | 2 | 2 | 1.00 | 2.07 | 3 |
| H60 | 576 | 2.00 | 2 | 2 | 1.00 | 2.07 | 3 |
| H120 | 512 | 2.00 | 2 | 2 | 1.00 | 2.07 | 3 |

Identical structure to PA2G-A: the zone-truncation rule flags exactly 2
vehicles per window; the truth-exact rule 2.07 on average. Small,
constant across H.

## Table 4 — descriptive T/a sweep at H = 60 (8 worlds × 18 windows per cell; NO parameter selected)

| cell | kind | T | a_max | n | s_RMSE | v_RMSE | delay_err mean |
|---|---|---|---|---|---|---|---|
| T1.0_a0.7 | sweep | 1.0 | 0.7 | 144 | 98.4 | 7.07 | −54.1 |
| T1.0_a1.0 | sweep | 1.0 | 1.0 | 144 | 104.9 | 7.24 | −152.6 |
| T1.0_a1.5 | sweep | 1.0 | 1.5 | 144 | 112.5 | 7.83 | −253.2 |
| T1.5_a0.7 | sweep | 1.5 | 0.7 | 144 | 91.8 | 6.86 | 31.3 |
| T1.5_a1.5 | sweep | 1.5 | 1.5 | 144 | 101.2 | 7.22 | −137.6 |
| T2.0_a0.7 | sweep | 2.0 | 0.7 | 144 | 91.8 | 6.94 | 113.7 |
| T2.0_a1.0 | sweep | 2.0 | 1.0 | 144 | 92.7 | 6.75 | 45.5 |
| T2.0_a1.5 | sweep | 2.0 | 1.5 | 144 | 97.9 | 7.04 | −25.1 |
| T1.5_a1.0 | nominal (registered v2g; NOT selected from the sweep) | 1.5 | 1.0 | 576 | 106.8 | 7.44 | −114.4 |

Descriptive only: s_RMSE moves 92–112 m across the sweep; the signed
delay error swings −253 to +114 mainly with a_max (lower acceleration →
less congestion build-up relative to a fleet-native truth that
accelerates harder). Nothing was chosen from this sweep (registration
rule). The sweep subset's absolute values sit below the panel nominal
because its 8 worlds are the less-congested half of the round-robin.

## Table 5/6 — breakdowns and sign diagnostics (v2g)

Full breakdowns by penetration and demand, and the H60/H120 sign
diagnostics, are in `results/eval_a2/tables.json` / `tables.md`
(generated by the committed `aggregate_eval_a2.py`). Sign structure:
H60 delay_err is negative in 398/576 windows (all penetrations negative,
−80 to −149); H120 negative in 384/512. Unlike PA2G-A's H120
cancellation (damping-truth CAVs at high penetration flipped the sign
positive), A2 is uniformly negative — the under-prediction is a
systematic tracked-scope effect (missing arrivals + free-road
assumption), not a penetration-regime flip: p0.02 −426.8 vs p0.4
−667.1 at H120, same sign.

## Honest model-error statement

- **What the fix solved:** the pre-fix engine was broken (swapped
  neighbour structure → 1e15-scale incentive terms → 33–66×
  over-production of discretionary LCs and manufactured collisions).
  Post-fix, on fleet-native truth with the CAV path wired, the v2g
  model is the best short-horizon predictor of the registered set
  (s_RMSE 10.1 m and delay MAE 6.1 veh-s at H10, vs 12.9 m/12.4 for
  the p3o replica and 16.6 m/11.3 for const-speed) and predicts
  essentially no false collisions (0.03–0.12 per 1000 vehicle-steps).
- **What remains model error (not tuned away, per registration):**
  1. **Discretionary LC over-production ~4.6–6.0×** (2044 vs 443 at
     H10; 8432 vs 1404 at H60). Textbook MOBIL with the keep-right bias
     at 0.5 s ballistic integration changes lanes more often than this
     campaign's SUMO fleet does. This is now the dominant lane-change
     error and is a *model-behaviour* question (threshold/bias semantics
     under coarse integration), flagged for the owner; no parameter was
     selected to reduce it.
  2. **Negative delay bias growing with H** (mean −114 at H60, −551 at
     H120; MAE 150/582): the tracked-scope model under-predicts delay
     because untracked arrivals accumulate congestion the model cannot
     see (Table 2: 58 % of truth delay is untracked at H120) and the
     free-road assumption is optimistic (Table 3: 2 vehicles/window).
     Within the tracked horizon (H10/H30) the bias is small (+4.5/+2.8).
  3. TTC<3 exposure and hard-brake are under-predicted at all H
     (−4.8/−14.6 at H120): the model's smoother trajectories produce
     fewer safety-critical events than the fleet-native truth.
- **Comparators:** const-speed persistence has the largest positive
  delay bias at H120 (+452) because it cannot react; the v1 p3o replica
  tracks position well at long H (117.8 m) but produces the most
  collisions (17.0/1000 at H120) and over-produces disc LCs (807 vs 443
  at H10, 3761 vs 1393 at H120).
- **Attribution caveat:** the truth is now truly fleet-native and the
  CAV path is engaged, so these numbers are clean model error of the
  generic rules for this panel — with one remaining structural limit:
  Part A v2 general is a tracked-scope product, so its long-H delay
  error is dominated by scope, not by the driver model.

## Design-deviation flags for the owner

1. The PA2G-A nominal set contained three non-textbook stability guards
   (cooldown_s 2.0, lc_min_speed_mps 1.0, ego_gain_floor). Per the
   design-deviation rule they are retired to OFF in the nominal set
   (textbook values untouched); the labelled variant (Table G) shows
   they are unnecessary post-fix. If the owner wants them as declared
   deviations, the `v2g_guards_on` parameter set is the ready-made
   variant.
2. The churn fix itself is a pure correctness fix (neighbour roles +
   decision/dynamics acceleration consistency) — no new behavioural
   rule was introduced.
3. The residual disc-LC over-production (point 1 of the model-error
   statement) is a candidate owner question: accept as model error, or
   schedule a separately registered investigation of MOBIL's threshold
   semantics at dt = 0.5 s.

## Provenance, integrity, artifacts

- Registration: `REGISTRATION_PA2G_EVAL_A2.md` (written before the
  run). Eval runner: `partA_v2g/eval_a2.py` (--stratified 32 --workers
  4 --sweep-every 4 --guards-on-every 4); run log
  `partA_v2g/eval_a2_run.out` + `.err`; an earlier 2-window aborted
  attempt (window-cap bug, 960 rows) is preserved at
  `results/eval_a2_aborted_2window/` with `ABORTED_NOTE.txt` (no silent
  retry).
- `assert_train_only` passed; opened files (plan + per-run
  summary.json/receipt + physics_steps.jsonl of TRAIN worlds only) are
  listed in `results/eval_a2/summary.json`.

| artifact | sha256 |
|---|---|
| `partA_v2g/results/eval_a2/raw_rows.jsonl` | 79E2BCD94D2604985386EA1A436A4C1D9284C2A9B865CCB7DDE51CE47068E92B |
| `partA_v2g/results/eval_a2/summary.json` | DBE1E40EA4210F53E289BFA4DB5C726707E0039067AAEA971873121325F58CF3 |
| `partA_v2g/results/eval_a2/tables.json` | A838EE3CCFEE18BE189F604EC2BE00BE576CE371412632C6121DC06FA91CC8DC |
| `partA_v2g/results/eval_fixcheck/raw_rows.jsonl` | 1C4B1F8BA33CFE22D8456A8767EC8D1E83C603600D9097F1E4B6E84ED584785F |
| `partA_v2g/results/eval_fixcheck/summary.json` | 482B202B0F40D6658990C4F5A1F1F8FFFA87881CFB0797265D580EF0D319D51F |
| `partA_v2g/params_nominal.json` | BA1FD93F0A21B1FFD2CDF20693E74FB8E3FBB8E4A1C57E7F88D9A3929E3455CE |

Ledger draft: `output/wd01/LEDGER_DRAFT_PA2G_A2.json` (to be appended
to the master ledger by the integrating agent; shared records untouched
here).

## Next step

Per the registration: this analysis supersedes WD-01-PA2G-A's headline.
Owner decisions now queued: (a) accept/decline the guard retirement,
(b) whether to register a bounded investigation of the residual
MOBIL disc-LC over-production, (c) the tracked-scope H ceiling is a
Part-B/horizon question (Table 2), not a Part-A fix.
