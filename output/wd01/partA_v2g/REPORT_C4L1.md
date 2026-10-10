# REPORT (FROZEN LOOK): WD-01-C4L1 — Part A v2 general, first CALIBRATION look

Registration: `output/wd01/REGISTRATION_C4L1.md` (LF-sha256 `5e4a772137b8b427823b4860a20440eaa65abcbea735551eff681aa67b80b124`, verified by the eval gate before the pass and pinned in `C4L1_FREEZE.json`). Engine: zcode/wd01c4l1 at `0c1cb4051bf5` (== zcode/wd01 9f716b4 for `partA_v2g`; engine.py/metrics.py/cavctrl.py unchanged since 9dda3a3). Params: `params_nominal.json` LF-sha256 `079a5576…35f7` (guards OFF, owner decision (a)). Zero SUMO; the pass replayed saved `__native_base` logs and ran the model rollouts in one process (workers 1, below-normal priority, WD_FORBID_D=1). Wall clock 1703 s.

## Verdict (frozen rule, H = 30 s): **PASS**

PASS iff (a) AND (c) AND (d) at H30; (b) is REPORTED ONLY (owner decision 2026-09-30 ~09:55, LC1 option 1). Computed from the saved raw rows only (`results/c4l1/criteria_verdict.json`).

| criterion | value | limit | ok |
|---|---|---|---|
| (a) collisions / 1000 veh-steps | 0.0105 | <= 1.0 | True |
| (c) delay MAE v2g (tracked scope) | 32.47 s | < const-speed 69.90 s (same windows) | True |
| (c) mean signed delay error | -1.99 s | within ±15 % of mean truth tracked delay 926.0 s (±138.9 s) | True |
| (d) v_RMSE v2g | 4.959 m/s | <= 1.05 x min(const-speed 8.069, p3o replica 5.834) = 6.125 | True |

**Registered PASS claim:** "Part A v2 general meets C4L1 at H = 30 on the true state." This claim does NOT extend to recon-sample inputs (that is C4L2, a separate registered look once the reconstructor provides K recon samples from a model never fitted on these worlds).

## Data, worlds, refusals

- Truth: chunk-4 `<world>__native_base/run/physics_steps.jsonl`, receipt `complete` + coverage, CAV identity from the physics rows' `cav_ids`. 9/9 registered CALIBRATION worlds usable; **refused worlds: none** (not replaced — nothing to replace).
- Windows: starts s = t0 + 30k with s + H <= t0 + 540 (the registered rule, enforced via `log.t_end = min(t_end, t0+540)`); exact counts from the code: n = 153 per model at H10/H30 (9 x 17), 144 at H60 (9 x 16), 126 at H120 (9 x 14); 1920 raw rows total.
- Comparators scored in the same pass: constant-speed persistence, v1 p3o replica (frozen knobs, sha 458890d9-era knobs per the A2 harness). Guards-ON variant: labelled, every 3rd world (3 of 9), never a gate item.
- Exposure audit at run time: PASSED (zero hits for the 9 CAL world names across all 49 Part A output roots; `partA_v2g/EXPOSURE_AUDIT_C4L1.md`).

## Table 1: per model x H (main registered models)

| model | H | n | s_RMSE (m) | v_RMSE (m/s) | delay_err mean (s) | delay_err MAE (s) | TTC<3 err mean | hard-brake err mean | coll/1000 |
|---|---|---|---|---|---|---|---|---|---|
| v2g | 10 | 153 | 8.4 | 2.120 | 4.9 | 5.8 | -2.18 | -5.20 | 0.029 |
| v2g | 30 | 153 | 50.3 | 4.959 | -2.0 | 32.5 | -4.76 | -14.63 | 0.010 |
| v2g | 60 | 144 | 116.2 | 7.878 | -154.5 | 181.1 | -4.98 | -15.51 | 0.003 |
| v2g | 120 | 126 | 155.1 | 9.154 | -709.2 | 735.2 | -5.12 | -16.85 | 0.000 |
| const_speed | 10 | 153 | 16.7 | 3.950 | -11.0 | 13.0 | 27.96 | -8.72 | 0.000 |
| const_speed | 30 | 153 | 94.1 | 8.069 | -44.1 | 69.9 | 145.61 | -18.24 | 0.000 |
| const_speed | 60 | 144 | 143.1 | 9.416 | 23.6 | 153.3 | 237.55 | -19.15 | 0.000 |
| const_speed | 120 | 126 | 154.6 | 10.202 | 555.8 | 605.8 | 311.63 | -20.52 | 0.000 |
| v1_p3o_replica | 10 | 153 | 11.9 | 2.936 | 13.8 | 13.8 | 1.17 | 6.61 | 0.193 |
| v1_p3o_replica | 30 | 153 | 65.0 | 5.834 | 42.5 | 47.7 | 1.44 | 13.47 | 1.816 |
| v1_p3o_replica | 60 | 144 | 101.5 | 7.009 | -15.6 | 101.0 | 2.17 | 17.81 | 9.617 |
| v1_p3o_replica | 120 | 126 | 128.1 | 8.284 | -339.2 | 403.6 | 2.70 | 21.48 | 28.944 |

## Table 2: lane changes by type — **(b) REPORTED ONLY, not binding**

Textbook MOBIL (owner decision: keep the model) over-produces discretionary LCs, as characterised on TRAIN by WD-01-PA2G-LC1; merge LC is under-produced. Ratios are shown per H; these numbers did NOT enter the verdict.

| model | H | disc pred | disc true | ratio | merge pred | merge true | ratio |
|---|---|---|---|---|---|---|---|
| v2g | 10 | 460 | 90 | 5.11 | 85 | 169 | 0.50 |
| v2g | 30 | 1843 | 241 | 7.65 | 116 | 470 | 0.25 |
| v2g | 60 | 2287 | 300 | 7.62 | 390 | 585 | 0.67 |
| v2g | 120 | 1966 | 303 | 6.49 | 491 | 523 | 0.94 |
| v1_p3o_replica | 10 | 243 | 90 | 2.70 | 380 | 169 | 2.25 |
| v1_p3o_replica | 30 | 618 | 241 | 2.56 | 617 | 470 | 1.31 |
| v1_p3o_replica | 60 | 995 | 300 | 3.32 | 594 | 585 | 1.02 |
| v1_p3o_replica | 120 | 1105 | 303 | 3.65 | 523 | 523 | 1.00 |

## Table 3: missing-arrivals curve + delay-scope note (v2g windows)

| H | n (v2g windows) | truth delay tracked (mean) | truth delay all (mean) | arrivals/window | untracked delay share |
|---|---|---|---|---|---|
| H10 | 153 | 314 | 420 | 8.2 | 0.253 |
| H30 | 153 | 926 | 1277 | 24.8 | 0.275 |
| H60 | 144 | 1577 | 2557 | 49.4 | 0.383 |
| H120 | 126 | 2175 | 5111 | 98.8 | 0.574 |

**Delay scope (registered decision (c)):** the tracked scope. Part B carries the untracked arrivals. The scope share grows with H (27.5 % of truth delay is untracked-arrival delay at H30, 57.4 % at H120), which is why long-H delay errors are scope-dominated, as on TRAIN (A2).

## Table 4: n_free_road_assumed (v2g windows)

| H | n | assumed mean | median | max | share windows >0 | truth-flag mean | truth-flag max |
|---|---|---|---|---|---|---|---|
| H10 | 153 | 2.00 | 2 | 2 | 1.00 | 2.82 | 5 |
| H30 | 153 | 2.00 | 2 | 2 | 1.00 | 2.82 | 5 |
| H60 | 144 | 2.00 | 2.0 | 2 | 1.00 | 2.83 | 5 |
| H120 | 126 | 2.00 | 2.0 | 2 | 1.00 | 2.83 | 5 |

The predictor-side zone-truncation rule assumes 2 free-road vehicles in every window; the truth-exact flag averages 2.82-2.83 (max 5).

## Table 5: labelled `v2g_guards_on` variant (design-deviation comparison; every 3rd world; NEVER headline)

| H | n | s_RMSE | delay_err mean | LC disc pred | coll/1000 |
|---|---|---|---|---|---|
| H10 | 51 | 8.8 | 4.6 | 169 | 0.041 |
| H30 | 51 | 51.1 | 8.2 | 527 | 0.015 |
| H60 | 48 | 106.7 | -79.9 | 656 | 0.010 |
| H120 | 42 | 133.5 | -458.5 | 562 | 0.000 |

Post-LC1-fix the retired guards move nothing material on CAL either (cf. A2's <2 % disc-LC delta on TRAIN).

## Table 6: breakdowns (v2g) by penetration and demand — s_RMSE / delay_err mean

| cell | H30 n | H30 s_RMSE | H30 delay mean | H120 n | H120 s_RMSE | H120 delay mean |
|---|---|---|---|---|---|---|
| pen p0.02 | 34 | 50.8 | 1.7 | 28 | 159.2 | -733.7 |
| pen p0.05 | 34 | 52.1 | 5.3 | 28 | 130.4 | -452.2 |
| pen p0.1 | 34 | 55.5 | -2.1 | 28 | 152.3 | -556.9 |
| pen p0.2 | 34 | 45.9 | -13.5 | 28 | 171.5 | -982.5 |
| pen p0.4 | 17 | 43.6 | -0.7 | 14 | 169.2 | -932.6 |
| dem m2000 | 17 | 51.7 | 15.9 | 14 | 87.6 | 28.5 |
| dem m2600 | 51 | 47.3 | -3.4 | 42 | 171.1 | -911.0 |
| dem m2700 | 85 | 51.8 | -4.7 | 70 | 159.0 | -735.7 |

## Table 7: H60/H120 delay-error sign diagnostics (v2g)

| H | n | >=0 | <0 | mean signed | MAE |
|---|---|---|---|---|---|
| H60 | 144 | 26 | 118 | -154.5 | 181.1 |
| H120 | 126 | 17 | 109 | -709.2 | 735.2 |

## TTC<3 and hard-brake errors (registered always-reported)

| model | H | TTC<3 err mean | TTC<3 err MAE | hard-brake err mean | hard-brake err MAE |
|---|---|---|---|---|---|
| v2g | 10 | -2.18 | 2.19 | -5.20 | 5.58 |
| v2g | 30 | -4.76 | 4.76 | -14.63 | 14.70 |
| v2g | 60 | -4.98 | 4.99 | -15.51 | 15.56 |
| v2g | 120 | -5.12 | 5.12 | -16.85 | 16.89 |

v2g slightly under-predicts TTC<3 exposure and hard-braking counts per window (same direction as A2 on TRAIN).

## Protocol receipts and disclosures

1. **Launch history (all disclosed, none result-driven — no metric number was printed, read, or seen before the final pass):**
   - Attempt 1: crashed BEFORE any row (params assert compared a raw-bytes sha to the registered LF-sha). Receipt: `c4l1_run_attempt1_aborted_paramsassert.out`. The registered one-restart allowance was used.
   - Attempt 2: killed EXTERNALLY by the shell wrapper's 20-min timeout while the laptop was saturated by an unrelated concurrent job (PB-RESCORE1, 4 workers; untouched). 467 shard rows existed (w8621 complete, w8622 partial); NO aggregation, NO numbers seen. Preserved at `results/c4l1/aborted_attempt2/` with ABORTED_NOTE.md. That partial exposed a mechanical deviation: the A2 `hs_ok` rule scores `w+H <= t_end` (t0+600 on these logs) while the registration requires `s+H <= t0+540` — provable from row counts alone (280 rows/world vs the registered 256), outcome-independent.
   - Attempt 3 (FINAL): fixed the horizon to the registered `t0+540`, freeze refreshed and committed before the pass, ONE continuous process over all 9 worlds: 1920 raw rows saved BEFORE aggregation; aggregation re-rolled nothing. This report and the verdict come only from attempt-3 artifacts.
2. **Window-rule note:** the registration states the window rule follows the A2 rule; in fact A2's effective horizon was the log end (t0+600), the registration's text (t0+540) is implemented here and is authoritative. Under the A2 horizon the verdict inputs would shift slightly; the registered rule is what is reported.
3. **Opened-file assertion:** every opened run file belongs to the 9 registered CAL worlds and nothing else (`results/c4l1/opened_run_files_complete.json`; `assert_cal_only` passed on the complete list). Defect disclosed: `summary.json`'s embedded list holds only the 9 worker-recorded physics logs because `_worker` cleared the shared opened-set after the main process recorded the plan + summary/receipt screening opens; the complete list is the deterministic reconstruction, cross-checked against the worker-recorded physics paths byte-for-byte. `eval_c4l1.py` stays byte-frozen (freeze-pinned); patch only under a new registration.
4. **Registration-header disclosures (unchanged):** 5 of the 9 CAL worlds (w8657, w8658, w8674, w8675, w8692) are also in the J4 wall pilot's CALIBRATION split — that does not feed Part A. Part B v0.4 may use chunk-4 CAL for calibration only (disclosed). This look CONSUMED the 9 chunk-4 CAL worlds for the nominal model; a later structural variant (LC2 option 2) needs fresh worlds and a new registration.
5. **Access control held:** the loader refused to start unless the registration LF-sha and the whole freeze matched (verified: engine 0c1cb4051bf5, 6 eval-script sha256s, params, registration); HOLDOUT was refused by split check + the holdout-exclusions guard; no HOLDOUT path appears anywhere in the opened-file lists.
6. **Distributional coverage (gate-v2 (c)) is NOT tested here** — with true-state input the model is deterministic; C4L2 will test coverage over recon samples.

## Artifacts

- Raw rows (1920, saved before aggregation): `results/c4l1/raw_rows.jsonl`
- Summary/aggregation + verdict: `results/c4l1/summary.json`, `results/c4l1/criteria_verdict.json`
- Opened files: `results/c4l1/opened_run_files_full.json` (worker-recorded physics logs), `results/c4l1/opened_run_files_complete.json` (complete list + assertion)
- Run stdout: `c4l1_run.out`; abort receipts: `c4l1_run_attempt1_aborted_paramsassert.out`, `c4l1_run_attempt2_aborted_timeout.out`, `c4l1_run_attempt3_aborted_mkdir.out`, `results/c4l1/aborted_attempt2/`
- Ledger drafts: `output/wd01/LEDGER_DRAFT_C4L1_ALLOCATION.json` (pre-pass), `output/wd01/LEDGER_DRAFT_C4L1.json` (final)
- Freeze: `partA_v2g/C4L1_FREEZE.json`; audit: `partA_v2g/EXPOSURE_AUDIT_C4L1.md`
