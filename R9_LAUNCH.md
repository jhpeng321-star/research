# R9 launch (S-FINAL): counted exclusion, integration, qualification, launch

Branch `c4/v10r9excl` (worktree `E:/agent-work/c4-v10r9excl`, from `c4/v10r9int2` d5a528d7). Date 2026-10-22/23.

## Commits

| commit | content |
|---|---|
| ebc901ae | counted exclusion: one shared function `merge_pivot_support.counted_exclusion` used by `MergePivotTerm.prepare`, `tools/r9_fit_exact.drop_counted_exclusions`, `tools/r9_support_audit.audit_episode` (and reconcile/report) |
| 983c947e, 22f6268a | merge `c4/v10r9int2` (R9_OBJECTIVE 953ceaff TF+F2 only, `merge_core.weight = 0`; c584d46f fitter TF normalisation) |
| 75294980 | merge `c4/v10r9m0` (88e99b5e, R9_GATE2) |
| 3ed8311e | merge `c4/v10r9tdiag` (84842fee entrant-slot geometry capture; landed before 00:45) |
| 7880bedf | gate verdict edited per owner (on-policy rules replaced by `merge_pivot_on_objective_disabled == 1` on every row; TF mass / zero-mass / grad / stages kept; excluded-motion totals and rate reported). cpuN fit ran on this commit |
| f91d8753 | stage-2E consistency (below). Mask semantics identical to 7880bedf; adds reporting, loud failure on non-finite eligible geometry |
| 3e5610fe | gate: stage-end validation capped (see "Gate attempt 1"); this is the SHIPPED and LAUNCHED commit (`repo_r9_3e5610fe` on gpuR) |

Not used: `c4/v10r9motion` (postponed). Not pushed.

## The exclusion (owner decision 2026-10-22, counted, no threshold)

A risk chain (one car, one identity) is removed from the loss WHOLE (its event and its waiting factors; an excluded event is never turned
into a censor; the identity tail is dropped so a re-entry is not born as a new chain) iff its exact likelihood is zero for every parameter value:
1. `event_vetoed_explained | event_vetoed_wm_motion | event_vetoed_other`: the recorded event step has E_T = 0 (vetoed under the WM one-step geometry);
   split by the existing classifier (vacated leader / junction occupant; successor clear = WM motion; other).
2. `event_zero_clearance`: eligible but a present side with zero clearance (c = -inf for every parameter).
3. `prefix_survival_zero`: the certified F2 pre-window prefix contains a certain passage (eligible, no neighbour either side, c = +inf) -> denominator exactly 0.
4. `wait_survival_zero`: an in-window step of a certain passage that did not end in the event (event or censor).
Ordinary censored cars are kept (also if ineligible at the censor step). No floors, thresholds, padding or tolerances. Non-finite speeds/gaps on an
ELIGIBLE row raise `ValueError` (numerical error, not a structural zero). The decision depends only on geometry, never on the 8 boundary parameters,
and only those 8 (`merge_pivot.*`) are trainable (verified from the preset; `freeze_prefixes` covers the other 144 tensors), so cohort membership cannot
move during training. Cohort version `r9-excl-1` (logged as `merge_pivot_support_exclusion_version = 1`; audit `summary.json` and fit `cohort` carry it).

Logged per training row: `merge_pivot_tf_excluded_motion` (excluded scored cars), `merge_pivot_tf_scored_cars`, `merge_pivot_support_{original,retained}_{cars,scored_cars,events}`,
`..._retained_censor_cars`, `..._excluded_{cars,event_cars,censor_cars,overlap_cars}`, per-class `..._excluded_<class>` (disjoint primary) and `..._excluded_bit_<class>` (overlapping),
`..._scored_excluded_*` (scored cars only), `..._empty_retained_batch`. Gate smoke JSON `metrics.support_totals` carries the sums, `tf_excluded_motion_rate` the rate.
All-excluded batch: finite zero loss, logged (`test_all_excluded_batch_has_finite_zero_loss_and_is_logged`).

Tests added: `tests/test_merge_pivot_exclusion.py` (13: infinity-pattern parity with the boundary, classes/priority/overlaps, supported chain bit-identical, tail, censor rules, batch
invariance, NaN raises, fitter-flow == training-flow), `test_merge_pivot_support.py` (+4: all-excluded batch, parameter independence of the cohort, training-vs-audit parity on one scene),
`test_r9_support_audit.py` (+1 audit wiring), gate verdict/launcher tests updated to the owner's gate edit.

## Tests

Round-9 suites (`tests/test_merge_pivot*.py tests/test_r9_*.py tests/test_checkpoint_forward_exact.py tests/test_round9_gate_*.py tests/unit/config
tests/test_merge_fit_stage2a.py tests/test_merge_fit_v2.py tests/test_route_frame.py tests/test_merge_core*.py tests/test_merge_context.py`, `--basetemp` inside the worktree,
`PYTHONPATH=src;.;tests` because `test_r9_tdiag.py` imports a sibling test module by bare name): **937 passed** at f91d8753. After 3e5610fe (script + launcher test only): `test_round9_gate_launcher.py` 9 passed.
No test weakened; the changed gate / support tests encode the owner's decisions.

## Support count: all 192 TRAIN episodes (cpuN, f91d8753, `tools/r9_support_reconcile.py`, round-9 physics, results `~/r9fit_sfit/work_f91d8753/results/r9_support_excl`)

| quantity | value |
|---|---|
| episodes | 192 (96 v5_native, 96 genrand_plain), 1536 analysis windows |
| HDV recorded merge events (audit windows, warm-up excluded) | 14,676 |
| supported | 14,563 |
| counted exclusions (cars = events) | **113 = 0.77 %** (v5_native 43, genrand_plain 70) |
| - event_vetoed_explained (vacated leader / junction occupant) | 95 |
| - event_vetoed_wm_motion (successor clear: WM's own motion) | 15 |
| - event_vetoed_other | 3 |
| - event_zero_clearance / prefix_survival_zero / wait_survival_zero | 0 / 0 / 0 |
| censored cars excluded | 0; overlapping causes 0; unique excluded cars 113 |
| unexcluded vetoes | 0 |
| F1 / exact M0 zero-mass events on the retained set | 0 of 14,563 / 0 of 14,563 (all positive) |
| audit-vs-fitter mismatches | 0 (fitter dropped 106 of the 113; the other 7 are outside its cohort) |
| deprecated strict P1-P3 law zero-mass | 4,145 (that law is not used) |

## Fit: F1 vs exact M0 on the IDENTICAL cohort (cpuN, 7880bedf, 154 fit worlds, 11,651 cars / 11,640 events; audit worlds NOT opened, `audit_worlds = 0`)

One counted-exclusion mask per run (shared frozen-checkpoint geometry cache), applied before any law is scored: F1, exact M0 and all comparators score the same retained
sequences. Selected-cohort TF+F2 scores, not a full-population likelihood. Selected penalties (tau, w) = (0, 0).

| quantity | exact M0 | M0-refit | F1 |
|---|---|---|---|
| NLL / car (11,651 cars, finite for all) | 7.256 | 2.528 | **2.450** |
| out-of-fold NLL (4 inner folds of 39/39/38/38 held-out worlds) | 7.25 / 7.30 / 7.17 / 7.30 | - | 2.453 / 2.447 / 2.443 / 2.461 |
| early mass (mean) | 0.421 | 0.386 | **0.387** |
| zero-mass events | 0 | 0 | 0 |
| signed median timing error | 0.0 s | 0.0 s | 0.0 s |
| \|median timing error\| (not point-identifiable: 40 % of predicted medians unreached at the horizon) | bounds [0, 34.0] s | [0, 13.0] s | bounds [0, 38.5] s |

Paired world bootstrap (154 worlds, car-weighted per-world means, 4000 resamples, seed 73): F1 - exact M0 = **-4.806 nats/car, 95 % CI [-5.001, -4.615]**, F1 better in 154/154 worlds;
F1 - M0-refit = -0.078 [-0.135, -0.026], better in 76/154 worlds. Early-mass reduction vs M0 -0.034. **Launch condition met.**
Fitted F1 (diagnostic, NOT used as init): s_f 2.65, T_f 0.27, b_f 0.99, s_l 1.18, T_l 0.32, b_l 1.18, tau 0.281, w 0.255. Run init is the round-8 stage300 checkpoint (tau 0.073, w 1.0).
Files: `E:/agent-work/r9fit-stage/results_7880bedf/direct/r9fit_excl1_{F1,SH,P}.{json,md,log}`.

## Gate attempt 1 (stopped, not a model failure)

`repo_r9_f91d8753`, gate PID 297883 / train 297886, stopped by me by PID after 52 min (artifacts renamed `*_a1`, nothing deleted). Steps 1-2 (120 s, batch 4): finite loss, 23/23 positive TF mass,
0 zero-mass, peak 2.4 GB, 77 s and 195 s. Then the stage-end long-horizon validation (preset `val_max_windows = 256`, `val_all_horizons`) ran for 40+ minutes: it executes the full merge_pivot
`prepare` (2120-step F2 prefix replay) per eval batch. cProfile of one 300 s step on gpuR (`/root/autodl-tmp/c4/v5/prof_r9excl.prof`): train step 234 s, then 3 eval batches 377 s (125 s each) -> 64 batches x 4-5 horizons would take hours,
and the 0.5 h smoke cap could never be met; the production run would have the same cost at every stage end. Operational fix in `scripts/round9_gate.sh` (3e5610fe): smoke `selection.val_max_windows=4`, `curriculum.max_hours=1.5`;
production `selection.val_max_windows=16`. No owner decision touched (monitoring/selection windows only; loss, data, config recipe unchanged). Note: the preset keeps `val_max_windows = 256`; any other launch of this config pays this cost.

## Gate attempt 2: smoke verdict OK, full run launched automatically

`/root/autodl-tmp/c4/v5/repo_r9_3e5610fe`, `scripts/round9_gate.sh`, log `round9_gate_3e5610fe.out`, smoke log `smoke_r9.log`, verdict `smoke_r9.verdict` = `smoke: ok`, report `smoke_r9.gate.json`, smoke run `runs/smoke_r9`.

| smoke quantity | value |
|---|---|
| optimizer steps | 4 (2 at 120 s, 2 at 300 s), batch 4 |
| step time | 77.6 s, 199.3 s (120 s); 180.0 s, 259.4 s (300 s) |
| peak CUDA memory | 3,309 MB (2,264 / 2,370 / 3,309 / 3,309 per step) |
| TF positive-mass events / zero-mass | 81 / 0 |
| `merge_pivot_on_objective_disabled` | 1 on every row |
| scored cars / excluded_motion | 85 / 0 (rate 0.0; smoke windows contain no impossible car) |
| merge_pivot grad norm (sum) | 0.045 |
| scored seconds >= 120 / >= 300 | 3,237 / 1,565 |
| checkpoints reloaded (strict, CPU) | stage120, stage300, final; sha256 953cf8bb..., 23ee963f..., c7301445... |
| smoke total wall time | ~52 min (training ~12 min, validation the rest) |

**Full run: LAUNCHED** (the gate `exec`s it): PID **299868** (`python -m wms train --config configs/c4_wm_v10round9.toml --mode run --out /root/autodl-tmp/c4/v5/runs/c4_wm_v10_round9 ...`),
run dir `/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round9`, log `/root/autodl-tmp/c4/v5/round9_gate_3e5610fe.out`, `train_log.jsonl` in the run dir. Stages 120 s (30 steps, batch 4) and 300 s (60 steps, batch 2),
`max_hours = 2.5`, `val_max_windows = 16`. First two steps healthy (78 s, 197 s; 24 scored cars, 0 excluded, 0 zero-mass).
gpuR disk is tight (1.6 GB free of 50 GB at launch; the run itself needs ~20 MB). Own scratch left on gpuR: `repo_r9_f91d8753`, `ship_r9_*`, `prof_r9excl.*`, `runs/prof_r9excl`, `runs/smoke_r9_a1`, `smoke_r9_a1.log`, `.round9_gate.lock_a1`.

## Caveats / honest notes

- The cpuN fit used 7880bedf; the audit/support numbers used f91d8753; the shipped code is 3e5610fe. The mask logic is byte-identical across the three (later commits add reporting, a loud failure on non-finite eligible geometry, and the script cap). Not re-fitted after 7880bedf.
- Support count covers the 192 TRAIN episodes' physical diagnostics (reconcile); the fit's held-out audit worlds stayed unopened (`--audit-once` never passed). "Held-out" above = inner folds / in-sample per-world paired bootstrap on fit worlds.
- The exclusion applies to chains regardless of scoredness; scored vs total counts are both logged. In the 192-episode count all 113 excluded cars are scored event cars.
- Fine-tuned F1 parameters (tau 0.28, w 0.26) were NOT injected into the run init (not part of the owner decision).
- The preset still carries `val_max_windows = 256`; the cap is applied in the gate script only.
