# SECOND-PASS (critical) REVIEW — WM-N1 Idea A, piece P3 ROLLOUT2

**Scope:** `output/wd01/wm_n1/rollout2/**` at branch `zcode/revA3g` ==
`zcode/wmn1p3` 5ee16fe. Reviewed against `INTERFACES_V2.md` (frozen
zcode/wmn1ifv2), `REGISTRATION_WM_N1_AMENDMENT_3_IDEA_A.md`
(zcode/wd01 263ea0b) and pre-flight gate **G1**
(`git show zcode/wd01:output/wd01/wm_n1/PREFLIGHT_GATES_IDEA_A.md`).
Reviewer: GLM second pass, 2026-10-01. Zero SUMO; no training; no
subagents; `WD_FORBID_D=1` throughout.

**Verdict: PASS with fixes applied** (all fixes are in `rollout2/equivalence.py`
— the G1 data path; the step/baseline/validity math needed no change).
The critical first-pass defect — G1 verified only on the synthetic
fallback because the selector could never open a real J4 run — is fixed
and G1 now PASSES on **real J4 TRAIN data** (see §9).

## 1. Batched baseline math-identical to partA_v2g — PASS

* Same rule functions imported READ-ONLY from the comparator
  (`baseline.py:43` reuses `partA_v2g.engine.{GAP_MAX, LANES, idm_accel}`);
  `params_nominal.json` is loaded from the partA_v2g file, never copied
  (`baseline.py:46-54`). The labelled v2g-side `lc_variants` diagnostics
  are refused (`baseline.py:64-69`).
* Scalar decision forms mirrored with the gap/v0 floors (`Params.idm1`,
  `baseline.py:117-127` == `engine._idm1`), vectorized dynamics path via
  the engine's own `idm_accel` (`baseline.py:538-561` == engine.py:634-648,
  incl. the accel-lane end obstacle term and `a_clip`).
* LC order identical: mandatory merges front-most-first with occupancy
  refreshed after every acceptance, then discretionary 1->0 and 0->1,
  front-most-first per direction (`baseline.py:468-535` ==
  partA_v2g `engine.py:573-633`); stable s-sorts keep tie order
  (`baseline.py:171-185, 191-196`).
* Merge/discretionary decisions split into (safe, margin) with the exact
  v1-kinematic d-gap + b_safe conjunction and MOBIL politeness/threshold
  arithmetic (`baseline.py:262-345`, `347-413`); safety gates consume no
  residual (§4).
* Commanded CAVs use the RAW (pre-obstacle, pre-clip) own-lane IDM term
  (`baseline.py:553,561` -> `step.py:300-310`) with
  `ctrl.mover.execution_speed` (`ctrl/mover.py:81`), math-identical to
  partA_v2g `engine.py:549-560, 649-658`; 1 s control cadence, command
  held across the two 0.5 s physics steps (`step.py:220-245` ==
  engine.py:568-572).
* End guard, collision counting (never repaired) and `s > lane_len`
  exits are the engine's (`step.py:358-391` == engine.py:662-697).
* Pinned by `rollout2/tests/test_engine_parity.py` (one-step exactness
  on free/merge/LC/dense/end-clamp scenes; 120-tick multi-step;
  replayed commands) and now by the REAL-data G1 run (§9): 4867
  vehicle-tick pairs over 2 J4 TRAIN windows, max error exactly 0.0.

## 2. params_nominal.json unchanged — PASS

* `output/wd01/partA_v2g/params_nominal.json`: git-clean, last touched by
  615ac1f (pre-P3). **LF-sha256 = 079a5576…e75935f7** (matches the
  registered pin; file bytes are CRLF, LF-normalized sha computed).
* `rollout2` never writes it: `load_params` is read-only
  (`baseline.py:51-54`); the equivalence tool labels the set as
  `partA_v2g/params_nominal.json` in every report (`equivalence.py:344-346`).

## 3. D1 integrator correct — PASS

* Frozen rule: `v_new = max(0, v + a*dt)`; `s_new = s + 0.5*(v + v_new)*dt`
  (`validity.py:41-51`, `integrate_d1_vnew` `54-63` used by the engine
  path `step.py:352-354` so an exactly-computed `v_new` (execution rule,
  speed clip) is integrated without a re-clamp round trip).
* Idea A default is d1 (`state.py:84`); the like-for-like `v2g` ballistic
  mode (`validity.py:66-75`; `step.py:355-357`) exists only for G1 and
  also skips the gap repair (`state.py:98-100`), matching the v2g
  engine that never repairs overlaps (engine.py:661). The documented
  per-step difference `s_d1 - s_v2g = 0.5*(v - v_new)*dt` is pinned by
  test (`test_engine_parity.py`, d1-vs-v2g case) and reported in
  `EQUIVALENCE.md`.
* Gap projection (d1 only) places followers at the contact limit and
  reduces speed feasibly, front-to-back per lane
  (`validity.py:103-137`), counted in `ValidityCounts`.

## 4. Validity never lets the residual override safety gates — PASS

* The P2 residual enters exactly two places: the HDV acceleration sum
  (`step.py:247-263`, zeroed on absent/CAV slots `step.py:263`) and the
  MOBIL **incentive margin** (`baseline.py:496-497, 520-521`). The
  `safe` conjunctions (merge d-gap/b_safe tests `baseline.py:297-311`;
  discretionary gap-floor + new-follower b_safe `baseline.py:371-385`)
  never see a residual term.
* No-overlap protection is structural, not learned: `enforce_legality`
  rejects an executed change overlapping a target-lane vehicle in d1
  mode (`baseline.py:499-503, 523-527` + `step.py:267`), and the
  same-lane gap projection runs after integration (`step.py:366-372`).
  Speed never goes negative (`step.py:296-299`, counted).

## 5. Teacher-forced CAV mode — PASS

* Recorded states replace CAV slots BEFORE any decision, so HDVs react
  to the executed motion (`step.py:200-216`; plumbed per-step from
  `engine.py:114-131`); `StepAux.teacher_forced` is reported
  (`step.py:415`).
* No label leak: the HDV network input is the 7-field kinematics view
  only — `s, lane one-hot(3), v, a, length` — never `is_cav`, actions or
  strategy (`step.py:58-71`; INTERFACES_V2 Mismatch 3). The residual is
  forced to 0 on CAV slots regardless of what the net returns
  (`step.py:263`).
* No gradient leak: every recorded tensor is detached at the numpy
  boundary (`_f64`, `step.py:54-55`); the rollout path is pure
  numpy/float64 math, so no gradient flows from the rollout into any
  module (P7's residual NLL consumes one-step tensors separately).

## 6. K-sample determinism — PASS

* One `numpy.random.default_rng(seed)` per `Rollout2Engine.run`, reused
  sequentially across samples (`engine.py:87, 91-144`); fresh
  `Rollout2State` per sample (`engine.py:92-96`). Pinned by
  `test_engine_parity.py` (two samples identical under a seed).

## 7. CUDA readiness — PASS (documented hot path)

* Boundary tensors are the frozen `[B, K]` float32/int64/bool contract
  (`state.py:7-21`); internal float64 is a documented G1-fidelity choice
  with outputs cast back to the input dtype (`engine.py:12-13, 80,
  150-159`; `world_from_batch` keeps float32, `engine.py:165-210`).
* Per-vehicle Python loops exist ONLY in the documented
  faithfulness-critical decision rules (LC/merge/discretionary margins,
  `baseline.py:262-535`; gap projection `validity.py:121-137`) and are
  per-world compacted (`step.py:269-291`); the IDM dynamics path is
  vectorized (`baseline.py:538-561`). This matches INTERFACES_V2
  Mismatch 5 ("P3 reimplements batched") and the module docstrings
  (`baseline.py:5-9`). Non-blocking note for P7: with `B>1` the LC stage
  cost scales linearly in world count; fine for the CPU smoke and G1.
* No `.item()`/`.numpy()` sync inside the vectorized dynamics path;
  counters aggregate at step end (`step.py:285, 298-299`).

## 8. torch 2.5.1 compatibility — PASS

* Ops used are within the INTERFACES_V2 "Torch-2.5 notes" allow-list:
  `torch.{full_like,zeros,full,where,clamp,isfinite,arange,from_numpy,
  stack,no_grad}`; no `torch.compile`, no `torch.amp`/autocast, no
  scaled-dot-product attention, no `torch.export`; no `torch.load` in
  rollout2 (so the `weights_only` 2.5.1-default question does not arise).
* Verified on the local env (torch 2.12.1+cpu) — 43/43 rollout2 tests
  pass; the API surface is 2.5-compatible by construction per the frozen
  allow-list. GPU-box 2.5.1 run remains P7/G5's pinned-env check.

## 9. G1 data-root defect — FOUND AND FIXED (this pass)

* **Defect (critical):** `select_j4_train_run` filtered
  `arm == "native_base"` BEFORE the J4 walls-pilot preference, but the
  J4 chunk-4 package (`campaign_walls_pilot_chunk4`, splits from
  `E:/research-dense/output/campaign_walls_v1/results_plan/WORLD_PLAN.json`)
  has ONLY wall arms (`W-lo-early`, `W-lo-late`, `W-mid-early`,
  `W-mid-late`) — so the J4 branch was dead code, the tool always fell
  through to the workspace (no `runs/`), and the first-pass G1 evidence
  was the labelled synthetic fallback only.
* **Fix (review fix):** `equivalence.py:218-266` — J4 TRAIN windows are
  selected first with ANY complete non-smoke arm (the zero-residual
  window table is built from the run's own physics rows and is
  arm-independent); `native_base` remains the preference for the generic
  chunk-4 fallback only; windows dedupe by `(world, t0_s)`;
  `--n-windows` (default 2, `equivalence.py:358-360`) runs and
  aggregates distinct windows; a missing data root now FAILS loudly
  instead of silently producing a synthetic pass (fallback requires
  explicit `--synthetic`); every path passes `assert_not_d`
  (`equivalence.py:65-71`).
* **REAL G1 result (WD_FORBID_D=1, zero SUMO, TRAIN rows only —
  HOLDOUT/CAL/chunk-5, `evaluate_summary`, `dispatcher_rows`,
  `insertions` never opened):**

```
$env:WD_FORBID_D='1'; python rollout2/equivalence.py --data-root E:/research-dense --n-steps 240
G1 PASS max_pos_err=0.000e+00 max_speed_err=0.000e+00 (tol 1e-06) source=j4_train_inventory
```

  * `w8659m2700p10t420` / `w8659m2700p10t420__w-lo-early`, window t0=450 s,
    30 veh (3 CAV): max position err **0.0**, max speed err **0.0**
    (1903 compared pairs; 10 lane changes exercised).
  * `w8660m2700p10t300` / `w8660m2700p10t300__w-lo-early`, window t0=330 s,
    37 veh (3 CAV): max position err **0.0**, max speed err **0.0**
    (2964 pairs; 18 lane changes exercised).
  * Aggregate 4867 pairs within 1e-6, zero collisions; full stdout in
    `rollout2/g1_j4_train_stdout.txt`, results recorded in
    `rollout2/EQUIVALENCE.md`. Splits were taken from the J4
    WORLD_PLAN (baked into the inventory parquet by
    `inventory/inventory_lib.py:110-114`); both worlds are TRAIN there.

## 10. Non-blocking notes

* `state.py:20` documents `slot_id` as "vehicle identity (-1 empty)";
  only arrivals ever assign `-1`-free new ids and nothing enforces the
  sentinel — harmless, but P9/P11 should not rely on `-1`.
* The engine's inline validity path (`step.py:350-391`) implements the
  same rule set as the exported reference `project_rollout_step`
  (`validity.py:140-213`); the interface tests pin the reference. A
  future refactor should converge them to one implementation.
* `window_table_from_run` (`equivalence.py:268-307`) builds the eval_a2
  zone table; the G1 tool sizes K to the table (`equivalence.py:147`),
  not the corpus K=128 — G1 compares math, not slot layout; corpus
  consumers keep K=128 via `world_from_batch`.

## Test evidence (this workspace)

```
$env:WD_FORBID_D='1'; python -m pytest rollout2/tests -q
...........................................                        [100%]
43 passed in 11.12s
python -m ruff check rollout2
All checks passed!
```
