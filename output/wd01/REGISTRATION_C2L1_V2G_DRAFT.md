# DRAFT (not frozen, not runnable): WD-01-C4L1: Part A v2 general, first CALIBRATION look

**Status:** DRAFT, written 2026-09-30 by the world model session (docs only).

- It replaces the HELD C2L1 (REGISTRATION_NATIVE_PARTA_CHUNK2.md plus
  AMENDMENT_1). Those files are kept unedited, and C2L1 was never run.
- Nothing may run until:
  1. the owner answers the open items in §8;
  2. this file is frozen, and its LF-sha256 is acknowledged by main
     discussion.

**Authority:**

- owner decision 2026-09-29, via main discussion: HOLD C2L1 and re-register
  it for v2 general (PARTA_V2_GENERAL_DESIGN.md §5, §7 Q4);
- owner decision on the native-arm defect: the chunk 1–3 `__native` arms are
  switching mixes, and true native truth is chunk-4 `__native_base`.

## 1. Question

Does Part A v2 general reach pre-specified accuracy and safety criteria at
H = 30 s on fresh, true fleet-native worlds? The model is the one-parameter-set
Treiber IDM + MOBIL + merge engine, deterministic given its input, with no
vType and no insertions.

Scoring is over TRACKED vehicles (contract v2.1, part_a_scope = "tracked").

**Closest prior:** WD-01-PA2G-A2 (REPORT_PA2G_EVAL_A2.md). It used the same
engine (zcode/wd01 9dda3a3), the same truth type (chunk-4 `__native_base`)
and the same metrics code, on 32 TRAIN worlds.

**The exact difference:**

- previously unopened CALIBRATION worlds;
- a pre-specified pass/fail rule;
- one pass.

The C2L1 truth (chunk-2 `__native`) is invalid because it is a switching mix,
not native, so C2L1 cannot be reused.

## 2. Model under test (frozen before the look; no refit)

- **Engine:** output/wd01/partA_v2g at the commit recorded in
  `C4L1_FREEZE.json`, currently 9dda3a3.
- **Parameters:** partA_v2g/params_nominal.json.
  - Current LF-sha256:
    079a55768f803b23c92af188680c1df91dc63f04acbbbac948870b46e75935f7
    (guards OFF).
  - This hash becomes final only after owner flag (a): whether the stability
    guards stay retired.
- **Initial state:** setting (a) of the design, the TRUE logged state
  restricted to the recon zone. Each CAV gets is_cav and the native action
  (CavControllerLoop native), as in A2.
- **Comparators** (scored in the same pass and reported, not tested):
  - constant-speed persistence;
  - the v1 p3o replica with its frozen parameters (sha 458890d9…).

## 3. Criteria (primary H = 30 s; all must hold for PASS)

The thresholds are set here from the TRAIN-only A2 numbers (disclosed). None
of them is chosen from CALIBRATION data. A2 values at H30 are shown in
brackets for context.

- **(a) Safety.** Predicted collisions ≤ 1 per 1000 vehicle-steps [0.04].
- **(b) Lane changes.** Total predicted lane changes per window (merge +
  discretionary) within ±20 % of the truth, as a mean over windows (the gate-v2
  form) [A2: about 2.4×, FAIL].
  - Merge lane changes are also reported separately, within ±20 % [A2: 0.36×].
  - **Open item §8-1:** keep (b) as a binding criterion, or report it only?
- **(c) Delay.** The mean absolute window delay error (tracked scope) is
  below const-speed's MAE on the same windows [30.3 vs 61.3], AND the mean
  signed error lies in ±15 % of the mean truth tracked delay [+2.8 vs about
  1249].
- **(d) Speed.** v_RMSE ≤ 1.05 × min(v_RMSE of const-speed, v_RMSE of the p3o
  replica) [5.02 vs 5.90].

**Distributional coverage (gate-v2 (c)) is NOT tested in this look.**

- With a true-state input the model is deterministic, so coverage is only
  defined over recon samples.
- It becomes a separate registered look (C4L2) once the reconstructor
  provides K recon samples for these worlds from a recon model that was never
  fitted on them (§8-3).

**Always reported (not gate items):**

- H ∈ {10, 60, 120} with the same statistics;
- s_RMSE;
- TTC<3 and hard-brake errors;
- lane changes by type;
- n_free_road_assumed;
- the missing-arrivals curve (tracked vs all-vehicle truth delay);
- breakdowns by demand and by penetration;
- the labelled guards_on variant.

## 4. Data

- **Plan:** output/campaign_stage1_v3/results_plan_chunk4/WORLD_PLAN.json,
  sha256 (raw bytes) 65510b27f651f477c3de9acc663b9138916a07dfb46468718dc9932147e1ffc4.
- **Worlds:** all 9 `CALIBRATION` worlds (chunk 4 has no VALIDATION split).

  | world | demand (veh/h) | penetration | t0 (s) |
  | --- | --- | --- | --- |
  | w8621m2700p02t240 | 2700 | 0.02 | 240 |
  | w8622m2700p02t360 | 2700 | 0.02 | 360 |
  | w8639m2700p05t360 | 2700 | 0.05 | 360 |
  | w8640m2000p05t540 | 2000 | 0.05 | 540 |
  | w8657m2700p10t300 | 2700 | 0.10 | 300 |
  | w8658m2600p10t360 | 2600 | 0.10 | 360 |
  | w8674m2600p20t300 | 2600 | 0.20 | 300 |
  | w8675m2700p20t300 | 2700 | 0.20 | 300 |
  | w8692m2600p40t300 | 2600 | 0.40 | 300 |

- **Truth:** `<world>__native_base/run/physics_steps.jsonl` over (t0, t0 + 540].
  - A run counts only with a receipt exit_status "complete" and coverage.
  - A refused world is reported and is not replaced.
  - CAV identity comes from the physics-row cav_ids, never from
    insertions.jsonl or dispatcher_rows.jsonl.
- **Windows:** starts s = t0 + 30k with s + H ≤ t0 + 540. This follows the A2
  window rule; the exact count is computed by the code, not assumed.
- **Access control:**
  - The loader refuses these 9 worlds unless it is given this frozen file with
    a matching LF-sha256.
  - HOLDOUT is blocked via tools/holdout_guard.py and
    protocols/holdout_exclusions.json.
  - The opened-file list is asserted at close-out.
- **Exposure audit BEFORE freezing** (registry absence is not proof):
  - Confirm that no Part A code, Part A analysis or tuning has opened these 9
    CAL runs. The A2 opened-file list is TRAIN-only, which I have verified.
  - Confirm that no reconstructor or planner artefact derived from them feeds
    Part A.
  - Chunk-4 CAL use by a future Part B build (for calibration of V) does not
    touch Part A and is disclosed.

## 5. Protocol

- **Before the look:**
  - commit the evaluation code;
  - write `C4L1_FREEZE.json`: engine commit, params sha, the eval script
    sha256s, this file's sha;
  - send the freeze sha to main discussion.
- **ONE pass.** Raw rows are saved before aggregation; aggregation re-rolls
  nothing.
- **Crash rule:** a crash before any row is written may be restarted once,
  with disclosure. A crash after rows exist is reported as is.
- **FAIL:** diagnosis continues on TRAIN only. There is no second look on these
  9 worlds. A changed model needs fresh worlds and a new registration.
- **PASS:** "Part A v2 general meets C4L1 at H = 30 on the true state". That
  claim does NOT extend to recon-sample inputs (C4L2).
- **Records:**
  - the ledger entry is appended by the integrating agent;
  - OPEN_QUESTIONS and ROUTE are updated.

## 6. Interaction with other uses

- The planner may use Part A v2 general on TRAIN worlds for development
  without affecting this look.
- Chunk-2/3 CAL worlds stay unopened by Part A. They are Part B's
  CALIBRATION, and their `__native` truth is a switching mix anyway.

## 7. Budget

Zero simulation, CPU only. From A2, 32 worlds took 1377 s at 4 workers
(including the p3o replica). This look is under 10 minutes, as one ZCode task.

## 8. Open items (owner, via main discussion) — must be resolved before freezing

1. **Lane-change criterion (b).** The TRAIN A2 run already shows a large miss:
   discretionary lane changes are over-produced 4.6–6×, and merges
   under-produced at 0.36×. Running a binding look now would spend fresh CAL
   worlds on a predictable FAIL.
   - Option 1 (recommended): first resolve flag (b), a bounded investigation
     of the residual discretionary over-production and merge
     under-production, done on TRAIN, with no parameter tuning unless the
     owner approves it.
   - Option 2: make (b) reported-only.
2. **Guards (flag (a)).** Should the stability guards stay retired in the
   frozen nominal set? This fixes the params sha.
3. **C4L2 (recon-sample coverage).**
   - Which recon model produces the K samples for these 9 worlds? It must not
     have been fitted on them.
   - What K?
   - Is the coverage target the physics-delay 10–90 % band ≥ 0.80 (gate v2
     (c)) over K samples?
4. **Delay scope (flag (c)).** Criterion (c) is tracked-scope only. The
   untracked-arrival remainder is Part B's job and is not judged here.
