# REGISTRATION: WD-01-PA2G-LC1: Part A v2 general lane-change mismatch investigation (TRAIN only, zero SUMO)

**Registered:** 2026-09-30 by the world model session, before execution.

**Authority:** owner decision (b), 2026-09-30, about 02:05
(OWNER_DECISIONS_WD01.md).

**Type:** bounded diagnostic ANALYSIS. It is not a validation look, and it
produces no acceptance claim.

## 1. Question

Why does Part A v2 general, in PA2G-A2 on chunk-4 `__native_base` TRAIN
truth, get the lane-change counts wrong in two ways?

- It over-produces discretionary lane changes: 2044 predicted vs 443 true at
  H10, and 7323 vs 1393 at H120, i.e. 4.6–6×.
- It under-produces merges: 390 vs 611 at H10, and 631 vs 1762 at H30.

Each part of the mismatch is attributed to one of four classes:

- **(i) Implementation bug** in engine.py, adapter.py or eval_a2.py.
- **(ii) Measurement or classification mismatch.** Truth and prediction must
  be counted identically. Candidates:
  - junction/internal-lane rows;
  - the lane_phys mapping;
  - a merge counted as discretionary, or the reverse;
  - lane changes that happen outside the zone or after an exit;
  - the 0.5 s sampling.
- **(iii) Model-structure difference.** Textbook MOBIL + IDM + the merge
  rule, with its nominal parameters, differs from SUMO LC2013 behaviour.
  Examples:
  - no anticipation or strategic lane choice;
  - the symmetric vs asymmetric criterion;
  - how the keep-right bias works;
  - the merge-zone geometry and the timing of the urgency term;
  - vehicles entering the accel lane.
- **(iv) Input-state effects.** The recon-zone truncation (a free-road
  leader), and vehicles starting mid-manoeuvre at window start.

**Closest prior:** WD-01-PA2G-A2 (REPORT_PA2G_EVAL_A2.md) and its
DIAGNOSIS.md (the `_neighbors` swap fix). The difference is that this
investigation decomposes the RESIDUAL mismatch that remains after that fix.

## 2. Model and data (unchanged)

- **Engine:** zcode/wd01 at 257ac6c or later (partA_v2g as merged in
  9dda3a3).
- **Nominal parameters:** params_nominal.json, LF-sha256
  079a55768f803b23c92af188680c1df91dc63f04acbbbac948870b46e75935f7. The
  owner fixed this set with guards retired (decision (a)).
- **Worlds:** the 32 PA2G-A2 TRAIN worlds of chunk 4, the same stratified
  selection, with the list read from the A2 summary/opened-files record.
  - For diagnosis only, the other chunk-4 TRAIN worlds may be added. The
    budget in §4 still applies.
  - No CALIBRATION or HOLDOUT file is opened, not even to inspect formats. A
    split check runs before every open, and the opened-file list is asserted
    at close-out.
- **Truth:** `<world>__native_base/run/physics_steps.jsonl`. CAV identity
  comes from the physics-row cav_ids.

## 3. Method (diagnosis first)

1. **Counting audit, done FIRST.**
   - Re-count truth lane changes from the raw physics rows with an
     independent counter.
   - Classify each true lane change by (from lane_phys, to lane_phys,
     s_down position, internal/junction flag).
   - Show whether truth and prediction use identical definitions.
   - Report any correction as class (ii), with a test.
2. **Decomposition.**
   - Predicted vs true lane changes by: lane pair; s-position bin (10 m);
     local density and speed regime; CAV vs HDV; time since window start
     (early spike from the start state?); changes per vehicle (repeat
     changers); change followed by a change back within 10 s.
   - For merges: the merge position distribution (predicted vs true), the
     time from accel-lane entry to merge, stop-at-lane-end events, and
     vehicles that never merge within H.
3. **Per-event traces.**
   - For about 10 worst windows, give per-vehicle traces of the MOBIL terms
     (ego gain, new-follower and old-follower terms, bias, threshold,
     safety) for predicted-but-not-true changes, and the truth trajectory
     beside them.
4. **Descriptive ablations.** These are NOT selection and NOT a
   parameter change to the nominal set; they only indicate which mechanism
   matters. Each is labelled and reported, and none is adopted:
   - politeness ∈ {0.2, 0.5, 1.0};
   - keep-right bias ∈ {0, 0.3};
   - threshold ∈ {0.1, 0.3};
   - the textbook asymmetric (European) MOBIL criterion vs the symmetric one;
   - merge-urgency start at the zone start vs a later point.

   Adopting any of them requires a further owner decision.
5. **Fixes.** Only class (i) bugs and class (ii) counting errors may be fixed.
   - Each fix gets a regression test.
   - After fixes, the A2 metrics are re-run on the same 32 worlds, labelled
     "post-LC1 fixes", with the before/after table.
   - The nominal parameters are NOT changed.

## 4. Budget and stop rule

- Zero SUMO.
- CPU at most 3 h in total at --workers 4 or fewer (A2 took 1377 s for 32
  worlds).
- One ZCode task (Flash, 240 min).
- Stop and report when the budget is reached, or when the mismatch is
  attributed to classes with evidence, whichever comes first.

## 5. Outputs

- partA_v2g/DIAGNOSIS_LC1.md: the counting audit, decomposition tables,
  traces, the ablation table (labelled descriptive), and the class
  attribution with rough shares;
- recommended owner options, for example:
  - keep the textbook model and make C4L1 (b) reported-only;
  - approve a specific structural addition;
  - approve a parameter change on TRAIN evidence.
- Raw rows are saved before aggregation.
- LEDGER_DRAFT_PA2G_LC1.json in the experiment_history ENTRY schema (id, name,
  controller, question, scenario, budget_registered, budget_actual, results,
  criteria_verdict, limitations, artifacts).
- Every report states the delay-scope note (owner decision (c)): Part A is
  tracked-scope, and Part B carries untracked arrivals.
