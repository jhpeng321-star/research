# REGISTRATION: WD-01-PA2G-LC2: would a standard structural lane-change addition be a big improvement? (TRAIN only, zero SUMO, sensitivity)

**Registered:** 2026-09-30 by the world model session, BEFORE execution.

**Authority:** owner decision, 2026-09-30 about 09:55 (LC1 option 1 plus
this sensitivity study).

**Type:** descriptive SENSITIVITY ANALYSIS.

- Nothing changes in the nominal Part A v2g model.
- Every variant is OFF by default, and its value is chosen a priori.
- No fitting, and no selection feeds the nominal model.

## 1. Question

LC1 found that the residual discretionary lane-change over-production
(4.6–6×) and the late merging are MODEL STRUCTURE. The causes are:

- instantaneous MOBIL at dt 0.5 s;
- no lane-hold or manoeuvre time;
- no strategic merging.

Would a standard structural addition from the literature improve the
accuracy of Part A v2g substantially, enough to justify owner option 2?

**Closest prior:** WD-01-PA2G-LC1 (DIAGNOSIS_LC1.md) and WD-01-PA2G-A2. The
engine, truth, worlds and metrics code are the same. The difference is the
structural variants.

## 2. Variants (each OFF by default; ONE a-priori value; never tuned)

- **V-a, persistence:**
  - A discretionary change fires only if the MOBIL criterion (incentive +
    safety) has held continuously for T_persist = 2.0 s (4 steps at dt 0.5).
  - This is a standard anti-flicker / decision-persistence device. The value
    is chosen a priori as 4 integration steps. It is NOT fitted.
- **V-b, manoeuvre time:**
  - After any lane change, the same vehicle may not start another
    DISCRETIONARY change for T_lc = 4.0 s. The change itself stays
    instantaneous.
  - Empirical lane-change durations are reported at around 4–5 s (e.g. Toledo
    & Zohar 2007). The worker states the exact citation it relies on, or
    marks it "to verify". The value is chosen a priori, not fitted.
- **V-c, strategic early merge:**
  - A vehicle on the acceleration lane applies the mandatory-merge rule from
    the START of the merge zone with zero incentive threshold, plus a merge
    bias of b_merge = 1.0 m/s² added to its MOBIL gain toward main_right. The
    safety test is unchanged.
  - This follows the on-ramp bias convention of Treiber/Kesting-style MOBIL
    implementations. The value is chosen a priori.
- **V-abc:** all three combined.

The nominal model is params_nominal.json, sha 079a5576…, unchanged. The
variants are implemented as flags with the default OFF, and a test proves
the nominal path is byte-identical with the flags off.

## 3. Data and metrics

- **Worlds:** the same 32 PA2G-A2 TRAIN worlds of chunk 4
  (lc1_a2_worlds32.json).
  - Windows: (t0, t0 + 540] every 30 s.
  - H ∈ {10, 30, 60, 120}.
  - The post-LC1 counting convention applies.
- **TRAIN only:** no CAL/HOLDOUT file is opened; the split is checked before
  each open, and the opened-file list is asserted.
- **Metrics** per variant and H:
  - s_RMSE and v_RMSE;
  - delay error (signed and MAE);
  - the discretionary-LC and merge-LC predicted/true ratios;
  - merge dwell (median) and the early-edge merge share;
  - dead-end stops;
  - collisions per 1000 veh-steps;
  - TTC<3 and hard-brake errors.
- **Uncertainty:** a world-bootstrap 90 % CI on each variant − nominal
  difference.

## 4. Decision rule (fixed BEFORE running)

A variant is a **BIG IMPROVEMENT** if BOTH conditions hold.

**(1) Benefit.** At least one of:

- s_RMSE at H = 30 AND at H = 60 each ≥ 20 % lower than nominal, with the
  world-bootstrap 90 % CI of the relative change excluding 0;
- the discretionary-LC ratio at H = 30 is within [0.67, 1.5] AND the merge-LC
  ratio at H = 30 is within [0.67, 1.5]. Nominal is 5.5× and 0.37×.

**(2) No harm.** All of:

- H10 and H30 s_RMSE not worse than nominal by more than 5 %;
- the H30 delay MAE not worse by more than 10 %;
- collisions ≤ nominal + 0.1 per 1000 veh-steps.

**Otherwise:**

- MODEST if benefit (1) is only partly met and no-harm (2) holds;
- NO IMPROVEMENT otherwise.

**Recommendation to the owner:**

- option 2 is worth registering only if a variant is a BIG IMPROVEMENT;
- the recommendation names the smallest such variant.

**Caveat, stated up front:** TRAIN-only evidence with a priori values. A
variant adopted later needs fresh evaluation worlds, because C4L1 consumes the
chunk-4 CAL for the nominal model.

## 5. Budget

- Zero SUMO.
- CPU: 5 runs (nominal plus 4 variants) × 32 worlds. From A2, about 1377 s at
  4 workers per full run including the p3o comparator. The comparators are
  not re-run, so the total is ≤ 1.5 h at ≤ 4 workers.
- One ZCode task (Flash).
- The Option-B wall work has CPU priority.

## 6. Outputs

- partA_v2g/REPORT_PA2G_LC2.md:
  - the variant specs with their literature provenance;
  - the full table with CIs;
  - the decision-rule verdict per variant;
  - the recommendation.
- Raw rows are saved before aggregation.
- LEDGER_DRAFT_PA2G_LC2.json in the experiment_history ENTRY schema.
