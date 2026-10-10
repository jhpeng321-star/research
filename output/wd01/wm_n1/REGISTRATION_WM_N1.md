# REGISTRATION (FROZEN): WD-01-WMN1: fully neural world model, and neural + tree combination

**Frozen:** 2026-09-30 by the world model session. It is derived from
REGISTRATION_WM_N1_DRAFT.md, which is kept unchanged.

**Owner decisions applied (2026-09-30, relayed by main discussion):**

- the §4 targets are APPROVED as stated;
- chunk 5 will be GENERATED as the TEST set (busy loading only, wall arms
  included);
- three variants (N, T, C), with the planner's variant chosen on CAL before
  TEST;
- the fallback is to keep today's hybrid.

**Design:** WM_N1_DESIGN.md (same directory).

**Gates before training:**

- (1) a ledger allocation per step;
- (2) the 1-h GPU timing gate, approved by the owner;
- (3) the chunk-5 TEST generated and graded for integrity only, with its
  outcomes SEALED.

**Gates before the TEST read:** PLANNER_VARIANT.json frozen, and the WM-N1
freeze sidecar recording the model shas and the chunk-5 registration sha.

## 1. Question

On FRESH test worlds, does one question hold? Can a fully neural world model
(N), or a CAL-selected combination of it with tree Part B (C), predict the
policy contrasts ΔV(damping / PI / wall vs native) better than the current
hybrid (Part A v2g plus Part B tree, T)? The outcome accuracy, the
short-horizon accuracy and physical validity must not be worse.

## 2. Models compared (the same TEST windows for all)

- **N:** WM-N1 pure neural, a 5-member ensemble.
- **T:** the hybrid.
  - Part A v2g, frozen nominal (params 079a5576…).
  - Part B tree: v0.4a, or v0.4b if walls exist, as registered. The frozen
    shas are recorded at freeze.
- **C:** the combination.
  - Primary: stacking plus the disagreement signal. w_τ is fitted on CAL by
    world-grouped 2-fold cross-fitting, followed by conformal recalibration
    on CAL.
  - Ablations:
    - residual (a neural head on OUT-OF-FOLD tree predictions);
    - an IDM/MOBIL physics prior plus a learned residual on the step
      decoder.
- **Planner variant:** chosen on CAL BEFORE TEST, by the design §5 rule. It
  is frozen in PLANNER_VARIANT.json (with its sha) before the TEST read.

## 3. Data and splits

- **TRAIN:** every TRAIN world of chunks 1–4, the J3 truearms and the J4
  wall branches (when available).
- **CAL:** the chunk 2–4 CAL worlds, used ONLY for early stopping,
  calibration, blend weights and the variant choice.
- **HOLDOUT:** sealed.
- **TEST: a NEW chunk 5** (owner-approved 2026-09-30: generate it; busy
  loading {2400, 2600, 2700} only; wall arms included). It has its own frozen
  registration, REGISTRATION_CHUNK5_TEST.md, whose LF-sha256 is recorded in
  the WM-N1 freeze sidecar before any TEST read. Reconstructor allocation:
  - 30 fresh worlds (3 demand × 5 penetration × 2);
  - per world: base, native_base, restored native, PI, 2 fixed, 2 switching
    and 4 wall arms;
  - about 207 k sim-s.
  - Outcomes are sealed until the single final evaluation.
- **Leakage rules:**
  - OOF tree predictions on TRAIN (world-grouped 5-fold);
  - CAL-only blend weights and calibration;
  - no TEST read before PLANNER_VARIANT.json is frozen;
  - one TEST pass.
- **Inputs exclude:** driver type, true penetration, pre_fix, speed_mode,
  world strings, seed and t0.

## 4. Targets and decision rule (OWNER-APPROVED 2026-09-30; fixed before training)

**Primary (planner relevance):** "The planner variant beats the hybrid" iff,
on TEST, ALL of:

1. ΔV CRPS is lower than T's for ≥ 2 of the gating contrasts {damping−native,
   PI−native, wall−native (if available)} at ≥ 2 of the τ {60, 120, 300}. The
   world-bootstrap 90 % CI of the difference must exclude 0.
2. Outcome CRPS is not worse than T's by more than 5 % at any τ (delay; the
   safety heads are reported).
3. H10/H30 s_RMSE is not worse than Part A v2g by more than 10 %.
4. Collisions ≤ T + 0.1 per 1000 veh-steps. Interventions are reported.

**Also reported per variant (N, T, C):**

- sign accuracy with CIs;
- coverage and PIT;
- the s/v/delay % errors at H10–H120;
- the disagreement-signal correlation and abstain curve;
- the PL-NOISE1 budget check, when that exists.

**If the primary rule fails,** the hybrid stays the working version. N or C
may still be reported as a research result.

## 5. Budget (to be finalised after the timing gate)

- **Featurization:** laptop or rented CPU, about 4–6 h at 4 workers, zero
  SUMO.
- **Training on the rented GPU:** a 1-h timing gate first. The estimate is
  25–50 GPU-h for 5 members, plus 30 % for rollout fine-tuning and 2 GPU-h
  for evaluation.
- **Chunk-5 TEST:** about 207 k sim-s on the rented CPU box, as a separate
  registration.

## 6. Records

- A LEDGER allocation draft is committed before each allocation (the
  featurization, the timing gate, the training, the TEST pass), in the
  experiment_history ENTRY schema.
