# WM-S-R5 post-run diagnosis plan (pre-registered, 2026-10-12)

Written BEFORE the round-5 GPU run, on branch `wms/r5-prep`. Scope: attribution of the
residual 120 s error of the round-5 model (`configs/r5_f23.toml`, one combined GPU run)
against the recorded truth, using EVALUATION-ONLY tooling. Nothing in this plan changes
training; the inference-time switches below are default OFF and live in the scorer
(`eval.flags.diag_*` / `wms.eval.predictor.RolloutFlags`), never in model training code
or configs (owner rule: no driver-type list, labels or true parameters in
model/training code).

## Questions (ordered)

Q1. How much of the 120 s position error is LANE-decision error vs longitudinal error
    on correctly-laned cars?
Q2. Does the learned per-driver settings encoder (FIX 1) help or hurt at 120 s,
    relative to population-default physics settings?
Q3. Is the late-horizon error still jam-wave timing (R4-DIAG-1/3), visible as survivor
    speed drift after 60 s?
Q4. Do the FIX 3 lane guards (cap + saturating lane gate + flip-rate loss) move the
    predicted lane-change rate toward the recorded rate without costing position error?

## Measurements (all on the unseen eval sets of the R4 recipe, comparators C1 and R4)

M1. Error split wrong-lane vs right-lane cars (Q1). For each horizon h in
    {10, 30, 60, 120} s: per car, |ds| at h split by lane agreement at h
    (`predicted.lane == recorded.lane`, both present). Report median/p90 per split and
    the fraction wrong-lane. R4 baseline numbers from WM-S-R4-DIAG-LANE/ERR.
M2. Late speed drift / survivor speed (Q3). For cars present at t0 and still present
    at 120 s ("survivors"): predicted vs recorded speed error at 60/90/120 s, and the
    section-level total_delay at 120 s (existing `section_delays`), R4 comparators.
M3. Lane rates vs recorded (Q4). Predicted lane changes per HDV-step per world vs the
    TRUE rate 0.00124327163 (LANECAL allowance), at 0-10/10-60/60-120 s; plus the
    training monitor `predicted_lane_change_rate` from TRAIN_RECEIPT history.
M4. Switch-off attribution (Q1/Q2/Q4), each with everything else unchanged:
    - `flags.diag_lane_gate_shared=true`  — lane correction falls back to the shared
      acceleration gate (tests the separate saturating lane gate);
    - `flags.diag_lane_cap_off=true`      — lane-logit cap removed;
    - `flags.diag_encoder_population=true`— encoder settings replaced by population
      defaults (zero shrinkage; interaction-only embedding);
    - `flags.diag_recorded_lanes=true`    — lane decisions teacher-forced to the
      recorded lanes (upper bound of fixing lane errors; needs recorded arm).
    Report the same M1/M2 tables per switch; a switch that improves M1 wrong-lane
    error without hurting M2 locates the responsible component.
M5. Encoder settings vs true driver types, EVALUATION-ONLY (Q2). Uses the existing
    evaluation-side sidecar plumbing (never training): replay the scored states with
    physics-only + true driver types (R4-DIAG-ORACLE protocol) and compare against
    M4's population-default switch: if true-type physics >> learned settings at 120 s,
    the encoder is the bottleneck; if learned ≈ true-type, lane decisions dominate.

## Decision rules (pre-committed)

- If M4 `diag_recorded_lanes` cuts the 120 s median |ds| by > 25% while
  `diag_encoder_population` changes it by < 10%, round 6 targets lane decisions, not
  the encoder.
- If `diag_encoder_population` IMPROVES 120 s error by > 10%, the hybrid encoder
  settings overfit; round 6 adds a settings-shrinkage prior instead of dropping FIX 1.
- If M3 predicted lane rate stays > 2x recorded after FIX 3, escalate to a hard lane
  budget; if < 1.2x and M1 wrong-lane share did not fall, lane-rate was never the
  binding constraint.
- If M2 survivor speed error grows monotonically past 60 s on jams (low recorded
  speed at t0) but not free flow, keep the jam-discharge hypothesis (R4) and add a
  wave-timing feature in round 6.

## Tooling notes

- All switch-offs run through the existing scorer (`python -m wms eval` with
  `--set flags.diag_*=true`), same sets, same subsets, same seed; comparators C1
  (`ckpt_c1fast`) and the R4 main checkpoint, per the previous-best-only rule.
- M1/M2 need one small extension of the eval report (split by lane agreement at h and
  survivor masks); it reads only predicted/recorded frames — no sidecar, no labels.
