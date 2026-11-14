# WD-01 owner decisions log (world model session)

## 2026-09-29, about 15:00: Part A v2 GENERAL (relayed by main discussion)

**Decision.** Part A becomes a general rule-based predictor:

- it uses ONLY the reconstructor's sample (no oracle vType, no future
  arrivals, no inserted vehicles, no SUMO internals);
- one nominal textbook parameter set (IDM + MOBIL/LC2013-style + merge rule),
  not SUMO-exact and not per-type tuned;
- it tracks every input vehicle to the horizon, and CAVs follow the commanded
  action;
- uncertainty comes only from the spread across recon samples;
- arrivals after t are not Part A's job (Part B and the horizon handle them).

**Evaluation.** Later: model error vs SUMO truth, from the true state and from
recon samples, reported honestly.

**Effect on running work.** The "v2 exact" task d03703e2 is not cancelled.
Its engine is reused, and its exactness harness is demoted to an equation
unit check.

**Design.** PARTA_V2_GENERAL_DESIGN.md.

## 2026-09-29, about 21:40: Part B deployable inputs (relayed by main discussion)

- **#5 entrance detectors.** The AGENTS.md RSU input rule stands.
  - The CAV-STRICT deployable variant (CAV self-reports + CAV-detected
    neighbours only) is the HEADLINE.
  - The all-vehicle entrance-detector variant is reported only as a clearly
    labelled SENSITIVITY. It is not a deployable claim.
- **pre_fix.** For any deployable claim it is a label/stratifier only. If it
  is kept as a training feature, deployment fixes it to the post-fix value,
  and the report says so. This matches the ruling given to the reconstructor
  for penetration, pre_fix and speed_mode.

## 2026-09-30, about 02:05: Part A v2 general flags and Part B v0.4 source (owner: "do as recommended", relayed by main discussion)

- **(a) Guards retired.**
  - The non-textbook lane-change guards (cooldown_s 2.0, lc_min_speed_mps 1.0)
    are removed from the nominal Part A v2g parameter set.
  - The labelled `guards_on` variant is kept as a sensitivity only.
  - The nominal set is now fixed: partA_v2g/params_nominal.json, LF-sha256
    079a55768f803b23c92af188680c1df91dc63f04acbbbac948870b46e75935f7
    (zcode/wd01 9dda3a3).
- **(b) Lane-change investigation.**
  - A bounded, TRAIN-only, zero-SUMO investigation (WD-01-PA2G-LC1) is
    registered. It covers the residual discretionary lane-change
    over-production (4.6–6× in PA2G-A2) and the merge under-production.
  - Diagnosis comes first. No parameter is tuned without a further owner
    decision.
  - C4L1 (the chunk-4 CALIBRATION look, REGISTRATION_C2L1_V2G_DRAFT.md) is
    NOT frozen or run until one of these holds:
    - (b) is resolved;
    - the owner decides that the lane-change criterion is reported-only.
- **(c) Delay scope.**
  - The long-horizon negative delay bias of Part A v2g comes from the tracked
    scope: untracked arrivals carry about 58 % of the true delay at H = 120.
  - Part B carries the untracked arrivals.
  - Every Part A report states this.
- **(d) Part B v0.4 training source.**
  - v0.4 trains on chunk-4 `__native_base` plus the wall-pilot branches, and
    the corrected true native/PI arms when available. This makes the native,
    damping and wall contrasts identifiable.
  - The v0.4 registration is drafted after v0.3.2 lands.

## 2026-09-30, about 09:40: walls, hold draws, LC1, Part B v0.4, re-scoring (relayed by main discussion)

- **Walls: OPTION B is approved** ("allowed to assemble"). This is an
  assembly-closure law change, identical in traffic_shepherd/wall_group.py and
  the mirrored partA_ctrl wall_pace:
  - the front member holds v_wall;
  - the rear member runs at v_wall + min(2 m/s, 0.1 s⁻¹ · span) until
    span ≤ 6 m;
  - it stays inside the existing acceleration clip and safe following;
  - a new `assembly_timeout` safety abort fires at 60 s.
  - Option A's A1, A3 and A4 are kept (upstream edge only, road-left check,
    road-end handback).
  - The 75 m feasibility cap is RESTORED (A2 is dropped).
  - Required: unit tests, WM-43 re-validation, and a bounded wide-pair smoke,
    each registered in the ledger first.
- **Hold draws:** keep {60, 120, 240} s, and record the road-end truncation
  (achieved vs requested hold).
- **Pilot scope:** 44 worlds, as applied. W-2 stays deferred.
- **J4 machine:** J4 runs on the LAPTOP after B is validated (no rented box
  now). CPU use is coordinated with the planner's PL-EXP1c.
- **LC1:** the owner wants an explanation before deciding. C4L1 is held.
- **Part B v0.4:** YES (chunk-4 true arms + wall pilot). The registration is
  finalised after the J4 data exists.
- **Re-scoring:** YES, zero-simulation. Old analyses are re-run against the
  new true references (Part B contrast re-scoring against the chunk-2/3
  truearms where applicable).

## 2026-09-30 (afternoon): C4L1 ruling and v0.4a data choices (relayed by main discussion)

- **C4L1 is ACCEPTED** as "PASS (protocol deviation: restarts beyond the
  registered allowance, one after partial rows; no numbers seen; the
  implementation corrected to the registered window rule)".
- **v0.4a data choices are CONFIRMED as owner-confirmed deviations:**
  - the PI token comes from the planned schedule action, because command
    owners label PI as "damping", its writer;
  - the chunk-1/fix27 damping bases are auxiliary training data only.

## 2026-10-01 — WM-N1 Idea A (relayed by main discussion)
- Amendment 3 `REGISTRATION_WM_N1_AMENDMENT_3_IDEA_A.md` (LF-sha256 a597d986…) ACKNOWLEDGED by the owner (verified by main), incl. tie order T < A < C < N and held-out test (iii) = remove the highest J4 wall-speed factor.
- GPU plan: ~3 GPUs for WM-N1 (final count after the 1-h timing gate). The WM-N1 GPU queue must run members and held-out configs in parallel across boxes (one config per box).
- Wall law v2.2 owner spec (merge-release at merge start, planner hold capped at merge start, 2-abreast, benefit trigger, D_min = assembly distance + minimal hold): walls_design/merge_release/OWNER_SPEC_V22_20261001.md (zcode/wallmr).

## 2026-10-01 night — WM-N1 FEAT1 output root (relayed by main discussion)
- The FEAT1 output (memmap + compressed upload) goes to `D:/rd-staging/wmn1_feat/` (201 GB free). NOT C:, NOT E: (E: is only ~38 GB free).
- This is an owner-approved EXCEPTION to the no-D rule for staging/upload artifacts ONLY. Code and the active tree stay on E:.
- `D:/research-dense` must not be touched.
- The featurizer's D: guard needs an explicit, documented override flag valid for this output root only.

## 2026-10-01 night — the world model is the priority (relayed by main discussion)
- The WORLD MODEL is now THE priority; the reconstructor model work is deferred.
- All rented GPUs (3–4 boxes) go to WM-N1: the Idea A members, the N comparator, and the held-out tests.
- J4B and J-PULSE get CPU-box priority.
- WM agent cap: GLM ≤ 6 + DeepSeek ≤ 3 (GLM total ≤ 8 across sessions; the recon keeps 2).
- Priorities: (1) Idea A code + pre-flight gates; (2) the FEAT1 fix + run (to D:); (3) the v0.4b direct-difference registration.
- Paper framing: the WM is trained and evaluated on the TRUE state. The "sparse sensing → reconstruction" link (the recon-input path, C4L2) is FUTURE WORK: keep it in the design, don't build it now.
