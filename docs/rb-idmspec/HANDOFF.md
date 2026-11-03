# RBH-IDMSPEC HANDOFF — SUMO 1.27.1 MSCFModel_IDM extraction

Status: COMPLETE (2026-10-31, ~23:20-00:30 window). Deliverables consumed from
`E:/agent-work/rb-shared/idmspec/` (canonical copies; identical files committed
here under `docs/rb-idmspec/`).

## What exists
- `spec.md` — exact MSCFModel_IDM behaviour spec for SUMO v1_27_1 with a
  file:line citation for every rule: core `_v` loop (sub-stepping = old
  idmStepping), followSpeed/stopSpeed/freeSpeed, getSecureGap, minNextSpeed,
  finalizeSpeed clamp chain, speedFactor application (multiplies the LANE limit
  and desiredMaxSpeed; NOT a post-hoc clamp), actionStepLength semantics at
  0.5 s step (parse rounding, action gating, aMax/ASL, stop-fallback headway,
  auto-ballistic switch), Euler vs ballistic integration, known artifacts
  (mid-loop zero clamp re-acceleration).
- `ref_idm.py` — pure-Python transcription (no SUMO dependency), API map in
  spec.md §9. Default passenger vType params built in; explicit lane-limit and
  actionStepLength arguments.
- `test_vectors.json` — 20 numeric leader/follower vectors (TS=0.5 default,
  one TS=1.0 case), inputs + expected vNext.
- `run_tests.py` — regenerates/verifies vectors + 8 independent analytic
  invariants (A1-A8). Current result: **ALL PASS (20 vectors + 8 invariants)**.
- `_sumo_src/` (rb-shared only, not committed) — the v1_27_1 source files for
  citation checking; filenames use `__` for `/`.

## Switches / integration
None — this is a reference artifact; no code changes to src/ were made.
Consumers: rb-cf / rb-int should port `ref_idm.py` into
`src/wms/physics/idm_sumo.py` (keep function boundaries: `_v`, `follow_speed`,
`stop_speed`, `free_speed`, `finalize_speed`, `idm_step`).

## Gotchas for the rebuild (full list in spec.md §7)
1. gap passed to followSpeed is the NET gap (bumper distance minus ego minGap);
   `_v` adds minGap back when respectMinGap=true (followSpeed) but not for
   stopSpeed.
2. speedFactor multiplies lane speed limit + desiredMaxSpeed (MSLane.h:591,
   MSBaseVehicle.cpp:211-213); desSpeed = min(vTypeMax, desiredMax*sf, limit*sf).
3. IDM ignores leader apparentDecel and has no decel cap in followSpeed;
   braking bounded only by vMin (emergencyDecel rate per Euler step).
4. myIterations = max(1, int(TS/0.25 + 0.5)) → 2 sub-steps at TS=0.5 (0.25 s
   internal), 4 at TS=1.0; gap pre-advanced inside the loop.
5. actionStepLength≠step-length silently switches SUMO to ballistic position
   update (MSVehicleType.cpp:463-476); aMax divides by ASL, stop-fallback
   headway = ASL.

## Tests
`python docs/rb-idmspec/run_tests.py` (pure Python 3, no deps, no SUMO run).

## Known gaps / not covered
DriverState perception errors, IDMM adaptation, stops (processNextStop), TraCI
influencer, lane-change patchSpeed, insertion speed procedures, junction
right-of-way — all documented as out-of-scope no-ops in spec.md §8.

## Re-verification (fresh session, 2026-11-04)
Resumed after session freeze; nothing redone. Independently confirmed:
- rb-shared/idmspec/ copies byte-identical to the committed docs/rb-idmspec/
  (SHA-256 match on all 4 deliverables).
- `run_tests.py` → ALL PASS (20 vectors + 8 invariants); test_vectors.json
  regeneration is deterministic (hash unchanged).
- Citations spot-checked against _sumo_src downloads: MSCFModel_IDM.cpp
  (delta :41, iterations :44, _v :246-268, stopSpeed :178-200, freeSpeed
  :104-127), MSCFModel.cpp (finalizeSpeed :199-265, maximumSafeStopSpeed
  :806-884, calcEmergencyDecel :1037-1086), EMERGENCY_DECEL_AMPLIFIER=1.2
  (MSCFModel.h:35), NUMERICAL_EPS=0.001 (config.h.cmake:236),
  processActionStepLength (SUMOVehicleParserHelper.cpp:1772-1789). All exact.
