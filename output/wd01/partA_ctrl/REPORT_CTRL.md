# WD-01 Part A - Ablation A2 step 1: torch emulators of the CAV commanded speed

Generated: 2026-09-26T21:20:45.798215+00:00
Branch: zcode/wd01-ctrl (workspace D:/rd-wt/wd01j).  ZERO SIMULATION:
no SUMO/TraCI process was started; the chunk-1 logs and the frozen
D:/research-dense reference code were only read.

## Scope

Batched, device-agnostic torch emulators of the three chunk-1 controller
commanded-speed paths, proven equivalent to the traffic_shepherd reference
implementations and validated against the logged commands of every
restored chunk-1 TRAIN branch:

- **CIRCLES damping** - the sha256-pinned ONNX policy
  (baselines/circles/model.onnx) ported to a torch graph executor
  (vectorized-singleton semantics), plus the exact decide() fallback gates
  and the direct_target speed bridge.
- **PI external-U** - the cv4 PIExternalU recurrence (registered
  U=31.29, PISaturationParameters 7/30/1/2) with the cv7 PIOwnerAdapter
  lifecycle (measured-speed re-initialization, one-sample clock,
  front-coverage-only engagement gate, infeasible-bridge reset), batched
  with explicit state.
- **wall-v2 member command law** - a = clip((v_target - v)/dt, -3, +1.5)
  plus the ownership direct_target bridge (synthetic frames only; chunk 1
  has no walls).

## E1 - damping equivalence (torch port vs onnxruntime reference)

- model sha256 verified: `762fde0763c693c0e88f41b98b22fc0a11135a312b78c4c567e6211a3035ccee`
- 100,000 random valid SI rows vs the ORT singleton contract
  (the production infer path): max |accel diff| =
  **9.537e-07** (tolerance 1e-5) - PASS
- decide-level parity (gates + bridge) on 1000 frames:
  0 fallback-reason mismatches, max target
  diff 7.153e-07
- identical fallbacks also pinned on logged chunk-1 TRAIN evidence frames
  and on the full frozen dispatcher stack over a fake-api scenario
  (tests/test_e1_damping.py, tests/test_e2_pi.py)

## E2 - PI external-U equivalence

- cv4 PIExternalU (reference class) vs the batched emulator on synthetic
  sequences with random lifecycle invalidations: every committed raw
  command within 1e-9, final running targets within 1e-6
  (tests/test_e2_pi.py::test_recurrence_synthetic_sequences)
- full frozen stack (MixedAssignmentSession + PISession + CommandOwnership
  + SumoLocalSensor on a scripted fake api - no SUMO) vs the emulate() API
  across a switching schedule (damping -> pi -> native -> pi re-entry):
  every issued target, fallback reason and PI running target matches
  (tests/test_e2_pi.py::test_fullstack_mixed_dispatcher_equivalence)

## E3 - wall-v2 member command law equivalence

- WallGroupSession.tick member commands == batched pacing law on
  synthetic frames through assembly -> hold -> release and a
  member_span_exceeded safety abort (tests/test_e3_wall.py)

## E4 - validation vs logged chunk-1 TRAIN commands

Data: 49 TRAIN worlds x 7 restored branches
(native, pi, fixed1, fixed2, switching1, switching2, switching3); receipt exit_status==complete
checked per branch: 332 of
343 branches validated
(11 skipped without a complete
campaign receipt - listed in the results json).  HOLDOUT worlds were never opened.  Each live CAV's
ideal sensor frame is reconstructed at every command tick from
physics_steps.jsonl (same-lane nearest front/rear by bumper gap,
lane length/speed from the frozen network, getAllowedSpeed =
lane_speed x vType speedFactor - verified exact against the logged
evidence frames) and the emulators reproduce the logged issued
targets (match = |diff| < 0.01 m/s and identical
fallback classification).

| controller | matched/rows (rate) | max |diff| on matched rows |
| --- | --- | --- |
| native | 1139689/1139689 = 1.000000 | 0 |
| damping | 1164578/1164578 = 1.000000 | 9.54e-07 |
| pi-external-u | 1180856/1180856 = 1.000000 | 1.73e-18 |

Mismatch causes (all controllers): {"sensing_reconstruction": 0, "controller_logic": 0, "fallback_ownership": 0, "timing": 0, "unexplained": 0}

Per arm (rows / matched):

| arm | native rows | native ok | damping rows | damping ok | pi rows | pi ok |
| --- | --- | --- | --- | --- | --- | --- |
| native | 159447 | 159447 | 164206 | 164206 | 159297 | 159297 |
| pi | 163112 | 163112 | 170477 | 170477 | 174469 | 174469 |
| fixed1 | 184945 | 184945 | 171627 | 171627 | 188560 | 188560 |
| fixed2 | 170654 | 170654 | 188095 | 188095 | 187834 | 187834 |
| switching1 | 150734 | 150734 | 150954 | 150954 | 150174 | 150174 |
| switching2 | 142774 | 142774 | 146924 | 146924 | 148031 | 148031 |
| switching3 | 168023 | 168023 | 172295 | 172295 | 172491 | 172491 |

PI on the logged pi_source_evidence frames (exact reference inputs,
second arm isolating controller logic from sensing):
927273 proposal rows, 0 mismatches, 0 clock resets -
the batched recurrence reproduces every logged raw proposal and issued
target.  Reconstructed-vs-logged frame sanity over 927273 evidence
frames: max |diff| per field - speed 3.47e-18, allowed
0, front_cov 0,
rear_cov 0, front_gap 0,
front_rel 1.73e-18 m/s (0
presence flips) - exact reproduction of the ideal sensing pipeline up to
float64 noise.

Reconstruction notes (declared, all verified against the logs):
- getAllowedSpeed = lane speed x speedFactor (aggressive 1.08 / normal
  1.0 / cautious 0.92, uncapped; speedDev=0 everywhere).
- The PI engagement gate is front-coverage-only (cv7 adapter); the
  damping gate additionally requires min(front, rear) coverage > 0.
  Both reproduced exactly.
- The t0 control tick has no branch physics row (the first is t0+0.5);
  frames there are reconstructed from the base run's final row (the
  exact saveState restore source).
- Every chunk-1 CAV runs speedMode 31 (pre-fix) - irrelevant to
  commanded-speed emulation, recorded per the data contract.
- Dispatcher rows at the second physics substep are re-stamped copies
  (declared in the WM-42 schema); only the first-substep stamps are used,
  gated by physics-row liveness.

## E5 - thin emulate() API

`emulators.ControllerEmulators.emulate(action_kind, state, inputs) ->
dict(target_speed_mps, reason_code, raw_command_mps, new_state,
pi_reset)` - one control tick for a batch of CAVs, device from WD_DEVICE
(default cpu; loud error on unavailable cuda).  Batched-invariance,
shapes and device contract pinned in tests/test_e5_api.py.

## Tests

```
..................                                                       [100%]
18 passed in 3.12s
```

## Artifacts

- emulators: onnx2torch.py (ONNX protobuf reader + torch graph
  executor), emulators.py (batched damping / PI / wall + emulate()),
  refload.py (read-only reference loader)
- validation: gen_e1_random.py, validate_chunk1.py
- tests: tests/ (18 tests: E1, E2, E3, E5)
- results: results/ (E1 random equivalence, E4 validation json + stdout,
  lane table, pytest output)
