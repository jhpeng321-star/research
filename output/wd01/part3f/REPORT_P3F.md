# WD-01 Part 3F — structured emulator validation eval (gate PASS)

**HOLDOUT NOT ACCESSED.**  Validation only (4 roots, t0 {120..480 step
60}, 4 windows per root = 16 windows).

## Gate verdict: PASS

| criterion | value | threshold | pass |
| --- | --- | --- | --- |
| ens speed RMSE H=60 | **4.481** | < B2 4.90 | YES |
| collisions /1000 | **0.008** | <= 1.0 | YES |

The structured emulator (per-vehicle sampled vType + SUMO dawdle +
stochastic LC hazard at scale 0.002) beats B2 IDM-normal on speed RMSE
and has effectively zero collisions.

## Implementation

engine.py kind="idm_sampled":
- per-vehicle vType sampled from the posterior at t0 (CRN per member)
- IDM with per-vehicle params (tau, a_max, b, s0, v0 = sf x limit)
- SUMO dawdle: v' = max(0, v_idm - U(0,1) * sigma * accel * dt)
- stochastic LC hazard (d kept constant, attrs refreshed, merge-zone
  gated)
- collisions counted, never repaired

## Limitations

- M=3 members on validation (M=5 was the target; reduced for CPU
  budget)
- LC scale calibrated on 2 roots x 2 t0 = 4 windows
- Delay proxy and TTC<3 exposure coverage not yet computed (ens
  member band)
- B4 v3 variant requires E.V3Baseline (separate class); not integrated
  into the eval loop

## Files

- p3f_eval.py: validation eval
- results/val_results.json: per-H metrics
- results/gate.json: gate verdict
