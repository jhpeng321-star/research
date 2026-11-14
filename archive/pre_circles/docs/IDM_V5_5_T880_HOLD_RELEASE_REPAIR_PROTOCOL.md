# IDM v5.5 t=880 hold/release repair protocol (review-only)

Status: frozen proposal; implementation is intentionally not authorized yet.

## Causal finding

The t880 safety-causality diagnosis passed exact zero-effect sham replay on
seeds 4401, 4402 and 4404. A one-second assembly-only arm produced no excess
treatment-only hard-brake onsets. The full 20 s wall added hard braking during
HOLDING/TAPERING and additional downstream events after release. The supported
limiting mechanism is accumulated follower compression and release-wave
discharge, with metastability affecting the late response.

## Single proposed change

Create version `v5.5.1` with a hold/release-only safety law:

1. preserve the existing member selection, one-second assembly, 20-second
   activation window, compact geometry, full occupied-lane requirement,
   analytical IDM damping control, and 1080 s outcome horizon;
2. during HOLD, monitor the existing follower-chain compression telemetry and
   freeze the wall target/tau state when the predeclared compression threshold
   is reached (no further tightening);
3. begin a monotone release to the exact damping targets as soon as the frozen
   compression threshold is reached, with no discharge step that increases
   follower closing speed;
4. keep the existing hard-safety abort and return to identical damping.

The exact compression threshold, release slope and machine fingerprint must be
chosen and recorded before any SUMO branch. Duration, activation time, number
of walls, lane-coverage definition, base damping, congestion endpoint and
outcome thresholds remain unchanged.

## Pass/fail gate after approval

Run only seeds 4401, 4402, 4404 and 4405, each with a paired damping control,
after focused and full pytest. The repair must pass exact replay/predecision
identity, sham identity, full planned dose and lane coverage, safety and service
admissibility for all four, retain physical benefit on 4402 and 4405 beyond
repeat noise, show no material harm on 4401/4404, and return smoothly to
damping. Harmful and failed cases must remain in the report.

If the gate fails, do not add seeds or train a predictor; report whether the
wall concept, geometry or lifecycle remains limiting and pivot honestly.

## Approval boundary

This proposal changes the hold/release treatment law. Per the project contract,
implementation must wait for explicit scientific approval of this exact repair
definition and its frozen threshold/slope values.

