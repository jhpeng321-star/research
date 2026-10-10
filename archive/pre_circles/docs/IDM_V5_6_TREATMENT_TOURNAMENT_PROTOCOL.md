# IDM v5.6 treatment tournament protocol

Status: **frozen, development-only treatment tournament**

Date: 2026-09-01

## Purpose and fixed causal contract

This protocol tests separately defined repairs to the safety limitation of the
IDM compact moving-wall overlay.  It does not train a predictor or RL policy.
The frozen control is `idm_analytical_disturbance_p30_v1`.  Every engaged arm
is the identical damping controller plus its candidate overlay; all CAVs that
are not explicitly in that candidate's declared role retain the identical
damping action.

The following are invariant for every candidate: IDM scenario with 2700 vph
mainline demand, 400 vph ramp demand, 30% CAV penetration, absolute SUMO
activation time 880 s, maximum 20 s intervention window, and outcome endpoint
at absolute time 1080 s (or demonstrated natural clearance).  The primary
physical endpoint remains the pre-existing sustained-congestion definition.
The only development seeds are 4401, 4402, 4404, and 4405.  Seeds 4407--4430,
4601--4660, and 5001--5060 are protected.

The four seeds are a development selection cohort, not evidence of
generalization.  Their use creates a multiple-comparison risk.  Only one
candidate may advance, unchanged, to a separately frozen fresh-seed protocol.

## Candidate routes and individual entry gates

| Route | Frozen treatment difference | Entry decision | Outcome screen |
| --- | --- | --- | --- |
| A | Same compact moving wall; both adaptive-servo assembly and HOLD/taper deceleration caps are 0.3 m/s2 rather than 1.0 m/s2. | Eligible: the 1.0 cap is materially active and source-compatible formation remains possible. | Required. |
| B | One full-lane wall plus at most one separately latched upstream support CAV per lane, using bounded IDM-compatible smoothing. | Stop unless a treatment-blind audit finds a non-reserve support cohort with runway, safe gaps, observability, and separation in every required lane. | Required only if the entry gate passes. |
| C | Discharge-aware simultaneous monotone taper; formation, HOLD authority, and geometry unchanged. | Stop if control telemetry proves it cannot trigger before a known unsafe path diverges. | Required only if the entry gate passes. |
| D | Corridor-wave-growth HOLD termination; 5 s minimum HOLD, then existing taper; no increase in authority after taper begins. | Stop if it cannot alter every known unsafe path before its adverse divergence. | Required only if the entry gate passes. |
| E | One alternate compact geometry selected using a treatment-blind geometry projection. | Stop unless exactly one alternative forms safely with the specified 20 s headroom in at least three cases. | Required only if the entry gate passes. |
| F | One two-cohort mild-wall design with a lower per-cohort cap and bounded total authority. | Stop unless a treatment-blind audit demonstrates two disjoint, spatially separated cohorts, storage headroom, and noninteracting releases. | Required only if the entry gate passes. |

No candidate may borrow a mechanism from another route.  A candidate stopped by
its own treatment-blind feasibility, action-headroom, integrity, or focused
safety gate is recorded as a no-go and does not stop the remaining routes.

### Candidate A exact profile

Candidate A is `idm_compact_moving_wall/5.6.a-low-authority`.  Relative to
v5.4 it changes only `compact_assembly_a_max_dec_mps2` and
`compact_servo_a_max_dec_mps2`, both from 1.0 to 0.3 m/s2.  It also changes the
adaptive-servo formation projection to use that same 0.3 m/s2 value, so the
admission model and runtime actuator agree.  `wall_params.a_max_dec` already
equals 0.3 m/s2 and is not a new change.  Geometry, members, speed reference,
tau/headway law, jerk/safety projection, handoff rule, planned duration, and
five-second linear release are unchanged.

Units are explicit: acceleration/deceleration is m/s2, speed-target change is
m/s, and jerk is m/s3.

## Common diagnostics and integrity

Each executed candidate writes a separate profile fingerprint and output root;
reports requested/realized assembly, HOLD, taper/release, member and role IDs,
lane coverage, compactness, controlled CAV-seconds, applied speed and tau
deltas, acceleration/deceleration, jerk, projection survival, gaps, TTC,
closing speed, follower-chain compression, braking followers, hard-brake
exposure/onsets, collision/teleport, storage, discharge, released-cohort
arrival rate, queue displacement, delay/queue area, congestion duration, and
release/abort reasons.  Every treatment-only hard-brake onset is attributed by
time, vehicle, lane, leader, gap, TTC, lane-change state, role, lifecycle
phase, and time since release.

Integrity requires exact damping replay, exact predecision identity, declared
role-only override, exact out-of-role damping, full occupied-lane coverage,
compactness, no collision/teleport, no hard lifecycle abort, exact damping
return, common horizon, and effects larger than replay noise.  An integrity
failure is not recoded as a neutral result.

## Four-case screen and selection

For every candidate that clears its own entry gate, run matched damping and
treatment branches for all four development seeds, using at most six parallel
SUMO workers.  It is compared against the stored v5.4 and v5.5.1 traces.

To pass, all four cases must satisfy integrity, safety, and service gates;
4402 and 4405 must retain material congestion prevention/shortening; 4401 and
4404 must show no material congestion, safety, or service harm; authority
must be non-negligible; compression/release-wave/hard-braking must improve
against v5.4/v5.5.1; queue must not merely move outside the measured region;
and all members must return to exact damping.

The tournament never selects a winner when the first route passes.  Audits are
completed for A--F and every independently feasible route completes its
four-case screen before selection.  Among passing candidates the winner is
chosen lexicographically by integrity, safety, service noninferiority,
preservation of 4402/4405 benefit, no harm in 4401/4404, simplicity, lower
total authority, lower delay/storage burden, then larger congestion benefit.
If none pass, treatment development stops and only the decentralized-authority
boundary paper framing remains supported.

## Fresh-seed protection

Only an unchanged tournament winner may receive a new 24-seed protocol for
4407--4430.  Losing or individually stopped candidates never open fresh seeds.
No predictor-scale generation, fit, calibration, generalization claim, or RL
training is authorized by this protocol.
