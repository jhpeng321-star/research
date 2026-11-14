# Braking limit and HOLD duration: bounded factorial check

Owner authorization: test whether the −0.3 m/s² braking limit is applicable and
improves outcomes, and whether HOLD duration materially changes wall effects.
This is a diagnostic experiment, not training or an accepted escalation gate.

Freeze one factorial policy before new physical outcomes. Reuse the unchanged
GentleWallPolicy through a constructor-only subclass. Factors are absolute
wall-command minimum −0.3 versus −3.0 m/s², and HOLD 0/20/40 seconds. All other
profile fields are frozen: 0.15 m/s³ command ramp, +1.5 m/s² maximum acceleration,
50 s assembly deadline, three target-settled readiness samples, target law,
cohort, 10 s maximum release, sensing and safety settings. These limits apply
throughout wall ownership, including transition. The cap also enters existing
clearance, stopping-distance and handback predicates. Thus this tests the
braking contract as a whole, not a pure fixed-brake pulse or assembly-only
actuator effect. Do not add a lifecycle-feasibility repair during this factorial.

HOLD 0 means complete assembly/readiness and start the same release logic
immediately. It is not no intervention. HOLD 20 is the incumbent; HOLD 40 tests
whether doubling that commitment helps or becomes infeasible. They are three
predeclared contrasts, not a search for an optimal duration. Nominal HOLD is not
realized HOLD when an attempt aborts. Report both, including exact phase clock,
command extrema, measured member/follower responses, and time/distance consumed.

Use only existing exposed seeds 4107 and 4108, mainline 2700 veh/h, nominal CAV
share 20%, decision 600 s, identical previously registered cohorts, full 2400 s
service horizon. Damping references are reused only with identity, complete
artifact, predecision state/report/command and exogenous-future verification.
The two existing −0.3/H20 trajectories may be reused only after exact physical
policy-output replay with the new subclass and unchanged execution modules.

Twelve factorial cells are defined. Two are reused −0.3/H20 runs. In seed 4108,
−0.3/H0 and −0.3/H40 are expected to be trajectory-equivalent because readiness
is never reached before terminal assembly abort. Establish this by replaying
all 2400 policy decisions against actual acquired reports and reconstructed
pure local CIRCLES decisions, checking commands, phases, reasons and member
sets against the original. Profile fingerprints and diagnostic estimates may
differ. Identical commands plus unchanged deterministic simulator/exogenous
state justify reuse; these are derived cells, not newly simulated replications.
If equivalence fails, stop registration and revise the evidence plan before
allocating any corridor simulation. Do not silently charge replacement runs.

Physical budget: two isolated 120 s mechanics fixtures, strong cap/H0 and strong
cap/H40, at most 240 simulated seconds, zero retries. Verify actual ownership,
clock, commands/ramp, readiness, zero or full 40 s HOLD, handback and unchanged
HDV/native protection. Existing gentle mechanics and exact output replay cover
the reused policy path. Then at most eight new 2400 s corridor runs = 19,200 s.
Total new ceiling 19,440 simulated seconds. No automatic retries or successor
campaign, fresh seeds, predictor fitting, damping changes or acceptance access.
Source and protocol are frozen before mechanics registration. Stop the block on
implementation/hash/replay failure; physical aborts and negative outcomes are
retained and completed to the full horizon. Run independent jobs in parallel
within available CPU resources; each receives an exclusive immutable allocation.

Primary comparisons use treatment costs per requested vehicle. For each HOLD
and seed, report gentle minus strong-cap cost. For each cap and seed, report
H20 minus H0 and H40 minus H20. Also report the difference of these contrasts
between caps, and absolute costs against matched damping. Retain all cells and
both seeds; no post-hoc winner selection. Two seeds do not support population
significance, optimal duration, a validated predictor or a safety claim. The
three equivalent 4108 gentle cells are one realized trajectory, not independent
observations. Practical significance means effect size and the original physical
guardrails, not an invented p-value. Descriptive averages cannot override case
failures or undelivered HOLDs.

Unchanged benefit criteria: service improvement ≥1 s/request OR severe-braking
improvement ≥0.02 vehicle-seconds/request; allowable cost increases service 0.5,
source waiting 0.1, severe braking 0.005, acceleration effort 0.5, and TTC<3 s
exposure 0.02, all per requested vehicle. Both arms must complete all requested
service, with zero treatment collision involvements and starting teleports.
Retain baseline safety anomalies. Severe braking is actual a<−3 m/s²; effort is
∫a²dt in m²/s³, not fuel. Decompose positive/deceleration effort, and inspect
upstream followers during wall ownership and the delayed remainder. Measured
braking exceeding −0.3 during a gentle attempt fails physical gentleness even
if command bounds pass. Never disable SUMO protection to force measured bounds.

Deliver a versioned implementation, tested reuse/equivalence proof, immutable
registrations/allocations, full physical contrasts and actual lifecycle outcomes,
and a decision on applicability, comparative benefit and HOLD sensitivity. A
negative/infeasible result is an answer; it does not authorize more training.
