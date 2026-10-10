# Registered target-speed check: one uncapped-rule wall per exposed seed

Status: registered bounded physical block, authorized by the 2026-09-07 owner
instruction to investigate moving-wall harm and whether changing the
maintained wall speed can produce useful escalation. This document freezes the
scientific question, the exact change, and the budget **before** any new
simulation. It authorizes no sweep, no retry, no replication, no training and
no acceptance use.

## Question

Does holding the repaired gentle moving wall at the formation rule's own
uncapped target (reported member mean speed − 2 m/s, still limited by member
allowed speeds) change the delivered maneuver and the full-horizon paired
outcomes relative to (a) the same wall at the cap-bound 24 m/s target and
(b) matched continued damping, in the two already exposed development seeds?

## Why this exact alternative

Parts 1–4 of `output/wall_harm_speed_v1/` show:

1. The 24 m/s cap binds in every realized engagement (entry means 27.681 and
   29.432 m/s, so the cap forces −3.68 m/s and −5.43 m/s slowdowns where the
   rule itself requests −2.0 m/s) and lengthens assembly to 12/18 s.
2. The shared 4108 harm scales with this slowdown dose (assembly/HOLD pack
   delay shifts queue phase and lane allocation); in 4107 the cap-bound depth
   contributed to hard post-release queue entry and to lane-1 acceptance
   differences. The requested slowdown depth is an entry-dataset variable.
3. Historical speed studies used different actuators, information contracts,
   traffic and membership; they do not answer this comparison (Part 4).
4. No target-speed alternative has ever been tested for the current
   controller; this is the transparent design rule itself, not a tuned value.

## Exact change

`target_speed_wall.TargetSpeedProfile`: `maximum_target_speed_mps` 24 → 40
(non-binding in this corridor), version `target_speed_wall/1`. Every other
profile field, the cohort, entry time 600 s, H20 commitment, −0.3 m/s²
absolute braking, 0.15 m/s³ ramp, readiness law, coast exit and
identical-damping handback are unchanged. The policy class repeats the
existing lifecycle initialization; no new control behavior is added.

## Mechanics prerequisite (120 s, charged)

`target_speed_mechanics`: one isolated two-lane fixture (2 members + 2
followers, cruise 27 m/s, road limit 33 m/s), decision at 5 s. Must show:
commanded target exactly 25.0 m/s (uncapped rule, above the old cap), three
settled readiness samples before HOLD, compact 20 s HOLD at target, verified
coast exit and identical-damping handback, command contract, zero
collisions/teleports. If it fails, the block stops and the failure remains
charged.

## Corridor episodes (2 × 2400 s, charged)

- `s4107_m2700_p20_t600_target_v1_wall` — same cohort/members/lanes/timing as
  the H20 source `s4107_m2700_p20_t600_lifecycle_v2_wall`.
- `s4108_m2700_p20_t600_target_v1_wall` — same for
  `s4108_m2700_p20_t600_cohort_v1_wall`.

Verified reuse: H20 wall traces (wall-versus-wall primary contrast) and
observer-replay damping arms (wall-versus-damping, original criteria). Exact
identity required before 600 s in intervals, traffic, policy, telemetry and
predecision decision-snapshot fields against the H20 sources. Both episodes
run in parallel. Zero automatic retries. Failures and incomplete service
remain outcomes.

## Accounting and accounting boundaries

New budget: exactly 120 + 4800 = 4920 simulated seconds, below the authorized
ceiling of 2 mechanics fixtures + 4 episodes; the remainder stays unallocated.
Changed assembly time is part of the maneuver effect: results are reported as
full-maneuver contrasts, never as a pure HOLD-speed effect. No post-assembly
speed-change arm is added.

## Stopping rules and gates

- Stop on any command, ownership, sensing, prefix, target-law or contract
  defect; preserve the charged failure.
- No speed sweep, no further seeds, no H40/duration/braking grid, no map
  change, no training, no predictor fitting, no acceptance read.
- A harmful or nonbeneficial label closes this direction with evidence; it
  does not by itself authorize any alternative rule. A qualifying result
  would still require separate registration for replication and prospective
  evaluation before any claim.
