# Decentralized Structured IDM Controller Protocol

Status: implemented software contract, version
`idm_decentralized_structured_controller/1`. This document does not authorize
PPO training or make an efficacy claim. The controller must pass an authority
screen before observations, reward, and training are frozen.

## Scientific role

This is the low-cost decentralized layer in the paper architecture. Every CAV
runs one parameter-shared actor using its own physical observation and bounded
neighbor/V2V messages. No learned lane-level agent observes a candidate list or
chooses an absorber. Centralized training of a critic is optional; execution of
the actor and contention contract remains decentralized.

The later moving wall is a distinct centrally coordinated treatment. The
escalation contribution must compare continuation of this decentralized
controller with temporary centralized wall control.

## Local action

Each eligible CAV emits one categorical action:

0. abstain;
1. mild analytical damping, gain 0.75;
2. nominal analytical damping, gain 1.00;
3. strong analytical damping, gain 1.25;
4. conservative reference, analytical reference offset -1.0 m/s.

Actions are requests around the analytical controller, never direct
accelerations. An invalid or malformed action becomes abstain. An inadmissible
CAV has only abstain enabled. While an option is latched, only its selected mode
is enabled; its repeated policy output has no control authority.

## Candidate admission

Candidate geometry is lane-local: a CAV must be on the pre-merge edge, upstream
of a detected same-lane disturbance, and have 150--800 m runway. Every
geometrically feasible candidate is tested before ranking. Failure of the
nearest candidate therefore does not suppress a safe fallback.

Admission requires:

- front TTC at least 3 s when closing;
- rear TTC at least 4 s when the follower is closing;
- leader gap at least SUMO `getSecureGap` when a leader exists;
- follower gap at least the follower's SUMO `getSecureGap` when it exists;
- predicted dose at least 16 s for the requested 20-step option.

Unavailable secure-gap evaluation is an admission failure, not a silent pass.
The existing follow-speed, comfortable-acceleration, jerk, TTC, allowed-speed,
and speed-mode projection remains the final command shield.

## Distributed contention and lifecycle

Admissible non-abstaining CAVs broadcast only lane, runway,
distance-to-merge, vehicle ID, and requested mode. All participants can compute
the same lexicographic priority: smaller runway, vehicle ID, then mode index.
At most one bid wins per lane. Winners and retained members must be at least
150 m apart longitudinally, including across lanes. A lower-priority admissible
bid remains a fallback rather than being rejected before contention.

The winning CAV and mode are latched for 20 controller steps. The latch ends on
completion, leaving the pre-merge edge, or a hard safety failure. Completion or
handoff uses the existing jerk-limited blend to native IDM before
`setSpeed(-1)`. Nonmembers receive no TraCI speed override.

## Observation and learning contract

The physical observation retains the existing ego, zone, local-neighbor,
absorber-eligibility, analytical-controller, and temporal-risk fields. The
five action-mask bits are appended after physical features and are not embedded
as measurements. The actor consumes only its local vector. A MAPPO critic may
consume packed peer states during training, but those states are unavailable
to the deployed actor. IPPO is also supported with the same action mask.

## Required diagnostics

Each step records all candidate admissions and rejection reasons, rank, runway,
predicted dose, front/rear TTC, front/rear gap and secure-gap values, local bids,
contention losses, active members, latched mode, member changes, commands,
projection reasons, release state, simultaneous absorber count, and controlled
CAV-seconds.

## Verification before training

Software gates:

- legacy residual/gain/reference behavior remains unchanged while the new flag
  is disabled;
- all safe candidates are exposed and the first safe bidding fallback can win;
- out-of-band and losing CAVs receive no override;
- spacing, option latch, safety abort, masks, and smooth release are exact;
- observation and action spaces match in both IPPO and MAPPO;
- focused and full repository tests pass.

Empirical authority gates remain mandatory. A small, development-seed-only
screen must demonstrate meaningful dose for all four active modes, placement
and mode response above replay noise, helpful and nonhelpful outcomes, no
collisions, bounded TTC/follower response, and acceptable service loss. Failure
of dose requires controller repair; adequate dose without outcome response
requires action redesign. Neither outcome authorizes a predictor claim.
