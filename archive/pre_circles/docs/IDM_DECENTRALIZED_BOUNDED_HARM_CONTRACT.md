# IDM Decentralized Bounded-Harm Damping Contract

Status: frozen development contract before new decision-rich evidence or
training. This document does not claim that a trained controller works.

## Scientific role

The decentralized controller is a low-authority intervention for lower and
transition realized loading. It is not required to eliminate residual risk in
structural overload. A high-density action may be physically helpful yet still
be insufficient when braking, wave growth, low-TTC exposure, service loss or
actuator saturation remains above the frozen acceptable target. That residual
risk is evidence for the later moving-wall escalation stage, not permission to
make decentralized braking more aggressive.

The current deployment architecture is decentralized per-CAV execution with a
shared actor, deterministic V2V contention and bounded infrastructure-provided
traffic context. The centralized peer critic is training-only. It is not an
onboard-sensor-only architecture.

## Fixed damping treatment

- Action 0: abstain and retain native IDM.
- Action 1: bid for the fixed lane-disturbance p30 terminal-speed ceiling.
- Requested duration: 20 one-second controller steps.
- Adequate dose: at least 16 command steps.
- SUMO integration: two 0.5-second ticks per controller step.
- Reference floor: 8 m/s; the ceiling does not increase braking authority.
- Comfort, jerk, TTC, secure-gap, SUMO safe-speed and smooth-release
  projections remain authoritative.
- No lane locking, treatment handoff, bounded-drop target or lane-only mask is
  part of this contract.

## Paired decision outcome

Every development decision branches from an identical predecision state into
abstain, exact abstain replay and activation. The outcome horizon is 60
seconds. The safety composite is the equal-weight mean of reference-normalized
reductions in corridor severe braking and acceleration energy. Reference files
are normalization scales only; the identical abstain branch is the causal
control.

- Helpful: safety composite greater than +0.02 and every constraint passes.
- Neutral: composite in [-0.02, +0.02] and every constraint passes.
- Harmful: composite below -0.02 or any dose, safety or service constraint
  fails.
- Service-admissible decision: delay-rate increase no more than 1 s/vehicle,
  arrival loss no more than one, and merge-discharge loss no more than one.
- Hard failure: added collision or teleport, incomplete horizon, unexpected
  command, or safety-projection bypass.

The 0.02 deadband is increased if duplicate abstain replay shows larger
numerical noise. No favorable composite may hide a constraint failure.

## Development and training gates

Initial development uses seeds 4107--4120. After the broad-observation and
zone-plus-recurrence hypotheses failed, seeds 4121--4132 were frozen as a
one-time prospective-within-development check of the ranked-three-neighbor
representation. Reference seeds 4101--4106 and every opened development seed
are permanently excluded from independent claims. Seeds 4201--4240 remain
untouched checkpoint-acceptance evidence.

Before training:

- exact discovery, predecision and abstain-replay identity;
- zero collisions and teleports;
- at least 80% of treatments deliver at least 16 of 20 steps;
- at least 90% pass safety and service constraints;
- at least 24 qualified nonneutral decisions;
- helpful and harmful outcomes in at least three development seeds each;
- at least 30% of qualified effects exceed replay noise;
- seed-grouped actor-observation AUROC at least 0.70 and balanced accuracy at
  least 0.65;
- reward sign agrees with every nonneutral paired action and reward/effect
  correlation is at least 0.80.

Training starts with seed-grouped supervised initialization and then a fresh
2,000/10,000-step MAPPO progression. No 50,000-step run is authorized until a
10,000-step checkpoint improves physical outcomes over always-activate and a
frozen conservative rule without TTC or service exploitation.

## Loading and escalation interpretation

Configured veh/h is recorded but is not called density and cannot define a
universal critical threshold. Evaluation records realized occupancy, flow,
speed, variance, gradient, lane imbalance, braking/TTC state, action dose and
saturation. The authority transition may be gradual and seed-dependent.

The primary damping-development cells are 1900, 2000, 2100 and 2200 veh/h
mainline with the frozen 400 veh/h ramp. Existing 2700 results remain a
development-only structural-overload diagnostic. They are not pooled into the
lower-loading training gate merely to increase sample count.

After an independently verified damping checkpoint is frozen, moving-wall
benefit is measured only as damping-plus-wall minus identical damping.
