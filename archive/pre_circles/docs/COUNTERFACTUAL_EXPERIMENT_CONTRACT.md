# Counterfactual Treatment-Effect Gate — Experimental Contract v0.1

Status: **replay-validity milestone passed; a ten-seed hyper no-control
calibration produced a Phase-1 freeze candidate, with held-out validation still
required.** This contract defines the first paper experiment before
treatment-effect model development.

Implementation and run instructions: [COUNTERFACTUAL_PILOT.md](COUNTERFACTUAL_PILOT.md).

## Research question

Given the observable mixed-traffic state at a freeway merge, which available
headway-control action—off, soft, or strong—improves the future relative to no
control enough to justify its delay, safety, and actuation costs?

This is not merely congestion prediction. A high-risk state may be
uncontrollable, while a less severe state may be an inexpensive rescue. The
target is the **conditional benefit of action**, not the health score itself.

## Causal scope

The causal unit is one complete SUMO traffic realization at one pre-treatment
decision time. Vehicle interactions occur inside the unit; vehicles are not
treated as independent samples.

Paired branches identify intervention effects inside the frozen SUMO model,
scenario, platform, and version. They do not establish real-road causal effects.
The paper will call these **simulator-grounded counterfactuals**.

## Pre-treatment state `X_t`

Only values available before action selection are legal features:

- health `h`, short trend `dh`, and lagged health trajectory;
- lane-level and aggregate speed/density patterns;
- pre-merge queue length, queue growth, and tail position;
- observed inflow and time within the loading pulse;
- CAV count, longitudinal spread, lane coverage, and in-band availability;
- current rung, recent duty, time since activation/release, and action history.

The health factor remains a useful compressed feature but is neither the sole
input nor the treatment-effect target. Post-treatment values and future labels
are forbidden features.

## Phase-1 decision design

The first model is a **one-decision treatment rule**, not a repeated-action RL
policy. The provisional eligibility landmark is the first post-warmup entry
into a broad warning band (`h <= 0.45`) while the short trend is non-positive.
Eligibility selects when to ask for a decision; it does not select the action.

The landmark will be checked on no-control training seeds before dataset
generation. If it selects too few non-rescue or overload states, eligibility
will be broadened and then re-frozen before treatment comparisons.

## Treatment arms

All arms share the same pre-treatment action history and exogenous seed.

- `A0 — off`: wall disabled for the 120-step treatment window.
- `A1 — soft`: current `RUNGS[1]` wall applied for 120 controller steps.
- `A2 — strong`: current `RUNGS[2]` wall applied for 120 controller steps.

After the treatment window, every arm is released to no wall. The primary
outcome horizon is 300 controller steps after the decision; episode-end
outcomes are secondary. A later experiment may compare fixed-pulse and
hysteretic-release protocols, but they must not be mixed within one treatment
definition.

## Potential outcomes and estimands

For each action `a`, store the vector

`Y(a) = [congestion(a), incremental_delay(a), safety(a), recovery(a), duty(a)]`.

Primary conditional estimands are:

- congestion-risk reduction versus off;
- incremental delay versus off;
- safety-exposure change versus off;
- recovery-time change versus off.

The deployed gate will choose the least intensive action whose predicted
benefit clears a confidence margin and whose delay/safety costs satisfy frozen
validation-set budgets. It will not optimize a single unreported weighted
reward.

## Outcome contract

### Primary outcome

An independently defined sustained-congestion event derived from queue growth,
speed state, and time-aligned bottleneck discharge. Its thresholds must be
calibrated only on no-control training seeds, validated against episode
trajectories, and frozen before any treatment-effect comparison.

The current `h < 0.15 for 60 steps` label is retained only as a diagnostic. It
cannot be the primary outcome because `h` is also a policy feature.

### Secondary outcomes

- incremental network time loss over the treatment and outcome horizons;
- completed trips and properly aligned bottleneck discharge;
- minimum TTC, unique low-TTC encounters, and low-TTC exposure duration;
- actual hard-braking exposure, collisions, and teleports;
- recovery time and residual queue;
- controller duty, number of CAVs treated, and simultaneous lane coverage.

## Counterfactual branch protocol

The reference protocol replays every branch from time zero with:

- identical SUMO seed;
- identical CAV-selection seed;
- identical scenario and executable version;
- identical pre-treatment actions;
- only the treatment arm changed at the decision step.

TraCI save/load is not accepted until it passes against full replay. Current
route files use flows, SUMO RNG state is not saved by default, and SUMO does
not serialize every car-following/lane-changing internal state.

The first acceptance test is off versus off. At six-decimal precision, the two
branches must match at every step on supervisor observation, active vehicle
IDs, road/lane, lane position, speed, acceleration, cumulative time loss,
route index, and vehicle type.

## Dataset split and leakage controls

- split by whole `(scenario, car-following model, demand, penetration, seed)`
  groups, never by snapshot row;
- keep all treatment branches from one pre-treatment state in the same split;
- keep every snapshot from one episode in the same split;
- reserve an untouched seed set for prospective policy evaluation;
- reserve IDM or a second geometry as a strict zero-shot transfer test;
- fit eligibility and outcome thresholds on training/no-control data only.

## Required baselines

- no control;
- always-on wall;
- tuned health threshold and hysteresis;
- congestion-risk predictor that treats high-risk states;
- current PPO supervisor;
- counterfactual treatment-effect gate;
- paired-outcome oracle as a non-deployable upper bound.

## Phase-1 success criteria

Proceed to treatment-effect data generation only if:

1. two full off-to-off replays pass the state-fingerprint check;
2. density and incremental-delay defects are corrected;
3. the independent congestion label is calibrated and frozen;
4. all branch configurations, seeds, executable versions, and hashes are
   recorded.

Proceed to a paper claim only if the treatment-effect gate improves prospective
policy value over both the risk-prediction gate and tuned health gate, while
reducing unnecessary or harmful interventions at matched congestion risk.
