# IDM wall v5.5 density-duration headroom protocol

Status: **approval-bound frozen protocol; implementation and execution are not
authorized by this document.**

Date: 2026-08-31

## Purpose and scientific boundary

This development-only pilot is the smallest matched experiment that jointly
tests whether the current wall fails because it is too short, is activated at
the wrong time, lacks upstream runway, or is applied in structural overload.
It does not train a predictor, change a label, open validation/final data, or
claim collapse prevention.

The 20-second arm must reproduce `idm_compact_moving_wall/5.4` exactly.  The
30- and 40-second arms are separately fingerprinted v5.5 duration variants.
Their only treatment difference is requested pulse duration.  Wall geometry,
target-strength rule, full occupied-lane coverage, compactness, member and
reserve selection, one-handoff limit, safety projection, emergency handoff,
five-second linear taper, analytical damping base, IDM parameters, demand
generation and outcome definitions remain unchanged.

Changing duration changes the scientific treatment.  The required new
contract and runner may be implemented only after owner approval of this
document.  No adaptive controller is part of this pilot.

## Cohort and causal arms

- Development seeds: 4401--4412 only.  They are already development-exposed
  and remain excluded from later prospective evaluation.
- Mainline demands: 2200, 2400 and 2700 veh/h.
- Ramp demand: 400 veh/h.
- CAV penetration: 30%, using the existing seeded-scatter contract.
- Common causal origin: all arms replay from time zero with the same seed.
- Candidate activation steps: 620 and 700.
- Requested durations: 20, 30 and 40 seconds.
- Common terminal controller step: 960.
- Control: frozen `idm_analytical_disturbance_p30_v1` damping.
- Treatment: identical damping plus exactly one v5.5 duration arm during the
  declared compact-wall override.

Step 620 is the activate-early arm.  Step 700 is a true defer-to-late arm: it
follows the identical damping trajectory until 700.  Every treatment branch
must be identical to damping through its own declared decision.  All outcomes
end at step 960, so later activation cannot gain or lose follow-up merely by
shifting the evaluation horizon.

The runner must record nominal demand and measured predecision mainline/ramp
flow, density, occupancy, mean speed, speed variance, queue, discharge,
ramp-arrival phase, sustained-congestion state, member travel runway and
release-boundary margin.  No nominal demand is called the critical density in
advance.

## Branch accounting and repeats

There are 36 `(demand, seed)` units.  Each unit has one damping control and six
wall branches from two timings by three durations:

- primary branches: `36 * (1 + 2 * 3) = 252`;
- exact repeats: all seven branches for two treatment-blind units per demand,
  `6 * 7 = 42`;
- total: **294 SUMO branches** and **216 primary wall-versus-damping
  comparisons**.

Within each demand, repeat units are the first two seeds in numeric order with
at least one timing that Gate F admits for all three durations using only
predecision geometry and runway.  If two such units do not exist, the pilot
stops without importing or substituting a seed.  Refused timing/duration arms
remain refusal records; they are not silently removed or replaced.

Duration response is estimated only on timing states whose declared member
identities are identical and whose predecision Gate-F check admits all three
durations.  Duration-dependent refusals diagnose runway/placement limitations
and are reported separately.

## Treatment-dose contract and diagnostics

Requested duration includes assembly, HOLD and the existing five-second
in-horizon taper.  The existing hard-safety abort retains precedence and may
return immediately to zero wall authority and identical damping.

Every branch records:

- requested and realized duration;
- assembly, HOLD, taper and zero-authority steps;
- controlled CAV-seconds and occupied-lane-seconds;
- lane coverage and compact span per active sample;
- member and reserve identities, changes and handoffs;
- requested target, realized member speed and command/projection survival;
- release and abort reason;
- front/rear gaps, TTC, hard-brake and low-TTC event streams;
- ramp arrivals, lane changes, queue/storage, speed variance and discharge;
- exact out-of-band damping action deltas;
- postrelease safety/service effects through common step 960.

The pilot persists bounded 1-Hz aggregate trajectories plus event-level
vehicle diagnostics.  Full anonymous vehicle histories are not required.

## Integrity gates

All applicable repeats and branches must satisfy:

1. exact damping-versus-damping replay identity;
2. exact branch identity through each declared decision;
3. wall override only for declared treatment members during treatment;
4. identical out-of-band damping actions;
5. no partial-lane HOLD and the existing compactness contract;
6. smooth taper or explicit emergency zero-authority handoff back to identical
   damping;
7. zero new collision or teleport;
8. deterministic fingerprints whose only cross-duration change is requested
   duration;
9. treatment effects larger than exact-repeat numerical noise.

An integrity failure stops outcome interpretation.  It is not converted into
a neutral or harmful treatment label.

## Frozen treatment-qualification gates

For a duration arm to remain a candidate:

- at least 80% of feasible activations receive at least 80% of requested
  duration;
- at least 80% occupied-lane coverage is maintained over active treatment;
- assembly completes and at least 50% of realized override time is HOLDING;
- at least 80% of active samples meet the existing 50-m compact-span limit;
- at least 90% of dose-qualified completed treatments are safety-admissible;
- no existing safety or service budget is relaxed;
- completed beneficial outcomes span at least three independent seeds;
- completed nonbeneficial outcomes span at least three independent seeds;
- delay remains a service-noninferiority guardrail, while hard braking,
  low-TTC exposure and merge discharge define meaningful physical benefit;
- results and failures are reported by nominal demand, measured traffic state,
  timing and duration, with no single seed supplying the conclusion.

If collapse prevention remains a primary paper target, at least one measured
transition-state stratum at 2200 or 2400 veh/h must contain prevention in three
independent seeds and nonprevention in three independent seeds.  No collapse
diversity is imposed on 2700 unless the pilot itself establishes recoverable
headroom there.  Otherwise 2700 is reported as structural-overload
safety/service mitigation.

## Duration and timing decisions

A 30- or 40-second arm supports the next adaptive-duration design only when it:

1. passes every integrity and treatment-qualification gate;
2. produces a meaningful safety/service improvement over the matched
   20-second arm in at least three independent seeds;
3. has strictly more matched improvements than matched degradations;
4. exceeds exact-repeat noise and is not supplied by a single demand or timing;
5. does not conceal upstream storage, ramp-service loss or postrelease harm.

Timing is considered actionable only if early versus late activation changes
the frozen joint safety/service outcome beyond replay noise in at least three
independent seeds without one timing violating the 90% safety gate.  This pilot
may demonstrate timing headroom, but it does not authorize a timing learner.

## Predeclared interpretation tree

1. If 30/40 seconds help safely, design an overlay-owned adaptive-duration
   controller with a minimum probe, sustained recovery rule and hard maximum.
2. If a longer arm helps but loses dose because the cohort reaches the release
   boundary, test a sequential successor wall with one atomic effective front.
3. If all durations share the same early braking during their common first 20
   seconds, duration is not the repair; test a separately frozen weaker,
   slower-forming wall with rear-feedback limits.
4. If longer arms are refused for runway before activation, test a small fixed
   library of upstream placement/control-area options.
5. If early/late results differ safely, retain timing as a later Gate-B or
   discrete-option question after treatment qualification.
6. If ramp insertions or lane changes dominate newly captured adverse-event
   precursors, design CAV-only merge-gap/ramp coordination; do not enable the
   legacy autonomous `ramp_pace` path.
7. If several separately qualified actions work in different states and a
   constrained retrospective oracle beats every fixed action across seeds,
   consider MPC or learned discrete selection.
8. If 2200/2400 respond but 2700 does not prevent collapse, treat 2700 as
   structural overload rather than manufacturing collapse headroom with seeds.
9. If safety/service improve while collapse persists, pivot honestly to
   conditional safety-and-service escalation with collapse prevention as a
   transition-regime secondary outcome.
10. If dose passes but no treatment changes outcomes, redesign the actuator;
    do not add predictor seeds.

Sequential walls are rejected as the first change because two complete compact
cohorts are unverified at p30 and an atomic full-cohort handoff does not yet
exist.  Strength and placement are held as separately fingerprinted fallbacks,
not mixed into this duration test.  Merge coordination is deferred because the
existing audit cannot causally attribute most delayed events to ramp insertion.
MPC is deferred because the ten-second shadow misses delayed safety harm and no
qualified multi-action oracle exists.

## Required implementation slice after approval

Only the main agent may implement this protocol.  The expected files are:

- new `experiments/idm/wall_v5_5_duration_contract.py`;
- new `experiments/idm/run_idm_wall_v5_5_density_duration_pilot.py`;
- new `tools/analyze_idm_wall_v5_5_density_duration.py`;
- focused contract, runner and analyzer tests under `tests/`;
- `idm/contract.py` only for a tested 2200/2400/2700 scenario resolver if the
  runner cannot use the existing `scenario_config` and
  `transition_scenario_config` functions directly.

The fixed-duration discriminator requires no change to
`policies/counterfactual_escalation.py`.  A later adaptive-duration controller
would require a separately approved version and changes to its initialization,
`act()` lifecycle and new closed-loop release logic, plus versioned adaptive
dose semantics in `counterfactual/twin_replay.py` and the dose analyzer.

## Candidate command, runtime and storage

The module below does not exist yet and this command must not be run before
implementation, focused tests and full `python -m pytest -q` pass:

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
$env:PYTHONPATH = "D:\research-dense;$env:PYTHONPATH"
Set-Location D:\research-dense

python -m experiments.idm.run_idm_wall_v5_5_density_duration_pilot `
  --mainline-demands 2200 2400 2700 `
  --ramp-demand 400 `
  --cav-penetration 0.30 `
  --seeds 4401 4402 4403 4404 4405 4406 4407 4408 4409 4410 4411 4412 `
  --common-start-step 620 `
  --activation-steps 620 700 `
  --durations-s 20 30 40 `
  --terminal-step 960 `
  --repeat-count-per-demand 2 `
  --workers 4 `
  --output runs/probe/idm_wall_v5_5_density_duration_p30_s4401_4412
```

Using the completed v5.4 development rate of approximately 209 worker-seconds
per branch, 294 branches require about 17.1 worker-hours.  Expected elapsed
time is 5--7 hours at four workers or 3--4.5 hours at eight workers after
allowing for SUMO and CPU contention.  Bounded aggregate/event output should
remain below 0.5 GB; reserve 1 GB.  Persisting full vehicle trajectories is not
authorized and would increase storage to several GB.

## Boundary after this protocol

No validation seed 4601--4660 or final seed 5001--5060 may be read.  No
predictor-scale generation, Gate-B fitting, probability calibration claim or
IDM generalization claim follows from this protocol.  If a fixed duration
qualifies and action diversity remains, the next owner decision is whether to
implement adaptive duration.  Only after a revised treatment qualifies may
Gate F be frozen and Gate B be developed on a separately frozen cohort.
