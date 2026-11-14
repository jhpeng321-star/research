# Path B — Sequential P30 Activation-Timing Headroom Pilot (Revised Protocol)

**Status:** frozen revision per user directive 2026-08-30; **executed
2026-08-30** — results in `docs/IDM_SEQUENTIAL_TIMING_HEADROOM_RESULTS.md`.
The pilot stopped after the oracle analysis; no observation model, reward
redesign or RL were started, and no seed above 4132 was touched.
**Date:** 2026-08-30
**Scope note:** the earlier candidate-1-vs-candidate-2 no-go did not test
true sequential timing; this protocol does.

## 1. Single frozen landmark schedule

For every (seed, cell) one deterministic native-IDM discovery trajectory is
run once from reset, under the frozen configuration (network, IDM/LC2013
types, 0.5 s integration / 1 s controller step, CAV selection seed,
penetration, landmark contract: `risk_onset` priority, persistence 3,
minimum separation 60 controller steps, max 3 landmarks, runway rank-1
candidate contract, admission contract). The complete ordered schedule
`t1, t2, t3` is recorded, with for each landmark:

- state/configuration hash (protocol hash + context hash + per-step
  physical-state and observation fingerprints);
- seed and cell;
- `t_i` in controller steps;
- rank-1 candidate ID, its observation fingerprint, admission feasibility;
- missing-landmark reason (no candidate state, no persistent signal, horizon
  limit, separation limit).

Recurrences 1/2/3 are **never** assembled from separate runs. The schedule
trajectory is verified deterministic by running the same discovery twice and
requiring identical instants, candidate IDs, observation fingerprints and
schedule hash (exact trajectory identity). The deferred arms remain identical
to native IDM until their activation landmark: `defer_r2` must match abstain
exactly through `t2` and `defer_r3` must match abstain exactly through `t3`.
Explicit branch-identity gates are evaluated at `t1`, `t2` and `t3` (the
recorded candidate observation fingerprints must match the arm's state at
the same controller step before activation; any mismatch invalidates the
state and is reported, never silently repaired).

## 2. Common starting state and terminal horizon

All arms begin at the exact state at `t1`:

- always abstain (native IDM to `H`);
- abstain replay (native IDM to `H`, exact replay identity);
- activate at `t1` (frozen P30 at `t1`, then native IDM to `H`);
- abstain until `t2` and activate once, then native to `H`;
- abstain until `t3` and activate once, then native to `H`.

One common terminal time per state:

```text
H = t3 + 60 controller steps   if t3 exists
H = t2 + 60 controller steps   if t2 exists (no t3)
H = t1 + 60 controller steps   otherwise
```

Treatment-relative 60-step effects from different starting times are never
compared as common-horizon policy values; every arm is measured over
`[t1, H]`.

## 3. Variable-horizon normalization (frozen equations)

Let `L = H − t1` (the common state horizon in controller steps), and let
`B_a` and `E_a` be the corridor severe-braking count sum and
acceleration-energy sum of arm `a` accumulated over the identical
`[t1, H]` window. References are the frozen per-step rates
`B_ref` and `E_ref` from
`runs/eval/idm_decentralized_lower_reward_refs_v1.json`
(SHA-256 `737b9ce4bb1945f7681b807941c86509ae3143128bd0729d2cb2aee5237c5b58`).

```text
u_B = (B_abstain − B_a) / (L * B_ref)
u_E = (E_abstain − E_a) / (L * E_ref)
u   = 0.5 * u_B + 0.5 * u_E

R(abstain)          = 0
R(activate)         = tanh(u)   if every paired constraint passes
R(activate)         = -2        otherwise
```

The rate form divides by the exact common-horizon length `L`, so a longer
`t2`/`t3` schedule does **not** receive more weight merely because its
common horizon contains more steps. Safety and service metrics (low-TTC
exposure steps, corridor delay rate, arrivals, merge discharge, collisions,
teleports, unexpected commands) are all computed over the identical
`[t1, H]` window for every arm. Treatment-dose constraints remain specific
to the single 20-step P30 activation (≥ 16 of 20 commands delivered).
Deadband: 0.02 fixed (never raised).

## 4. Action feasibility

- `activate_now` feasible iff the `t1` landmark and its candidate exist and
  pass the frozen admission contract;
- `defer_r2` feasible iff `t2` and its candidate exist and pass admission;
- `defer_r3` feasible iff `t3` and its candidate exist and pass admission;
- action masks are frozen from the native-IDM schedule;
- missing actions are reported per seed/cell, never imputed or silently
  dropped;
- no candidate-2 or multiple-activation action exists; exactly one P30
  activation per branch.

## 5. Treatment identity

The physical treatment remains exactly: `activation_speed_ceiling_v1`;
20 active steps; runway rank-1 candidate at the selected landmark; hard
safety projection authoritative (comfort/jerk/TTC/secure-gap/SUMO
speedMode 31, front dominates); projected release to exact native IDM;
8 m/s reference floor. Path B changes timing only — no change to P30
strength, duration, release or safety.

## 6. Minimum analyzable cohort (frozen from the schedule inventory)

Before any treatment outcome is opened, the schedule inventory must satisfy
all of the following (otherwise: report an insufficient-headroom cohort and
stop — never pool incomparable states):

1. at least 25 analyzable states (seed/cell with a `t1`);
2. coverage across at least six independent seeds;
3. more than one demand;
4. both 5% and 10% penetration where landmarks exist;
5. `defer_r2` feasible in at least six states across multiple seeds;
6. `defer_r3` feasible in at least six states across multiple seeds;
7. baseline (always-now) outcomes include both beneficial and
   nonbeneficial timing realizations.

(These minimums were predeclared against the schedule inventory before
treatment branches; the inventory values are recorded with the run.)

**Frozen inventory check (recorded 2026-08-30, before any treatment branch):
the inventory satisfies all criteria 1--6: 69 states with `t1` across 12
distinct seeds; 58 states with `t2`; 43 states with `t3`; all three demands
(1900/2000/2100); both penetrations with landmarks (p05: 33 states, p10: 36);
full schedule-identity exactness in 72/72 seed-cell checks. Criterion 7
(outcome diversity of the always-now baseline) is evaluated at analysis time
with the gate; if the always-now outcomes are all one-sided, the cohort is
reported as insufficient for timing diversity and no oracle is interpreted.**

## 7. Integrity gates (before efficacy interpretation)

1. exact abstain-versus-abstain replay identity;
2. exact common `t1` state identity (all arms match the schedule state);
3. exact preactivation identity for each deferred arm (`t2`/`t3` match);
4. zero treatment commands before the selected activation instant;
5. only the selected target commanded;
6. complete common `[t1, H]` horizon for every arm;
7. correct treatment dose (≥ 16 of 20 at the single activation);
8. smooth native-IDM handoff (projected release, no uncontrolled interval);
9. zero unexpected commands;
10. no safety-projection bypass;
11. zero added collisions/teleports;
12. exact reward-reference identity;
13. every missing landmark and infeasible action reported.

## 8. Oracle baselines

Compare the constrained timing oracle against, on the identical states:

- always abstain;
- always activate at `t1` when feasible;
- always defer to `t2` when feasible;
- always defer to `t3` when feasible;
- the current third-risk-onset rule (= defer to `t3` when `t3` exists;
  where `t3` is infeasible the rule abstains and is reported as infeasible,
  never treated as activated);
- the best fixed deployable timing rule selected from the predeclared fixed
  candidate set (never / now / r2 / r3, all frozen before outcomes; the
  maximum of their mean policy reward is the best fixed rule).

**Frozen fixed-policy semantics:** a fixed timing policy attempts every
scheduled feasible action; a post-activation dose or constraint failure
receives its frozen penalty (reward −2, harmful classification) and is
never retrospectively substituted by abstention; only a missing or
pre-action-infeasible timing abstains. The constrained oracle — and only
the oracle — may exclude outcome-inadmissible actions, because it is an
explicitly unattainable upper bound. Identical semantics must be used by
the global, subgroup and leave-one-seed-out computations (regression-tested).

## 9. Oracle pass gate

The constrained retrospective oracle (all actions with safety/service
constraints and adequate dose) must, on the joint state set:

1. gain ≥ +0.02 mean policy reward over the best fixed deployable timing
   rule;
2. meaningful gain (> 0.02) in at least three independent seeds;
3. more than one activation timing selected;
4. no single seed/state dominating the gain (gain remains ≥ 0 when each
   seed is removed);
5. adequate helpful/harmful/neutral diversity among oracle selections;
6. hard safety/service compliance (zero constraint failures);
7. effects beyond replay noise.

Additional mandatory reports: oracle gain with each seed removed; gain by
penetration; gain by demand; gain by recurrence; fraction of states where
each action is feasible; fraction of oracle value attributable to each
action. No observation model, reward redesign or RL is authorized if this
gate fails.

## 10. Testing and execution

Implement the separately versioned timing runner and analyzer. Focused tests
must cover: schedule determinism; common-state identity; defer-r2 and
defer-r3 preactivation identity; common-horizon calculation;
variable-horizon reward normalization (rate form); action masks; missing
landmarks; branch resume; exact replay; oracle calculations (including
leave-one-seed-out and value attribution). Then `python -m pytest -q`; the
exact test count is reported before launching SUMO. Use only seeds
4121–4132; never open 4201 or above; at most four SUMO workers; resumable
artifacts; periodic progress reports. Stop after the oracle analysis; do
not automatically proceed to an observation screen or RL.

## 11. After Path B (frozen decision tree)

- If the timing oracle **fails**: execute the approved Path A protocol on
  seeds 4201–4220 and validate the fixed rule (see Path A below).
- If the timing oracle **passes**: keep 4201–4240 untouched; perform the
  separately gated observation/value screen, reward audit and training using
  development data only; freeze the learned checkpoint; then use 4201–4220
  for its first prospective acceptance and 4221–4240 only as confirmation
  after the controller and acceptance analysis are frozen.
- Path A and Path B are never run in parallel.
