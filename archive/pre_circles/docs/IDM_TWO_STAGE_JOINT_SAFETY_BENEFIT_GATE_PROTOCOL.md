# IDM two-stage joint safety-and-benefit escalation gate

Status: **owner-approved current scientific contract (2026-08-30); Phase A
authorizes protocol adoption and a diagnostic development observation-value
screen only. Predictor-scale generation, fitting, and prospective claims
remain separately gated.**

Date: 2026-08-30

## Why this protocol exists

The v5.2 wall has genuine treatment-effect diversity but is not uniformly
safe: three independent seeds contain safe, meaningful wall benefits, while
five of ten dose-qualified cases exceed the full-horizon hard-brake budget.
Artifact diagnosis shows that the unsafe effects are delayed downstream
trajectory divergences, not a mechanically obvious release failure. A raw
90%-safe treatment gate therefore blocks the very conditional-selection
problem the paper proposes to solve.

This protocol defines the alternative estimand explicitly: the RSU should
activate a physically deliverable wall only when it is predicted to be both
safe and beneficial relative to frozen damping. It does not claim that the
wall is universally safe or that the current predictor works.

This protocol replaces the old requirement that 90% of all raw
wall activations be safe with a prospective requirement that at least 90% of
the **gate-selected** wall activations be safe. This is a material scientific
change and requires approval before any execution.

## Frozen counterfactual arms

- Control: exact deterministic `idm_analytical_disturbance_p30_v1` with zero
  learned residual.
- Treatment: the same damping controller plus the frozen compact,
  full-occupied-lane moving wall.
- Outside the active override, including after release, both arms use identical
  analytical damping.
- The next treatment is separately versioned as v5.3. It adds only the frozen
  treatment-integrity repairs in `IDM_WALL_V5_3_MECHANISM_REPAIR_PROTOCOL.md`:
  pre-latched same-lane reserves with bounded handoff, compactness enforcement,
  and a fixed in-horizon linear taper. These mechanics may not be tuned on
  outcomes and are not claimed to repair delayed downstream safety failures.
- Full-horizon outcomes and common terminal horizons remain unchanged.

## The two distinct gates

### Gate F: treatment feasibility and dose reliability

Gate F answers only: **can the frozen wall be delivered as specified from this
decision state?**

Its label is the compact composite dose contract:

`D = duration>=80% AND lane_coverage>=80% AND assembly_complete AND
holding>=50% AND compact_geometry>=80%`.

It may use decision-time geometry, occupied lanes, candidate/reserve CAV
locations, gaps, TTC, projected control authority, member speed mismatch, and
traffic state. It may not use delay, discharge, collapse, hard braking after
the decision, treatment utility, or the Gate-B label.

The first implementation should include a deterministic physical screen plus
a separately calibrated dose-reliability estimate. A failed realized dose
remains a failed attempted action; it may not be silently converted to
abstention during analysis.

### Gate B: conditional joint safety and benefit

Gate B answers: **conditional on an admissible attempted wall, should the RSU
prefer wall-plus-damping to damping alone?**

The primary binary target is:

`Y = safety_admissible AND traffic_noninferior AND meaningful_benefit`.

Safety retains the frozen full-horizon collision, teleport, low-TTC, and
hard-brake budgets. Benefit retains paired delay and merge-discharge effects
and the frozen replay/noise deadband. Queue area is diagnostic rather than a
primary loss because controlled inflow can intentionally move a queue
upstream. Collapse prevention remains secondary unless prospective diversity
supports restoring it as a primary target.

Every dose-qualified unsafe, harmful, or neutral case is a negative Gate-B
example. None may be removed as “hopeless” after observing its outcome.

Gate B should expose both:

- calibrated `P(safety_admissible | x, D=1)`; and
- calibrated `P(meaningful benefit and service noninferiority | x, D=1,
  safety-admissible)` or an equivalent constrained expected-utility estimate.

This decomposition makes a safety refusal inspectable even if a wall might
improve service.

## Deployment rule

The deployable policy is:

`ACTIVATE = physical_screen(x) AND P(D=1|x)>=t_D AND
P(safe|x,D=1)>=t_S AND E[constrained_benefit|x,D=1,safe]>t_B`.

Otherwise it uses damping alone. Thresholds must be selected on development
and calibration partitions, frozen, and then applied once to untouched test
seeds. They may not be tuned on paper-test outcomes.

## Observation contract

Only information available at the declared decision is permitted:

- local wall-candidate geometry and per-lane availability;
- candidate/reserve front and rear gaps, relative speeds, and TTC;
- projected command survival and available deceleration authority;
- RSU aggregates over predeclared upstream, merge, and downstream regions;
- density/occupancy, harmonic speed, speed variance, flow, discharge trend,
  queue-growth trend, and lane imbalance;
- short causal history windows computed strictly before the decision;
- demand, penetration, decision time, and fixed scenario parameters.

No future state, realized dose, treatment phase, arm identity, outcome metric,
or seed identifier may appear as a feature. Feature computation must be
identical for both counterfactual branches through the decision.

The v5.2 observation audit found that the existing contract has no explicit
post-merge edge state, despite delayed unsafe divergence concentrating on that
edge. Before fitting, observation v2 must add a compact, frozen RSU aggregate
for post-merge occupancy/density, speed, variance, safety events, discharge,
and their strictly predecision trends. The similarly named existing
`local_downstream_minus_upstream_speed_mps` is an upstream-approach gradient;
it is not a post-merge measurement. See
`IDM_WALL_V5_2_DECISION_OBSERVATION_AUDIT.md`.

## Model-development order

1. Freeze schemas, feature timestamps, labels, seed partitions, thresholds,
   and deployable baselines.
2. Run an observation-value/headroom screen on development artifacts only.
   With the current ten completed cases this is diagnostic, not validation.
3. Compare simple interpretable models first: regularized logistic regression,
   shallow gradient-boosted trees, and a conservative rule baseline. A neural
   network is justified only after those baselines and sample size support it.
4. Use grouped cross-fitting by seed. No decision from the same seed may span
   train and validation folds.
5. Calibrate probabilities on a held-out calibration partition and lock the
   activation thresholds.
6. Evaluate once on untouched prospective seeds, including every eligible
   decision and every refusal.

## Minimum gates before scale-up

The initial development pilot must show:

- exact replay and predecision identity;
- no leakage or duplicate starting states across folds;
- both `D=1` and `D=0` across at least three seeds each for Gate F;
- both `Y=1` and `Y=0` across at least three dose-qualified seeds each for Gate
  B;
- treatment effects above replay noise;
- leave-one-seed-out value not dominated by a single seed;
- a deployable cross-fitted policy with positive constrained value over always
  damping, always wall, the deterministic feasibility-only rule, and a fixed
  density threshold;
- no evidence that seed, absolute vehicle ID, or postdecision variables carry
  the signal.

If these conditions fail, do not add a neural network. Diagnose observation
sufficiency or accept a conservative rule/no-go.

## Prospective acceptance gates

On untouched seeds, the frozen two-stage policy must satisfy all of:

- at least 90% of selected activations are safety-admissible, with every
  collision, teleport, low-TTC, and hard-brake failure reported;
- selected actions have positive paired constrained value versus always
  damping and versus feasibility-only activation;
- helpful selected treatments exceed harmful selected treatments;
- at least 70% of selected treatments are helpful or neutral;
- benefits and refusals span independent seeds rather than one outlier;
- dose success among selected activations is at least 80%;
- calibration and discrimination are reported for both gates, including Brier
  score, reliability intervals, AUROC/AUPRC where defined, and coverage;
- abstained cases are evaluated counterfactually so conservative selection
  cannot hide missed benefit;
- all claims are restricted to the frozen IDM scenario and tested penetration
  range.

## Seed discipline

- 4301--4312: opened mechanism-development and diagnosis only.
- 4313 onward: must be assigned prospectively before generation; no seed may be
  moved between development, calibration, and test after outcomes are read.
- 5001--5060: remain reserved for the final paper evaluation unless a later
  approved protocol assigns an explicit subset in advance.

## Owner decision and authorized scope

The owner approved this estimand on 2026-08-30. The paper claim is now: “a
two-stage RSU gate selectively escalates to a high-authority wall when physical
delivery, safety, and conditional benefit are jointly supported.”

The approval authorizes Phase A protocol adoption, v5.3 treatment-integrity
mechanics, the bounded seven-case mechanism replay, and a diagnostic
development observation-value screen. It does not authorize predictor-scale
generation, fitted models, threshold selection, calibration, or prospective
performance claims. Those require a separate Phase B approval.
