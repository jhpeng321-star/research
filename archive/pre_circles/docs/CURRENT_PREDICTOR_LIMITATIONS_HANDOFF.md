# Current Predictor Limitations and Verification Handoff

## Workspace and paper

Workspace: `D:\research-dense`

Proposed IEEE IV paper title:

> Counterfactual Escalation from Decentralized CAV Damping to Coordinated
> Moving-Wall Control

This document records the current status after the moving-wall lifecycle was
rewritten. It supersedes older statements that the source code still terminates
the wall whenever runtime formation coverage changes. That failure mode existed
in the previous implementation, but the current source now contains an explicit
assemble-hold-release controller. The new implementation has not yet been
validated by a real SUMO pilot.

Preserve the dirty worktree and all user changes. Do not resume medium
optimization in this workspace.

## Bottom-line diagnosis

The software contract is currently healthy, but the predictor is not verified.
The remaining blockers are empirical and statistical:

1. No SUMO artifact was generated after the current assemble-hold-release
   rewrite.
2. Existing pilot rows predate the rewrite and do not contain the new
   `treatment_dose` contract.
3. Previous pilots still delivered incomplete and variable wall treatments.
4. The available data contain essentially no collapse-prevention headroom.
5. The current model predicts a composite `wall_selected` label, not collapse
   probability or conditional collapse-prevention effect.
6. Formation feasibility, safety, service, and heterogeneous benefits remain
   combined in the primary target.
7. The traffic states do not yet isolate the metastable transition band.
8. There are too few independent positive groups for reliable probability
   calibration.

More seeds using the old data-generating protocol will not solve these issues.

## What the current predictor actually predicts

The primary model in `counterfactual/escalation_v2_model.py` predicts the
probability that the safety/service oracle assigns:

`wall_selected = True`

The label in `counterfactual/safety_service_oracle.py` requires all of the
following:

- The wall actually engaged.
- The treatment remained safety-admissible.
- Traffic service remained noninferior.
- At least one meaningful benefit occurred:
  - fewer hard-brake onsets;
  - less low-TTC exposure; or
  - greater merge discharge.

Therefore, the current predictor does **not** directly estimate:

- `P(future collapse | predecision state)`;
- `P(wall prevents collapse | predecision state)`; or
- a clean conditional average treatment effect for collapse.

The earlier all-case AUROC was largely driven by predicting whether CAV
placement allowed the wall to engage. It did not demonstrate reliable benefit
prediction after a valid wall treatment was delivered.

## Current implementation status

`policies/counterfactual_escalation.py` now includes:

- explicit `DAMPING`, `ASSEMBLING`, `HOLDING`, `RELEASING`, and `COOLDOWN`
  states;
- frozen activation lanes;
- latched wall members;
- deterministic same-lane handoffs;
- runtime formation gaps treated as dose diagnostics rather than automatic
  aborts;
- safety aborts for genuinely dangerous conditions;
- return to the identical frozen damping controller after release; and
- continued identical damping actions for out-of-band CAVs.

The revised generator and replay path now record treatment-dose diagnostics,
including duration, CAV-seconds, lane coverage, handoffs, and control deltas.

Verification performed on the current source:

- Focused counterfactual suite: 37 tests passed.
- Complete repository suite: 468 tests passed.
- There were no test failures. The warnings were a tensor-conversion warning
  and an inability to write the pytest cache; neither invalidated the tests.

These tests establish software invariants under test fixtures. They do not
establish treatment efficacy or sufficient treatment dose in SUMO.

## Artifact/source timing mismatch

The current controller and generator were modified during the morning of
2026-08-26. The latest available treatment-development artifact under
`runs/probe/counterfactual_escalation_predictor_v2/` completed around 00:49,
before the rewrite.

Consequences:

- No available pilot validates the current assemble-hold-release treatment.
- Existing rows do not contain the top-level `treatment_dose` fields required
  by the current treatment-repair gate.
- Old model reports and pilot summaries must not be used as evidence that the
  current controller works.
- A new small pilot is required; another predictor-scale dataset is not.

## Evidence from the most recent pre-rewrite pilots

The four newest development pilots showed:

### Strict occupied-lane coverage, 20% CAV

- 12 cases from 4 seeds.
- 2 cases engaged.
- Mean realized override: 3 of 60 requested steps.
- No full pulse.
- No selected wall cases.
- Neither engaged arm experienced sustained congestion.

### Strict occupied-lane coverage, 30% CAV

- 12 cases from 4 seeds.
- 4 cases engaged.
- Mean realized override: 6 of 60 requested steps.
- No full pulse.
- No selected wall cases.
- Neither engaged arm experienced sustained congestion.

### Partial-lane development variant, 20% CAV

- 8 cases from 4 seeds.
- 5 cases engaged.
- Mean realized override: 7.8 of 60 requested steps.
- No full pulse.
- One case passed the older delay-oriented development oracle.
- Neither engaged arm experienced sustained congestion.

### Partial-lane development variant, 30% CAV

- 8 cases from 4 seeds.
- 6 cases engaged.
- Mean realized override: 31.7 of 60 requested steps.
- No full pulse.
- No selected wall cases.
- Neither engaged arm experienced sustained congestion.

Partial-lane operation improved engagement and dose, especially at 30% CAV,
but did not validate a completed wall treatment or collapse-prevention effect.
These numbers describe the old implementation and are treatment-development
evidence only.

## Remaining scientific and statistical limitations

### 1. No current treatment-dose evidence

The highest-priority question is whether the new controller can sustain the
declared treatment. Until a real pilot answers this, model work is premature.

The current treatment-repair criteria in `counterfactual/headroom.py` require:

- at least 80% realized duration for a dose-qualified case;
- at least 80% lane coverage for a dose-qualified case;
- at least 80% of feasible activations to be dose-qualified;
- at least 90% safety-admissible completed treatments;
- beneficial completed treatments across at least 3 independent seeds; and
- nonbeneficial completed treatments across at least 3 independent seeds.

### 2. Collapse prevention is not a mandatory gate

The treatment-repair summary reports collapse-prevention counts, but collapse
prevention is not currently included in its mandatory pass conditions. A pilot
could pass through safety or service diversity while showing zero collapse
prevention.

If the paper's primary claim is escalation to prevent metastable collapse,
freeze an additional criterion before running the pilot. A reasonable
development criterion is collapse prevention across at least 3 independent
seeds, with nonprevented or harmful completed treatments also present for
action diversity. The exact threshold should be declared as a development gate,
not a final efficacy claim.

### 3. The treatment geometry is not scientifically frozen

The generator currently defaults to `static_all`, which controls all eligible
CAVs in the formation band. This can resemble distributed in-band speed
control, rather than a compact aligned moving wall.

The newer `compact_moving` geometry selects spatially and kinematically coherent
members but remains treatment-development-only. The team must decide and freeze
whether the paper's treatment is:

- a distributed in-band CAV wall;
- a compact cross-lane moving wall; or
- a supervisor-consistent partial-lane wall.

Do not choose the final treatment definition after inspecting the model or
locked-test outcomes.

### 4. There is essentially no collapse-treatment-effect target

In the older 300-case primary data, the wall prevented sustained congestion in
only one case. In the latest small IDM pilots, none of the engaged cases
collapsed under either arm.

This produces stable cases where neither arm collapses and saturated cases
where both arms collapse, but almost no cases in which the wall changes the
collapse outcome. A collapse-treatment-effect predictor cannot be learned from
this label distribution.

### 5. The primary label mixes feasibility and several outcomes

The current all-eligible target combines:

- initial and runtime feasibility;
- realized engagement;
- treatment safety;
- service noninferiority; and
- an OR across hard-braking, TTC, and discharge benefits.

This is a valid overall operational-success label, but it is not a clean causal
treatment-effect target. It also permits apparently good performance driven
mainly by CAV placement and formation feasibility.

Use a two-stage deployment structure:

1. A deterministic predecision formation/safety eligibility gate.
2. A probabilistic benefit and risk model for eligible cases expected to
   receive an adequate treatment dose.

Do not train by conditioning on eventual post-decision engagement. Instead,
record and use the formation and activation-safety status known at the declared
decision time.

### 6. Fixed decision times mix different causal tasks

Decisions at fixed times mix:

- stable traffic;
- pre-breakdown traffic;
- already-collapsed traffic; and
- saturated traffic that is effectively guaranteed to collapse.

Separate prevention from recovery. A prevention cohort must exclude states that
have already met the sustained-collapse definition at the decision time.

Use observable, predecision event landmarks such as:

- merge-speed decline and recent slope;
- speed or headway variance growth;
- queue growth rate;
- upstream/downstream imbalance;
- ramp pressure;
- discharge deterioration; and
- CAV lane coverage and longitudinal placement.

### 7. The metastable transition band has not been mapped

First estimate damping-only collapse probability across a narrow dense/hyper
grid. Select demand, ramp-flow, penetration, and decision windows where future
collapse probability is approximately 20--80%. Do not resume medium
optimization.

A probabilistic model needs repeated observations in this transition band. It
cannot learn useful discrimination from regimes that almost never or almost
always collapse.

### 8. Model dimensionality exceeds the evidence

The richest current feature set contains approximately 58 variables. The older
engaged cohort contained only 67 independent seed groups and 20 selected
examples.

The contract minimum of 60 development groups and 20 positives permits an
exploratory model fit; it is weak support for a calibrated probability claim.
Begin with 6--12 mechanism-based variables and seek substantially more
independent positive and negative groups. If feasible, target at least 50
positive treatment-effect groups for stable calibration, while reporting
uncertainty honestly if that count cannot be achieved.

### 9. Probability calibration and prospective value remain untested

The current code produces grouped out-of-fold probabilities and reports AUROC,
average precision, Brier score, and log loss. It has not yet demonstrated:

- reliability/calibration curves;
- calibration intercept and slope;
- seed-clustered confidence intervals;
- frozen prospective performance on locked seeds;
- complete closed-loop policy value;
- improvement over health and congestion-risk gates; or
- lower intervention duty than always-wall.

AUROC alone is insufficient for claiming a useful probability predictor.

### 10. Observability and transfer assumptions must be explicit

Several features use mixed-traffic state for HDVs and CAVs. The paper must
assume RSU/infrastructure sensing plus V2X aggregation, or restrict features to
measurements realistically available to CAVs.

The damping checkpoint was trained primarily under the frozen Krauss mixture.
IDM experiments therefore test distributional transfer as well as wall
treatment. Do not claim IDM generalization until the damping controller and
escalation treatment have separate evidence under IDM.

### 11. Protocol-development seeds must not become evaluation seeds

Multiple wall variants have been examined on seeds 401--408. These are
treatment-development seeds and should be explicitly excluded from model
validation and locked testing. Freeze the treatment parameters before opening
new evaluation seeds.

## Required immediate experiment

Run a corrected 12--20-seed headroom pilot using the current source and new
`treatment_dose` schema. This is not a predictor-training dataset.

Before starting:

1. Freeze the wall geometry and parameter set being tested.
2. Freeze whether full occupied-lane coverage or a scientifically justified
   partial-lane formation defines the treatment.
3. Freeze the decision windows or event-landmark rule.
4. Add a collapse-headroom pass condition if collapse prevention remains the
   paper's main claim.
5. Mark every previously inspected treatment-development seed as excluded from
   later evaluation.

The pilot should pass only if:

- damping-versus-damping replay identity passes;
- predecision branch identity passes;
- in-band override and out-of-band damping identity pass;
- most feasible treatments receive the declared dose and lane coverage;
- completed treatments are acceptably safe;
- beneficial and nonbeneficial completed treatments span independent seeds;
- treatment effects exceed replay/numerical noise; and
- collapse-prevention diversity exists if collapse prevention is primary.

If treatment-dose criteria fail, repair geometry or handoffs. If dose passes but
the wall does not change outcomes, redesign the wall action. If the wall changes
outcomes but only outside the collapse definition, pivot the primary target
honestly to safety stabilization or service/discharge benefit.

## Recommended predictor architecture after headroom passes

Use separate stages and heads:

1. **Deterministic feasibility gate** using only declared predecision state.
2. **Baseline collapse-risk head** estimating collapse under damping.
3. **Treatment-effect head** estimating the change in collapse risk under a
   completed wall treatment.
4. **Safety head** estimating hard-braking, TTC, collision, and teleport risk.
5. **Service head** estimating discharge and delay effects.

A decision can then be framed as:

`expected collapse cost avoided + expected service benefit - expected safety and intervention cost`

Start with regularized logistic regression, a generalized additive model, or
shallow constrained boosting using 6--12 physically motivated features. Split,
bootstrap, and calibrate by complete scenario-seed group.

## Claim discipline

The defensible current claim is:

> The damping-preserving assemble-hold-release overlay has passed software
> contract tests and is ready for a corrected treatment-dose headroom pilot.

Do not yet claim:

- that the moving wall reliably completes its treatment;
- that it prevents metastable collapse;
- that the counterfactual predictor works;
- that predicted values are calibrated probabilities;
- that the method generalizes to IDM; or
- that the learned gate improves closed-loop performance.

## Immediate instruction for the main agent

Read `AGENTS.md`, `docs/COUNTERFACTUAL_ESCALATION_HANDOFF.md`,
`docs/COUNTERFACTUAL_ESCALATION_V2_PLAN.md`,
`docs/PREDICTOR_DIAGNOSIS_HANDOFF.md`, and this file. Inspect the dirty worktree
without discarding any changes. Confirm the current source and artifact timing
mismatch. Then design and run only the corrected 12--20-seed treatment-dose
pilot after freezing its geometry, cohort, and pass/fail criteria. Do not train
another predictor or generate a paper-scale dataset until the pilot passes.
