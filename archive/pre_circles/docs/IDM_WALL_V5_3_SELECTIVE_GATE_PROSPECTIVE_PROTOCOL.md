# IDM wall v5.3 selective-gate prospective support protocol

Status: **owner-authorized and frozen before prospective outcomes are opened.**

Date: 2026-08-31

## Scope and non-retroactivity

This protocol does not overturn the failed v5.3 seven-case Stage-1 result.
It prospectively tests the unchanged v5.3 treatment under the owner-adopted
two-stage estimand. The treatment, outcome budgets, replay precision and
observation-v2 schema are unchanged. The new question is whether a selective
Gate F/Gate B program has adequate physical-delivery and outcome support; raw
wall attempts are not required to be universally safe.

## Frozen causal arms

- Control: `idm_analytical_disturbance_p30_v1` with zero learned residual.
- Treatment: the identical analytical damping plus
  `idm_compact_moving_wall/5.3` during its declared override.
- Outside the override and after release, both arms continue identical
  analytical damping.
- v5.3 remains full occupied-lane `compact_moving`, with pre-latched ranked
  same-lane reserves, at most one handoff per lane, no partial-lane HOLD,
  two-sample compactness enforcement, and a fixed five-second linear taper.
- No parameter or taper shape may be selected using prospective outcomes.

## Prospective cohort and seed discipline

- IDM demand: 2700 veh/h mainline and 400 veh/h ramp.
- Treatment-blind penetration grid: 10%, 15%, 20%, 25%, 30%.
- Prospective support seeds: **4313--4332** (20 independent seeds).
- Decision steps: 620, 660, 680 and 700.
- Common terminal controller step: 960; paired outcome horizon: 260 steps.
- Seeds 4301--4312 remain diagnosis only.
- Seeds 4401--4520 remain predictor-development candidates and are not opened
  by this support pilot.
- Seeds 4601--4660 remain validation candidates.
- Seeds 5001--5060 remain sealed for final paper evaluation.

Before execution, the runner must fail closed if any seed 4313--4332 already
has a non-protocol run artifact. All windows from a seed remain one correlated
group.

## Stage P0: treatment-blind feasibility scan

Run analytical damping only. At each frozen decision, evaluate v5.3 formation,
full occupied-lane coverage, projected compactness, authority, gaps and TTC.
No wall is applied and no outcome, benefit or treatment-effect field may be
written.

Starting at 10% penetration, select the first level satisfying all of:

- at least 14 of 20 seeds contain an admitted window;
- at least 20 admitted seed-window cases;
- admission spans at least two decision steps;
- every stored admission reconstructs exactly from predecision fields.

If no level passes, stop. Do not relax full-lane coverage, inspect outcomes,
or move to a higher unlisted penetration.

## Stage P1: bounded paired headroom pilot

At the selected penetration, replay every frozen seed/window. A deterministic
physical veto writes a refusal record and never opens a treatment branch.
Every admitted start receives a paired damping and v5.3 branch. The first two
admitted cases by seed/decision order receive exact damping and treatment
repeats.

Every case must persist observation-v2 features, feature/profile hashes,
predecision scan, dose components, wall lifecycle, member/handoff diagnostics,
full-horizon safety/service outcomes and the joint Gate-B label.

## Frozen support gates

Integrity must satisfy all of:

- exact damping and treatment repeats;
- exact branch identity through the declared decision;
- wall override restricted to current members and active treatment;
- identical out-of-band damping;
- no partial-lane HOLD;
- taper reaches zero and release returns to analytical damping;
- zero excess collisions or teleports;
- treatment effects exceed exact replay noise.

Physical and label support must satisfy all of:

- at least 80% of admitted attempts achieve the compact composite dose;
- dose-qualified treatments span at least six seeds;
- Gate-F positive and negative/refusal support each span at least three seeds;
- safety-admissible and unsafe dose-qualified outcomes each span at least three
  seeds;
- joint safe/service-noninferior/meaningfully-beneficial `Y=1` outcomes span
  at least three seeds;
- `Y=0` dose-qualified outcomes span at least three seeds;
- no class is created by duplicate windows or one seed outlier.

The raw attempted-wall safety fraction is reported but is not a pass criterion
under the approved selective estimand. The later frozen deployable gate must
achieve at least 90% safety among selected activations on untouched seeds.

## Decisions

- If P0 fails: geometry/penetration support no-go; do not generate outcomes.
- If dose fails: diagnose formation, handoff or compactness; do not add seeds.
- If integrity fails: treatment no-go; do not fit a predictor.
- If label diversity fails: the pilot is insufficient for learning; do not
  manufacture balance with post-hoc seed selection.
- If all support gates pass: freeze hashes and authorize grouped
  predictor-development generation on separately assigned seeds 4401--4520.

No model fitting, threshold selection, probability claim or final-test opening
is part of this pilot.

## Planned commands and runtime

The versioned runners must be implemented and tested before execution. Planned
commands are:

```powershell
python -m experiments.idm.scan_idm_wall_v5_3_feasibility `
  --workers 4 `
  --output runs/probe/idm_wall_v5_3_feasibility_p10_p30_s4313_4332

python -m experiments.idm.run_idm_wall_v5_3_support_pilot `
  --workers 4 `
  --feasibility-scan runs/probe/idm_wall_v5_3_feasibility_p10_p30_s4313_4332 `
  --output runs/eval/idm_wall_v5_3_support_pilot_s4313_4332
```

Estimated wall time is 2--5 hours for the treatment-blind grid and 1--2 hours
for admitted paired branches on four workers. Both runners must be resumable.

