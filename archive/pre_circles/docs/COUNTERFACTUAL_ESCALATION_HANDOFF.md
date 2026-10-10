# Counterfactual Escalation — Session Handoff

Use this document to resume the project in another Codex session or on the main
working branch without relying on the previous conversation history.

## Copy/paste resume prompt

```text
Continue the IEEE IV counterfactual-escalation project in D:\research-dense.
Read AGENTS.md and docs/COUNTERFACTUAL_ESCALATION_HANDOFF.md first, then read
docs/COUNTERFACTUAL_ESCALATION_V2_PLAN.md and the directly referenced code.

The paper title is:
Counterfactual Escalation from Decentralized CAV Damping to Coordinated
Moving-Wall Control

The immediate task is Phase 1 of the v2 plan: freeze the exact decentralized
damping checkpoint and implement a damping-preserving moving-wall overlay.
The control arm must be frozen decentralized damping; the treatment arm must be
the identical damping controller plus a temporary coordinated moving wall.
Do not generate more paper-scale data or claim the predictor works until the
new damping-vs-damping-plus-wall replay passes identity/prefix tests and a
12–20-seed corrected headroom pilot shows safe action diversity.

Preserve the dirty worktree and all existing user changes. Do not resume medium
optimization; this workspace is for dense/hyper experiments. Run relevant unit
tests and then python -m pytest -q when the implementation slice is complete.
```

## Project and paper target

- Workspace: `D:\research-dense`
- Conference target: IEEE Intelligent Vehicles Symposium (IEEE IV)
- Current title:
  **Counterfactual Escalation from Decentralized CAV Damping to Coordinated
  Moving-Wall Control**
- Paper worksheet:
  `paper/ieee_iv_counterfactual_gate/main.tex`
- Full v2 plan:
  `docs/COUNTERFACTUAL_ESCALATION_V2_PLAN.md`
- Visual plan:
  `C:\Users\jpeng\.cursor\projects\d-research-dense\canvases\counterfactual-escalation-v2-plan.canvas.tsx`

The intended system is an RSU that receives recent mixed-traffic and CAV state,
keeps decentralized CAV damping as the normal controller, and escalates to a
coordinated moving wall only when the predicted incremental benefit is positive
with sufficient confidence and safety/formation constraints pass.

## Core scientific premise

The road is metastable, so the paper should not claim exact prediction of
future collapse. The prediction target is narrower and more defensible:

> Given the observable traffic history at the current decision time, would
> adding a temporary coordinated moving wall improve the future relative to
> continuing decentralized damping?

For a complete pre-treatment history `H_t`, define:

```text
A0: frozen decentralized damping for the complete outcome horizon
A1: the same frozen damping + one frozen moving-wall pulse
```

For cost metric `k`, use a benefit-positive effect:

```text
tau_k(H_t) = Y_k(A0 | H_t) - Y_k(A1 | H_t)
```

Reverse the subtraction for benefit metrics such as merge discharge. These are
**simulator-grounded counterfactual effects** inside the frozen SUMO model, not
real-road causal effects.

The health factor is retained as an input feature and as the main interpretable
baseline. It is not the outcome label and should not directly determine the
counterfactual target.

## What makes the proposed contribution novel

Moving bottlenecks, decentralized damping, cooperative merge control, RSU
coordination, and counterfactual prediction each exist separately. The proposed
contribution is their safety-constrained escalation architecture:

> An RSU estimates the state-specific incremental value of adding coordinated
> moving-wall control to an already active decentralized damping policy and
> escalates only when benefit, uncertainty, formation feasibility, and safety
> criteria pass.

If fully implemented and prospectively validated, the estimated novelty was
approximately 8/10 for IEEE IV. This is primarily systems-and-decision novelty,
not an entirely new machine-learning theory.

## Current implementation foundation

Reusable code already exists:

- `counterfactual/twin_replay.py`
  - deterministic replay from time zero;
  - branch fingerprints and trace comparison.
- `counterfactual/pilot.py`
  - pre-treatment traffic-pattern features;
  - CAV penetration, placement, lane coverage, local shock, spacing, braking,
    and TTC history features.
- `counterfactual/outcomes.py`
  - congestion, queue, delay, discharge, hard braking, TTC, collision,
    teleport, recovery, and intervention-duty outcomes.
- `counterfactual/oracle.py`
  - paired-outcome oracle and treatment-effect summaries.
- `counterfactual/readiness.py`
  - safety budgets and deployability screening.
- `counterfactual/gate_model.py`
  - grouped logistic/ridge models and grouped evaluation utilities.
- `experiments/counterfactual/train_counterfactual_gate.py`
  - preliminary binary gate.
- `experiments/counterfactual/train_counterfactual_two_head_gate.py`
  - continuous traffic-effect and safety-risk heads;
  - seed-grouped evaluation and optional locked-test inputs.
- `policies/wall_over_champion.py`
  - existing composition pattern where a wall controls in-band CAVs while the
    frozen champion controls out-of-band CAVs.
- `env/wall_supervisor.py`
  - RSU wall environment and wall rungs.

Relevant tests already exist under `tests/test_counterfactual_*.py` and
`tests/test_wall_supervisor.py`.

## Critical architectural gap

The current paired generator uses `WallSupervisorEnv`. In that environment,
wall action 0 releases active CAVs to SUMO. Therefore, the current estimand is:

```text
wall off vs wall on
```

It is not yet:

```text
decentralized damping vs decentralized damping + wall
```

This must be corrected before generating new paper-scale data. In v2:

1. The frozen damping policy computes actions for every active CAV in both
   branches.
2. When escalation is off, all damping actions are returned unchanged.
3. When escalation is on, the moving wall temporarily overrides only feasible
   in-band CAVs.
4. Out-of-band CAVs continue receiving the same damping actions.
5. After the pulse, all CAVs return to damping—not uncontrolled SUMO.
6. The old internal health trigger must not decide the experimental treatment;
   the replay schedule must issue an explicit intervention command.

## Current evidence and limitations

### Ten-seed / 30-case paired pilot

The current tau-free hyper-density dataset contains:

- 10 independent SUMO seeds, 101–110;
- three decision windows per seed: steps 120, 180, and 240;
- 30 paired cases;
- 21 off labels and 9 safety-admissible soft-wall labels;
- nine positive labels across six seeds;
- no strong-wall labels because the strong wall violated the calibrated safety
  envelope whenever extra traffic benefit was attractive.

Across the nine oracle-selected soft cases, the preliminary average effects
were favorable for queue area, delay, discharge, hard braking, and low-TTC
exposure. However, always-soft was not a good policy, so treatment heterogeneity
exists in the old comparison.

### Predictor result

The preliminary gate did not generalize to held-out seeds:

- health and health-change features were exactly zero across all 30 hyper cases;
- the health-only model always selected off;
- richer traffic and placement logistic models performed at or below chance;
- diagnostic threshold sweeps did not reveal a deployable operating point;
- the learned policy did not improve on always-off.

This result must be preserved and reported honestly. It establishes that:

- a scalar health gate saturates in the hyper-density regime;
- 10 independent seed groups are insufficient for the current predictor;
- random row splitting would create seed leakage;
- additional features or seeds cannot repair the paper title until the control
  arm is changed to actual damping.

Do not claim that the existing counterfactual predictor works.

Detailed current-result documents:

- `docs/COUNTERFACTUAL_EXPERIMENT_CONTRACT.md`
- `docs/COUNTERFACTUAL_PILOT.md`
- `docs/COUNTERFACTUAL_TAU0_N30_PRELIMINARY_RESULTS.md`
- `docs/COUNTERFACTUAL_GATE_N30_TRAINING_RESULTS.md`

## Traffic and vehicle-model facts

- Default scenario: `osm.hyper.sumocfg` through `config/defaults.py`.
- Dense experiments remain in scope; medium optimization must not be resumed in
  this workspace.
- The primary route files use a three-type HDV mixture without an explicit
  `carFollowModel`; SUMO therefore uses its default Krauss family.
- CAVs are retyped to `normal_car`; HDVs retain the route-file distribution.
- IDM configurations/probes exist and should be treated as zero-shot transfer
  or robustness evaluation unless the contract is deliberately changed before
  training.
- CAV penetration and realized placement/coverage are legal decision-time
  predictors and must be included in the experimental analysis.

## Target runtime gate

The final RSU decision should be structurally equivalent to:

```text
activate = eligibility
        AND formation_feasible
        AND lower_confidence_bound(any meaningful traffic benefit) > margin
        AND all traffic noninferiority bounds pass
        AND all safety upper-confidence bounds remain within budget
        AND no hard safety veto is active
```

Recommended continuous prediction heads:

Traffic effects:

- sustained-congestion event reduction;
- queue-area reduction;
- incremental-delay reduction;
- additional merge discharge.

Safety risks under treatment:

- hard-brake exposure;
- low-TTC exposure;
- collision/teleport hard vetoes until enough rare events exist to model risk.

Begin with interpretable regularized models. Use seed-grouped bootstrap
ensembles or grouped conformal calibration for uncertainty. A neural temporal
model should be considered only if simple models fail after an adequately
powered, correctly defined dataset exists.

## Immediate implementation slice

Do not generate more seeds before completing these steps:

1. Identify and freeze the exact decentralized damping checkpoint.
2. Record its SHA-256, training configuration, action space, SUMO version,
   network/route hashes, step length, and CAV-selection behavior.
3. Add `policies/counterfactual_escalation.py` with explicit commands:
   - `DAMPING_ONLY`
   - `DAMPING_PLUS_WALL`
4. Extract the useful overlay behavior from `WallOverChampionPolicy` without
   retaining its internal health trigger.
5. Guarantee that action 0 continues damping and never calls the supervisor's
   release-to-SUMO behavior.
6. Add tests for:
   - off-path action identity with the frozen damping policy;
   - in-band wall override only during the pulse;
   - out-of-band damping identity across both arms;
   - deterministic formation veto;
   - release back to damping;
   - safety abort and cooldown.
7. Refactor `counterfactual/twin_replay.py` to accept a base-policy factory and
   an intervention schedule.
8. Pass damping-vs-damping replay equality, treatment-prefix equality, and
   repeatability checks on at least two seeds.
9. Recalibrate outcomes and safety budgets under damping-only trajectories.
10. Run a corrected 12–20-seed headroom pilot before any large dataset.

## Headroom decision rule

Proceed to paper-scale data generation only if:

- the safety-first oracle selects both damping-only and damping+wall;
- safe beneficial wall cases span several independent seeds;
- the oracle materially improves on damping-only;
- always-wall is worse than the oracle in traffic, safety, or intervention duty;
- treatment success/failure varies with observable pre-treatment conditions.

If the oracle nearly always selects damping, redesign the treatment or formation
mechanism. If it nearly always selects the wall, an always-wall policy is the
honest answer and the gate is not a strong contribution.

## Recommended paper-scale experiment

Primary matrix:

- demand: dense and hyper;
- CAV penetration: 10%, 20%, and 30%;
- HDV model: frozen Krauss mixture;
- development: 20 independent seeds per demand/penetration cell;
- three or four predeclared decision windows per seed;
- locked test: 10 new seeds per cell;
- transfer: at least 20 untouched IDM groups at the primary penetration.

All rows from one complete group
`(demand, penetration, HDV model, traffic seed, CAV-selection seed)` must remain
in the same split. Never split by snapshot or decision row.

Before launching the matrix, benchmark ten branches and report measured runtime,
disk use, and safe parallelism. The 8/10 version was estimated as approximately
10–14 focused working days end to end, although the corrected headroom result
should be available much earlier.

## Required locked-test baselines

1. No CAV control.
2. Frozen decentralized damping.
3. Always-wall over damping when formation is feasible.
4. Tuned health/hysteresis gate over damping.
5. Congestion-risk prediction gate.
6. Learned counterfactual treatment-effect gate.
7. Paired-outcome oracle as a non-deployable upper bound.

The final policy must be tested in complete closed-loop episodes. Offline AUC or
classification accuracy is not enough.

## Paper-level success criteria

Freeze the criteria before opening the locked set:

- no excess collisions or teleports relative to damping;
- hard-braking and TTC exposure stay within frozen safety noninferiority budgets;
- delay or queue improvement over damping has a seed-clustered confidence
  interval excluding zero, with discharge noninferior;
- the counterfactual gate beats both the health gate and congestion-risk gate;
- intervention duty is lower than always-wall;
- the gate captures a meaningful portion of paired-oracle improvement;
- benefit direction is reasonably consistent across density/penetration cells;
- failures and negative results are reported without test-set retuning.

## Paper authorship and claim discipline

The LaTeX paper is intentionally an author worksheet. The authors should write
and verify the substantive prose, citations, and numerical claims. Assistance
can be used to organize, critique, and correct author-written material, but the
paper must not present unverified generated prose or results.

Do not write any of these claims yet:

- "the counterfactual gate works";
- "the method improves safety and efficiency";
- "the system predicts metastable collapse";
- "the approach generalizes to IDM or other geometries";
- "this is the first counterfactual traffic controller."

Those statements require the corrected paired design and locked prospective
evidence. The defensible current claim is that the existing health signal
saturates in hyper density and motivates learning the incremental value of
escalating from damping to coordinated wall control.

## Repository safety notes

- The worktree is heavily modified and contains many untracked experiment
  artifacts. Treat all existing changes as user work.
- Do not run destructive Git commands or discard/rewrite unrelated files.
- Do not overwrite `NEW_AGENT_HANDOFF.md`; this handoff is intentionally
  separate.
- New dense/hyper runs belong under `runs/rllib`, `runs/probe`, or `runs/eval`
  according to `AGENTS.md`.
- Verification command:

```powershell
python -m pytest -q
```

## One-sentence next action

Freeze the damping checkpoint, implement and test the explicit
`damping-only`/`damping-plus-wall` overlay, and prove paired replay validity
before running another training dataset.
