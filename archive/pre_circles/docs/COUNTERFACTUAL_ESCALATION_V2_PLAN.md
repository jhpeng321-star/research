# Counterfactual Escalation v2 — IEEE IV Implementation Plan

Status: implementation plan for the paper target
`Counterfactual Escalation from Decentralized CAV Damping to Coordinated
Moving-Wall Control`.

## 1. Paper-level objective

Build and prospectively evaluate an RSU policy that observes only information
available at decision time, keeps a frozen decentralized CAV damping policy as
the default controller, and escalates to a coordinated moving wall only when
the predicted incremental traffic benefit is sufficiently positive, formation
is feasible, and safety risk is within a frozen budget.

The paper claim is not that traffic collapse can be predicted exactly. The
claim is that the **conditional value of a specific intervention** can be
estimated well enough to avoid harmful or unnecessary moving-wall activations.

The final scientific sentence should be defensible in this form:

> Given an RSU-observed traffic history, the proposed safety-constrained gate
> estimates the simulator-grounded incremental value of adding a temporary
> coordinated moving wall to a frozen decentralized damping controller and
> escalates only when benefit, uncertainty, safety, and formation criteria
> pass.

## 2. What already exists and what must change

Reusable foundations already present in the repository:

- deterministic replay from time zero in `counterfactual/twin_replay.py`;
- pre-treatment traffic, placement, and local-shock features in
  `counterfactual/pilot.py`;
- health-independent traffic and safety outcomes in
  `counterfactual/outcomes.py`;
- oracle/readiness analysis in `counterfactual/oracle.py` and
  `counterfactual/readiness.py`;
- grouped ridge/logistic evaluation in `counterfactual/gate_model.py`;
- a preliminary continuous traffic-effect and safety-risk trainer in
  `experiments/counterfactual/train_counterfactual_two_head_gate.py`;
- a working wall-over-damping composition pattern in
  `policies/wall_over_champion.py`.

The critical architectural defect is that the current paired generator uses
`WallSupervisorEnv`: action 0 releases CAVs to SUMO. It therefore estimates
`wall versus no wall`, not `damping plus wall versus damping`. The existing
30-case dataset is valid for its old question and must remain archived, but it
cannot be pooled with v2.

## 3. Frozen causal contract

For a complete pre-treatment history `H_t`, define two potential outcomes with
the same SUMO seed, CAV-selection seed, vehicle realization, damping checkpoint,
and action history:

```text
A0: frozen decentralized damping for the complete horizon
A1: the same frozen damping + one frozen moving-wall pulse
```

Outside the wall band, A1 must continue to use exactly the damping actions from
A0. Inside the band, the wall overlay temporarily has priority. When the pulse
ends, every CAV returns to the frozen damping policy—not uncontrolled SUMO.

The treatment is a complete protocol, not merely the word "wall":

- wall geometry and target band;
- speed/headway commands;
- formation feasibility mask;
- pulse duration;
- minimum dwell and cooldown;
- runtime safety-abort rule;
- post-pulse return to damping.

All of these must be frozen before paper-scale paired data are generated.

For metric `k`, use a benefit-positive convention:

```text
tau_k(H_t) = Y_k(A0 | H_t) - Y_k(A1 | H_t)
```

for costs such as delay and queue area, and reverse the subtraction for
benefits such as discharge. The paper must call these **simulator-grounded
counterfactual effects**, not real-road causal effects.

## 4. Target runtime architecture

```text
RSU traffic history H_t
        |
        v
eligibility + formation feasibility mask
        |
        v
traffic-effect heads + safety-risk heads + uncertainty intervals
        |
        v
safety-constrained escalation rule
        |                         |
        | reject                  | accept
        v                         v
frozen decentralized       frozen decentralized damping
CAV damping                + temporary coordinated wall
```

The health factor remains one candidate feature and the existing health gate
remains a required baseline. Health is not the label and cannot directly define
the primary outcome.

## 5. Implementation phases and stage gates

### Phase 0 — Freeze provenance and primary controller

Tasks:

1. Select one frozen damping checkpoint and record its path, SHA-256, training
   configuration, action space, SUMO version, network files, route files, step
   length, and CAV-selection logic.
2. Freeze the primary HDV model as the route-file Krauss mixture. Treat the IDM
   configuration as transfer evaluation, not pooled primary training.
3. Assign non-overlapping seed blocks for calibration, development, model
   selection, locked testing, and transfer testing.
4. Write a v2 experiment manifest and dataset schema version.

Exit gate: the same command can reproduce a checksum-identical configuration
manifest, and the selected damping checkpoint is never tuned again using v2
test results.

### Phase 1 — Implement damping-preserving wall overlay

Create `policies/counterfactual_escalation.py` by extracting the useful
composition behavior from `WallOverChampionPolicy`:

1. Always compute the frozen damping action for every active CAV.
2. When escalation is off, return all damping actions unchanged.
3. When escalation is on, apply the wall only to feasible in-band CAVs and
   return damping actions for all out-of-band CAVs.
4. After release, resume damping for every CAV and restore cached wall-specific
   headway state without erasing the damping controller's state.
5. Expose explicit commands `DAMPING_ONLY` and `DAMPING_PLUS_WALL`; do not use
   the old health trigger inside the treatment implementation.

Required tests:

- escalation off is action-for-action identical to the frozen damping policy;
- in-band CAVs receive wall commands only during the pulse;
- out-of-band CAVs receive the same damping actions in both arms;
- release returns to damping rather than `release_to_sumo`;
- infeasible formation causes a deterministic veto;
- safety abort and cooldown behave identically in data generation and deployment.

Exit gate: all composition tests pass and a trajectory diff shows that the two
arms first diverge at the declared treatment step.

### Phase 2 — Build paired replay v2

Refactor `counterfactual/twin_replay.py` to accept a base-policy factory and an
intervention schedule rather than assuming `WallSupervisorEnv` actions.

Add metadata for:

- base checkpoint/hash;
- base policy actions and wall overrides;
- treatment start/end/abort reason;
- eligible and actually controlled CAV IDs;
- wall lane coverage and formation geometry;
- scenario, HDV model, penetration, all seeds, and executable hashes.

Acceptance tests:

1. damping versus damping replay matches at six-decimal state fingerprints;
2. damping+wall matches damping through the full pre-treatment prefix;
3. repeating one branch reproduces identical outcomes;
4. treatment mutation changes only the overlay command path;
5. no feature is computed from post-treatment state.

Exit gate: at least two seeds pass every twin/repeat test before any large run.

### Phase 3 — Recalibrate outcomes under the correct control arm

The old no-wall calibration is not automatically valid under damping. Re-run
calibration using frozen damping-only trajectories.

Freeze on calibration seeds only:

- sustained-congestion thresholds and persistence;
- queue-area and delay definitions;
- time-aligned merge discharge;
- hard-brake and low-TTC exposure definitions;
- collision and teleport handling;
- operational noninferiority margins;
- safety budgets and the runtime abort threshold.

Validate the congestion event against plotted trajectories on a separate
validation seed block before freezing it. Report controller steps in seconds.

Exit gate: labels distinguish meaningful future outcomes, safety metrics are
auditable at vehicle level, and thresholds are written to a read-only manifest.

### Phase 4 — Run a small headroom/heterogeneity screen

Before model training, generate paired A0/A1 outcomes on 12–20 fresh seeds with
three or four predeclared decision windows per seed.

Continue only if all are true:

- both damping-only and damping+wall are selected by the safety-first oracle;
- safe beneficial wall cases occur across several independent seeds rather
  than one lucky realization;
- always-wall is worse than the oracle on intervention burden or at least one
  traffic/safety metric;
- the oracle materially improves over damping-only;
- treatment failures can be explained by observable pre-treatment conditions.

If the oracle almost always chooses damping, redesign the wall protocol or
formation rule. Do not try a more complicated predictor. If the oracle almost
always chooses the wall, the correct controller is an always-wall policy and a
gate is not a meaningful contribution.

### Phase 5 — Generate the development dataset

Recommended core design:

- demands: dense and hyper only;
- CAV penetration: 10%, 20%, and 30%;
- primary HDV model: the frozen Krauss mixture;
- development: 20 independent seeds per demand/penetration cell;
- decision windows: three or four predeclared times or causal eligibility
  landmarks per seed;
- locked test: 10 new seeds per cell;
- transfer: at least 20 untouched IDM scenario-seed groups at the primary
  penetration, with no model retuning.

This yields approximately 360–480 development causal units from 120 independent
scenario-seed groups before any on-policy augmentation. All branches from one
history remain in the same split.

Every row must store both potential outcomes. Split by the complete group
`(demand, penetration, HDV model, traffic seed, CAV-selection seed)`, never by
snapshot or decision row.

Before launching the full matrix, benchmark ten branches and calculate the
measured wall-clock time, disk use, and safe parallel-worker count. Save each
completed causal unit atomically so interrupted runs resume without duplication.

### Phase 6 — Train an uncertainty-aware treatment-effect gate

Start with interpretable regularized models; do not begin with a neural network.
Use the existing continuous trainer as the base and predict separate outcomes:

Traffic-effect heads:

- sustained-congestion event reduction;
- queue-area reduction;
- delay reduction;
- additional merge discharge.

Safety-risk heads:

- hard-brake exposure under treatment;
- low-TTC exposure under treatment;
- collision/teleport risk when enough events exist; otherwise use a hard veto
  and report that rare-event risk cannot be learned from the available data.

Feature ablations:

1. health only;
2. aggregate traffic history;
3. traffic + placement/penetration;
4. traffic + placement + local shock/spacing history.

Estimate uncertainty with seed-grouped bootstrap ensembles or grouped conformal
calibration. Hyperparameters and activation margins may use only development
and validation groups.

The frozen gate should activate only if:

1. eligibility and formation feasibility pass;
2. the lower confidence bound shows at least one meaningful traffic benefit;
3. lower bounds satisfy all traffic noninferiority margins;
4. upper confidence bounds satisfy every safety budget;
5. no hard safety veto is active.

The final report should emphasize policy value, unsafe activation rate,
intervention precision, activation burden, and regret to the paired oracle—not
classification accuracy alone.

### Phase 7 — Wire the gate into closed-loop RSU control

Create a deployment wrapper, for example `policies/escalation_gate.py`, that:

- maintains the exact history buffer used by training;
- evaluates at a frozen decision interval;
- blocks decisions during an active pulse or cooldown;
- applies the same feasibility and safety shield used in the paired arms;
- logs every feature vector, prediction interval, veto, action, and reason;
- continues decentralized damping at every non-escalated step.

First run closed loop on development seeds. If the learned policy visits states
outside the original damping-only dataset, perform one prespecified on-policy
augmentation round on development seeds only: replay paired alternatives from
those visited histories, retrain once, and freeze the model. Never augment from
locked-test trajectories.

Exit gate: the online wrapper reproduces offline actions for identical feature
records and repeated runs are deterministic.

### Phase 8 — Locked prospective evaluation

Run complete episodes on untouched seeds for these policies:

1. no CAV control;
2. frozen decentralized damping;
3. always-wall over frozen damping when feasible;
4. tuned health/hysteresis gate over frozen damping;
5. congestion-risk gate that treats high predicted risk;
6. learned counterfactual escalation gate;
7. paired oracle as a non-deployable upper bound on sampled decision states.

Primary evidence must be computed at the independent scenario-seed level using
paired, cluster-aware bootstrap confidence intervals.

Predeclare paper-level success criteria before opening the locked set:

- zero excess collisions or teleports relative to damping;
- safety exposure remains within the frozen noninferiority budgets;
- delay or queue improvement over damping has a confidence interval excluding
  zero, with discharge noninferior;
- the counterfactual gate beats the health and congestion-risk gates on policy
  value or beneficial-intervention precision;
- intervention duty is lower than always-wall;
- the gate captures a meaningful fraction of the oracle improvement;
- the direction of benefit is consistent across density/penetration cells.

If a criterion fails, preserve the result and narrow the claim. Do not tune on
the locked set or replace the locked seed block.

### Phase 9 — Robustness, ablation, and failure analysis

Required analyses:

- remove health features;
- remove placement/penetration features;
- remove local shock features;
- remove uncertainty abstention;
- compare one-head binary labels with continuous traffic/safety heads;
- vary pulse duration only as a post-freeze sensitivity analysis;
- zero-shot IDM transfer;
- inspect false-positive activations and missed beneficial interventions;
- stratify by demand, penetration, wall lane coverage, and CAV distance/spread.

The paper should include at least one failure case where a wall is harmful and
the gate refuses it, and one rescue case where health alone is uninformative but
the treatment-effect gate escalates.

## 6. Paper evidence package

Minimum figures/tables:

1. Architecture: decentralized damping, RSU gate, and wall overlay.
2. Paired replay: identical history splitting into A0 and A1.
3. Oracle headroom: heterogeneity of treatment effects and safety admissibility.
4. Closed-loop policy comparison with seed-level confidence intervals.
5. Activation/abstention analysis against safety and operational benefit.
6. Feature ablation and transfer table.
7. Two trajectory case studies: beneficial escalation and correct refusal.

Every reported number must be generated from locked artifacts through one
reproducible analysis command. The current health-gated result belongs in the
motivation/baseline story; the current 30-case failed predictor belongs in the
development history or limitations, not as evidence that v2 works.

## 7. Realistic schedule

Assuming the existing foundation is reused and compute is available:

- 1 day: freeze contract, checkpoint, manifests, and treatment protocol;
- 1–2 days: damping-preserving overlay and unit/integration tests;
- 1 day: paired replay v2 and determinism validation;
- 1 day: damping-baseline outcome calibration and headroom screen;
- 1–3 days: measured dataset generation, depending on branch runtime and safe
  parallelism;
- 1 day: grouped models, uncertainty calibration, and offline evaluation;
- 1–2 days: closed-loop integration and development-only augmentation;
- 1–2 days: locked evaluation, ablations, transfer, and failure analysis;
- 2–3 days: figures, manuscript, and advisor review.

An honest 8/10 version is therefore roughly a 10–14 focused working-day effort,
not a few-hour training run. The first decisive milestone—the corrected paired
damping-versus-damping-plus-wall headroom screen—should be reachable much
earlier and determines whether the remaining investment is justified.

## 8. Immediate next implementation slice

Do these tasks before generating another seed:

1. Freeze and hash the exact decentralized damping checkpoint.
2. Add `CounterfactualEscalationPolicy` with explicit off/on commands.
3. Make action 0 mean **damping-only**, never release-to-SUMO.
4. Add composition tests for in-band override and out-of-band identity.
5. Extend twin replay metadata with base-policy and overlay fingerprints.
6. Pass damping-vs-damping and prefix-identity tests on two seeds.
7. Run a 12–20-seed corrected headroom pilot.

Only after step 7 should the paper-scale dataset or predictor training begin.
