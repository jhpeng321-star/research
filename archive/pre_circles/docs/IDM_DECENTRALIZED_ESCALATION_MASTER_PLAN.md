# IDM-Native Decentralized Damping and Counterfactual Escalation

## Master implementation and paper plan

**Workspace:** `D:\research-dense`

**Working paper title:**

> Counterfactual Escalation from Decentralized CAV Damping to Coordinated
> Moving-Wall Control

**Purpose:** This is the controlling handoff for pivoting the project from the
Krauss-trained discrete controller to an IDM-native decentralized damping
controller, then rebuilding and prospectively validating the moving-wall
counterfactual escalation predictor. It supersedes plans to train a predictor
on the presently unverified treatment/data contract. Existing handoffs remain
important historical evidence and must be read before changes are made.

---

## 1. Required reading and workspace rules

Read completely before acting:

1. `AGENTS.md`
2. `docs/COUNTERFACTUAL_ESCALATION_HANDOFF.md`
3. `docs/COUNTERFACTUAL_ESCALATION_V2_PLAN.md`
4. `docs/PREDICTOR_DIAGNOSIS_HANDOFF.md`
5. `docs/CURRENT_PREDICTOR_LIMITATIONS_HANDOFF.md`
6. This document

Rules:

- Preserve the dirty worktree and every existing user change.
- Do not resume medium optimization. This workspace is for dense/hyper and the
  IDM/counterfactual pivot described here.
- Do not reuse old Krauss paired rows as primary evidence for the new IDM
  causal contract.
- Do not train another predictor until the decentralized actuator and wall
  treatment pass their gates.
- Do not generate paper-scale data until replay, branch identity, treatment
  dose, safety, and outcome-diversity gates pass.
- Record every development seed inspected. Exclude those seeds from final
  prospective evaluation.
- Run focused tests after each implementation slice and `python -m pytest -q`
  at the end of every major phase.
- Do not claim successful prediction, calibration, collapse prevention, or
  real-world generalization before prospective evidence exists.
- Before a design change that alters the scientific definition of
  decentralized damping or the moving wall, stop and discuss it with the user.

### Current execution status (2026-08-29)

Update 2026-08-30: the frozen fixed-P30 limited-domain rule completed its
prospective Path-A acceptance on seeds 4201--4220 and failed. Over 120
opportunities it activated 58 times (12 helpful, 13 harmful, 33 neutral), had
slightly positive mean physical effect (+0.00382) but negative
constraint-aware policy value (-0.04325), three merge-discharge failures, and
negative leave-one-seed-out value. It is not promoted as a validated damping
controller. See `IDM_P30_RULE_ARM_PROSPECTIVE_ACCEPTANCE_RESULTS.md`.

The wall audit found and repaired a physical release defect: wall-only tau
changes are now restored for removed, handed-off, aborted and final members
without releasing their speed control to uncontrolled SUMO. The analytical
`disturbance_p30` controller is frozen as the identical control arm. A
treatment-blind p10--p30 scan selected 30% as the lowest compact full-lane
placement level. The corrected 12-seed v5.2 paired pilot then passed replay,
causal identity, dose and outcome-diversity gates, but failed runtime safety:
only 5/10 dose-qualified treatments were admissible versus the frozen 90%
requirement. Predictor training remains unauthorized pending an approved,
versioned response to the post-release hard-braking mechanism. A read-only
release-aligned diagnosis subsequently found that the harm is not localized at
release: across the five unsafe completed cases, one positive excess onset
occurred in the first 20 seconds after release and 248 occurred more than 60
seconds later, predominantly on the downstream merge edge. Terminal wall
mechanics do not separate safe from unsafe cases. A taper-only safety repair is
therefore rejected. On 2026-08-30 the owner adopted the two-stage joint
safety-and-benefit selective-escalation estimand and froze v5.3 as a treatment-
integrity version only (bounded same-lane handoff, compactness enforcement and
a fixed in-horizon linear taper). The prospective 90% safety requirement now
applies to gate-selected activations; no fitted gate or performance claim yet
exists. See
`IDM_COMPACT_WALL_FEASIBILITY_SCAN_RESULTS.md` and
`IDM_WALL_V5_2_P30_PILOT_RESULTS.md`,
`IDM_WALL_V5_2_RELEASE_SAFETY_DIAGNOSIS.md`, and
`IDM_TWO_STAGE_JOINT_SAFETY_BENEFIT_GATE_PROTOCOL.md`.

A subsequent treatment-blind observation audit found that the current feature
contract has usable upstream history and preflight geometry but no explicit
post-merge state for the edge where delayed harm concentrates. Gate-B outcome
diversity reaches the minimum diagnostic threshold (three joint-positive and
seven joint-negative seeds), while Gate F has only two failed-dose seeds. No
model has been fitted. The completed Phase-A replay passes identity,
no-partial-wall, and damping-return checks but fails v5.3 safety
non-worsening and safe-beneficial retention. Observation v2 passes exact-prefix
recapture and descriptive headroom checks, but it cannot override that
treatment no-go. Predictor-scale generation, fitting, calibration and
prospective evaluation remain unauthorized. See
`IDM_WALL_V5_2_DECISION_OBSERVATION_AUDIT.md`,
`IDM_TWO_STAGE_JOINT_SAFETY_BENEFIT_GATE_PROTOCOL.md`, and
`IDM_OBSERVATION_V2_SCREEN_RESULTS.md`.

The projected native-IDM-to-P30 blend is implemented and its frozen 15-case
development screen is complete. Replay, predecision, target and exact P30
endpoint identity passed. Intermediate strengths were command-distinct in most
cases, resolving the earlier projection-collapse diagnosis. Authority still
failed: one 0.25 treatment aborted after 15/20 commands, two branches violated
the literal executed-jerk gate when front-safe follow speed dominated, only two
cases across two seeds gained more than the replay deadband over both endpoints,
and no harmful P30 case became helpful. The contingent observation/reward pool
was not generated and RL remains unauthorized. See
`IDM_P30_PROJECTED_BLEND_AUTHORITY_RESULTS.md`. A materially different
closed-loop bilateral law would change the scientific treatment definition and
must be discussed before implementation.

The simpler binary abstain-versus-P30 reward has now been separately repaired.
The fixed 60-second paired option reward reproduced all 90 stored effects and
constraints exactly, achieved 100% nonneutral sign and substantial-pair
agreement, Pearson correlation 0.99944 and Spearman correlation 1.0, and passed
the component-balance and hard-failure gates. It authorizes implementation of a
paired contextual-bandit/semi-Markov wrapper only. The ordinary variable-
lifetime multi-agent reward and RL training remain unauthorized until that
wrapper and its treatment-independent actor observation are verified. See
`IDM_BINARY_OPTION_REWARD_RESULTS.md`.

---

## 2. Scientific pivot and final architecture

The frozen Krauss damping checkpoint uses the pruned actions `coast`,
`recover`, and `guard`. It is a merge-assistance/safety controller, not a
purpose-built IDM wave damper. The optional legacy `wave_damp` action makes
many CAVs chase a short-horizon spatial speed mean; under string-unstable IDM,
this can create multiple moving perturbations. Raw continuous acceleration is
also inadequate because it directly commands the next speed without an
explicit nominal controller, jerk projection, follower constraint, or
coordination rule.

The target architecture is:

```text
IDM mixed traffic
        |
        v
IDM-native decentralized residual damping
  - local analytical car-following/smoothing baseline
  - one selected absorber per lane/influence region
  - bounded learned residual
  - deterministic safety/jerk/follower projection
        |
        v
Two-stage counterfactual escalation gate
  - Stage 1: can a safe, adequate wall dose be delivered?
  - Stage 2: conditional on feasibility, would it improve safety/service?
        |
        v
Temporary coordinated moving wall
        |
        v
Smooth release to the identical frozen IDM damping controller
```

The decentralized controller is the always-available, low-authority control
arm. The wall is temporary, coordinated, high-authority treatment. The paper's
central question is when escalation from the first to the second is warranted.

---

## 3. Known code and evidence issues that must shape the work

### 3.1 Optional observation suffixes are assembled but not encoded

`ObservationAssembler` appends optional meter load, CAV count, wave signal,
health, injection velocity difference, and runway margin after the fixed ego,
zone, and neighbor block. `FeatureEncoder.forward_components()` slices only
the fixed blocks. Consequently, enabled suffixes can be present in the raw
vector but absent from actor and critic embeddings. Some still affect external
masks or rewards, but they are not necessarily learned inputs.

Repair this before interpreting new training.

### 3.2 Prior IDM training does not establish actuator failure

Earlier IDM runs had incomplete episodes/NaN summaries, very little training,
process failure, or incompatible checkpoint observation/action spaces. They do
not constitute a clean rejection of an IDM-native learned damper.

### 3.3 Existing IDM results delimit control authority

Current evidence shows a stochastic transition around 2200 veh/h and nearly or
fully structural collapse at higher demands. Decentralized damping should be
tested primarily below and near the transition. It should not be required to
prevent collapse at 2400--2700 veh/h. Those regimes motivate escalation and
stress/service evaluation.

### 3.4 Critic truncation and training bookkeeping

The centralized critic truncates peers above `max_cavs_critic=32`, and this has
already occurred in high-density IDM runs. Resolve or explicitly avoid that
regime during controller training. Short smoke runs must also demonstrate
completed episodes, finite rewards/losses, meaningful environment/agent step
counts, and checkpoint restoration before long training.

---

## 4. Phase 0: freeze the IDM experimental contract

**Estimated human time:** 2--4 hours.

Create `docs/IDM_DECENTRALIZED_CONTROL_CONTRACT.md` recording:

- SUMO configuration and version;
- exact IDM and lane-changing parameters for every vehicle type;
- CAV type conversion/selection behavior;
- SUMO step and policy interval;
- CAV penetration and placement method;
- controller observation/action timing;
- development, validation, and final untouched seed partitions;
- density/demand cells and their scientific role;
- collapse definition;
- wave, safety, service, and control metrics;
- moving-wall definition and outcome horizon;
- replay/numerical tolerance.

Recommended demand roles:

- 1900--2100 veh/h: lower/near-critical decentralized damping evaluation;
- 2200 veh/h: stochastic transition and primary headroom cell;
- 2400 veh/h: escalation/service-management cell;
- 2700 veh/h: structural-overload stress test, not primary decentralized
  collapse-prevention evidence.

Reproduce no-control IDM behavior on a small seed sample. The phase passes only
if paired resets are reproducible and demand roles remain empirically valid.

---

## 5. Phase 1: repair observation consumption

**Estimated human time:** 4--8 hours.

Likely files:

- `env/features.py`
- `models.py`
- `env/core.py`
- `train.py`
- model/observation unit tests

Add an explicit context-feature branch:

```text
ego block       -> ego MLP
neighbor set    -> shared neighbor MLP + masked/trust pooling
zone block      -> zone MLP
optional context-> context MLP
                   |
                   v
             concatenated embedding
```

The feature schema and optional-field order must have a single source of truth.
Actor and critic must receive the same intended raw physical fields. Action
mask bits must remain separate from the physical embedding.

Required tests:

1. Raw observation dimensions match the declared schema.
2. Changing each enabled suffix changes the encoded output.
3. Every enabled suffix receives a nonzero gradient.
4. Actor and centralized critic consume the context branch.
5. Action-mask fields are not embedded as traffic features.
6. NaN/Inf sanitization remains correct.
7. Legacy checkpoints either load correctly or fail with a clear compatibility
   message rather than silently shifting fields.

Gate: do not proceed to learning until every enabled physical field is proven
to affect the intended representation.

---

## 6. Phase 2: build the analytical IDM damping actuator

**Estimated human time:** 1--2 working days.

Do not begin with RL. Implement and validate a deterministic actuator first.

### 6.1 New control mode

Add a mode with a distinct name such as `idm_residual_smoothing`. Avoid altering
the meaning of existing checkpoint modes.

The command is:

```text
a_cmd = safety_project(a_nominal + absorber_gate * delta_a_rl)
```

Zero residual must exactly reproduce a useful analytical controller.

### 6.2 Nominal controller

Use ego speed/acceleration, leader gap/speed/relative speed, follower state,
desired headway, and a slowly filtered downstream reference. A starting form
is:

```text
a_nominal = kv * (v_ref - v)
          + ks * (gap - desired_gap)
          + kd * (v_lead - v)

desired_gap = s0 + T * v
```

The downstream reference must be low-pass filtered, outlier-robust, bounded,
and slowly varying. Do not command every CAV to a raw instantaneous mean. Test
a physically justified percentile/planner reference and log its source,
distance, age, and filter state.

### 6.3 Deterministic absorber selection

Initially learn only control magnitude, not who acts. Implement deterministic
selection:

1. Detect a disturbance and its influence region.
2. Enumerate CAVs with enough upstream runway and acceptable local/follower
   state.
3. Select the most useful upstream feasible absorber.
4. Permit at most one primary absorber per lane or non-overlapping influence
   region.
5. Enforce minimum spacing between active absorbers.
6. Latch membership for a minimum hold period.
7. Handoff only when the member exits, becomes unsafe, or loses authority.
8. Keep nonselected CAVs in identical IDM behavior or a separately declared,
   extremely gentle support law.

Log selections, reasons, runway, disturbance location, hold time, handoffs, and
simultaneous active count.

### 6.4 Learned residual interface

Start with one scalar:

```text
delta_a_rl in [-0.5, +0.3] m/s^2
```

The exact bounds must be frozen after the analytical pilot, not optimized on
final seeds. Acceleration may be converted to a desired speed before the safety
layer, but the mapping and controller period must be explicit.

### 6.5 Deterministic projection

After combining nominal and residual commands, enforce:

- physical and comfortable acceleration/deceleration bounds;
- jerk bound relative to the prior applied command;
- secure-gap and safe-following-speed constraint;
- TTC floor/emergency behavior;
- follower-aware braking limit so a close follower does not inherit a shock;
- speed-limit and SUMO car-following safety checks;
- smooth release to ordinary IDM.

Review the exact SUMO `speedMode` bits. Preserve the car-following safe-speed,
acceleration, and deceleration checks intentionally; do not rely on a magic
integer without documenting its semantics.

Required tests:

- zero residual reproduces the analytical law;
- inactive control returns to IDM;
- only selected absorbers receive damping overrides;
- nonselected/out-of-region CAV actions remain identical;
- jerk, acceleration, gap, TTC, and follower bounds hold;
- latching/handoff are deterministic;
- controller/filter state resets between episodes;
- release has no acceleration or desired-speed discontinuity;
- command and projection diagnostics match actual TraCI commands.

---

## 7. Phase 3: six-seed analytical actuator pilot

**Estimated compute:** 3--8 hours.
**Estimated analysis/iteration:** about one working day.

Use declared development seeds only. Compare:

1. uncontrolled/no-op IDM CAVs;
2. the current pruned checkpoint;
3. analytical selected-absorber IDM damping;
4. the same analytical controller through the new RL interface with an exact
   zero residual;
5. optional all-CAV damping ablation.

Test 1900, 2000, 2100, and 2200 veh/h unless Phase 0 moves the measured
transition.

Required metrics:

**Wave:** upstream/downstream speed-variance amplification, oscillation energy,
low-speed-wave duration, minimum speed, propagation duration, vehicles exposed.

**Safety:** collisions, TTC count/minimum, all-vehicle severe/hard braking,
integrated negative acceleration, jerk, and follower response after absorber
commands. Do not substitute a policy-wrapper hard-brake counter for an
all-vehicle trajectory audit.

**Service:** time loss, completed/unfinished vehicles, discharge, throughput,
and spillback.

**Control:** controlled CAV-seconds, absorber count, simultaneous absorbers,
control magnitude, clipping/projection frequency, and handoffs.

Frozen pilot gate at at least one transition-band cell:

- about 10--15% reduction in wave energy or severe-braking exposure;
- benefit in at least four of six development seeds;
- zero collisions;
- no material TTC or follower-braking deterioration;
- no more than about 0.5--1.0 s/vehicle added delay;
- no material discharge loss;
- effects exceed paired replay/numerical noise.

If the actuator fails, diagnose reference construction, preview distance,
absorber geometry, simultaneous actions, and follower projection. Do not train
RL and do not add seeds merely to search for significance. If all-CAV control
fails but selected-absorber control passes, preserve that as an important
mechanism ablation.

---

## 8. Phase 4: train only the bounded residual

**Estimated preparation:** one working day.
**Estimated compute per serious campaign:** 8--24 hours.

Only begin after Phase 3 passes.

Keep absorber selection deterministic in the first campaign. Initialize the
residual mean at zero with narrow exploration. Use a zero-residual analytical
controller as the meaningful default rather than starting from random direct
acceleration.

Train mainly at 1900--2200 veh/h with controlled randomization over:

- IDM parameters;
- CAV penetration and placement;
- demand perturbations;
- declared lane-changing variations.

Do not primarily train at 2400--2700 and expect decentralized collapse
prevention.

Replace the Krauss fairness-heavy objective with an IDM damping objective whose
priority is:

1. collisions/TTC/severe braking;
2. wave and braking attenuation;
3. service-budget compliance;
4. smooth, sparse control.

An initial structure is:

```text
R = -w_wave * wave_energy
    -w_brake * severe_braking_exposure
    -w_ttc * ttc_events
    -w_jerk * jerk
    -w_control * control_effort
    -w_delay * max(0, delay - delay_budget)
    +w_flow * discharge
```

Do not use delayed binary collapse prevention as the main dense per-step
reward. Avoid duplicating a large global penalty independently for every agent
without checking scaling and credit assignment.

Before long training, a 1--3 iteration smoke run must show:

- completed episodes;
- finite rewards, values, losses, and gradients;
- correct environment and agent step counts;
- residual distribution near zero at initialization;
- no exploration-induced gridlock;
- no critic truncation in the training cells;
- successful checkpoint save and restoration.

---

## 9. Phase 5: select and freeze the IDM decentralized checkpoint

**Estimated time:** 1--2 working days including evaluation.

Select using untouched validation seeds and a predeclared multi-objective
criterion. Do not use final prospective seeds. Required evidence:

- repeatable wave or severe-braking reduction;
- safety admissibility;
- service noninferiority/budget compliance;
- reasonable cross-seed consistency;
- abstention or near-zero residual in healthy traffic.

Freeze and record:

- checkpoint path and hash;
- code revision/diff identity;
- feature schema and normalization;
- action bounds and safety parameters;
- IDM/demand/lane-change configuration;
- seed partitions.

Run damping-versus-damping paired replay identity. The new checkpoint becomes
the control arm for every subsequent paired experiment. Old Krauss treatment
rows cannot be pooled into this final contract.

---

## 10. Phase 6: revalidate the moving-wall overlay

**Estimated time:** 1--2 working days plus 4--10 compute hours.

Control arm:

> Frozen IDM residual damping.

Treatment arm:

> The identical frozen IDM residual damping plus a temporary coordinated
> moving wall.

Required invariants:

1. Damping-versus-damping replay identity passes.
2. Control and treatment branches are identical before the decision.
3. Only treatment members in the declared band receive wall overrides.
4. Out-of-band/nonmember CAVs retain identical damping actions.
5. Released members return smoothly to the identical damping controller.
6. Requested and realized treatment dose are recorded.

Record requested/realized duration, controlled CAV-seconds, lane coverage,
membership changes, handoffs, abort reasons, speed/tau deltas relative to the
damping counterfactual, and safety projection frequency.

Run only a corrected 12--20-seed headroom pilot. Pass criteria:

- at least 80% of feasible activations receive at least 80% of planned
  duration;
- at least 80% lane coverage;
- at least 90% safety-admissible completed treatments;
- beneficial completed treatments in at least three independent seeds;
- nonbeneficial completed treatments in at least three independent seeds;
- treatment effects exceed replay/numerical noise;
- collapse-prevention diversity if collapse prevention remains a primary
  paper target.

If dose fails, repair geometry, membership, or handoff. If dose passes but
outcomes do not change, redesign the treatment rather than adding more seeds.
If the wall improves safety/service without preventing collapse, discuss an
honest outcome pivot before changing the primary target.

---

## 11. Phase 7: two-stage counterfactual escalation predictor

**Estimated time after valid data exist:** 2--4 working days.

Start with regularized logistic/linear models and gradient-boosted trees. Add a
neural network only if dataset size and grouped validation show a reason. A
large network is not automatically more appropriate for a modest paired
simulation dataset.

### Stage 1: feasibility and dose

Estimate:

```text
P(safe adequate wall dose | predecision X)
```

A frozen label can be:

```text
feasible_completed =
    realized_duration >= 0.8 * requested_duration
    and lane_coverage >= 0.8
    and safety_admissible
```

Candidate predecision features include eligible CAV count, lane coverage,
CAV spacing, arrival-time spread, control runway, follower gaps, active lanes,
expected handoffs, penetration, local density/speed, disturbance position, and
wall geometry. Never include postdecision dose or outcome information.

### Stage 2: conditional treatment effects

Train only/evaluate explicitly on feasible completed treatments. Use paired
branches with identical predecision state:

```text
delta_metric = metric(wall + damping) - metric(damping)
```

Predict separate outcomes rather than only the legacy composite label:

- severe-braking effect;
- TTC effect;
- collapse-status effect;
- delay effect;
- discharge effect;
- control/treatment cost.

The operational gate should be constrained:

```text
activate if
    predicted feasibility >= threshold
    and predicted safety benefit exceeds its minimum
    and predicted delay cost is within budget
    and predicted discharge loss is within budget
```

Prefer a conservative lower-confidence-bound rule when uncertainty is high.

Freeze the beneficial-treatment definition before paper-scale generation. A
reasonable starting contract is:

```text
no collision
and severe-braking benefit exceeds a minimum meaningful effect
and TTC is noninferior
and delay increase <= max(2 s/vehicle, 10%)
and discharge loss <= 2--3%
```

Collapse prevention is an additional high-value channel, not a license to hide
unbounded service cost. If it remains the primary paper claim, require explicit
collapse-prevention headroom/diversity in the pilot and final test.

Split data by scenario-seed group, never by individual arm rows. Report AUROC,
average precision, Brier score, calibration, threshold confusion matrix,
missed-benefit and false-escalation rates, constraint violations, and most
importantly prospective policy value.

Compare against:

- never escalate;
- always wall;
- health-threshold escalation;
- feasibility-only gate;
- retrospective oracle clearly labeled as an unattainable upper bound.

The predictor is verified only if the frozen gate prospectively outperforms the
deployable baselines on untouched seeds while satisfying safety/service
constraints. Training-set or cross-validation signal alone is insufficient.

---

## 12. Phase 8: paper-scale prospective evaluation

**Estimated compute/analysis:** 2--5 days.

Only start after the controller, treatment, and predictor gates pass. Use fully
untouched seeds and the frozen code/configuration.

Primary arms:

1. no control;
2. frozen IDM decentralized damping;
3. always/eligible wall plus identical damping;
4. health-threshold escalation;
5. learned two-stage escalation;
6. retrospective oracle upper bound.

Evaluate transition, moderate-overload, and structural-overload cells; CAV
penetration; and prespecified IDM parameter variants. Krauss can be a separate
model-form/transfer sensitivity test but not pooled with primary IDM evidence.

Use seed-level paired effects, paired bootstrap confidence intervals,
appropriate paired tests, effect distributions, calibration uncertainty,
policy value, and documented failure/abstention cases.

---

## 13. Phase 9: paper framing and required ablations

**Estimated first complete draft:** 2--4 working days once results are frozen.

Proposed central claim:

> Low-authority IDM-compatible decentralized control attenuates disturbances
> when local control authority is sufficient, but cannot reliably manage every
> metastable or overcritical realization. A counterfactual two-stage gate
> therefore estimates whether a temporary coordinated moving wall is both
> deliverable and beneficial.

Candidate contributions:

1. IDM-native decentralized residual damping with selected absorbers and a
   deterministic safety projection.
2. A damping-preserving temporary moving-wall overlay with measured treatment
   dose and branch identity.
3. A two-stage gate separating physical treatment feasibility from conditional
   safety/service benefit.
4. Prospective paired comparison against never-escalate, always-escalate, and
   health-threshold baselines.
5. Empirical characterization of the authority boundary between decentralized
   damping and coordinated inflow control.

Required ablations:

- all-CAV versus selected-absorber damping;
- raw acceleration versus residual action;
- with/without follower-aware safety projection;
- with/without downstream context;
- health-only versus two-stage gate;
- feasibility-only versus benefit-aware gate;
- damping versus damping-plus-wall.

Claims prohibited without evidence:

- general collapse prevention;
- calibrated individual treatment effects;
- real-world safety/generalization;
- IDM being inherently more realistic based only on these simulations;
- successful prediction based only on training or retrospective cross-validation;
- safety improvement based only on the wrapper hard-brake counter.

---

## 14. Time and decision schedule

### Fastest credible technical checkpoint

Within about 2--3 working days:

- freeze the IDM contract;
- repair observation consumption;
- implement/test the analytical residual actuator;
- run a six-seed development pilot;
- decide whether the actuator has a repeatable effect worth training.

This is a progress deliverable, not a verified predictor.

### Optimistic end-to-end schedule

- Contract and encoder repair: 1 day
- Analytical actuator and tests: 1--2 days
- Pilot and one tuning iteration: 1 day
- RL setup/training: 1--2 days
- Fresh controller validation/freeze: 1 day
- Wall overlay revalidation: 1--2 days
- Initial paired data and predictor: 2--4 days
- Prospective evaluation and draft: 3--5 days

**Optimistic total:** about 10--15 working days.

**More realistic paper-ready total:** about 2--3 weeks.

### Approximate compute time

- analytical pilot: 3--8 hours;
- one residual RL campaign: 8--24 hours;
- checkpoint comparison: 4--10 hours;
- wall-dose pilot: 4--10 hours;
- initial paired predictor dataset: 12--36 hours;
- final prospective experiment: 1--4 days.

If the analytical actuator fails, budget an additional 2--5 working days for
reference-speed, preview-distance, absorber-selection, and safety redesign.

---

## 15. Immediate execution order for the next agent

1. Read all required documents and inspect the current dirty worktree.
2. Report, before editing:
   - exact active IDM configuration;
   - exact observation suffix/encoder mismatch;
   - proposed controller equation and action semantics;
   - absorber selection rule;
   - safety projection;
   - files to change;
   - six-seed pilot command/runtime;
   - frozen pass/fail criteria.
3. Stop for discussion if any choice materially changes the scientific
   decentralized-treatment definition.
4. Otherwise implement only Phases 0--2 with focused tests and full pytest.
5. Report code/test results and the exact pilot command before launching the
   pilot.
6. Run only the six-seed analytical pilot, analyze it, and apply the Phase 3
   gate.
7. Do not automatically continue to RL if the actuator gate fails.

The decisive near-term question is not whether a neural predictor can be fit.
It is whether a selected-absorber, safety-projected IDM actuator produces a
repeatable physical effect without unacceptable safety or service cost. Only a
positive answer supports the remaining predictor and paper program.
