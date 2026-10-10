# Documentation index

**Current owner priority, 2026-09-06: reuse an existing trained damping baseline
and focus the contribution on escalation.** Start with
[published-controller provenance and graph audit](PUBLISHED_DAMPING_REUSE_AUDIT_2026-09-06.md).
The coordinating task owns the single bounded development transfer check in
`runs/probe/published_idm_fit_check_2026-09-06/`; do not duplicate or extend its
simulations. The model is a frozen external RL candidate, not yet an accepted
baseline or an exactly identified paper checkpoint. No training or acceptance
data were used in the source/inference audit.

**Completed repair:** [Native-selectable damping v2.0](NATIVE_SELECTABLE_DAMPING_V2_PROTOCOL.md).
**Result:** [Action/reward repair, damping efficacy fails](NATIVE_SELECTABLE_DAMPING_V2_RESULTS.md).
The bounded check is finished: 20,400 charged global seconds, 18,000 completed,
no accepted checkpoint and no authorization to extend training. Native/reward
semantics are tested; a later negative-control audit rejects the old handback
speed-equality statistic, and a separately bounded paired unfinished-command
test now verifies physical release/re-engagement directly (20 actual simulated
seconds, 40 charged). Short pulse/recovery and corridor outcomes still fail.
The separate [v2.1 recovery prototype](NATIVE_DAMPING_RECOVERY_V21_RESULTS_2026-09-06.md)
completed its 480-second mechanism budget with no learning: rebound removed,
pulse effort 18.94% worse, NO-GO. Do not create another bespoke variant or
train it automatically. Both versions and all negative evidence are preserved.
The old campaign is stopped at verified 36/8/8 checkpoints. Do not resume its
serial or parallel coordinator. This repair has a separate 22,800-global-second
ceiling, explicit native release, bounded intervention cost, short SUMO learning
check and two exposed corridor diagnostic cases. No full campaign or acceptance
evaluation is authorized by that check. Historical scheduling text below is
superseded. Wall training remains paused.

**Historical implementation route, 2026-09-05 (stopped).** See
`TRAINED_DAMPING_TO_PAPER_ROUTE_2026-09-05.md`, then
`NATIVE_DAMPING_V1_IMPLEMENTATION.md` and `NATIVE_DAMPING_V1_LEARNING_CHECK.md`.
The stopped stage is `NATIVE_DAMPING_FULL_CAMPAIGN_2026-09-05.md`.
Its historical scheduling used three independent workers; see
`NATIVE_DAMPING_PARALLEL_EXECUTION_2026-09-05.md` and
`runs/rllib/native_damping_campaign_v1/parallel_execution/` for preserved records.
The earlier serial coordinator is stopped and must not be restarted.
A separate two-case magnitude diagnostic uses the remaining two development
rollout slots; see `NATIVE_DAMPING_MAGNITUDE_ABLATION_PROTOCOL_2026-09-05.md`.
Its completed findings are in `NATIVE_DAMPING_MAGNITUDE_ABLATION_RESULTS_2026-09-05.md`:
substantial effects persist with microunit corrections, while reducing magnitude
also changes outcomes. The capped policy remains harmful in the 2200 case.
The first complete checkpoint assessment and action-record audit are in
`NATIVE_DAMPING_O71_E020_DEVELOPMENT_RESULTS_2026-09-05.md`: episode 20 is
ineligible; the registered campaign was subsequently stopped at 36/8/8.
Prospective acceptance criteria and remaining execution prerequisites are in
`NATIVE_DAMPING_ACCEPTANCE_PROTOCOL_2026-09-05.md`; acceptance data remain sealed.
The current manuscript methods draft is
`TRAINED_DAMPING_METHODS_DRAFT_2026-09-05.md`; the earlier
`IDM_PAPER_SKELETON.md` is historical and supplies no current-pipeline results.
The primary-source introduction and related-work draft is
`TRAINED_DAMPING_INTRO_RELATED_WORK_2026-09-05.md`; its proposed contribution
remains conditional on validated damping and escalation results.
The learning check is bounded development work, not an accepted checkpoint.
Wall/predictor campaigns remain paused. Earlier wall documents below are
historical evidence and do not override this objective.

- `IDM_V5_7_1_ORIGINAL_CORRIDOR_EXPANDED_ZONE_PROTOCOL.md` — corrected,
  unexecuted original-corridor 200--1,900 m / 58 s low-authority mechanism
  protocol. It replaces neither the v5.6.2 no-go nor the unexecuted v5.7
  derived-network design.

- `IDM_V5_7_EXTENDED_CONTROL_ZONE_SOURCE_AUDIT.md` — treatment-blind
  kinematic derivation of the single 900 m extended approach / 1,900 m
  eligibility-boundary mechanism test.
- `IDM_V5_7_EXTENDED_CONTROL_ZONE_FROZEN_PROTOCOL.md` — frozen four-seed
  development protocol for the distance-triggered, all-0.3 m/s2,
  multi-step-assembly / 40 s HOLD / 8 s taper wall. No result is implied.

- `IDM_V5_5_T880_SAFETY_CAUSAL_DIAGNOSIS_RESULTS.md` — exact sham/assembly-only
  diagnosis showing hold/release as the supported safety mechanism.
- `IDM_V5_5_T880_HOLD_RELEASE_REPAIR_PROTOCOL.md` — review-only, versioned
  repair proposal; no controller change or repair gate has run.
- `IDM_V5_5_T880_SAFETY_CAUSAL_DIAGNOSIS_PROTOCOL.md` — diagnosis-only sham
  and assembly-only ablations for t880 wall safety causality.
- `experiments/idm/run_idm_v5_5_t880_safety_causal_diagnosis.py` — frozen
  12-branch diagnosis runner.

- `IDM_WALL_V5_3_SELECTIVE_GATE_SUPPORT_RESULTS.md` — prospective v5.3 support
  no-go and exact interrupted-taper diagnosis.
- `IDM_WALL_V5_4_EMERGENCY_HANDOFF_SUPPORT_PROTOCOL.md` — frozen fresh-cohort
  protocol for the explicit emergency zero-authority damping handoff.
- `IDM_WALL_V5_4_P25_AVAILABILITY_RESULTS.md` — treatment-blind p25 placement
  no-go on fresh seeds; no wall outcomes opened.
- `IDM_WALL_V5_4_P30_SUPPORT_AMENDMENT.md` — one frozen treatment-blind p30
  operating-point amendment with unchanged wall and outcome gates.
- `IDM_WALL_V5_4_P30_AVAILABILITY_RESULTS.md` — p30 treatment-blind placement
  pass (27 windows/16 seeds); bounded paired support is the next gate.
- `IDM_WALL_V5_4_P30_SUPPORT_RESULTS.md` — v5.4 physical/dose integrity and
  safety-first class-support pass authorizing grouped development generation.
- `IDM_V5_4_TWO_STAGE_PREDICTOR_DEVELOPMENT_PROTOCOL.md` — frozen p30 Gate-F /
  Gate-B partitions, class minima, models and grouped threshold rule.
- `IDM_V5_4_OBSERVATION_SUFFICIENCY_DIAGNOSIS.md` — development-only audit of
  all saved decision-time observations on seeds 4401--4520: dose is partly
  observable, benefit is unstable, safety remains near chance under fully
  nested grouped selection, and new shadow-rollout information is recommended.
- `IDM_V5_4_OBSERVATION_V2_REVIEW_PROTOCOL.md` — review-only causal 10-second
  damping-versus-wall shadow observation contract; no implementation, fitting
  or new-seed execution is authorized by the document.
- `IDM_V5_4_SHADOW_OBSERVATION_V2_PROTOCOL.md` — frozen executable,
  development-only observation-repair protocol for exact 10-second SUMO twins,
  causal predecision trajectories, grouped model comparison and a bounded
  scientific go/no-go.
- `IDM_V5_4_SHADOW_OBSERVATION_V2_RESULTS.md` — completed 120-seed leakage-safe
  grouped observation screen: shadow features strongly predict dose and improve
  benefit ranking, but safety and threshold-policy gates fail; the appended
  73-case mechanism audit localizes only 16 failures to taper/release and finds
  insufficient 2700 collapse-prevention headroom; no validation or final
  partition was opened.
- `IDM_WALL_V5_5_DENSITY_DURATION_HEADROOM_PROTOCOL.md` — approval-bound frozen
  2200/2400/2700 by early/late timing by 20/30/40-second matched development
  discriminator; defines the evidence required before adaptive duration,
  successor walls, strength/placement changes or a safety/service pivot.  No
  implementation or execution is authorized yet.
- `IDM_V5_5_DAMPING_DENSITY_MAPPER_PROTOCOL.md` — frozen executable 21-branch
  damping-only development screen using an absolute 620--1080 s clock,
  prevalence/incidence separation, three exact repeats and bounded telemetry;
  it selects a later treatment regime but cannot establish damping efficacy,
  critical density, wall headroom or predictor validity.
- `IDM_V5_5_DAMPING_DENSITY_MAPPER_RESULTS.md` — completed integrity-clean
  development mapper: 2200/2400 are 0/6 subcritical candidates, 2700 is a
  4/6-versus-2/6 late drain-down upper-transition candidate, and legacy health
  overlabels the physical endpoint. Selects one treatment-blind 880 s Gate-F
  check for an unchanged 20-second wall; no wall or predictor claim yet.
- `IDM_V5_5_T880_GATE_F_AND_WALL_PILOT_PROTOCOL.md` — frozen timing
  reconciliation and two-stage development gate: treatment-blind v5.4
  formation/safety/runway at absolute 880 s, followed conditionally by one
  unchanged 880--900 matched wall pilot through 1080/natural clearance.
- `IDM_V5_5_T880_GATE_F_AND_WALL_PILOT_RESULTS.md` — completed, integrity-
  reconciled development result: Gate F admits 4/6, only one treatment is both
  safe and beneficial, and predictor-scale expansion remains unauthorized.
- `IDM_V5_5_T880_CONTROL_IDENTITY_RECONCILIATION_PROTOCOL.md` — bounded
  no-wall audit separating full microscopic/control identity from the
  runner-dependent supervisor episode-progress feature in the omnibus hash.

- `IDM_DECENTRALIZED_BOUNDED_HARM_CONTRACT.md` — current frozen scientific
  contract for lower/transition IDM damping, bounded harm, and high-density
  residual risk reserved for later wall escalation.
- `IDM_DECENTRALIZED_LOWER_OBSERVABILITY_VALIDATION_PROTOCOL.md` — frozen
  lower-loading validation of the current observation and the single proposed
  local-zone-plus-risk-recurrence upgrade before any new RL training.
- `IDM_DECENTRALIZED_RANKED_NEIGHBOR_VALIDATION_PROTOCOL.md` — one-time fresh
  development validation of the fixed three-nearest-neighbor representation
  and conservative 2000-veh/h damping operating domain.
- `IDM_DECENTRALIZED_P30_FRESH_VALIDATION_REPORT.md` — replicated rejection of
  p30 benefit prediction, fresh operating-domain results, placement diagnosis,
  p50 projection diagnosis, and rejection of a universal full-duration gate.
- `IDM_DECENTRALIZED_P50_DEVELOPMENT_SCREEN_PROTOCOL.md` — frozen matched
  p30-versus-p50 treatment and predeclared development decision rule.
- `IDM_DECENTRALIZED_P50_DEVELOPMENT_SCREEN_RESULTS.md` — completed matched
  screen, failed harmful-rate retention gate, seven-mechanism diagnosis and
  proposed rear-aware feedback replacement; no training authorized.
- `IDM_RSU_OBSERVATION_DIAGNOSTIC_RESULTS.md` — reproduced 29/30 padded classic
  RSU decisions, exact 2000 m observation-only replay, failed grouped live-RSU
  observability ablation and no-MAPPO decision.
- `IDM_ACTOR_CRITIC_OBSERVATION_REPAIR_PLAN.md` — confirmed loss of explicit
  follower identity, missing action-authority fields and ego-omitting critic,
  with a versioned future repair/test plan.
- `IDM_REAR_AWARE_FEEDBACK_TREATMENT_PROPOSAL.md` — historical proposal that led
  to the separately frozen modified-FollowerStopper P30 treatment.
- `IDM_MODIFIED_FOLLOWERSTOPPER_P30_DEVELOPMENT_PROTOCOL.md` and
  `IDM_MODIFIED_FOLLOWERSTOPPER_P30_VALIDATION_PROTOCOL.md` — frozen treatment,
  viewed design-smoke disclosure and 14-case treatment-naive gate.
- `IDM_MODIFIED_FOLLOWERSTOPPER_P30_VALIDATION_RESULTS.md` — completed mechanism
  diagnosis: mechanics and bounded safety passed, but the treatment reproduced
  P30 in 13/14 cases and failed efficacy/headroom; no RL training authorized.
- `IDM_P30_PROJECTED_BLEND_AUTHORITY_PROTOCOL.md` and
  `IDM_P30_PROJECTED_BLEND_AUTHORITY_RESULTS.md` — implemented native-IDM-to-P30
  projected authority, exact 15-case development screen, real command diversity
  but failed mechanics and seed-diverse outcome-headroom gates; no observation,
  reward, or RL work authorized.
- `IDM_DECENTRALIZED_HIGH_LOADING_ESCALATION_DIAGNOSIS.md` — delivered-authority
  evidence that the highest-loading p30 insufficiency is residual traffic risk,
  plus the explicit absence of collapse-prevention headroom.
- `IDM_DECENTRALIZED_REWARD_CREDIT_DIAGNOSIS.md` — fresh evidence that the
  selected-agent reward has a structural lifetime/horizon mismatch and the
  required option-level training abstraction after treatment selection.
- `IDM_BINARY_OPTION_REWARD_PROTOCOL.md` and
  `IDM_BINARY_OPTION_REWARD_RESULTS.md` — frozen paired 60-second binary
  activation reward and passing 90-decision audit; authorizes a paired option
  wrapper but not the ordinary multi-agent reward or RL training.

This index distinguishes current scientific contracts from historical evidence.
When documents disagree, use the first applicable document in the current-work
section and treat the older document as provenance only.

The repository restructuring and deletion audit is recorded in
`CLEANUP_REPORT_2026-08-27.md`.
Maintained module ownership and dependency rules are in `CODE_ARCHITECTURE.md`.

## Current work

- `NATIVE_DAMPING_V1_IMPLEMENTATION.md` — **current implementation status**:
  sensor-local continuous controller foundation, causal team-credit primitives,
  executable actor/actuator adapter, regression tests and a passing exposed-seed
  mechanical comparison. The collector and learner are now implemented.
- `NATIVE_DAMPING_V1_LEARNING_CHECK.md` — **bounded learning and repair evidence**:
  explicit request accounting, fixed-clock PPO, the stopped per-episode-centering
  defect, its reproduced/fixed terminal-penalty regression, exact reuse/budget
  accounting, and deterministic development evaluation. No accepted checkpoint.
- `TRAINED_DAMPING_TO_PAPER_ROUTE_2026-09-05.md` — **read first: the owner
  requires standalone trained damping, then escalation**. Current evidence,
  distinction between residual control and fixed-P30 activation learning,
  proposed bounded training route, checkpoint acceptance, and downstream
  paper dependencies. No experiment launched by this planning review.
- `PAPER_COMPLETION_PLAN_2026-09-05.md` — **owner-requested post-stop
  evidence review and wall-only alternative**, superseded as the primary
  route by the owner's trained-damping clarification above.
  Stage 2 must not resume unchanged: horizon, command restoration, and
  CAV-only observation requirements are unresolved. This is a proposal,
  not authorization to execute or alter frozen experiments.
- `HANDOFF_2026-09-05.md` — **prior scientific and operational handoff**
  for the next agent: frozen treatment (onset + loading threshold +
  duration map), measured boundary, remaining Stage 2 → freeze →
  prospective order, seed registry, and claim boundaries.
- `STAGE2_NEXT_AGENT_PROMPT.md` — **stopped historical execution prompt**;
  do not resume before the trained-damping dependencies above are resolved.
- `STAGE1C2_RESULTS.md` — hold-duration / onset-trigger / fair-threshold
  / lead-time results that froze the current policy shape.
- `ESTIMAND_RESET_2026-09-02.md` — owner-directed reset of treatment and
  label definitions, superseding the v5.x treatment treadmill (diagnosis,
  decisions, retained machinery).
- `IDM_V5_8_MULTIWALL_SKELETON_HANDOFF.md` — v5.8 multiwall skeleton
  engineering handoff, preparatory only (halted 2026-09-02; see
  ESTIMAND_RESET).
- `IDM_REAR_AWARE_FEEDBACK_V2_FROZEN_PROTOCOL.md` — frozen v2 closed-loop
  bilateral pulse treatment, cohort, gates and exclusions.
- `IDM_REAR_AWARE_FEEDBACK_V2_SIX_SEED_PILOT_RESULTS.md` — completed matched
  six-seed mechanism pilot: integrity gates passed, primary development
  efficacy gates failed (1/5 harmful reduction, 1/5 benefit seeds); no
  training authorized, no seeds added, no treatment change.
- `IDM_REAR_AWARE_FEEDBACK_V2_FAILURE_DIAGNOSIS.md` — diagnosis-only phase on
  existing artifacts: probe/release were neutral, harm arose 30--60 s after
  handoff in the ramp/merge zone; 4127 and 4123 were not distinguishable
  before the harmful response; no v2.1 mechanism can pass the primary gates;
  pulse declared a mechanism no-go, fallback program follows.
- `IDM_FIXED_P30_LIMITED_DOMAIN_PROTOCOL.md` — frozen limited-domain (5--10%
  penetration) fixed-P30 observation/value screen protocol.
- `IDM_LIMITED_DOMAIN_P30_SCREEN_RESULTS.md` — screen results: no causal
  observation model beats the fixed 5--10% rule; learned activation not
  supported; no RL authorized.
- `IDM_FIXED_P30_LIMITED_DOMAIN_CONTROLLER.md` — frozen rule-based P30
  damping arm (development arm, prospective acceptance pending).
- `IDM_RL_TIMING_CANDIDATE_HEADROOM_PROTOCOL.md` — frozen overnight RL
  timing/candidate headroom protocol.
- `IDM_RL_TIMING_CANDIDATE_NOGO_DIAGNOSIS.md` — 27-state paired library,
  oracle-headroom gate failure (single-state gain, inadequate diversity),
  value-screen failure, no-RL conclusion; fixed rule retained. Scope:
  candidate-1-vs-candidate-2 only; sequential timing not tested.
- `IDM_P30_RULE_ARM_PROSPECTIVE_ACCEPTANCE_PROTOCOL.md` and
  `IDM_P30_RULE_ARM_PROSPECTIVE_ACCEPTANCE_RESULTS.md` — frozen Path A and its
  completed no-go: 58 activations over 120 opportunities, 12 helpful / 13
  harmful / 33 neutral, slightly positive physical effect but negative
  constraint-aware value and failed service/LOO gates; the rule remains
  development-only.
- `IDM_SEQUENTIAL_TIMING_HEADROOM_PROTOCOL.md` — frozen sequential
  activate-now/defer-r2/defer-r3 timing headroom pilot protocol (common
  single-trajectory schedule, common horizon, rate normalization, oracle
  gate); executed 2026-08-30.
- `IDM_SEQUENTIAL_TIMING_HEADROOM_RESULTS.md` — completed Path B pilot:
  69-state common-horizon cohort, all fixed timing rules net-negative, but
  the constrained timing oracle passes all seven gates (gain +0.048, robust
  LOO); stopped after oracle analysis per protocol; next gated phase is the
  observation/value screen (not started).
- `IDM_TIMING_OBSERVATION_VALUE_SCREEN_PROTOCOL.md` — frozen causal
  observation/value screen protocol for the sequential timing decision
  (recursive targets, feature provenance, LOO, frozen gates).
- `IDM_TIMING_OBSERVATION_VALUE_SCREEN_RESULTS.md` — screen result:
  **gate fails** — coarse family-A features recover no deployable timing
  signal (ridge worse than abstain and below shuffled control);
  families B–D not evaluable from existing artifacts; no RL.
- `IDM_TIMING_OBSERVATION_REPAIR_PROTOCOL.md` and
  `IDM_TIMING_OBSERVATION_REPAIR_RESULTS.md` — observation-repair phase:
  native-IDM capture of 40 compact B–D features at all 170 landmarks with
  exact identity (0 mismatches), then family-incremental screen: every
  family increment fails; fixed-P30 timing prediction unsupported;
  rule-based damping + moving-wall escalation recommended.
- `IDM_DAMPING_AUTHORITY_MAP_DEVELOPMENT_RESULTS.md` — development-only
  continuous authority map over 182 existing paired opportunities; response
  is nonmonotonic across demand and penetration, so no fixed critical-density
  switch or binary safe-but-insufficient label is claimed.
- `IDM_COMPACT_WALL_FEASIBILITY_SCAN_PROTOCOL.md` and
  `IDM_COMPACT_WALL_FEASIBILITY_SCAN_RESULTS.md` — treatment-blind p10--p30
  placement scan; 30% is the lowest level passing the predeclared availability
  rule (12 windows across 9 seeds).
- `IDM_WALL_V5_2_PROSPECTIVE_PROTOCOL.md` and
  `IDM_WALL_V5_2_P30_PILOT_RESULTS.md` — damping-preserving p30 paired wall
  pilot: exact identity and dose/diversity gates passed, but safety was 5/10
  completed cases versus the frozen 90% requirement; predictor unauthorized.
- `IDM_WALL_V5_3_MECHANISM_REPAIR_PROTOCOL.md` — owner-approved frozen next
  treatment version: bounded pre-latched same-lane handoff, compactness
  enforcement, and a fixed five-second in-horizon linear taper. These are
  treatment-integrity repairs, not a claimed repair for delayed downstream
  safety harm; a treatment-blind feasibility scan is required before Phase B.
- `IDM_WALL_V5_2_RELEASE_SAFETY_DIAGNOSIS.md` — read-only paired diagnosis of
  the five safety failures: 248 positive excess hard-brake onsets occur more
  than 60 seconds after release versus one in the first 20 seconds; terminal
  mechanics do not separate safe from unsafe cases.
- `IDM_TWO_STAGE_JOINT_SAFETY_BENEFIT_GATE_PROTOCOL.md` — owner-approved current
  estimand separating dose feasibility from conditional joint safety and
  benefit. The prospective 90% safety requirement applies to gate-selected
  activations, while every raw unsafe treatment remains a Gate-B negative.
- `IDM_OBSERVATION_V2_SCREEN_RESULTS.md` — completed Phase-A result: exact
  observation-v2 prefix capture and descriptive class headroom pass, but the
  frozen v5.3 seven-case replay fails safety non-worsening and retention gates;
  scale-up, predictor fitting, and Phase B remain unauthorized.
- `IDM_WALL_V5_3_STAGE1_FAILURE_DIAGNOSIS.md` — read-only event-level diagnosis:
  the 4312 one-step regression is a delayed post-treatment trajectory
  divergence that remains safety-admissible, while the safe-beneficial
  retention gate used mismatched starts; the original no-go is preserved.
- `IDM_WALL_V5_3_SELECTIVE_GATE_PROSPECTIVE_PROTOCOL.md` — frozen prospective
  4313--4332 treatment-blind feasibility and bounded paired-support protocol
  for the unchanged v5.3 wall under the adopted selective-gate estimand.
- `IDM_WALL_V5_2_DECISION_OBSERVATION_AUDIT.md` — read-only audit of all 48
  frozen decisions: Gate-B outcome diversity exists, Gate-F has only two
  failed-dose seeds, and the 252-field contract lacks explicit post-merge state
  where delayed safety divergence concentrates.

1. `CLOSED_LOOP_BILATERAL_PULSE_HANDOFF.md` — authoritative transfer record for
   the user-authorized replacement of fixed P30 with a causal, rear-aware,
   probe-hold-release pulse; records current partial code, exact next tests,
   proposed six-seed matched pilot, gates and authorization boundaries.
2. `IDM_P30_SIX_PENETRATION_VALUE_DIAGNOSIS.md` — mature-wave six-penetration
   action-headroom analysis, failed causal temporal and conditional expected-
   value gates, limited 5--10% fallback, and the closed-loop treatment decision
   boundary before any further training.
3. `IDM_DECENTRALIZED_TRAINING_CONTAMINATION_2026-08-28.md` — quarantines the
   duplicate-writer MAPPO run and records the clean-smoke safeguards.
4. `IDM_DECENTRALIZED_CLEAN_SMOKE_V3_DIAGNOSIS.md` — failed checkpoint and
   threshold evaluation, action-specific sufficiency evidence, and the current
   p30 treatment-dose blocker.
5. `IDM_DECENTRALIZED_ACTIVATION_GATE_V1_REPORT.md` — verified p30 terminal-speed
   activation treatment, pooled pretraining gate, reward alignment, and
   resumable PPO training status.
6. `IDM_DECENTRALIZED_ESCALATION_MASTER_PLAN.md` — controlling implementation
   and paper plan.
7. `IDM_DECENTRALIZED_CONTROL_CONTRACT.md` — frozen decentralized actuator.
8. `IDM_DECENTRALIZED_STRUCTURED_CONTROLLER_PROTOCOL.md` — implemented local
   five-mode candidate, contention, safety and option contract; efficacy not yet
   established.
9. `IDM_RESIDUAL_PHASE4_PROTOCOL.md` — analytical baseline plus bounded residual.
10. `IDM_DECENTRALIZED_CONTROLLER_SMOKE_V1_REPORT.md` — passing real-SUMO
   integration smoke; not efficacy evidence.
11. `IDM_DECENTRALIZED_AUTHORITY_VERIFICATION_PROTOCOL.md` — frozen paired
   candidate-and-mode pretraining gate.
12. `IDM_DECENTRALIZED_AUTHORITY_PILOT_V1_REPORT.md` — passed placement and
   activation authority, failed corrected structured-mode projection survival;
   PPO remains unauthorized.
13. `IDM_PHASE5_ENDPOINT_ALIGNED_READINESS.md` — current training-readiness gate.
14. `IDM_PHASE5_ACTION_LOCAL_PULSE_PROTOCOL.md` — current resumable action-credit gate.
15. `IDM_PHASE5_DOSE_RESPONSE_V1_DIAGNOSIS.md` — completed 20-second residual
   pilot, causal dose audit and decision not to authorize PPO.
16. `IDM_STRUCTURED_AUTHORITY_SCREEN_PROTOCOL.md` — gain/reference/mode screen
   required before any new IDM RL training.
17. `IDM_STRUCTURED_AUTHORITY_SMOKE_V1_REPORT.md` — passing two-seed compact
   instrumentation and parallel-execution smoke.
18. `IDM_STRUCTURED_AUTHORITY_SCREEN_V1_REPORT.md` — completed gain/reference
   results, demand-stratified failures, and no-PPO decision.
19. `IDM_PHASE5_TRAINED_POLICY_ACCEPTANCE.md` — independent-seed checkpoint gate.
20. `IDM_BINARY_OPTION_REWARD_RESULTS.md` — passing causal binary-option reward
    and exact boundary before paired-wrapper implementation.
21. `IDM_P30_PROJECTED_BLEND_AUTHORITY_RESULTS.md` — current negative graded-
    authority result and no-observation/no-training decision.
22. `IDM_MODIFIED_FOLLOWERSTOPPER_P30_VALIDATION_RESULTS.md` — prior negative
    action-mechanism result and no-training decision.
23. `IDM_TWO_STAGE_PREDICTOR_CONTRACT.md` — eventual escalation predictor contract.
24. `COUNTERFACTUAL_ESCALATION_V2_PLAN.md` — damping-preserving paired treatment.
25. `COUNTERFACTUAL_EXPERIMENT_CONTRACT.md` — paired experimental identity rules.
26. `CURRENT_PREDICTOR_LIMITATIONS_HANDOFF.md` and
    `PREDICTOR_DIAGNOSIS_HANDOFF.md` — known limitations and failed assumptions.
27. `IDM_PAPER_SKELETON.md` — current manuscript framing.

The reusable seed/provenance contract is implemented in `idm/contract.py`; IDM
pilot metric helpers are in `idm/pilot_metrics.py`.

Current treatment-development records include the `ADAPTIVE_WALL_*`,
`COMPACT_MOVING_WALL_*`, `IDM_WALL_*`, `IDM_MULTWINDOW_*`, and
`TERMINAL_WALL_SPEED_*` documents. They are development evidence, not efficacy
claims. Current predictor pilot documents beginning with `COUNTERFACTUAL_V5_*`
are frozen protocols or diagnoses, not proof that the predictor works.

## Historical evidence

`LEGACY_RESEARCH_RECORD.md` is the compact entry point for the completed Krauss,
early IDM wall, ExpC, reward, action-space, and metastable-health campaigns.
Detailed provenance remains in:

- `RESEARCH_OVERVIEW.md`
- `metastable_merge_control.md`
- `metastable_probe_program.md`
- `expC_metastable_health_mask.md` and `expC_runbook.md`
- `GOAL_FAIR_SAFE.md`, `PLAN_V9_IMPROVE_PASS.md`, and `PROJECT_GUIDE.md`
- `superpowers/plans/`, `superpowers/specs/`, and `superpowers/reports/`
- preliminary `COUNTERFACTUAL_*RESULTS.md` and sensitivity documents

These files explain why earlier approaches were rejected and preserve exact
reported values. Their obsolete one-off Python probes and launch scripts are
not part of the maintained codebase.

## Artifact navigation

See `ARTIFACT_MANIFEST.md`. `runs/` is intentionally ignored by Git and contains
both current evidence and historical experimental artifacts. Do not cite a run
only from its directory name; pair it with the relevant frozen protocol and
verify its row/schema version.

## Claim status

As of 2026-08-30:

- software baseline: 803 tests passed after the v5.2 wall, treatment-blind
  feasibility integration, release-safety diagnosis, and decision-observation
  contract audit;
- IDM residual/controller implementation exists;
- current Phase-5 training readiness is not fully passed;
- no trained IDM damping checkpoint has passed prospective acceptance;
- the mechanically valid P50 action tied P30's development harmful fraction
  and is not retained for training;
- the modified-FollowerStopper P30 ceiling passed identity, dose and bounded
  safety checks but materially changed commands in only 1/14 treatment-naive
  cases, tied P30's 3/14 harmful count and is not retained for training;
- projected native-IDM-to-P30 blending delivered distinct commands in most
  cases but failed one dose/conflict case, the literal executed-jerk gate and
  predeclared seed-diverse intermediate-outcome headroom; its observation/reward
  expansion and RL training are not authorized;
- the paired fixed-horizon binary activation reward passed all 90-decision
  alignment gates, but requires a new counterfactual option wrapper; the old
  variable-lifetime multi-agent reward remains rejected;
- live existing classic RSU fields did not improve grouped P30 benefit
  observability in the matched 11-case development diagnostic;
- the wall treatment and escalation predictor are not paper-verified;
- prior Krauss and early IDM results remain historical evidence only.
# IDM v5.5 t880 safety-causality diagnosis (diagnosis-only)

- `docs/IDM_V5_5_T880_SAFETY_CAUSAL_DIAGNOSIS_PROTOCOL.md`
- `experiments/idm/run_idm_v5_5_t880_safety_causal_diagnosis.py`

## Current v5.5.1 adaptive-release no-go

- `IDM_V5_5_1_ADAPTIVE_HOLD_RELEASE_PROTOCOL.md` — frozen follower-aware,
  damping-preserving treatment definition and fingerprint.
- `IDM_V5_5_1_FOUR_CASE_REPAIR_RESULTS.md` — completed paired 4401/4402/4404/4405
  mechanism gate; failed safety/lifecycle requirements.
- `IDM_V5_5_1_REPAIR_NOGO_DIAGNOSIS.md` — required stop decision and ranked
  review-only paper routes. Predictor generation/training remains unauthorized.

## Current v5.6 treatment tournament no-go

- `IDM_V5_6_TREATMENT_TOURNAMENT_PROTOCOL.md` — frozen six-route, development-
  only selection protocol and all-case winner rule.
- `IDM_V5_6_LOW_AUTHORITY_AUDIT.md`,
  `IDM_V5_6_SUPPORT_CAV_AVAILABILITY_AUDIT.md`,
  `IDM_V5_6_RELEASE_AND_DURATION_AUDIT.md`,
  `IDM_V5_6_GEOMETRY_FEASIBILITY_AUDIT.md`, and
  `IDM_V5_6_MULTIWALL_FEASIBILITY_AUDIT.md` — independent route-entry audits.
- `IDM_V5_6_TREATMENT_TOURNAMENT_RESULTS.md` — Candidate A four-case failure;
  no winner, fresh-seed opening, or predictor authorization.

## Current v5.6.1 early-formation no-go

- `IDM_V5_6_1_EARLY_LOW_AUTHORITY_FORMATION_PROTOCOL.md` — frozen early-
  assembly causal definition and treatment-blind common-start selection rule.
- `IDM_V5_6_1_EARLY_LOW_AUTHORITY_NOGO_DIAGNOSIS.md` — no common safe,
  continuous early formation in the frozen 860--875 s grid; no treatment
  implementation, SUMO outcome, fresh-seed, or predictor authorization.

## Current v5.6.2 phase-split treatment no-go

- `IDM_V5_6_2_PHASE_SPLIT_AUTHORITY_PROTOCOL.md` — separately versioned
  one-step high-authority assembly, actual-command jerk-limited settling, and
  0.3 m/s² sustained-authority definition.
- `IDM_V5_6_2_PHASE_SPLIT_AUTHORITY_SOURCE_AUDIT.md` — source-path evidence
  for the phase caps and the newly frozen transition jerk bound.
- `IDM_V5_6_2_PHASE_SPLIT_AUTHORITY_FOUR_CASE_RESULTS.md` — completed
  4401/4402/4404/4405 gate: authority/safety/service passed, but compactness
  and benefit retention failed; no fresh screen or predictor authorization.
- `IDM_V5_6_2_PHASE_SPLIT_AUTHORITY_NOGO_DIAGNOSIS.md` — required stop
  decision and the bounded diagnosis; this treatment must not be retuned.
