# Restored route: trained damping first, then escalation, then the paper

> Implementation has started under the active goal. See
> [the first controller implementation and mechanical evidence](NATIVE_DAMPING_V1_IMPLEMENTATION.md).
> The earlier planning-only status below is historical. Mechanical and collector
> checks have run, and a bounded learning check plus its repair are complete; see
> [the fixed training protocol and current evidence](NATIVE_DAMPING_V1_LEARNING_CHECK.md).
> The short prototype is rejected as finished damping; full training remains
> outstanding and has now started under
> [the registered full campaign](NATIVE_DAMPING_FULL_CAMPAIGN_2026-09-05.md).
> Wall generation remains paused.

Original planning status: owner-corrected project objective and proposed implementation route, 2026-09-05. The linked implementation records above supersede the original planning-only status. This supersedes the prior paper plan's decision to make trained damping optional. Historical failed experiments remain failed; they are not erased or authorized for unchanged resumption.

## 1. The objective that must not change implicitly

Deliver a trained decentralized damping controller that works without a wall, freeze that controller, then learn when an RSU should add coordinated sustained control from CAV-acquired reports. Finally evaluate the integrated system on untouched scenarios.

"Works on its own" means a trained checkpoint produces useful traffic stabilization within a declared operating range without an RSU issuing damping actions or a wall rescuing its evaluation. It does not mean universal improvement, removal of the safety shield, or that physical equations cannot be part of the controller.

The prior September 5 plan was a smaller wall-policy paper. It did not satisfy this objective. Its 1208-rollout total does not budget the restored learned-damping project and must not be quoted as the cost of completing it.

## 2. Where the work actually stands

There are three distinct objects that previous navigation conflated:

1. **Analytical damping:** a hand-designed selected-absorber controller, used in current wall comparisons. It is neither an accepted learned checkpoint nor evidence about every possible learned controller.
2. **Continuous residual learning:** Phase 4 provided a bounded scalar acceleration residual in [-0.5,+0.3] m/s² around the analytical controller, with a selected-absorber gate and safety projection. RLlib smoke/checkpoint infrastructure exists. Later action-local tests found substantial eligibility/filter/projection attenuation; the 20-second pilot delivered adequate duration in 6/14 treatments and found no helpful dose-qualified effects. This is evidence against that tested interface, not a reason to claim learned damping was finished.
3. **Learned fixed-maneuver activation/timing:** later work trained or screened whether to apply a fixed 20-second P30 maneuver. It learned selection of a scripted treatment, not an ongoing continuous damping response. Fresh observation and timing screens did not recover useful deployable value. That failure does not establish impossibility for a different closed-loop control law.

Actual checkpoint evidence:

- The clean v3 10k activation smoke became selective (29/233 eligible decisions), but its development safety/TTC comparison failed. This is a failed tested checkpoint, not absent training infrastructure.
- The later paired-option wrapper **is implemented** in `idm/paired_option_env.py`; the August 29 reward report's statement that it still needed implementation is now stale.
- `runs/rllib/idm_paired_option_smoke_10pct_v1/summary.json` records 2000 stored-sample training steps: train mean paired reward +0.0558, development validation mean reward -0.0643. Validation activated 8/28 cases, with two helpful and two harmful. These are stored one-step decision samples, not 2000 seconds of fresh SUMO and not independent final evaluation.
- No accepted standalone learned IDM damping checkpoint was identified in the authoritative acceptance records. A checkpoint directory name alone does not establish acceptance. Contaminated activation-smoke v1 remains quarantined.

Useful retained work: IDM scenarios; deterministic replay; front/rear safety projection; physical command telemetry; checkpoint infrastructure; fixed-horizon causal reward/accounting lessons; seed exclusion machinery; evidence about weak/projected actions and P30 timing; sustained-wall physical implementation and provisional mechanism results.

Avoidable project drift: the analytical fallback became the wall control arm before the requested trained controller was delivered; several documents contain incompatible active directions; readiness claims outlived the code/artifacts supporting them. The stopped Stage 2 additionally has the horizon/observation mismatches documented in the prior audit. The right response is to restore dependency order, not to call every prior negative finding worthless.

## 3. One recommended damping design, not a new menu of pilots

**Proposed successor:** a parameter-shared, closed-loop per-CAV policy producing a bounded acceleration correction around native IDM, with a deterministic safety projection. This is a new versioned hypothesis, not resumption of the rejected Phase-5 residual or fixed-P30 policy.

For CAV i at time t:

`a_request(i,t) = a_nominal_local_IDM(i,t) + delta_theta(local_history(i,t))`

`a_command(i,t) = safety_project(a_request(i,t))`

Initial proposed residual bounds retain [-0.5,+0.3] m/s². In the implementation, exact zero sends no command (or releases this controller's prior override); the local IDM equation is only a nominal approximation, not an exact reconstruction of SUMO native acceleration. SUMO's native safety may override issued targets. The nominal term for a CAV uses its own known controller parameters and measured leader state; no private HDV parameter is an actor input.

Why change the nominal term? Learning around the existing analytical P30 target preserves its imposed response and can leave the learner with little effective freedom. Native IDM provides a meaningful zero-action baseline without forcing that maneuver. Retain the old analytical controller as an explicit comparator. This changes the controller contract and requires new data; the historical paired labels cannot be relabeled as samples of this action.

Each CAV uses the same weights, selects its own correction at the 1 s control cadence, and can respond again to new measurements. Do not retain the old 20 s P30 latch or selected-absorber-only mask in this successor. These are explicit design changes, not an instruction to remove local safety limits. Retain collision-avoidance, leader/rear risk constraints, acceleration and jerk limits. Avoid stacking a redundant five-second residual filter on top of jerk limiting unless the engineering audit establishes a reason for it.

Inputs: self speed/acceleration, previous executed correction, front and rear gaps/relative speeds with visibility flags, local causal trends, lane/map position, and measurement age. Use a fixed short history (proposed 5 s). The nominal sensor assumption is nearest visible same-lane front/rear measurements within 100 m. Report unsupported/missing measurements explicitly. No global SUMO health, hidden demand, future driver intention, true penetration, or all-vehicle density enters the damping actor. No RSU is needed to execute damping. RSU report fusion belongs to escalation.

This is a recommendation to test, not an assertion that these inputs or bounds are already sufficient. Local-feedback learned traffic control has precedent, but those results do not establish this merge's performance: [Flow benchmarks](https://proceedings.mlr.press/v87/vinitsky18a.html) and [Flow framework](https://arxiv.org/abs/1710.05465). [Stern et al.](https://arxiv.org/abs/1705.01693) supports the physical possibility of wave damping in a different ring-road setting, not a guarantee here.

## 4. Finish the learning contract before spending on training

There is one engineering work package, with four outputs:

1. **Effective authority:** zero equals native IDM; deliberate nonzero requests change executed commands in states where the safety bounds allow them; projected/forbidden actions remain blocked and logged. Verify requested -> projected -> executed actions, entry/exit, and jerk behavior. A numerically changing network output is insufficient.
2. **Observation boundary:** the actor uses the same sensor-derived fields in training and execution. A privileged centralized critic is training-only. Hidden-state changes that leave local measurements unchanged must not change the actor action.
3. **Team credit:** collect a fixed-clock corridor trajectory. Compute one physical team reward per simulator step, not a reward multiplied by the number of live CAVs. Give active actor samples advantages derived from that common trajectory, so credit can include downstream consequences after a contributing CAV leaves. Mask absent actors without ending the team return. Verify the loss/normalization and checkpoint restore. Do not simply plug the paired binary reward into the old variable-lifetime environment.
4. **Auditable objective:** penalize severe braking and acceleration energy, with explicit service and safety constraints. Freeze reference scales, penalty semantics, discount/return horizon, and checkpoint-selection metric together before training. Audit synthetic reward cases and saved physical outcomes for sign, duplicate-agent scaling, downstream delay and source-queue accounting. The old binary reward's 0.99944 correlation cannot certify this new continuous team objective.

The exact reward coefficients and rollout/loss implementation remain engineering deliverables; it would be misleading to invent a validated formula in this review. Use the existing equal-weight normalized braking/energy composite as the starting physical objective, not a new broad weight search. Selection must use measured physical outcomes, never training reward alone.

Suggested initial mechanical smoke: four exposed scenarios x native/zero/scripted-bounded-response, at most 12 rollouts. This does not need to demonstrate a statistically significant traffic benefit before learning. Do **not** require another static helpful/harmful classifier to achieve AUC 0.70 as a universal prerequisite to continuous RL. That old classifier answered a different action question. Require correct measurements, usable physical authority, coherent credit, and a functioning learner; then test whether training improves control.

## 5. One capped damping training campaign

Proposed operating scope: the existing IDM merge, p20, with randomized development demand in 1900–2200 veh/h plus a smaller prespecified overload component at 2600/2700 for learning conservative behavior. Hold ramp demand and driver family fixed initially. Register development seeds before use; configured demand is never an actor feature.

Use the existing PPO/MAPPO infrastructure with a small shared actor and training-only critic after the team-credit correction. Do not carry over a failed or contaminated checkpoint. Initialize zero correction.

- First 20,000 **global simulator control steps** verify finite learning, changing actor parameters, effective safe action diversity, stable losses, and save/restore. Count active actor samples separately; many CAV decisions in one second are not many seconds of traffic evidence.
- Continue one frozen configuration to 250,000 global steps for each of three optimizer seeds. Total campaign cap is 750,000 global steps, including the first seed's 20k check.
- Evaluate at fixed 50k/150k/250k checkpoints on eight development scenarios. The first 20k check, those evaluations, baselines, and engineering smoke fit inside a proposed cap of 100 full development evaluation rollouts. Reuse identical baseline rollouts.
- Select the checkpoint by predeclared constrained physical performance across development scenarios. Show all three training lineages. Do not choose from the training reward curve or repeatedly test final seeds.

The 20k point is a software/learning check, not a claim that 20k must solve traffic control. The 250k cap is a first bounded scientific campaign, not a convergence theorem. If all three lineages fail at the cap, diagnose whether authority, optimization, sensing, or physical efficacy is responsible using the existing logs; do not automatically change reward weights, add seeds, or jump back to walls. A further campaign would be a concrete revised hypothesis with a new budget.

## 6. Prove that damping works without a wall and freeze it

After development selection, compare one frozen checkpoint with native IDM and the analytical controller on 32 untouched scenarios (proposed eight each at 2000/2200/2600/2700): 96 paired-comparator rollouts. Audit exposure before allocating seed IDs; old reservation labels are not proof of untouched status. Do not open sealed artifacts in a planning pass.

Proposed acceptance target: within the stated lower/transition operating range, at least 5% mean improvement in braking or acceleration-energy burden, no material worsening of the other, and a paired confidence interval supporting aggregate improvement. Declare the tolerated harmful tail and numerical/reference conventions before evaluation. Service bounds should include native-IDM comparison, mean delay regression no more than the predeclared budget (existing protocol used +2 s/vehicle), source backlog, completions and unfinished traffic. Report uncertainty; eight seeds per loading is not a precise universal loading boundary.

At overload, require conservative/non-worsening behavior within the declared margins, not elimination of congestion. This separates a useful damping operating range from the residual problem escalation should address. A no-op policy does not pass the improvement requirement; an aggressive policy does not pass merely by lowering one wave metric while causing harmful braking or service loss.

If accepted, freeze D*: actor weights, code, observer, normalization, action/safety limits, training scope, and scenario configuration. The first paper result is then **standalone learned damping**. If acceptance fails, report that failure; do not relabel the analytical fallback as the completed learned controller or quietly downgrade the user's objective.

## 7. Only now rebase the wall and collect escalation labels

Rebase the sustained intervention on **D***, not the historical analytical arm. Correct the known out-of-band `setSpeed(-1)` cancellation. Enforce full 2400 s outcome/termination accounting and preserve D* exactly before engagement and on non-overridden vehicles.

Keep the physical wall settings initially: 200–400 m mainline band, target 10 m/s, original tau +3 s, 0.3 m/s² command clip, fixed hold in {300,450,650}, and 5 s taper. Do not start another geometry/strength/release scan.

Reuse the prior plan's 80-seed paired development-library concept, now with four arms D*, D*+H300, D*+H450, D*+H650: at most 320 rollouts. Make its first eight exposed scenarios the engineering integration batch; they count toward the 80, not another cohort. Pause only for an implementation/safety defect, not a noisy eight-seed efficacy estimate.

Old wall results inform the candidate action set and hypotheses. Their effect sizes, best durations, thresholds, and labels do not transfer numerically to a newly trained D*. A stronger D* may reduce, increase or change wall headroom. If that library shows no practically useful intervention headroom, do not train an escalation predictor to manufacture a second contribution; the completed damping result remains distinct and must be preserved.

## 8. Train the RSU escalation gate from CAV reports

Every CAV reports timestamped self state, local measurements, and current damping command/mode. The RSU combines the reports with explicit coverage, age, loss, and missingness. Save raw report histories in the paired library once. No exact all-vehicle health, hidden demand or simulator counterfactual is an online input.

Predict the expected full-horizon cost difference `J(D*+wall_H) - J(D*)` for each duration, and include no wall as zero. The deterministic availability check uses received reports. Compare a fitted simple rule and a small multifeature action-value model on equal action choices and seed-grouped folds. Select the model by held-out policy value, not an AUC gate or a preferred architecture.

For a bounded first paper, retain the proposed single scheduled opportunity at t=250 using 60 s report history. The lower damping actor runs continuously; only the wall opportunity is one-shot. This explicitly does not claim a learned event-time detector. Recurring or learned-timing escalation is a different intervention dataset and is not silently included.

Once D* is frozen, use the same development library for self-only versus local sensing, history and coverage ablations, duration analysis, and fitted-rule/model comparisons without more traffic simulation.

## 9. Evaluate the integrated hierarchy and finish the manuscript

Freeze both candidate gates before fresh validation. Proposed remaining evaluation budget:

- One conventional VSL comparator: at most 60 development tuning rollouts, with actuator differences disclosed.
- 24 fresh validation scenarios x six arms = at most 144 rollouts. Arms: native, D*, D*+best constant wall, simple report gate, multifeature report gate, VSL. Use a predeclared rule for selecting the simple versus complex gate; no threshold retuning on these results.
- 60 locked final scenarios x the same six arms = at most 360 rollouts, deduplicating identical policies. Final reserved seeds require the existing exposure audit and frozen-bundle conditions; do not read them for this plan.
- One communication stress cohort, 30 fresh scenarios x native/D*/selected integrated policy = 90 rollouts. Predeclare loss/delay and stale-command behavior. No automatic multi-network/penetration/driver-model expansion.

The post-damping ceiling is therefore 320+60+144+360+90 = **974 rollouts**, conditional on useful wall headroom. Damping adds at most **750k global training steps + 100 development evaluation rollouts + 96 untouched acceptance rollouts**. These are proposed compute limits, not a statistical power guarantee. Do not translate the prior 16–21 h wall-only estimate into an end-to-end training promise. Benchmark the actual corrected training sampler during the 20k check and forecast remaining wall time from global steps/second and rollout throughput.

Manuscript chain:

1. Standalone D* versus native and analytical baselines: stabilization, service and harmful tails.
2. Residual difficult regimes under D*: observed limits, not a universal critical density.
3. D*-preserving wall effect and cost by regime/duration.
4. CAV-report gate versus simple rule and always-on escalation.
5. Locked integrated results and communication stress, including source queues, throughput, ramp/mainline allocation, braking and TTC.

Working framing: **Learning Decentralized Traffic Damping with CAV-Informed Escalation to Coordinated Metering**. Positive claims remain conditional on measured results; a timeline or plan cannot guarantee successful learning or acceptance.

## 10. Project controls that prevent another two weeks of drift

- One active controller version, one active training configuration, one explicit next deliverable.
- Every run has a question, code/seed identity, compute cap, promised artifact, and a decision it enables.
- Separate software correctness from scientific efficacy. Do not demand that a four-seed smoke prove generalization; do not call a passing unit test a working controller.
- Once a configuration passes engineering checks, run the capped learning experiment instead of adding unbounded pretraining feature screens.
- Use fixed development evaluation checkpoints. Preserve negative results; do not repeatedly reinterpret a failure into a new success metric.
- No predictor-scale wall data while D* is still changing. No silent fallback from "trained damping" to "analytical base".
- If an assumption fails, explain the specific failed layer and propose one bounded correction. A failed run must change a concrete decision, not merely generate another report.

**Immediate deliverable:** the versioned native-IDM residual actor interface, admissible local observation schema, fixed-clock team credit/reward contract, and deterministic tests. Then the capped 20k learning check. No further wall generation is the next action.

## Source records

- `IDM_RESIDUAL_PHASE4_PROTOCOL.md`; `IDM_PHASE5_DOSE_RESPONSE_V1_DIAGNOSIS.md`.
- `IDM_DECENTRALIZED_CLEAN_SMOKE_V3_DIAGNOSIS.md`; `IDM_DECENTRALIZED_TRAINING_CONTAMINATION_2026-08-28.md`.
- `IDM_DECENTRALIZED_REWARD_CREDIT_DIAGNOSIS.md`; `IDM_BINARY_OPTION_REWARD_RESULTS.md` (historical implementation status corrected by current source).
- `IDM_TIMING_OBSERVATION_REPAIR_RESULTS.md`; `IDM_SEQUENTIAL_TIMING_HEADROOM_RESULTS.md`.
- `idm/paired_option_env.py`; `tools/train_idm_paired_option.py`; `runs/rllib/idm_paired_option_smoke_10pct_v1/summary.json`.
- `PAPER_COMPLETION_PLAN_2026-09-05.md` and the CAV/health/escalation audits linked there: retained evidence and wall-only alternative, superseded as the primary route.
