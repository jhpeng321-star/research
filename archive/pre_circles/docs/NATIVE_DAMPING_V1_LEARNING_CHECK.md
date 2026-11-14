# Native damping v1: bounded learning check

Status: **engineering package complete; smoke checkpoint rejected as finished damping.** The first learner was stopped after a demonstrated credit defect. PPO schema v2 repairs it; four corrected updates and all four deterministic comparisons are complete. One exposed case improves and three regress. The owner's trained-damping goal remains active, with no accepted checkpoint and no generalization claim.

The final deterministic comparison averages **3.63% more severe braking,
5.57% more acceleration energy, +1.734 s/requested vehicle, and +41.625
vehicle-seconds of TTC<3 exposure per case**. These are unweighted means over
four already exposed seed/loading cases, not independent prospective evidence.
All vehicles finish, with no reported collisions or teleports. One case exceeds
the +2 s/requested-vehicle service margin. The data do not justify freezing D*.

## Repair discovered during the first check

The v1 learner subtracted the mean advantage across times in one episode. Under gamma=lambda=1, a terminal service-failure penalty shifts every earlier team advantage by the same constant. Subtracting that episode's mean **erased that direct penalty from all actor samples** whenever other costs varied. The critic still received penalized targets, and the physical reward log remained correct; neither fact rescued the immediate actor-credit contract. This is a concrete new implementation failure, not a claim that it caused every historical negative result.

The new regression `test_terminal_service_penalty_reaches_actor_with_variable_earlier_costs` failed on v1: adding a -2 terminal penalty changed the actor advantages by exactly zero. It passes on v2: every corresponding advantage decreases by -2. The corrected learner uses the actual signed team advantages, the existing frozen physical reward scales, and the learned critic, without per-episode centering or rescaling. Gradient clipping and the other PPO settings remain unchanged. The old standalone `gather_actor_advantages(normalize=True)` helper is not used by this trainer and must not be substituted into its one-episode update path.

The stopped run preserves four completed policy episodes (9,600 s), a fifth interrupted episode charged conservatively as a full 2,400 s slot, all references, and a source snapshot. Its `summary.json` marks it stopped and forbids resuming the failed checkpoints. The interrupted run had reported at least 1,500 s before termination; the full allocation is charged so the missing tail cannot hide computation.

The defect was also reproduced on the recorded harmful 4107/2200 rollout,
using the exact archived v1 source and the repaired v2 source:
`runs/rllib/native_damping_v1_learning_check_o71/penalty_credit_diagnosis.json`.
Turning on the terminal penalty changes raw return by -2 in both versions.
Across all **32,729 actor samples**, the old actor-advantage change is exactly
zero; the repaired change is -2, up to float32 precision. This offline diagnosis
uses no new simulation and does not claim the penalty bug explains all traffic
harm or all historical project failures.

The repair restarts from the exact original initial weights. It may reuse the first policy rollout because that sample preceded every failed update; it verifies the saved generating actor/critic weights and action log probabilities before using it. It also reuses native references only if all physical source/scenario/SUMO identities and reference digests match. Four corrected updates comprise one verified reused rollout plus **three new full episodes (7,200 s)**, covering the same four cases. No trained v1 weights are resumed. Total policy simulation allocation stays at 12,000 + 7,200 = **19,200 s**, with one 2,400 s data sample reused for the new optimizer rather than simulated again.

Repaired run directory: `runs/rllib/native_damping_v2_credit_repair_o71/`. Its four updates completed with finite gradients, nonzero actor changes, and exact inference checkpoint round-trips. A completed-run `--resume` also succeeded without additional episodes. The final inference file is `inference_004.pt`, explicitly unaccepted. Repaired exploratory paired rewards are -0.1393, +0.0605, +0.1193 and -0.2283; corresponding system-time changes are +1.194, -0.294, -0.334 and +1.671 s/requested vehicle. Every request completes and no collision involvements or starting teleports are reported. These are learning/software results, not deterministic policy efficacy.

Four deterministic comparisons completed in `runs/eval/native_damping_v2_credit_repair_o71_development/`; their final checkpoint is the repaired fourth update, not any stopped v1 checkpoint. The evaluator reads only the actor, with no critic or native reference needed to choose its actions.

- 4107 / 2000: braking -13.87%, energy -10.53%, system time -1.316 s/vehicle, TTC exposure -89.5 vehicle-seconds.
- 4109 / 2000: braking +13.06%, energy +16.28%, system time +1.774 s/vehicle, TTC exposure +75.0 vehicle-seconds.
- 4107 / 2200: braking +9.11%, energy +11.01%, system time +4.490 s/vehicle, TTC exposure +127.0 vehicle-seconds; service fails.
- 4109 / 2200: braking +6.23%, energy +5.54%, system time +1.987 s/vehicle, TTC exposure +54.0 vehicle-seconds.

Only four corrected optimizer updates were made. This is enough to verify an
active learner and reject this particular prototype, not to conclude that PPO
has converged, local observations are insufficient, or trained damping is
impossible. Do not substitute the analytical controller for the required D*.

At completion of this short package, no experiment remained running. The later
full campaign is tracked in `NATIVE_DAMPING_FULL_CAMPAIGN_2026-09-05.md`.
This package's total policy simulation allocation stayed at
19,200 s (including the conservatively charged interruption); references cost
9,600 s, deterministic comparisons 9,600 s, and the collector diagnostic 900 s.
The earlier separate mechanical check adds its own 900 s; it was the preceding
implementation package, not hidden inside this check's 39,300 s allocation.

## Exact next work

1. Finish the full-training specification: audited training/development/final
   seed separation, the prespecified loading family, and a strong fixed or
   analytical comparison using the same sensing/actuation. Freeze checkpoint
   selection on physical outcomes. Do not choose it from training reward or
   start another P30/health-classifier observation screen.
2. Extend this corrected fixed-clock learner into the single bounded campaign
   in the restored route: three optimizer lineages, at most 250k global steps
   each, fixed development checks. Account for this check's spent computation;
   do not relabel reused samples as new simulated time. Four updates are not a
   completed training campaign, and the cap is not a guarantee of convergence.
3. Compare one selected frozen candidate on the untouched damping acceptance
   cohort, including braking/energy, service, safety, and harmful cases. No-op
   does not satisfy useful damping; improvement in one case does not satisfy
   robustness. Only an accepted standalone checkpoint becomes D*.
4. Only then resume the RSU wall-feasibility/benefit work from CAV reports and
   the paper's prospective integrated evaluation. No wall/predictor run was
   launched during this implementation package.

## Frozen purpose and budget

Answer whether an ongoing local policy receives usable full-episode credit, updates its behavior, exercises bounded commands, and saves/restores correctly. This check does not select the final checkpoint or prove generalization.

- Four already exposed cases: seeds 4107 and 4109, each at mainline 2000 and 2200 veh/h; ramp 400; independent fixed p20 CAV assignment.
- Four native reference episodes: 9,600 simulated seconds, cached once.
- Original allocation: eight policy episodes cycling those four cases, **19,200 global control steps**, under the proposed 20k first-check cap. The repair/reuse accounting above supersedes the uninterrupted eight-update execution. Each complete episode lasts exactly 2,400 s, with 0.5 s SUMO integration and 1 s actor decisions. CAV samples are counted separately.
- One optimizer seed (71), one frozen configuration, no hyperparameter search. The remaining 750k development budget is not launched automatically by this runner.
- Before training, a separate 300 s native/zero/stochastic collector integration diagnostic uses only the first exposed case. These three prefixes are never admitted into a training batch and cost another 900 simulated seconds.
- After the eight updates, evaluate the deterministic final checkpoint on these same four exposed cases, reusing the native references: another 9,600 simulated seconds. This distinguishes noisy training rollouts from deployed policy behavior. There is no checkpoint search or fresh-seed evaluation. Total allocated traffic time for the complete engineering package is 39,300 simulated seconds (19,200 training + 9,600 reference + 9,600 evaluation + 900 diagnostic).

The explicit request generator is a new, versioned fixture. It fixes desired departures, vehicle identities, IDM driver types, and CAV designations before either arm starts. Mainline demand holds through 1200 s, declines over three 100 s segments toward 1500 veh/h, then continues to 1800 s; ramp demand ends at 1800 s. Full service outcomes are observed to 2400 s. Drivers retain their assigned IDM type in all arms; CAV assignment does not replace them with normal-car parameters. This is not a numerical continuation of historical pulse-flow results.

## What is trained

The shared actor has two 64-unit tanh layers and receives only the 85-component causal local history defined in `native_damping.py`. It chooses an ongoing acceleration correction in [-0.5,+0.3] m/s². A separate training-only critic receives a 22-component corridor summary and cannot enter the actor's inference interface.

This implementation uses a small explicit **PyTorch PPO** learner rather than adapting the old RLlib per-vehicle terminal path. The relevant engineering requirement is verified full-clock team credit and optimizer behavior; passing this check would not certify the old RLlib pipeline.

Learning rate 1e-4, two PPO epochs per complete episode, 128 global times per minibatch, clip 0.2, gradient norm bound 0.5, and per-actor approximate-KL early stop 0.03 are fixed. Gamma and GAE lambda equal one for the undiscounted fixed-horizon team objective. Advantages are calculated on the team timeline before being gathered at active CAV decisions, with no per-episode mean subtraction in repaired schema v2. The surrogate sums CAV score contributions and averages over global times; adding CAVs does not multiply the physical reward. The policy distribution remains stochastic during collection; evaluation/inference uses its deterministic transformed mean.

## Reward and service accounting

For interval t, physical cost is:

`C_t = 0.5 B_t/B_ref + 0.5 E_t/E_ref + 0.1 S_t/N + 2 H_t`.

Here B is vehicle-seconds with acceleration below -3 m/s²; E is the integral of squared acceleration; S is requested-but-not-arrived vehicle-seconds, **including source waiting**; N is the fixed total requested population; and H counts newly reported colliding-vehicle involvements and starting teleports. B_ref and E_ref are the means of the four native episode totals with a numerical floor of one, frozen before the first policy update. These are normalization scales, not runtime actor observations.

The learning reward is `r_t = C_native,t - C_policy,t`, using the cached same-request native trajectory. The subtraction is an action-independent training baseline: it centers returns without requiring a native reference at deployment. The terminal reward loses another 2 if policy system time exceeds native by more than 2 s/requested vehicle or policy arrivals are more than one vehicle lower. These scalar penalties do not replace independent safety and service acceptance constraints.

Requested departure time is integrated exactly within each 0.5 s interval. SUMO insertion and arrival events are assigned to their reported tick boundary, consistently across arms. Requests still waiting or unfinished at 2400 s remain in the outcome accounting. Pair identities must match, and short prefixes cannot be treated as completed training episodes.

The collector also records closing TTC below 3 s, minimum observed closing TTC, hard events, command counts, fallback reasons, and measured command tracking. TTC instrumentation is privileged outcome measurement only. TraCI leader distances exclude the follower's minGap; the metric adds it back to obtain a bumper gap. It is a longitudinal closing-time surrogate, not a collision-probability estimate.

## Evidence retention and stopping rules

The manifest fingerprints source files, SUMO binary, network, explicit requests, controller schema, PPO settings, and scenario assignments. Saved rollout arrays preserve actor inputs, sampled actions, log probabilities, actor identities/time indices, critic observations, and values. JSON stores physical interval costs and summary accounting. Trainer checkpoints contain both networks, optimizers, random states, history, and the frozen identities. Inference weights are separately exportable and explicitly marked unaccepted.

A per-directory operating-system lock prevents concurrent writers. A completed rollout can be reused after restart. A started but incomplete simulation cannot be silently repeated: the runner stops for budget reconciliation. Resume requires the same code/scenario/optimizer identity. Checkpoint inference is checked against the in-memory actor after each update.

Stop for nonfinite values/gradients, mismatched population/scenario/clock, failed zero-action identity, no stochastic command authority, or broken save/restore. Report safety degradation rather than treating lower training cost as success. A 19,200-step check can establish working learning software but cannot decide that the chosen observations are sufficient or that damping works on its own. That requires the subsequent prespecified development campaign and untouched-seed acceptance.

## Results

Before the larger development campaign, distinguish useful control from useful
learning: compare the learned feedback with a strong fixed/analytical correction
using the same sensing and actuator. Nonzero commands engage a nominal IDM
approximation, whereas exact zero hands back to SUMO. Therefore an improvement
over native alone does not isolate the value of learned observation-dependent
feedback. This is a required design question for the next prespecified baseline
comparison, not permission to launch an extra sweep or relabel the present four
cases as a learning-necessity test.

Regression suite: 1,077 tests passed in 126.53 s before the diagnostic/training launch. After adding the deterministic comparison report and its service/safety regression test, the full suite passed **1,078 tests in 169.08 s** (concurrent training was running). All ten focused training tests passed.

After the penalty-credit repair, reference-budget guard, and JSON-roundtrip identity correction, the full suite passed **1,081 tests in 131.97 s** before repaired simulation started. The JSON correction addresses tuple/list representation after serialization; its failed preflight started no SUMO. Physical source fingerprints and the network hash were rechecked before deterministic evaluation.

Collector diagnostic artifact: `runs/probe/native_damping_collector_v1_s4107/summary.json`. Native and zero have identical interval costs; zero issues no ramps across 3,255 decisions. Untrained stochastic actions issue 2,981 ramps and no reported collisions or teleports. However, severe braking rises from 31.0 to 47.5 vehicle-seconds, acceleration energy from 2,509.4 to 3,076.1, and TTC<3 s exposure from 10.0 to 25.5 vehicle-seconds. This is a mechanics pass with a negative physical outcome, not a safety/benefit pass. Do not hide this observation behind a `passed` software field.

Stopped v1 artifact directory: `runs/rllib/native_damping_v1_learning_check_o71/`. Its updates are invalid as acceptance evidence because of the demonstrated actor-credit defect. Record completed repaired learning and deterministic comparisons before deciding whether to extend the budget.

The four frozen normalization references are B_ref=613.0 vehicle-seconds and E_ref=41,204.235963623185 acceleration-energy units (m²/s³). Native references complete their fixed requested populations of 1,150, 1,109, 1,232 and 1,185 vehicles. These normalization values remain fixed across updates, while each case uses its own fixed N.

Read-only credit audit: `tools/audit_native_damping_rollouts.py` reconstructs team returns and population/physical cost identities from saved samples without new SUMO. In the first episode, 26,321 CAV samples correspond to 2,400 global times, 22,143 ramps (84.13% of decisions), and 241 distinct CAVs. All 241 have nonzero later team return after their last observed decision; that later return is retained. Maximum credit reconstruction discrepancy is 7.45e-9. This verifies retention, not attribution of each later cost to that particular CAV.

Early training records are mixed: the first exploratory episode has paired reward -0.1393 and +1.194 s/requested vehicle, while the second has +0.0272 and -0.266 s/requested vehicle. Both complete all vehicles and have finite parameter updates and exact inference save/restore. Different cases and exploratory trajectories must not be presented as a learning improvement curve. Final deterministic development comparisons remain pending.
