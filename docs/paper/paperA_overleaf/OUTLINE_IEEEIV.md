# Paper A — IEEE IV outline for professor review

**Working title:** World-Model-Guided Exploration for Learning Multi-CAV Cooperation

**Authority and status — v2.** The owner's latest redesign supersedes the reward-credit method in v1: **Arm B is WM-guided exploration, SUMO-graded**. Cooperation is rare and pays late; the WM points CAVs at likely-winning joint moves, while real SUMO outcomes teach the policy and govern adaptive trust. Paper A is the main paper. Existing opportunity/ranking evidence is distinct from pending learned-policy success. Red markers are unresolved evidence or implementation details, never estimated results. <!-- Source: E:/agent-work/_sol/coopmwm_prompt.md:42–51 (MAIN17:12); E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md MAIN17:12,17:14; owner S-PAPERA-OUTLINE addendum17:14. -->

## Source key and writing rules

- **OWNER:** `E:/agent-work/_sol/coopmwm_prompt.md:24–51` and shared authority `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`, MAIN17:07, MAIN17:12, MAIN17:14. Latest redesign controls over earlier reward-credit instructions; design choices and panel counts are prospective, not measured performance.
- **PLAN:** `E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md`; **BANDIT:** `E:/agent-work/c4-coopbandit/docs/paper_results/COOPBANDIT.md`; **TEACH:** `E:/agent-work/c4-wmteach/docs/paper_results/WMTEACH.md`; **DISTILL:** `E:/agent-work/c4-distill-1101/docs/paper_results/DISTILL.md`.
- **MWM-DESIGN:** `E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM_DESIGN.md`; inspected design still describes the superseded reward-credit arm. Architecture and registered panel/budget lineage can be cited, but Arm B implementation must be rebound after redesign. **WMPLAN:** `E:/agent-work/c4-wmplan/docs/paper_results/WMPLAN.md`. **BIB:** `E:/agent-work/c4-papera/docs/paper/paperA_cooperation/{refs.bib,reference_verification.csv,RELATED_WORK.md}`. **SCOPE:** the same directory's `CLAIMS.md`, `THREATS.md`, and `EVIDENCE_SOURCES.md`.
- **RANK:** `E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md`; completed cached action-ranking diagnostics support expected-delay proposals, not live efficacy or exact synergy qualification.
- Every empirical number below carries an actual source path, directly or through an adjacent source comment. IDs, algebra, editorial dimensions/page budgets and proposed experimental choices are not empirical findings. All reported delay savings use **MAPPO minus comparator**, so positive is better; interaction is separately reported in veh-s/action. Confidence intervals retain the producer's sampling frame.

## Six-page allocation, including references

Editorial targets, not measured outcomes; figures/tables are included within their section allocations. The compiled outline is a review artifact, not a camera-ready submission.

- Title, abstract and index terms: **0.35 page**; I Introduction: **0.60**; II Related Work: **0.45**.
- III Problem Formulation: **0.65**; IV Method: **1.05**; V Experimental Setup: **0.55**.
- VI Results: **1.35**; VII Discussion/Limitations: **0.25**; VIII Conclusion: **0.10**; References: **0.65**. Total **6.00 pages**. <!-- Source: editorial allocation for owner six-page brief; not experimental evidence. -->

## Title, abstract draft and index terms

### Abstract — paragraph A1
- MAPPO-style multi-agent learning can produce competent individual CAV driving while missing rare cooperative opportunities. Walls, gap creation and metering may pay off minutes later; random exploration rarely finds the coordinated moves, and noisy shared returns make them harder to learn. This is a task-specific hypothesis, not a theorem that standard MARL cannot cooperate.
- We propose a physics-informed WM as an **exploration guide**: replay candidate joint moves and policy/baseline actions on identical sampled hidden futures, rank by expected delay saving, and execute promising proposals early. The model needs useful action rankings rather than exact future reconstruction; SUMO alone supplies learning reward and advantage.
- Existing exact-model planning saves **+648.407 [209.699,1173.616] veh-s/world at J600** and **+1583.821 [588.246,2765.414] at J900** on **36 worlds**, establishing headroom rather than learned-policy success. <!-- Source: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:22–27. -->
- Starting from MAPPO, use importance-corrected PPO and trust-weighted imitation; decay proposal trust adaptively from SUMO-realized advantage versus the policy's own actions. Evaluate against frozen MAPPO and same-budget continued training, and test persistence after trust reaches zero. <span style="color:red">[PENDING: S-COOPMWM learned gains, proposal advantage/trust traces and exact deployed synergy; insert outcome sentence only after complete evidence.]</span>

### Index Terms — paragraph A2
- Connected and automated vehicles; cooperative merging; multi-agent reinforcement learning; world models.
- Guided exploration; counterfactual evaluation; common random numbers; mixed-autonomy traffic.

## I. Introduction — 0.60 page

### I-P1 — why cooperation is the missing control resource
- Introduce mixed-traffic merging: an individually costly slowdown can create a gap, meter arrivals or stabilize a cross-lane wall; the downstream group benefit can emerge over minutes.
- Explain the exploration bottleneck: a useful joint slowdown is both rare and initially costly, so independent/random exploration has little chance of discovering its long-delayed team payoff. Shared-return noise compounds that difficulty; retain this as a tested hypothesis.
- Establish rarity using **45/300 helpful states = 15%, 95% CI [10,20.33]%** in the selected TRAIN corpus. Only **81** states were group-eligible, and sampling was deliberately oversampled: this is corpus prevalence, not an on-road event rate. <!-- Source: E:/agent-work/c4-coopbandit/docs/paper_results/COOPBANDIT.md:7–18. -->

### I-P2 — the central idea
- Use a physics-informed WM to answer which intervention is better under the same uncertain futures, rather than reconstructing the one future that happened.
- Pair candidate joint moves and policy/MAPPO comparisons with CRN to reduce background variation in predicted savings. Joint/member-alone/baseline twins separately measure excess-over-solos synergy. WM output changes exploration, while all reward/advantage comes from real SUMO.
- Warm-start from MAPPO, try WM-supported moves early, then reduce trust as the policy's own SUMO performance catches up. <span style="color:red">[PENDING: guided discovery versus random exploration, adaptive trust and cooperation persisting after beta→0 / S-COOPMWM.]</span>

### I-P3 — four contribution statements (C1–C4)
- **C1 — opportunity:** exact counterfactual synergy demonstrates that beneficial joint actions exist and a perfect-predictor planner improves on MAPPO; sparse helpful states explain a difficult learning target. Supported within the reported simulator/development panels. <!-- Sources: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:22–43,95–102; E:/agent-work/c4-coopbandit/docs/paper_results/COOPBANDIT.md:7–18. -->
- **C2 — guided exploration, honestly graded:** observable-start WM ranks joint proposals using paired hidden futures; SUMO-only PPO reward, behaviour-mixture correction and adaptive trust connect proposals to learning. Exact replay fidelity/noise at **60/120/300/600 s** remains diagnostic, not a training gate. Cached S-BDRANK rankings are delivered; <span style="color:red">[PENDING: proposal implementation freeze and S-COOPMWM fidelity/noise.]</span> <!-- Sources: E:/agent-work/_sol/coopmwm_prompt.md:42–51; E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md:24–46. -->
- **C3 — learned emergence:** compare MAPPO, same-budget MAPPO+training control A, MAPPO+WM-guided exploration B and exact-planner empirical upper reference; include S-WMPLAN separately. Report whether cooperation persists after trust→0. <span style="color:red">[PENDING: paired learned-policy gains, B−A attribution and proposal-free persistence.]</span>
- **C4 — mechanism discovery:** show learned walls, gap creation and metering with exact matched joint/solo/baseline twins, group benefit and synergy. <span style="color:red">[PENDING: independently replicated deployed-policy behaviours / S-COOPMWM, S-COOPBEHAV.]</span>

## II. Related Work — 0.45 page

### II-P1 — counterfactual credit and coalition interaction
- Cite difference-reward lineage (`wolpert_tumer2002`), COMA (`foerster_coma2018`) and difference-reward policy gradients (`castellini_drreinforce`): subtraction for multi-agent credit is established.
- Distinguish the group excess-over-solos estimand from marginal-in-team credit, COMA's critic baseline and a Shapley allocation (`han_shapley2022`). For larger groups, C_G aggregates beyond-singleton interactions; it is not an isolated highest-order effect.
- Cite `barnett_counterfactual2026` for contextual cooperation measurement. Here C_G is an evaluation/diagnostic estimand; the novelty target is traffic-specific long-horizon proposal-guided exploration, SUMO grading and trust withdrawal, not a new subtraction reward.

### II-P2 — model-based learning and the closest counterfactual WM work
- Discuss Dreamer-style imagination conceptually, with **[CITE NEEDED: Dreamer primary paper; no verified entry in inherited bibliography]**. Do not invent a BibTeX entry. Cite verified `chai_macd2024` (Multi-Agent Counterfactual Dreamer) as direct prior art for imagined counterfactual credit.
- Cite `liu_physics_wm2026` for physics-informed traffic MARL. Here the WM proposes exploration actions; actor transitions, reward and advantage come from real SUMO. This avoids directly optimizing imagined reward, while biased proposals can still waste real budget or distort learning.
- Cite `oefinger_validate2026` for action-response validity. Action-delay ranking motivates proposals; it does not certify synergy, calibrated win probability, PPO mixture correctness or live improvement.

### II-P3 — MAPPO, merging, platoon cooperation and wave damping
- Cite MAPPO (`yu_mappo2022`), mixed-traffic on-ramp MARL (`chen_merge2021`) and Flow (`wu_flow2022`) for baseline learning/control lineage.
- Cite CIRCLES (`lee_circles2024`) for open-road wave damping and AV coalition reward work (`han_shapley2022`) for cooperation incentives. Separate observed coordination from exactly measured excess-over-solos benefit.
- Discuss platoon/merge coordination without claiming a dedicated platooning benchmark or new verified paper. **[CITE NEEDED: dedicated platoon-cooperation reference if a specific mechanism/prior claim is added]**. Retain `cardei_coordination2026` for process-aware evaluation if space permits.

## III. Problem Formulation — 0.65 page

### III-P1 — partially observed multi-CAV control
- Define a variable-agent Dec-POMDP with latent traffic state x_t, transition law P, shared team reward, active CAV set, causal local histories h_i and RSU observations z_t. The policy may use only those declared observations.
- Hidden human behaviour and future arrivals are uncertainty, not actor features. A WM samples an observation-conditioned/fitted-prior future distribution q(ξ|h,z); it never receives the realized hidden future.
- State action ownership: residual speed/time-gap setpoints around frozen MAPPO commands, native lane behaviour and unchanged safety executor. Exact labels may use simulator replay state offline; deployment observations remain causal. <!-- Source: E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM_DESIGN.md:3–23. -->

### III-P2 — source-inclusive team objective and clocks
- Define J_H(B) = integral from t to t+H of [sum over on-road vehicles of max(0,1−v_j/v_free,j) + Q_src(τ)] dτ. Q_src counts requested vehicles waiting to enter; include future entrants and all traffic, not just CAVs. Units are veh-s; lower J is better.
- State v_free and source-request accounting as simulator scoring-contract quantities, not additional actor inputs. <span style="color:red">[PENDING: bind final evaluator's exact free-speed/reference and source-wait implementation / S-COOPMWM.]</span>
- Distinguish intervention duration W, replay lookahead H and episode scores. The existing planner's **J600/J900** are after a **300 s** warmup, while event credits use **H120** or a separately labelled **H300**. Never compare these as identical clocks. <!-- Source: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:14–15,22–30,73–84. -->

### III-P3 — cooperation as excess over member-alone effects
- Freeze group G, nonmember actions, baseline B, joint targets, intervention window and continuation. Define J_H(S) with only members S issuing their joint-proposed targets; all other members of G use B. Solos retain joint targets/context rather than replanning.
- Define D_H(S)=J_H(B)−J_H(S) and C_G,H=D_H(G)−Σ_i∈G D_H({i}) = Σ_i J_H({i})−J_H(G)−(|G|−1)J_H(B). Require both D_H(G)>0 and C_G,H>0 to call a case beneficially cooperative.
- C_G is neither intent nor fairness. Never substitute member-removal marginals J(G\{i})−J(G) into the solo sum. The earlier member-revert credit design is historical; the new arm has no WM reward. <span style="color:red">[PENDING: bind coalition-alone exact diagnostic/discovery exports and the declared baseline to the redesigned policy.]</span> <!-- Sources: E:/agent-work/c4-papera/docs/paper/paperA_cooperation/CLAIMS.md:M-DEF,M-BASE; E:/agent-work/_sol/coopmwm_prompt.md:42–51. -->

## IV. Method — 1.05 pages

### IV-P1 — MAPPO residual policy and observable coordination
- Initialize both learned arms from the same frozen speed-only MAPPO policy. A zero residual must reproduce MAPPO command execution, not merely its neural output.
- Add trainable residual speed/log-time-gap setpoints and an RSU broadcast latent derived only from causal sensed traffic. Retain native lane logic, actuator ownership and safety clamping.
- Report capacity, input and exploration matching across arms. The inherited implementation proposes a **four-dimensional** zero-initialized broadcast and **10 s** residual windows; carry these only after final freeze. <!-- Source: E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM_DESIGN.md:3–14. -->

### IV-P2 — rank likely-winning cooperative proposals
- On a registered random sample of decision windows, oversample cooperation-likely RSU-observable situations and include ordinary ones. FULL-v4 **22cc32f6**, inherited **K=8** paired-future option code and **H300** primary compare joint moves against the policy's own action / MAPPO. Propose only positive expected saving, ranked by predicted mean delay. <!-- Source: E:/agent-work/_sol/coopmwm_prompt.md:44–45; E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md:24–32. -->
- Options **O1 slow together / O2 wall / O3 yield gap / O4 longer gap** map to member setpoints through inherited executors; the residual actor is not thereby restricted to an option policy. Reuse identical hidden-driver/arrival draws, nonmember actions and continuation in each comparison; reject incomplete paired sets. <!-- Source: E:/agent-work/_sol/coopmwm_prompt.md:45,50. -->
- Write a* = argmax_a mean_k[J_WM(a_policy;ξ_k)−J_WM(a;ξ_k)] over available proposals, with an explicitly declared MAPPO reference when used. <span style="color:red">[PENDING: exact baseline-selection rule, sampling fraction, availability mapping and proposal likelihood/export freeze / S-COOPMWM.]</span>

### IV-P3 — replay fidelity and noise are diagnostics
- For branch costs X and Y, Var(X−Y)=Var(X)+Var(Y)−2Cov(X,Y); general synergy variance is aᵀΣa. Positive covariance under matched futures can cancel shared variation, but neither universal nor complete cancellation is guaranteed.
- Compare paired replay contrasts against independently resampled branches and team-return advantage on matched starts/groups/horizons. Distinguish conditional WM-estimated noise from actual PPO-advantage dispersion across real windows; those are different sampling distributions.
- Exact SUMO playback checks ranking, D and coalition-alone C_G at **60/120/300/600 s**, with signs, ties, scale and world uncertainty. A low-variance biased/zero contrast is not useful. The earlier fidelity-gated multi-horizon reward weighting and reward ramp are superseded: no diagnostic horizon gates training. <span style="color:red">[PENDING: S-COOPMWM fidelity/noise; any multi-horizon proposal weighting is an explicitly registered ablation.]</span> <!-- Sources: E:/agent-work/_sol/coopmwm_prompt.md:26–28,49; E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM_DESIGN.md:55–58. -->

### IV-P4 — SUMO-only PPO under a proposal mixture
- At sampled eligible windows execute the WM proposal with probability beta; otherwise execute the policy action. Elsewhere use the policy. Let μ_beta=(1−beta)π_old+beta q_WM be the actual joint behaviour law; clipped PPO uses π_theta/μ_beta under the correct common reference measure. All reward and advantage come from SUMO's source-inclusive team J; no WM delay or C_G is added. <!-- Source: E:/agent-work/_sol/coopmwm_prompt.md:46–47. -->
- Add imitation toward executed proposals with weight proportional to beta. <span style="color:red">[PENDING: proposal distribution/likelihood including deterministic atoms, joint-density correction after masks/executor mapping, PPO loss normalization and imitation coefficient / S-COOPMWM.]</span> A Gaussian density evaluated at a deterministic proposal is not its mixture probability; importance correction must match the actual behaviour law.
- Exact full-prefix playback verifies parity, branch identity, shared exogenous coupling and source bookkeeping. It supplies diagnostic labels, not model-based reward or actor features. Compare with A using the same policy capacity, causal inputs and real-SUMO budget; guided proposal injection is the treatment.

### IV-P5 — adaptive trust and cooperation after withdrawal
- Beta starts at **1** on sampled proposal windows and decays adaptively, flooring at **0**, as the policy's own actions match/beat WM proposals in SUMO. Use critic-based running estimates with confidence intervals of realized proposal-versus-own-action advantage. This is an observed training diagnostic, not an exact causal twin contrast. <!-- Source: E:/agent-work/_sol/coopmwm_prompt.md:48–50. -->
- <span style="color:red">[PENDING: pre-registered beta update rule, CI estimator, sampling/selection correction, imitation scaling and redesigned implementation receipt / S-COOPMWM.]</span> At chance replay fidelity, report the diagnostic; it does not stop B. SUMO evidence governs trust, with its speed of decay to be measured.
- Report beta, proposal availability/acceptance/execution, option types and realized advantage over updates; assess exact learned cooperation when beta=0. <span style="color:red">[PENDING: whether trust reaches zero, policy-only evaluation and persistence; never equate continued proposal execution with autonomous cooperation.]</span>

## V. Experimental Setup — 0.55 page

### V-P1 — common plant, panels and comparators
- Describe the preserved merge network, human/CAV population, demand and penetration cells, causal sensing, warmup, safety/lanes, full-population/source delay and independent-world pairing. Bind exact simulator/config/policy hashes in the artifact; cite verified SUMO (`lopez_sumo2018`).
- Main rows: frozen MAPPO; MAPPO+same-budget continued training A (same architecture/inputs/base exploration); MAPPO+WM-guided exploration B; S-WMPLAN online planner; exact SUMO planner empirical upper reference. Native is a registered final-test reference. The finite-option exact planner is not a certified optimum.
- Existing exact-planner development worlds differ from the learned-policy/WMPLAN panel. Its numbers are contextual, unpaired until a common-panel comparison is delivered. WMPLAN has resource-limited early intervention eligibility; expose refusals rather than treating it as unrestricted planning. <!-- Sources: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:11–27,62–65; E:/agent-work/c4-wmplan/docs/paper_results/WMPLAN.md:25–43,95–121. -->

### V-P2 — outcomes, uncertainty and fair budgets
- Report paired source-inclusive J, D_G, sum of solos, C_G, helpful/eligible proportions, braking/TTC, collisions/teleports, ramp/mainline service and execution-clamp shares; retain null and adverse cases.
- Cluster uncertainty at world level, preserving all events and inactive worlds; event means with no-event resamples are undefined, not zero. Report original, extension and pooled planner cohorts without representing the adaptive extension as pristine confirmation.
- Match added real-SUMO budget, policy updates, capacity and observable inputs; separately disclose WM fitting, proposal compute, exact diagnostics and MAPPO pretraining. <span style="color:red">[PENDING: redesigned source/configuration freeze, proposal GPU quote and complete comparator export / S-COOPMWM.]</span>

### V-P3 — registered final panel and primary-first replication
- Final-test seed block **989600–989635**, **36** worlds, **3** per **12** demand×share cells; shares **.05/.10/.20/.40**, demand **1900/2400/2700**. Final-test outcomes remain sealed; only public registration metadata is used here. **300 s warmup + 900 s control**, J600/J900. <!-- Source: E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md S-COOPMWM EXACT PLAN16:46/17:07 and AMENDMENT17:07; E:/agent-work/_sol/coopmwm_prompt.md:30–39. -->
- Primary training seed **20261101**, then replications **20261102/20261103**; fixed **update-24** checkpoints for A/B. Once both primary checkpoints exist, evaluate MAPPO/native/A/B once; later evaluate replication A/B on the same panel, reusing MAPPO/native. No test-based checkpoint selection, tuning or reruns. <!-- Source: same shared REGISTRATION.md EXACT PLAN and AMENDMENT17:07; E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM_DESIGN.md:43,60–68. -->
- Paired whole-world **95%** bootstrap CIs, **10,000** resamples, RNG **20261101**, for B−MAPPO, B−A and A−MAPPO J600/J900; define signs (cost difference versus positive saving), per-cell safety descriptives and training-seed dispersion. Exact B discovery uses detector-first complete twins. Total cap **6,000,000 sim-s**; primary panel **172,800**, replication panels **172,800**, amended allocated **5,738,400**, unallocated **261,600**. These are registered budgets, not consumed compute or completed results. <!-- Source: same shared REGISTRATION.md AMENDMENT17:07; E:/agent-work/_sol/coopmwm_prompt.md:37–40,51. -->

## VI. Results — 1.35 pages, E1–E5

### VI-A / E1 — opportunity: does MAPPO leave cooperation on the table?
- Exact planner: **36-world J600 +648.407 [209.699,1173.616]**, **J900 +1583.821 [588.246,2765.414] veh-s/world**; **95%**, **10,000** whole-world bootstrap draws, seed **20261101**. Preserve original/extension strata in artifact. <!-- Source: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:22–27. -->
- All **52** executed group actions have **104** solo twins; **24/52** have C_G>0. Pooled mean **+21.771 [−42.074,+90.467] veh-s/action** does not establish positive population-mean synergy. Selected joint-benefit/solo-harm case: **D_G +568.333707**, solos **−302.399975/−18.034632**, **C_G +888.768314**, at **H120**. <!-- Source: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:29–40. -->
- Longer-lookahead wall: **H300 D_G +13.851684**, solos **−205.778376/+38.059435**, **C_G +181.570625**. Counterexample wall: **D_G +2442.101011** but **C_G −428.858726**; helpful is not automatically superadditive. Different panels/budget pruning prevent a causal horizon comparison. <!-- Source: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:95–105. -->
- Helpful-option prevalence: **45/300 = 15% [10,20.33]%** in the oversampled TRAIN corpus. BANDIT always falls back to MAPPO; DISTILL has **0/45** helpful-teacher agreement and does not establish MAPPO superiority. These motivate guided discovery followed by real-outcome learning, rather than assuming easy option imitation. <!-- Sources: E:/agent-work/c4-coopbandit/docs/paper_results/COOPBANDIT.md:7–18,35–42; E:/agent-work/c4-distill-1101/docs/paper_results/DISTILL.md:12–21,31–40. -->

### VI-B / E2 — fidelity, ranking and noise reduction
- Report exact-vs-WM action order, C_G sign/rank, regret, ties, coverage and world CIs at **60/120/300/600 s**. <span style="color:red">[PENDING: S-COOPMWM proposal fidelity and coalition-alone synergy diagnostics.]</span> This diagnoses proposal quality; no horizon qualification gates B. <!-- Source: E:/agent-work/_sol/coopmwm_prompt.md:26–28,49. -->
- Compare team-return advantage variance, independent-future replay variance and CRN replay variance under matched starts/horizons, distinguishing conditional model noise from empirical real-advantage dispersion. <span style="color:red">[PENDING: variance ratios and exact signal preservation / S-COOPMWM.]</span>
- Existing FULL-v4 teacher diagnostic: helpful **4/8** exact-argmax matches at **H300**; all-state exact chosen saving **+680.3 veh-s/state** on **12** selected states; within-state delay rank **0.516**. The helpful-state mean is a different denominator and must not be labelled +680.3. This small roster-only, source-prior/cold-memory option diagnostic does not qualify continuous residual C_G. <!-- Source: E:/agent-work/c4-wmteach/docs/paper_results/WMTEACH.md:80–90,132–136. -->
- S-BDRANK: available-option delay-pair concordance **65.5/75.9% (v4)** versus **63.8/51.7% (v5)** at **120/300 s**, **58 pairs per horizon/version**. Breakdown probability saturates: ties on **112/120** development controller pairs at **600 s**; do not transfer that denominator to option pairs. Mean delay motivates ranking; development-selected rules still lose on the reserved panel. No live efficacy or synergy qualification is implied. <!-- Source: E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md:24–46,48–55. -->

### VI-C / E3 — main comparison: does cooperation emerge from MAPPO?
- T1 main rows: **MAPPO / MAPPO+training control A / MAPPO+WM-guided exploration B / S-WMPLAN / exact planner empirical upper reference**, with native reference. Columns: paired J savings, exact C_G and D_G/solos, moves/coverage, safety, real/model compute and trust/proposal use at evaluation.
- <span style="color:red">[PENDING: S-COOPMWM final-panel outcomes, B−A attribution, primary + replication results, beta/acceptance/realized-advantage trajectories and beta=0 persistence.]</span> Test improvement over matched continued training, not merely frozen MAPPO. If beta remains positive, distinguish assisted performance from proposal-free evaluation.
- <span style="color:red">[PENDING: S-WMPLAN complete paired efficacy, exact twins and resource-refusal counts.]</span> Existing exact-planner values occupy a separately marked different-panel reference row; do not paste them into missing learned-arm cells.

### VI-D / E4 — discovered behaviours with exact twins
- F4 panels should show a learned wall, gap creation and metering episode: sensed geometry, member actions and resulting merge/wave change, each with same-prefix baseline, joint and each-alone traces.
- Attach D_G, each solo saving and C_G at the declared clock; identify beneficial cases, retain adverse contrasts, report detector exposure/eligibility and avoid outcome-selected prevalence claims.
- <span style="color:red">[PENDING: WM-guided learner behaviour examples, option-type/exposure counts, independent replication and exact twins after beta→0 / S-COOPMWM, S-COOPBEHAV.]</span> Exact-planner walls support opportunity, not discovery by the learned policy.

### VI-E / E5 — ablations
- Compare FULL-v4 versus FULL-v5 for matched proposal tasks, horizons and seeds; published teacher-option diagnostics are not a learned-policy ablation. <span style="color:red">[PENDING: separately registered matched model-version ablation / S-COOPMWM.]</span>
- Compare proposal horizon, any registered multi-horizon weighting, adaptive versus fixed/zero trust, imitation weight and CRN versus independent futures under matched real/model budgets. The original credit-weight/ramp ablation is superseded. <span style="color:red">[PENDING: proposal-horizon, trust, imitation and pairing ablations; no automatic extra runs authorized.]</span>
- Record sign/rank disagreement, ties, proposal refusals, safety-erased commands and training drift. A is the no-WM-exploration control; proposal-free evaluation isolates persistence from teacher assistance.

## VII. Discussion / Limitations — 0.25 page

### VII-P1 — what the evidence can support
- Ranking is sufficient only for the decision being made; preserving delay order does not preserve coalition synergy or calibrated win probabilities. CRN can cancel shared noise but leaves model bias and missing-observation uncertainty.
- Source-prior/cold-memory replay differs from fully observed exact playback. Freeze interventions/backgrounds/continuations, distinguish member-alone and member-removal estimands, and report realized motion after safety caps.
- Finite-option oracle planning is an empirical upper reference under privileged prediction, not an optimum or deployable policy. Existing cohorts differ; rare selected examples do not establish population-level cooperation.

### VII-P2 — attribution and generalization
- A positive C_G with negative D_G is not beneficial cooperation. SUMO-only grading avoids direct exploitation of WM reward, but biased proposal sampling, critic estimates, imitation and incorrect mixture correction can still mislead learning. Match capacity/input/budget and retain safety/service outcomes.
- Simulator-only evidence, one merge configuration, limited observation/communication assumptions, finite training seeds and development reuse limit real-road inference. <span style="color:red">[PENDING: final reproducibility/freeze and seed-variability evidence.]</span>

## VIII. Conclusion — 0.10 page

### VIII-P1 — close the intended argument, conditioned on results
- Existing exact replays establish rare useful joint interventions and planner headroom. WM-guided exploration aims to find them more often; SUMO-only grading and adaptive trust aim to turn guidance into autonomous cooperation from MAPPO.
- <span style="color:red">[PENDING: claim emergence and gains over both MAPPO and continued training only with complete E2–E4 evidence, and claim persistence only after proposal-free evaluation; otherwise report the measured boundary and negative outcome.]</span>

## Figure and table plan (included in page allocations)

### F1 — method schematic, Section IV
- Show real SUMO → causal histories → MAPPO residual actor; a WM path compares available joint proposals using paired futures and expected delay. A beta-controlled action mixture feeds SUMO; SUMO reward/advantage feeds importance-corrected PPO and realized proposal advantage updates beta.
- Show beta-weighted imitation, the proposal-free persistence check and exact twin diagnostics. Draw no WM-to-reward arrow. Symbolic schematic only; empirical annotation stays pending.

### F2 — opportunity, Section VI-A
- Panel (a): exact-planner J600/J900 saving with source CIs, original/extension/pooled cohort labels. Panel (b): all-action synergy distribution and helpful-state prevalence with its selected-corpus denominator.
- Show positive and adverse wall exemplars compactly; caption distinguishes episode savings from event synergy. <!-- Source: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:22–43,95–102; E:/agent-work/c4-coopbandit/docs/paper_results/COOPBANDIT.md:7–18. -->

### F3 — proposal fidelity/noise and trust, Sections VI-B/C
- Aligned horizons: exact-vs-WM action/C_G sign/rank/regret, paired versus unpaired noise and team-advantage variance; no training-admission boundary. Add update traces of beta, proposal acceptance and SUMO-realized advantage.
- Cached option ranks are available from `BDRANK.md`; <span style="color:red">[PENDING: S-COOPMWM exact diagnostics and trust traces; identify uncertainty, sampling and normalization.]</span> <!-- Source: E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md:24–32. -->

### F4 — learned behaviour twins, Section VI-D
- Compact space–time/action traces for wall, gap and metering, with baseline/joint/member-alone branches and exact D_G/solos/C_G beneath each case.
- <span style="color:red">[PENDING: policy identities, trace manifests, clocks, exposure, trust at execution and independently replicated proposal-free examples.]</span>

### T1 — main control/planning comparison, Section VI-C
- Rows/columns as E3; visually separate on-policy learned rows from planning references, and mark different-panel exact reference. Include paired contrast to continued training, coverage and safety rather than only headline means.
- <span style="color:red">[PENDING: common-panel learned and WM-planning data; no placeholder numerical entries.]</span>

### T2 — ablations, Section VI-E
- Rows: proposal model version, horizon/weighting, adaptive/fixed/zero trust, imitation weight and paired-future control. Columns: matched budget, fidelity/variance, J gain, exact cooperation/persistence and safety.
- <span style="color:red">[PENDING: S-COOPMWM ablation exports; preserve adverse/null runs.]</span>

## Professor-review package and handoff

- `main.tex` mirrors this structure, contains red pending/citation markers, number-source comments and empty framed figure/table placeholders. `refs.bib` uses only inherited verified keys; the inherited verification CSV is authoritative for identity.
- Critical insertion slots are E2–E5 evidence, the redesigned behaviour-mixture likelihood and adaptive-trust rule, coalition-alone diagnostic exports, matched controls and proposal-free persistence. Earlier reward-credit design remains historical. Desktop `PaperA_outline_v1.pdf` is preserved; this package builds `PaperA_outline_v2.pdf`.
