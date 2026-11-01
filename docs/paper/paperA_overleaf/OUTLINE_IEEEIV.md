# Paper A — IEEE IV outline for professor review

**Working title:** Learning to Cooperate from Imagined Counterfactuals: World-Model Replay Credit for Multi-CAV Merge Control

**Authority and status.** The owner's 2026-11-01 16:40 brief supersedes the older teacher–student outline in `../paperA_cooperation/OUTLINE.md`. Paper A is the main paper. The story below is the intended submission argument; existing opportunity evidence is distinguished from the still-pending demonstration of cooperation learned through WM replay. Red markers are unresolved evidence or implementation details, never estimated results. <!-- Source: owner S-PAPERA-OUTLINE brief; E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md, MAIN16:22–16:40. -->

## Source key and writing rules

- **OWNER:** this task's approved framing and the shared authority at `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`, MAIN16:22, MAIN16:25, MAIN16:38, MAIN16:40. Design choices/horizons from this authority are prospective, not measured performance.
- **PLAN:** `E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md`; **BANDIT:** `E:/agent-work/c4-coopbandit/docs/paper_results/COOPBANDIT.md`; **TEACH:** `E:/agent-work/c4-wmteach/docs/paper_results/WMTEACH.md`; **DISTILL:** `E:/agent-work/c4-distill-1101/docs/paper_results/DISTILL.md`.
- **MWM-DESIGN:** `E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM_DESIGN.md`; **WMPLAN:** `E:/agent-work/c4-wmplan/docs/paper_results/WMPLAN.md`. Current MWM design is an earlier speed amendment and does not yet encode the owner's latest horizon instruction. **BIB:** `E:/agent-work/c4-papera/docs/paper/paperA_cooperation/{refs.bib,reference_verification.csv,RELATED_WORK.md}`. **SCOPE:** the same directory's `CLAIMS.md`, `THREATS.md`, and `EVIDENCE_SOURCES.md`.
- **RANK (late delivery):** `E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md`; completed cached action-ranking diagnostics arrived during drafting. They update the brief's pending S-BDRANK status, but do not supply residual-policy cooperation-credit validation.
- Every empirical number below carries an actual source path, directly or through an adjacent source comment. IDs, algebra, editorial dimensions/page budgets and proposed experimental choices are not empirical findings. All reported delay savings use **MAPPO minus comparator**, so positive is better; interaction is separately reported in veh-s/action. Confidence intervals retain the producer's sampling frame.

## Six-page allocation, including references

Editorial targets, not measured outcomes; figures/tables are included within their section allocations. The compiled outline is a review artifact, not a camera-ready submission.

- Title, abstract and index terms: **0.35 page**; I Introduction: **0.60**; II Related Work: **0.45**.
- III Problem Formulation: **0.65**; IV Method: **1.05**; V Experimental Setup: **0.55**.
- VI Results: **1.35**; VII Discussion/Limitations: **0.25**; VIII Conclusion: **0.10**; References: **0.65**. Total **6.00 pages**. <!-- Source: editorial allocation for owner six-page brief; not experimental evidence. -->

## Title, abstract draft and index terms

### Abstract — paragraph A1
- MAPPO-style multi-agent learning can produce competent individual CAV driving while missing rare cooperative opportunities. Walls, gap creation and metering can pay off minutes later, when arrivals, human drivers and other CAVs obscure their credit in a shared team return. This is the paper's motivation, not a theorem that standard MARL cannot cooperate.
- We propose a physics-informed world model as a **credit evaluator**: starting from observable state, replay group, member-alone and baseline interventions under identical sampled hidden futures. The model must preserve useful action/credit rankings; matching every realized future is unnecessary. Common random numbers cancel shared variation to the extent that paired outcomes covary.
- Existing exact-model planning saves **+648.407 [209.699,1173.616] veh-s/world at J600** and **+1583.821 [588.246,2765.414] at J900** on **36 worlds**, establishing headroom rather than learned-policy success. <!-- Source: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:22–27. -->
- Starting from MAPPO, ramp in replay cooperation credit and evaluate against both frozen MAPPO and matched continued training. <span style="color:red">[PENDING: S-COOPMWM ranking/sign qualification, noise reduction, learned delay gains and exact deployed synergy; insert outcome sentence only after complete evidence.]</span> The intended conclusion is emergent cooperation; it is not yet a supported result.

### Index Terms — paragraph A2
- Connected and automated vehicles; cooperative merging; multi-agent reinforcement learning; world models.
- Counterfactual credit assignment; common random numbers; mixed-autonomy traffic.

## I. Introduction — 0.60 page

### I-P1 — why cooperation is the missing control resource
- Introduce mixed-traffic merging: an individually costly slowdown can create a gap, meter arrivals or stabilize a cross-lane wall; the downstream group benefit can emerge over minutes.
- Explain the long-horizon assignment problem: a shared traffic reward mixes a CAV's contribution with hidden drivers, arrivals and other agents. Good individual control need not discover rare joint interventions; avoid a blanket claim about all MARL.
- Establish rarity using **45/300 helpful states = 15%, 95% CI [10,20.33]%** in the selected TRAIN corpus. Only **81** states were group-eligible, and sampling was deliberately oversampled: this is corpus prevalence, not an on-road event rate. <!-- Source: E:/agent-work/c4-coopbandit/docs/paper_results/COOPBANDIT.md:7–18. -->

### I-P2 — the central idea
- Use a physics-informed WM to answer which intervention is better under the same uncertain futures, rather than reconstructing the one future that happened.
- Pair joint, each-alone and baseline replays with common random numbers (CRN); subtract shared background variation and measure excess-over-solos synergy. The actor still collects real SUMO trajectories; the WM supplies auxiliary training credit.
- Warm-start from MAPPO and ramp the new credit to retain individual competence while making coordinated long-horizon effects learnable. <span style="color:red">[PENDING: measured cancellation, action-dependent WM bias and learned emergence / S-COOPMWM.]</span>

### I-P3 — four contribution statements (C1–C4)
- **C1 — opportunity:** exact counterfactual synergy demonstrates that beneficial joint actions exist and a perfect-predictor planner improves on MAPPO; sparse helpful states explain a difficult learning target. Supported within the reported simulator/development panels. <!-- Sources: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:22–43,95–102; E:/agent-work/c4-coopbandit/docs/paper_results/COOPBANDIT.md:7–18. -->
- **C2 — replay credit:** observable-start physics-informed WM, paired hidden futures and multi-horizon credit; check ranking/sign against exact SUMO playback at **60/120/300/600 s**, and quantify variance versus team-return advantage. Cached S-BDRANK action rankings are delivered; <span style="color:red">[PENDING: S-COOPMWM residual-credit fidelity/noise.]</span> <!-- Sources: owner brief; shared REGISTRATION.md MAIN16:38; E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md:24–46. -->
- **C3 — learned emergence:** compare MAPPO, same-budget MAPPO+training control, MAPPO+WM credit and exact-planner empirical upper reference; include S-WMPLAN as a separate planning row. <span style="color:red">[PENDING: learned-policy and WM-guided-planner paired outcomes.]</span>
- **C4 — mechanism discovery:** show learned walls, gap creation and metering with exact matched joint/solo/baseline twins, group benefit and synergy. <span style="color:red">[PENDING: independently replicated deployed-policy behaviours / S-COOPMWM, S-COOPBEHAV.]</span>

## II. Related Work — 0.45 page

### II-P1 — counterfactual credit and coalition interaction
- Cite difference-reward lineage (`wolpert_tumer2002`), COMA (`foerster_coma2018`) and difference-reward policy gradients (`castellini_drreinforce`): subtraction for multi-agent credit is established.
- Distinguish the group excess-over-solos estimand from marginal-in-team credit, COMA's critic baseline and a Shapley allocation (`han_shapley2022`). For larger groups, C_G aggregates beyond-singleton interactions; it is not an isolated highest-order effect.
- Cite `barnett_counterfactual2026` for contextual cooperation measurement. The novelty target is the traffic-specific, long-horizon, CRN-paired replay-to-learning pipeline and its validation, not the subtraction equation.

### II-P2 — model-based learning and the closest counterfactual WM work
- Discuss Dreamer-style imagination conceptually, with **[CITE NEEDED: Dreamer primary paper; no verified entry in inherited bibliography]**. Do not invent a BibTeX entry. Cite verified `chai_macd2024` (Multi-Agent Counterfactual Dreamer) as direct prior art for imagined counterfactual credit.
- Cite `liu_physics_wm2026` for physics-informed traffic MARL. Here the actor executes in real SUMO and WM output supplies credit, reducing exposure to model-generated training transitions; biased credit can still be exploited.
- Cite `oefinger_validate2026` for checking action-response validity. Ranking sufficiency is decision-specific; action-delay ranking alone does not certify synergy subtraction or confidence calibration.

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
- C_G is neither intent nor fairness. Never substitute member-removal marginals J(G\{i})−J(G) into the solo sum. <span style="color:red">[PENDING: producer reconciliation—the inspected MWM-DESIGN uses member-revert marginal credits; freeze coalition-alone semantics or explicitly label a different estimand before importing results.]</span> <!-- Sources: E:/agent-work/c4-papera/docs/paper/paperA_cooperation/CLAIMS.md:M-DEF,M-BASE; E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM_DESIGN.md:21–28. -->

## IV. Method — 1.05 pages

### IV-P1 — MAPPO residual policy and observable coordination
- Initialize both learned arms from the same frozen speed-only MAPPO policy. A zero residual must reproduce MAPPO command execution, not merely its neural output.
- Add trainable residual speed/log-time-gap setpoints and an RSU broadcast latent derived only from causal sensed traffic. Retain native lane logic, actuator ownership and safety clamping.
- Report capacity, input and exploration matching across arms. The inherited implementation proposes a **four-dimensional** zero-initialized broadcast and **10 s** residual windows; carry these only after final freeze. <!-- Source: E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM_DESIGN.md:3–14. -->

### IV-P2 — observable-start paired WM replay
- At a real SUMO decision, freeze the observable start, selected group and candidate targets. Sample complete hidden-driver/arrival futures ξ_k from the fitted population/prior; the inherited backend uses **K=8** paired draws. <!-- Source: E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM_DESIGN.md:15–23. -->
- Reuse each identity-bound future in joint, each member-alone and all-baseline branches, including the same nonmember commands and continuation. Reject incomplete future sets rather than dropping adverse branches.
- Compute mean imagined J, D and C_G from these paired branches. A monotone but nonlinear transformation can preserve action ranking while changing synergy; therefore check both action ranking and exact C_G sign/rank, not only absolute delay fit.

### IV-P3 — why CRN can reduce credit noise
- For branch costs X and Y, Var(X−Y)=Var(X)+Var(Y)−2Cov(X,Y); general synergy variance is aᵀΣa. Positive covariance under matched futures can cancel shared variation, but neither universal nor complete cancellation is guaranteed.
- Compare paired replay credit against independently resampled replay branches and the real team-return advantage on the same starts, groups and horizons. Report raw variance, standardized signal-to-noise and variance ratios with world-level uncertainty.
- A small-variance biased or identically zero estimator is not useful credit. Include sign agreement, ties, execution coverage and exact effects beside noise statistics. <span style="color:red">[PENDING: numerical variance reduction and effective signal / S-COOPMWM.]</span>

### IV-P4 — fidelity-gated multi-horizon credit
- Candidate horizons are **60/120/300/600 s**. Owner rule: admit a horizon only if the **95%** exact-vs-WM sign-agreement interval has lower bound **>0.5**; weight qualified horizons by measured agreement and use **300 s** as primary if qualified. <!-- Source: shared REGISTRATION.md MAIN16:38; owner task. -->
- Write c̄ = Σ_H α_H ĉ_H, α_H≥0, Σ_H α_H=1 over qualified horizons. <span style="color:red">[PENDING: frozen agreement-to-weight mapping, tie handling, qualification sample size and no-qualified-horizon fallback / S-COOPMWM.]</span> Do not choose horizons by downstream DEV gain.
- Exact SUMO full-prefix playback checks no-op/trajectory parity, joint and singleton branch identity, source bookkeeping and shared exogenous coupling. Gate on sampled TRAIN windows; exact costs are verifier/calibration data, not optimization reward labels.

### IV-P5 — ramp cooperation into MAPPO learning
- Optimize PPO/MAPPO-style real-trajectory learning with original team-return learning plus an auxiliary WM replay credit. A proposed aggregate form is w_ind Σ_i D̂({i}) + w_coop Ĉ_G, with 0≤w_coop≤w_ind; algebraically this preserves nonnegative expanded singleton weight. This is a design guard, not a safety or efficacy guarantee.
- Ramp the auxiliary credit from zero while keeping a matched no-credit continuation arm. <span style="color:red">[PENDING: final per-CAV allocation, PPO score/advantage insertion, normalization, scale and ramp under coalition-alone C_G / S-COOPMWM.]</span> The earlier producer ramp is update-based; do not silently claim an implemented allocation.
- Policies act and gather outcomes in real SUMO; WM replay provides credit only. This reduces direct transition-model exploitation but does not eliminate exploitation of action-dependent credit bias. F1 should show this boundary and exact playback as an external check.

## V. Experimental Setup — 0.55 page

### V-P1 — common plant, panels and comparators
- Describe the preserved merge network, human/CAV population, demand and penetration cells, causal sensing, warmup, safety/lanes, full-population/source delay and independent-world pairing. Bind exact simulator/config/policy hashes in the artifact; cite verified SUMO (`lopez_sumo2018`).
- Main rows: frozen MAPPO; MAPPO+same-budget continued training (same architecture/inputs/exploration); MAPPO+WM replay credit; S-WMPLAN world-model-guided online planner; exact SUMO planner empirical upper reference. The planner is a finite-option privileged-information reference, not a certified optimum.
- Existing exact-planner development worlds differ from the learned-policy/WMPLAN panel. Its numbers are contextual, unpaired until a common-panel comparison is delivered. WMPLAN has resource-limited early intervention eligibility; expose refusals rather than treating it as unrestricted planning. <!-- Sources: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:11–27,62–65; E:/agent-work/c4-wmplan/docs/paper_results/WMPLAN.md:25–43,95–121. -->

### V-P2 — outcomes, uncertainty and fair budgets
- Report paired source-inclusive J, D_G, sum of solos, C_G, helpful/eligible proportions, braking/TTC, collisions/teleports, ramp/mainline service and execution-clamp shares; retain null and adverse cases.
- Cluster uncertainty at world level, preserving all events and inactive worlds; event means with no-event resamples are undefined, not zero. Report original, extension and pooled planner cohorts without representing the adaptive extension as pristine confirmation.
- Match added real-SUMO budget, policy updates, capacity and observable inputs; separately disclose WM fitting, replay compute, exact calibration and MAPPO pretraining. <span style="color:red">[PENDING: final seed/checkpoint/exposure/budget lineage, seed replication and complete comparator export / S-COOPMWM.]</span>

## VI. Results — 1.35 pages, E1–E5

### VI-A / E1 — opportunity: does MAPPO leave cooperation on the table?
- Exact planner: **36-world J600 +648.407 [209.699,1173.616]**, **J900 +1583.821 [588.246,2765.414] veh-s/world**; **95%**, **10,000** whole-world bootstrap draws, seed **20261101**. Preserve original/extension strata in artifact. <!-- Source: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:22–27. -->
- All **52** executed group actions have **104** solo twins; **24/52** have C_G>0. Pooled mean **+21.771 [−42.074,+90.467] veh-s/action** does not establish positive population-mean synergy. Selected joint-benefit/solo-harm case: **D_G +568.333707**, solos **−302.399975/−18.034632**, **C_G +888.768314**, at **H120**. <!-- Source: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:29–40. -->
- Longer-lookahead wall: **H300 D_G +13.851684**, solos **−205.778376/+38.059435**, **C_G +181.570625**. Counterexample wall: **D_G +2442.101011** but **C_G −428.858726**; helpful is not automatically superadditive. Different panels/budget pruning prevent a causal horizon comparison. <!-- Source: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:95–105. -->
- Helpful-option prevalence: **45/300 = 15% [10,20.33]%** in the oversampled TRAIN corpus. BANDIT always falls back to MAPPO; DISTILL has **0/45** helpful-teacher agreement and does not establish MAPPO superiority. These motivate a rare, long-horizon credit mechanism rather than easy option imitation. <!-- Sources: E:/agent-work/c4-coopbandit/docs/paper_results/COOPBANDIT.md:7–18,35–42; E:/agent-work/c4-distill-1101/docs/paper_results/DISTILL.md:12–21,31–40. -->

### VI-B / E2 — fidelity, ranking and noise reduction
- Report exact-vs-WM action order, C_G sign/rank, regret, ties, coverage and world CIs at **60/120/300/600 s**. <span style="color:red">[PENDING: S-COOPMWM residual-credit fidelity.]</span> Cached S-BDRANK rankings below are a separate completed diagnostic. <!-- Source: owner brief; shared REGISTRATION.md MAIN16:32,MAIN16:38. -->
- Compare team-return advantage variance, independent-future replay variance and CRN replay variance under matched starts/horizons. <span style="color:red">[PENDING: variance ratios, exact signal preservation and admitted horizon weights / S-COOPMWM.]</span>
- Existing FULL-v4 teacher diagnostic: helpful **4/8** exact-argmax matches at **H300**; all-state exact chosen saving **+680.3 veh-s/state** on **12** selected states; within-state delay rank **0.516**. The helpful-state mean is a different denominator and must not be labelled +680.3. This small roster-only, source-prior/cold-memory option diagnostic does not qualify continuous residual C_G. <!-- Source: E:/agent-work/c4-wmteach/docs/paper_results/WMTEACH.md:80–90,132–136. -->
- Late S-BDRANK delivery: available-option **H300** delay-pair concordance **75.9% (v4) / 51.7% (v5)** on **58 pairs per horizon/version** in the same small diagnostic. Separate controller diagnostics favor mean delay over saturated breakdown probability on development, but development-selected rules lose on the separate reserved panel. Ranking alone is not policy efficacy; option-breakdown ranking is unavailable from the cached trajectories. <!-- Source: E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md:24–46,48–55. -->

### VI-C / E3 — main comparison: does cooperation emerge from MAPPO?
- T1 main rows: **MAPPO / MAPPO+training control / MAPPO+WM credit / S-WMPLAN / exact planner empirical upper reference**. Columns: paired J savings at declared clocks, exact C_G and D_G/solos, moves/coverage, safety and added real/model compute.
- <span style="color:red">[PENDING: S-COOPMWM common-panel MAPPO/control/WM-credit results, B−A attribution and learning trajectories.]</span> Test improvement over the matched continued-training arm, not merely frozen MAPPO.
- <span style="color:red">[PENDING: S-WMPLAN complete paired efficacy, exact twins and resource-refusal counts.]</span> Existing exact-planner values occupy a separately marked different-panel reference row; do not paste them into missing learned-arm cells.

### VI-D / E4 — discovered behaviours with exact twins
- F4 panels should show a learned wall, gap creation and metering episode: sensed geometry, member actions and resulting merge/wave change, each with same-prefix baseline, joint and each-alone traces.
- Attach D_G, each solo saving and C_G at the declared clock; identify beneficial cases, retain adverse contrasts, report detector exposure/eligibility and avoid outcome-selected prevalence claims.
- <span style="color:red">[PENDING: deployed WM-credit policy behaviour examples and independent replication / S-COOPMWM, S-COOPBEHAV.]</span> Exact-planner walls support opportunity, not discovery by the learned policy.

### VI-E / E5 — ablations
- Compare FULL-v4 versus FULL-v5 using the same frozen credit task, horizon and seeds; do not treat published teacher-option diagnostics as a learned-policy ablation. <span style="color:red">[PENDING: matched model-version ablation / S-COOPMWM.]</span>
- Compare qualified single horizons with multi-horizon weighting; compare ramp/credit weights and CRN versus independent future draws, with unchanged real rollout and replay budgets. <span style="color:red">[PENDING: horizon, credit-weight, ramp and pairing ablations.]</span>
- Record failure modes: sign/rank disagreement, zero/tie credit, safety-erased commands and training drift. Same-budget continuation is the essential credit-off control, not a substitute for every mechanism ablation.

## VII. Discussion / Limitations — 0.25 page

### VII-P1 — what the evidence can support
- Ranking is sufficient only for the decision being made; preserving delay order does not preserve coalition synergy or calibrated win probabilities. CRN can cancel shared noise but leaves model bias and missing-observation uncertainty.
- Source-prior/cold-memory replay differs from fully observed exact playback. Freeze interventions/backgrounds/continuations, distinguish member-alone and member-removal estimands, and report realized motion after safety caps.
- Finite-option oracle planning is an empirical upper reference under privileged prediction, not an optimum or deployable policy. Existing cohorts differ; rare selected examples do not establish population-level cooperation.

### VII-P2 — attribution and generalization
- A positive C_G with negative D_G is not beneficial cooperation. Real SUMO execution still permits auxiliary-reward gaming; compare against capacity/input/budget-matched continued MAPPO and retain all safety/service outcomes.
- Simulator-only evidence, one merge configuration, limited observation/communication assumptions, finite training seeds and development reuse limit real-road inference. <span style="color:red">[PENDING: final reproducibility/freeze and seed-variability evidence.]</span>

## VIII. Conclusion — 0.10 page

### VIII-P1 — close the intended argument, conditioned on results
- Existing exact replays establish useful rare joint interventions and planner headroom. Proposed WM replay credit turns imagined matched-future comparisons into a learnable cooperation signal while MAPPO supplies competent driving.
- <span style="color:red">[PENDING: conclude that cooperation emerged and exceeded both MAPPO and matched continued training only if E2–E4 establish it; otherwise report the measured boundary and negative outcome.]</span>

## Figure and table plan (included in page allocations)

### F1 — method schematic, Section IV
- Show real SUMO → observable CAV/RSU histories → MAPPO residual actor → real SUMO. A separate replay path samples ξ_k and fans out to baseline/joint/each-alone branches with identical draw IDs, then C_G, horizon weights and ramp feed auxiliary credit.
- Draw exact playback as verifier, not the actor's imagined transition source. Symbolic schematic only; empirical annotation stays pending.

### F2 — opportunity, Section VI-A
- Panel (a): exact-planner J600/J900 saving with source CIs, original/extension/pooled cohort labels. Panel (b): all-action synergy distribution and helpful-state prevalence with its selected-corpus denominator.
- Show positive and adverse wall exemplars compactly; caption distinguishes episode savings from event synergy. <!-- Source: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:22–43,95–102; E:/agent-work/c4-coopbandit/docs/paper_results/COOPBANDIT.md:7–18. -->

### F3 — credit fidelity and noise, Section VI-B
- Aligned horizons: exact-vs-WM C_G sign/rank/regret with admission boundary, paired versus unpaired replay noise and team-advantage variance. Show missing clocks as pending, not zero bars.
- Cached option-rank data are available from `BDRANK.md`; <span style="color:red">[PENDING: S-COOPMWM exact residual-credit/noise exports; identify world CIs and normalization in caption.]</span> <!-- Source: E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md:24–32. -->

### F4 — learned behaviour twins, Section VI-D
- Compact space–time/action traces for wall, gap and metering, with baseline/joint/member-alone branches and exact D_G/solos/C_G beneath each case.
- <span style="color:red">[PENDING: policy identities, trace manifests, clocks, exposure and independently replicated examples.]</span>

### T1 — main control/planning comparison, Section VI-C
- Rows/columns as E3; visually separate on-policy learned rows from planning references, and mark different-panel exact reference. Include paired contrast to continued training, coverage and safety rather than only headline means.
- <span style="color:red">[PENDING: common-panel learned and WM-planning data; no placeholder numerical entries.]</span>

### T2 — ablations, Section VI-E
- Rows: model version, horizon rule, credit weight/ramp and paired-future control. Columns: matched budget, fidelity/variance, J gain, exact cooperation, safety.
- <span style="color:red">[PENDING: S-COOPMWM ablation exports; preserve adverse/null runs.]</span>

## Professor-review package and handoff

- `main.tex` mirrors this structure, contains red pending/citation markers, number-source comments and empty framed figure/table placeholders. `refs.bib` uses only inherited verified keys; the inherited verification CSV is authoritative for identity.
- Critical blockers are E2–E5 evidence, exact coalition-alone credit semantics versus the earlier member-revert design, the latest horizon rule, and final matched-control implementation. These are explicit research insertion slots, not omissions hidden behind the exact planner's positive result.
