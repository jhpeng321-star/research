# Paper A — claim → evidence → status

Status vocabulary follows the existing checklist but uses the requested three labels: **SUPPORTED** (only the stated scope), **PENDING** (including inconclusive directional hypotheses), **CONTRADICTED** (completed evidence opposes the assertion). A supported design is not a supported outcome. Source reports were inspected, not rerun. `EVIDENCE_SOURCES.md` records unavailable deliveries; `REFRESH.md` records the latest cut.

**Binding hierarchy:** WM-trained cooperation is primary; SUMO-oracle training is backup/reference. The W-VALUE → W-POLICY → W-RISE/W-WIN chain must resolve on its own evidence. Existing oracle design/receipts cannot fill WM-primary outcome slots. <!-- Source: owner MAIN ADDENDUM; E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md, OWNER FRAMING. -->

## Measurement contribution

### M-DEF — Interaction is distinct from fleet improvement
- **Exact claim:** For fixed background B and group G, define D(S)=J(B)−J(S), then C_G=D(G)−Σ_i D({i}). Positive C_G is superadditive delay saving under these interventions; it neither guarantees D(G)>0 nor identifies intention or altruism.
- **Evidence:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`, estimand; `E:/research-dense/docs/paper_results/COOPMEAS2.md`, definitions and interpretation.
- **Status: SUPPORTED — definition and completed historical use.** This is a traffic operationalization of established counterfactual/coalition ideas, not a claim to invent difference rewards.
- **Scope:** Algebraically C_G=Σ_i J({i})−J(G)−(|G|−1)J(B). With a baseline-normalized coalition value it aggregates interactions beyond singleton terms; it is not a Shapley allocation or, for larger groups, an isolated highest-order interaction. This is algebra, not an empirical estimate.

### M-EXACT — Matched simulator branches remove learned-transition approximation
- **Exact claim:** The historical protocol reproduced native branches and preserved paired initial state, group and exogenous coupling. Exactness is relative to the simulator/replay contract.
- **Evidence:** `E:/research-dense/docs/paper_results/COOPMEAS2.md`, verification; `E:/agent-work/c4-coopmeas2/COOPMEAS2.md`, snapshot/controller-memory design.
- **Status: SUPPORTED — historical contract.** Native repeats passed **56/56** controller-world checks; this does not certify the new audit or learner canary. <!-- Source: E:/research-dense/docs/paper_results/COOPMEAS2.md:78-85. -->
- **Current qualification:** Earlier audit/learner canaries passed on their checked states, but later learner zero-noise saved-state checks failed. **CONTRADICTED as a blanket qualification of the learner's saved-state branches.** Original SNR/interaction values are invalid; full-prefix rerun gates remain pending. Audit canaries are retained as task-specific mechanics evidence, with the full audit held. [[PENDING: corrected exact-branch integrity and complete audit / S-SUMOCF, S-COOPMEASURE]]. <!-- Sources: E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md; E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md. -->

### M-ARTIFACT — Original oracle probe credits are invalidated by replay failure
- **Exact claim:** Identical issued actions did not preserve physical trajectories in later loaded branches; apparent branch gains cannot be attributed to the intervention. Original absolute-held-latent and decoder-path defects were also identified. No optimizer update completed.
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md`; `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md`.
- **Status: SUPPORTED — completed negative mechanics result.** False zero-noise D_G was **+1233.339747428894 / −1555.7833967208862 veh-s** at the two checked later states. These are artifact magnitudes, never cooperation estimates. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md:41-53. -->
- **Recovery scope:** current design requires complete same-seed prefix reruns for every branch, persistent offsets about current policy means through the original Torch/safety/memory path, and a corrected fixed probe. [[PENDING: reviewed full-rerun artifact PASS and corrected design/selection binding / S-SUMOCF]]. Earlier source bytes, charges and failed receipts remain preserved.

### M-BASE — Audit and training use different counterfactual backgrounds
- **Exact claim:** The controller audit disables control to native release; CTRL0 is an always-commanded adapter. Learner credit instead compares a sampled latent-window intervention with checkpoint-mean continuation/background. These estimate different interventions.
- **Evidence:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`, intervention/release; `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, binding amendment, exploration/continuation and explicit `policy_mean_reference` handoff.
- **Status: SUPPORTED — design scope; actual learner binding PENDING.** Never pool learner exploration C_G with native-release audit C_G or call one an increase in the other.
- **Insert:** [[PENDING: confirm sampled-window B/subset/continuation semantics from actual learner receipts and audit the final frozen policy against native release / S-SUMOCF, S-COOPMEASURE]].

### M-COHORT — Entrants, source waiting and missing groups are explicit
- **Exact claim:** Team J includes fleet plus source delay; initial GROUP/ONLY and ALL semantics are separately labeled. Historical entrant-controlling ALL must not be confused with the current learner's ALL over policy-eligible identities present at window start. Missing/undersized groups are missing, not zero; tiny-gain eligible groups remain included. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md:288-290; E:/research-dense/docs/paper_results/COOPMEAS2.md. -->
- **Evidence:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`; `E:/research-dense/docs/paper_results/COOPMEAS2.md`.
- **Status: SUPPORTED — historical analysis/design.** [[PENDING: new audit coverage, group-size distribution, local-cohort definition and denominator / S-COOPMEASURE]]. Local observable-cohort delay cannot replace the fleet primary.

## Existing-controller audit

### A-PANEL — The current primary audit is MAPPO plus wall and a delivered learned arm
- **Exact claim:** The owner narrowed primary existing arms to speed-only MAPPO and wall mechanism control. The producer reserves a trained SUMOCF arm when available; PI/CIRCLES/CTRL0 are optional/history. A WM-trained arm requires an explicit updated binding.
- **Evidence:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`.
- **Status: SUPPORTED — amended prospective design only; artifact HOLD.** [[PENDING: corrected complete MAPPO/wall/learned-arm audit C_G, D_G, ΣD_i, fraction positive, coverage, safety and world CIs / S-COOPMEASURE]], plus [[PENDING: WM-policy arm registration and matching / MAIN, S-COOPMEASURE]]. No full-panel result is available.

### A-WALL — The wall is empirically positive in C_G
- **Evidence:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`, full-roster masked singleton rule and refusal/abort accounting.
- **Status: PENDING.** A designed coordinated mechanism is not guaranteed beneficial. Preserve refused/aborted worlds and identical wall context in singleton branches.
- **Insert:** [[PENDING: wall interaction and command/abort/coverage receipt / S-COOPMEASURE]].
- **Canary scope:** A coordinated wall was assembled, but its single-pair **180 s C_G was −154.834 veh-s**; MAPPO's was **zero** in that same mechanics probe. This is not complete-panel inference and does not contradict a still-pending average positive-control hypothesis. <!-- Source: E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md:19-25. -->

### A-OLD — Earlier cooperation evidence establishes a positive interaction advantage
- **Evidence:** `E:/research-dense/docs/paper_results/COOPMEAS2.{md,csv}`.
- **Status: PENDING — directional inference inconclusive, completed estimates available.** At the primary **300 s**, historical cycle-1 C_G **+138 [−439,+728]**, historical MAPPO **−189 [−1028,+638]**, paired difference **+327 [−395,+1035] veh-s**, on **28 worlds**, reported **95% CIs**. Achieved contrast half-width **715.2 veh-s**, planning projection **9.84 veh-s**. <!-- Source: E:/research-dense/docs/paper_results/COOPMEAS2.md:39-62. -->
- **Scope:** Historical lane-head MAPPO `aecd7d70` and cycle-1 `64f70af4`, not the speed-only export. Conclude “not established,” not “no cooperation.”

### A-ADVERSE — Historical short-clock interaction was adverse
- **Evidence:** `E:/research-dense/docs/paper_results/COOPMEAS2.csv`, `cohort=all,clock_s=60,term=C_G,contrast=ours`.
- **Status: SUPPORTED — secondary, unadjusted.** **−2.982 [−6.131,−0.196] veh-s** at **60 s**, rounded from export. Retain alongside the primary; do not select only favorable clocks. <!-- Source: E:/research-dense/docs/paper_results/COOPMEAS2.csv:2. -->

### A-HEADROOM — Existing speed-only MAPPO clearly beats CTRL0
- **Evidence:** `E:/agent-work/c4-speedhead/docs/paper_results/SPEEDHEAD.md`.
- **Status: PENDING — exploratory comparison inconclusive.** The **600 s** real team-cost contrast is **+32.6 [−3920.5,+3868.4] veh-s**, **eight worlds**. This does not establish equivalence or predict the larger audit. <!-- Source: E:/agent-work/c4-speedhead/docs/paper_results/SPEEDHEAD.md:4-12,45,67. -->
- **Later completed DEV baseline:** MAPPO−CTRL0 **J600 −905.06 [−1904.59,+8.85] veh-s**, **J900 +560.82 [−864.84,+2035.35] veh-s**, **24 worlds**, source **95% CIs**. Neither establishes a win; mean ranking changes across horizons. This is baseline development evidence, not a learned-arm or cooperation result. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:19-26. -->

### A-BEHAV — Behavior patterns establish beneficial cooperation
- **Evidence:** `E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md`, completed descriptive corpora and original probe discovery; `E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md`.
- **Status: CONTRADICTED as an inference from these delivered data.** Descriptive detector counts/exposure are delivered, but whole-controller comparisons do not isolate members. Original probe positive-credit discovery uses now-invalidated branch costs and cannot support beneficial interaction, even if its trace geometry/counting is reproducible.
- **Scope:** retain descriptive corpora separately and disclose censoring/observation coverage; do not pool repeated probe designs as independent worlds or label outcome-selected clusters causal mechanisms. [[PENDING: valid corrected paired traces with audited credit, coverage and selection / S-COOPBEHAV, corrected trace producer]].

## Primary contribution — WM-trained cooperative policy

### W-VALUE — Short WM imagination plus a real-trained value predicts long outcomes
- **Exact claim to test:** A short WM integral plus terminal observable V trained on real SUMO labels predicts paired remaining fleet+source costs more accurately than full WM, a command-aware no-WM value and FIT-only arm/time means.
- **Evidence:** `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md`; result expected at `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md` (absent at this cut).
- **Status: PENDING — frozen offline design SUPPORTED; predictive success unmeasured.** Candidates use **60/90 s** imagination and a fixed absolute endpoint **t=900 s**, with starts **300/360/420/480/540 s**. The gate has **400 pair/time rows across eight reserved TRAIN worlds**; confidence resamples entire worlds, not rows. It requires one short-horizon method to have strictly positive **95%** lower bounds for both sign improvement and MAE reduction against each declared baseline, using **10,000** world bootstrap draws. <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md:5-11,51-71. -->
- **Admissibility:** sensor masks/reproducible observation adapter, actual causal source queues and real fleet+source clocks precede fitting. Missing telemetry yields INCOMPLETE/NA, not a truth-feature or road-only substitute. The first frozen test excludes optional SUMOCF traces. [[PENDING: telemetry gate, all baseline/horizon results, error/ranking/sign CIs and data/model/code hashes / S-WMVALUE]].
- **Scope:** recorded policy-continuation prediction; later starts use each arm's own state. This is not randomized same-state counterfactual interaction, a final evaluation panel or policy efficacy. Do not call the proposed short horizon verified under the new contract before its gate.

### W-POLICY — The primary policy is trained by WM counterfactual group credit plus V
- **Exact proposed method:** For each matched background/subset branch, approximate remaining J by the short imagined fleet+source integral plus a real-trained terminal V, then form D and C_G from those costs. Use the owner's MAPPO-warm-started per-CAV/observable-RSU coordination route, subject to a separately frozen implementation.
- **Evidence:** owner MAIN ADDENDUM; `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`, WM-centric direction; `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md`, delivery contract.
- **Status: PENDING — owner method direction only.** [[PENDING: W-VALUE PASS, separate policy registration, actor/value/WM hashes, continuation/horizon conditioning, matched group/background/exploration and executed objective / MAIN WM policy producer, S-WMVALUE]]. No training run is implied by the offline design.
- **Scope:** algebraic proposal is Jhat(S)=imagined integral(S)+V(s_terminal(S),remaining time; continuation contract). A value learned from fixed-controller mixtures may not evaluate a new policy's continuation or preserve small branch differences; state-only value may alias distinct future actions. Validate this transfer explicitly.

### W-RISE — WM training increases exactly measured cooperation
- **Evidence needed:** frozen WM policy versus initial speed-only MAPPO, common native-release audit worlds/groups/windows and exact-valid replay with world-cluster paired CIs.
- **Status: PENDING.** [[PENDING: WM checkpoint, matching audit arm, C_G/D_G/solo changes and all-candidate lineage / MAIN WM policy producer, S-COOPMEASURE, S-COOPEVAL]]. WM-imagined credit, offline pair accuracy and backup learner credit cannot fill this claim.

### W-WIN — WM-trained cooperation beats MAPPO on real traffic outcomes
- **Evidence needed:** exact paired fleet+source delay, source/ramp/mainline and safety outcomes, frozen policy identity, selection/exposure history and training-seed scope.
- **Status: PENDING.** [[PENDING: WM-policy real outcome contrasts, later-horizon ranking, uncertainty and safety / MAIN WM policy producer, S-COOPEVAL]]. Offline prediction PASS alone does not establish efficacy or credit-specific benefit.

### W-RATIO — WM policy recovers a fraction of the SUMO-trained gain
- **Evidence needed:** WM and oracle candidates against the same MAPPO reference, metric/horizon/weighting, worlds, actuator/input contracts and paired uncertainty.
- **Status: PENDING.** Intended efficacy ratio is (J_MAPPO−J_WM)/(J_MAPPO−J_oracle), computed jointly on matched worlds only after a beneficial oracle denominator is established. Report raw paired gains first; absent, nonpositive or poorly resolved denominator means ratio not estimable. Do not substitute a cooperation ratio for traffic gain.
- **Scope:** the oracle-trained result is a finite-budget empirical reference, not a proven optimal upper bound. [[PENDING: corrected oracle and WM gains, compatible evaluation and ratio uncertainty / S-SUMOCF, MAIN WM policy producer, S-COOPEVAL]].

## Backup/reference contribution — SUMO-oracle training

### L-METHOD — Warm-started observable per-CAV counterfactual learner
- **Exact claim:** The amended method warm-starts speed-only MAPPO, trains its final hidden layer and mean row plus an observable RSU broadcast branch, and uses correlated latent exploration with matched subset branches.
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, binding amendment only.
- **Status: SUPPORTED — corrected design/CPU checks, not learning efficacy or exact replay qualification.** Coordination latent is at most **four-dimensional**; training exploration mixes **ρ∈{0,0.5,1}**. Full-rank cases use subset joint scores; full correlation uses a **common-mode projected surrogate**. Credit outcome horizon is **600 s after sampled start**. Corrected intervention is a persistent offset around the current policy mean through the original Torch execution path, not an absolute held start mean. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md:3-60,68-95. -->
- **Scope:** Native lanes and original safety executor. The original probe and in-flight pre-HOLD samples are not optimizer input. No completed optimizer update exists. [[PENDING: corrected full-rerun artifact release, trained implementation/config/checkpoint hashes and executed objective / S-SUMOCF]]. <!-- Sources: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:3-17; E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md. -->

### L-SNR — Probe selection has established a usable cooperation signal
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, SNR gate; `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`, WAIT_SUMOCF.
- **Status: CONTRADICTED for the original published probe; PENDING for the corrected probe.** The original table landed, but its saved-state credits were invalid artifacts. Its initial CV-based ranking was also corrected to the registered raw D_G rule before the replay defect was confirmed. Neither original winner licenses learning or an audit.
- **Insert:** [[PENDING: full-rerun artifact PASS, corrected raw-D_G SNR table, chosen group/window, frozen SHA and C_G precision / S-SUMOCF]]. Corrected prospective reduced probe fixes **ρ=0.5** on **two** previously bound states across **10/30/60 s** windows; other correlation cells are unmeasured, not zero, and the absent larger group is NA. Local CV coefficient is fixed to **zero**. Group-benefit SNR still does not imply C_G precision. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md:12-20,39-48. -->

### L-ANTIGAME — The objective guards against worsening solo behavior
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, credit weights; `E:/agent-work/c4-papera/docs/WM_C4_DESIGN.md`, late update.
- **Status: SUPPORTED — algebraic guard, not a learned-outcome guarantee.** w_coop≤w_ind preserves nonnegative expanded singleton weight; report D_G and ΣD_i separately even when C_G improves. [[PENDING: actual weighted-objective accounting and observed singleton outcomes / S-SUMOCF]].

### L-RISE — Training increases measured cooperation
- **Evidence needed:** matched native-release C_G of initial MAPPO and every declared frozen successor on common worlds/groups, with shared world resamples and exposure history.
- **Status: PENDING.** [[PENDING: final candidate C_G, paired candidate−MAPPO interaction CI, eligible/positive fractions, all candidates and credit-off comparison / S-SUMOCF, S-COOPMEASURE, S-COOPEVAL]]. Training reward or adaptively sampled training C_G alone cannot establish policy-level improvement.

### L-WIN — The new learner beats MAPPO
- **Evidence needed:** complete paired team-delay and physical-vector dev exports, checkpoint-selection history and frozen candidate identity.
- **Status: PENDING.** [[PENDING: learned−MAPPO and learned−CTRL0 fleet+source cost CIs, later-clock ranking, subgroup and safety outcomes / S-COOPEVAL]]. Development selection is exploratory; a fresh fixed confirmation and training-seed replication remain separate gaps.
- **Contract issue:** current harness lists **120/300/600 s**; amended design requires **600/900 s after control start**. No later-clock value can be extrapolated from the older loader. <!-- Sources: E:/agent-work/c4-coopeval/tools/coopeval/__init__.py:13; E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md:161-177. -->
- **Latest dev contract:** planned **six** checks include a **three-world**, geometrically selected **pair** diagnostic, rotating high-demand cells. It measures learned deployed means versus MAPPO means during the window with learned continuation thereafter, not random exploration credit. It remains distinct from native-release audit C_G and selected training groups; rotating worlds do not supply a matched policy-level learning curve. Completed baseline values are recorded in A-HEADROOM; no learned check is available. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md:161-177; E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:19-26. -->

### L-OLDWIN — The older frozen imagined policy already beat MAPPO
- **Evidence:** `E:/research-dense/docs/paper_results/E3FINAL_A.md`.
- **Status: CONTRADICTED — historical checkpoint/panel only.** Registered primary **600 s** cycle-1−MAPPO **+683.5 [+256.9,+1122.4] veh-s** on **228 paired worlds**; positive cost is worse. Native-relative benefit was **−1093.9 [−1637.1,−558.0] veh-s**. <!-- Source: E:/research-dense/docs/paper_results/E3FINAL_A.md:3-14,32-34. -->
- **Scope:** This is neither a result for the speed-only warm start nor for the new learner. Use as motivation/evidence, not a substitute successor baseline.

### L-DRIFT — Earlier WM-trained candidates provide evidence of instability
- **Evidence:** `E:/agent-work/c4-pilotdiag/docs/paper_results/PILOTDIAG.md`; `E:/agent-work/c4-speedhead/docs/paper_results/CANDB_INTEG.md`.
- **Status: SUPPORTED for B/B-L diagnostic drift; PENDING for complete A/B/B-L inventory.** Credit tails and harmful braking are observed; remedies were implementation/diagnostic proposals, not demonstrated efficacy.
- **Insert:** [[PENDING: candidate A final receipt, B/B-L immutable checkpoint/outcome inventory and distinction between WM telemetry and real-SUMO cost / S-PILOTDIAG, owner integrator]]. Do not label pilot WM rescoring as a new real-SUMO result.

### L-TRACK3 — Track 3 learned an improvement
- **Evidence:** `E:/agent-work/c4-sumoes/docs/paper_results/SUMOES_RUN.md`.
- **Status: CONTRADICTED — completed CTRL0-base and MAPPO-base Track 3B searches.** Track 3B stopped after **four consecutive real-update rejections, zero accepted updates**, retaining an exactly zero/MAPPO-equivalent head. Proposed panel cost deltas were **+1602.4032432138847, +594.04014733433, +477.04223105311394, +1241.7322083711551 veh-s**. Combined historical+Track 3B consumption was **171,900 sim-s**. These are adaptive fixed TRAIN-panel decisions, not fresh confirmatory efficacy CIs or a proof that every residual direction fails. Older screen-only no-step decisions remain distinct from these real-update rejections. <!-- Source: E:/agent-work/c4-sumoes/docs/paper_results/SUMOES_RUN.md:1-10,36-85. -->

### L-CREDIT — A prior cooperation-loss ablation proved causal benefit
- **Evidence:** `E:/research-dense/docs/paper/CLAIM_EVIDENCE_CHECKLIST.md`, E5-CREDIT (secondary report pointer).
- **Status: CONTRADICTED as a positive causal claim.** The checklist records byte-identical ON/OFF trajectories despite different commands. [[PENDING: direct E5 report binding and meaningful matched-capacity/credit-off successor ablation / S-COOPEVAL, owner integrator]].

## World-model reliability motivating the primary value-tail route

### W-SHORT — WM-S is accurate to about 90 s
- **Evidence:** `E:/research-dense/docs/paper_results/E2_direction_horizon.csv` versus `E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md`.
- **Status: SUPPORTED only for the historical direction panel; CONTRADICTED as a general/current-contract claim.** Older **90 s** accuracy **0.8895 [0.8424,0.9341]**, source PASS; **120 s 0.7526 [0.6824,0.8187]**, FAIL, threshold **0.8684**. Checkpoint alias not bound. Corrected matched native-lane sign accuracy at **60/90 s** is **0.571/0.444**; overall NOT QUALIFIED. <!-- Sources: E:/research-dense/docs/paper_results/E2_direction_horizon.csv:5-6; E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md:3-18,29-30. -->
- **Safe wording:** “Short-horizon direction fidelity is panel-dependent; it does not transfer automatically to the native-lane counterfactual contract.”

### W-LONG — Corrected long-horizon failure is quantified upstream
- **Evidence:** `E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md`; `tools/wmrequal3/evidence/ANALYSIS.json`.
- **Status: SUPPORTED — recorded-intervention diagnostic.** At **300 s**, matched total-effect MAE **1376.6 veh-s**, sign accuracy **0.556**, correlation **−0.235**; upstream/merge-road MAE **1497.6 veh-s**, covariance share **96.0%**. <!-- Source: E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md:17,31-43. -->
- **Scope:** Covariance share is signed additive error accounting, not a causal fraction; source cancellation allows regional MAE above total. Recorded-command replay with synthetic entrant feedback is not full policy-level native-release C_G fidelity.

### W-ENTRY — Source correction fixes the merge-queue problem
- **Evidence:** `E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_RETARGET.md`.
- **Status: CONTRADICTED for the tested zero-head entry path.** At **300 s**, clock-corrected upstream MAE **1456.9→1393.0 veh-s** while source MAE **584.0→82.5**; total J MAE worsens **1449.1→1536.2 veh-s**. <!-- Source: E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_RETARGET.md:25-36,44-48. -->
- **Scope:** Supports residual upstream mismatch after entry repair; does not prove a unique causal merge mechanism or validate the trained repair head. [[PENDING: later trained-head verdict / S-WMFIXMERGE]].

### W-DEPLOY — WM-S is a validated deployable replacement for exact C_G
- **Evidence needed:** same B/G/ONLY interventions, inputs, population, horizons and controller memory; paired predicted-versus-exact C_G error, sign/ranking and calibrated abstention/latency.
- **Status: PENDING.** Existing total-effect fidelity and observability designs are insufficient. The primary proposal now uses WM plus a real-trained value tail; success of either component alone does not validate complete C_G. [[PENDING: checkpoint-bound WM+V interaction fidelity, continuation transfer, runtime observability and latency / S-WMVALUE, MAIN WM policy producer, S-COOPEVAL]]. Distinguish offline WM training from a claim that WM inference is required or validated at deployment.

## Reporting constraints

### R-SAFETY — Efficiency comes without safety/fairness trade-offs
- **Evidence:** `E:/research-dense/docs/paper_results/E3FINAL_A.md`; `E:/research-dense/docs/paper_results/COOPMEAS2.md`.
- **Status: CONTRADICTED for comparator-neutral historical wording; PENDING for successor.** Historical final cycle-1−MAPPO hard braking **+2.1 [+1.1,+3.1] veh-s**, TTC-below-**3 s** **+27.5 [+13.8,+41.9] veh-s**, worst-group delay **+2.3 [+1.4,+3.1] s/request**. Incident zeros do not establish noninferiority. <!-- Source: E:/research-dense/docs/paper_results/E3FINAL_A.md:110-121. -->

### R-FAIR — Warm-started richer-input training establishes credit-specific superiority
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`; `E:/agent-work/c4-papera/docs/WM_C4_DESIGN.md`.
- **Status: PENDING.** Need matched RSU/latent capacity, warm start, additional simulator budget and correlated-exploration controls, with/without cooperation credit. Count MAPPO pretraining, acquisition and failed attempts separately; no equal-budget or credit-causation claim from a frozen speed-only baseline alone.

### R-NOVEL — This is the first counterfactual reward or cooperative merging method
- **Evidence:** verified works in `RELATED_WORK.md`/`refs.bib`.
- **Status: CONTRADICTED as a broad novelty assertion.** Defensible contribution: a specified simulator-exact longitudinal coalition-interaction audit in mixed merging, followed by an explicitly tested learning and surrogate-fidelity question. Final novelty scope needs comparison of full closest papers, not titles alone.

### R-FORMAT — The package is already a paginated conference submission
- **Evidence:** owner brief and `OUTLINE.md` editorial allocation.
- **Status: PENDING.** Target **six pages including references**, deadline **2026-11-15**, IEEE IV **2027** are owner-provided. [[PENDING: official venue author instructions, final IEEEtran compilation, authors and pagination / owner]]. <!-- Source: owner S-PAPERA task, 2026-10-31. -->
