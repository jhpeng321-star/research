# Paper A — claim → evidence → status

Status vocabulary follows the existing checklist but uses the requested three labels: **SUPPORTED** (only the stated scope), **PENDING** (including inconclusive directional hypotheses), **CONTRADICTED** (completed evidence opposes the assertion). A supported design is not a supported outcome. Source reports were inspected, not rerun. `EVIDENCE_SOURCES.md` records unavailable deliveries; `REFRESH.md` records the latest cut.

**Binding hierarchy:** WM-trained cooperation is primary; SUMO-oracle training is backup/reference. The W-VALUE → W-POLICY → W-RISE/W-WIN chain must resolve on its own evidence. Existing oracle design/receipts cannot fill WM-primary outcome slots. <!-- Source: owner MAIN ADDENDUM; E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md, OWNER FRAMING. -->

**Night verdict:** WMVALUE is RUNNING with frozen real-trained values and audited adapters; its predictive metrics remain NA. The SUMO backup is FINAL STOPPED without useful deployed cooperation or a primary-horizon MAPPO win. Calibration is NOT USABLE and the queue/service pilot is KILLED. These completed negative results replace earlier delivery placeholders; the WM policy itself is still PENDING. <!-- Sources: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md; E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md; E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md; E:/agent-work/c4-wmqueue/docs/paper_results/WMQUEUE.md. -->

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
- **Current qualification:** Earlier canaries passed on their checked states, but later learner zero-noise saved-state checks failed. **CONTRADICTED as a blanket qualification of saved-state branches.** Original SNR/interaction values remain invalid. Corrected full-prefix reruns reproduce checked zero-offset baseline cost/trajectory bitwise; final learner reconciliation PASS is delivered. **SUPPORTED — corrected mechanics scope only.** [[PENDING: audit-specific release/versioned complete-panel migration / S-COOPMEASURE]]. <!-- Sources: E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md; E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:59-88; E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md. -->

### M-ARTIFACT — Original oracle probe credits are invalidated by replay failure
- **Exact claim:** Identical issued actions did not preserve physical trajectories in later loaded branches; apparent branch gains cannot be attributed to the intervention. Original absolute-held-latent and decoder-path defects were also identified. No optimizer update completed before that original HOLD; corrected training subsequently ran and stopped.
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md`; `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md`.
- **Status: SUPPORTED — completed negative mechanics result.** False zero-noise D_G was **+1233.339747428894 / −1555.7833967208862 veh-s** at the two checked later states. These are artifact magnitudes, never cooperation estimates. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md:41-53. -->
- **Recovery scope:** complete same-seed prefix reruns and persistent offsets through the original Torch/safety/memory path pass the published gate. Corrected launch, checkpoint receipts and final reconciliation are delivered; original failures remain preserved and invalidated. [[PENDING: audit-specific release / S-COOPMEASURE]]. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:3-88,213-226. -->
- **Execution distinction:** larger issued-safe command deviation need not change realized physical motion: one checked state had identical trajectories/cost under both tested nonzero scales because final SUMO caps erased the intervention. This is a valid zero physical effect after mechanics recovery, unlike the old false branch gains. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md:62-83. -->

### M-BASE — Audit and training use different counterfactual backgrounds
- **Exact claim:** The controller audit disables control to native release; CTRL0 is an always-commanded adapter. Learner credit instead compares a sampled latent-window intervention with checkpoint-mean continuation/background. These estimate different interventions.
- **Evidence:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`, intervention/release; `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, binding amendment, exploration/continuation and explicit `policy_mean_reference` handoff.
- **Status: SUPPORTED — design and executed learner scope.** Corrected TRAIN exploration uses sampled offsets versus checkpoint means; DEV uses learned-versus-MAPPO deployed means with learned continuation. Neither is native-release audit C_G. Never pool them or call one an increase in the other. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:26-46,184-225. -->
- **Insert:** [[PENDING: corrected native-release frozen-policy audit / S-COOPMEASURE]].

### M-COHORT — Entrants, source waiting and missing groups are explicit
- **Exact claim:** Team J includes fleet plus source delay; initial GROUP/ONLY and ALL semantics are separately labeled. Historical entrant-controlling ALL must not be confused with the current learner's ALL over policy-eligible identities present at window start. Missing/undersized groups are missing, not zero; tiny-gain eligible groups remain included. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md, D_all clarification; E:/research-dense/docs/paper_results/COOPMEAS2.md. -->
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
- **Later completed DEV baseline:** MAPPO−CTRL0 **J600 −905.06 [−1904.59,+8.85] veh-s**, **J900 +560.82 [−864.84,+2035.35] veh-s**, **24 worlds**, source **95% CIs**. Neither establishes a win; mean ranking changes across horizons. This is baseline development evidence, not a learned-arm or cooperation result. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md, Completed fixed DEV baseline. -->

### A-BEHAV — Behavior patterns establish beneficial cooperation
- **Evidence:** `E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md`, completed descriptive corpora and original probe discovery; `E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md`.
- **Status: CONTRADICTED as an inference from these delivered data.** Descriptive detector counts/exposure are delivered, but whole-controller comparisons do not isolate members. Original probe discovery remains invalid. Corrected paired DEV and TRAIN reports now exist; they do not demonstrate beneficial deployed cooperation.
- **Delivered corrected evidence:** deployed **u3**, **two eligible worlds/one missing**, **all four paired-supported detector counts zero**. At sampled TRAIN **u15**, **31 eligible windows/six worlds**, reference-supported yield/opening/hold/damping counts **0/1/0/5**; these are exploration mechanisms, not deployed means. Its largest positive-credit record has **C_G +2482.05** but **D_G −3413.52 veh-s**; the unanticipated single-record signature has **C_G 0.0000296** and **D_G −0.000000954 veh-s**. Neither establishes a novel beneficial mechanism. <!-- Source: E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md:165-236. -->
- **Scope:** corrected examples are delivered, not missing. Retain exposure/censoring and immutable-index provenance; no pooling overlapping snapshots or outcome-selected clusters as independent worlds. [[PENDING: independently replicated beneficial deployed mechanism / MAIN WM policy producer, S-COOPBEHAV]].

## Primary contribution — WM-trained cooperative policy

### W-VALUE — Short WM imagination plus a real-trained value predicts long outcomes
- **Exact claim to test:** A short WM integral plus terminal observable V trained on real SUMO labels predicts paired remaining fleet+source costs more accurately than full WM, a command-aware no-WM value and FIT-only arm/time means.
- **Evidence:** `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md`; delivered verdict `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md`.
- **Status: PENDING — frozen decisive inference RUNNING; no predictive verdict.** Candidates use **60/90 s** imagination and fixed absolute endpoint **t=900 s**, with starts **300/360/420/480/540 s**. Gate: **400 pair/time rows across eight reserved TRAIN worlds**, whole-world resampling and strictly positive **95%** lower bounds for sign improvement and MAE reduction against every baseline, using **10,000** draws. These are design quantities, not completed prediction metrics. <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md:5-11,51-71. -->
- **Resolved blocker / SUPPORTED execution:** registered causal reconstruction replaces unavailable sensing channels with **111 observable features**, no source queue/request/demand features; **460 FIT episodes/82 worlds/9660 states**. Full-team labels reconcile **1380** recorded cost clocks, maximum discrepancy **7.87e−6 veh-s**, source exactly **zero**. V and command-only D fit once in **9.89 CPU s**, E also frozen. **36 CPU tests**, actual WM repeats and **134-source** cross-host parity pass. Reserved real cells were exported after the adapter gate; **120** reserved objective clocks reconcile. All A–E sign/rank/MAE/CI cells remain NA pending complete **320-task** inference. <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md:17-55. -->
- **Next evidence:** [[PENDING: complete once-only A–E predictions, contrasts and hash-bound gate verdict / S-WMVALUE existing dispatcher and finalizer]]. Optional SUMOCF traces stay excluded. Matched new A/B/C trajectories replace uncertified cached long-WM aggregates; preserve the caches, do not mix their exit clocks or CRNs. No duplicate inference is needed from this package. <!-- Sources: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md:38-57; E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md:113-146. -->
- **Scope:** recorded policy-continuation prediction; later starts use each arm's own state. This is not randomized same-state counterfactual interaction, a final evaluation panel or policy efficacy. Do not call the proposed short horizon verified under the new contract before its gate.

### W-POLICY — The primary policy is trained by WM counterfactual group credit plus V
- **Exact proposed method:** For each matched background/subset branch, approximate remaining J by the short imagined fleet+source integral plus a real-trained terminal V, then form D and C_G from those costs. Use the owner's MAPPO-warm-started per-CAV/observable-RSU coordination route, subject to a separately frozen implementation.
- **Evidence:** owner MAIN ADDENDUM; `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`, WM-centric direction; `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md`, delivery contract.
- **Status: PENDING — policy direction only; awaits the running W-VALUE test.** [[PENDING: W-VALUE PASS, separate policy registration, actor/value/WM hashes, continuation/horizon conditioning, matched group/background/exploration and executed objective / MAIN WM policy producer, S-WMVALUE]]. Real fitting and adapter audit are complete; they do not authorize policy training. <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md:38-57. -->
- **Scope:** algebraic proposal is Jhat(S)=imagined integral(S)+V(s_terminal(S),remaining time; continuation contract). A value learned from fixed-controller mixtures may not evaluate a new policy's continuation or preserve small branch differences; state-only value may alias distinct future actions. Validate this transfer explicitly.

### W-RISE — WM training increases exactly measured cooperation
- **Evidence needed:** frozen WM policy versus initial speed-only MAPPO, common native-release audit worlds/groups/windows and exact-valid replay with world-cluster paired CIs.
- **Status: PENDING.** [[PENDING: WM checkpoint, matching audit arm, C_G/D_G/solo changes and all-candidate lineage / MAIN WM policy producer, S-COOPMEASURE, S-COOPEVAL]]. WM-imagined credit, offline pair accuracy and backup learner credit cannot fill this claim.

### W-WIN — WM-trained cooperation beats MAPPO on real traffic outcomes
- **Evidence needed:** exact paired fleet+source delay, source/ramp/mainline and safety outcomes, frozen policy identity, selection/exposure history and training-seed scope.
- **Status: PENDING.** [[PENDING: WM-policy real outcome contrasts, later-horizon ranking, uncertainty and safety / MAIN WM policy producer, S-COOPEVAL]]. Offline prediction PASS alone does not establish efficacy or credit-specific benefit.

### W-RATIO — WM policy recovers a fraction of the SUMO-trained gain
- **Evidence needed:** WM and oracle candidates against the same MAPPO reference, metric/horizon/weighting, worlds, actuator/input contracts and paired uncertainty.
- **Status: CONTRADICTED for a currently reportable positive gain-retention percentage.** Intended ratio is (J_MAPPO−J_WM)/(J_MAPPO−J_oracle). The completed backup has no beneficial primary-horizon denominator, and WM policy outcomes are missing: **NOT ESTIMABLE at this cut**. Report raw costs, not a percentage or a cooperation ratio. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:3-37. -->
- **Scope:** finite-budget simulator training is an empirical reference, not an optimal upper bound or successful teacher. A future ratio requires separately established positive gains, compatible evaluation and joint uncertainty; the stopped backup is not an unfinished positive result.

## Backup/reference contribution — SUMO-oracle training

### L-METHOD — Warm-started observable per-CAV counterfactual learner
- **Exact claim:** The amended method warm-starts speed-only MAPPO, trains its final hidden layer and mean row plus an observable RSU broadcast branch, and uses correlated latent exploration with matched subset branches.
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, binding amendment only.
- **Status: SUPPORTED — corrected design/CPU checks and checked rerun mechanics, not learning efficacy.** Coordination latent is at most **four-dimensional**; training exploration mixes **ρ∈{0,0.5,1}**. Full-rank cases use subset joint scores; full correlation uses a **common-mode projected surrogate**. Credit outcome horizon is **600 s after sampled start**. Corrected intervention is a persistent offset around the current policy mean through the original Torch execution path, not an absolute held start mean. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md, corrected resumption and Policy/observability/density/horizon; E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md. -->
- **Executed scope:** Native lanes and original safety executor. Original invalid credits are excluded from optimizer input. Corrected launch and **15 completed optimizer updates** are delivered, followed by final owner STOP; no next optimizer step occurred. All checkpoint/Adam pairs are preserved and verified. Last evaluated **u12** policy SHA `c8b69d163adf6d6903b5356f2eacd565f9ca246566c6c88fca3b4b9814bbdb2d`; later **u13–u15** have no DEV outcomes. This supports implementation/execution, not efficacy. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:3-88. -->

### L-SNR — Probe selection has established a usable cooperation signal
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, SNR gate; `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`, WAIT_SUMOCF.
- **Status: CONTRADICTED for a useful established signal; corrected mechanics/selection SUPPORTED.** Original table credits are invalid artifacts, after a separately disclosed CV-ranking defect. The corrected full-rerun probe now reports **C_G mean/SD = 0/0** for every estimable design on **two worlds**. Exact registered maximum selects **pair/60 s, nominal ρ=0.5**; the SNR distinction from the shorter nonzero design is floating-point roundoff, not scientific evidence. Other correlation cells are unmeasured; larger groups are missing; all-zone labels alias the same pair. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md:85-109. -->
- **Measured group signal:** corrected **60 s** D_G mean **0.001293659**, SD **0.001829510 veh-s**, SNR **0.7071067811865476**, arising from one tiny effect and one zero. Artifact actual=charged **72,881 sim-s**, including failed saved-state gate. CV coefficient remains **zero**. No SNR cutoff was registered. Corrected learning subsequently completed without establishing useful deployed interaction; see L-RISE. <!-- Sources: E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md:85-109; E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:26-46. -->

### L-ANTIGAME — The objective guards against worsening solo behavior
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, credit weights; `E:/agent-work/c4-papera/docs/WM_C4_DESIGN.md`, late update.
- **Status: SUPPORTED — algebraic guard, not a learned-outcome guarantee.** w_coop≤w_ind preserves nonnegative expanded singleton weight; report D_G and ΣD_i separately even when C_G improves. Executed sampled/DEV group and solo quantities are delivered, with no beneficial deployed cooperation established. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:26-46,105-126. -->

### L-RISE — Training increases measured cooperation
- **Evidence needed:** matched native-release C_G of initial MAPPO and every declared frozen successor on common worlds/groups, with shared world resamples and exposure history.
- **Status: CONTRADICTED for useful deployed improvement in the completed backup; full native-release audit PENDING.** Fixed-pair deployed-mean diagnostic C_G at **u3/u7/u12** is **−0.002589/+0.001270/0 veh-s**, on **2/2/3 eligible worlds**, with **1/1/0 missing groups**. At **u12**, **D_G=ΣD_i=−0.000234 veh-s**. The tiny positive **u7 CI [+0.000854,+0.001686] veh-s** does not establish useful cooperation; the whole-panel traffic cost worsens. Sampled TRAIN credit is not deployed-policy interaction. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:26-40. -->
- **Remaining:** [[PENDING: complete matched native-release audit and credit-off attribution / S-COOPMEASURE, S-COOPEVAL]]. Sparse diagnostics do not prove that every group is noncooperative.

### L-WIN — The new learner beats MAPPO
- **Evidence needed:** complete paired team-delay and physical-vector dev exports, checkpoint-selection history and frozen candidate identity.
- **Status: CONTRADICTED for demonstrated primary-horizon superiority in this completed backup.** Fixed **24-world** DEV learned−MAPPO **J600** at **u3/u7/u12**: **+508.3 [−343.5,+1328.3] / +857.9 [+118.4,+1657.3] / +548.5 [−397.8,+1520.6] veh-s**. Corresponding **J900 −633.4 [−2141.1,+866.1] / −465.0 [−1869.1,+974.4] / −584.5 [−2079.7,+1002.3]**. The middle primary check is adverse beyond its pointwise interval; other intervals include zero. Mean ranking flips with horizon. At **u12**, vs CTRL0 **J600 −356.6 [−1402.3,+752.1] veh-s**. **95%** world-bootstrap intervals, **2000 draws/seed 0**. Owner STOP executed; this is a completed unsuccessful backup, not a pending efficacy delivery. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:3-24. -->
- **Contract issue:** current harness lists **120/300/600 s**; amended design requires **600/900 s after control start**. No later-clock value can be extrapolated from the older loader. <!-- Sources: E:/agent-work/c4-coopeval/tools/coopeval/__init__.py:13; E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md, corrected DEV plan. -->
- **Executed stop/budget:** checks stopped after **u3/u7/u12**; no later DEV was dispatched. TRAIN **3,194,716 actual / 3,205,151 charged sim-s**, DEV **184,049 actual=charged sim-s**, including invalidated work, full prefixes and final drain. Final reconciliation PASS; no live continuation. DEV incident counts are zero, not proof of equivalence. [[PENDING: complete fairness/service vector, MAPPO pretraining and matched-control budget lineage / S-COOPEVAL, owner integrator]]. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:48-88. -->

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
- **Scope:** Supports residual upstream mismatch after entry repair; does not prove a unique causal merge mechanism or validate the trained repair head. Latest v2-DATA report has a complete zero-head entry baseline but no complete candidate panel/gate verdict; optimizer completion is not an accuracy result. Scope is gate-only, not automatic post-gate training. [[PENDING: complete candidate panels and registered gate verdict / S-WMFIXMERGE existing scorer/watcher]]. <!-- Source: E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_PILOT_V2DATA.md, Current scope and Complete step0 baseline. -->

### W-DEPLOY — WM-S is a validated deployable replacement for exact C_G
- **Evidence needed:** same B/G/ONLY interventions, inputs, population, horizons and controller memory; paired predicted-versus-exact C_G error, sign/ranking and calibrated abstention/latency.
- **Status: PENDING.** Existing total-effect fidelity and observability designs are insufficient. The primary proposal now uses WM plus a real-trained value tail; success of either component alone does not validate complete C_G. [[PENDING: checkpoint-bound WM+V interaction fidelity, continuation transfer, runtime observability and latency / S-WMVALUE, MAIN WM policy producer, S-COOPEVAL]]. Distinguish offline WM training from a claim that WM inference is required or validated at deployment.

### W-CALIB — Calibration qualifies long-horizon credit
- **Evidence:** `E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md`.
- **Status: CONTRADICTED for this frozen calibration track.** Verdict NOT USABLE. In world-grouped CV, raw WM2 MAE at **120/300/600 s** is **263/1719/6013 veh-s**, versus no-WM arm means **185/999/3298**. Selected ridge improves CV at **300 s** to **865 veh-s**, sign **0.692**, but every SPEEDHEAD horizon fails the frozen multi-metric gate. These are point estimates, not invented CIs. <!-- Source: E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md:14-50. -->
- **Scope:** CV selection is optimistic; raw WM is not worse on every metric or panel. SPEEDHEAD raw **300 s** MAE **1399** versus arm mean **1568 veh-s** is an explicit exception. Changed-reference WMREQUAL3 transfer is diagnostic, not policy-credit qualification. <!-- Source: E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md:29-60. -->

### W-QUEUE — A conservative queue/service tail fixes prediction
- **Evidence:** `E:/agent-work/c4-wmqueue/docs/paper_results/WMQUEUE.md`.
- **Status: CONTRADICTED for the registered prototype.** KILL: on **eight reserved TRAIN worlds/48 contrasts per horizon**, queue-model MAE at **120/300/600 s** is **276/1362/5296 veh-s**, versus state-only **239/932/3642**; sign **0.438/0.313/0.229** versus **0.625/0.625/0.667**. At **300 s**, baseline-minus-queue MAE **−429.5 [−645.2,−213.3] veh-s**. No coupling adopted. Raw-WM comparison is separately limited to **six complete worlds/36 contrasts**, never pooled with the full panel. <!-- Source: E:/agent-work/c4-wmqueue/docs/paper_results/WMQUEUE.md:3-32. -->
- **Scope:** conservation checks do not guarantee admissions/effects fidelity. Failure of this fixed model does not falsify every structural queue model or decide WMVALUE's different test.

## Paper B boundaries relevant to Paper A

### B-TRANSFER — Historical prediction success validates the primary WM-policy route
- **Status: CONTRADICTED as a transfer of evidence; historical result SUPPORTED within scope.** Recovered wall **60 s** teacher-forced WM−BB-traj **−9.98 [−16.78,−5.09] m**, **67 starts/29 worlds**, is controller transfer on seen worlds. It does not certify WM2/value-tail interaction or autonomous feedback. <!-- Source: E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:6-28. -->
- **Coverage/history:** old C1 and BBs saw the **40 PI/13 wall** diagnosis worlds under other arms; newer audited corpora saw none, while initialization lineage remains incomplete. Genuine **60 s** history on separate shifted PI starts gives WM2−BB **−45.05 [−52.73,−25.25] m** at **60 s**; original-start regression and reset-controller adverse feedback remain. This is a conditional extension, not a universal history cure or isolated coverage cause. <!-- Source: E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md:3-45. -->

### B-FLOOR — Unavoidable chaos explains the long-horizon model error
- **Status: CONTRADICTED as an established explanation at the tested amplitude.** Under native driving, **six worlds**, two one-step **+0.01 m/s** HDV pulses per world give median HDV position/speed errors exactly **zero through 120 s**. Endpoint **300 s** is unavailable because no original HDVs survive jointly, not zero error. This is amplitude/target-dependent sensitivity, not an irreducible RNG floor, general determinism proof or PI/MAPPO reference. <!-- Source: E:/agent-work/c4-c1floor/docs/paper_results/C1FLOOR.md:7-22. -->

### B-SWITCH — Controller diversity proves WM-guided switching improves traffic
- **Status: PENDING for switching efficacy; headroom SUPPORTED descriptively.** Recorded **90-world J600** winners native/CTRL0/MAPPO/PI/CIRCLES **22/17/19/29/3**; the figure already exists at `E:/agent-work/c4-c1switch/docs/paper_results/c1switch/recorded_winner_by_condition.png`. Exact-rerun **16-state** headroom passes, but median best-to-second margins are **1.832371/6.853775 veh-s** at **60/90 s**. Different winners do not establish predictability, switching gains or cooperative credit. Latest ledger: corrected canary RUNNING after anonymity/seeding repair; main remains gated, no efficacy outcome supplied. <!-- Sources: E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_WINNERS.md:4-19,31-42; E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md, MAIN20:52 and v7 STAGED / WM PARITY PASS / CORRECTED CANARY RUNNING. -->

### B-CLOSED — Current complete autonomous feedback validates the WM
- **Status: PENDING.** Producer is RUNNING; complete-panel predictions, world-bootstrap CSV/PNG and verdict remain missing. Do not quote partial-panel performance as a headline or substitute completion counters for scientific outcomes. [[PENDING: complete frozen feedback panel / S-C1CLOSED existing collectors]]. <!-- Source: E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md. -->

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
