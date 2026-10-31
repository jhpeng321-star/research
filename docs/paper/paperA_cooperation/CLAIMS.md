# Paper A — claim → evidence → status

Status vocabulary follows the existing checklist but uses the requested three labels: **SUPPORTED** (only the stated scope), **PENDING** (including inconclusive directional hypotheses), **CONTRADICTED** (completed evidence opposes the assertion). A supported design is not a supported outcome. Source reports were inspected, not rerun. `EVIDENCE_SOURCES.md` records unavailable deliveries; `REFRESH.md` records the latest cut.

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
- **Refresh:** New audit live-native/load/repeat, wall GROUP repeat and MAPPO GROUP repeat are reported bit-identical; learner receipt says `branch_mode=saved`, `saved_bitwise=true`, `zero_update_bitwise=true`. **SUPPORTED — these mechanics canaries only.** Full-panel replay/coverage remains [[PENDING: complete new audit and training integrity receipt / S-COOPMEASURE, S-SUMOCF]]. Same scalar seeds alone are insufficient. <!-- Sources: E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md:19-31; E:/agent-work/c4-sumocf/tools/sumocf/evidence/CANARY_V2.json. -->

### M-BASE — Audit and training use different counterfactual backgrounds
- **Exact claim:** The controller audit disables control to native release; CTRL0 is an always-commanded adapter. Learner credit instead compares a sampled latent-window intervention with checkpoint-mean continuation/background. These estimate different interventions.
- **Evidence:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`, intervention/release; `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, binding amendment, exploration/continuation and explicit `policy_mean_reference` handoff.
- **Status: SUPPORTED — design scope; actual learner binding PENDING.** Never pool learner exploration C_G with native-release audit C_G or call one an increase in the other.
- **Insert:** [[PENDING: confirm sampled-window B/subset/continuation semantics from actual learner receipts and audit the final frozen policy against native release / S-SUMOCF, S-COOPMEASURE]].

### M-COHORT — Entrants, source waiting and missing groups are explicit
- **Exact claim:** Team J includes fleet plus source delay; initial GROUP/ONLY interventions and entrant-controlling ALL are separately labeled. Missing or undersized groups are missing, not zero; tiny-gain eligible groups remain included.
- **Evidence:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`; `E:/research-dense/docs/paper_results/COOPMEAS2.md`.
- **Status: SUPPORTED — historical analysis/design.** [[PENDING: new audit coverage, group-size distribution, local-cohort definition and denominator / S-COOPMEASURE]]. Local observable-cohort delay cannot replace the fleet primary.

## Existing-controller audit

### A-PANEL — The prospective panel covers the named existing controllers
- **Exact claim:** Speed-only MAPPO, PI external-U, frozen CIRCLES damping and CTRL0 are planned audit arms; wall is a coordinated-mechanism control.
- **Evidence:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`.
- **Status: SUPPORTED — prospective design only.** [[PENDING: completed controller C_G, D_G, ΣD_i, fraction positive, coverage, safety and stratified world CIs / S-COOPMEASURE]].

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

## Learning contribution

### L-METHOD — Warm-started observable per-CAV counterfactual learner
- **Exact claim:** The amended method warm-starts speed-only MAPPO, trains its final hidden layer and mean row plus an observable RSU broadcast branch, and uses correlated latent exploration with matched subset branches.
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, binding amendment only.
- **Status: SUPPORTED — design plus zero-update mechanics, not learning efficacy.** Coordination latent is at most **four-dimensional**; exploration mixes **ρ∈{0,0.5,1}**. Full-rank cases use subset marginal joint scores; at full correlation the amended design explicitly calls the score a **common-mode projected surrogate**, since individual mean shifts can be off support. Training samples correlation per window even after a nominal probe winner is selected. Credit outcome horizon is **600 s after sampled start**. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md:10-60; E:/agent-work/c4-sumocf/tools/sumocf/evidence/CANARY_V2.json. -->
- **Scope:** Native lanes, original safety executor; not a new lane policy, from-scratch actor, parameter-ES learner, or already validated WM-trained successor. Zero-update parity and CPU sampled-score checks are reported passing. [[PENDING: final trained implementation/config/checkpoint hashes and executed objective receipt / S-SUMOCF]]. <!-- Sources: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md; E:/agent-work/c4-sumocf/tools/sumocf/evidence/CANARY_V2.json. -->

### L-SNR — Probe selection has established a usable cooperation signal
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, SNR gate; `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`, WAIT_SUMOCF.
- **Status: PENDING.** The design selects by |mean D_G|/sd(D_G), not directly by C_G precision. A good group-benefit signal does not automatically make an interaction difference precise.
- **Insert:** [[PENDING: measured SNR table, chosen group/window/correlation, actual sim-s, raw/CV comparison, frozen SHA and C_G precision / S-SUMOCF]]. Cross-fitted local control variates may become biased if their expectation changes with policy.

### L-ANTIGAME — The objective guards against worsening solo behavior
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, credit weights; `E:/agent-work/c4-papera/docs/WM_C4_DESIGN.md`, late update.
- **Status: SUPPORTED — algebraic guard, not a learned-outcome guarantee.** w_coop≤w_ind preserves nonnegative expanded singleton weight; report D_G and ΣD_i separately even when C_G improves. [[PENDING: actual weighted-objective accounting and observed singleton outcomes / S-SUMOCF]].

### L-RISE — Training increases measured cooperation
- **Evidence needed:** matched native-release C_G of initial MAPPO and every declared frozen successor on common worlds/groups, with shared world resamples and exposure history.
- **Status: PENDING.** [[PENDING: final candidate C_G, paired candidate−MAPPO interaction CI, eligible/positive fractions, all candidates and credit-off comparison / S-SUMOCF, S-COOPMEASURE, S-COOPEVAL]]. Training reward or adaptively sampled training C_G alone cannot establish policy-level improvement.

### L-WIN — The new learner beats MAPPO
- **Evidence needed:** complete paired team-delay and physical-vector dev exports, checkpoint-selection history and frozen candidate identity.
- **Status: PENDING.** [[PENDING: learned−MAPPO and learned−CTRL0 fleet+source cost CIs, later-clock ranking, subgroup and safety outcomes / S-COOPEVAL]]. Development selection is exploratory; a fresh fixed confirmation and training-seed replication remain separate gaps.
- **Contract issue:** current harness lists **120/300/600 s**; amended design requires **600/900 s after control start**. No later-clock value can be extrapolated from the older loader. <!-- Sources: E:/agent-work/c4-coopeval/tools/coopeval/__init__.py:13; E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md:88-93. -->
- **Latest amendment:** planned **six** dev checks now include a **three-world**, geometrically selected **pair** interaction diagnostic, rotating high-demand cells, alongside the full efficacy panel. That diagnostic uses exploration-background credit and differs from selected training groups and native-release audit groups. Rotating worlds do not supply a matched policy-level C_G learning curve. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md:95-107. -->

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
- **Status: CONTRADICTED — historical CTRL0-base search; PENDING for MAPPO-base learning.** Historical final head is zero with no accepted update. Track 3B's post-canary logging stop was repaired and explicitly superseded by continuation: its MAPPO-base panel baseline is complete and its first update is in progress. No completed learned-improvement result has landed. Screen-only rejections are not additional measured panel costs. <!-- Source: E:/agent-work/c4-sumoes/docs/paper_results/SUMOES_RUN.md:1-9,94-127. -->

### L-CREDIT — A prior cooperation-loss ablation proved causal benefit
- **Evidence:** `E:/research-dense/docs/paper/CLAIM_EVIDENCE_CHECKLIST.md`, E5-CREDIT (secondary report pointer).
- **Status: CONTRADICTED as a positive causal claim.** The checklist records byte-identical ON/OFF trajectories despite different commands. [[PENDING: direct E5 report binding and meaningful matched-capacity/credit-off successor ablation / S-COOPEVAL, owner integrator]].

## World-model substitute and reliability

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
- **Status: PENDING.** Existing total-effect fidelity and observability designs are insufficient. [[PENDING: checkpoint-bound counterfactual-interaction surrogate validation and runtime observability/latency / S-WMREQUAL3, S-WMFIXMERGE, S-COOPEVAL]]. Describe “candidate substitute” until resolved.

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
