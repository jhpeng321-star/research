# Paper A — paragraph-level drafting outline

**Headline:** Do CAVs really cooperate? Measuring and learning cooperation in mixed-traffic merging with exact counterfactual replays.

**Owner target:** IEEE IV 2027; six pages including references; deadline 2026-11-15. Use IEEEtran conference two-column layout on conversion. Actual pagination and official rules remain pending. <!-- Source: owner S-PAPERA task, 2026-10-31. -->

All bullets below specify the exact claim a paragraph may make, evidence/claim IDs, and required insertions. This is drafting scaffolding, not prose. A positive learning result is a question until L-RISE and L-WIN resolve. Keep the four contribution slots, but describe completed measurements and prospective tests accurately.

## Editorial page budget (targets, not measured pagination)

- Title, abstract and I Introduction: 0.75 page.
- II Related work: 0.45 page.
- III Counterfactual definition/protocol: 0.95 page, including F1.
- IV Audit and learner methods: 0.90 page, including compact T1.
- V Results: 1.70 pages, including F2/F3/F4 as compact panels.
- VI Discussion, limitations and conclusion: 0.45 page.
- References: 0.80 page. Total target: 6.00 pages.
- Put detailed detector taxonomy, all cells, every candidate, complete physical vector, and budget lineage in a linked reproducibility artifact. Main text must still show primary adverse/null outcomes and essential coverage. Merge figure panels rather than shrinking labels to fit.

## Abstract — one paragraph

- **AB-P1 exact claim:** “We operationalize cooperation as excess group delay saving over singleton savings in matched simulator branches; we audit existing controllers, test an oracle-credit warm-started learner, and assess a learned substitute's horizon-dependent fidelity.” Use “test,” not “show increases,” until supported. Add the actual primary audit/learning outcome and explicitly state the model's limitation. Map M-DEF, A-PANEL, L-RISE/L-WIN, W-SHORT/W-LONG.
- **Insert:** [[PENDING: headline speed-only audit C_G and paired successor interaction/efficacy with CIs / S-COOPMEASURE, S-SUMOCF, S-COOPEVAL]]. Do not fill with COOPMEAS2's different checkpoint/panel.

## I. Introduction

- **I-P1 exact claim:** Traffic improvement and visibly coordinated maneuvers do not establish superadditive group interaction. Explain why a merge can shift delay upstream/sourceward. Evidence M-DEF, M-COHORT; motivating historical contrast A-OLD. Avoid claiming existing controllers categorically do not cooperate.
- **I-P2 exact claim:** A simulator permits a fixed-state intervention comparison that a learned model can approximate incorrectly. State the audit/learning/deployment distinction and “exact within the replay contract,” not exact real-world causal effects. Evidence M-EXACT, W-LONG/W-DEPLOY.
- **I-P3 exact claim:** Contributions are (a) operational interaction metric and replay protocol, (b) a controller audit with a coordinated wall mechanism control, (c) a warm-started per-CAV learning experiment using long-horizon group credit, and (d) surrogate-fidelity diagnosis. Contribution (c) is an experiment, not a guaranteed gain; (d) includes failure. Evidence all corresponding CLAIMS sections. Mention prior WM-trained drift and historical inconclusive interaction briefly; do not bury them.

## II. Related work — three compact paragraphs

- **II-P1 exact claim:** Difference rewards, COMA, Dr.Reinforce and MACD establish counterfactual credit-assignment precedent; our C_G measures total group excess over singleton effects and is not the same as an agent's marginal-in-team reward. Cite verified entries from `RELATED_WORK.md`. No first-counterfactual claim.
- **II-P2 exact claim:** Mixed-autonomy/merging MARL, AV Shapley reallocation and Flow/CIRCLES establish control and cooperation precedent. Our intended distinction is intervention-specific simulator-replayed measurement rather than inferring interaction from return or coordination signatures. Do not assert those papers lack all causal analysis unless full text confirms it.
- **II-P3 exact claim:** Counterfactual-planning cooperation measurement, process-aware evaluation and world-model admissibility are close evaluation precedents; physics-informed offline traffic MARL is close model/control precedent. State complementary operational scope and the need to validate surrogate intervention effects. Cite verified works; spell out any remaining full-text comparison gaps.

## III. What does “cooperate” mean?

- **III-P1 exact claim:** Define scenario, permitted observations, simulator background B, selected G, time of selection, intervention duration and outcome horizon. Disabled audit members release to native; controller memory/context remain correctly masked. Evidence M-BASE/M-COHORT. **[[PENDING: selected SNR design and exact new audit contract / S-SUMOCF, S-COOPMEASURE]].**
- **III-P2 exact claim:** Define full fleet+source J, saved D(S), C_G and marginal-in-team M_i=D(G)−D(G\{i}). State sign/units, group-benefit and individual-benefit distinctions, and why positive C_G alone can be gamed by harmful solos. Evidence M-DEF/L-ANTIGAME. Algebraic interpretation: beyond-singleton aggregate, not general intent, altruism or Shapley fairness.
- **III-P3 exact claim:** Exact replay requires physical state/RNG, exogenous requests, controller memory/dwell, branch ownership and native-repeat checks. Same seed alone is insufficient. New audit repeat canaries and learner saved-branch/zero-update parity are now reported passing; full-panel integrity remains pending. Evidence M-EXACT; F1. <!-- Sources: E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md; E:/agent-work/c4-sumocf/tools/sumocf/evidence/CANARY_V2.json. -->
- **III-P4 exact claim:** Estimate paired world-level outcomes retaining tiny/null eligible groups, report missingness and safety, and keep local delay secondary/control-variate-only under the frozen rule. World clusters, source weighting and pointwise uncertainty are protocol-specific. Evidence M-COHORT/L-SNR. **[[PENDING: frozen bootstrap, selected-design SHA and analysis receipt / S-COOPMEASURE, S-COOPEVAL]].**

## IV. Audit and learning experiments

- **IV-P1 exact claim:** Controller versions and actuators differ: native release, forced-command CTRL0, speed-only MAPPO, PI, CIRCLES and full-context wall. Frozen wall roster is kept in singleton commands so refusal is not confused with lost coordination. Evidence A-PANEL/A-WALL; T1. **[[PENDING: controller hashes and wall coverage/abort evidence / S-COOPMEASURE]].**
- **IV-P2 exact claim:** The amended learner initializes from speed-only MAPPO, adds an observable RSU coordination latent and fine-tunes a bounded actor subnetwork under the original safety executor and native lane ownership. Evidence L-METHOD. Supersedes the historical from-scratch/no-imitation design. **[[PENDING: actual implementation, zero-addition parity and observation schema / S-SUMOCF]].**
- **IV-P3 exact claim:** Shared-plus-independent latent exploration and matched marginal joint scores are used for local sampled-window credit, with singleton/group/flip/all terms and anti-sabotage weights. Full-correlation mean shifts are explicitly a common-mode projected score surrogate; do not claim a general unbiased gradient. The probe freezes group/window, while training continues sampling correlation. Outcome credit runs **600 s after sampled intervention start**; this is not the old fixed-end episode or a differentiable WM objective. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md:22-74. --> **[[PENDING: nominal probe winner/window/group, SNR table, actual weights/updates and charged totals / S-SUMOCF]].**
- **IV-P4 exact claim:** Development/checkpoint selection and final frozen comparisons are separate; matching is required for claims about credit, correlation or richer RSU inputs. Distinguish native-release audit C_G from learner mean-background exploration C_G. Evidence M-BASE, R-FAIR, L-RISE/L-WIN. **[[PENDING: capacity/input/budget-matched controls, training seeds and freeze/exposure receipts / S-SUMOCF, S-COOPEVAL]].**

## V. Results — keep these questions separate

### A. Do existing controllers cooperate? (F2 primary; historical F2H available)
- **V-A-P1 exact claim:** Report full speed-only controller audit C_G and its uncertainty alongside D_G/ΣD_i, positive fraction and coverage; interpret wall as a measured control, not an assumed positive. **[[PENDING: complete audit / S-COOPMEASURE]].** Do not fill with whole-controller SPEEDHEAD efficiency.
- **V-A-P2 exact claim:** Historical COOPMEAS2 could not establish beneficial interaction or advantage over MAPPO, despite fleet gains. Its primary **300 s** paired C_G difference **+327 [−395,+1035] veh-s** has half-width **715.2 veh-s**, on **28 worlds**. Include the adverse short-clock estimate in the caption/artifact. Explain controller/panel difference. <!-- Source: E:/research-dense/docs/paper_results/COOPMEAS2.md:12-20,39-62. --> Claims A-OLD/A-ADVERSE; F2H.

### B. Does learning improve interaction and traffic? (F3)
- **V-B-P1 exact claim:** State final frozen successor C_G, paired change from the same MAPPO audit, team-cost comparison and all declared candidates; resolve cooperation and efficacy independently. **[[PENDING: paired frozen candidate results and CIs / S-SUMOCF, S-COOPMEASURE, S-COOPEVAL]].** Positive training reward alone fills neither result.
- **V-B-P2 exact claim:** Attribute any gain only as far as controls permit. Report credit-off, exploration and input/capacity controls, actual budget, physical vector and training-seed scope. **[[PENDING: ablations, safety/fairness and lineage / S-SUMOCF, S-COOPEVAL]].** If unavailable, explicitly state the attribution gap.
- **V-B-P3 exact claim:** Prior WM-trained drift and Track 3's zero final head explain the oracle-learning motivation; they are not successor successes. Track 3B later resumed from its retained MAPPO canary, completed a baseline and rejected its first real update with zero-head rollback; its final search outcome remains pending. Historical cycle-1 failed its MAPPO comparator. Claims L-DRIFT/L-TRACK3/L-OLDWIN. Complete A/B/B-L inventory is **[[PENDING: original candidate A and pilot receipt inventory / owner integrator, S-PILOTDIAG]].** Do not call pilot imagined rescoring a real-SUMO outcome. <!-- Source: E:/agent-work/c4-sumoes/docs/paper_results/SUMOES_RUN.md:1-11. -->

### C. Can WM-S replace the oracle? (F4)
- **V-C-P1 exact claim:** Historical direction fidelity was bounded by horizon, but the corrected native-lane contract failed qualification. Plot separate panels rather than one common-model curve. Claims W-SHORT/W-LONG; F4. “Accurate to about **90 s**” applies only to the old panel. <!-- Sources: E:/research-dense/docs/paper_results/E2_direction_horizon.csv; E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md. -->
- **V-C-P2 exact claim:** Source/entry correction reduces source error but leaves the upstream/merge-road residual and can worsen net error by removing cancellation. Quantify measured regional errors; localization does not prove a unique causal merge-queue defect. Claim W-ENTRY. **[[PENDING: direct predicted-versus-exact C_G, calibrated deployment abstention and latency / S-WMREQUAL3, S-WMFIXMERGE, S-COOPEVAL]].**

## VI. Discussion and conclusion

- **VI-P1 exact claim:** Measurement is intervention-, baseline-, group- and horizon-dependent. Exact SUMO replay does not give real-road causality; behavior exemplars and signed error components are not causal mediation. Mention observer coverage, finite precision, exploratory selection and uncertain safety explicitly. See `THREATS.md`.
- **VI-P2 exact claim:** Conclude only resolved contributions: operational cooperation audit, learning outcome (positive, null, or adverse as delivered), and measured limits of a candidate surrogate. **[[PENDING: final outcome-specific conclusion / owner after S-COOPMEASURE and S-SUMOCF deliveries]].** Do not conclude deployable replacement or MAPPO superiority while pending.

## Drafting acceptance checklist

- Paragraph claims map to `CLAIMS.md`; any newly inserted numerical result includes its actual source path in an HTML comment.
- No historical lane-on checkpoint or whole-controller efficacy plot is relabeled speed-only interaction evidence.
- Both interaction and team improvement are required for the intended positive learner claim; preserve adverse/null outcomes.
- Retain primary CIs, eligible-world denominator, missingness, native-repeat status, controller identities, horizon clock origin and cost sign in main captions.
- Verify every used citation key against `refs.bib`; unresolved reference identities never become author-year guesses.
- Render IEEEtran and count references within the page cap before calling the manuscript submission-ready.
