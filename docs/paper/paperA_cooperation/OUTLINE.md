# Paper A — paragraph-level drafting outline

**Headline:** Do CAVs really cooperate? Measuring and learning cooperation in mixed-traffic merging with exact counterfactual replays.

**Owner target:** IEEE IV 2027; six pages including references; deadline 2026-11-15. Use IEEEtran conference two-column layout on conversion. Actual pagination and official rules remain pending. <!-- Source: owner S-PAPERA task, 2026-10-31. -->

All bullets below specify the exact claim a paragraph may make, evidence/claim IDs, and required insertions. **Primary method: WM-trained cooperative policy, using short WM counterfactual rollouts plus a real-SUMO-trained terminal cost-to-go. Backup/reference: SUMO-oracle-trained policy.** This owner framing supersedes the earlier oracle-first outline. Positive WM learning awaits W-VALUE, W-POLICY and W-RISE/W-WIN; backup efficacy awaits L-RISE/L-WIN. Design, offline prediction and policy efficacy are separate claims. <!-- Source: E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md, OWNER FRAMING and S-WMVALUE pre-registration; owner MAIN ADDENDUM. -->

## Editorial page budget (targets, not measured pagination)

- Title, abstract and I Introduction: 0.75 page.
- II Related work: 0.45 page.
- III Counterfactual definition/protocol: 0.95 page, including F1.
- IV WM-primary method, offline gate and oracle reference: 0.90 page, including compact T1.
- V Results: 1.70 pages, including F2/F3/F4 as compact panels.
- VI Discussion, limitations and conclusion: 0.45 page.
- References: 0.80 page. Total target: 6.00 pages.
- Put detailed detector taxonomy, all cells, every candidate, complete physical vector, and budget lineage in a linked reproducibility artifact. Main text must still show primary adverse/null outcomes and essential coverage. Merge figure panels rather than shrinking labels to fit.

## Abstract — one paragraph

- **AB-P1 exact claim:** “We operationalize cooperation as excess group delay saving over singleton savings; the primary learning route computes counterfactual group credit with short world-model imagination and a terminal value trained on real simulator outcomes, while exact SUMO comparisons audit cooperation and verify traffic gains.” This is a proposed/tested route until actual deliveries establish the outcome. Map M-DEF, W-VALUE/W-POLICY/W-RISE/W-WIN, A-PANEL and L-RISE/L-WIN. Include the WM qualification limitation and oracle artifact failure rather than implying all short rollouts are verified.
- **Insert:** [[PENDING: WMVALUE offline verdict, registered WM policy outcome and exact paired cooperation/efficacy CIs / S-WMVALUE, MAIN WM policy producer, S-COOPMEASURE, S-COOPEVAL]]. Backup comparison remains [[PENDING: corrected SUMO-trained policy / S-SUMOCF]].

## I. Introduction

- **I-P1 exact claim:** Traffic improvement and visibly coordinated maneuvers do not establish superadditive group interaction. Explain why a merge can shift delay upstream/sourceward. Evidence M-DEF, M-COHORT; motivating historical contrast A-OLD. Avoid claiming existing controllers categorically do not cooperate.
- **I-P2 exact claim:** Long-horizon WM error motivates short imagination plus a real-outcome value tail, rather than a value bootstrapped entirely inside the same biased WM. Exact simulator intervention labels remain essential for verification; their replay mechanics must themselves pass state-specific checks. Evidence W-LONG, W-VALUE, M-EXACT/M-ARTIFACT.
- **I-P3 exact claim:** Contributions are (a) operational interaction metric and replay protocol, (b) MAPPO/wall audit with historical optional controller evidence, (c) the WM-primary cooperative-policy experiment with SUMO training as backup/reference, and (d) horizon/value-tail reliability assessment. Claim success for (c) only after exact policy verification. Prior drift, historical inconclusive interaction and oracle artifact failure motivate the tests; none establishes the proposed remedy.

## II. Related work — three compact paragraphs

- **II-P1 exact claim:** Difference rewards, COMA, Dr.Reinforce and MACD establish counterfactual credit-assignment precedent; our C_G measures total group excess over singleton effects and is not the same as an agent's marginal-in-team reward. Cite verified entries from `RELATED_WORK.md`. No first-counterfactual claim.
- **II-P2 exact claim:** Mixed-autonomy/merging MARL, AV Shapley reallocation and Flow/CIRCLES establish control and cooperation precedent. Our intended distinction is intervention-specific simulator-replayed measurement rather than inferring interaction from return or coordination signatures. Do not assert those papers lack all causal analysis unless full text confirms it.
- **II-P3 exact claim:** Counterfactual-planning cooperation measurement, process-aware evaluation and world-model admissibility are close evaluation precedents; physics-informed offline traffic MARL is close model/control precedent. State complementary operational scope and the need to validate surrogate intervention effects. Cite verified works; spell out any remaining full-text comparison gaps.

## III. What does “cooperate” mean?

- **III-P1 exact claim:** Define scenario, permitted observations, simulator background B, selected G, time of selection, intervention duration and outcome horizon. Disabled audit members release to native; controller memory/context remain correctly masked. Evidence M-BASE/M-COHORT. **[[PENDING: corrected artifact-approved SNR design and WM-policy audit arm binding / S-SUMOCF, S-COOPMEASURE, MAIN]].** The producer's present audit reservation names the SUMO-trained arm only.
- **III-P2 exact claim:** Define full fleet+source J, saved D(S), C_G and marginal-in-team M_i=D(G)−D(G\{i}). State sign/units, group-benefit and individual-benefit distinctions, and why positive C_G alone can be gamed by harmful solos. Evidence M-DEF/L-ANTIGAME. Algebraic interpretation: beyond-singleton aggregate, not general intent, altruism or Shapley fairness.
- **III-P3 exact claim:** Exact replay requires physical state/RNG, exogenous requests, controller memory/dwell, ownership and branch identity checks. Earlier learner canaries passed, but later zero-noise saved-state branches failed: original SNR/interaction findings are artifacts. Corrected learner design uses complete same-seed prefix reruns, still requiring bitwise zero-intervention verification. Historical audit canaries do not license other states. Evidence M-EXACT/M-ARTIFACT; F1. <!-- Sources: E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md; E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md. -->
- **III-P4 exact claim:** Estimate paired world-level outcomes retaining tiny/null eligible groups, report missingness and safety, and keep local delay secondary/control-variate-only under the frozen rule. World clusters, source weighting and pointwise uncertainty are protocol-specific. Evidence M-COHORT/L-SNR. **[[PENDING: frozen bootstrap, selected-design SHA and analysis receipt / S-COOPMEASURE, S-COOPEVAL]].**

## IV. WM-primary learning and oracle reference

- **IV-P1 exact claim:** Proposed WM branch cost is the imagined fleet+source integral through a short horizon plus real-trained V at the terminal sensed state. Compute D and C_G from branch costs with the same background/group/noise contract. Proposed horizons are **60/90 s**; they are candidates for verification, not current-contract accuracy guarantees. V must have the correct remaining horizon and continuation conditioning. Evidence W-VALUE/W-POLICY. **[[PENDING: registered policy implementation, value input/continuation contract, actor warm start, RSU latent and matched exploration objective / MAIN WM policy producer]].** <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md; owner MAIN ADDENDUM. -->
- **IV-P2 exact claim:** WMVALUE first tests offline paired remaining-cost prediction: full WM versus short WM+V, command-aware no-WM V and FIT-only arm/time means. Real labels and observable sensing/queues are gates; test worlds are reserved TRAIN, not final confirmation. All declared baselines and both short horizons must be reported. PASS precedes separately registered policy training. Evidence W-VALUE; F3V. **[[PENDING: telemetry admissibility, decisive result and hashes / S-WMVALUE]].**
- **IV-P3 exact claim:** Backup S-SUMOCF warm-starts speed-only MAPPO, adds observable RSU latent capacity, uses persistent offsets about current means, correlated joint-score surrogate and long-horizon exact branch credit under original safety/native lanes. Full-correlation score is projected; no unbiased full-policy gradient claim. Backup horizon is **600 s after sampled intervention start**, unlike WMVALUE's fixed absolute endpoint. Original probe/selection is invalidated; optimizer remains held. Evidence L-METHOD/L-SNR/M-ARTIFACT. **[[PENDING: corrected gate, probe freeze and trained backup / S-SUMOCF]].** <!-- Sources: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md; E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md. -->
- **IV-P4 exact claim:** Controller audit primary arms are speed-only MAPPO/wall plus a trained oracle arm when delivered; the WM-trained arm still needs binding. Keep native/CTRL0 and historical lane-head/speed-only versions distinct. Freeze final comparisons and separate dev selection, matched-capacity/credit controls and actual budget accounting. An oracle-trained finite search is an empirical reference, not a mathematical upper bound. Evidence A-PANEL, R-FAIR, W-RATIO; T1. **[[PENDING: paired WM/oracle audit/evaluation contract and controls / MAIN, S-COOPMEASURE, S-COOPEVAL]].**

## V. Results — keep these questions separate

### A. Do existing controllers cooperate? (F2 primary; historical F2H available)
- **V-A-P1 exact claim:** Report full speed-only controller audit C_G and its uncertainty alongside D_G/ΣD_i, positive fraction and coverage; interpret wall as a measured control, not an assumed positive. **[[PENDING: complete audit / S-COOPMEASURE]].** Do not fill with whole-controller SPEEDHEAD efficiency.
- **V-A-P2 exact claim:** Historical COOPMEAS2 could not establish beneficial interaction or advantage over MAPPO, despite fleet gains. Its primary **300 s** paired C_G difference **+327 [−395,+1035] veh-s** has half-width **715.2 veh-s**, on **28 worlds**. Include the adverse short-clock estimate in the caption/artifact. Explain controller/panel difference. <!-- Source: E:/research-dense/docs/paper_results/COOPMEAS2.md:12-20,39-62. --> Claims A-OLD/A-ADVERSE; F2H.

### B. Does the WM-primary route work? (F3V then F3)
- **V-B-P1 exact claim:** Report WMVALUE gate and every baseline/horizon before policy outcomes. Better paired continuation prediction is necessary evidence for this route, but does not establish C_G fidelity or learned efficacy. If INCOMPLETE/FAIL, say so and keep policy outcome pending rather than substituting the oracle result. Claims W-VALUE/W-POLICY. **[[PENDING: offline verdict / S-WMVALUE]].**
- **V-B-P2 exact claim:** Report exact frozen WM-policy C_G change against MAPPO and team-delay/safety outcomes independently. Then compare the SUMO-trained backup on matching worlds/contracts and compute a gain-retention ratio only with an established positive oracle denominator. Claims W-RISE/W-WIN/W-RATIO and L-RISE/L-WIN. **[[PENDING: WM policy, corrected oracle policy, paired exact audit/evaluation and valid ratio / MAIN WM policy producer, S-SUMOCF, S-COOPMEASURE, S-COOPEVAL]].**
- **V-B-P3 exact claim:** Credit-off, no-WM/value-only, no-tail WM, input/capacity and exploration controls determine attribution. Include real-label acquisition and warm-start costs. Prior drift and historical failure motivate real-trained value testing, not a demonstrated cure. Track 3B finished with **four real rejected updates, zero accepted updates**, retaining MAPPO equivalence; the earlier CTRL0-base null is distinct. Claims L-DRIFT/L-TRACK3/L-OLDWIN/R-FAIR. Complete A/B/B-L inventory is **[[PENDING: original candidate A and pilot receipt inventory / owner integrator, S-PILOTDIAG]].** <!-- Source: E:/agent-work/c4-sumoes/docs/paper_results/SUMOES_RUN.md:1-10,36-68. -->

### C. Why short imagination plus a real-trained tail? (F4)
- **V-C-P1 exact claim:** Historical direction fidelity was bounded by horizon, but the corrected native-lane contract failed qualification. Plot separate panels rather than one common-model curve. Claims W-SHORT/W-LONG; F4. “Accurate to about **90 s**” applies only to the old panel. <!-- Sources: E:/research-dense/docs/paper_results/E2_direction_horizon.csv; E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md. -->
- **V-C-P2 exact claim:** Source/entry correction reduces source error but leaves the upstream/merge-road residual and can worsen net error by removing cancellation. Quantify measured regional errors; localization does not prove a unique causal merge-queue defect. Claim W-ENTRY. **[[PENDING: direct predicted-versus-exact C_G, calibrated deployment abstention and latency / S-WMREQUAL3, S-WMFIXMERGE, S-COOPEVAL]].**
- **V-C-P3 exact claim:** Completed behavior counts/exposure describe traffic patterns only. Original SUMOCF positive-credit discovery used now-invalidated branch costs and cannot validate beneficial cooperation. Corrected paired exemplars must bind valid credit and reference semantics. Claim A-BEHAV; F5. **[[PENDING: artifact-valid paired mechanism examples / S-COOPBEHAV, corrected trace producer]].**

## VI. Discussion and conclusion

- **VI-P1 exact claim:** Measurement is intervention-, baseline-, group- and horizon-dependent. Exact SUMO replay does not give real-road causality; behavior exemplars and signed error components are not causal mediation. Mention observer coverage, finite precision, exploratory selection and uncertain safety explicitly. See `THREATS.md`.
- **VI-P2 exact claim:** Conclude separately on measurement, the primary WM/value route, and SUMO-trained backup/reference. **[[PENDING: outcome-specific conclusion after WMVALUE and policy deliveries / owner, S-WMVALUE, MAIN WM policy producer, S-SUMOCF]].** A successful backup cannot be written as WM-trained cooperation; a failed or unavailable WM test remains central to the result.

## Drafting acceptance checklist

- Paragraph claims map to `CLAIMS.md`; any newly inserted numerical result includes its actual source path in an HTML comment.
- No historical lane-on checkpoint or whole-controller efficacy plot is relabeled speed-only interaction evidence.
- Both interaction and team improvement are required for the intended positive learner claim; preserve adverse/null outcomes.
- Retain primary CIs, eligible-world denominator, missingness, native-repeat status, controller identities, horizon clock origin and cost sign in main captions.
- Verify every used citation key against `refs.bib`; unresolved reference identities never become author-year guesses.
- Render IEEEtran and count references within the page cap before calling the manuscript submission-ready.
