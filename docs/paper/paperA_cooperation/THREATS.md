# Threats, reviewer objections and bounded answers

## “The WM is no longer the method; SUMO does the learning.”
- **Answer:** The owner's primary method is WM-trained cooperation: short imagined subset branches plus a terminal cost-to-go supervised by real simulator outcomes. Exact SUMO is the verifier; oracle-trained S-SUMOCF is the backup/reference. An oracle gain cannot fill the WM-policy claim. Keep offline value prediction, WM policy training and exact traffic/interaction verification as separate gates.
- **Evidence:** owner MAIN ADDENDUM; `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md`; `CLAIMS.md` W-VALUE/W-POLICY/W-RISE/W-WIN.
- **Gap:** [[PENDING: decisive offline test, then separately registered WM policy and paired exact outcomes / S-WMVALUE, MAIN WM policy producer, S-COOPMEASURE, S-COOPEVAL]].

## “A real-trained terminal value just moves the long-horizon bias into a critic.”
- **Answer:** Real-outcome supervision avoids training the tail entirely from the WM's own biased long rollout; it does not ensure correct continuation values or accurate small differences across imagined states. The frozen offline gate compares short WM+V with full WM, command-aware value-only and FIT-only arm/time baselines on common reserved TRAIN worlds. Missing causal sensing/queue telemetry yields INCOMPLETE, not an enriched global-state substitute.
- **Evidence:** `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md`, observation/label gate, models and decisive comparison.
- **Remaining objection:** offline data use recorded fixed-controller continuation and arm-specific states after the common warmup. A new cooperative policy and its imagined terminal states may be off distribution; unconditioned V may alias continuation actions. Better offline pair prediction is not same-state C_G accuracy, a policy-gradient theorem or a learned-policy result.
- **Gap:** [[PENDING: observation/label admissibility, full baseline results, value time/continuation contract and transfer to WM policy branches / S-WMVALUE, MAIN WM policy producer]]. Do not tune on the reserved worlds or interpret pair/time rows as independent samples.

## “The WM recovers a percentage of an oracle gain that may be zero.”
- **Answer:** Show matched WM−MAPPO and oracle−MAPPO effects first. A gain-retention ratio requires compatible metrics, clocks, worlds, controller/input contracts and an established beneficial oracle denominator with uncertainty. If the denominator is absent, nonpositive or poorly resolved, report the ratio as not estimable. A finite-budget oracle-trained policy is an empirical reference, not a certified optimum or mathematical upper bound.
- **Evidence:** `CLAIMS.md` W-RATIO; owner MAIN ADDENDUM.
- **Gap:** [[PENDING: corrected oracle/WM policies and jointly paired gain/ratio uncertainty / S-SUMOCF, MAIN WM policy producer, S-COOPEVAL]].

## “This is just difference rewards / Shapley / COMA.”
- **Answer:** Counterfactual credit is established prior art. Our proposed object is group excess above singleton effects under an explicit longitudinal simulator intervention and full-population cost, followed by a controller audit and learning test. M_i and C_G are different; C_G is neither a Shapley allocation nor evidence of intentional cooperation. For larger groups it aggregates beyond-singleton interaction terms, not only the highest-order term.
- **Evidence:** `RELATED_WORK.md`; `CLAIMS.md` M-DEF/R-NOVEL. The equation alone is not sufficient novelty; avoid first-of-kind claims pending full closest-paper comparison.

## “Positive synergy could reward two bad solo controllers.”
- **Answer:** Correct. Present C_G, D_G, ΣD_i and solo gain distributions together. A positive interaction with a harmful group is not beneficial traffic control. The weighted objective's singleton guard prevents a negative expanded singleton incentive but cannot guarantee all learned solos help.
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, credit; `E:/agent-work/c4-papera/docs/WM_C4_DESIGN.md`, late cooperation update.
- **Gap:** [[PENDING: executed objective audit and final solo/group outcomes / S-SUMOCF]].

## “The baseline changed, so cooperation rose only by definition.”
- **Answer:** Native-release audit and checkpoint-mean exploration credit are different estimands. State B and all nonmember/entrant/continuation actions in each caption. A training-credit curve cannot establish policy-level cooperation relative to native. Compare final policy and initial MAPPO under the same audit worlds/groups/background.
- **Evidence:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`; `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`.
- **Gap:** [[PENDING: common-background frozen-policy audit / S-COOPMEASURE, S-SUMOCF]].

## “Exact replay is only a common seed, or leaks privileged state.”
- **Answer:** Branching requires physical and RNG state, identity-coupled requests, controller memory/dwell and ownership release. Simulator state is an offline intervention label; actors must use only declared CAV/RSU observations. Historical native-repeat verification supports that historical implementation, not a new one. Full-state diagnostic initialization is not sparse-sensor deployment validation.
- **Evidence:** `E:/research-dense/docs/paper_results/COOPMEAS2.md`; `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`; `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`.
- **Gap:** [[PENDING: complete-panel integrity, observation binding and memory/entrant receipts / S-COOPMEASURE, S-SUMOCF]]. Replay exactness remains simulator- and execution-contract-specific.
- **Current counterexample:** the learner's earlier canary passed, but later zero-noise saved-state branches failed at the first physical step despite identical commands. Original probe credits are invalid. Corrected full-prefix reruns require state/trajectory and zero-intervention verification; the audit is held pending reviewed release and corrected binding. Full-prefix reruns alone are not proof of paired exactness. <!-- Sources: E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md; E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md. -->
- **Data lineage:** the audit producer disclosed that an early scanner opened legacy parameter JSON; it states no values were used and the scanner was corrected. Preserve the disclosure. This package did not open those files. <!-- Source: E:/agent-work/c4-coopmeasure/tools/coopaudit/evidence/EXPOSURE_NOTE.md. -->

## “Disabling a wall member changes its planner, invalidating the solo comparison.”
- **Answer:** The prospective audit retains the full wall roster/context and masks issued member commands; it does not request a singleton wall and call refusal its solo outcome. Record assembly failure, absent coverage and aborts. A coordinated wall mechanism can still have negative interaction or no fleet benefit.
- **Evidence:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`, wall provenance and singleton masking.
- **Gap:** [[PENDING: full-context mask/command and refusal/abort traces / S-COOPMEASURE]].

## “Groups or local delay were selected after observing gains.”
- **Answer:** Freeze geometric eligibility/identities before intervention; keep missing and tiny/null gains. Publish SNR selection rule/table and its exposure boundary. Selection by D_G SNR does not guarantee C_G precision. Local control variates need frozen centering/beta; a policy-dependent expectation can bias credit. Full fleet+source outcome remains primary.
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, SNR gate; `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`, WAIT_SUMOCF/coverage.
- **Current evidence:** the initial CV-based selection violated the raw-D_G rule and was corrected deterministically; subsequently the saved-state branch credits themselves were invalidated. Preserve both defects and all original receipts. No original winner establishes a usable signal. [[PENDING: artifact-approved full-rerun probe table/selection SHA and corrected audit binding / S-SUMOCF, S-COOPMEASURE]]. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md; E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md. -->

## “More vehicles/ticks make the confidence interval artificially narrow.”
- **Answer:** Use paired world resamples and the original stratification/weighting; worlds, not vehicles/ticks or repeated checkpoints, are independent experimental units. Preserve pointwise/unadjusted secondary interpretation. Historical wide interaction uncertainty is evidence of limited precision, not absence of an effect.
- **Evidence:** `E:/research-dense/docs/paper_results/COOPMEAS2.md`; `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`; `E:/agent-work/c4-coopeval/tools/coopeval/stats.py`.
- **Gap:** [[PENDING: new complete-panel inference and fixed selection/stop accounting / S-COOPMEASURE, S-COOPEVAL]]. No post-result sample enlargement is justified by the old planning projection.

## “More training, RSU inputs or correlation caused the gain, not cooperation credit.”
- **Answer:** A warm-started richer-input actor compared with frozen speed-only MAPPO tests the combined method. Credit-specific attribution additionally needs capacity/input/additional-budget-matched continued MAPPO, credit-off, and correlation controls, with the same safety executor. WM-specific attribution needs no-WM/value-only and no-tail/long-WM controls. Separate warm-start and additional training cost; include real label acquisition, value fitting, inference and failed branch costs.
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`; `E:/agent-work/c4-papera/docs/WM_C4_DESIGN.md`.
- **Gap:** [[PENDING: WM/oracle matched controls, seed replication and total-budget lineage / S-WMVALUE, MAIN WM policy producer, S-SUMOCF, S-COOPEVAL]]. No equal-budget efficiency ratio is currently established.
- **Related estimator objection:** correlated exploration requires the correct marginal joint score. The amended fully correlated case is a common-mode projection, not an unrestricted individual-mean score; frozen continuation and any changing local-control-variate expectation also limit unbiased-gradient claims. Report this as the implemented local surrogate unless an applicable proof is supplied. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md:22-60. -->

## “Your evaluation is used for checkpoint selection.”
- **Answer:** Label iterative dev checks and selected exemplars exploratory. Report all declared candidates and stop reasons; do not treat the best dev checkpoint as prospectively frozen confirmation. Baseline warm-start hash and final policy freeze must precede any claimed independent confirmation. Existing historical panels and new speed-only panels are not pooled.
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`; `E:/research-dense/docs/paper_results/E3FINAL_A.md`; `EVIDENCE_SOURCES.md`.
- **Gap:** [[PENDING: final freeze/exposure receipt and permitted independent confirmation / owner integrator, S-COOPEVAL]].

## “Native, CTRL0 and the two MAPPO versions are conflated.”
- **Answer:** Native releases longitudinal control; CTRL0 issues commands through an adapter. Historical lane-head MAPPO and current speed-only MAPPO are separate artifacts. Record checkpoint, wrapper, actuator mode, lanes, safety limits and realized execution per arm. Native-relative gain is not MAPPO superiority.
- **Evidence:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`; `E:/agent-work/c4-speedhead/docs/paper_results/SPEEDHEAD.md`; `E:/research-dense/docs/paper_results/COOPMEAS2.md`.
- **Gap:** [[PENDING: final per-arm execution table / S-COOPMEASURE, S-COOPEVAL]].

## “Why claim a short-horizon WM success if corrected qualification fails?”
- **Answer:** Do not make an unqualified success claim. The historical direction-only panel and corrected recorded native-lane intervention panel have different populations/contracts/checkpoint bindings. Preserve both, in separate panels. The proposed short WM+real-value route needs its own verification; shortening imagination does not itself resolve the corrected failure. Direction alone is insufficient for effect magnitude, safety, policy ranking or interaction.
- **Evidence:** `E:/research-dense/docs/paper_results/E2_direction_horizon.csv`; `E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md`.
- **Gap:** [[PENDING: WMVALUE gate and checkpoint-specific WM+value/native-release C_G validation / S-WMVALUE, MAIN WM policy producer, S-COOPEVAL]]. Current support is a bounded historical result plus a corrected failure.

## “The merge queue is not a causally isolated failure.”
- **Answer:** Correct. Regional signed covariance contributions localize error; they do not estimate causal mediation. Source and upstream error cancellation explains why removing source error can worsen total error. The tested entry-path contrast supports residual upstream mismatch, not a unique physical cause or a completed fix.
- **Evidence:** `E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md`, regional accounting; `E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_RETARGET.md`.
- **Gap:** [[PENDING: trained-head verdict and intervention-matched surrogate comparison / S-WMFIXMERGE, S-COOPEVAL]].

## “Signatures of yielding or synchrony are being called causal cooperation.”
- **Answer:** Behavior detectors describe geometry and timing. Accepted-merge timing/gap differences require valid paired reference branches with explicit native-release or policy-mean semantics; cross-lane overlap alone is a wall-like pattern, not useful metering. Delivered whole-controller counts/exposure are descriptive. Original SUMOCF positive-credit discovery uses invalidated costs and cannot support beneficial cooperation, even if geometric detector outputs reproduce. Corrected outcome-selected exemplars require denominators, censoring and selection disclosure.
- **Evidence:** `E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md`.
- **Gap:** [[PENDING: corrected artifact-valid paired traces and matched coverage/censoring / S-COOPBEHAV, corrected trace producer]]. <!-- Sources: E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md; E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md. -->

## “Efficiency hides source queues, unfairness or safety deterioration.”
- **Answer:** Use all requested service, entrant/source bookkeeping and road delay together; report ramp/mainline service, worst-group delay, braking/TTC, collision/teleport/dropped-step totals. Preserve original observed safety outcomes. Incident zeros are not equivalence, certification or a risk bound.
- **Evidence:** `E:/research-dense/docs/paper_results/E3FINAL_A.md`; `E:/research-dense/docs/paper_results/COOPMEAS2.md`; `E:/agent-work/c4-speedhead/docs/paper_results/SPEEDHEAD.md`.
- **Gap:** [[PENDING: successor safety/fairness vector and comparator-consistent noninferiority design if claimed / S-COOPEVAL]].

## “Results cannot generalize to real heterogeneous traffic.”
- **Answer:** Simulator-only evidence on one preserved bottleneck with structured human dynamics has limited external validity. The WM prior shares simulator structure; unseen controller, unseen world and unseen penetration are different claims. Single-checkpoint development does not estimate training-seed variability. RSU coverage, communication delay and runtime latency require separate validation.
- **Evidence:** `E:/agent-work/c4-papera/docs/WM_C4_DESIGN.md`; `E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md`; prior checklist `E:/research-dense/docs/paper/CLAIM_EVIDENCE_CHECKLIST.md`.
- **Gap:** [[PENDING: permitted cross-network/driver/seed/deployment evidence / owner]]. No new experiments are part of this writing task.

## Honest result routing for the owner

- WMVALUE PASS only: report improved offline continuation prediction; WM-policy cooperation/efficacy remain unresolved until separately trained and exactly verified.
- WMVALUE FAIL or INCOMPLETE: state the decisive result or missing telemetry explicitly. Keep WM as the primary tested route; a successful backup is labeled SUMO-trained.
- Only oracle backup improves: report that exact-credit learning succeeds within its stated scope; WM learning remains unsupported. No gain-retention ratio without comparable WM outcomes.
- A policy improves fleet cost only: report control improvement without claiming interaction discovery. A policy improves C_G only: report interaction alongside D_G/solos without MAPPO superiority.
- Neither improves or intervals remain wide: preserve negative/inconclusive results and reliability limits. Track 3B is already a completed no-accepted-gain result; oracle probe artifacts are mechanics failures, not negative cooperation estimates. Package completeness does not promote pending scientific claims.
