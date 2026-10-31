# Threats, reviewer objections and bounded answers

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
- **Gap:** [[PENDING: producer canaries, observation binding and memory/entrant receipts / S-COOPMEASURE, S-SUMOCF]]. Replay exactness remains simulator- and execution-contract-specific.

## “Disabling a wall member changes its planner, invalidating the solo comparison.”
- **Answer:** The prospective audit retains the full wall roster/context and masks issued member commands; it does not request a singleton wall and call refusal its solo outcome. Record assembly failure, absent coverage and aborts. A coordinated wall mechanism can still have negative interaction or no fleet benefit.
- **Evidence:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`, wall provenance and singleton masking.
- **Gap:** [[PENDING: full-context mask/command and refusal/abort traces / S-COOPMEASURE]].

## “Groups or local delay were selected after observing gains.”
- **Answer:** Freeze geometric eligibility/identities before intervention; keep missing and tiny/null gains. Publish SNR selection rule/table and its exposure boundary. Selection by D_G SNR does not guarantee C_G precision. Local control variates need frozen centering/beta; a policy-dependent expectation can bias credit. Full fleet+source outcome remains primary.
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`, SNR gate; `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`, WAIT_SUMOCF/coverage.
- **Gap:** [[PENDING: selected table SHA, group/window/correlation freeze and raw versus CV evidence / S-SUMOCF, S-COOPMEASURE]].

## “More vehicles/ticks make the confidence interval artificially narrow.”
- **Answer:** Use paired world resamples and the original stratification/weighting; worlds, not vehicles/ticks or repeated checkpoints, are independent experimental units. Preserve pointwise/unadjusted secondary interpretation. Historical wide interaction uncertainty is evidence of limited precision, not absence of an effect.
- **Evidence:** `E:/research-dense/docs/paper_results/COOPMEAS2.md`; `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`; `E:/agent-work/c4-coopeval/tools/coopeval/stats.py`.
- **Gap:** [[PENDING: new complete-panel inference and fixed selection/stop accounting / S-COOPMEASURE, S-COOPEVAL]]. No post-result sample enlargement is justified by the old planning projection.

## “More training, RSU inputs or correlation caused the gain, not cooperation credit.”
- **Answer:** A warm-started richer-input actor compared with frozen speed-only MAPPO tests the combined method. Credit-specific attribution additionally needs capacity/input/additional-budget-matched continued MAPPO, credit-off, and correlation controls, with the same safety executor. Separate warm-start and additional training cost; include failures and acquisition.
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`; `E:/agent-work/c4-papera/docs/WM_C4_DESIGN.md`.
- **Gap:** [[PENDING: matched controls, seed replication and total-budget lineage / S-SUMOCF, S-COOPEVAL]]. No equal-budget efficiency ratio is currently established.

## “Your evaluation is used for checkpoint selection.”
- **Answer:** Label iterative dev checks and selected exemplars exploratory. Report all declared candidates and stop reasons; do not treat the best dev checkpoint as prospectively frozen confirmation. Baseline warm-start hash and final policy freeze must precede any claimed independent confirmation. Existing historical panels and new speed-only panels are not pooled.
- **Evidence:** `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`; `E:/research-dense/docs/paper_results/E3FINAL_A.md`; `EVIDENCE_SOURCES.md`.
- **Gap:** [[PENDING: final freeze/exposure receipt and permitted independent confirmation / owner integrator, S-COOPEVAL]].

## “Native, CTRL0 and the two MAPPO versions are conflated.”
- **Answer:** Native releases longitudinal control; CTRL0 issues commands through an adapter. Historical lane-head MAPPO and current speed-only MAPPO are separate artifacts. Record checkpoint, wrapper, actuator mode, lanes, safety limits and realized execution per arm. Native-relative gain is not MAPPO superiority.
- **Evidence:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`; `E:/agent-work/c4-speedhead/docs/paper_results/SPEEDHEAD.md`; `E:/research-dense/docs/paper_results/COOPMEAS2.md`.
- **Gap:** [[PENDING: final per-arm execution table / S-COOPMEASURE, S-COOPEVAL]].

## “Why claim a short-horizon WM success if corrected qualification fails?”
- **Answer:** Do not make an unqualified success claim. The historical direction-only panel and corrected recorded native-lane intervention panel have different populations/contracts/checkpoint bindings. Preserve both, in separate panels. Direction alone is insufficient for effect magnitude, safety, policy ranking or interaction.
- **Evidence:** `E:/research-dense/docs/paper_results/E2_direction_horizon.csv`; `E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md`.
- **Gap:** [[PENDING: final checkpoint-specific native-release C_G validation / S-WMREQUAL3, S-COOPEVAL]]. Current support is a bounded historical result plus a corrected failure.

## “The merge queue is not a causally isolated failure.”
- **Answer:** Correct. Regional signed covariance contributions localize error; they do not estimate causal mediation. Source and upstream error cancellation explains why removing source error can worsen total error. The tested entry-path contrast supports residual upstream mismatch, not a unique physical cause or a completed fix.
- **Evidence:** `E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md`, regional accounting; `E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_RETARGET.md`.
- **Gap:** [[PENDING: trained-head verdict and intervention-matched surrogate comparison / S-WMFIXMERGE, S-COOPEVAL]].

## “Signatures of yielding or synchrony are being called causal cooperation.”
- **Answer:** Behavior detectors describe geometry and timing. Accepted-merge timing/gap differences require valid paired member-release branches; cross-lane overlap alone is a wall-like pattern, not useful metering. Selected positive-C_G traces are exploratory exemplars with denominators, censoring and selection disclosed.
- **Evidence:** `E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md`.
- **Gap:** [[PENDING: paired traces, detector coverage and censoring / S-COOPBEHAV]].

## “Efficiency hides source queues, unfairness or safety deterioration.”
- **Answer:** Use all requested service, entrant/source bookkeeping and road delay together; report ramp/mainline service, worst-group delay, braking/TTC, collision/teleport/dropped-step totals. Preserve original observed safety outcomes. Incident zeros are not equivalence, certification or a risk bound.
- **Evidence:** `E:/research-dense/docs/paper_results/E3FINAL_A.md`; `E:/research-dense/docs/paper_results/COOPMEAS2.md`; `E:/agent-work/c4-speedhead/docs/paper_results/SPEEDHEAD.md`.
- **Gap:** [[PENDING: successor safety/fairness vector and comparator-consistent noninferiority design if claimed / S-COOPEVAL]].

## “Results cannot generalize to real heterogeneous traffic.”
- **Answer:** Simulator-only evidence on one preserved bottleneck with structured human dynamics has limited external validity. The WM prior shares simulator structure; unseen controller, unseen world and unseen penetration are different claims. Single-checkpoint development does not estimate training-seed variability. RSU coverage, communication delay and runtime latency require separate validation.
- **Evidence:** `E:/agent-work/c4-papera/docs/WM_C4_DESIGN.md`; `E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md`; prior checklist `E:/research-dense/docs/paper/CLAIM_EVIDENCE_CHECKLIST.md`.
- **Gap:** [[PENDING: permitted cross-network/driver/seed/deployment evidence / owner]]. No new experiments are part of this writing task.

## Honest result routing for the owner

- Audit complete, learner pending: the audit and operational measurement can be drafted as results; learning remains a clearly labeled unresolved experiment.
- Learner improves fleet cost only: report control improvement without claiming interaction discovery.
- Learner improves C_G only: report interaction change alongside D_G/solos; do not claim MAPPO superiority.
- Neither improves or intervals remain wide: preserve the negative/inconclusive learning result and the measured-surrogate limits. Package completeness does not convert pending scientific claims into completed contributions.
