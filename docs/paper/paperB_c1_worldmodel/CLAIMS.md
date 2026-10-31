# Paper B claim–evidence ledger

Extends `E:/research-dense/docs/paper/CLAIM_EVIDENCE_CHECKLIST.md`; that checklist is an older evidence cut, not overriding authority. Status refers to the exact claim below. Keep source comments when transferring numbers into prose. `[[PENDING: ... / agent]]` is an insertion request, never a result.

**Evening refresh:** use the recovered historical response result as the supported headline; full autonomous feedback superiority is still unresolved. Diagnosis, calibration, perturbation and switching-headroom results below supersede afternoon pending language. Exact source cut: `SOURCE_REFRESH.json`; review receipt: `NIGHT_SOURCE_REVIEW.md`.

## C1-FACTOR — Executable controllers can be evaluated separately from human dynamics

- **Status: SUPPORTED (architecture).** WM-S factors controller code from the learned traffic transition. In feedback evaluation, controllers receive predicted observations; teacher-forced CAV motion is a different contract.
- **Evidence:** `E:/research-dense/docs/WM_STATE_PLAN.md`, Model; `E:/agent-work/c4-paperb/src/wms/model/{factory,spec}.py`; main checklist C1-FACTOR.
- **Exact scope:** IDM-style longitudinal core, checkpoint-dependent MOBIL/learned lane rules, history-conditioned settings and gated residual, open-road entry/exit and merge machinery. Architecture does not prove arbitrary-controller accuracy, plant parity, or sparse-sensor deployment.
- **Draft insertion:** `[[PENDING: final checkpoint/config/source hashes, requested→issued→realized actuation and lane/entrant ownership / S-C1CLOSED]]`.

## C1-WALL — Historical structured prediction beats the in-house black boxes on unseen wall response

- **Status: SUPPORTED (historical, teacher-forced; recovered report).** At the reported wall clock of **60 s**, median HDV absolute longitudinal position error is **8.71 / 25.41 / 40.07 m** for structured / BB-traj / BB-id, on **67 states from 29 worlds**. <!-- Source: E:/agent-work/c4-c1gap/docs/paper_results/c1gap/WALL_C1_RESULT.recovered.md:9,19,23,27,280. -->
- Paired **BB-traj minus structured** is **+9.98 [5.09,16.78] m**; **BB-id minus structured +31.78 [18.61,45.37] m**; reported world-bootstrap **95%** intervals. These are paired state-error contrasts, not differences of the displayed marginal medians. <!-- Source: E:/agent-work/c4-c1gap/docs/paper_results/c1gap/WALL_C1_RESULT.recovered.md:133-154,280. -->
- **Evidence identity:** recovered report plus adjacent `recovery.json`, source commit `c55520803f63648a4676f493dce62d5622f4baca`, reported Markdown SHA `173ade43c9009aeec1942cd011288de0e04746e80483208dccdd2c887f663ae7`. Recovery is not a new scoring run.
- **Caveats:** entries disabled; historical training includes evaluation worlds' other controller arms; no unseen-world claim. BB-traj uses recorded CAV motion, but BB-id learns CAV motion and ignores the teacher; do not call the latter teacher-matched. Wall cohorts overlap and must not be pooled. Wall/seen ratios measure cohort difficulty as well as transfer.
- **Correction to older checklist:** the original report was previously unavailable; recovered report now supplies paired intervals. The subsequently delivered `E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md` independently reconstructs the wall contrasts and supplies marginal world-bootstrap CIs for the displayed medians: **8.71 [7.10,13.61] / 25.41 [16.96,31.09] / 40.07 [33.61,54.68] m**. <!-- Source: E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:27. -->
- **Independent decision correction:** historical wall trajectory-error advantage does not imply wall selector advantage: short-clock regrets tie and the recovered report's “every wall beneficial” interpretation conflicts with its costs (native is optimal/tied). Entries are disabled and the historical ranking metric is road-only, not full fleet-plus-source cost. Use the corrected world-cluster reconstruction rather than the original start-resampled ranking intervals. Evidence: C1_GAPS.md, C1-PAIRED. `[[PENDING: unbroken from-initialization exclusion proof / S-C1GAP]]`.

## C1-PI — Diagnosis supports a conditional genuine-history extension, not autonomous PI superiority

- **Status: SUPPORTED (completed diagnosis, teacher-forced/conditional presence).** Original PI snapshot-start WM2−BB-traj common-ID position-error contrast at **60 s** is adverse: **+17.85 [3.24,31.27] m**; original-start **120 s** contrast is favorable, **−47.82 [−95.20,−27.75] m**. These are distinct clocks, not controller-choice evidence. <!-- Source: E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md:7-14. -->
- On separately registered shifted starts `t0′=t0+60 s`, genuine full-history WM2−BB-traj is **−45.05 [−52.73,−25.25] m at 60 s** and **−144.74 [−250.11,−36.22] m at 120 s**. The frequently quoted roughly **48 m** advantage belongs to the original-start later clock, not this shifted panel. <!-- Source: E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md:14,25-32. -->
- Within identical shifted PI starts, full history versus snapshot improves WM2 by **−13.47 [−26.92,−1.49] m at 60 s**. Later-start selection also changes rankings, so history alone does not explain the entire between-panel improvement. Full-history wall comparison remains inconclusive. <!-- Source: E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md:33-37. -->
- **Scope:** C1 `87ed02e0`, 13E `952ec059`, WM2 `b8797242`, frozen BB-traj `afaaa32c`; no baseline retraining. Recorded CAV teacher, conditional presence/exits, entries OFF and dormant mix. Shifted denominators differ by clock/survivors. Do not call this production WM2+mix or autonomous closed-loop superiority.
- **Adverse feedback retained:** reset-controller late PI feedback gives WM2−BB **+201.00 [141.73,623.94] m at 120 s**. Prior PI command state is unbound: a conditional-reset stress test, not a fair continuation or isolated HDV-model error. `[[PENDING: fully initialized autonomous PI/Wall feedback comparison / S-C1CLOSED]]`. <!-- Source: E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md:39-45. -->

## C1-COVERAGE — Historical/newer differences isolate architecture or history

- **Status: CONTRADICTED as an isolated causal attribution.** Old C1 and frozen BBs saw other-controller training from **40/40 PI and 13/13 wall diagnosis worlds**; audited 13E/WM2 corpora overlap **0/40 and 0/13**. Historical C1 tests unseen controllers on seen worlds; newer comparisons also change world exposure. Corpus, weights and physics change together. <!-- Source: E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md:3-4. -->
- No tested component rollback restores the original PI advantage; runtime replay retains regression but is not exact archived parity. Diagnosis supports a major documented coverage confound plus a within-panel history effect, not a coverage-only causal proof or universal history fix. Evidence: C1PIDIAG.md, components/runtime/publication boundary.

## C1-CLOSED — Zero-shot full-feedback superiority to black-box prediction is complete

- **Status: PENDING (registered campaign running).** The delivered `E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md` exists; the task-named root alias is absent. Complete-panel estimates remain unavailable. Historical wall motion replay and partial task completion cannot supply this claim.
- **Bound design:** WM2+mix `b8797242`, frozen BB-traj `afaaa32c`, legacy BB-id `e12c81c0` and historical C1 `87ed02e0`; common longitudinal/gap-based ramp plant without ordinary CAV lane changes. BB-id keeps a different learned-CAV contract; production-native WM2 is a separate diagnostic. Cross-host exact parity failed and remains disclosed. Evidence: C1CLOSED.md and C1CLOSED_RUNTIME.md. No partial performance values enter the headline.
- **Required evidence:** controller exclusion before training/adaptation; identical starts/histories; actual controller feedback on imagined observations; native/controlled original and entrant lane ownership; command-to-plant parity; endpoint and effect errors; paired model contrasts with world uncertainty; baseline data/compute contracts.
- **Disposition:** retain the headline as the paper question; assert a completed full-feedback comparison only after `[[PENDING: eligible result and hashes / S-C1CLOSED]]`.

## C1-MAPPO — Strict-history offline controller choice has measured short-clock regret

- **Status: SUPPORTED (exploratory, exported feedback-selection panel).** On **32 starts / 21 worlds**, **60 s** `team` oracle regret is **13.64 [5.382,26.2] veh-s** for `wm13e`, versus **22.05 [10.05,37.86]** for BB-traj. Always-native and always-MAPPO regrets are **48.3 [26.93,70.98]** and **21.88 [9.728,37.69] veh-s**. <!-- Source: E:/research-dense/docs/paper_results/E1_C1_fair.md:159-199, selectors decisions/B_history_strict/{wm13e,bbtraj}/60/team. -->
- WM-selected minus always-MAPPO is **−8.241 [−18.99,−0.08363] veh-s** and minus always-native **−34.66 [−60.51,−8.163] veh-s**; source **95%** intervals. <!-- Source: E:/research-dense/docs/paper_results/E1_C1_fair.md:195-199; interval level: E:/research-dense/docs/paper/IEEE_IV_2027_DRAFT.md:107,133. -->
- **Evidence:** corresponding CSV selectors; main checklist C1-MAPPO and C1-PAIRED; results README.
- **Scope:** offline selection among evaluated controllers, not deployed receding-horizon selection or proof of a learned controller's efficacy. `team` cost needs its original implementation binding; do not silently rename it all-request fleet-plus-source J.
- **Refresh correction to the older checklist:** the current export DOES contain a paired strict-history team regret contrast: **WM-13E minus BB-traj −8.412 [−19.23,−0.3071] veh-s**, on the same **21 worlds / 32 starts**. This is **SUPPORTED as an exported exploratory paired advantage**, not a contrast inferred from marginal intervals. <!-- Source: E:/research-dense/docs/paper_results/E1_C1_fair.md:2100; exact values E:/research-dense/docs/paper_results/E1_C1_fair.csv, panel paired_decisions/B_history_strict/60/team/13e_minus_bbtraj, metric regret_oracle. -->
- **Independent reconstruction now SUPPORTED:** C1_GAPS reproduces the paired result using **2,000 whole-world percentile bootstrap draws**, seed **0**, retaining the original state-weighted estimand; equal-world sensitivity is **−8.85 [−19.72,−0.11] veh-s**. <!-- Source: E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:26,29. -->
- **Important metric distinction:** on the same B/MAPPO panel the common-HDV endpoint contrast is **−15.35 [−28.21,+1.45] m**, INCONCLUSIVE. Better exploratory regret is not established endpoint-error superiority on that panel. <!-- Source: E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:30. -->
- **Still pending:** `[[PENDING: exact team population/cost, candidate controllers, full initialization exposure and feedback execution binding / S-C1GAP + S-C1CLOSED]]`. Numerical pairing and its bootstrap recipe are available; complete zero-shot eligibility/full-feedback plant validation are not.

## C1-LONG — Relative black-box error gains establish long-clock decision trust

- **Status: CONTRADICTED as a general implication.** C1_GAPS reports supported relative absolute traffic-error gains on the later textbook panel while long-clock regret and common-HDV endpoint contrasts remain inconclusive; no common endpoint cohort survives at the longest reported clock. Do not conflate “beats a worse black box on traffic error” with accurate controller verdicts.
- **Evidence:** `E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md`, C1-PAIRED. These are 13E outputs, not current WM2 measurements; heterogeneous history/data prevent an isolated architecture attribution.

## C1-EXPOSURE — Controller exclusion is the unit of zero-shot eligibility

- **Status: SUPPORTED (scope constraint and newly audited corpora); full initialization lineage PENDING.** Same-world other-controller training is not unseen-world transfer. A checkpoint fitted to an evaluated controller's outcomes is adapted, even if initialized from an eligible earlier checkpoint.
- **New evidence:** `E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md`, exposure audit. C1FAST `87ed02e0` and its frozen BB corpus contain no wall/PI/MAPPO algorithm; configured but unused wall data are not actual exposure. Audited later `13E b2 FINAL 952ec059` and `WM2 c02 b8797242` corpora also exclude those algorithms; their B/C evaluation worlds do not overlap the audited training worlds. WM2 own-policy data are adaptation for that policy, not MAPPO exposure. No alias-to-evaluation binding is inferred solely from these abbreviated identities.
- **Remaining audit gap:** copied R16 initialization for WM2, fitted R9 initialization for 13E and an interrupted training stage prevent an unbroken from-initialization hash-linked exclusion proof. `[[PENDING: completed lineage audit and evaluation binding / S-C1GAP + S-C1CLOSED]]`. Do not use a model alias/date alone as proof.
- **Historical scope preserved:** C1FAST used the wall evaluation worlds under other arms, whereas the audited later B/C corpus relationships differ. Do not turn the historical caveat into a universal assertion that every later evaluation world was seen.

## WM-SETTINGS — Learned settings caused every historical result

- **Status: CONTRADICTED.** Historical `residual` variants did not wire inferred parameters into the physics; later `hybrid` variants do. The encoder can improve prediction without identifying true driver types.
- **Evidence:** `E:/research-dense/docs/WM_CHANGE_LOG.md`, R4-DIAG-HIST, R5, R5B-DIAG; local `src/wms/model/spec.py`, rollout-mode contract.
- **Permitted replacement: SUPPORTED (versioned architecture).** Describe learned settings as part of the later family, and label historical residual-only evidence separately.
- `[[PENDING: final-model ablation under the same controller/history/plant / S-C1CLOSED]]`.

## WM-LANES — All checkpoints share the same MOBIL and residual gate

- **Status: CONTRADICTED.** MOBIL, learned discretionary/hazard lanes, forced merge, and cumulative/saturating gates are version-dependent.
- **Evidence:** local `src/wms/model/spec.py`; `E:/research-dense/docs/WM_CHANGE_LOG.md`; WMFIX prior audit.
- **Permitted replacement:** IDM/MOBIL-style physics-informed family with explicitly checkpoint-bound lane, merge, gate, and admission components. A design sketch is not the final result configuration.

## HORIZON-SIGN — Frozen direction export supports a bounded reported pass

- **Status: SUPPORTED (named frozen panel only).** Tested **10/30/60/90 s** are PASS; **120/180 s** are FAIL. Accuracy at **90 s** is **0.8895 [0.8424,0.9341]**, at **120 s 0.7526 [0.6824,0.8187]**, and at **180 s 0.6263 [0.5474,0.7071]**, across **54 worlds**, with threshold **0.8684**. <!-- Source: E:/research-dense/docs/paper_results/E2_direction_horizon.md:5-10; CSV preserves full precision. -->
- **Evidence:** E2 CSV/Markdown; results README; main checklist HORIZON-SIGN.
- **Caveats:** export lacks checkpoint and exact action/population bindings. Reproduce source PASS/FAIL; the source does not establish a lower-CI-bound rule. The headline shorthand “at least **120 s** fails” means the tested failing clocks, not every unobserved later horizon. <!-- Source: E:/research-dense/docs/paper_results/E2_direction_horizon.md:9-10. -->
- `[[PENDING: original direction registration, checkpoint/config alias, sign denominator/tie rule / S-C1GAP]]`.

## HORIZON-UNIVERSAL — A short direction pass certifies WM-S generally

- **Status: CONTRADICTED.** `r13e/r14b/r16` retain overall NOT-QUALIFIED verdicts in the fidelity suite. The newer corrected WM2 recorded-action panel also fails; it is a different panel from E2.
- **Evidence:** `E:/research-dense/docs/paper_results/E1_WM_fidelity.md`; `E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md`.
- Corrected matched J at **300 s**: MAE **1376.6 veh-s**, Pearson **−0.235**, sign agreement **0.556**; these are recorded original-command replay plus feedback entrants, not full-feedback controller evaluation. <!-- Source: E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md:7-17. -->

## ERR-REGION — Upstream/merge road dominates the corrected long-clock error account

- **Status: SUPPORTED (descriptive covariance decomposition).** On matched native-lane recorded interventions, upstream/merge road contributes **96.0%** of **300 s** J squared-error covariance; source **−7.1%**, downstream **+11.1%**. Upstream/merge MAE is **1497.6 veh-s**, bias **+1031.1 veh-s**. Region boundary is **s ≤ 75.44 m**. <!-- Source: E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md:31-40. -->
- **Definition:** share(component) = `100 E[e_component e_J] / E[e_J²]`, with common total-J median order-statistic weights preserving additivity. Negative shares express cancellation.
- **Forbidden replacement: CONTRADICTED.** “Merge queue explains almost all MAE” or “fixing the merge will remove that share of error.” Region includes upstream longitudinal traffic; neither queue-specific causal mediation nor recoverable-error fraction is measured.
- **Protocol:** WM2 `b8797242`, mix ON, paired K8 median; original longitudinal commands replayed, entrants same-arm feedback; MAPPO remains an unmatched diagnostic. <!-- Source: E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md:4-8. -->

## ERR-ENTRY — Correct entrance bookkeeping sharply improves source error, not total accuracy

- **Status: SUPPORTED (corrected-clock entry-path diagnostic).** At **300 s**, source-component MAE falls **584.0 → 82.5 veh-s** (owner shorthand **584 → 83 veh-s**). Upstream MAE remains **1456.9 → 1393.0 veh-s**. <!-- Source: E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_RETARGET.md:25-29. -->
- **Evidence:** same report, zero-head comparison; WM2 clock-corrected versus `conserve_queue` + learned admission with zero response head. The source-clock correction and entry-path change must be named.
- **Caveat:** total J MAE at that clock **worsens 1449.1 → 1536.2 veh-s**; pooled J **512.7 → 546.1**. Source error had canceled road error. Do not combine these with the one-tick WMREQUAL3 values as though the clocks agree. <!-- Source: E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_RETARGET.md:13-14,30-48. -->
- **Status for “entrance repair solves long-horizon fidelity”: CONTRADICTED.**

## WM-REPAIR — A successful merge-response repair now qualifies the model

- **Status: CONTRADICTED for completed reviewed candidates.** The pilot failed its registered cross-horizon gate; the smoothed-objective feasibility also failed and its successor pilot was not launched.
- **Evidence:** `E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_PILOT_FAIL.md`; `WMFIX_V2_FEAS.md`; historical failures in `WMFIX_PRIOR.md`.
- **Refresh:** `E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_PILOT_V2DATA.md` now records completed training updates and an assembled zero-head baseline, with candidate panels/gate still pending. An entrant-controller binding failure was repaired without repeating saved updates. MAIN restricts it to its registered gate; no passing repair or extension is established. Training completion and baseline metrics are not efficacy.
- **Scope:** retain objective/horizon and hard-event gradient mismatches as failure evidence. Finite gradients, local loss improvement, or passing unit fixtures are not prediction-accuracy qualification. Later candidate outcomes remain `[[PENDING: any completed replacement verdict / S-WMFIXMERGE]]`.

## WM-CALIB — Completed calibration restores trustworthy policy credit

- **Status: CONTRADICTED for the frozen calibration pipeline; completed verdict NOT USABLE as long-horizon policy credit.** Final `E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md` supersedes handoff/smoke pending fields. The gate requires better MAE, sign and Spearman than both raw WM2 and arm-mean at every horizon in both splits.
- **Grouped-CV baseline warning:** on **40 fit worlds / 120 contrasts per horizon**, raw WM2 paired-effect MAE at **120/300/600 s** is **263/1719/6013 veh-s**, versus no-WM arm-mean **185/999/3298 veh-s**. Selected calibrators achieve **174/865/3152 veh-s**, but do not pass the combined gate. <!-- Source: E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md:2-14,17-28. -->
- **Transfer retained:** SPEEDHEAD selected-calibrator MAE **212/1783/4754 veh-s**, versus raw **176/1399/4701**, arm-mean **198/1568/4073**, at the same clocks; all transfer horizon gates FAIL. Raw WM is not worse than arm-mean in every transfer cell: avoid promoting the CV ordering into a universal claim. <!-- Source: E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md:29-50. -->
- **Scope:** arm−CTRL0 effects, mean paired futures (median sensitivity separately), objective J rather than silently substituted continuous team delay; world-grouped CV is selection evidence, not an independent test. WMREQUAL3 transfer changes reference/arms and is diagnostic only. No restoration of unseen-controller qualification or calibrated epistemic uncertainty follows.

## WM-QUEUE — A conservative queue/service prototype repairs the long tail

- **Status: CONTRADICTED for this registered prototype; KILL.** At **300 s**, queue paired-effect MAE/sign are **1362 veh-s / 0.313**, versus state-only **932 / 0.625**, arm-mean **930 / 0.646**, on **eight test worlds**. Queue−state-only sign difference is **−0.313 [−0.563,−0.104]** and baseline−queue MAE **−429.5 [−645.2,−213.3] veh-s**. Exact conservation does not ensure correct controller effects. <!-- Source: E:/agent-work/c4-wmqueue/docs/paper_results/WMQUEUE.md:13-16,24-28. -->
- The matched cached-WM comparison uses a smaller complete subset; do not attach the full-panel baseline cells to that subset. This falsifies this frozen prototype, not all queue models. Evidence: WMQUEUE.md, matched coverage and final scope.

## C1-FLOOR — Unavoidable chaos explains the short-window WM error

- **Status: CONTRADICTED as an explanation supplied by the measured perturbation test.** On **six native development worlds**, two one-tick **+0.01 m/s** HDV pulses per world yield world-median original-HDV position/speed errors **exactly zero through 120 s**, excluding both target IDs. Aggregate team-delay absolute-error median is **0.5193 veh-s at 120 s** and **55.61 [0.0000382,421.18] veh-s at 300 s**. <!-- Source: E:/agent-work/c4-c1floor/docs/paper_results/C1FLOOR.md:7-20. -->
- **Scope:** amplitude-, target-, arm- and median-dependent sensitivity; not proof that traffic is never chaotic, and not an irreducible stochastic lower bound. No joint original survivors remain at the last endpoint clock, so those endpoint cells are unavailable, never zero. RNG-only floor remains `[[PENDING: valid restore/reseed reference / S-C1FLOOR + MAIN]]`; the accepted fallback is complete, not a running floor job. Saved-state mismatch is a software/execution issue, not measured RNG divergence.

## C1-SWITCH — State-dependent choice has headroom, while WM-guided switching benefit is complete

- **Status: SUPPORTED for opportunity; PENDING for predictor/selector benefit.** Recorded **90-world, 600 s** unique winners: native **22**, CTRL0 **17**, MAPPO **19**, PI **29**, CIRCLES **3**. These fixed-episode winners establish scenario diversity, not same-moment switching. <!-- Source: E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_WINNERS.md:5-13. -->
- Full-rerun Step0b passes its headroom gate on **16 native-trunk states / eight worlds**. Unique **60 s** wins: native **2**, CTRL0 **9**, MAPPO **3**, plus **two ties**; unique **90 s** wins: native **2**, CTRL0 **7**, MAPPO **1**, PI **3**, CIRCLES **1**, plus **two ties**. Median best–second margin **1.832371 / 6.853775 veh-s** at those clocks, including ties. Every controller wins somewhere at the later clock, not the earlier one. <!-- Source: E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_WINNERS.md:31-36. -->
- **Validation/continuation:** the first paired canary is mechanical evidence only. Detector IDs affected its stochastic key despite anonymous tensors; correction changed a WM pick and triggered the retention STOP. MAIN subsequently authorized a corrected canary and continuation, still gated by parity and BB reuse; no complete main efficacy result is delivered at this cut. Evidence: `E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_DESIGN.md`, final continuation clauses; shared REGISTRATION.md, MAIN evening authorization. `[[PENDING: corrected-adapter complete switching regret/episode comparison / S-C1SWITCH]]`.
- **Contract:** full observable history; WM2 `b8797242` versus frozen BB; predictions optimize current-belief on-road delay with entries OFF/dormant mix, evaluation uses full fleet+source delay. Controller memory and approximate native plant remain explicit; deterministic repeat noise is not an RNG floor.

## COMPANION-STATUS — A validated value tail or successful SUMO backup licenses WM-based control

- **Status: PENDING for WMVALUE, CONTRADICTED for the completed backup win.** WMVALUE inference is running with frozen real-value/no-WM comparators and no efficacy verdict. `[[PENDING: complete frozen A–E verdict / S-WMVALUE]]`. Evidence: `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md`.
- Companion SUMO backup stopped: development learned−MAPPO **J600 +508.3 / +857.9 / +548.5 veh-s** at **u3/u7/u12**, with no useful deployed cooperation. This is context, not a Paper B result or a positive teacher. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:3-37. -->
- Saved-state restore and held-absolute-latent/decoder artifacts were identified; full reruns plus the common current-mean/decoder path pass the mechanical gates. Old artifact-contaminated credits remain invalid. Evidence: `E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md`, failure, full-rerun and CPU-correction sections. Mechanical repair establishes neither cooperation nor accuracy.

## CTRL-USE — Short-clock selection evidence licenses long-run controller choice

- **Status: CONTRADICTED as a general claim.** SPEEDHEAD reports inconclusive real speed-only headroom and discordant per-world predictions; its exploratory panel is separate from E1 and E2.
- **Evidence:** `E:/agent-work/c4-speedhead/docs/paper_results/SPEEDHEAD.md`, design, per-world contrasts and final verdict.
- **Permitted implication (interpretation):** use ranking/regret, signed effects, magnitude, action fidelity and complete-population accounting together. Treat short-clock results as a declared empirical envelope, not a safety certificate or authority for long-run policy credit. This implication is a recommendation, not a tested abstention policy.

## BIB / FORMAT

- **Bibliography:** reference existence/metadata and comparison scope are recorded in `RELATED_WORK.md`; BibTeX contains only verified entries.
- **Formatting: PENDING (final typesetting).** Use `IEEEtran` conference layout and IEEE numbered references. The page budget and deadline are owner constraints: **six pages including references**, deadline **2026-11-15**. <!-- Source: owner S-PAPERB brief; E:/research-dense/docs/paper/IEEE_IV_2027_DRAFT.md:3-4. -->
- `[[PENDING: owner-authored prose, authors, conference template/CFP check and compiled page count / owner]]`.
