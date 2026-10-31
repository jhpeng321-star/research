# Paper B claim–evidence ledger

Extends `E:/research-dense/docs/paper/CLAIM_EVIDENCE_CHECKLIST.md`; that checklist is an older evidence cut, not overriding authority. Status refers to the exact claim below. Keep source comments when transferring numbers into prose. `[[PENDING: ... / agent]]` is an insertion request, never a result.

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
- **Correction to older checklist:** the original report was previously unavailable; recovered report now supplies paired intervals. Full recovery/exclusion audit remains `[[PENDING: C1_GAPS.md and authoritative eligibility decision / S-C1GAP]]`.

## C1-PI — Existing PI summaries establish current closed-loop PI transfer

- **Status: PENDING.** The recovered wall report's comparison-to-PI paragraph and change log support a historical qualitative lead only. Do not import its approximate PI numbers as a current feedback result.
- **Evidence:** recovered wall report, Comparison to PI; `E:/research-dense/docs/WM_CHANGE_LOG.md`; main checklist C1-PI.
- `[[PENDING: original PI exclusion receipt, checkpoint binding, feedback PI/Wall results and comparable black-box contracts / S-C1GAP + S-C1CLOSED]]`.

## C1-CLOSED — Zero-shot full-feedback superiority to black-box prediction is complete

- **Status: PENDING.** The requested `E:/agent-work/c4-c1closed/C1CLOSED.md` was absent at initial review. Historical wall motion replay cannot supply this claim.
- **Required evidence:** controller exclusion before training/adaptation; identical starts/histories; actual controller feedback on imagined observations; native/controlled original and entrant lane ownership; command-to-plant parity; endpoint and effect errors; paired model contrasts with world uncertainty; baseline data/compute contracts.
- **Disposition:** retain the headline as the paper question; assert a completed full-feedback comparison only after `[[PENDING: eligible result and hashes / S-C1CLOSED]]`.

## C1-MAPPO — Strict-history offline controller choice has measured short-clock regret

- **Status: SUPPORTED (exploratory, exported feedback-selection panel).** On **32 starts / 21 worlds**, **60 s** `team` oracle regret is **13.64 [5.382,26.2] veh-s** for `wm13e`, versus **22.05 [10.05,37.86]** for BB-traj. Always-native and always-MAPPO regrets are **48.3 [26.93,70.98]** and **21.88 [9.728,37.69] veh-s**. <!-- Source: E:/research-dense/docs/paper_results/E1_C1_fair.md:159-199, selectors decisions/B_history_strict/{wm13e,bbtraj}/60/team. -->
- WM-selected minus always-MAPPO is **−8.241 [−18.99,−0.08363] veh-s** and minus always-native **−34.66 [−60.51,−8.163] veh-s**; source **95%** intervals. <!-- Source: E:/research-dense/docs/paper_results/E1_C1_fair.md:195-199; interval level: E:/research-dense/docs/paper/IEEE_IV_2027_DRAFT.md:107,133. -->
- **Evidence:** corresponding CSV selectors; main checklist C1-MAPPO and C1-PAIRED; results README.
- **Scope:** offline selection among evaluated controllers, not deployed receding-horizon selection or proof of a learned controller's efficacy. `team` cost needs its original implementation binding; do not silently rename it all-request fleet-plus-source J.
- **Not established:** paired WM-minus-BB regret significance. Marginal intervals cannot provide it. `[[PENDING: paired regret contrast, exact team population/cost, candidate controllers and hashes / S-C1GAP + S-C1CLOSED]]`.

## C1-EXPOSURE — Controller exclusion is the unit of zero-shot eligibility

- **Status: SUPPORTED (scope constraint); per-result audit PENDING.** Same-world other-controller training is not unseen-world transfer. A checkpoint fitted to an evaluated controller's outcomes is adapted, even if initialized from an eligible earlier checkpoint.
- **Evidence:** main checklist C1-EXPOSURE; `E:/research-dense/docs/WM_CHANGE_LOG.md`, eval-world list correction; recovered wall report training/caveats.
- `[[PENDING: checkpoint→corpus→controller-first-exposure→evaluation mapping / S-C1GAP]]`. Do not use a model alias/date alone as proof.

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
- **Scope:** retain objective/horizon and hard-event gradient mismatches as failure evidence. Finite gradients, local loss improvement, or passing unit fixtures are not prediction-accuracy qualification. Later candidate outcomes remain `[[PENDING: any completed replacement verdict / S-WMFIXMERGE]]`.

## WM-CALIB — Calibration restores trustworthy policy credit

- **Status: PENDING.** Only handoff/resume reports were available initially; old calibration outputs are smoke artifacts and cannot be quoted as completed results.
- **Evidence:** `E:/agent-work/c4-wmcalib/docs/paper_results/{WMCALIB_HANDOFF,WMCALIB_RESUME}.md`.
- `[[PENDING: WMCALIB.md, complete grouped-CV selection and independent SPEEDHEAD transfer versus raw WM and arm-mean, plus separate WMREQUAL3 diagnostic / S-WMCALIB]]`.
- Calibration is controller/reference/population dependent. Same-panel fitting cannot establish zero-shot controller generalization or calibrated epistemic uncertainty.

## CTRL-USE — Short-clock selection evidence licenses long-run controller choice

- **Status: CONTRADICTED as a general claim.** SPEEDHEAD reports inconclusive real speed-only headroom and discordant per-world predictions; its exploratory panel is separate from E1 and E2.
- **Evidence:** `E:/agent-work/c4-speedhead/docs/paper_results/SPEEDHEAD.md`, design, per-world contrasts and final verdict.
- **Permitted implication (interpretation):** use ranking/regret, signed effects, magnitude, action fidelity and complete-population accounting together. Treat short-clock results as a declared empirical envelope, not a safety certificate or authority for long-run policy credit. This implication is a recommendation, not a tested abstention policy.

## BIB / FORMAT

- **Bibliography:** reference existence/metadata and comparison scope are recorded in `RELATED_WORK.md`; BibTeX contains only verified entries.
- **Formatting: PENDING (final typesetting).** Use `IEEEtran` conference layout and IEEE numbered references. The page budget and deadline are owner constraints: **six pages including references**, deadline **2026-11-15**. <!-- Source: owner S-PAPERB brief; E:/research-dense/docs/paper/IEEE_IV_2027_DRAFT.md:3-4. -->
- `[[PENDING: owner-authored prose, authors, conference template/CFP check and compiled page count / owner]]`.
