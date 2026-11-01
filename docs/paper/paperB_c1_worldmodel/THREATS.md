# Paper B — limitations and reviewer response notes

Each response is an argument the current evidence permits, not polished rebuttal prose. Claim IDs refer to `CLAIMS.md`.

## “The headline says closed loop; the strongest wall result replays the controller.”

- **Threat:** realized CAV motion supplies information unavailable to an autonomous feedback forecast. BB-id does not even share that teacher contract.
- **Answer:** separate historical response prediction from feedback transfer and offline selection. Completed C1CLOSED now supplies autonomous common-plant PI/MAPPO evidence, but its principal-clock paired endpoint contrasts are inconclusive. Earliest clocks favor BB; lower aggregate error does not imply useful controller ranking. Wall executor restoration remains pending. No relabeling of teacher-forced CAVs as executable wall control. See C1-WALL/C1-CLOSED.

## “Is the excluded controller actually absent from training?”

- **Threat:** reused outcomes, adaptation and same-world other-arm exposure can undermine zero-shot wording.
- **Answer:** define eligibility by checkpoint and evaluated controller before first exposure. The recovered wall report names the historical training arms; historical same-world training prevents unseen-world claims. The newly delivered C1_GAPS audit supports algorithm exclusion in the audited historical/later corpora, and distinguishes own-policy adaptation from MAPPO exposure. `[[PENDING: unbroken initialization lineage and evaluation binding / S-C1GAP + S-C1CLOSED]]`. Later adaptation is a separate result, not retroactive confirmation.

## “Did a newer architecture fail, or did the comparison become harder?”

- **Threat:** checkpoint comparisons also change world exposure, corpora, history and physics. Historical matched-data advantages cannot isolate the effect of newer components.
- **Answer:** C1PIDIAG documents old C1/frozen-BB exposure to all its PI/wall worlds under other arms and none in the newer audited corpora. Treat this as a major fairness confound, not a causal coverage-only proof. Keep genuine original-start regression, unsuccessful component rollbacks and incomplete initialization ancestry. C1-COVERAGE/C1-EXPOSURE.

## “Was the history rescue obtained by giving the forecast its future?”

- **Threat:** shifting the forecast start changes state difficulty and survival as well as available history. A later controller restart may omit its command memory.
- **Answer:** original-start forecasts never receive later frames. Genuine-history diagnosis uses separately registered later starts; within those identical starts the history contrast is informative, while between-panel gains are not pure history effects. Teacher forcing still supplies future CAV motion. Late PI feedback with unbound previous-command state is retained as a reset stress test, not deployed continuation or isolated human-dynamics error. Entries/source remain unavailable on these bindings. C1-PI; C1PIDIAG.md.

## “IDM traffic predicted by an IDM prior is an inverse crime.”

- **Threat:** synthetic dynamics share the prior's structure; no real-road transfer is demonstrated.
- **Answer:** explicitly frame simulator-based controller-response transfer and fidelity diagnosis. MOBIL/learned merge behavior and SUMO lanes differ, but those differences do not establish realism. Discuss physics-prior suitability, not real-driver identification or physical safety certification. Evidence WM-SETTINGS/WM-LANES, recovered wall caveats, change log.

## “Are BB-traj and BB-id state-of-the-art predictors?”

- **Threat:** frozen in-house baselines, data differences and different CAV contracts limit the comparison.
- **Answer:** name them as in-house model ablations. Local `src/wms/model/factory.py` identifies physics-free trajectory and controller-id learned-CAV variants. Published forecasting references are context, not executed baselines. Do not equate our BB-traj to Wayformer/Trajectron++, or claim superiority to all black-box prediction. `[[PENDING: final training/input/parameter/budget comparability / S-C1GAP]]`.

## “The architecture says learned settings, but historical settings were unused.”

- **Threat:** describing later hybrid wiring as the cause of historical success is false attribution.
- **Answer:** show version labels in the schematic and result captions. Historical residual-only and later hybrid settings are explicitly distinguished. A useful latent setting is not recovered true behavior/type. Evidence WM-SETTINGS and change-log diagnostics. Component attribution awaits a matched ablation.

## “Is the short-clock regret difference statistically established?”

- **Threat:** comparing displayed marginal CIs is not paired inference; changing cost populations or candidate sets changes regret.
- **Answer:** report the source-specific oracle regret and constant selectors, plus the separately exported paired WM–BB contrast now available in the current E1 export and independently reconstructed by C1_GAPS with world resampling. Use exact strict-history/team selectors only; keep plant/exposure/cost binding pending. The same-panel endpoint contrast is inconclusive and longer-clock regret remains inconclusive. Evidence C1-MAPPO/C1-LONG. The result supports exploratory offline choice, not deployed selector efficacy or a numerical guarantee on future regret.

## “Does the horizon plot contradict corrected qualification?”

- **Threat:** different checkpoints, controller plants, references, populations and sign denominators can produce different apparent envelopes.
- **Answer:** show panel identities separately. E2's frozen registered direction panel is not the corrected WM2 recorded-command J panel or an all-criteria fidelity pass. Transfer of its envelope is `[[PENDING: original export binding / S-C1GAP]]`. In particular, retain the failed fidelity verdicts instead of treating a short-clock PASS as model-wide authority.

## “The regional percentage is causal storytelling.”

- **Threat:** error covariance is not explained MAE, queue mediation, or the improvement achievable by repairing a component.
- **Answer:** state the source formula, the upstream/merge boundary, common order-statistic weights and negative cancellation explicitly. Keep signed bars; do not use a pie chart or label the quantity 'queue error explained'. Evidence ERR-REGION. Localizing residual error justifies a research target, not its causal necessity.

## “The entrance fix improves the desired metric, so why does the total worsen?”

- **Threat:** additive components can cancel and hide the wrong intervention response; clock conventions also differ.
- **Answer:** report source, upstream and total side by side under the same corrected clock. Improvement of source accounting is genuine contract evidence; it is not overall WM qualification. Do not splice corrected-clock diagnostics into one-tick qualification. Evidence ERR-ENTRY, WMFIX_RETARGET.

## “Is this really native-lane control for every vehicle?”

- **Threat:** original versus entrant callbacks, ramp stays and native references can have different ownership. A lane-head MAPPO arm with missing lane requests is not matched native-lane evidence.
- **Answer:** use the corrected native-lane qualification only within its reported matched arms; label MAPPO unmatched diagnostic. Original commands are replayed and entrant feedback is explicit. Common controller code alone does not establish requested/issued/realized plant parity. Evidence WMREQUAL2/WMREQUAL3 and C1-FACTOR.

## “Fewer lane changes or better local losses prove a fix.”

- **Threat:** bundled changes, saturated/indirect gradients, hard events and objective/horizon mismatch undermine mechanistic attribution.
- **Answer:** keep failed pilot/feasibility outcomes and version history. Expanded-data repair stopped at its failed early screen; later checkpoints were not scored, so the original full gate is unmeasured. Local loss, finite gradients, reduced lane counts or smaller source error do not imply correct paired J. Evidence WM-REPAIR and WMFIX_PRIOR.

## “Can calibration rescue the verdict?”

- **Threat:** flexible calibrators can use arm identity or same-panel outcomes, fit aggregate means and still fail state-dependent ordering.
- **Answer:** the final WMCALIB report is complete and NOT USABLE for long-horizon policy credit. Grouped-CV gains of a selected calibrator fail the independent transfer gate. Retain no-WM arm-mean as a substantive comparator and show both splits: raw WM's CV deficit is not its ordering in every transfer cell. Distinguish arm-minus-CTRL0 from arm-minus-native, mean from median reduction and J from continuous team cost. WMREQUAL3 changed-arm transfer is diagnostic. WM-CALIB.

## “Is long-horizon prediction impossible because SUMO is chaotic?”

- **Threat:** irreducibility could be used as an unfounded excuse for model error; median endpoint error could hide localized propagation.
- **Answer:** C1FLOOR's tiny-pulse experiment provides no measured floor explaining large WM endpoint errors there. Its native-only scope and nonzero aggregate sensitivity are explicit. ENSPROBE later found identical tapes across its configured continuation seeds, holding driver population/arrivals fixed; that does not measure hidden-state uncertainty, prove real-road determinism, or repair the earlier un-restored-continuation mismatch. Late empty cohorts stay unavailable. Never subtract these references from WM MAE. C1-FLOOR/WM-RNG.

## “Does each controller winning somewhere establish that switching works?”

- **Threat:** fixed-world winners are not same-moment opportunities, and an oracle opportunity is not predictable advantage. Small best-to-second margins may be practically negligible.
- **Answer:** present scenario diversity and full-rerun same-state headroom separately. Preserve ties/small margins; native repeats measure replay error, not stochastic uncertainty. The reference switching study stopped incomplete; its authorized low-penetration complete-case analysis does not establish WM−BB superiority or a full-panel null. The wall-inclusive main study and final RL comparators remain pending. Selection predicts road delay while evaluation includes source cost, with entries OFF/dormant mix. C1-SWITCH/C1-SWITCH2.

## “Do simulation restore and random-key artifacts contaminate the conclusions?”

- **Threat:** saved-state continuation can diverge with identical commands; a held absolute latent changes the intervention; anonymous tensors can still leak detector IDs through the forecast seed.
- **Answer:** ARTIFACT_CHECK invalidates old contaminated credits and validates full reruns plus common-decoder execution. C1SWITCH headroom uses full reruns and prefix/repeat gates. The detector-ID key correction changed a canary pick; corrected-canary/parity/BB-reuse gates subsequently passed before the later budget stop. Preserve superseded schedules and distinguish mechanical validation from efficacy. C1CLOSED's failed exact cross-host parity and prospective numerical acceptance are separately disclosed; ensemble-scale acceptance does not prove identical lane/queue trajectories. COMPANION-STATUS/C1-SWITCH/C1-CLOSED.

## “Why not replace the microscopic long tail with a queue model or value function?”

- **Threat:** a structural replacement may conserve traffic yet miss intervention-dependent service; a value head may merely reproduce arm means.
- **Answer:** WMQUEUE's fixed prototype failed and was killed. WMVALUE now also FAILS: both short-imagination methods fail necessary no-WM comparator gates, so its long-WM conjunction fails regardless of the missing A estimate. Retain A as NA and the frozen value/data scope. These reject the tested routes, not all queue/value models. The stopped SUMOCF backup remains negative; later exact-planner findings are distinct companion evidence and do not validate a WM. WM-QUEUE/WM-VALUE/COMPANION-STATUS.

## “Point errors are irrelevant if the model gets the traffic distribution right.”

- **Threat:** exact realized outcomes may be unpredictable, but good pooled means or broad future spread can conceal biased action forecasts.
- **Answer:** WMBREAKDOWN supplies distributional failure too: conditioned climatology is stronger than the WM on the evaluation slice, absolute range coverage is poor, and paired effects can be excessively dispersed. Do not claim all uncertainty is too narrow. Saved cost-only predictions cannot score new-panel imagined event rates/timing; historical rate evidence belongs to a different panel. Cells are sparse and reused, preventing untouched or powered qualification. The proposed distributional gate is not a passed theorem. WM-DISTRIBUTION.

## “Does faithful physics prove a best-achievable information ceiling?”

- **Threat:** observable initialization can lose internal state; a physics prior can improve fields while misjudging controller effects.
- **Answer:** SUMOCORE retains initialization collision flags, different encoder/ensemble contracts and small cohorts. Its short-clock improvement does not establish a clean ceiling or long-clock action accuracy. Latest causal-motion FULL-v4 improves delay-distribution/endpoint point estimates but worsens breakdown-rate distance, failing the conjunctive adoption rule at this cut. Earlier saved-ID concerns belong to older variants; they neither validate nor describe this binding. Earlier-source A/B/C versus later FULL changes physics too, so it is not a settings-only ablation. Final identity-bound qualification remains pending. WM-PHYS.

## “Do the rebuilt model's confidence intervals establish superiority?”

- **Threat:** an interim scorer resampled unscored same-cell peers; after correction, mixed-cell bootstrap intervals still condition on singleton worlds. Quoting either as full-panel uncertainty overstates evidence.
- **Answer:** retain the explicit retraction. Report the latest candidate's favorable point estimates with full-world intervals unavailable under its registered resampling plan. The breakdown-distance point condition already fails, independently of missing uncertainty. Later engineering variants need their own frozen identity, matched complete cohorts and corrected inference; no inference of qualification from missing CIs. WM-PHYS; WMPHYS_MORNING.md, switching-eligibility section.

## “Is WM-guided switching fairly compared with RL?”

- **Threat:** cadence, wall lifecycle, memory resets, observation budget, source objective, training expenditure and joint action space can favor one selector.
- **Answer:** C1SWITCH2 registers shared library/cadence/handovers/evaluation panel; evaluation is peer-owned once, with frozen final RL weights. Fleet PPO is the like-for-like primary selector; per-CAV MAPPO has a different joint action space. Disclose road-only WM/BB prediction versus full-team evaluation, both model-data and RL training costs, and independent WM-PHYS adoption. Training progress and complete fixed-controller arms cannot substitute for the pending paired picker result. C1-SWITCH2.

## “Can the incomplete main switching panel be treated as a null result?”

- **Threat:** shared cpuQ connectivity loss leaves current training/WM accounting unknown; model-service timeout and GPU release leave selective completed episodes. A staged relocation is not completed inference.
- **Answer:** label the main paired campaign incomplete and infrastructure-interrupted. Keep completed fixed/random arms, distinguish historical lower accounting bounds from exact final totals, and retain missing final RL checkpoints. Do not analyze the reachable-host subset as the planned population or infer remote job death from transport loss. Source owners already have read-only observers and registered continuation/placement plans; this package supplies no replay or new dispatch. C1-SWITCH2; `E:/agent-work/c4-c1switch2/HANDOFF.md`, Current execution state; selector status reports.

## “Where is uncertainty calibration or an abstention controller?”

- **Threat:** sampled future spread measures neither all model error nor calibrated epistemic uncertainty. Diagnostic confident calls can be wrong.
- **Answer:** no claim of calibrated uncertainty, formal admissibility level, or tested refusal policy. Latest WM-PHYS selected-switch win rates are below their forecast probabilities despite favorable delay-distribution point estimates; a probability threshold is a rule, not demonstrated calibration. Horizon/plant/population scope is an empirical reporting discipline. A deployment gate would require independently measured coverage, calibration, cost and efficacy; none is supplied by a direction curve. WM-PHYS.

## “How independent are the data and intervals?”

- **Threat:** overlapping wall cohorts, repeated starts, vehicle/tick pseudoreplication, multiple models and post-selection invalidate naive confidence claims.
- **Answer:** preserve source sampling units, WORLD clustering where supplied, source-specific CIs and exploratory labels. Do not sum overlapping cohorts, synthesize missing intervals or pool teacher-forced, timeline and feedback panels. Source start-cluster delay diagnostics remain labeled as such. Exposure prevents calling development results fresh confirmation.

## “Does this prove safety, real-world benefit or sample efficiency?”

- **Threat:** incident zeros, isolated checkpoints, incomplete baseline lineage and short-clock selection cannot establish these broad outcomes.
- **Answer:** limit claims to simulator predictions and measured offline decisions. Safety fidelity failure stays visible; no universal incident bound, training-seed robustness, end-to-end efficiency ratio, or real-road claim. SPEEDHEAD's inconclusive headroom is context, not a favorable control result. Policy efficacy belongs to the companion paper's own evidence.

## Owner submission dependencies

- Completed current common-plant feedback is available; strict initialization lineage and causal wall feedback remain dependencies for stronger zero-shot/wall wording.
- Essential for exact wording of trust limits: original direction-panel binding and distinction between covariance share, queue causality and source bookkeeping.
- Completed evidence to retain: calibration/value failures, killed queue prototype, repair early-stop partial result, distributional bias, bounded pulse/RNG references and mixed closed-loop results. Reference switching stays partial. Latest rebuild interim fails the breakdown-distance adoption condition and retracts earlier CIs. Pending updates: connectivity/accounting reconciliation, complete main wall-inclusive paired pickers/final RL weights and final independent physics-rebuild qualification; no partial efficacy promotion.
- Formatting: `[[PENDING: final prose, authors, conference requirements, accessible final figure sizing and compiled pagination / owner]]`.
