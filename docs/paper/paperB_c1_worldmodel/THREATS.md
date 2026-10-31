# Paper B — limitations and reviewer response notes

Each response is an argument the current evidence permits, not polished rebuttal prose. Claim IDs refer to `CLAIMS.md`.

## “The headline says closed loop; the strongest wall result replays the controller.”

- **Threat:** realized CAV motion supplies information unavailable to an autonomous feedback forecast. BB-id does not even share that teacher contract.
- **Answer:** separate historical response prediction from feedback transfer and offline selection. Historical superiority is confined to its declared contract; full-feedback superiority remains `[[PENDING: eligible comparison / S-C1CLOSED]]`. No relabeling of teacher-forced CAVs as executable wall control. See C1-WALL/C1-CLOSED.

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
- **Answer:** keep failed pilot/feasibility outcomes and version history. Local loss, finite gradients, reduced lane counts or smaller source error do not imply correct paired J. Evidence WM-REPAIR and WMFIX_PRIOR. No new experiment is proposed as a completed answer.

## “Can calibration rescue the verdict?”

- **Threat:** flexible calibrators can use arm identity or same-panel outcomes, fit aggregate means and still fail state-dependent ordering.
- **Answer:** the final WMCALIB report is complete and NOT USABLE for long-horizon policy credit. Grouped-CV gains of a selected calibrator fail the independent transfer gate. Retain no-WM arm-mean as a substantive comparator and show both splits: raw WM's CV deficit is not its ordering in every transfer cell. Distinguish arm-minus-CTRL0 from arm-minus-native, mean from median reduction and J from continuous team cost. WMREQUAL3 changed-arm transfer is diagnostic. WM-CALIB.

## “Is long-horizon prediction impossible because SUMO is chaotic?”

- **Threat:** irreducibility could be used as an unfounded excuse for model error; median endpoint error could hide localized propagation.
- **Answer:** C1FLOOR's completed tiny-pulse native experiment has near-zero short-window endpoint medians, so it provides no measured floor explaining large WM errors there. Its targets, amplitude, native-only scope, target-ID exclusions and nonzero aggregate delay sensitivity are explicit. RNG-only floor was blocked by restore mismatch. Late empty endpoint cohorts stay unavailable. This does not prove universal determinism, nor license subtracting a floor from WM MAE. C1-FLOOR.

## “Does each controller winning somewhere establish that switching works?”

- **Threat:** fixed-world winners are not same-moment opportunities, and an oracle opportunity is not predictable advantage. Small best-to-second margins may be practically negligible.
- **Answer:** present recorded scenario diversity and full-rerun same-state headroom separately. Preserve ties and small margins; native repeats measure replay error, not stochastic uncertainty. Main WM/BB switching regret and episode gains remain `[[PENDING: complete corrected-adapter results / S-C1SWITCH]]`. Selection predicts on-road delay while evaluation includes source cost; entries are OFF and mix dormant. No inferred switching efficacy from winners. C1-SWITCH.

## “Do simulation restore and random-key artifacts contaminate the conclusions?”

- **Threat:** saved-state continuation can diverge with identical commands; a held absolute latent changes the intervention; anonymous tensors can still leak detector IDs through the forecast seed.
- **Answer:** ARTIFACT_CHECK invalidates old contaminated credits and validates full same-seed reruns plus current-mean/common-decoder execution. C1SWITCH same-state headroom uses full reruns and complete prefix/repeat gates. The later detector-ID stochastic-key correction changed a canary decision and triggered STOP; MAIN authorized a corrected continuation, not an efficacy finding. Preserve superseded schedules and require corrected-adapter gates before any final selector claim. Mechanics do not supply a useful cooperation or prediction result. COMPANION-STATUS/C1-SWITCH.

## “Why not replace the microscopic long tail with a queue model or value function?”

- **Threat:** a structural replacement may conserve traffic yet miss intervention-dependent service; a value head may merely reproduce arm means.
- **Answer:** WMQUEUE's fixed conservative prototype failed its registered paired-effect gate and was killed. This rejects that prototype, not every queue tail. WMVALUE compares short WM plus frozen real-value prediction with endpoint WM and no-WM alternatives; inference is running, so its benefit remains `[[PENDING: complete A–E gate / S-WMVALUE]]`. The stopped SUMO backup supplies no positive teacher or useful deployed cooperation. Keep these as context for controller-judge limits. WM-QUEUE/COMPANION-STATUS.

## “Where is uncertainty calibration or an abstention controller?”

- **Threat:** sampled future spread measures neither all model error nor calibrated epistemic uncertainty. Diagnostic confident calls can be wrong.
- **Answer:** no claim of calibrated uncertainty, formal admissibility level, or tested refusal policy. Horizon/plant/population scope is an empirical reporting discipline. A deployment gate would require independently measured coverage, calibration, cost and efficacy; none is supplied by a direction curve.

## “How independent are the data and intervals?”

- **Threat:** overlapping wall cohorts, repeated starts, vehicle/tick pseudoreplication, multiple models and post-selection invalidate naive confidence claims.
- **Answer:** preserve source sampling units, WORLD clustering where supplied, source-specific CIs and exploratory labels. Do not sum overlapping cohorts, synthesize missing intervals or pool teacher-forced, timeline and feedback panels. Source start-cluster delay diagnostics remain labeled as such. Exposure prevents calling development results fresh confirmation.

## “Does this prove safety, real-world benefit or sample efficiency?”

- **Threat:** incident zeros, isolated checkpoints, incomplete baseline lineage and short-clock selection cannot establish these broad outcomes.
- **Answer:** limit claims to simulator predictions and measured offline decisions. Safety fidelity failure stays visible; no universal incident bound, training-seed robustness, end-to-end efficiency ratio, or real-road claim. SPEEDHEAD's inconclusive headroom is context, not a favorable control result. Policy efficacy belongs to the companion paper's own evidence.

## Owner submission dependencies

- Essential for the intended closed-loop headline: C1 protocol/exposure and feedback result bindings.
- Essential for exact wording of trust limits: original direction-panel binding and distinction between covariance share, queue causality and source bookkeeping.
- Completed negative evidence to retain: calibration transfer failure, killed queue prototype and bounded perturbation reference. Pending updates: complete C1CLOSED, corrected-adapter switching, candidate repair gate and WMVALUE verdict; no partial efficacy promotion.
- Formatting: `[[PENDING: final prose, authors, conference requirements, accessible final figure sizing and compiled pagination / owner]]`.
