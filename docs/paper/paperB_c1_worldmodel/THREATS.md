# Paper B — limitations and reviewer response notes

Each response is an argument the current evidence permits, not polished rebuttal prose. Claim IDs refer to `CLAIMS.md`.

## “The headline says closed loop; the strongest wall result replays the controller.”

- **Threat:** realized CAV motion supplies information unavailable to an autonomous feedback forecast. BB-id does not even share that teacher contract.
- **Answer:** separate historical response prediction from feedback transfer and offline selection. Historical superiority is confined to its declared contract; full-feedback superiority remains `[[PENDING: eligible comparison / S-C1CLOSED]]`. No relabeling of teacher-forced CAVs as executable wall control. See C1-WALL/C1-CLOSED.

## “Is the excluded controller actually absent from training?”

- **Threat:** reused outcomes, adaptation and same-world other-arm exposure can undermine zero-shot wording.
- **Answer:** define eligibility by checkpoint and evaluated controller before first exposure. The recovered wall report names the historical training arms; same-world training prevents unseen-world claims. `[[PENDING: hash-linked corpus/exposure audit / S-C1GAP]]`. Later adaptation is a separate result, not retroactive confirmation.

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
- **Answer:** report the source-specific oracle regret and constant selectors, and retain the unresolved paired WM–BB contrast. Use exact strict-history/team selectors only. Evidence C1-MAPPO. The result supports exploratory offline choice, not deployed selector efficacy or a numerical guarantee on future regret.

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
- **Answer:** require complete world-grouped CV, frozen selection, independent transfer versus raw WM and arm-mean; distinguish arm-minus-CTRL0 from arm-minus-native and J from continuous team cost. Old smoke outputs are not results. `[[PENDING: final WMCALIB report and transfer / S-WMCALIB]]`.

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
- Optional result update: final calibration, provided it preserves the failed/unmatched evidence and transfer scope.
- Formatting: `[[PENDING: final prose, authors, conference requirements, accessible final figure sizing and compiled pagination / owner]]`.
