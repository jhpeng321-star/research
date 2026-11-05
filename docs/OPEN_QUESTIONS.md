<!-- WM-S-open-questions-20261009 -->
<!-- S-DIAG-closeout -->
**2026-11-06, S-DIAG (partially answered):** [Clean1 diagnosis](../runs/active/C4-V7FT-20261017/CLEAN1_DIAGNOSIS_20261105.md) rules out none-collapse/dead residual and identifies weak, reversed instantaneous move actuation, concentrated supervision, feasibility limits and local-credit mismatch. Still open: causal move-versus-none physical benefit, exact identities of three usable CE contexts, and whether production-consistent actuation plus feasible diverse practice produces group-over-best-solo benefit. Final clean1 owner STOP retained; existing clean2/plan-fix/mining work must be consulted before any follow-up.
<!-- /S-DIAG-closeout -->

<!-- S-WMCHECK-closeout -->
**2026-11-05, S-WMCHECK (partially answered):** [Gain-disagreement audit](../runs/active/C4-V7FT-20261017/WM_ERROR_CHECK_20261105.md) finds 90/221 positive predictions win on recovered selected training pairs. How wrong is the WM at raw/calibrated P>=0.8 remains **open**: no genuine candidate-bound probability evidence exists in these records. Matched original prediction-horizon provenance is also missing. No confidence inferred from gains and no new runs authorized by this analysis.
<!-- /S-WMCHECK-closeout -->

# WM-S open questions update (2026-10-09)

Current direction: WM-S ([WM_STATE_PLAN.md](WM_STATE_PLAN.md),
[WM_S2_ROADMAP.md](WM_S2_ROADMAP.md)). Questions changed by the recent results:

- **OQ-WMS-1 (open):** does the braking fix (asymmetric too-fast loss, decel weight,
  leader-braking propagation, asymmetric residual cap) reduce the 60–120 s blow-up rate
  and the too-fast bias versus the same model without it? Models trained 2026-10-09;
  scoring in progress (`E:/rd-wt/ws-main2score`), not yet answered.
  [REGISTRATION.md](../runs/active/WM-S-MAIN2-20261009/REGISTRATION.md).
- **OQ-WMS-2 (open):** why did pack_v3 contribute 0 training windows, and does the fix
  restore them? Fix in progress (`E:/rd-wt/ws-packv3fix`).
- **OQ-WMS-3 (open):** does the entries-ON eval change the car/traffic accuracy picture?
  Rescoring in progress (`E:/rd-wt/ws-entrieseval`).
- **OQ-WMS-4 (partly closed):** does C1 generalise to an unseen controller beyond PI?
  **Yes for the WALL** — 2026-10-09, structured 8.71 m versus bb_traj 25.41 / bb_id
  40.07 m HDV median |ds| at 60 s on `wall_same`, verdict SUPPORTED with caveats
  (`E:/rd-wt/ws-wallc1/_out/WALL_C1_RESULT.md`). Closed-loop C1 remains open.
- **OQ-WMS-5 (open):** can the planner use traffic-level delay deltas now? At 300 s the
  model picks the better of native/PI in ~80% of cases (regret 19 vs 128/194 for fixed
  choices), but all ranking so far replays recorded CAV motion; closed-loop ranking is
  untested ([DESIGN_DISCUSSION_20261008.md](DESIGN_DISCUSSION_20261008.md)).
- **OQ-WMS-6 (open):** C2 per-driver identification and C3 fragility are deferred; C4
  learned cooperative multi-CAV control comes later. Closest prior work and required
  evidence: [LITCHECK_MULTICAV_WM_RL_20261009.md](LITCHECK_MULTICAV_WM_RL_20261009.md).
- **Closed / retired:** the WM-N1 delay-head direction (A8–A13) and wall-only planning
  are retired by owner decision ([WM_STATE_PLAN.md](WM_STATE_PLAN.md)).
<!-- /WM-S-open-questions-20261009 -->

Label fix + timing audit (2026-10-03): [FIX1](../runs/active/WM-N1-LABEL-TIME-FIX1-20261003/REPORT.md). The switching-run label lookup is fixed (196 chunk2/3 campaign switching items rebound to their own swnative/swpi labels; 0 excluded), and the corrected TRAIN labels pass every identity gate. The training mapper chose a later decision time than t0 in all 5,480 labelled cells. Owner approved (a) the t0 mapper, (b) excluding 263 base/native_base items from outcome supervision, and (c) the 10 s eval correction window. M1 retrain package: [RETRAIN_READY](../runs/active/WM-N1-LABEL-TIME-FIX1-20261003/RETRAIN_READY.md); GPU50 smoke is the only training-path test. M1 unchanged.

Full TRAIN timing audit (2026-10-03): [stopped at corrected policy gate](../runs/active/WM-N1-LABEL-TIME-FULL-20261003/REPORT.md). With source-cited policy correspondence, 900 labelled items map, but 196 chunk2/3 campaign switching items (13,524 rows) carry true-fleet labels (native_fleet/pi_fleet): suspected label-binding defect. Last-row-per-tau timing not measured; exclude/rebind decision and any further read need owner approval. M1 unchanged.

TRAIN identity diagnostic (2026-10-03): [exact row and policy-name resolution](../runs/active/WM-N1-LABEL-TIME-DIAG-20261003/REPORT.md). Global row69 uses label class damping_fleet versus receipt controller damping; frozen source explains this naming difference. The prior exact-string audit check was overstrict for this row. Full TRAIN timing remains unresolved; another read needs owner approval. M1 unchanged.

TRAIN metadata timing (2026-10-03): identify which label identity field disagrees with its registered TRAIN item, then resolve last-row-per-horizon timing. [Stopped audit](../runs/active/WM-N1-LABEL-TIME-20261003/STOP_REPORT.md); no model-accuracy or corruption claim. Follow-up metadata read awaits owner approval; no new simulation/training.

Evaluation bridge (2026-10-03): [source foundation review](../runs/active/WM-N1-EVAL-BRIDGE-20261003/SOURCE_REVIEW.md), 47 generated cases covered with failure/repair receipts retained. Real scoring remains disabled pending action/temporal binding, comparator and truth-roster contracts plus explicit data release. Source stays in its managed branch; M1 unchanged.

EFFH-01 update (2026-10-03): GLM reports substantial 300-to-600s changes in TRAIN physical contrasts. [Source review](../runs/active/EFFH-01-REVIEW-20261003/SOURCE_REVIEW.md) corrects the unobserved after-600/steady-state claim and the signed-ratio/first-crossing interpretation. Numerical/input provenance remains report-only. [Model source audit](../runs/active/EFFH-01-REVIEW-20261003/HORIZON_LIMITATION_SOURCE_AUDIT.md) identifies timestamp binding and 10s-train/120s-eval summary risks; actual M1 impact is unverified. Keep M1 locked; no longer-horizon run or retraining is authorized.

<!-- coordinator-readiness-20261002 -->
Member 1 completed all 30,000 CUDA training steps. Checkpoint b37a25a7 is independently hash-verified and preserved under runs/active/WM-N1-M1-20261003; all 30,000 training records pass the owner-approved loss checks. The completed run remains attributed to 45c9032 plus its documented box-side CUDA patch.

Source branch zcode/wd01 is now d96b57d: the four CUDA host copies and pandas 2.3.3 pin are upstreamed, and the J4 campaign/frozen wall-factor selector and G6 metadata report are corrected. The host-copy regression passed on CPU (CUDA unavailable), and all 28 guarded synthetic J4 checks passed.

TTC15/A7 remains unmerged because of integration defects. A real-checkpoint evaluation bridge is missing; its source-only foundation passed generated checks; real scoring remains incomplete and disabled. The frozen J4-only held-out protocol also conflicts with the completed full-TRAIN M1 and needs owner resolution before such a claim. CAL/J4B access, INC1, PL-WM1 and checkpoint cloud backup remain gated. No model accuracy or held-out outcome result is claimed.
<!-- /coordinator-readiness-20261002 -->

<!-- literature-question-20261002 -->
LIT-WMN1-NOVELTY-20261002-01: Can integrating current hidden-state uncertainty improve matched intervention ranking, calibration and physical outcomes over a point reconstruction with the same transition ensemble? Close primary prior art includes arXiv2608.17739 and PMA-Diffusion2512.06183. Current true-state WM-N1 training preparation does not answer this deployment question. Comparison ideas in the [review](../tmp/zcode_review/LIT_WMN1_NOVELTY_REVIEW_20261002.json) are proposals only; no new allocation is launched.
<!-- /literature-question-20261002 -->

<!-- WM-N1-RECOVERY-01-resume -->
**WM-N1-RECOVERY-01 resumed by owner (2026-10-02):** GLM repairs continue as code-only while an effective synthetic-test guard is built and independently reviewed. Prior split-metadata exposure remains disclosed; real40/GPU and main restructuring are still held.
<!-- /WM-N1-RECOVERY-01-resume -->

<!-- WM-N1-RECOVERY-01-stop -->
**WM-N1-RECOVERY-01 stopped (2026-10-02):** a completed executor test read the frozen CAL split world-name list despite the prohibition. Four workers stopped; candidates and logs preserved. Zero real training/SUMO; GPU readiness held pending owner direction and guarded review. [Stop receipt](../tmp/zcode_review/WMN1_CAL_SPLIT_EXPOSURE_20261002.md).
<!-- /WM-N1-RECOVERY-01-stop -->

<!-- WM24-31-complete -->
**WM-S-TRAINSETUP-20261009 (engineering):** slot-binding, sampling and CPU argv
contracts are verified. The real-pack retained-window census after PI/wall and
held-family exclusions, GPU smoke stability/timing, and forecasting generalization
remain open; synthetic checks supply no efficacy evidence.
[Readiness](../_out/TRAINSETUP_READY.md).
**WM-24 to WM-31 (2026-09-25):** open questions after the first fresh-world test: why queues found on development worlds are missed on fresh worlds; how to stop realistic sensing noise from producing false queues; how to get calibrated uncertainty (current calibration has a code defect); whether a better blind-cell speed prior can close the speed gap; and whether a stable per-vehicle dynamics model can roll the hybrid state forward (open-loop rollouts reach 42 m/s). Fresh random worlds show a queue at t=600 only about 8% of the time, so queue claims need larger or multi-time batches. [Report](../output/fresh_validation_v1/WM31_REPORT.md).
<!-- /WM24-31-complete -->
<!-- WM23-complete -->
**WM-23 complete (2026-09-25):** speed fill for occlusion-shadowed cells answers part of WM-22's occluded miss (2/5 queues at both radii, 0 false positives). Still open: hidden queue cells are undercounted (for example 5 vs 13 vehicles at 7011), and nothing yet tests the rule on worlds it was not designed on. [Report](../output/shadow_speed_v1/REPORT.md).
<!-- /WM23-complete -->
<!-- WM22-complete -->
**WM-22 complete (2026-09-24):** removing prior mass from visible-empty cells answers the WM-18 overcount and ideal-sensing queue misses: 5/5 true queues found, signed bias near zero, 0 false positives. Still open: occluded queue tails (1/5 at 50 m), because blind cells take the F prior's free-flow speed; and whether p_miss = 0 holds for real sensors. [Report](../output/prior_fusion_v1/REPORT.md).
<!-- /WM22-complete -->
<!-- WM20-complete -->
**WM-20 complete (2026-09-24):** on all 21 exposed worlds, a 100 m all-vehicle radius improves the frozen overlay (ideal occupied count MAE 0.43183 at 50 m to 0.26329 at 100 m; identity gate matches WM-18). Sensing-only reconstruction stays accurate on declared cells and abstains elsewhere (100 m abstention 0.872 at 2% to 0.100 at 40%). The 60 s memory plus nearest-cell fill does not: signed total bias on declared cells is about +140 to +230 vehicles and occupied speed RMSE stays several m/s. Conservation counting and a physics fill were declared out of this package and are untested. [Report](../output/radius_penetration_v1/REPORT.md).
<!-- /WM20-complete -->

<!-- WM19-reviewed -->
**WM-19 complete (2026-09-24):** 21 radius-input exports independently verified; 16 adapter tests pass, including actual synthetic mass and hidden-control separation. [Reviewed report](../output/worldmodel_radius_input_v1/REPORT_FINAL.md). No real forecasts/training or WM-16 resumption. Remaining priority: observed/prior speed fusion hides queues (WM-18 occluded detects 1/5). One saved-grid prior-speed test is proposed, not executed; action-response dynamics remain unvalidated. This supersedes earlier WM-19 in-progress statements below.
<!-- /WM19-reviewed -->

<!-- WM18-WM19-current -->
WM-18 answers the bounded richer-sensing question positively for local error and mean absolute total-count bias (21 exposed worlds), but not queue recovery: ideal0/5,occluded1/5 under the frozen definition. Root traced the remaining queue failure to small fast prior mass combined with correct slow observations. Next constructor question: can causally supported prior assimilation correct that without hiding uncertainty or degrading uncovered cells? No new variant tested. WM-19 input integration is underway; real future-grid evidence/controller-response validity remain open, and WM-16 stays paused. [Verified result](../output/radius_sensor_v1/REPORT_FINAL.md).

<!-- WM17-open-questions -->
WM-17 current-state construction closed its registered four-arm comparison. Open: can CAV-only evidence calibrate a 50 m count posterior and recover slow-queue presence without fixed detectors, true penetration, or hidden HDV state? The existing 25 exposed worlds cannot validate a new fit; a separate 12-prefix proposal and pre-fit stop condition are in [WM-17 close-out](../output/state_constructor_v1/WM17_CLOSEOUT_REPORT.md). Also unresolved: a provable speed/acceleration bound and out-of-grid mass for V7 reachable sets, actual nonzero receipt delays, and whether better state reconstruction improves any physical controller outcome. No successor run is authorized by this note.

<!-- WM15-spatial-correction -->
WM-15 open questions after its [first structural milestone](../output/worldmodel_spatial_v1/FIRST_MILESTONE_REPORT.md): Can sparse, receipt-causal CAV reports calibrate 10 m cell count and occupied speed; can an action-conditioned model predict future grid states beyond persistence; and do mixed spatial assignments improve the complete physical outcome vector over fixed/native/PI and direct prediction? The saved physical corpus has full vehicle positions only through t=600.0, so it cannot grade post-action future-grid forecasts. Fleet-uniform arms cannot answer the mixed-interaction question. No positive spatial-planner evidence yet; the WM-14 fit budget remains unspent.

<!-- WM14-spatial-supersession -->
WM-14 stopped as a uniform-arm baseline/component diagnostic; its fit allocation was not spent. [Close-out](../output/worldmodel_repair_wm14/SPATIAL_SUPERSESSION.md).

<!-- WM13-review -->
WM-13 independent review qualifies WM-12: useful prototype, but no validated closed-loop safety/selection. Reproduced dispatcher stable-request starvation; planner later-tick labels are static600s outcomes; reconstruction headline is oraclebest-head and sensorablation has unequal membercounts. Next: offline repairs and one same-state operational benchmark before expanded mixed/wall campaigns. No code patch/simulation/training. [Review](../output/worldmodel_review_wm13/REPORT.md).

<!-- WM12-status -->
WM-12 Phase A complete (2026-09-23, zero simulation, 8 workstreams via parallel subagents): every component of the outcome-scored architecture now has acceptance evidence. Per-vehicle car-following dynamics PASSED its gate (2.17 vs 2.33 advected persistence; bounded 60-step rollout); post-switch arm-aware dynamics proved controller identity INFORMATIVE (0.1228 vs 0.1364 m/s, p=0.0002); reconstructor v5 beat v4 on every stored metric (RMSE 2.27, ramp 1.55 via the owner-authorized RSU ramp sensor, ablation-proven -39%); horizon study proved 60 s scoring INSUFFICIENT (top-1 0.56 vs 0.72-0.76 at 600 s) -> planner scoring moved to 600 s with ties-default-native; wall audits: history multiplies episodes ~30x (era-tagged) but wall-v2 engagement is 380 vehicle-seconds and assembly dv needs command/owner inputs - Phase-B wall matrix specified (12 runs); dispatcher built and unit-tested (13 tests) with wall-subset execution stubbed pending a subset-roster wall session. PLANNER VERDICT: SAFE but NOT ADDITIVE - raw imagined argmax mis-ranks vs measured outcomes (18/54 holdout recovery); the tie-to-native rule recovers exactly always-native performance (regret 0.19; fleet 0.534 vs 0.509); 30,566 labeled decisions generated for a future learned proposer. Registered next steps: Phase-B mixed-assignment runs (per-CAV effects are extrapolation until measured), Gate-3 coverage-input fix (degenerate regularizer input identified), command-aware wall features. [Reports](../output/worldmodel_v3_planner/RESULTS.json).

<!-- CV313210-status -->
CV-31 + CV-32 + WM-10 completed (2026-09-22): the coverage extension delivered 15 accepted arms across five fresh cells (7533 safety-closed by a retain-arm collision, evidence preserved) plus the first five layered-verified wall arms on the fixed hold-phase supervisor (zero collisions). New structure discovered: the first retain-best cell ever (7535m2400 - damping wins at mid-demand), and opposite native/PI winners at matched m2700 demand (7531 PI-best vs 7534 native-best). Wall verdict: operable and safe; fully engaged at 7531 and lost to all three controllers; gate correctly suppressed entry at 7535; never beat a cells best controller, never hurt vs retained damping. Selector verdict (WM-10): the enlarged corpus did NOT fix generalization - a fresh never-trained congested cell (7534) still picked PI on 91% of windows (+7.6 s/veh regret), so the native-vs-PI moderator is NOT identifiable from the current lane-split state features at this scale; more same-kind seeds is ruled out as the fix and the registered next step is moderator identification. Always-native remains the correct deployment policy (+0.38 vs selector +7.14 s/veh mean holdout regret). [Reports](../output/worldmodel_v2_full/REHEARSAL_REPORT.md).

<!-- CV31-status -->
CV-31 completed: 15 of 18 full-horizon coverage arms (cells 7531-7534 at m2700, 7535-7536 at m2400, three arms each); 38400 s charged / 38400.0 s actual; per-cell 600 s prefixes verified by the reviewed CV31 oracle; closed cells preserved with their charges. Sign-consistency over cells with all three arms valid (5/6): native beat retained CIRCLES on all five costs in 4; PI beat retained CIRCLES on all five in 3; PI beat native on all five in 0. Per-request service deltas, treatment minus reference (negative = improvement): 7531m2700 native-retain -1.203252, pi-retain -5.022173 | 7532m2700 native-retain -18.432568, pi-retain -17.740674 | 7534m2700 native-retain -20.746110, pi-retain -12.935644 | 7535m2400 native-retain +6.086056, pi-retain +3.524303 | 7536m2400 native-retain -1.248216, pi-retain +0.241475. One scene per cell; no retry, fitting or acceptance access; no switching rule, generalization or world-model claim. [Report](../output/controller_verification_v31/REPORT.md).

<!-- CU01-status -->
CU-01 complete (2026-09-22, zero simulation): per-controller use-case research + benefit-categorization redesign audit; six reports + synthesis in output/controller_use_case_research_v1/. Findings: native IDM wins service 9/9 shared-prefix cells (all-five 7/9; zero native-arm collisions) and is the evidence-supported retention default; retained CIRCLES has no remaining active-controller use case locally (0/11 stage-1 best-suffix; dominated on all five costs at m2700 in CV-30) - its load-bearing role is the frozen reference instrument; owner pivot offered: re-baseline the reference to native. PI: partial winner (effort 3/6 cells; beats retained CIRCLES all-five 4/6), never all-five dominant vs native (0/6); the unexplained same-load service sign flip (+23.7/-29.2 across 4107/4108) is the top selection gap. FollowerStopper: braking specialist whose literature window (slow covered leader + wave state + wave-matched U) is unreachable at the fixed road-limit U=31.29 (CV-06 +36.25 s/req vs native at m2700). Wall: the four-joint beneficial window (discharge-inflection entry at >= capacity demand, near-merge compact hold, hold until storage drains, capped-outflow taper) captured every executed benefit (-23.14/-11.30 s/req; WL-04 harness -30.2 s/veh) and no current gate captures it (K1 B 4/30 / C 2/30 with 2 shared; wall-v2 E1-E5 also refuses the beneficial K1 entry). Categorization: four incompatible yardsticks coexist and 12/29 recorded HARMFUL contrasts have service improving; recommendation adopted-as-proposal: five-category Pareto taxonomy now (57 recorded contrasts -> 28 dominant / 4 trade-off / 2 mixed / 13 guardrail-harm / 10 service-harm), switching-regret as the declared decision estimand with abstention default, two-seed replication as the action gate (all current per-cell labels UNRATED under it). Owner decisions requested: (1) reference re-baselining (arm A -> native), (2) taxonomy adoption. No controller accepted/deployed/switched; no simulation/fitting/acceptance access. Ledger CU-01; validate 0 hard errors; verify-sources 0 mismatches; 47 ledger tests pass. [REPORT](../output/controller_use_case_research_v1/REPORT.md)

<!-- WM09-status -->
WM-09 reconstructor fix complete (zero simulation): v3’s Gate-1 failure was root-caused to an unmasked speed loss over ~67% empty cells (−10.1 m/s occupied bias; the trained model was 2.5× worse than copying the last report) plus rank-64 linear heads unable to carry 240 per-cell priors. Fix: occupancy-masked regression + per-head climatology-initialized per-cell bias, encoder untouched. Holdout: RMSE 11.85→3.28 m/s (beats persistence 4.66 and climatology 6.75), bias +0.70, correlation ≥0.84, worst ramp cell 14.9→4.60, Gate 1 PASS (hit@10 100%, hit@5 84.8%), Gate 2 PASS improved (64.2%); Gate 3 spread still FAIL (architectural). Rehearsal: reconstruction GO — but the fix exposed the selector as the sole blocker: always-native policy scores +0.38 s/veh mean holdout regret vs the learned selector’s +9.13 (cell 7012 +26.0). Closed-loop switching stays NO-GO; registered fixes are more outcome-labeled cells per demand regime or a validated moderator feature. [Report](../output/worldmodel_v2_full/REHEARSAL_REPORT.md).

<!-- WM08-status -->
WM-08 world-model build complete (zero simulation; trained on registered CV-17/19/20/21/23/25/26/27/28/30 evidence, 52 train / 15 unique holdout runs): three-stage model trained and saved (best-of-5 lane-split reconstructor, 3-member per-action dynamics ensemble, outcome-supervised 4-arm selector; labels rebuilt programmatically from all 70 complete run summaries — retained damping is never argmin-cost across 22 cells; native best in 12/20 comparable cells, PI in 8). Rehearsal on never-trained cells: dynamics beats persistence on all 13 holdout runs incl. the wall arm; consistency reweighting passes head honesty (54.6% vs 20% chance); lane-cell reconstruction (hitting@10 22.7%) and cross-cell selector generalization (7012 closed-loop regret +14.3 s/veh; cell-identity confound from one-trajectory-per-cell-arm labels) FAIL — overall NO-GO for closed-loop use; spread-coverage failure replicated a 4th time (architectural). Fixes landed this pass: hyphen run-tag parsing, reconstructor dedup key keeping same-cell different-arm runs, CV-30 + wall extraction, duplicate holdout extractions removed, placeholder outcome table replaced. [Report](../output/worldmodel_v2_full/REHEARSAL_REPORT.md).

<!-- CV30-status -->
CV-30 completed: 18 of 18 full-horizon extension arms (cells 7301-7306, three arms each; 7301-7303 at m2000, 7304-7306 at m2700); 43200 s charged / 43200.0 s actual; per-cell 600 s prefixes verified by the reviewed CV30 oracle; closed cells preserved with their charges. Sign-consistency over cells with all three arms valid (6/6): native beat retained CIRCLES on all five costs in 5; PI beat retained CIRCLES on all five in 4; PI beat native on all five in 0. Per-request service deltas, treatment minus reference (negative = improvement): 7301m2000 native-retain -1.160107, pi-retain -1.082290 | 7302m2000 native-retain -1.565059, pi-retain +0.388080 | 7303m2000 native-retain -1.000450, pi-retain -1.196670 | 7304m2700 native-retain -17.517986, pi-retain +0.380216 | 7305m2700 native-retain -23.752149, pi-retain -11.365688 | 7306m2700 native-retain -10.224050, pi-retain -0.626827. One scene per cell; no retry, fitting or acceptance access; no switching rule, generalization or world-model claim. [Report](../output/controller_verification_v30/REPORT.md).

<!-- stage1-corpus-status -->
Stage-1 corpus COMPLETE: 30/30 point-controller arms accepted across ten labeled cells (7001-7008, 7010-7012; 7009 safety-closed by 4 collisions on its retain arm) plus CV-17/19 origins. Per-seed best-suffix by service: native on 7001/7003/7006/7008/7012; PI on 7002/7004/7005/7007; at m2000 the labels split 3/3 — the flip is not explained by demand regime. Wall suffix (CV-26): 7012 accepted at run time (first verified wall arm, layered tier); 7010 re-adjudicated PASS after the reviewed automaton vocabulary fix; 7007 FAILS the registered span contract — genuine supervisor defect: hold-time member drift to 160 m when member distance reports are unavailable (layered verification surfacing an observation-dependent enforcement gap); 7008/7011 lost to a concurrent canonical-rebuild launch race (zero evidence; successor re-run candidates). Lock-family failures decisively repaired (any-denial retry + non-fatal progress marking). Stage-2 training data complete: receipt-causal inputs + oracle-verified ground truth, 4-action labels. [Report](../output/worldmodel_stage1_corpus/REPORT.md).

<!-- CV19-status -->
CV-19 completed: 6 accepted integrity-verified full-horizon shared-prefix arms across exposed 4107/4108 m2000; 14400 s charged / 14400.0 s actual; all per-cell 600 s prefixes verified by the reviewed CV19 oracle. The same-state suffix ordering differs between the two cells. Per-request service deltas, treatment minus reference (negative = improvement): 4107 native-retain -0.224348, pi-retain -1.182609 | 4108 native-retain -1.684659, pi-retain -1.547822. One scene per cell; no retry, fitting or acceptance access; no switching rule, generalization or world-model claim. [Report](../output/controller_verification_v19/REPORT.md).

<!-- CV17-status -->
CV-17 completed: 3 integrity-verified full-horizon shared-prefix arms; 7200 s charged / 7200.0 s actual, with the stopped CV15 charge of 2400 s preserved separately. Both retain-to-native and retain-to-PI 600 s prefixes verified by the reviewed CV19 oracle; zero collisions or teleports in all arms. Descriptive same-state suffix ordering on exposed 4108 m2700: native lowest, PI intermediate, retained CIRCLES highest on all five continuous costs (native minus retained service -9.471691 s/request; PI minus retained -4.380260; PI minus native +5.091431). PI suffix delivery fell below target on 15393/25802 assessed endpoints (mean absolute error 0.8895 m/s). One scene; no retry, fitting or acceptance access; no switching rule, generalization or world-model claim. [Report](../output/controller_verification_v17/REPORT.md).

> Current update CV-16: offline bounded budget-write repair verified (32 scoped tests); no simulation. CV15 remains stopped, 2400s charged/190.5s reached. Controller switching benefit remains untested. See ../output/controller_verification_v16/REPORT.md.

<!-- CV15-status -->
CV-15 stopped: first retained-CIRCLES replay failed at190.5s on atomic budget-file replacement (Windows PermissionError).2400s charged/190.5s reached; complete metrics through190.0. Original accounting preserved and reconciled; no switch, efficacy result, retry or successor allocation. Next: zero-simulation bounded I/O-retry repair with permanent-failure tests, preserving sticky STOP and reservation charges. CV14 offline interface proof stands; physical shared-prefix comparison remains untested. [Report](../output/controller_verification_v15/REPORT.md).

<!-- CV14-status -->
CV-14 complete: opt-in one-boundary fleet switch between native/CIRCLES/PI verified through the actual evaluator with mocked physics. Shared-prefix traces, actual native handback, PI resets/retained recurrence and failure evidence checked; existing default traces retained. Zero simulation or fitting. Next: separately registered physical identical-prefix replay, charging every replay and retaining native as mandatory comparator. Switching benefit and world-model utility remain unproven. Supersedes CV13 pending offline switch proof. [Report](../output/controller_verification_v14/REPORT.md).

<!-- CV13-status -->
CV-13 completed: 2 integrity-verified full-horizon PI cases; 4800 s charged / 4800.0 s actual. Physical benefits and harms versus native/CIRCLES and tracking limitations are preserved separately. No retry, fitting or acceptance access. Unchanged-controller replication completed; repeatability and tradeoffs are reported without a predictor claim. At high load, PI service effects reverse across exposed seeds4107/4108; there is no universal superiority or verified selection rule. Next is an offline proof of a shared-history controller-switch interface, with native retained as a mandatory comparator. World-model training remains downstream of useful control evidence. This supersedes CV12 pending replication. [Report](../output/controller_verification_v13/REPORT.md).

<!-- CV12-status -->
CV-12 completed: 2 integrity-verified full-horizon PI cases; 4800 s charged / 4800.0 s actual. Physical benefits and harms versus native/CIRCLES and tracking limitations are preserved separately. No retry, fitting or acceptance access. Lower-load service improves while higher-load service worsens; next is unchanged-controller replication on both loads of a second exposed seed, separately registered. world-model training remains downstream of useful control evidence. This supersedes CV11 pending characterization. [Report](../output/controller_verification_v12/REPORT.md).

<!-- CV11-status -->
**CV-11 complete:** [Supported multi-CAV PI integration](../output/controller_verification_v11/REPORT.md). Fixed-U31.29 opt-in path verified offline with real ownership and mocked physics, including partial failures and lifecycle reset. Default command/cost regressions preserved. Zero new simulation. Next: source-compatible matched PI traffic characterization, proposed two2400s cases with registration and first-run integrity/safety gate; no PI highway benefit or world-model claim yet. This supersedes CV10 pending evaluator integration.

<!-- CV10-status -->
**CV-10 complete:** [PI remaining branch mechanics](../output/controller_verification_v10/REPORT.md). Two2s cases passed source/owner/safety checks;4s charged/actual. Braking target delivered exactly; mixed target undershot by0.3993m/s. CV08+CV10 close planned PI mechanics. Next: supported multi-CAV integration, then bounded matched traffic characterization; no highway benefit or world-model claim yet.

<!-- CV09-status -->
**CV-09 complete:** [PI source/plant reconstruction](../output/controller_verification_v9/REPORT.md). All14 CV08 half-step speeds match pinned SUMO IDM/TraCI calculations exactly:2 ramp-limited,8 PI steps IDM-limited,4 native. Zero new simulation. This refines CV08 delivery interpretation for this fixture only. Next: two proposed missing PI branch fixtures (4s not allocated), then supported integration/characterization; explicit literature-based highway-FS adaptation remains an alternative.

<!-- CV08-status -->
**CV-08 complete:** [PI live mechanics and source audit](../output/controller_verification_v8/REPORT.md). One8s run passed recurrence/release/reset/safety checks; 1/5 targets delivered within0.1m/s,4/5 below. No highway efficacy or world-model claim. Modified-FS numerical source calibration remains unresolved. Continue controller-first characterization under the reviewed actual plant interface.

<!-- CV07-status -->
**CV-07 completed:** [PI ownership integration](../output/controller_verification_v7/REPORT.md). Real owner path verified offline for per-vehicle acknowledgements, raw PI recurrence, resets and partial failures. No live PI, fitting or simulation. CV06 high-load tracking diagnosis completed; published highway-FS variant identified for audit. Next: source-supported highway candidate audit and bounded PI live-mechanics preparation; useful controller/world-model benefit still unproven. This supersedes CV06 pending offline PI integration; preserves all physical negatives.

<!-- CV06-status -->
CV-06 completed: 2 integrity-verified full-horizon FS cases; 4800 s charged / 4800.0 s actual. Physical benefits and harms versus native/CIRCLES and tracking limitations are preserved separately. No retry, fitting or acceptance access. Next controller work follows the reviewed PI integration plan; world-model training remains downstream of useful control evidence. This supersedes CV05 pending characterization. [Report](../output/controller_verification_v6/REPORT.md).

<!-- CV05-status -->
**CV-05 completed (2026-09-20):** [Supported FS evaluation integration](../output/controller_verification_v5/REPORT.md). Opt-in explicit-U FollowerStopper now reaches the supported evaluator with distinct causal audit/issued/measured telemetry. Successful mocked integration and failure accounting verified; eight exposed CUR-04 reference bindings audited for scoped reuse. Zero simulation/fitting. Next: register and verify the budgeted wrapper/source freeze, then two proposed full-horizon FS cases using four matched references. 4800 s is proposed, not allocated; no merge efficacy or world-model claim follows. This supersedes CV-04 pending evaluator integration; PI runtime remains open.

<!-- CV04-status -->
**CV-04 completed (2026-09-20):** [Close-leader controller verification](../output/controller_verification_v4/REPORT.md). Three 6 s FollowerStopper cases passed command/safety integrity, with 14/15 issued targets delivered and one measured response 0.697238409606755 m/s below target. 18 s charged/actual, no retries or collisions/teleports. Across CV-03/04 all four source-law regions and native release have been exercised. PI external-U is an isolated proposed adaptation, not a runtime controller. Next: versioned supported merge evaluation integration and dependency-audited reference reuse, then bounded physical characterization. Merge benefit and world-model value remain unproven. This supersedes CV-03 close-leader pending status; historical evidence is preserved.

<!-- CV03-status -->
**CV-03 completed (2026-09-20):** [Live FollowerStopper mechanics](../output/controller_verification_v3/REPORT.md). Two matched10s branches:13/13 issued-target endpoints matched exactly, real native release verified (+2.5119134514880788m/s at2s), zero collisions/teleports.20s charged/actual;no retries. Mechanics only: close-leader response,PI highway contract and matched merge efficacy remain open;world-model learning has not started. Frozen CIRCLES unchanged. Verification limitations are explicit in the report. This supersedes CV-02's pending live-check status.

<!-- CV02-status -->
**CV-02 completed (2026-09-20):** [Analytical source and target-speed interface verification](../output/controller_verification_v2/REPORT.md). PI source/startup and opt-in FollowerStopper adapter verified offline by Luna execution and Astra review; 52 scoped tests pass. Next: registered 20-second FS live mechanics. Zero simulation/fitting; default CIRCLES unchanged.

<!-- CV01-goal-start -->
**Active owner goal (2026-09-20):** [Uncertainty-aware traffic regulation from sparse CAV observations](WORLD_MODEL_RESEARCH_GOAL.md). Verify literature-supported controllers first, then establish state-dependent control opportunity before world-model learning. Walls are optional. Astra plans/reviews; Luna executes bounded tasks. This direction supersedes wall-only planning below; frozen baselines, evidence and safety/information constraints remain preserved. CV-01 initial audit and isolated FollowerStopper equation verification are complete (zero simulation/fitting); full controller verification is ongoing.
<!-- CV01-goal-end -->

<!-- OE71-current-start -->
**OE-71 completed (2026-09-16):** [Bounded B-versus-damping predictor check](../output/oe71_selection_v1/REPORT.md). The existing 28 causal R1 features and fixed Ridge model were tested by holding out each of 12 exposed admitted seed groups. The model selects B seven times: 2 beneficial, 4 harmful, 1 nonbeneficial. It misses two benefits and avoids two harms. Always B and the training-only constant retain all four benefits with six harmful interventions. Mean service and effort are worse than always B; useful policy selection is not demonstrated. OE-70 physical benefits remain valid. Zero simulation; one 12-fold exploratory fit, no fresh holdout or successor work.
<!-- OE71-current-end -->

<!-- OE70-current-start -->
**OE-70 verified (2026-09-16):** [Independent source reconstruction](../output/oe70_verification_v1/REPORT.md). OE-69 retirement rationale withdrawn: 30 new triplets across five loadings; B=4 beneficial/6 harmful/20 nonbeneficial, C=2/5/23. Two both, two B-only, no C-only. Historical m2700 A/C separate; wrong-policy B excluded. B-versus-damping selection remains unvalidated. Current task: zero simulation/fitting; prior90newcorridors charged216000s. Earlier OE-69 current-state blocks below are superseded and retained as history. No successor execution authorized.
<!-- OE70-current-end -->

<!-- OE69-current-start -->
**OE-69 completed (2026-09-16):** [Definitive analysis: retire the observation-triggered K1 formulation](../output/oe69_integration_v1/REPORT.md). 32 complete A/B/C triplets analyzed from source summaries. C (filtered gate): 4 QUALIFIED / 28 HARMFUL. B (original gate): 4 QUALIFIED / 28 HARMFUL. Both qualify on the same 2 seeds; each uniquely on 2 more; neither on 26. The filter exchanges which scenarios benefit but does not increase the qualifying count. 81% of scenarios show no qualifying wall. Action value concentrated at m2700. DECISION: RETIRE this formulation. No learning pilot justified. Infrastructure contributions preserved. Zero simulation; zero fitting; zero acceptance access.
<!-- OE69-current-end -->

<!-- OE67-current-start -->
**OE-67 completed (2026-09-15):** [Integration of the repaired launch path](../output/oe67_integration_v1/REPORT.md). Gate guard wired into production B/C construction (filter references the same guarded gate); scenario binding wired into launch_corridor; CLI built with validate-package/dry-run/run-panel verbs. Successor package: 36 scenarios x 3 arms; 6 reuse (m2700-registered seeds) + 102 new = 244,800 s proposed. READY FOR AUTHORIZED EXECUTION with noted CLI limitations. Zero simulation.
<!-- OE67-current-end -->

<!-- OE66-current-start -->
**OE-66 completed (2026-09-15):** [End-to-end repair of OE-65](../output/oe66_integration_v1/REPORT.md). Four root causes established and corrected: (D1) arm B was constructed as fixed-t600/1s instead of the observation gate; (D2) loading hardcoded 2700 in the runner job dict; (D3) gate episode provenance became stale after expiry (live property recomposed from mutable markers); (D4) the legacy runner consumed OE-65 configs instead of the dedicated executor. All repaired: arm B now uses the observation gate (no filter/no fixed time); arm C = identical + zero-slow filter; loading propagates from config; GateEpisodeProvenanceGuard freezes episode provenance at crawl confirmation; scenario_binding.py verifies registered (seed,loading) against generated route hashes with TOCTOU re-hash. Evidence corrected: 20 completed runs classified (4 valid_for_executed_config / 7 wrong_policy / 9 wrong_loading / 1 partial_failed); all B arms verified bit-identical to A; all used m2700 loading. VERDICT: READY FOR AUTHORIZED EXECUTION - the corrected launch path produces the right policies with the right scenarios. Zero simulation; zero fitting.
<!-- OE66-current-end -->

<!-- OE65-current-start -->
**OE-65 completed / INCOMPLETE (2026-09-15):** [Dataset-and-learning study](../output/oe65_integration_v1/REPORT.md). 20/108 corridors completed before the zero-slow filter's temporal defect (armed/triggered inconsistency in delayed admissions) caused a campaign STOP. Preliminary results: 920038 C-vs-A **QUALIFIED** (all five costs improve, service -11.30 s/req) on a fresh world where the original gate never fired - the second demonstrated beneficial configuration after OE-59 920002. All B arms verified-null on fresh worlds. Lower-demand scenarios (1900-2200) showed little action value. 50,400 s charged of 259,920 authorized; 209,520 lapsed. The filter temporal defect requires a versioned fix before panel resumption. Zero fitting; zero training; zero acceptance access.
<!-- OE65-current-end -->

<!-- OE64-current-start -->
**OE-64 completed (2026-09-15):** [Controller-contract investigation](../output/oe64_integration_v1/REPORT.md). Zero simulation. The 470 s abort was CORRECT under the implemented contract: with measured -0.4923 and prior command +0.0537, the command intersection was EMPTY and the floor band [-0.3,-0.15] vs measured band [-0.6423,-0.3423] are disjoint (gap 0.0423); commanding -0.3 requires +0.1923 > the 0.15 ramp (OE-63's "inside every limits" claim corrected; strong-form H3 withdrawn). The plant braking was leader-driven (slower leader acquired at 469, -3.05 m/s closing); the exact SUMO mechanism is unidentified. The 463 plan was kinematically honest (min 24 closure steps vs the 48-step coordination horizon); the failure was model-structural. OWNERSHIP verified from raw rows: 318 -> live damping (-0.4056, below the wall floor - the floor binds only the wall envelope, by design); 312 -> native (missing_or_uncovered_leader, setSpeed -1), resuming damping at 523; zero wall commands after 469; re-engagement prohibited. Option-B prototype built and verified (isolated, bounded re-entry): ZERO eligible re-engagement ticks on the recorded 920004 world through 2399 - the contract change would never have fired. DECISION: contract A - permanent handback is the coherent policy; filtered formulation PAUSED on filling-crawl states (K1 cannot assemble at 184-200 m span under real safety responses); option-B prototype retained as an owner-proposal artifact; training-readiness unchanged (not justified). Zero simulation.
<!-- OE64-current-end -->

<!-- OE63-current-start -->
**OE-63 completed (2026-09-14):** [Continuation-or-pivot checkpoint: the 470 s abort root cause](../output/oe63_integration_v1/REPORT.md). Zero simulation. ROOT CAUSE (three independent reconstructions, 16-digit agreement, over-determined): the wall was still ASSEMBLY at 470 (ready_at never set; member span 184-200 m vs the 50 m formation requirement; HOLD never started); mainline.318 acquired a slower leader at 469 and SUMO safety-braked it to measured -0.4923 m/s2 while the stale wall command was +0.0537 - the ramp-band interval was EMPTY (lo -0.0963 > hi -0.3423, gap 0.246) and the protective branch correctly aborted. The 463 admission passed under a commanded-response convergence forecast the real traffic violated within 7 ticks. H-VERDICTS: H3 strong form REFUTED (the excluded continuation was smooth and inside all limits); H2 confirmed at 470; H1 DEMONSTRATED on the neighboring band (admission/runtime ramp-anchor inconsistency, outcome-determining for measured in (-0.45, prior-0.3); does not flip the actual 470 state); H4 resolved. The abort was CORRECT protection. ACTION CENSUS: 6 executed K1 activations -> 2 qualified (both the 920002 discharge-inflection world), 4 harmful (all mid-crawl/filling entries); 2 executed protective aborts; 38 of 40 offline admissions unexecuted. DECISION B: action limitation identified (ramp-band bookkeeping without recovery semantics; anchoring inconsistency; assembly convergence model violated) - the filtered formulation is PAUSED; a versioned ramp-recovery repair is a controller-contract change requiring its own validation; 920005-920007 remain registered-unspent. Zero simulation.
<!-- OE63-current-end -->

<!-- OE62-current-start -->
**OE-62 completed (2026-09-14):** [Staged continuation: zero-slow filter productionized, 920004 development test](../output/oe62_integration_v1/REPORT.md). STAGE 1 (offline): OE-61 counts corrected (original admissions 58/63; revised 40/63; 18 removed; 5 already nonadmitted - the summary bug was the category NAME, not aggregation); 920004 cohorts pinned (orig 354s mainline.235/236; revised 463s mainline.312/318); 920002 REUSE_VERIFIED (the executed beneficial outcome IS the revised policy's outcome - binding fingerprints recomputed from the executed run telemetry); 25 unchanged confirmed not reusable physical outcomes; production integration causal_h5_entry_live/3 (post-admission behavior identity-tested vs the executed runtime; W3 caught a wrong-module BLOCKER, fixed and re-verified). STAGE 2 (920004, 2400 s, prefix identity VERIFIED through both admission ticks): admission at exactly 463 as predicted, but the wall NEVER REACHED HOLD - protective EXIT_INFEASIBLE abort during assembly; REVISED-vs-A service +4.7383/TTC +0.0848 HARMFUL (vs original C +32.8844; revised is 28.15 s/req less harmful). FROZEN GATE: GATE_CLOSED_NOT_QUALIFIED - stage 3 NOT proceeded; 920005-920007 registered and unspent. TRAINING-READINESS: observation causal and online; it changes deployed decisions; its one live test REDUCED HARM but remained harmful; no action headroom demonstrated on filling-crawl states; a learning pilot is NOT justified. Zero fitting; zero training; zero acceptance access; 2,400 s charged of 24,720 authorized; OE-57/59 STOPs and configuration-stop disclosures preserved.
<!-- OE62-current-end -->

<!-- OE61-current-start -->
**OE-61 completed (2026-09-14):** [Decision replay: the zero-slow admission check](../output/oe61_integration_v1/REPORT.md). Offline decision replay of the retrospectively-motivated rule (at admission, additionally require slow_count==0 over the gate's current near-merge reporter snapshot; slow = speed < 1.0 strictly; refusal named and non-mutating). All four source checkpoints reproduce exactly (920002@609 0/7; 920003@453 1/8; 920004@344 1/5; @354 2/5). Population: 63 damping streams, original decisions reproduced exactly first; revised outcomes UNCHANGED 25 (39.7%) / DELAYED 9 / CHANGED_COHORT 6 / NONADMITTED 23 (36.5%). Panel fates: 920002 UNCHANGED (beneficial wall stands); 920003 NONADMITTED (the +13.96 harm would not have occurred); 920004 CHANGED_COHORT (refused 354, admitted 463 with a different cohort; the +32.88 harm would not have occurred as observed; delayed outcome UNKNOWN). Behavior: wait-for-observed-recovery dominant (12 genuine recoveries vs 2 turnover), feasibility loss the main cost (15 streams/4 episodes). DECISION: minimal bounded physical test proposed (revised policy vs damping, same three fresh scenarios, 2x3x2400 = 14,400 s, NOT authorized) - decision-level evidence only, changed-decision outcomes remain unknown. Zero simulation; zero fitting; zero acceptance access.
<!-- OE61-current-end -->

<!-- OE60-current-start -->
**OE-60 completed (2026-09-14):** [OE-59 panel diagnosis and record repair](../output/oe60_integration_v1/REPORT.md). Zero simulation. ADMISSION DISCRIMINATOR identified (receipt-causal at each decision tick): 920002 fired at the discharge-to-crawl inflection (5 s after discharge, slow share 0.000) and qualified; 920003 fired mid-crawl (90 s search, slow share 0.125) and 920004 into a still-slow crawl (0.400, with an in-episode infeasibility receipt) - both harmed. Static quantities flip direction at B@600 (decision-moment properties, not seed discriminators). K1 hold placement couples with outcome in both both-activate seeds (closer-to-merge hold better); delivered speed reduction refuted as carrier. FLOW: effort diverges at entry, braking/TTC +46-77 s, service +110-310 s; 920002 drains upstream storage, 920003/920004 grow it; final crossings equal (transient discharge/storage). INTEGRITY repaired: overage exactly +2,380 s recorded component-wise (2,400 quarantined - 20 unspent B headroom; master ledger corrected with explicit OE-59 corrections entry); quarantined-reservation record fixed (2400 completed, path repointed, originals preserved); study.py exact enforcement restored via REGISTRATION_IDENTITY_MIGRATION.json (unapproved changes rejected, doctored-input tested); B CONFIG reconciled; exposure re-check run retrospectively (clean=false, labeled); budget-chain enforcement mock-tested. DECISION A: "refuse to arm while any near-merge slow reporter remains at crawl confirmation" - prospective test specified (state-conditioned admission on the same three scenarios vs existing A arms; rejected if conditioned admissions still harm or the gain fails), no fitting, no thresholds tuned here. Zero simulation; OE-59 STOPs and receipts preserved.
<!-- OE60-current-end -->

<!-- OE59-current-start -->
**OE-59 completed (2026-09-14):** [Integration finish and the conditional physical panel](../output/oe59_integration_v1/REPORT.md). Both mechanics PASSED with the scheduled HOLD+5 release executing LIVE (C s90031: hold 499-503, request exactly 504; B s90051: hold 607-611, request 612) - the capability that blocked OE-57. Nine-corridor panel on fresh seeds 920002-920004: all arms complete, zero collisions/teleports, 5/5 executed walls delivered exactly 5 hold ticks with scheduled release at hold_start+5, one refusal (920004 B, infeasible at 600) retained as damping with bitwise-verified identity. RESULTS (per-request, negative=improvement): 920002 C-vs-A service -23.14/braking -0.101/effort -1.79/TTC -0.233/wait +0.004 = BENEFICIAL (all five improve; C-vs-B service -15.42 also beneficial); 920003 C-vs-A +13.96 service/+0.153 wait over allowance HARMFUL; 920004 C-vs-A +32.88 service/+0.099 TTC over allowance HARMFUL (B refused at 600, bitwise-identical to damping). Ladder: UNRESOLVED_MIXED - the unresolved question is why gate-derived entry helps strongly on one fresh world and harms on two (cohort/entry-state/dynamics). Three seeds are a screen, not validation. Charged 26,275 s of the 22,895 s envelope (overage = one 2400 s first-corridor run quarantined after the pre-repair clock-check false positive; check repaired, replacement charged within the panel; OE-57 STOP untouched; predecessor campaign STOP preserved per OE-51/36 combined-accounting pattern). Zero fitting; zero acceptance access.
<!-- OE59-current-end -->

<!-- OE58-current-start -->
**OE-58 completed (2026-09-14):** [Consolidated OE-57 repair](../output/oe58_integration_v1/REPORT.md). Zero new simulation. CAUSE corrected from source (three independent reconstructions; margin reproduced bit-for-bit): the wall committed to HOLD until 344 s (ready_at 324 + hold_s 20); the per-tick remaining-plan check found the committed path intersecting reported traffic (same-lane downstream reporter mainline.168, decelerating; forecast margin -2.61153 m; forecast violation, not a collision/clearance violation); exit_now feasible -> protective taper 326, coast 327. The OE-57 receipt's admission+30-timer claim WITHDRAWN (arithmetic coincidence); "commitment arithmetic" claim withdrawn; the receipt REMAINS FAILED under its original criteria. Legitimate: the protective exit and the wrapper not firing post-exit. Defective: no preemption event/marking and the RELEASED_HANDBACK misclassification (no protective-taper class); wrapper fail-closed unreachable after self-taper. K1 contract ACCEPTED (H5 = early-exit-of-H20; single +5 attempt through the full proof; protection governs; explicit events; missed release fails explicitly) and implemented as causal_h5_entry_live/2 (37 tests + 9 adversarial over real objects). PLANNING DIAGNOSTIC (conditional, labeled): H20 margin -2.61153 reproduced; hypothetical H5-remaining plan +263.01 m - the same reporter does not intersect the shorter plan; proves nothing about safety/delivery. HASHES fully explained (reports: window double-count, live side correct; gate_features: two projections of one state; canonical serializer makes the saved pair MATCH). CLOCK checks repaired (opportunity-derived, gaps fail, ownership distinct). MECHANICS staged: M-A1 = s90031, admission 475, horizon 575 s <= 720 (only pre-qualifying stream; PASS requires the scheduled path; protective preemption recorded as UNMET); requirement B answered offline on the saved C run (12/12). Proposed next: authorize M-A1 (575 s) under new registration. Zero fitting; zero acceptance access; OE-57 STOP and 396 s receipt preserved.
<!-- OE58-current-end -->

<!-- OE57-current-start -->
**OE-57 completed / BLOCKED AT STAGE 2 (2026-09-13):** [Runtime integration attempt and mechanics failure](../output/oe57_integration_v1/REPORT.md). Stage-1 offline integration complete (runtime bridge, strict frozen grading, guarded runner with offline B-admission pre-check, fresh-seed registration, analysis pipeline, independent review - all suites green). The C mechanics allocation (396 s, exposed s90028) executed the gate->admission->binding->assembly chain LIVE and reproduced the offline replay decision exactly (admission 296, cohort {mainline.180,183}) - then FAILED the frozen HOLD+5 release check: delivered hold 2 s, taper at admission+30 s via the wall's own commitment clock while the prospective release wrapper silently yielded (fire branch and missed-release error both gated on taper_at-is-None; requested/verified stayed False throughout - precise location in live_policy.py _wall_step). B mechanics and all nine corridors NEVER launched per the stop rule; 396/23,040 s charged; campaign oe57_runtime_panel STOP preserved; no physical benefit or harm measured. Remaining requirement: repair the release wrapper's fail-closed semantics (+ the missing self-preempting-wall test), decide the wall-commitment root cause, re-register under NEW authorization reusing the staged package. Zero fitting; zero acceptance access.
<!-- OE57-current-end -->

<!-- OE56-current-start -->
**OE-56 completed (2026-09-13):** [Consolidated OE-55 repair and readiness verdict](../output/oe56_integration_v1/REPORT.md). Zero-simulation consolidated repair. GATE_CONTRACT_V2 frozen before results (queue requires stopped presence; discharge requires a vehicle-level recovery witness; stop-free crawl; mutually exclusive predicates; per-tick report-derived admission eligibility replacing the v1 held-true convention). Gate v2 implemented and independently reproduced from the contract text alone (6 deep cases exact; 57/60 population exact, 3 witness-count-only differences - G-F3 resolved as a recorded conservative deviation). One frozen 60-stream replay (manifest-bound): 55 ADMITTED / 5 TRIGGER_THEN_PHASE_LOSS (phase_condition_lost); trigger median 457.5->538.5; retention 28 identical / 27 re-timed / 5 lost; 646 state-condition + 313 infeasible + 38 no-cohort honest refusals; eligibility lapsed on 59/60 streams; 90037 unchanged (531/571/{403,396}) - the motivating 600/{409,411} entry still NOT reproduced and the gate still fires on 60/60 (weak selectivity carried honestly). Receipt-window fractional-time repair in the production helper (latent defect: zero fractional receipts in the 60 streams). Comparator gap resolved by ONE shared prospective wrapper causal_h5_entry/1 (B/C byte-identical machinery on identical inputs; historical fixed-H5 contract shown unreproducible from damping snapshots). DECISION READY-TO-REGISTER: launch package = 3 arms x 3 fresh seeds x 2400 s + 1,440 s mechanics = 23,040 s proposed (NOT authorized); ITT accounting; frozen decision ladder; named blocker BLOCKER-FRESH-SEEDS (0/60 of 90001-90060 fresh; owner allocates 3 unexposed identities). No claim that offline feasibility, triggering, or passing tests demonstrate beneficial wall selection. v1 documents bannered, results preserved; run-1 bookkeeping defect quarantined with byte-identical rerun.
<!-- OE56-current-end -->

<!-- OE55-current-start -->
**OE-55 completed (2026-09-13):** [Causal post-discharge-crawl entry gate](../output/oe55_integration_v1/REPORT.md). Zero-simulation implementation + 60-stream causal replay. Five OE-54 overclaims corrected and bannered (P2: true availability lead 17/1/-23 s, not ">=136 s"; the surviving re-development monitor is a different trigger than the crawl state evaluated here). Frozen queue->discharge->crawl gate (receipt-causal, missing-data-safe, research candidate only) + unchanged-selector action contract with frozen refusal taxonomy. Replay: ALL 60 compatible damping streams fire (median trigger 457.5 s) and ALL admit feasible 2-CAV cohorts (median lag 0 s; refusals retained) - but NO honest no-fire exists, 11/60 degenerate (vacuous discharge, slow-crawl detection), and the motivating 90037 entry is NOT reproduced (trigger 531, admission 571/{403,396} vs executed 600/{409,411}). Exact 3-stream cross-match vs independent implementation; counterfactuals stop at the activation boundary. DECISION A: candidate implemented and replay-verified; prospective test STAGED ONLY - 3 matched arms x 3 owner-allocated fresh seeds x 2400 s + one 240 s mechanics = 21,840 s ceiling, frozen ITT refusals and end-of-panel criteria, NOT authorized or launched. Zero simulation; STOPs untouched.
<!-- OE55-current-end -->

<!-- OE54-current-start -->
**OE-54 completed (2026-09-13):** [Why late H5 qualifies in 90037 while early H5 fails](../output/oe54_integration_v1/REPORT.md). Zero-simulation three-arm investigation + eight record corrections (A1/A2 confirmed; A3-A8 corrected and bannered on seven OE-53 documents: the actuation-envelope bound was invalid, handback states differ, same-localization was effort-only, pre-entry quantities do differ, spatial non-detections were inconclusive, "no repair handle" withdrawn). Matched-live denominator bug in oe53_d located and repaired (sample-based rates unaffected). Three-arm comparison (frozen detector): late 90037 qualifies on BOTH primaries via 249 FEWER slowdown events (-7.0%, per-event effort unchanged); early fails effort via +12.5% non-event re-acceleration work; harm confined to [600,1500) at near-merge bins 0-5 (HDV stop-and-go) while the service gain peaks q3 via queue head-start with effort improving and bins 6-8 improve - SEPARABLE except one entangled window (q2); late = different landing pattern, not milder early; 90026 does not replicate. RSU: nothing at t=240 separates (the queue cycle 284-594 had not occurred - early entry is structurally decision-blind); the late entry state is report-visible since ~464 s (>=136 s lead); the harm is report-invisible before 600. DECISION Option B: queue-phase-gated wall entry hypothesis specified (decision, causal feature block, baselines, 2-3 fresh-seed independent-data requirement ~9,600-14,400 s, pre-declared stopping rule) - proposal only, no fitting. Zero simulation; STOPs untouched.
<!-- OE54-current-end -->

<!-- OE53-current-start -->
**OE-53 completed (2026-09-12):** [Post-handback effort mechanism and route decision](../output/oe53_integration_v1/REPORT.md). Ten-worker zero-simulation investigation. Eight OE-52 review corrections verified (A/B/C/G/H confirmed; D/F refuted-as-stated: like-for-like local congestion onsets IDENTICAL across arms, the 39-63 s spatial lead was a mislabeled post-activation latency; true leads 41.0/26.0/54.5 s). Mechanism measured: 90026 harm = more repeated slow/recovery cycles (3423 vs 2798, +22.3%); 90031 saving = softer cycles at unchanged count (intensity -14.6%); 90037 harm = harder sub-threshold acceleration (non-cycle rate +14.2%, 90.1% inside [930,1500)); all >=99.8% post-handback, HDV-dominated; early walls shift neither wave origin nor propagation; no CAV-observable feature block separates benefit from harm before any decision (RSU blind over [2000,2400)). Decision: Option 4 - the narrow early-wall route stopped (act-earlier timing, effort repair, observation selection, abort-benefit formulations unsupported); damping remains the only policy consistent with all labels; no successor experiment staged. Zero simulation. Preserved OE-51/52 reports banner-linked to [CORRECTED_INTERPRETATION.md](../output/oe53_integration_v1/CORRECTED_INTERPRETATION.md).
<!-- OE53-current-end -->

<!-- OE50-current-start -->
**OE-50 completed (2026-09-12):** [OE-49 mechanics verification repair](../output/oe50_integration_v1/REPORT.md). Canonical mechanics-evidence schema + read-only adapter + independent verifier v2 built over the preserved OE-49 evidence; corrected retrospective verification (labeled retrospective): R1-R16 VERIFIED from raw artifacts - including corrected one-to-one ownership R11 (636 records, 26 required writes, 26 matched pairs; the original C11 failure was an evaluation bug on empty-network ticks) - and R17 FAILED (equality UNASSESSABLE; consistency FAILED - c11=false with empty failures). Decision B: insufficient by exactly R17; replacement <=240 s staged with three instrumentation items. Zero simulation.
<!-- OE50-current-end -->

<!-- OE51-current-start -->
**OE-51 completed (2026-09-12):** [Early-H5 execution package](output/oe51_integration_v1/REPORT.md). Mechanics PASSED (canonical v3 17/17, R17 equality VERIFIED) on ONE live 240 s allocation; three early-H5 corridor arms (90026 t*=240, 90031 t*=246, 90037 t*=240) executed to full horizon: ALL THREE HARMFUL under unchanged criteria. 90026: system +4.27, braking -0.124, effort +4.96, TTC -0.368; 90031 (protective abort 9 s, HOLD never reached): system +12.16, braking -0.186, effort -8.64; 90037: system -12.57 PRESERVED AND IMPROVED vs H5-at-600 (-4.05) but effort +7.42 violated (label harmful). Prefix verification passed 3/3; onsets verified; corrected influence margin 156/222.5/151.5 s. Act-earlier timing route CLOSED. Zero simulation beyond authorized blocks; OE-49 STOP and 240 s preserved; oe51_early_h5 mechanics STOP preserved; corridors under successor identity oe51_early_h5_corridors.
<!-- OE51-current-end -->

<!-- OE52-current-start -->
**OE-52 completed (2026-09-12):** [OE-51 numerical/mechanism repair and acceleration-effort diagnosis](docs/oe52_integration_v1/REPORT.md if false else '../output/oe52_integration_v1/REPORT.md'). Three parallel offline branches (zero simulation). Vectors/grading: all 30 cross-check numbers exact at 6 dp; frozen grading reproduced; interval reconciliation 0.0; corrected useful_intervention for 90031 flips True->False (saved OR semantics vs embedded BOTH-references definition; corrected: delivered but not useful). Timing: all six recorded onsets verified correct under frozen definitions; occupancy onset = whole-pre-merge-edge scalar (2,644 m edge), not a band; first spatial signature upstream in the 600-800 m bin 39-63 s before region onsets; slow-vehicle counts identical in both arms at every onset (one delayed non-cohort follower differs); OE-35 local queue candidates identical (merge queue pre-existing); corrected influence-before-development margin 156/222.5/151.5 s. Effort: formula = signed measured a^2 x 0.5 s over ALL live vehicles (HDVs included); reconciliation 7.1e-9; wall episode 0.06-0.23% of each difference; 99.8%+ post-handback, diffuse over 1200+ vehicles; balanced speed-loss/recovery oscillation; 90031 saving is post-abort divergence. All OE-51 physical conclusions survive; act-earlier route remains CLOSED. Zero simulation.
<!-- OE52-current-end -->

# Open questions — decision guide

<!-- OE49-current-start -->
**OE-49 completed (2026-09-12):** [Execution of the OE-48 staged early-H5 experiment](../output/oe49_integration_v1/REPORT.md). Campaign STOPPED at the live mechanics prerequisite per the frozen integrity rule; corridors never launched; 240 s charged and completed of the 7440 s ceiling. The wrapper chain demonstrably executed end-to-end on real SUMO (runtime-causal binding at t=5, assembly, measured readiness, exactly 5 hold ticks, release at ACTUAL hold start + 5 s, coast-terminal release, damping-matched handback, zero collisions, exact clock) - but the preregistered PASS contract evaluated FALSE on two verification-tooling defects: the C11 harness invariant demanded ownership records on every tick while the fixture traffic despawns at ~163 s (write ledger shows perfect 26/26 discipline), and eight Branch-A verifier criteria failed through artifact-interface mismatches with Branch-B formats. No live retry authorized. A future authorized task must repair the C11 invariant and the A/B artifact interface, then register a new mechanics campaign; the corridor design (3x2400 s, bindings, references, frozen analysis protocol) remains staged and reusable unchanged.
<!-- OE49-current-end -->

<!-- OE48-current-start -->
**OE-48 completed (2026-09-12):** [Repair of the OE-47 staged early-H5 experiment](../output/oe48_integration_v1/REPORT.md). READY FOR A SEPARATELY AUTHORIZED LIVE MECHANICS PREREQUISITE. Defects confirmed: OE-47 staged 90026/90031 as H20 (copied H20 bindings) and 90037 with a stale t=600-era branch spec - none executable as H5 (the executed H5 policy requires a pre-registered hold start). Repairs: new versioned wrapper early_hold_offset_release/1 (release at the ACTUAL committed hold start + 5 s, mirrored verification, EarlyReleaseError stop semantics, strict schema rejecting OE-47 carriers); repaired registration (all three jobs s{seed}_h5_early at 240/246/240 s with t* cohort bindings recomputed from the causal damping-prefix replay, t=600 digest path validated 3/3 against executed bindings); budget 7200 s corridors PLUS separate 240 s mechanics prerequisite wired by name. Premises corrected: 90037 H20 harmful (effort +7.0873) vs H5 beneficial (service -4.0496); 90026/90031 H5==H20 only because neither historical run reached committed HOLD. Integrator independently traced all three registered jobs through the wrapper (release at actual ready+5, delivered hold 5, coast handback) and confirmed real bindings atomically refuse mismatched evidence. Zero simulation/fitting/acceptance access.
<!-- OE48-current-end -->


<!-- OE50-current-end -->


<!-- OE51-current-end -->

<!-- OE47-current-start -->
**OE-47 completed (2026-09-12):** [State-dependent activation timing after the OE-46 negative entry predictor](../output/oe47_integration_v1/REPORT.md). Decision A (qualified): one bounded 3-case matched timing mechanism experiment is fully specified and staged, NOT authorized or launched. Findings: no executed current-controller arm ever moved first activation off t=600 (only CUR-09 within-realization t=1200 under an older action, 0/8, and PP-05/07's emergent, cohort-leaked PPO 560/565 s); the timing-value hypothesis is CANNOT-RESOLVE from executed data; new ordering facts: disturbance develops before entry in 3/32 groups (90026/90031/90037, development 476.5-557.5s), all three with harmful fixed-entry labels, while wall influence onset is 600.5s in 32/32 (executed regime already influence-before establishment almost uniformly). Observability: earliest-causal-admission (OBS-T1) is operational with 32/32 replay integrity at t=600 and strictly-earlier feasibility in 32/32 (24/32 window-censored); t* cohorts share no members with executed cohorts (timing-and-selection); a disturbance-conditioned online trigger is measurement-blocked. Staged: cases 90026/90031/90037, one new H5-at-t* arm each (240/246/240s), reusing all damping and H5-at-600 references, 7200s ceiling, integrity-only stopping, scaffolding dry-run 32/32 PASS with 39 tests; independent reviewer re-ran the replay and re-derived timelines. Zero simulation.
<!-- OE47-current-end -->

<!-- OE46-current-start -->
**OE-46 completed (2026-09-12):** [Entry-time relational observations for fixed-H5 entry selection - one frozen exploratory comparison](../output/oe46_integration_v1/REPORT.md). NEGATIVE: under the single preregistered LOSO L2-logistic comparison (lambda=10, fixed 0.5 rule) on the 32 OE-34 mode27 entry-feasible groups, the snapshot baseline (4 OE-20 scalars) scores worse than the fold-training constant (Brier 0.2052 vs 0.1998) and activates nothing; adding the never-fitted entry_relational block (288 columns, causally verified 11/11, extracted never fitted in OE-34) worsens out-of-fold prediction further (Brier 0.2658, log loss 0.8049) and its single activation is a harmful arm (90036, TTC +0.02004 violation); 0/8 beneficial retained. Determinism byte-verified; harmful activation persists 31/31 LOO variants; one disclosed implementation correction with preserved failure receipt. This exact representation/action/model/operating-point combination is closed; no variant, no fresh-validation spend. Zero simulation/PPO/acceptance access.
<!-- OE46-current-end -->

<!-- OE45-current-start -->
**OE-45 completed (2026-09-11):** [OE-44 audit and repair; fixed-H5 entry value versus duration value](../output/oe45_integration_v1/REPORT.md). Repaired OE-44's vacuous persistence baseline (strictly-historical v2: pooled MAE 103.3m vs estimator 100.0m on identical cells - weak-measurement verdict strengthened) and corrected the negative-MAE prose (stored JSON was correct). Separated entry from duration: fixed-H5 entry hindsight opportunity bound = 8 beneficial retained / 20 harmful avoided (OE34; 5/10 on 4xxx where executed walls added 10 escalation-induced collision involvements vs damping's own 12), with harm guardrail-dominated (harmful-arm mean service -0.22 s/request); the 10-vs-8 qualifying gap is duration choice (headroom 2 and 1), not entry value. Corrected recurring claims with file-linked evidence (OE-21 small model advantage 0.2221/0.6432 vs constant 0.2249/0.6535; OE-22 zero decisions; OE-34 300 columns, gate false 5/6; both-held-out selector attribution belongs to OE-32). One untested observation cell identified (entry-time relational tensor, extracted never fitted) but blocked by zero untouched groups. Decision: owner chooses one bounded fresh 12-pair entry panel (57600s fixed-sample ceiling, integrity-only stopping, NOT authorized/executed) or closing the formulation; no duration/exit experiment proposed. Zero simulation/fitting; all decisive numbers independently recomputed by the integrating agent.
<!-- OE45-current-end -->

<!-- OE44-current-start -->
**OE-44 completed (2026-09-11):** [Observation and action-selection feasibility audit](../output/oe44_integration_v1/REPORT.md). Three parallel zero-simulation branches with pre-frozen specs. Measurement: 60s report-based arrivals are a calibrated coverage-bounded CAV count (ratio 0.194, MAE=bias -31.86/window); queue boundary weak (MAE 108.8m, -107.4m HDV-invisibility bias, 70% abstention); downstream-constrained indicator unavailable (0 positives); sustained near-merge arrival-flow change absent in 30/32 wall arms, so time-to-influence prediction is inconclusive and OE40's cumulative count lag is an accumulation effect, not a rate step. Selection value: entry qualifying mass 10/32 (OE34), 3/9 (OE07), 1/8 (OE10 dev-only); best fixed H5 captures 8/10; duration headroom 2 (90007/90054) and 1 (4125); OE42 H0 adds zero; all headroom single-seed-dependent; prior selectors at/below constants. Mechanisms: the sign-discriminating information (late storage reversal, discharge pulse value, duration cost split) is observable only AFTERWARD; formation explains delivery, not sign (7 flips); crossing pulses are rescheduled timing, not throughput (deltas return to 0 by 2400). Decision: stop the observation-to-predictor formulation on this evidence - no untouched groups, no decision-time sign observable, no action repair identified. One bounded collection proposal recorded, NOT authorized or executed: a fresh 12-pair entry panel (57600s ceiling). No duration/exit experiment proposed (hold dose and exit/recovery timing confounded per OE42). All branch headline numbers independently recomputed by the integrating agent.
<!-- OE44-current-end -->

<!-- OE42-current-start -->
**OE-42 completed (2026-09-11):** [H0 formation-plus-immediate-exit probe](../output/h0_exit_probe_v1/REPORT.md). Executed OE-41's bounded H0 proposal on its three mechanism cases: at the actual committed-HOLD admission tick (642/607/618s) the existing controlled exit replaced holding (zero delivered hold), with verified shared prefixes against the matched H20 sources and a passing 120s mechanics fixture. 7320s charged/completed of the 7320s ceiling, no retries; one disclosed post-run accounting deviation (sticky STOP preserved, registered corridor-remainder identity). All three H0 arms fail the unchanged criteria: 90021 service +4.98s/request and worse than harmful H20 in 4/5 costs; 90051 +4.20 and loses H20's service improvement; 90030 keeps -15.02 of -16.93s/request but violates braking and TTC while reused H5/H20 qualify. Some sustained holding contributes materially to every good outcome on these cases; formation+exit alone is not a new action family and damping remains the only policy consistent with all labels. Greatest remaining question: HOLD dose versus later exit/recovery timing (H0-vs-H5 as the prediction target); no successor simulation, fitting or sweep.
<!-- OE42-current-end -->

<!-- OE43-current-start -->
**OE-43 completed (2026-09-11):** [Driver-mix ablation](../output/driver_mix_v1/REPORT.md). Replacing aggressive assignments with normal did not rescue the current H20 policy: both modified cases are harmful under unchanged criteria (90030 beneficial to harmful;90045 harmful to harmful). Modified damping service time also increases by12.46 and32.12s/request; assembly increases18 to23s and16 to28s. Driver mix changes outcomes, but removing this type is not a demonstrated fix. 9600 simulated seconds completed within combined9600s ceiling; three preallocation errors preserved, zero physical retries. Two exposed diagnostic seeds; no predictor/generalization claim or automatic successor.
<!-- OE43-current-end -->

<!-- OE41-current-start -->
**OE-41 completed (2026-09-11):** [Formation-harm investigation](../output/formation_harm_v1/REPORT.md). All32 feasible OE34 formations and64 dependent policy contrasts audited. Initially compact14/15 reach HOLD versus8/17 noncompact;delivered medians18/28s. Assembly has zero excess severe-braking/TTC/collision exposure but creates local motion differences. Seven H5/H20 qualification flips share identical assembly; long formation alone does not explain harm. Targets are source-dependent (3.045–24m/s),not universally24. Two qualifying whole policies never reach HOLD;original labels preserved. Costs appearing after handback do not identify handback as cause. H0 immediate controlled exit at actual readiness is a bounded proposal only (7320s ceiling if separately executed),not launched. No new simulation/fitting or validated predictor; CAV estimator work remains open. V1 independent interval-check boundary error preserved/superseded byV2.
<!-- OE41-current-end -->

<!-- OE40-current-start -->
**OE-40 completed / CORR-44 (2026-09-11):** [Measured wall flow effects](../output/flow_effect_v1/REPORT.md). All114 existing OE34 runs reconstructed with conservation (547314 samples); 32 feasible groups,64 dependent policy contrasts. In all22 actual duration branches, the first downstream cumulative-count difference follows HOLD+5 and actual handback (H20 median76.5s after entry). This is a count-response diagnostic, not onset of all effects. Flow and accumulation change in both directions; early gains can reverse later or fail other guardrails. H20/H5 qualifying counts remain5/32 and8/32 feasible groups. Next: report-only error/coverage audit for anticipated time-to-influence and future merge state; optimal activation and benefit selection remain unvalidated. Full-state reconstruction is complete; CAV measurement validation is not. OE39 equal-final-count/online-measurability inference is superseded in part. Zero new simulation/fitting; early nine read-only extraction chronology disclosed; no successor campaign launched.
<!-- OE40-current-end -->

<!-- OE38-current-start -->
**OE-38 / CORR-42 (2026-09-11):** [OE37 source review](../output/oe37_review_v1/REPORT.md). OE37 contains superseded feature/model claims and ambiguous pooled qualification counts; its RouteD conclusion is not established by the submitted synthesis. OE34 fitted tracked histories/interactions in a300-column combined kernel representation; its negative validation remains. Complete the requested existing-data flow/queue mechanism analysis and report-only measurement/coverage audit before choosing a successor route. Zero simulation/fitting; no new observation success or physical result claimed. OE37 original preserved with review status.

OE38 follow-up / CORR43: [correction review](../output/oe37_review_v1/FOLLOWUP_REVIEW.md). Nonlinear RBF and action-specific physical-vector prediction were already fitted. Flow/queue reconstruction and CAV measurement validation remain uncompleted; RouteC is not established. No new simulation or fitting.
<!-- OE38-current-end -->

<!-- OE36-current-start -->
**OE-36 completed (2026-09-11):** [Extra regulation and second-wall test](../output/multi_wall_probe_v1/REPORT.md). Two live mechanics checks and three2400s corridors completed; three source-equivalent policies reused through exact full replay. 7440s charged/completed of14640s, no physical retries. All three new arms fail original criteria: 90030 relay saves another4.67s/request versusH20 but increases braking/effort; 90045 extension and relay worsen service and remain TTC failures. Additional regulation does not produce a new qualifying case. Sequential relay is mechanically demonstrated; simultaneous walls, relocated queues and reliable observation-based selection remain unvalidated. Use the existing queue-response evidence to examine a specific inflow/storage hypothesis before another controller variant; no training or successor simulation follows. Mode27 diagnostic sources, frozen damping, sealed acceptance and original criteria preserved. Execution-accounting repairs and negative results retained.
<!-- OE36-current-end -->

<!-- OE35-current-start -->
**OE-35 completed (2026-09-11):** [Breakdown timing and prevention audit](../output/collapse_prevention_audit_v1/REPORT.md). Zero new simulation/fitting;114existingOE34arms fully source-verified. Under the frozen retrospective diagnostic,45/60dampingcases develop extended upstream queues by1200;only2confirmedbefore600. Local slowdown does not establish entry600 is too late to prevent spread. Among32entry-feasible cases either wall policy changes incidence20to18 (3absent by1200,1induced);only90030of3absence cases qualifies. Most measured onset windows show increased discharge, so a universal capacity-drop mechanism is unsupported. Original physical criteria and negative predictor result stand; CAVwarning/benefit selection and earlier-entry efficacy remain unvalidated. No successor campaign launched; OE34formalclose-out remains separate/pending.
<!-- OE35-current-end -->

<!-- OE34-current-start -->
**OE-34 active (2026-09-11):** [Frozen observation validation study](../output/observation_relational_v1/PROTOCOL.md). Owner-directed focus on CAV placement/history and actual action-selection value. Mode27 wall fixed;36 fresh training groups and24 independent validation groups;five exposed OE32 training groups reused. Aggregate and relational models share corrected scaling/decision rules. Ceiling432000s, no retries; validation follows model freeze. No validated predictor yet; acceptance sealed, PPO/release redesign paused.
<!-- OE34-current-end -->

<!-- OE33-current-start -->
**OE-33 completed (2026-09-11):** [Reviewed harm mechanisms and observations](../output/harm_mechanism_v2/REPORT.md). Four cohort-native60 diagnostics completed, 9600 s charged/completed, no retries; 15 original OE-32 arms reused offline. All four original wall prefixes are exact, all requested trips complete, zero collisions/teleports. Early recovery assignment reduces 4160 H20 service excess from +26.375463 to +5.088213 s/request (still harmful), but destroys 4158 H20 qualification; 4158 H5 effort harm barely changes. **State-dependent recovery-controller contribution, not a universal release defect.** Later noncohort traffic carries much of the cost; a unique cut-in or complete wave ancestry is unresolved. Causal observation extraction is reviewed, but no validated selector exists. OE-32 scaling/continuation-objective repair and independent observation validation remain the next design direction; no blanket native60 or gradual-release deployment, PPO, additional simulation or acceptance access follows. All prior physical evidence and negative results remain preserved.
<!-- OE33-current-end -->

<!-- OE32-current-start -->
**OE-32 completed (2026-09-10):** [Five-part observation and duration study](../output/observation_mode27_v1/REPORT.md); [artifact map](../output/observation_mode27_v1/ARTIFACTS.md). All 22 physical allocations completed: 52800 s charged/completed of 86400 s; zero collisions/teleports, all requested service complete. Twelve independent groups yield five feasible wall decisions (three training, two held-out), plus seven refusals retained as damping. Held-out 4158 favors H20 and 4160 favors H5 under the unchanged original criteria. Entry selectors equal the constant; the full duration selector chooses the harmful action in both. **Real duration headroom; no validated predictor.** Post hoc diagnosis identifies tiny-MAD amplification of controlled-speed noise and a continuation fallback that can retain a harmful action. Next justified design work: physically meaningful scaling and a continuation objective that distinguishes harm avoidance from new benefit, followed by additional independent causally eligible decision support and frozen validation. No model repair, successor campaign, PPO, acceptance access or recovery-release redesign is launched by this result. Same-prefix future uncertainty remains untested; CAV-observation impossibility is not established.
<!-- OE32-current-end -->

<!-- OE31-current-start -->
**OE-31 completed (2026-09-10):** [Emergency-braking interface and wall reassessment](../output/emergency_braking_v1/REPORT.md). The opt-in mode27 candidate passes all three mechanics cases; six matching damping/H20/H5 corridors finish all requested service without collisions or teleports. 4127 H5 qualifies (service -10.730994 s/request); the other three walls fail original continuous-cost criteria, including 4136 H5 service +23.174982 s/request. One qualifying wall across two exposed groups supports the interface repair on this panel, not reliable escalation or a predictor. Original mode31 remains the default/selectable; mode27 is a versioned candidate. Release redesign and training remain paused; acceptance sealed. 14580s charged/completed, 12 allocations, no retries. Gate/1 STOP and its CORR-41 interpretation repair are preserved. Remaining question: explain/predict state-dependent physical harm under the fixed corrected interface; no successor campaign is launched.
<!-- OE31-current-end -->


**OE-30 (2026-09-10), current reviewed result:** [Causal harm investigation](../output/harm_causality_v1/REPORT.md). Source-matched fixtures prove a braking-interface cause for two selected H5 collisions: speedMode31 clips the cautious follower’s emergency response; identical-command27 prevents contact. Two full corridors retain the original wall and have exact prefixes/leader trajectories, yet avoid collisions with one follower kept native. Both still fail the original continuous-cost criteria. Validate the damping safety interface before choosing a release redesign or restarting predictor work. 4980s charged/completed,8allocations,no retries; acceptance sealed and recovery monitor paused.


**OE-29 (2026-09-10), current route:** [Harm mechanism diagnosis](../output/harm_mechanism_v1/REPORT.md). Recovery-release v2 and its monitor remain paused at the owner’s request. Six saved arms reconcile; the two target collision followers show sustained 3 m/s² braking under damping, unchanged immediate leaders, and different gaps across H5/H20/damping. SUMO’s ordinary-deceleration floor is a specific hypothesis to test, not a proven historical cause. Diagnose that interface before selecting a release repair; no new simulation or controller change in OE-29.


**OE-28 (2026-09-10):** [Recovery release implementation and test](../output/recovery_release_v1/REPORT.md). The single120s fixture completed safely but recovery was refused by a predicted speed/slew conflict. Version2 repairs that constraint offline; no live v2 validation. Both selected corridor cases lack one lane reference and reproduce the original H5 policy across2400 saved ticks, so4800s of redundant simulation was omitted. The campaign is stopped; training remains paused and acceptance sealed.

## OE-27 reviewed update (2026-09-10)

[Evidence repair and warning diagnosis](../output/duration_selection_v11/REPORT.md)
closes the OE-26 accounting/event-count defects and the missing 4127 verification
receipt through new offline checks. It does not close entry or duration prediction.
Reports contain near-event hazards; the warning-specific collision partner is
identified only after wall handback in the motivating case. All successful
wall-arm warnings in this fixed comparison occur after handback, with many
warnings without recorded collision. These are limits of this detector/target,
not impossibility of CAV-observable benefit prediction.

Still open: a decision-time mechanism supporting entry/HOLD choice; an effective
response to post-handback local hazards; and independent physical validation of
any proposed selector. Do not treat post-handback warning as evidence for a past
decision, average service improvement as joint qualification, or one extra
duration-oracle qualification as the headroom of the separate entry problem.
No further simulation or fitting was performed by OE-27.

Infrastructure note (MIG-01, 2026-09-10): required shared inputs have been
preserved locally; `D:/research` is no longer needed by active execution.
[Migration details](RESEARCH_ASSET_MIGRATION.md) document historical path mapping
and unchanged map/controller bytes. This resolves a storage dependency only;
none of the scientific questions below changes status.

Companion to [EXPERIMENT_HISTORY](EXPERIMENT_HISTORY.md) and the master ledger
[protocols/experiment_history.json](../protocols/experiment_history.json) (schema
experiment_history/2; entry IDs cited below resolve there — `python
tools/experiment_history.py card TH-07`). Each question states its **exact scope**,
what is established, the strongest evidence and counterexamples, where the uncertainty
comes from, the **smallest comparison** that could resolve it, the closest prior
experiment with the exact novel difference, and dependencies/stopping conditions.

Uncertainty sources are typed: `missing-execution` (never run), `confounding`
(ran, but factors bundled), `limited-seeds` (ran, too few groups), `missing-observations`
(decision-time information absent), `failed-verification` (ran, integrity/verification
defect).

**Evidence-status cutoff:** 2026-09-07 (maintenance update 2). The continuation
target-adjust evidence is incorporated as TA-01/TA-02/TA-03 with source-verified
numbers; remaining pending items are the TTC warning lead/false-alarm
quantification, the PART4_V2 raise-row re-derivation, and raw artifacts (never
opened by this ledger). The v2 report prose for the 4107 raise row is superseded
(its effort/wait signs were inverted); quote raise-arm numbers only from the
registered analysis or TTC_GROUND_TRUTH_V3.

## Framing conventions (full definitions in the ledger `conventions` block)

- **Entry frame:** proposed wall vs continued damping from the pre-decision state —
  the overall-policy benchmark. **Continuation frame:** keep / adjust / release from
  the same executed wall history (same-prefix contrasts). Damping-from-the-beginning
  is not an action available after assembly; a better continuation does not justify
  the original entry.
- **Observation ladder:** obtainable-from-reports → implemented-and-extracted →
  descriptive-of-state → predicts-feasibility → outcome-associated →
  **action-validated-on-independent-cases**. As of the cutoff nothing reaches the last
  rung (zero positive labels). Retrospective full-state diagnostics are not
  automatically CAV-acquired features; future measurements cannot be predecision
  features.
- **Owner-authorized route restrictions are policy, not physics.** They are listed in
  §R and may be changed only by the owner. Scientific findings below hold for their
  stated scope regardless of policy.

---

## OQ-1. Does any wall policy produce qualifying benefit over frozen CIRCLES damping?

> WL-02 update (2026-09-20, zero simulation): the wall-v2 pacing controller's
> literature-grounded entry gate (E1–E5), replayed on the OE-59 panel report
> streams, refuses every harmful executed admission (920003/920004) AND the
> one beneficial executed K1 entry (920002@600: upstream band 3.93 m/s,
> below the 20 mph pacing floor). K1 hold-wall and literature-safe
> pacing-wall entry envelopes are disjoint on this evidence; the
> pacing-envelope openings (post-discharge, upstream still free, e.g.
> 276–428 s) have never been physically entered by any executed arm. Open:
> does a wall entered in the wall-v2 envelope beat damping/controllers?
> Requires registered shared-prefix arms; no throughput claim is available
> from any field study. [WL-02 report](../output/wall_v2_verification_v1/REPORT.md).
>
> WL-04 first answer, harness world only (2026-09-21): envelope-gated
> wall-v2 vs no-wall, shared-prefix exact, 3 exposed m2700 seeds —
> 920002 **−30.2 s/veh (−18.2%)** and effort −18.4% (2 episodes);
> 920003/920004 gate no-ops; zero collisions. The envelope exists and can
> capture value, but this is wall-vs-nothing in a world without CIRCLES
> damping — the pipeline question (wall vs damping/controllers under the
> CV oracle) is still open. [WL-04 report](../output/wall_v2_efficacy_v1/REPORT.md).

- **Scope:** CIRCLES baseline (model SHA 762fde07…), moving-wall treatments, seeds
  4107/4108 (m2700 p20 unless noted), 2400 s horizon, frozen criteria (service ≥1.0
  s/req or braking ≥0.02 veh-s benefit; cost allowances; zero treatment collisions).
- **Established:** every *tested* wall action fails — 12 factorial cells (LC-10), 5
  lifecycle/cohort/duration traces + 1 refusal (LC-05/07/08), 2 uncapped-target arms
  (TH-07), 2 gentle arms (CUR-15), 2 dissolution arms (CUR-12), 8 natural-panel
  attempts (CUR-09), 8 same-prefix continuation actions (TA-01/TA-03:
  lower-22/raise-26 at HOLD+5), and — OE-02 (2026-09-08) — the verified 4117
  configuration causally re-bound on two fresh realizations: 4119 (complete 20 s
  HOLD; service/braking/TTC improve but effort violates its allowance) and 4120
  (4 s protective abort; four guardrails worsen) both harmful. Zero qualifying
  benefits beyond the single unreplicated 4117 cell in that chain; zero
  strict-joint successes elsewhere.
- **OE-07 update (2026-09-09, executed single-contract 12-pair study):** under the
  verified fixed wall (OE-02 profile, causal binding) on 12 exposure-audited
  development pairs (4121–4132, m2700/p20/t600), the wall **qualified in 3 of 9
  executed arms** (4125: service −31.18 with all five costs improving; 4129:
  qualifying via braking −0.25/effort −10.46/TTC −0.36 with service +0.26 within
  allowance; 4132: service −15.08, all costs improving) — all three with zero
  treatment collisions — while 6 arms were harmful (0–6 collisions, including one
  pre-HOLD exit with 6 collisions) and 3 of 12 scenarios had cohort refusals.
  Qualifying rate 0.33 on executed arms, exact 95% CI [0.075, 0.70]; ITT 3/12.
  **Scope: one scenario cell (m2700/p20/t600), exposed development seeds, one
  fixed contract — not fresh-population evidence and not a deployable selector.**
  The claim "no tested wall qualifies" is therefore **no longer complete**: the
  tested fixed entry qualifies in roughly a third of realizations of this cell,
  with large outcome variance (service −31.2 to +11.0 s/request).  **OE-10 update (OE-09 block-2 execution, 2026-09-09):** the untouched
  evaluation block (4141–3144) produced **3/3 harmful executed walls, 0
  beneficial, 1 refusal** (4142: service −20.6 yet harmful with 2/2
  collisions; 4143: +10.9 with 2 new treatment collisions), and the
  development block 1 beneficial / 6 harmful / 1 nonbeneficial / 3 refusals.
  Pooled same-contract executed arms: **4 beneficial / 19 = 0.21** —
  real but rare, highly variable, and absent in this block. Pooled evidence
  remains exposed development data, never prospective.
- **OE-08 decision-level update (2026-09-09, zero simulation):** analysis of OE-07's
  saved predictions shows the qualifying outcomes are real but **not selectable with the
  saved models**: out-of-fold, the criteria-conjunction screen (thresholds = the original
  criteria) selects nothing — sign errors on braking/effort/service hit both beneficial
  held-out walls — while always-wall catches 2/3 at base rate. Improved TTC MAE
  (direction right 3/3) did not rescue the conjunction. See
  [output/entry_pairs_v2/REPORT.md](../output/entry_pairs_v2/REPORT.md).
- **Strongest counterexample to "walls can never help":** none in the current era —
  and that is the point: the claim "no tested wall qualifies" does **not** establish
  "no wall can qualify." IDM-era counterexamples exist under a different actuator and
  baseline (T880 4405 safe-beneficial, W5-11; 2700 delay-beneficial management
  channel, SW-06/08) but do not transfer.
- **Uncertainty type:** `limited-seeds` + `missing-execution` (no qualifying design
  has ever been executed under this baseline).
- **Smallest resolving comparison:** one separately registered candidate design that
  addresses a demonstrated mechanism (merge-saturation attractors / discrete
  lane-allocation flips; post-handback harm concentration), with a bounded mechanics
  fixture first, then a two-arm same-prefix corridor comparison on the two exposed
  seeds.
- **Closest prior + exact novel difference:** TH-07 (uncapped target) is closest; the
  novel difference must be the *mechanism addressed*, not another (target, HOLD, cap)
  setting — those three axes are answered (see §R-1).
- **Dependencies/stopping:** stop if the mechanics fixture fails; stop after the
  corridor comparison regardless of outcome (record and close); do not scale seeds
  before a development qualification exists.

## OQ-2. Is the measured wall harm a controllable dose or realization-dominated?

- **Scope:** current actuator, seeds 4107/4108, entry target and HOLD duration as the
  candidate controls.
- **Established:** 4108 harm scales with ENTRY slowdown dose (vanishes at the 2 m/s
  entry rule: TH-07); 4107's harmful realization reproduced within 2 s/vehicle across
  two different entry targets (TH-07) and its H5-vs-H20 gap is a 637.5 s
  lane-acceptance cascade (TH-01) — a **plausible mechanism**, not an
  experimentally isolated cause. In the CONTINUATION frame, maintained speed has now
  been varied separately at fixed pre-branch history and scheduled end (TA-01
  lower-22, TA-03 raise-26): 4107 raise is the strongest measured service result
  (-8.436 s/req, TTC-only failure) yet harmful; 4108 raise equals keep; no dose law
  in either direction.
- **Uncertainty type:** `limited-seeds` (two settings on two groups cannot separate
  dose-control from realization-dominance) + `confounding` in the ENTRY frame (the
  entry-target change alters assembly and handback together with maintained speed);
  the continuation frame removed that confound on these two seeds only.
- **Smallest resolving comparison:** the continuation contrast already run
  (TA-01/TA-03) repeated on NEW seed groups — if and only if a candidate qualifying
  action exists to test (standalone value sweeps are answered; see section R).
- **Closest prior + exact novel difference:** TA-03; a new experiment must change
  seed groups or the lifecycle window, not the adjustment value.
- **Dependencies:** requires an OQ-1 candidate worth diagnosing.

## OQ-3. Can acquired CAV reports select a beneficial action (not just feasibility)?

- **Scope:** CAV-acquired reports available at decision time; action selection across
  seed groups; all controllers/actuators tested to date.
- **Established:** reports reach the ladder rungs implemented/descriptive/
  feasibility-predictive (18 s assembly forecast delivered exactly, LC-07; queues
  age-0 before every decision, TH-03; regional contrasts, CUR-11) and are
  descriptively outcome-associated (slowdown depth vs harm, TH-03/04). Nothing is
  action-validated: current export 65 features, **zero positive labels** (LC-11);
  IDM-era fair threshold L0' unbeaten by multivariate L1' (+0.160 [−0.014,+0.328],
  SW-10); onset AUC 0.944 ≠ benefit (delay R² 0.168, SW-10); shadow/observation
  programs failed their confirmatory gates (W5-10); fresh-seed replication of
  development prediction signals failed (DEC-02).
- **Important scope limit:** those failures invalidate the *tested feature/action
  combinations on their populations*; they do **not** prove CAV-observation
  insufficiency under every controller or action space.
- **OE-01 entry-frame sufficiency update (2026-09-08, zero simulation):** the
  LC-11 "zero positive labels" statement is entry-frame-specific — the executed
  PPO-pilot development cells DO contain positive entry labels (fixed wall
  BENEFICIAL on 4117, improving all five continuous metrics; PPO arm beneficial
  on 4115), but OE-01's inventory finds exactly **one causally clean beneficial
  entry label** (4117): the PPO arms' 4115/4117 activations (560/565 s) and
  19/24 training activations used a cohort identity frozen from the damping
  reference's t=600 snapshot, i.e. future-selected membership at a pre-600
  decision (`LEAK_pre600`, not deployable-causal). Causal pre-decision report
  histories and provenance-bound observation datasets exist for all 12 executed
  entry comparisons; with 1 positive label no scenario-grouped fit/evaluate
  split exists, so **existing evidence is insufficient to decide** whether
  observations support wall-entry benefit selection — an instrumentation gap,
  not an observation-insufficiency finding. Diagnostics: 4107-t600 vs 4115 have
  near-identical observable profiles with opposite service signs (realization
  variation is live), while effect-magnitude class (m2000 ≈0 vs m2700 large) is
  observable-supported. Smallest missing evidence staged as proposal OE-01-P
  (replicate the 4117 configuration on audited 4119–4132 seeds; ceiling 28,800 s;
  not registered for execution). See
  [output/entry_observability_v1/REPORT.md](../output/entry_observability_v1/REPORT.md).
- **OE-02 replication outcome (2026-09-08, executed under owner authorization):**
  Stage A of OE-01-P ran with causal t=600 binding and 16-check prefix matching —
  **0/2 qualifying** (4119 harmful-by-effort despite improving service/braking/TTC
  with a complete 20 s HOLD; 4120 harmful after a 4 s protective abort), so Stage B
  was never activated (9,600 s of the 28,800 s ceiling charged [CORR-18]). 4117 remains the
  single beneficial entry label: fitting/validating an observation-to-benefit model
  is still inadmissible. The causal observation dataset is collected and
  provenance-bound for future use. See
  [output/oe02_replication_v1/REPORT.md](../output/oe02_replication_v1/REPORT.md).
- **OE-03 probabilistic re-specification (2026-09-08, zero simulation):** the
  question is restated as expected-outcome/risk/coverage estimation instead of
  'every intervention must succeed'. CORR-18 corrected the OE-02 chain charge
  to 9,600 s and the complete-cell count to 8. Exploratory LOSO analysis
  (14 causal cells, 8 seed groups): the predeclared single-feature model does
  NOT beat a demand-class constant (MAE 5.64) — no continuous-effect
  prediction is supported; no model carried forward. A fixed-sample
  12-scenario entry study (4121–4132, 57,600 s ceiling, no outcome-conditioned
  stopping) is SPECIFIED but not authorized; targets, owner decisions and
  precision limits in
  [output/entry_observability_v2/PROBABILISTIC_SPEC.md](../output/entry_observability_v2/PROBABILISTIC_SPEC.md).
- **OE-04 observation contract (2026-09-08, zero simulation, literature-grounded):**
  four primary sources synthesized — none predicts intervention benefit and none
  offers a feature ablation to import (CIRCLES's frozen controller observes a
  singleton [ego speed, leader speed, bumper gap]; the VSL and merging papers
  assume detector/ideal-observer infrastructure absent here). A compact
  3-group contract (G-SPATIAL recovery/growth, G-MERGE ramp/mainline split,
  G-BURDEN proposal dose) was implemented and tested on the 14 causal cells:
  no addition beats the benchmark on leave-one-seed-group-out MAE; G-MERGE is
  coverage-limited (3/14 snapshots contain a ramp reporter at p20). The
  fixed-sample study remains the power fix. See
  [output/entry_observability_v3/REPORT.md](../output/entry_observability_v3/REPORT.md).
  **Interpretation narrowed by CORR-20:** that negative holds for *univariate*
  linear fits of individual features, each on its own complete-case subset,
  against a per-subset **observed-regime** constant (the "demand class" is
  actually the observable `all_mean_speed_mps >= 25`, not the exogenous demand
  label), for the service-difference target only — it does not establish
  joint-group failure or decision unpredictability.
- **OE-05 reuse audit + joint comparison (2026-09-08, zero simulation; INTERPRETATION
  CORRECTED 2026-09-09, CORR-21):** the prior programs were classified and **combined
  causal observation histories were never tested for wall-entry benefit** (gates/two-head
  failed on other actuators; SW-10's multivariate L1' was IDM-era; OE-03/04 were
  univariate; PPO's trained actor was deterministically identical to initialization — no
  learned selectivity, and no predictive mapping was estimated). The joint 4-feature
  ridge vs the observed-regime constant was executed and **NEGATIVE** (service MAE 8.571
  vs 6.043, 13 cells / 8 groups; effort ties; TTC worsens). **CORR-21:** the freeze's
  "identical proposal in every cell" claim was false — the panel mixes **three treatment
  identities** (8 natural_longitudinal_wall/1, 4 PP-05 lifecycle_v2, 2 OE-02
  finite_lifecycle_wall/2 cells; different braking authority, binding, runners, decision
  times) — so the comparison is retained only as a **mixed-controller exploratory
  analysis** (arithmetic verified from saved predictions) and is withdrawn as a
  one-contract test; OE-03/04 inherit the same interpretive limit.
  See [output/entry_observability_v4/REPORT.md](../output/entry_observability_v4/REPORT.md).
- **OE-06 treatment-compatibility repair (2026-09-09, zero simulation, zero fitting):**
  every cell traced to its registered configuration
  ([COMPATIBILITY_MANIFEST.json](../output/entry_observability_v5/COMPATIBILITY_MANIFEST.json));
  the same-family subset is **6 cells / 6 groups (4115–4120)** with a binding/runner
  covariate; **no seed group contains two wall actions from a matched pre-decision
  state**, so offline selection analysis is impossible in the entry frame. Limitation
  diagnosis: combination of treatment inconsistency, no per-seed action variation, ≤8
  independent groups, and p20 observation coverage (ramp reporter 3/14). Observation
  sufficiency itself remains **untested in both directions** — single-snapshot,
  mixed-action linear fits at this sample size cannot settle it. Next step: **OE-03's
  12 matched-pair entry study retained with single-contract amendments** (57,600 s,
  exposure-audited seeds, owner-owned thresholds, original episode labels preserved;
  not authorized). See
  [output/entry_observability_v5/REPORT.md](../output/entry_observability_v5/REPORT.md).
- **OE-07 held-out prediction test (2026-09-09, executed):** the first admissible
  fit/evaluate split with positive labels. Frozen 13-feature causal predictor
  (single-snapshot + trend30/60 + coverage + proposal features, ridge α=1.0) vs
  frozen regime/global constants on the held-out pairs: the predictor **does not
  beat the constants on held-out service** (MAE 9.20 vs 6.66, n = 3 eval cells —
  exploratory); development LOSO is also negative (17.33 vs 15.91). Benefits
  vary strongly and the multi-metric criteria diverge from service (4129
  qualified via braking/effort/TTC while service worsened slightly) — the named
  gaps are sample size, service-only targets, and unpredicted cohort refusals
  (25%), not a demonstration that observations are uninformative. See
  [output/entry_pairs_v1/REPORT.md](../output/entry_pairs_v1/REPORT.md).
- **OE-08 screening diagnostic (2026-09-09, retrospective, zero fitting):** applying the
  original criteria as a fixed screen to OE-07's saved held-out predictions selects
  nothing (model or constants): allowance thresholds sit far below prediction noise, so
  sign-level errors — not MAE ranking — are the binding failure. Safety outcomes
  (collisions 4/9 arms, completion 2/9) were never predicted and remain explicit
  unknowns; the successor must be a data-collection/prediction study with a
  pre-registered screen and joint-criteria targets (OE-09 proposal, not authorized).
  Also recorded: OE-07 access-sequence deviation (labels materialized before the dev
  fit; split enforced by code path only) and the per-fold-persistence gap. See
  [output/entry_pairs_v2/REPORT.md](../output/entry_pairs_v2/REPORT.md).
- **OE-09 successor protocol finalized (2026-09-09, proposal only, NOT authorized):**
  entry-pairs block 2 — 12 additional matched pairs under the unchanged verified
  contract; explicit reuse manifest (training = 11 same-contract rows now, ~17–19
  after; PP-05 cells diagnostics-only; CUR-09 excluded; refusals = rate diagnostics);
  bounded α selection {0.1, 1, 10} by development LOSO; primary measure = mean
  per-metric MAE ratio vs the regime constant (fixed a priori) + sign-error rate;
  full persistence and five-step mechanically enforced evaluation access (freeze/unlock
  receipts, mocked boundary tests); four-way conclusions predefined. CORR-23 corrects
  six OE-08 readings and fixes the seed fact: **the exposed registry (4107–132) is
  fully consumed** — every new scenario needs an owner seed-allocation decision.
  See [output/entry_pairs_v3/OE09_PROTOCOL.md](../output/entry_pairs_v3/OE09_PROTOCOL.md). **OE-10 execution result (2026-09-09):** with 16 training rows
  (α = 10.0 by dev-LOSO; monotone in regularization — hypothesis
  directionally supported, CORR-24) the frozen predictor **still lost to the
  regime constant on untouched cells (aggregate 3.92; sign errors 8/15 vs
  4/15)** and its pre-registered screen selected 1 harmful wall (0 beneficial
  existed). Predefined conclusion executed: **the single-decision-time linear
  observation/screening approach is closed at this scale**; direction shifts
  away from predictor investment. See
  [output/entry_pairs_v3/REPORT.md](../output/entry_pairs_v3/REPORT.md).
- **OE-15 repair + literature-grounded representation test (2026-09-08, zero
  simulation):** (a) CORR-28 repairs OE-14's comparison — four audited
  defects (median-imputed assembly for 4119/4120 vs registered 18/36 s;
  trailing-rear-gap substitution in 5 cells; unlabelled stale I1 readings;
  gap-change mislabelled as closing rate) plus non-nested α selection had
  manufactured the v7 aggregate "improvement"; corrected, the interaction
  group WORSENS aggregate (1.1441 → 1.1577) and service. (b) Two new frozen
  literature-grounded groups — member-measured interaction pressure (front/
  rear closing speeds, headway, closing persistence) and reporter
  spatial/temporal structure (near-queue braking persistence, lane
  differential, upstream gradient, mean acceleration) — each WORSEN the
  nested-α aggregate (1.1671 / 1.1856 vs repaired base 1.1441) and service;
  no arm beats the regime constant; the screen still selects only harmful
  4141 in every arm. **The linear snapshot-summary representation family
  (base, interaction, pressure, structure) is closed on this contract**;
  the untested remainder is per-lane ahead/behind sensing with stop-time
  memory (2011.00120), which the saved snapshots do not contain.
  See [output/entry_pairs_v8/REPORT.md](../output/entry_pairs_v8/REPORT.md).
  **Corrected by CORR-29 / OE-16 (2026-09-10):** Group B's implementation
  admitted downstream reporters (negative distance_to_merge_m — 93/468
  decision and 1082/5180 history frames, all 19 seeds) and B1 used
  reporter-selected instead of per-frame history membership; v8's per-seed
  diagnostic denominator also included the held-out group (the primary
  aggregate was correct). Re-frozen before inspection and rerun as the only
  refit: corrected C2R aggregate **1.2103** (worse than the reported 1.1856
  and than A0 1.1441; service 14.23 vs 13.53); all four B features changed
  in all 19 seeds; every per-seed diagnostic value changed while "no
  distributed gain" survives (at best 4/16 seeds beat a fold-local
  constant). **The representation-family closure stands on a correct
  implementation.** A negative corrected result is limited to this
  representation, model family, and 16-row exposed dataset; 2011.00120's
  ablation does not establish cross-lane radar or stopped-duration benefit.
  See [output/entry_pairs_v9/REPORT.md](../output/entry_pairs_v9/REPORT.md).
- **OE-17 duration-selection pilot (2026-09-10, 38,400 s):** selection
  AFTER an identical wall history (HOLD+5) using post-entry CAV reports —
  the first test of temporal (E→ER change) features on a one-decision
  contrast. The frozen selector failed its predeclared gate at n=7 (and
  prospective branches collapsed to n=1), so observation-based duration
  selection remains **unestablished**, while the underlying physical
  contrast proved large and sign-varying (H5 beneficial on 3/7 exposed
  dev seeds, one flip from harmful). Branch yield (refusals, aborts
  before hold) is the binding constraint for any future attempt. See
  [output/duration_selection_v1/REPORT.md](../output/duration_selection_v1/REPORT.md).
- **OE-18 correction (2026-09-10, zero simulation; CORR-30):** OE-17's
  chosen-policy accounting, contrast sign, constant baseline and
  eligibility gate corrected from source summaries (chosen continuation
  was 2.85 s/service worse than always-release; dev-only constant 3/7; all
  17 dev sources decision-time eligible at HOLD+5, 9 without executed H5
  arms). The physical contrast and the selector's dev-gate failure stand;
  the corrected population picture shifts the open question to branch
  yield and target choice (service sign is not qualification). See
  [output/duration_selection_v2/REPORT.md](../output/duration_selection_v2/REPORT.md).
- **OE-19 duration-axis qualification study (2026-09-10, 21,600 s):** the
  9 decision-time-eligible sources received H5 arms (all verified);
  combined 16-pair picture: release-only 2 / continuation-only 1 / both 3 /
  neither 10; always-release 5/16 vs oracle 6/16. Duration choices are
  qualification-relevant in both directions, but the observation question
  (can features predict the labels?) remains untested by design. See
  [output/duration_selection_v3/REPORT_OE19.md](../output/duration_selection_v3/REPORT_OE19.md).
- **OE-20 entry-selection assessment (2026-09-10, zero simulation):** the
  H5-target question is NEW (prior fits targeted H20 outcomes); the
  preregistered selector never activates and loses to the training-only
  constant on 18 labeled pairs — negative result, fresh validation not
  justified. See
  [output/duration_selection_v4/REPORT.md](../output/duration_selection_v4/REPORT.md).
- **OE-21 correction (2026-09-10, zero simulation; CORR-32):** OE-20's
  held-out constant corrected to fold-training-only frequencies — the
  model shows a small unvalidated advantage over the constant rather than
  a deficit; all-decline and the untested temporal representation stand.
  See [output/duration_selection_v5/REPORT.md](../output/duration_selection_v5/REPORT.md).
- **OE-22 temporal-block comparison (2026-09-10, zero simulation):**
  adding the existing trend30/60 approach-speed means improves scoring
  descriptively but produces zero decision changes at the frozen 0.5
  rule; no selection value demonstrated. See
  [output/duration_selection_v6/REPORT_OE22.md](../output/duration_selection_v6/REPORT_OE22.md).
- **OE-23 diagnosis (2026-09-10, zero simulation; CORR-34):** the harmful
  H5 cases are mechanically heterogeneous (collision-cluster,
  effort/braking-despite-service-gain, marginal service excess,
  pre-branch inherited) and the entry-predictor formulation has one
  qualification case of headroom over always-release — stop this
  formulation; a read-only collision-event extraction is the only
  zero-cost next diagnostic. See
  [output/duration_selection_v7/REPORT_OE23.md](../output/duration_selection_v7/REPORT_OE23.md).
- **OE-24 collision-event diagnosis (2026-09-10, zero simulation;
  CORR-35):** OE-23's policy accounting corrected (refusals in absolute
  totals, cancel in differences; common-25 never 5118.84 vs always-H5
  5011.33; 4119 UNKNOWN; full-26 activated totals INCOMPLETE). All five
  H5 collision cases reconcile to intervals.jsonl vehicle-involvements;
  4124's event identical in the H20 arm; 4142's first event pre-entry
  and shared by all arms; H5-only post-release events in 4/5 cases
  (nearest 51 s after release), causally unresolved. Recommendation C:
  further re-instrumented evidence needed. See
  [output/duration_selection_v8/REPORT_OE24.md](../output/duration_selection_v8/REPORT_OE24.md).
- **OE-23 diagnosis (2026-09-10, zero simulation; CORR-34)**: policy
  populations repaired (18 labeled / 25 common / 4119 UNKNOWN / 26
  incomplete); the 12 harmful H5 cases decompose into at least four
  mechanisms (collision-cluster 5 with H5 collisions 2-6 vs damping 0-2;
  effort/braking violations despite service gains 2; marginal service
  excess 2; pre-branch inherited 2); the four highest temporal scores are
  all harmful. Recommendation: stop the entry-predictor formulation; the
  only zero-cost next diagnostic is a read-only collision-event
  extraction. See
  [output/duration_selection_v7/REPORT_OE23.md](../output/duration_selection_v7/REPORT_OE23.md).
- **OE-20 entry-selection assessment (2026-09-10, zero simulation):** the
  H5-target question is NEW (prior fits targeted H20 outcomes); the
  preregistered entry-feature selector (4 features, fixed lambda=10,
  threshold 0.5) never activates and loses to the training-only constant
  on 18 labeled pairs (5 qualifying) — negative result, fresh validation
  not justified. See
  [output/duration_selection_v4/REPORT.md](../output/duration_selection_v4/REPORT.md).
- **OE-21 correction (2026-09-10, zero simulation; CORR-32):** OE-20's
  held-out constant corrected to fold-training-only frequencies — the
  model shows a small unvalidated advantage over the constant rather than
  a deficit; all-decline (no escalation benefit) and the untested temporal
  representation stand. See
  [output/duration_selection_v5/REPORT.md](../output/duration_selection_v5/REPORT.md).
- **OE-38 OE-37 correction (2026-09-11, zero simulation; CORR-42):** OE-34
  fitted 300 features (12 aggregate + 288 relational across six tracked
  reporter slots, three lags, sixteen fields including tracked speed slope,
  accelerations, gaps and relative speeds). OE-21 showed a small numerical
  advantage. H5 qualifying is 8/60 fresh cases. Route D withdrawn; Route C
  (state measurable but little useful action improvement) is the corrected
  route. Physical flow/queue reconstruction remains to be done. See
  [output/duration_selection_v10/CORRECTED_ASSESSMENT.md](../output/duration_selection_v10/CORRECTED_ASSESSMENT.md).
- **Uncertainty type (updated by OE-07):** positive labels now exist (3 qualifying
  cells in the single-contract 12-pair study), so the binding uncertainty is
  `limited-seeds` (6 development / 3 held-out outcome cells; CI [0.075, 0.70] on θ_Q)
  plus untested temporal/joint-target information — not `missing-observations`
  (still unproven in both directions) and no longer plain `missing-execution`.
- **Smallest resolving comparison (updated by OE-07):** more single-contract matched
  pairs on exposure-audited seeds (the OE-07 design repeated, budget-gated, owner-
  authorized) to tighten θ_Q and give the pre-frozen rule-vs-model comparison —
  which must target the **joint criteria vector** (per-guardrail outcomes), not
  service alone — real held-out support. Any fresh-prospective claim still requires
  an explicit audited registration; exposed development seeds cannot become
  acceptance data.
- **Closest prior + exact novel difference:** SW-10 Part C (L0'/L1') — novel
  difference: positive examples exist and the action space is the current actuator.
- **Dependencies:** strictly downstream of OQ-1. Fitting on the current 0-positive
  export is inadmissible and would answer nothing.

## OQ-4. Does the assembly/HOLD/exit decomposition hide a repairable component?

- **Scope:** current actuator phases: assembly (12/18/7 s observed), HOLD (0–40 s
  nominal), taper/exit, handback.
- **Established:** HOLD length tested by same-prefix continuation (LC-08: keep-20 vs
  release-5; IDM-era H300/H450/H650 curves, SW-10). Exit/release laws tested
  (CUR-12, CUR-15). Assembly-only vs full wall tested once (CUR-09 + IDM T880 sham/
  assembly-only diagnosis, W5-11): assembly not dominant; harms concentrate
  post-handback (LC-09: C4108 late-window service share 63.8%; CUR-15: all 161.5
  gentle-4108 TTC vehicle-seconds post-handback). **Not isolated:** the current
  12/18 s assembly from the current handback — they are bundled in every executed
  4108 arm (TH-02).
- **Uncertainty type:** `confounding` (bundled phases) + `limited-seeds`.
- **Smallest resolving comparison:** a registered same-prefix arm varying one phase
  (assembly dose or handback timing) with all else fixed — as part of an OQ-1
  candidate, since phase isolation alone cannot qualify anything.
- **Closest prior + exact novel difference:** LC-08 machinery; novel difference:
  vary assembly or handback instead of HOLD.
- **Evidence-kind note:** window decompositions (LC-09, TH-01/02) are temporal
  localization, not experimental isolation; only sham/assembly-only/same-prefix arms
  isolate causes.

**OE-11 discharge-phase evidence (2026-09-09, zero simulation):** 4141's
+5.065 s/req harm is exactly zero through handback (648 s), slightly negative
to 1200 s (the wall arm discharged *faster*), then +7,925 veh-s in 1200–1800
s — 94% pre-merge, 97% on HDVs, both arms carrying the same standing queue
from t=0. This adds a measured instance of **discharge-phase rescheduling**
(the OQ-4 isolating factor) and a proposed phase-referenced activation
contrast (reusing exposed seeds and existing damping references, ~7,200–
9,600 s, proposal only). See
[output/entry_pairs_v4/DIAGNOSIS_4141.md](../output/entry_pairs_v4/DIAGNOSIS_4141.md).
**OE-12 trigger replay + OE-13 integrity repair (2026-09-09, zero
simulation):** the rising-approach-speed trigger, frozen before replay, was
replayed on all 12 block-2 damping streams — 8/12 fire within 2 s of
t=600, the only 2 later triggers lose cohort feasibility, 0 new feasible
proposals → **rule not ready**; blocker = near-queue CAV coverage
instrumentation (2–5 reporters on a ~94% HDV queue). OE-13's integrity
repair found the replay's current-bin boundary (now = t+1) was a real
tick-level defect (400/14,412 predicate differences under the repaired
strictly-causal window, which adds the last pre-decision second) but
**latent at the result level** (triggers/feasibility identical on the
corrected replay). CORR-26/27 correct the accounting (+7,075.5 =
+6,980.0 sampled presence, approximate + +64.5 measured source-wait +
+31.0 unexplained, 0.44%). See
[output/entry_pairs_v6/REPORT.md](../output/entry_pairs_v6/REPORT.md).

**OE-14 interaction-feature closure (2026-09-09, zero simulation; one bounded
offline comparison):** the neighbor-interaction hypothesis (front gap now /
min-history gap / front closing rate / trailing rear gap, frozen before any
association) was tested against the 13-feature aggregate set under the identical
frozen procedure on the 16 training rows: primary aggregate 1.131 -> 1.108 (both
worse than the regime constant; inside the noise band), service MAE WORSENED
with the added group (dev 12.43 -> 14.51; exposed 10.63 -> 12.99), violation
recognition 7/15 -> 9/15, screen still selects only 4141, and the I-features
completely overlap beneficial vs harmful classes. **This specific feature
hypothesis is closed**; general observation sufficiency remains unresolved. See
[output/entry_pairs_v7/REPORT.md](../output/entry_pairs_v7/REPORT.md).
**Corrected by CORR-28 / OE-15 (2026-09-08):** the aggregate "improvement"
1.131 -> 1.108 was an artifact of four defects (imputed assembly,
trailing-rear substitution in 5 cells, unlabelled stale I1, gap-change
mislabelling) and non-nested α selection; repaired, the group worsens the
aggregate (1.1441 -> 1.1577) and service; the service-worsening and screen
findings above stand. See
[output/entry_pairs_v8/REPORT.md](../output/entry_pairs_v8/REPORT.md).


## OQ-5. Do damping and wall interact differently at other penetrations/demands?

- **Scope:** current era: wall tested only at p20/m2000+m2700; damping characterized
  at those cells (CUR-04) plus adoption-era direct p20 (CUR-01).
- **Established:** nothing about p≠20 walls. Owner clarification: a wall may help at
  any traffic state; density is not an entry requirement.
- **Uncertainty type:** `missing-execution`.
- **Smallest resolving comparison:** an extended characterization + opportunity panel
  at p5–p30 (registered separately) — this **diagnoses the counterfactual** and
  captures candidates; it cannot create wall benefit by itself and does not require
  any beneficial controller to run. Distinguish it clearly from replication
  (OQ-7) and predictor-scale generation (OQ-3).
- **Closest prior + exact novel difference:** CUR-04; novel difference: penetrations
  and demand cells beyond the two development cells, with measured (not nominal)
  density recorded.

## OQ-6. Map length and recruitment geometry

- **Established:** the original map suffices; no evaluation truncation (LC-01);
  failures are downstream-traffic/attractor phenomena; a v5.7 +900 m extension design
  exists (frozen, self-flagged confound, never executed — W5-16).
- **Uncertainty type:** none currently binding (no demonstrated map-length
  limitation). A separately versioned local extension compared damping-vs-wall on the
  identical extended scenario remains an available owner-sanctioned repair **if** a
  demonstrated limitation appears; an extension does not itself establish benefit.

## OQ-7. Replication and prospective evidence

- **Established:** the 6-seed/2-setting replication panel was proposed and NOT
  activated (LC-11); Part F fitting not activated; Part G prospective not activated.
  Sealed and unopened: 4601–4660, 5001–5060, 912001–912032, current-era acceptance
  partitions. Seeds 201–210 are consumed as a locked split forever (CFG-05).
- **Uncertainty type:** `missing-execution` by design (gated).
- **Smallest resolving comparison / ordering:** replication and prospective
  evaluation are downstream of OQ-1/OQ-3 candidates; the ordered test plan
  ([ESCALATION_TESTS](../archive/circles_era/docs/ESCALATION_TESTS.md)) defines the sequence and gates.
- **Merely repeats:** opening sealed blocks to "check"; reusing 4107/4108 as if
  fresh.

## OQ-8. Continuation actions from an executed wall history

- **Established:** keep-20 vs release-at-5 (LC-08, same-prefix); lower-22 and
  raise-26 at HOLD+5 with unchanged scheduled end (TA-01/TA-03, source-verified).
  Eight delivered same-prefix continuation actions across two seed groups: eight
  harmful, zero qualifying, zero strict-joint successes. 4107 raise-26 improves
  service, braking, effort and wait versus damping and beats keep-24 by 4.282 s/req,
  failing only TTC; 4108 raise equals keep. No dose law in either direction; the v2
  upward-arm prose signs are superseded (ledger CORR-6).
- **Standing rule:** record entry and continuation comparisons separately; a better
  continuation never retroactively justifies the entry decision, and
  damping-from-the-beginning remains the overall-policy benchmark, not an action
  available after assembly.
- **Remaining pending:** TTC warning lead/false-alarm quantification; PART4_V2
  raise-row re-derivation (PU-1).

---

## §R. Route restrictions (owner-authorized policy — distinct from scientific findings)

These are current policy ([ROUTE](ROUTE.md)); they are stricter than the science
requires and only the owner changes them. Scientific scope statements follow each.

- **R-1 (policy):** no further speed/duration/braking search. *(Scientific scope: the
  completed comparisons are answered — do not repeat the same cell; the 24 m/s-vs-
  uncapped comparison, H0/H20/H40 and −0.3/−3.0 are done. A mechanism-isolating
  comparison that varies one phase with assembly/handback fixed, or any new
  controller/observation contract, would be a different experiment and is not
  excluded by the data.)*
- **R-2 (policy):** no replication, predictor-scale generation, training, or
  fresh-seed campaigns now. *(Scientific scope: replication is meaningful only after
  a development qualification exists.)*
- **R-3 (policy):** damping training paused; external CIRCLES controller frozen.
  *(Scientific scope: the bespoke-learning NO-GOs cover the tested interfaces; they
  are not a theorem about all damping learning.)*
- **R-4 (policy):** acceptance data sealed; wall/predictor generation gated.
- **R-5 (policy):** bounded diagnosis is permitted without a pre-existing beneficial
  controller — characterization panels, observer/replay diagnostics, offline
  localization and mechanics fixtures are the sanctioned way to develop a candidate
  (this guide's OQ-2/OQ-4/OQ-5 paths). Diagnosis ≠ replication ≠ predictor-scale
  generation ≠ prospective acceptance; each needs its own registration.

## What would merely repeat an answered experiment (quick test)

A proposed run repeats an answered question if, after substituting its design into
the ledger (`python tools/experiment_history.py closest --text "..."`), the closest
prior entry differs only in parameter values on an axis already swept (target depth,
HOLD length, braking cap, cohort rule, release law of the same family) under the same
controller/observation contract and the same two seeds. It is genuinely new if it
changes the mechanism addressed, the seed groups, the observation contract, the
action space, or the decision frame (entry vs continuation) — and it must say which.

*Any new experiment: register first with
[EXPERIMENT_RECORD_TEMPLATE](EXPERIMENT_RECORD_TEMPLATE.md); append outcomes and stop
decisions to the ledger on close; update this file as part of closing.*

- TTC observation audit v4 (output/ttc_diagnosis_v1/REPORT_TTC_V4.md): the
  raise-26 excess (+56 veh-s, 4107) is entirely post-release lane-0 following
  chains (703/707 allocation family). No fixed-definition follower-front
  warning leads the assessed events (max lead 0-3.5 s fleet-wide; specific
  events <= 0 s). Open: does the multi-report fusion candidate (reported
  deceleration ahead + follower approach) give repeatable lead across arms and
  seeds without unacceptable non-detection rates? v5 first pass (zero
  simulation): yes on lead (1-5 s, earlier than follower-front on 15/16 paired
  events) but recall is penetration-limited (335/518 4107-raise events
  unassessable); no threshold refinement is authorized Does reported downstream
  queue pressure modify the raise-vs-keep action effect (two-seed suggestion
  only)? Both answerable only with more independent seeds; nothing exhausted.

- Joint reassessment (output/joint_design_v1/DECISION.md): does ONE registered
  continue/release decision at HOLD+20 (delivered 30 s wall, keep-24, exposed
  seeds) change outcomes versus keep-24, and does the direction depend on the
  reported downstream state? Proposal staged in NEXT_EXPERIMENT_PROPOSAL.md
  (5040 s, not executed, not authorized). Untested beyond it: post-release
  reassessment designs; any second extension; target changes at other ticks.

- Selection potential (output/joint_design_v1/SELECTION_AND_SENSITIVITY_V1.md):
  oracle beats every fixed continuation action, but no action qualifies in
  either seed; the raise-vs-keep TTC sign reverses across seeds. Open: does a
  +10 s extension of the raise-26 HOLD repair its TTC-only failure
  (NEXT_BLOCK_PROPOSAL_V2, not executed), and is that duration effect
  state-dependent? Untested: repeated matched futures; any post-release
  design; G1-G5 predictor gates all unmet.

- HOLD-extension experiment executed (output/hold_extend_v1/REPORT_V3.md;
  ledger HE-01): the +10 s extension of the raise-26 history does not repair
  the tradeoff — 4107's extended wall lost compactness +6 s into the
  extension (protective abort at 26 s delivered; vs damping it improves
  service/braking/effort/waiting but TTC +0.020141 fails by 0.000141 and 2
  collision involvements fail; harmful), and 4108's 30 s HOLD raises five
  continuous costs vs damping with four guardrail violations (harmful). Vs
  the same-prefix raise-26 H20 arm: 4107 improves service/braking/TTC
  (−27.5 veh-s)/waiting but costs effort (+1.53 m²/s³ per request) and 2
  collisions (0→2); 4108 shows small measured differences. Open and
  unexhausted: other duration histories/adjustments are untested but no
  evidence currently motivates them; post-release reassessment remains a
  separate undesigned decision point; G1-G5 predictor gates all unmet.

- Hold-extension repair pass (output/hold_extend_v1/REPORT_V2.md,
  superseded in part by REPORT_V3.md): corrected signs (4107-ext improves
  service/braking/effort/wait vs damping; TTC +0.020141 fails by only
  0.000141; 2 collisions fail); the v1 admission evaluated the original H20
  plan's final tick (vacuous for H30) and hold_extend/2 now assesses the
  extended commitment before commit. Open: the repaired admission
  (hold_extend/2) has saved-branch replay support but no in-situ physical
  validation; whether any hold duration can retain the TTC improvement
  without losing compactness or adding collisions is untested.

- Comparator repair + budget-launch integration (hold_extend_patch_v1,
  output/hold_extend_patch_v1/, ledger HE-03/CORR-12; zero simulation): the
  corrected same-prefix comparison is machine-generated (COMPARISONS_V3.json)
  and machine-checked against the registered wall_vs_wall blocks; the earlier
  −92 veh-s "same-prefix TTC" figure was actually vs keep-24 (raise-H20 gives
  −27.5 veh-s). "Closest any arm has come" and "first continuation action to
  improve TTC" are withdrawn (no collision-aware ranking defined; the
  raise-26 adjustment itself improved TTC vs keep-24 by −103 veh-s). The
  campaign budget is now enforced at both real launch boundaries
  (mechanics start_sumo; paired evaluate_arm) against the seeded ledger
  (4920 s authorized / 6360 s charged / 1440 s overrun recorded as a
  deviation); every ceiling is exhausted, so campaign hold_extend blocks all
  further reservation. Smallest unresolved verification before any future
  hold_extend/2 use: one bounded 120 s mechanics fixture under the repaired
  code (requires simulation; not executed).

- PPO pilot (output/ppo_pilot_v1/REPORT.md, ledger PP-01): can a small PPO policy
  learn useful wall activation and continuation decisions from acquired RSU
  evidence? UNANSWERED - implementation-blocked at the mechanics gate. The
  interface (wall_ppo/1 contract, 29-feature observation, scaled physical reward
  with bounded intervention cost, masked actor-critic, launch-path budget
  enforcement) is complete and zero-simulation verified; both 120 s mechanics
  allocations failed on bare traci.simulationStep() action-boundary jumps
  (+1.5 s observed), exhausting the 240 s mechanics budget; the repair
  (absolute-target stepping) is implemented but unverified in-situ. Open: does
  the repaired fixture pass; then do references/training/evaluation under the
  registered remaining budgets. NOTE: `ppo_contract.py` also still names a
  narrower unresolved design question - whether 29 features + 4-tick history is
  the right minimum observation, which no evidence yet answers.

- PPO repair (output/ppo_repair_v1/, ledger PP-02): the clock discrepancy is
  now characterized with facts separated from hypotheses - the mechanism
  inside SUMO remains UNRESOLVED offline and requires the instrumented
  mechanics allocation proposed in MECHANICS_PROPOSAL.md (not executed).
  Delayed-credit delivery through the real trainer is now exact at all
  delays (Monte Carlo return with value baseline; the GAE-lambda trace
  attenuated +450 s consequences to ~1%); budget enforcement is sticky at
  campaign level and reclassification cannot restore capacity. Open: the
  one-allocation mechanics verification; then the bounded pilot's
  references/training/evaluation under the registered remaining budgets.

- PPO fixture repair (output/ppo_fixture_repair_v1/, ledger PP-03): the
  mechanics fixture's time/count contract is now derived and asserted (240
  physics samples, 119 control ticks, 120 reward rows, phase-dependent
  rollout rows cross-checked against the decision log), and the
  mocked-boundary full-runner suite passes a complete chain trace, fails
  injected clock anomalies with receipts, and blocks budget-rejected
  launches. The mocked harness also caught a live-run blocker (empty
  damping references would have refused the wall at assembly start) - now
  fixed. Open: the single instrumented mechanics allocation under the
  corrected proposal (not authorized, not executed).

- PPO mechanics verification PASSED (runs/active/ppo_mech_v1, ledger PP-04):
  the repaired fixture completed 120 s with exact grid accounting, the full
  masked-refusal → ACTIVATE → CONTINUE (H20→H30) → RELEASE → handback chain,
  zero collisions/teleports and intervention cost within its bound. The
  historical clock anomaly did not recur (cause still unresolved; a single
  clean run cannot exclude a rare mechanism). Open: the bounded PPO pilot's
  references (12×2400 s), training (32×2400 s) and development evaluation
  (12×2400 s) under their registered budgets - all requiring separate
  authorization; efficacy, observation usefulness and PPO learning remain
  entirely unproven.

- PPO evidence audit (output/ppo_audit_v1/, ledger PP-06): PP-05's dev
  labels corrected — the fixed wall was BENEFICIAL on 4117 (where the PPO
  policy was harmful with 2 collisions) and HARMFUL on 4115; the PPO
  4115 benefit verified. The actor converged to always-activate-when-
  feasible (P(ACTIVATE) 0.52 -> 0.56-0.64 on genuine-choice rows) but 98.4%
  of decision rows were feasibility-determined — no evidence of learned
  selectivity. On-policy integrity verified exactly across all updates.
  The executed sampler ignored the declared episode seeds (global RNG);
  corrected as SamplingActor v2, unverified in-situ. Open: why the policy
  releases ~14 s into a committed H40 (early-release mechanism unat
tributed); whether fresh-seed replication of the 4115-class result
  reproduces.

- PPO timing/decision replay (output/ppo_timing_v1/, ledger PP-07): the
  PP-06 'activated 14 s later' claim was reversed — the PPO actor activated
  40/35 s EARLIER than the fixed wall on 4115/4117, committed H40 via two
  continuations, and released early (delivered 14/11 s). Offline replay
  shows ZERO deterministic decision changes between the initialized and
  trained actor on all 6,716 recorded training states; the trained changes
  are probability-mass only. Open: what traffic mechanism made early
  activation beneficial on 4115 but harmful (2 collisions) on 4117 — and
  whether the frozen 'activate early, release early' controller replicates
  on fresh seeds.

- Duration semantics + candidates (output/ppo_duration_v2/, ledger PP-08):
  the executed pilot's hold releases were mask-FORCED (H40 committed ->
  only RELEASE exposed at 14/11 s delivered), so no duration preference was
  ever learned or chosen. wall_ppo/2 now separates MAINTAIN (interior) from
  EXTEND (scheduled end, admitted on the proposed maneuver). Separately,
  replaying the damping references' report streams shows the fixed t=600
  cohort was feasible at only 1/28 measured ticks per scenario while
  bounded current-report search found feasible cohorts at 13-22/28 —
  the fixed binding materially restricted opportunities (availability
  under damping trajectories is not a wall outcome). Open: in-situ
  wall_ppo/2 mechanics pass; whether selector integration changes physical
  outcomes.

- Candidate-opportunity correction (output/ppo_candidate_audit_v2/, ledger
  PP-09): PP-08's audit compared the fixed cohort and the candidate search
  under DIFFERENT readiness conventions; the corrected shared-path
  comparison re-verifies the opportunity gain (search-feasible 13-22/29
  samples vs 1/29 fixed-cohort; 4109/4110 feasible despite no t=600
  binding). Open: in-situ wall_ppo/2 mechanics pass (ready-to-register, not
  authorized); then selector integration and any future learning pilot.

- Duration mechanics PASS + selector implemented (runs/active/ppo_mech_v2/,
  output/ppo_selector_v1/, ledgers PP-10/PP-11): the ONE authorized 120 s
  allocation verified wall_ppo/2 duration semantics in-situ (12/12 checks
  incl. admitted H30 EXTEND at the deadline, exact 0.5 s clock, scheduled
  release, clean handback). The wall_ppo/3 causal current-report selector
  (selector causal_selector/1) was implemented behind the unchanged
  wall_ppo/2 hold semantics. CORRECTION (CORR-19): PP-11's "fully
  mocked-verified" claim covered a WAIT-only empty-world path only — the
  activation chain was never exercised and four integration defects existed
  (phase-blind driver that would RELEASE at the first HOLD decision; no
  offer binding so candidate observations were zero while inactive;
  execution re-ranking that could silently substitute a cohort; search
  ignored its profile parameter and hardcoded lanes). Repaired in PP-12
  (selector causal_selector/2, output/ppo_selector_repair_v1/): offer bound
  before observations, execution commits exactly the approved offer with
  atomic refusal, named phase-aware comparator, profile/lanes/eligibility
  consistency; verified on a POPULATED mocked environment through the
  complete select→activate→assemble→H20→EXTEND→H30→RELEASE→handback chain;
  876 tests + Ruff pass. Resolved: the PP-08/PP-09 open "in-situ wall_ppo/2
  mechanics pass" question, and (PP-13) the live selector mechanics
  question for the populated fixture: the ONE authorized 120 s live
  allocation (runs/active/ppo_mech_v3) passed 16/16 checks — live 4-pair
  search with an offer bound before the window, exact offer-to-admission
  binding (fingerprint equality), member fixation, H30 admitted at the H20
  deadline and fully delivered, clean handback, live search max 10.4 ms
  (fixture scale only). Open: whether selection changes physical outcomes
  (prospective benefit/risk evaluation design: OE-03's probabilistic
  protocol; the missing action-outcome evidence stands); corridor-scale
  search cost and population effects; any future learning pilot (requires
  its own authorization — none is implied by the mechanics PASS).

<!-- OE34-traffic-question -->
OE-34 supplementary [traffic-state review](../output/observation_relational_v1/review/COLLAPSE_QUESTION_V1.md): saved damping histories show low merge speeds before entry600; collapse incidence/onset and prevention efficacy remain unmeasured. Zero new simulation; full study close-out remains in progress.


<!-- MIG02-status -->
**MIG-02 (2026-09-17):** [Source-isolation verification](../output/research_removal_v1/REPORT.md) passes: 568 local asset hashes, 1020 tests, exact saved replay and archive path/size checks. No further dependencies needed copying. Automatic approval review blocked deletion; all D:/research contents restored, old folder NOT deleted. Scientific route unchanged.
<!-- WALL-C1-PREP-20261009 -->
**Unseen group controller C1:** can an LPWO-trained structured model predict HDV
response to coordinated walls at 10/30/60/120 s and rank matched wall/native deltas?
Engineering preparation is synthetic-only; corrected splits, engaged set sizes,
source guards, retrained comparators and scores remain pending. Existing C1 models
trained with wall windows cannot answer this question. See
[`WALL_C1_READY.md`](../_out/WALL_C1_READY.md). No scientific conclusion changed.

<!-- WD01-PA2G-A2-flags -->
**WD-01-PA2G-A2 (2026-09-30; supersedes WD-01-PA2G-A):** Part A v2 general predictor, fixed engine, on chunk-4 `__native_base` (fleet-native) TRAIN truth, 32 worlds, zero simulation ([report](../../rd-wt/wd01/output/wd01/REPORT_PA2G_EVAL_A2.md), ledger WD-01-PA2G-A2). Open owner flags: (a) accept or decline retiring the nominal stability guards to OFF; (b) discretionary lane-change over-production (4.6-6x) and merge under-production: characterize as model error, or register a bounded investigation (no tuning); (c) tracked-scope long-horizon delay bias (untracked arrivals dominate beyond H=60): a Part-B/horizon question. WD-01-PA2G-A's headline numbers are superseded (switching-mix truth + the `_neighbors` swap bug fixed in 615ac1f).

<!-- WM72-relabel -->
**Campaign arm relabel (WM-72, 2026-09-29):** in chunks 1/2/3 and fix27 the restored `__native` and `__pi` arms are random per-CAV switching mixes (the branch driver special-cased only `fixed`), and `dispatcher_rows.jsonl` is not an action source. Any "vs native" / "PI vs native" number from those arms is mislabelled. Chunk 4's branches run the fixed driver (native = fleet native, pi = fleet PI). Re-running native/PI for chunks 2/3 awaits an owner decision.

<!-- WD01-owner-decisions-20260930 -->
**Owner decisions on WD-01 Part A/B (2026-09-30, OWNER_DECISIONS_WD01.md 257ac6c):** (a) the LC stability guards are RETIRED; params_nominal (sha 079a5576...) is FIXED. (b) The residual LC mismatch is under attribution in WD-01-PA2G-LC1 (in progress; gates the C4L1 chunk-4 CAL look). (c) The long-horizon delay bias is a tracked-scope effect; Part B carries untracked arrivals. (d) Part B v0.4 trains on chunk-4 native_base + the wall-pilot branches + the corrected true native/PI arms.

<!-- PLEXP1b-abs-ttc-floor -->
**PL-EXP1b (2026-09-30): absolute min-TTC floor scope (owner decision #7).** The absolute bar (min TTC >= 1.5 s over the window, true-gap) fails for the NATIVE reference itself in all 10 TRAIN worlds (native min TTC 1.02-1.47 s), so the primary owner rule was non-informative (0/10). Options for BOTH the analysis rule and the in-loop rule: A keep as is / B apply only to CAV-involved pairs / C relative-to-native (the planner recommends C). Descriptive oracle gains in 3/10 worlds (delay -1.9 to -5.1 %) meanwhile. Ledger PL-EXP1b. Sensitivity addendum (zero-sim): pass rates A 0/10, B 4/10, C 8/10; under C the passing gains are w8581 (-2.0 %) and w8541 (-5.1 %); 80/107 in-loop vetoed candidates were HDV-HDV pairs.

<!-- PARTB-v03-hardlinks -->
**Part B v0.3 root hard links (2026-09-30, flag; owner decision):** 1432 files of `output/wd01_partB_v0data_v03` still have link count > 1 (hard-linked to another root, probably v0.3.1). An in-place write to that other root could silently alter v0.3. The v0.3.2 root has been de-linked and verifies 2149/2149. Whether to de-link v0.3 / v0.3.1 (copy + rename, hash-verified) is the owner's call. Ledger WD-01-PARTB-V032.

<!-- WD01-LC1-closed -->
**WD-01-PA2G-LC1 (complete, 2026-09-30):** no bug; one counting fix (tracked cohort, ~1.5-2 % of events; counters agree 2240/2240); the residual 4.6-6x discretionary LC over-production is model structure (2.32x floor at matched states) plus ~60 % divergence amplification; the merge lag is structural (dwell 34.5 s vs 12 s; 0 vs 48 dead-end stops). Owner options go to main (C4L1 gate).

<!-- WM73-truearms -->
**WM-73 / J3 (complete 2026-09-30):** the corrected chunk-2/3 restored arms exist: `runs/campaign_stage1_chunk{2,3}_truearms/<world>__native` (fleet native) and `__pi` (fleet PI, U=31.29), 140 worlds, same saved states as the recorded damping `__base` (CRN). Analyses that used the recorded switching-mix `__native`/`__pi` (PL-02a C1/C2/C3/F27, WD-01 Part B up to v0.3.2) can now be re-run against these true references (owner decision; no re-run started).

<!-- OWNER-20260930-0940 -->
**Owner decisions ~09:40 2026-09-30 (via main; the rented box is terminated, the bundle is kept for re-renting):** (1) walls = Option B (assembly closure law; WM implements + re-validates), hold draws kept, 44 chunk-4 worlds, W-2 deferred; J4 runs on the LAPTOP after B is validated. (2) Planner rule #7 = C (TTC floor relative to native) adopted -> PL-EXP1c on the laptop. (3) Re-run the old switching-mix-referenced analyses (PL-02a, Part B <= v0.3.2) against the WM-73 truearms references, zero-sim. RESOLVED 2026-09-30: the Part B v0.3 / v0.3.1 hard links (1432 files) were de-linked (owner GO); both roots verify fully before and after, manifests unchanged.
