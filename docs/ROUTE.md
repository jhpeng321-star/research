# Current route — WM-S readiness (2026-10-09)

<!-- S-GROUPDIAG-closeout -->
**2026-11-06, S-GROUPDIAG:** [Clean2 mechanism diagnosis](../runs/active/C4-V7FT-20261017/CLEAN2_GROUP_COOP_RESEARCH.md) prioritizes certified executable labels, earlier role-aware pair opportunities, complementary macros and sparse measured marginal credit. B's total benefit is essentially best-solo benefit, not group synergy. Preserve the registered u24 criterion and B/R symmetry; changed actions/rewards require a labelled successor. Existing mining/PLANFIX ownership must be consulted before follow-up.
<!-- /S-GROUPDIAG-closeout -->

<!-- S-DIAG-closeout -->
**2026-11-06, S-DIAG:** [Clean1 evidence through final A9/B9/R9](../runs/active/C4-V7FT-20261017/CLEAN1_DIAGNOSIS_20261105.md) prioritizes production-consistent move actuation/credit, feasible joint opportunities and broader usable practice over blanket KL/imitation increases. Clean1's durable owner STOP remains authoritative. Consult existing clean2/plan-fix receipts before overlapping follow-up work; this zero-simulation analysis authorizes no dispatch.
<!-- /S-DIAG-closeout -->

<!-- S-WMCHECK-closeout -->
**2026-11-05, S-WMCHECK:** [WM error audit](../runs/active/C4-V7FT-20261017/WM_ERROR_CHECK_20261105.md) measures disagreement on 542 recovered selected training pairs. Positive predictions win 40.7%; logged gain ranking is weak and error signs depend on seed. Raw/calibrated >=80% reliability remains unresolved because genuine paired probabilities are absent. Mixed horizons prevent an isolated matched-horizon WM-error claim. Zero new SUMO or charges; no calibration artifact fitted.
<!-- /S-WMCHECK-closeout -->

> Added 2026-10-09 (docs-only). The append-only route log below is historical and
> unchanged. WM-S plan: [WM_STATE_PLAN.md](WM_STATE_PLAN.md); contributions and weekly
> plan: [WM_S2_ROADMAP.md](WM_S2_ROADMAP.md); inventory:
> [DOCS_AUDIT_20261009.md](DOCS_AUDIT_20261009.md).

**Current route.** The project is **WM-S**, a physics-informed per-step traffic world
model (IDM/MOBIL core + gated learned residual) used to pick controllers per
situation. CAVs run the real controller code, so unseen controllers can be evaluated
zero-shot (C1). This week is **WM readiness**, not RL:

1. **Braking-fix verdict.** Braking-fix and no-braking models trained 2026-10-09
   ([REGISTRATION.md](../runs/active/WM-S-MAIN2-20261009/REGISTRATION.md)); scoring in
   progress (`E:/rd-wt/ws-main2score`). DIAG120 traced the 120 s blow-ups to
   under-braking / missed jams, not phantom jams
   ([DESIGN_DISCUSSION_20261008.md](DESIGN_DISCUSSION_20261008.md)).
2. **pack_v3 fix.** pack_v3 contributed **0** training windows (fix in progress,
   `E:/rd-wt/ws-packv3fix`).
3. **entries-ON eval.** Rescoring in progress (`E:/rd-wt/ws-entrieseval`).
4. **Then one GPU run with loss rebalancing.** Black-box retraining is dropped: new
   models are compared only with the previous C1 models. Both GPUs are currently shut
   down; owner approves GPU use.

**Standing decisions.** Walls are one action (not the plan); **regret** is the primary
planner score; direct output / per-CAV switching data is the main training data; RL
(C4) comes later. Retired: FollowerStopper, the per-lane mix layer, wall-only planning,
the WM-N1 delay head, D: staging (E: only), and self-recovery/pushforward training.
Executors: OpenCode only (no ZCode).

**Change log:** check [WM_CHANGE_LOG.md](WM_CHANGE_LOG.md) before proposing any model change.
**Latest overnight status:** [records/morning_report_20261010.md](records/morning_report_20261010.md).

**Fix protocol (owner, 2026-10-09).** One model line, always trained leave-PI-and-wall-out.
Each fix is ONE training run that differs from the current best only by that fix, scored
only against the current best on the fixed unseen-controller test (`wall_same`, plus PI
when available) at 60/120 s and blow-up rate; keep it if clearly better with no horizon
worse, else drop it. An optional CPU-only seen-controls sanity check follows a positive
result. No extra control runs, retrained black boxes or extra test sets. The first current
best is the winner of the 2026-10-09 braking-fix vs no-braking comparison.

C1 zero-shot unseen-controller prediction is SUPPORTED with caveats — verified on the
PI arm and, on 2026-10-09, on the unseen WALL (structured 8.71 m vs bb_traj 25.41 /
bb_id 40.07 m HDV median |ds| at 60 s on `wall_same`;
`E:/rd-wt/ws-wallc1/_out/WALL_C1_RESULT.md`).

---

Label fix + timing audit (2026-10-03): [FIX1](../runs/active/WM-N1-LABEL-TIME-FIX1-20261003/REPORT.md). The switching-run label lookup is fixed (196 chunk2/3 campaign switching items rebound to their own swnative/swpi labels; 0 excluded), and the corrected TRAIN labels pass every identity gate. The training mapper chose a later decision time than t0 in all 5,480 labelled cells. Owner approved (a) the t0 mapper, (b) excluding 263 base/native_base items from outcome supervision, and (c) the 10 s eval correction window. M1 retrain package: [RETRAIN_READY](../runs/active/WM-N1-LABEL-TIME-FIX1-20261003/RETRAIN_READY.md); GPU50 smoke is the only training-path test. M1 unchanged.

Full TRAIN timing audit (2026-10-03): [stopped at corrected policy gate](../runs/active/WM-N1-LABEL-TIME-FULL-20261003/REPORT.md). With source-cited policy correspondence, 900 labelled items map, but 196 chunk2/3 campaign switching items (13,524 rows) carry true-fleet labels (native_fleet/pi_fleet): suspected label-binding defect. Last-row-per-tau timing not measured; exclude/rebind decision and any further read need owner approval. M1 unchanged.

TRAIN identity diagnostic (2026-10-03): [exact row and policy-name resolution](../runs/active/WM-N1-LABEL-TIME-DIAG-20261003/REPORT.md). Global row69 uses label class damping_fleet versus receipt controller damping; frozen source explains this naming difference. The prior exact-string audit check was overstrict for this row. Full TRAIN timing remains unresolved; another read needs owner approval. M1 unchanged.

TRAIN label-timing audit (2026-10-03): [stop report](../runs/active/WM-N1-LABEL-TIME-20261003/STOP_REPORT.md). Exact ten-column metadata projection stopped on row/item identity disagreement. Timing counts and M1 effect remain unresolved; one bounded diagnostic requires owner approval. M1 and real scoring remain held. This supersedes earlier statements that the actual TRAIN table was unopened.

Evaluation bridge (2026-10-03): [source foundation review](../runs/active/WM-N1-EVAL-BRIDGE-20261003/SOURCE_REVIEW.md), 47 generated cases covered with failure/repair receipts retained. Real scoring remains disabled pending action/temporal binding, comparator and truth-roster contracts plus explicit data release. Source stays in its managed branch; M1 unchanged.

EFFH-01 update (2026-10-03): GLM reports substantial 300-to-600s changes in TRAIN physical contrasts. [Source review](../runs/active/EFFH-01-REVIEW-20261003/SOURCE_REVIEW.md) corrects the unobserved after-600/steady-state claim and the signed-ratio/first-crossing interpretation. Numerical/input provenance remains report-only. [Model source audit](../runs/active/EFFH-01-REVIEW-20261003/HORIZON_LIMITATION_SOURCE_AUDIT.md) identifies timestamp binding and 10s-train/120s-eval summary risks; actual M1 impact is unverified. Keep M1 locked; no longer-horizon run or retraining is authorized.

<!-- coordinator-20261002-current -->
Member 1 completed all 30,000 CUDA training steps. Checkpoint b37a25a7 is independently hash-verified and preserved under runs/active/WM-N1-M1-20261003; all 30,000 training records pass the owner-approved loss checks. The completed run remains attributed to 45c9032 plus its documented box-side CUDA patch.

Source branch zcode/wd01 is now d96b57d: the four CUDA host copies and pandas 2.3.3 pin are upstreamed, and the J4 campaign/frozen wall-factor selector and G6 metadata report are corrected. The host-copy regression passed on CPU (CUDA unavailable), and all 28 guarded synthetic J4 checks passed.

TTC15/A7 remains unmerged because of integration defects. A real-checkpoint evaluation bridge is missing; its source-only foundation passed generated checks; real scoring remains incomplete and disabled. The frozen J4-only held-out protocol also conflicts with the completed full-TRAIN M1 and needs owner resolution before such a claim. CAL/J4B access, INC1, PL-WM1 and checkpoint cloud backup remain gated. No model accuracy or held-out outcome result is claimed.
<!-- /coordinator-20261002-current -->

<!-- WM-N1-RECOVERY-01-resume -->
**WM-N1-RECOVERY-01 resumed by owner (2026-10-02):** Owner-selected Sol repairs and guarded acceptance continue; real TRAIN stages remain separately registered. Prior split-metadata exposure remains disclosed; real40/GPU and main restructuring are still held.
<!-- /WM-N1-RECOVERY-01-resume -->

<!-- WM-N1-RECOVERY-01-stop -->
**WM-N1-RECOVERY-01 stopped (2026-10-02):** a completed executor test read the frozen CAL split world-name list despite the prohibition. Four workers stopped; candidates and logs preserved. Zero real training/SUMO; GPU readiness held pending owner direction and guarded review. [Stop receipt](../tmp/zcode_review/WMN1_CAL_SPLIT_EXPOSURE_20261002.md).
<!-- /WM-N1-RECOVERY-01-stop -->

<!-- WM-N1-RECOVERY-01-current -->
**Current coordination (2026-10-02):** use [HANDOFF_20261002](HANDOFF_20261002.md)
for the owner-authorized WM-N1 route. Three bounded OpenCode code repairs cover
contrast support/A6 token parity, TRAIN-only outcome access, and the production
member-1 launch. Reviewed integration and a registered 40-step TRAIN-only laptop
check precede GPU power-on and the five-condition approval. No new SUMO,
real-data training, CAL/HOLDOUT/chunk-5/INC1 outcome access or GPU approval has
occurred in this recovery. Ledger `WM-N1-RECOVERY-01` is in progress;
[task record](../tmp/zcode_review/COORDINATOR_RECOVERY_20261002.json).
<!-- /WM-N1-RECOVERY-01-current -->

<!-- WM24-31-complete -->
**WM-S second integration (2026-10-09):** implementation integrated and CPU-checked
on `wms/main2`: **612 passed, 6 skipped**, **136 wall tests**, **28 chain checks**.
Next engineering step: stage the permitted TRAIN inputs and apply the shared-split
preflight/smoke contract in [MAIN2_READY](../_out/MAIN2_READY.md). No training or
scientific evaluation was allocated by this integration. Historical evidence-path
validation has 1,560 pre-existing errors, with zero new integration errors.

**WM-S main integration (2026-10-09):** all requested branches are integrated on
`wms/main`; CPU acceptance is **438 passed, 6 skipped**. The optional follow-up
is staging the two TRAIN-only real-state fixtures and frozen wall-source pair
inside the workspace, then running the [READY commands](../_out/MAIN_READY.md).
No new scientific evaluation was performed. [Merge record](../_out/MAIN_MERGE.md).

**WM-S-TRAINSETUP-20261009:** CPU preparation complete. Merge the adapter and chain,
supply the externally built leave-PI-and-wall-out `WINDOWS_TRAIN/WINDOWS_VAL/EXCLUDE`,
stage TRAIN switching data to the training box, then run the recorded preflight
and real-data smoke gates. [Exact commands](../_out/TRAINSETUP_READY.md).
**WM-S switch model (2026-10-07):** implementation-ready CPU path. Closed-loop native/CIRCLES/PI handles post-t0 CAV entrants with a plan-level default controller; switch schedules and suffix commands bind via the t0 identity sidecar and tick-local dense truth. Targeted tests pass on synthetic fixtures. [Status](../_out/SWITCH_MODEL_READY.md).
**WALL-C1-PREP-20261009 (2026-10-09):** engineering preparation is fixture-verified;
real split/test sets remain pending workspace-local staging of permitted TRAIN
sources and the canonical guard. Rebuild into a new directory, verify schedules,
then materialise/score recorded-CAV wall arms. Use LPWO-trained checkpoints only.
Closed-loop wall execution awaits the documented wallexec rollout hook.
[READY / exact commands](../_out/WALL_C1_READY.md). Zero training/GPU/SUMO spent.

**WM-S2 integrated (2026-10-06):** `wms/s2int` merges `wms/s2` + `wms/entry2` + `wms/trafloss` + `wms/stab` (all new flags default OFF) and adds the optional driver-parameter sidecar loader/recovery eval and the gpuA `s2_chain.sh`. Suites pass (298 passed, 2 skipped) and the real-data CPU dry run (driver_params + entries v2 + traffic loss + stability) is finite under the 6 GB memory rule. Next: run `s2_chain.sh` on gpuA once heterogeneous-driver data is staged. [Status](../_out/S2_READY.md).

**WM-24 to WM-31 complete (2026-09-25):** the reconstructor was tested on 60 fresh worlds for the first time. Counts transfer (ideal-sensing occupied count MAE 0.25-0.45, same as development); queue detection does not (0-2 of 5 true queues with ideal sensing, vs 5/5 on development worlds); realistic sensing produces false queues in 11-18% of worlds; speeds are the weak channel; calibrated uncertainty is broken by a code defect. The lost F prior was replaced by a report-scaled climatology (WM-29). A CTM+EnKF history filter (WM-24) did worse than the snapshot constructor. Realistic sensing roughly doubles count error (WM-25). Speed errors are the most damaging for forecasting (WM-27). A hybrid vehicle-plus-field state on physical lanes is feasible (WM-30). Frozen WM-22/23 results reproduce only under CPython 3.10.20. In progress: WM-32 (calibration and scoring repair), WM-33 (queue diagnosis on CALIBRATION worlds), WM-34 (100-world clean retest at t = 480/540/600). [Fresh-world report](../output/fresh_validation_v1/WM31_REPORT.md).
<!-- /WM24-31-complete -->

<!-- WM23-complete -->
**WM-23 complete (2026-09-25):** cells hidden behind detected vehicles now take the speed of nearby observed traffic instead of the prior's free-flow speed. Zero simulation, same 21 exposed worlds; the rule was designed after seeing WM-22's miss, so this is development evidence only. Occluded queue detection rises from 1/5 to 2/5 at both radii with no false positives, identical counts and lower occluded speed RMSE (3.84 to 2.66 m/s at 50 m). Remaining misses come from undercounted queue windows and fast adjacent-lane evidence. Next question, not started: a count fill for hidden queue cells. [Report](../output/shadow_speed_v1/REPORT.md).
<!-- /WM23-complete -->

<!-- WM22-complete -->
**WM-22 complete (2026-09-24):** the F prior now fills only cells the CAV reports could not see, using a report-only visibility mask. Scored offline on the 21 WM-18 worlds, with zero simulation. The identity gate reproduces WM-20 arm A. Primary arm vs frozen: signed bias drops from +8 to +21 vehicles to about -0.5 to +1.3, speed RMSE improves in every cell, and ideal sensing finds 5/5 true queues (frozen 0/5) with no false positives. Occluded 50 m still finds 1/5 (registered criterion 1 not met): queue tails shadowed by the queue itself are blind and get the prior's free-flow speed. Next question, not started: a speed or physics fill for blind cells. [Report](../output/prior_fusion_v1/REPORT.md).
<!-- /WM22-complete -->

<!-- WM20-complete -->
**WM-20 complete (2026-09-24):** 100 m all-vehicle radius vs 50 m, and sensing-only vs memory-plus-copy, scored offline on the 21 WM-18 worlds. Identity reproduces WM-18. Sensing-only abstains; the fill overcounts. Next constructor question, not started: conservation counting and a physics fill for blind cells. [Report](../output/radius_penetration_v1/REPORT.md).
<!-- /WM20-complete -->

<!-- WM19-reviewed -->
**WM-19 complete (2026-09-24):** 21 radius-input exports independently verified; 16 adapter tests pass, including actual synthetic mass and hidden-control separation. [Reviewed report](../output/worldmodel_radius_input_v1/REPORT_FINAL.md). No real forecasts/training or WM-16 resumption. Remaining priority: observed/prior speed fusion hides queues (WM-18 occluded detects 1/5). One saved-grid prior-speed test is proposed, not executed; action-response dynamics remain unvalidated. This supersedes earlier WM-19 in-progress statements below.
<!-- /WM19-reviewed -->

<!-- WM18-current -->
WM-18 is in progress: owner-authorized 50 m multi-vehicle, cross-lane CAV sensing for reconstruction. Versioned implementation is under independent acceptance repair; no real-data scores, SUMO, fitting or model forwards yet. WM-17 is preserved; WM-16 stays paused. World-model continuation is conditional on reconstruction evidence. [Scope and review](../output/radius_sensor_v1/ROOT_REVIEW_AND_SCOPE.md).

<!-- WM17-current-route -->
Current state-construction route (2026-09-23): [WM-17](../output/state_constructor_v1/WM17_CLOSEOUT_REPORT.md) completed the authorized 25-world F/M/N/MN comparison with 25 reads, 25 frozen forwards and 100 scoring blocks, no SUMO/fitting. Local CAV motion/neighbor evidence improved some 10 m placement and speed, but inherited 50 m count error, undercovered intervals and missed slow queues block use as a trusted world-model state. The one separate *proposal only* is a CAV-only probabilistic 50 m count correction, conditional on a defensible merge-aware likelihood and fresh exposure audit; no successor allocation is opened. WM-16/WM-15 remain preserved as prior evidence, and the paused WM-16 physical campaign stays paused.

<!-- WM15-spatial-correction -->
Current route WM-15: the [first structural milestone](../output/worldmodel_spatial_v1/FIRST_MILESTONE_REPORT.md) connects CAV reports, a 10 m grid, action-conditioned future-grid diagnostics and cell→CAV dispatch in a synthetic-only smoke. Production abstains without calibrated predictions and full physical costs. The immediate evidence blocker is missing post-600 full-vehicle positions in saved runs; a narrowly instrumented uniform-arm branch is the proposed next diagnostic, not yet launched. Compare physical outcomes with native/CIRCLES/PI and a direct predictor only after matched, supported spatial actions exist. WM-14's uniform-arm allocation remains unspent; wall remains excluded pending its own episode evidence.

<!-- WM14-spatial-supersession -->
WM-14 is stopped as a uniform-arm baseline/component diagnostic. Its draft fit allocation remains unspent. Repaired source pins and 150 focused passing tests are preserved; [scope close-out](../output/worldmodel_repair_wm14/SPATIAL_SUPERSESSION.md).

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

<!-- WL-status -->
WL-01..04 completed (zero pipeline simulation until WL-03/04 harness runs; 2026-09-20/21, owner-requested parallel wall track): literature synthesis corrected the effectiveness window (capacity drop 3-18%; entry post-discharge not mid-crawl; compact 100-300 m zone ending 100-200 m upstream; wall speed >= 0.7x ambient; hold until wave absorbed; gradual capped-outflow exit; no field-proven throughput gain; no published CAV-wall entry gate exists). WL-02 implemented the wall-v2 controller standalone (E1-E5 gate, pacing law, buffer assembly, taper, ALINEA-style override, fail-closed; 24 unit tests green) and replayed the gate on the OE-59 panel report streams: every harmful executed admission refused (E2), free-flow admission suppressed (E5), and the executed beneficial K1 entry also refused — K1 hold-wall and literature-safe pacing-wall envelopes are disjoint on this evidence. WL-03 verified the full actuation chain in SUMO mechanics (all five phases, all three exit paths, protective span abort, delivery MAE 0.011 m/s, 2250 s, zero collisions); CV-20's 00:20 abort (Windows PermissionError at spawn) is disclosed there. WL-04 ran ON-vs-OFF shared-prefix efficacy arms in the harness world (3 exposed m2700 seeds, 14400 s, prefix identity exact): 920002 -30.2 s/veh (-18.2%) and effort -18.4% with 2 episodes; 920003/920004 gate no-ops (+0.001/-0.003); zero collisions — first physical evidence that envelope-gated entry captures value where the envelope exists; NOT pipeline evidence. Wall-vs-damping/controllers under the CV oracle remains the registered next step (feasibility + incremental-benefit approval required). Reports: [WL-01](../output/wall_literature_review_v1/REPORT.md), [WL-02](../output/wall_v2_verification_v1/REPORT.md), [WL-03](../output/wall_v2_mechanics_v1/REPORT.md), [WL-04](../output/wall_v2_efficacy_v1/REPORT.md).

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

# Current scientific route

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

## Latest reviewed state — 2026-09-10, OE-27 / CORR-38

[Consolidated evidence and warning report](../output/duration_selection_v11/REPORT.md)
supersedes OE-26's policy totals, event count and incomplete verification claim.
Six instrumented arms now pass collision/physical-vector/phase reproduction checks;
both correct per-seed H5/H20 prefixes pass. Common-25 all-decline selectors equal
damping in every physical metric; always-H5 improves mean service/smoothness/TTC
but increases waiting and collision involvements. This is a trade-off, not approval.

One frozen front/rear warning comparison used all six saved streams, with no
simulation or fitting. Reports warn before collisions but have many predictions
without collision; all successful wall-arm warnings occur after handback. Offline
neighbor attribution distinguishes warnings about the eventual collision partner
from warnings about a different neighboring vehicle. This does not establish
entry, duration, or response benefit. Do not repeat the closed OE-14–16/OE-22
feature searches or restart PPO from this result. A next physical test needs a
decision-timed mechanism and a concrete feasible response; neither is established
for the existing wall by these late warnings. Full source and regression checks
are in the OE-27 report. Earlier dated blocks below are historical context where
they conflict with this reviewed state.

**Workspace migration MIG-01 (2026-09-10):** required `D:/research` inputs now
live under `assets/research/`; active configuration and historical SUMO input
resolution use those verified local copies. See
[migration details](RESEARCH_ASSET_MIGRATION.md). No physical experiment or
scientific conclusion changed. New registrations must record the new config/source
identity; do not rewrite old registrations or bypass source-identity checks.

## Current state (2026-09-07 — read this first; everything below is history)

- **Verified physical findings** (source-verified against registered analyses;
  ledger `verify-sources`: 12 sources, 1,230 checks, 0 mismatches): every executed
  wall action on the two exposed seeds (4107/4108, m2700 p20) fails the unchanged
  original criteria — entry walls (strong, gentle, finite-lifecycle, uncapped-target),
  the brake/HOLD factorial, all eight same-prefix continuation actions
  (keep/lower-22/raise-26/release), and the hold-extension arms
  (ledger HE-01). Strongest measured result: the 4107 raise-26
  continuation improves service −8.436 s/request vs damping and beats keep-24 by
  4.282 s/request, but violates the TTC guardrail (+0.0396 veh-s/request) — harmful,
  not qualifying. Extending its HOLD to 30 s improves the same-prefix TTC
  (−27.5 veh-s vs raise-H20) but costs effort and 2 collision involvements, and
  still fails vs damping (TTC +0.020141, over by 0.000141; 2 collisions). The
  comparator repair (hold_extend_patch_v1, HE-03/CORR-12) supersedes the
  REPORT_V2 continuation table; machine-generated comparisons:
  [COMPARISONS_V3.json](../output/hold_extend_patch_v1/COMPARISONS_V3.json),
  [REPORT_V3](../output/hold_extend_v1/REPORT_V3.md). Numbers:
  [RESULTS_SUMMARY](RESULTS_SUMMARY.md) and the ledger
  ([protocols/experiment_history.json](../protocols/experiment_history.json)).
- **Current executable action design and unresolved choices**: the finite-lifecycle
  wall with report-based cohort selection is the current working design with
  verified completed maneuvers; earlier designs also completed some cases. Its continuation frame (adjust/keep/release at
  HOLD+5) is mechanically proven. Unresolved: which mechanism could produce a
  *qualifying* action (none exists — see
  [OPEN_QUESTIONS](OPEN_QUESTIONS.md) OQ-1/OQ-2); assembly-vs-handback isolation;
  any fresh seed. Retaining CIRCLES damping remains the only policy consistent with
  all measured labels under the original joint criteria (not metric-wise dominant —
  e.g., 4107 raise/lower have zero collisions where damping has four).
- **Observation findings and review status**: reports support state discrimination
  and feasibility/execution; nothing is action-validated. The TTC/observation
  sequence is under versioned review (ledger `artifact_review`): ground truth and
  the v4 phase decomposition are current; the v3 warning claims and v5 evaluator
  are superseded. **A v7 closure now exists** in
  [REPORT_TTC_V7](../output/ttc_diagnosis_v1/REPORT_TTC_V7.md), superseding v6;
  its exploratory warning findings are distinct from independently validated
  action-selection evidence. Version-level ledger review metadata may lag this
  closure and must be reconciled before reuse. Warning candidates
  F/M are exploratory and retrospectively motivated, not action-validated.
- **Latest authorized task and execution status**: the single instrumented
  120 s mechanics verification of the repaired PPO fixture **PASSED**
  (ledger PP-04, campaign CAMP-PPO-MECH-V1, runs/active/ppo_mech_v1): 240
  exact 0.5 s advances, 119 control ticks, 120 reward rows, completed 120.0 s,
  full chain masked-ACTIVATE-refusal → ACTIVATE → CONTINUE (H20→H30) →
  RELEASE → identical-CIRCLES handback, zero collisions/teleports, no clock
  violations. The mechanics prerequisite for the bounded PPO pilot
  (PP-01) is SATISFIED; references (12×2400 s), training (32×2400 s) and
  development evaluation (12×2400 s) still require their own authorization
  and registration. The historical clock anomaly did not recur; its cause
  remains unresolved. The bounded PPO pilot itself was EXECUTED and
  AUDITED (PP-05/PP-06): the frozen policy produced one qualifying
  beneficial development case (4115, beating damping and the fixed wall)
  and one harmful case (4117); the actor's deterministic decisions are
  IDENTICAL to initialization on every recorded training state and equal
  to 'activate at the first admitted opportunity, release early'
  (98.4% of decisions feasibility-determined); comparator mislabels
  corrected (4117-fixed BENEFICIAL, 4115-fixed HARMFUL — see
  output/ppo_audit_v1/ and output/ppo_timing_v1/). Training and acceptance evaluation remain paused;
  further simulation requires new authorization.
- **Duration mechanics + causal selector (2026-09-08, PP-10/PP-11, repaired
  PP-12)**: the ONE authorized 120 s mechanics verification of the
  wall_ppo/2 duration semantics **PASSED** (ledger PP-10, campaign
  CAMP-PPO-MECH-V2, runs/active/ppo_mech_v2): 12/12 checks incl. full H30
  delivery — masked request t=5 → ACTIVATE t=10 → MAINTAIN×3 (H20) → EXTEND
  admitted H30 at the t=40 deadline via remaining_plan → MAINTAIN →
  scheduled RELEASE t=50 → coast-terminal handback to identical damping;
  240 exact 0.5 s advances, 0 collisions/teleports, no protective abort.
  The causal current-report candidate selector (wall_ppo/3) is IMPLEMENTED
  with zero simulation. CORRECTION (CORR-19): PP-11's verification covered
  a WAIT-only empty-world path only and carried four integration defects
  (phase-blind driver, no offer binding / zero candidate observations while
  inactive, execution re-ranking risking silent cohort substitution,
  ignored profile/lanes). REPAIRED in PP-12 (selector causal_selector/2,
  campaign CAMP-PPO-SELECTOR-REPAIR,
  [output/ppo_selector_repair_v1/REPAIR_REPORT.md](../output/ppo_selector_repair_v1/REPAIR_REPORT.md)):
  explicit offered-proposal binding before observations, execution bound to
  the exact approved offer with atomic refusal, named phase-aware scripted
  comparator, causal eligibility and profile/lanes consistency; verified on
  a POPULATED mocked environment through the complete
  select→activate→assemble→H20→EXTEND→H30→RELEASE→handback chain with
  external-actor and budget-rejection coverage; 876 tests + Ruff pass.
  **LIVE selector mechanics then VERIFIED (PP-13, campaign CAMP-PPO-MECH-V3,
  runs/active/ppo_mech_v3)**: the ONE authorized 120 s allocation passed
  16/16 checks — masked probe t=5 with an offer already bound from a live
  4-pair search → ACTIVATE t=10 committing exactly the approved offer
  (fingerprint equality) → assembly ready 19.0 → MAINTAIN×4 (H20) → EXTEND
  admitted H30 at t=39 → scheduled RELEASE t=49 = ready+30.0 → coast
  handback; 240 exact 0.5 s advances, 0 collisions/teleports; live search
  cost 10.4 ms max (fixture scale only). No live selector run beyond this
  is authorized; the updated proposal is
  [output/ppo_selector_v1/SELECTOR_NEXT_VERIFICATION.md](../output/ppo_selector_v1/SELECTOR_NEXT_VERIFICATION.md).
  Execution is verified; prospective benefit/risk evaluation design lives in
  OE-03's probabilistic protocol. Training, acceptance evaluation and any
  further simulation remain gated on new authorization.

- **Entry-observability sufficiency (2026-09-08, OE-01, zero simulation)**: the
  entry-frame evidence base is inventoried and diagnosed as INSUFFICIENT to
  decide whether pre-decision CAV reports can select when a feasible wall beats
  damping — 12 executed matched wall-entry comparisons, 7 complete maneuvers,
  exactly one causally clean beneficial label (4117 fixed wall), so no
  scenario-grouped observation-to-benefit test is admissible. The PPO arms'
  pre-600 activations used a future-selected t=600 cohort identity and are
  labeled non-causal. Causal report histories and a provenance-bound
  observation dataset exist for every cell; the smallest missing evidence is a
  staged replication of the 4117 configuration (proposal OE-01-P, ceiling
  28,800 s, NOT executed, requires its own registration and authorization).
  See [output/entry_observability_v1/REPORT.md](../output/entry_observability_v1/REPORT.md)
  and OPEN_QUESTIONS OQ-3.
- **OE-02 staged replication EXECUTED (2026-09-08)**: Stage A of OE-01-P ran
  under owner authorization with causal t=600 cohort binding (selector on the
  scenario's own damping snapshot; 16-check prefix matching per pair) —
  **0/2 qualifying**: 4119 (complete 20 s HOLD) improves service/braking/TTC
  but violates the effort allowance; 4120 aborts 4 s into assembly and four
  guardrails worsen. Stage B was never activated (registered stopping rule);
  9,600 of 28,800 s charged across the superseded campaign chain [CORR-18] (v1 0 s
  pre-launch defect; v2 freeze-gate-halted by concurrent ppo_env edits — off
  the campaign's control path by import-graph evidence — with its completed
  s4119 reference adopted). 4117 remains the single unreplicated beneficial
  entry; observation-to-benefit fitting stays inadmissible. Causal
  observation dataset collected (provenance-bound). See
  [output/oe02_replication_v1/REPORT.md](../output/oe02_replication_v1/REPORT.md).
- **OE-03 probabilistic specification and fixed-sample proposal (2026-09-08,
  zero simulation)**: OE-02 accounting corrected via CORR-18 (chain 9,600 s;
  8 complete-delivery entry cells); probabilistic targets (qualifying
  probability, per-cost expectations, violation rates, conditional q90
  tails, coverage) specified for the fixed-wall contract with all thresholds
  left as prospective owner decisions; exploratory continuous-effect
  prediction NOT supported beyond a demand-class constant (negative LOSO
  diagnostic, no model carried forward); a fixed-sample 12-scenario entry
  study (4121–4132, 57,600 s ceiling, independent of the selector's pending
  mechanics) is SPECIFIED, not authorized. See
  [output/entry_observability_v2/REPORT.md](../output/entry_observability_v2/REPORT.md).
- **OE-04 literature-grounded observation contract (2026-09-08, zero
  simulation)**: 4 primary sources synthesized (no intervention-benefit
  predictors, no importable feature ablations); hypotheses B/C/D audited as
  already implemented (`regional_reports`, `proposal_features`, registered
  admission) and aggregated for the first time into decision-time features on
  the 14 causal cells; no addition beats the observed-regime benchmark on
  leave-one-group-out MAE (G-MERGE coverage-limited:
  3/14 snapshots have a ramp reporter at p20); wave-front estimator left
  optional/unimplemented; OE-03 study observation requirements frozen (full
  streams retained, q90 replaced by violation counts + realized worst case).
  **Interpretation narrowed by CORR-20 (OE-05):** the fits were univariate
  (never joint), on the service-difference target only, against an observed
  mean-speed regime constant (not the exogenous demand class), each feature on
  its own complete-case subset.
  See [output/entry_observability_v3/REPORT.md](../output/entry_observability_v3/REPORT.md).
- **OE-05 observation-to-benefit reuse audit + joint comparison (2026-09-08;
  interpretation corrected 2026-09-09, CORR-21)**: prior observation programs
  classified — combined causal observation histories were **never tested** for
  wall-entry benefit (historical gates and SW-10's multivariate fit were other
  actuators/eras; OE-03/04 were univariate only; the PPO pilot's trained actor
  was deterministically identical to initialization — no learned selectivity, no
  observation-to-benefit mapping). A joint 4-feature ridge vs the
  observed-regime constant was executed and **NEGATIVE** (service MAE 8.571 vs
  6.043; 13 cells/8 groups; no model carried forward). **CORRECTION
  (CORR-21):** the freeze's identical-contract claim was false — the panel
  mixes three treatment identities (8 natural_longitudinal_wall/1, 4 PP-05
  lifecycle_v2, 2 OE-02 finite_lifecycle_wall/2 cells), so the comparison is
  retained only as a mixed-controller exploratory analysis (arithmetic
  verified from saved predictions) and is withdrawn as a one-contract test.
  See [output/entry_observability_v4/REPORT.md](../output/entry_observability_v4/REPORT.md).
- **OE-06 treatment-compatibility repair (2026-09-09, zero simulation, zero
  fitting)**: every cell traced to its registered configuration — three
  treatment identities confirmed (different braking authority, binding
  mechanisms, runners, decision times; damping reference identical throughout).
  Usable same-family evidence after filtering: **6 cells / 6 groups
  (4115–4120)** with a binding/runner covariate; no seed has two wall actions
  from a matched state, so offline selection analysis is impossible. Limitation
  diagnosis: treatment inconsistency + no per-seed action variation + ≤8
  independent groups + p20 observation coverage; observation sufficiency stays
  untested in both directions. Next step: **OE-03's 12 matched-pair entry study
  RETAINED with single-contract amendments** (57,600 s; per-cell action schema;
  exposure-audited seeds — never described as fresh; owner-owned thresholds;
  original episode labels preserved) — proposal only, not authorized. See
  [output/entry_observability_v5/REPORT.md](../output/entry_observability_v5/REPORT.md).
- **OE-07 single-contract wall-entry study EXECUTED (2026-09-09, owner-authorized,
  12 matched pairs)**: the OE-03 proposal as repaired by OE-06 ran under campaign
  CAMP-OE07-ENTRY-PAIRS (`runs/active/oe07_entry_pairs`): 50,400 s of the 57,600 s
  ceiling charged and verified completed (12 damping + 9 wall arms; 3 cohort refusals
  retain damping; no retries; 0 integrity failures; 9/9 pair-matching receipts
  17/17). Identity verified BY ARTIFACT pre-launch (registration profile == executed
  OE-02 summary == `LifecycleProfile()`; grader reproduces registered OE-02/PP-06
  labels; 23/24 control-path modules byte-identical to OE-02, the exception entering
  only through the pinned null-command `DampingPolicy`). **Physical: the fixed entry
  QUALIFIED in 3/9 executed arms** (4125 service −31.18 all-costs-improving; 4129
  qualifying via braking/effort/TTC with service +0.26 within allowance; 4132
  service −15.08 all-costs-improving; all three zero collisions) vs 6 harmful
  (service −31..+11 spread; collisions 0–6) — qualifying rate 0.33, exact 95% CI
  [0.075, 0.70], ITT 3/12; refusals 25%. **Prediction (frozen protocol,
  exploratory): the pre-registered 13-feature causal predictor does NOT beat the
  regime/global constants on held-out service (MAE 9.20 vs 6.66, n=3)** and is
  missed badly on the braking-qualified cell — named gaps: sample size,
  service-only target, unpredicted refusals. Decision frame: benefits vary; predictor
  does not yet add value; joint-criteria prediction needs more single-contract
  pairs (owner decision; no automatic successor). See
  [output/entry_pairs_v1/REPORT.md](../output/entry_pairs_v1/REPORT.md).
- **OE-08 decision-level analysis of OE-07's saved predictions (2026-09-09,
  zero simulation, zero fitting)**: out-of-fold, the criteria-conjunction
  screening rule (thresholds = the original criteria) **selects nothing** —
  predicted allowance thresholds sit far below prediction noise, and sign
  errors hit both beneficial held-out walls (4129: service/braking/effort all
  wrong-signed, despite its qualifying via braking/effort; 4132: effort) —
  while always-wall catches 2/3 at base rate. Better TTC MAE (direction right
  3/3) does not translate into violation recognition. Documented: OE-07's
  access-sequence deviation (all labels materialized before the dev fit; split
  by code path only) and the per-fold-persistence gap; budget corrected
  (28,800 damping exactly consumed; 7,200 s residue is not authorization).
  **Successor proposal (OE-09, NOT authorized):** 12 new matched pairs under
  the unchanged contract as a labelled data-collection/prediction study with
  joint-criteria targets, a pre-registered screen, per-fold persistence and
  separated grading — a policy evaluation is not yet defensible. See
  [output/entry_pairs_v2/REPORT.md](../output/entry_pairs_v2/REPORT.md).

- **OE-09 successor protocol FINALIZED (2026-09-09, proposal only, NOT
  authorized, zero simulation/fitting)**: 12 additional matched pairs under
  the unchanged verified contract (57,600 s new authorization); reuse
  manifest admits 11 same-contract rows into training now (~17–19 after),
  PP-05 cells diagnostics-only, CUR-09 excluded; ridge α ∈ {0.1, 1, 10}
  by development LOSO; primary measure = mean per-metric MAE ratio vs the
  regime constant on untouched eval cells + sign-error rate; per-fold
  persistence and five-step freeze/unlock access enforcement with mocked
  boundary tests; criteria-conjunction screen pre-registered with feasibility
  masking and identical baselines; four-way conclusions predefined (improved
  prediction / useful screening / continued failure / insufficient evidence).
  **Unresolved: owner seed-allocation decision** — the exposed registry
  (4107–132) is fully consumed (CORR-23 fixes OE-08's 4133 statement).
  See [output/entry_pairs_v3/OE09_PROTOCOL.md](../output/entry_pairs_v3/OE09_PROTOCOL.md).
- **OE-10 / OE-09 block-2 EXECUTED (2026-09-09, owner-authorized)**: campaign
  CAMP-OE09-BLOCK2 (`runs/active/oe09_entry_pairs`), seeds 4133–1444
  (owner-authorized audited block; Amendment 1 bound by hash). **48,000 s of
  57,600 charged, 20/20 allocations verified completed, 0 failures, no
  retries**; damping ceiling exactly consumed (incl. 4 refusals with explicit
  reference reuse). Development first (5 executed + 3 refusals), hashed
  freeze (α = 10.0 by dev-LOSO over 16 pooled training rows), then
  evaluation unlocked mechanically. **Held-out: 3/3 executed walls harmful,
  1 refusal; primary prediction aggregate 3.92 (worse than the regime
  constant; service-only ratio 0.70; sign errors 8/15 vs 4/15); the
  pre-registered screen selected 1 harmful wall and 0 beneficial (none
  existed).** Predefined conclusion: **continued failure — the
  single-decision-time linear observation/screening approach is closed at
  this scale**; no automatic successor. Pooled same-contract evidence: 4
  beneficial / 19 executed arms (exposed development data). See
  [output/entry_pairs_v3/REPORT.md](../output/entry_pairs_v3/REPORT.md).
- **OE-11 4141 selection-failure diagnosis (2026-09-09, zero simulation/
  fitting)**: CORR-25 corrects the OE-10 attribution — the screen selected
  **4141** (sole violation: service +5.065 s/req; 0/0 collisions; 4142 was
  *rejected* on the predicted effort allowance). The harm reconciles from
  per-tick regional presence: zero through handback (648 s), early wall
  advantage to 1200 s (−1,581 veh-s), stall concentrated 1200–1800 s
  (+7,925), 94% pre-merge, 97% on HDVs, both arms carrying the same standing
  queue from t=0 — the wall rescheduled its discharge. Prediction
  reproduced exactly and all features in-range (no extrapolation/preprocessing
  error); decision-time lane asymmetry is INVERTED between classes (beneficial
  19.6–21.2 vs harmful 7.6–15.5), so representation-loss of lane structure
  is not evidence-backed. **Next direction (proposal only, NOT authorized):**
  discharge-phase-referenced activation contrast on already-exposed block-2
  seeds reusing existing damping references (7,200–9,600 s) — mechanism
  isolation per OQ-4's carve-out, not a timing selector. See
  [output/entry_pairs_v4/DIAGNOSIS_4141.md](../output/entry_pairs_v4/DIAGNOSIS_4141.md).

- **OE-12 accounting repair + trigger replay (2026-09-09, zero simulation/
  fitting)**: the 4141 accounting closed with an exact identity
  (+7,075.5 = +6,980.0 presence + 95.5 insertion lag, corroborated by the
  source-wait difference and CAV first-response shifts; approximate,
  bounded 1.35% — CORR-26 refines the "tick edges" wording), and the
  frozen rising-approach-speed trigger was replayed on all 12 block-2
  damping streams: **8/12 fire within 2 s of t=600, the only 2 later
  triggers immediately lose cohort feasibility, 0 new feasible proposals
  → the rule is not ready and no physical contrast is proposed.** The
  blocker is instrumentation/coverage (2–5 near-queue reporters on a
  ~94% HDV queue), not thresholds. See
  [output/entry_pairs_v5/REPORT.md](../output/entry_pairs_v5/REPORT.md).

- **OE-13 trigger-replay integrity repair (2026-09-09, zero simulation/
  fitting; one authorized offline replay)**: v1's current-bin boundary
  (now = t+1) was a real tick-level defect — 400/14,412 predicate
  differences under the repaired strictly-causal window (which adds the
  last pre-decision second v1 dropped) — but **latent at the result
  level**: the corrected v2 replay (5/5 synthetic boundary tests)
  reproduces the OE-12 verdict exactly (9/12 within 2 s; later
  4134/4135/4137 all refused). CORR-27 corrects the accounting: +6,980.0
  sampled presence (approximate) + +64.5 measured source-wait + +31.0
  veh-s unexplained (0.44%). The rule-not-ready conclusion stands on the
  repaired implementation. See
  [output/entry_pairs_v6/REPORT.md](../output/entry_pairs_v6/REPORT.md).

- **OE-14 interaction-feature closure (2026-09-09, zero simulation; one
  bounded offline comparison)**: the neighbor-interaction hypothesis
  (per-member front gap now / min-history gap / front closing rate /
  trailing rear gap, frozen in FREEZE_INTERACTION.md before any
  association; neighbors anonymous — per-member histories, not
  per-neighbor tracking) was compared against the 13-feature aggregate set
  under the identical frozen procedure on the 16 training rows: primary
  aggregate 1.131 → 1.108 (both worse than the regime constant; inside
  the noise band), service MAE worsened (dev and exposed), violation
  recognition 7/15 → 9/15, screen unchanged (selects only 4141), and the
  I-features overlap beneficial vs harmful classes completely. **This
  specific feature hypothesis is closed**; per-neighbor tracking is
  impossible with anonymous detections, and general observation
  sufficiency remains unresolved. See
  [output/entry_pairs_v7/REPORT.md](../output/entry_pairs_v7/REPORT.md).
  **Corrected by CORR-28/OE-15**: the aggregate improvement was an
  artifact (defective feature extraction + non-nested α selection);
  repaired, the interaction group worsens the aggregate (1.1441 → 1.1577).

- **OE-15 repair + observation-representation closure (2026-09-08, zero
  simulation; one bounded offline comparison suite)**: all four audited
  OE-14 defects affected saved rows and are repaired with regression-locked
  tests (CORR-28; v7 preserved). Two new literature-grounded groups —
  frozen before testing in GROUP_SPEC.md: member-measured interaction
  pressure (front/rear closing speeds with verified signs, time headway,
  closing persistence) and reporter spatial/temporal structure (near-queue
  braking persistence, lane differential, far-upstream gradient, mean
  acceleration) — were compared against the repaired base under nested-α
  LOSO (α selected inside every outer fold): **both worsen the primary
  aggregate (1.1671 / 1.1856 vs 1.1441) and service; nothing beats the
  regime constant; the screen still selects only harmful 4141 in every
  arm.** 16 synthetic + 11 regression tests pass. The linear
  snapshot-summary representation family is closed on this contract; the
  untested remainder is per-lane ahead/behind radar with stop-time memory
  (2011.00120), absent from the saved snapshots. No successor simulation
  proposed. See
  [output/entry_pairs_v8/REPORT.md](../output/entry_pairs_v8/REPORT.md).
  **Group B corrected by CORR-29/OE-16**: its implementation admitted
  downstream reporters and reporter-selected history; corrected aggregate
  1.2103 (still worse than A0 and the regime constant) — closure stands.

- **OE-16 Group B spatial repair + corrected C2 (2026-09-10, zero
  simulation; one bounded refit)**: three verified defects — near-queue
  predicate lacked the 0 m lower bound (93/468 decision and 1082/5180
  history frames downstream of the merge, all 19 seeds), B1 counted
  reporter-selected full histories instead of per-frame membership, and
  the per-seed diagnostic denominator included the held-out group (the
  primary aggregate was already correct). Correction frozen in
  CORRECTION_SPEC.md before any predictive inspection; 8 synthetic tests
  pass (boundaries inclusive, per-frame membership, future/stale
  exclusion, missingness, I/A identity on all 19 seeds). All four Group B
  features changed in all 19 seeds; corrected C2R: **aggregate 1.2103,
  service 14.23 — worse than A0 (1.1441/13.53) and the regime constant
  (9.97); the pre-registered criterion (beating defective C2's 1.1856) is
  NOT met**; every per-seed diagnostic value changed and no distributed
  gain survives the clean fold-local denominator (at best 4/16 seeds).
  **The representation-family closure stands on a correct
  implementation**; the negative is limited to this representation,
  model family, and 16-row exposed dataset. Remaining question:
  outcome diversity (registry-exhausted) vs information the sensor
  interface never recorded (cross-lane neighbours, uncensored stop
  durations) — an owner instrumentation decision, not an offline one.
  See [output/entry_pairs_v9/REPORT.md](../output/entry_pairs_v9/REPORT.md).

- **OE-17 duration-selection pilot (2026-09-10, 38,400 of 48,120 s)**:
  three questions, one dataset. Physical: early H5 release vs H20
  continuation is a large sign-varying lever on 7 compatible dev pairs
  (C-R service -19.0..+28.6 s/request; H5 whole-policy beneficial on 3/7
  where H20 qualified on 2/7; 4123 flips harmful->beneficial under H5).
  Observation: the frozen ER logistic FAILED its predeclared dev gate
  (OOF log loss 4.83 vs constant 0.87) - unassessable. Policy: prospective
  branches collapsed to n=1 (4148 refusal; 4146/4147 aborts) - no
  improvement demonstrated. Reuse required byte-level compatibility proof
  (asset migration changed provenance hashes, not worlds; OE-02 lineage
  excluded). No duration law, no calibrated probabilities; branch yield is
  the binding constraint. See
  [output/duration_selection_v1/REPORT.md](../output/duration_selection_v1/REPORT.md).
- **OE-18 evaluation correction (2026-09-10, zero simulation; CORR-30)**:
  OE-17's numbers corrected from source summaries — chosen-policy
  accounting attributed continuation to release (vocabulary bug; the
  frozen continuation choice is 2.85 s/service WORSE than always-release,
  not better), the printed contrast sign was inverted, the prospective
  constant leaked the prospective label (corrected to dev-only 3/7), and
  the eligibility gate used a post-branch outcome: all 17 dev sources
  were decision-time eligible at HOLD+5 (9 with no executed H5 arm);
  4146/4147/4148 exclusions are genuine. Conclusions survive with
  corrected numbers; 4123 shows service preference differs from
  qualification. See
  [output/duration_selection_v2/REPORT.md](../output/duration_selection_v2/REPORT.md).
- **OE-19 duration-axis qualification study (2026-09-10, 21,600 s — exact
  ceiling)**: 9 new H5 runs on the decision-time-eligible sources (all
  9/9 preflight PASS, all runs verified bit-identical prefixes + receipts).
  Over 16 compatible branch pairs: release-only qualifies 2 (4123, 4137),
  continuation-only 1 (4125), both 3, neither 10; always-release 5/16
  qualifying vs always-continue 4/16 vs hindsight oracle 6/16 (headroom =
  one case); E/ER features saved with prefix-verified equality, no
  fitting. Duration creates two-sided qualification choices; a selector
  is not demonstrated and the residual value case is bounded. See
  [output/duration_selection_v3/REPORT_OE19.md](../output/duration_selection_v3/REPORT_OE19.md).
- **OE-20 entry-selection assessment (2026-09-10, zero simulation;
  CORR-31)**: first H5-target entry-selection comparison (26
  opportunities; 18 labeled groups); preregistered 4-feature logistic
  never activates; negative result. See
  [output/duration_selection_v4/REPORT.md](../output/duration_selection_v4/REPORT.md).
- **OE-21 OE-20 correction (2026-09-10, zero simulation; CORR-32)**:
  fold-training-only constant corrected (Brier 0.2249/log loss 0.6535) —
  small unvalidated model advantage; 4119's always-H5 UNKNOWN; temporal
  features untested. See
  [output/duration_selection_v5/REPORT.md](../output/duration_selection_v5/REPORT.md).
- **OE-38 OE-37 review + correction (2026-09-11, zero simulation;
  CORR-42)**: independent review demonstrated OE-37's factual errors: OE-34
  fitted 300 features (12 aggregate + 288 relational across six tracked
  reporter slots, three lags, sixteen fields); OE-21 showed a small
  numerical advantage (not a loss); H5 qualifying is 8/60 fresh (not
  5/18); historical mode31 and current mode27 collision evidence must be
  separated; the physical mechanism investigation was not completed. OE-37's
  Route D recommendation is withdrawn as unsupported; **Route C** (state
  measurable but available actions offer little useful improvement) is the
  corrected route. The physical flow/accumulation/queue reconstruction
  remains to be done. See
  [output/duration_selection_v10/CORRECTED_ASSESSMENT.md](../output/duration_selection_v10/CORRECTED_ASSESSMENT.md).
- **OE-22 temporal-block comparison (2026-09-10, zero simulation)**:
  adding trend30/60_approach_mean_speed improves scoring (Brier
  0.2140/log loss 0.6158 vs snapshot 0.2221/0.6431 vs constant
  0.2249/0.6535; 11 improved/7 worsened) but ZERO decision changes at
  the frozen 0.5 rule. See
  [output/duration_selection_v6/REPORT_OE22.md](../output/duration_selection_v6/REPORT_OE22.md).
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
- **OE-24 collision-event diagnosis (2026-09-10, zero simulation;
  CORR-35)**: OE-23's policy accounting corrected (refusals in absolute
  totals, cancel in differences; common-25 never 5118.84 vs always-H5
  5011.33; 4119 UNKNOWN; full-26 activated totals INCOMPLETE). All five
  H5 collision cases reconcile to intervals.jsonl vehicle-involvements
  (2 per event); 4124's event identical in the H20 arm; 4142's 458 s
  event pre-entry and shared by all arms; H5-only post-release events in
  4/5 cases (nearest 51 s after release), causally unresolved. No
  release-adjacent collision or ownership discontinuity. Recommendation
  C: further re-instrumented evidence needed before choosing maneuver
  repair vs observation study. See
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
- **OE-20 entry-selection assessment (2026-09-10, zero simulation;
  CORR-31)**: first H5-target entry-selection comparison — 26
  opportunities reconciled from metadata (7 refusals; 19 feasible: 16
  executed H5, 2 pre-branch-equivalent 4120/4126, 4119 missing); 18
  labeled groups (5 beneficial / 11 harmful / 2 nonbeneficial). The
  preregistered 4-feature logistic (fixed lambda=10, threshold 0.5)
  LOSES to the training-only constant (OOF 0.222/0.643 vs 0.201/0.591)
  and NEVER activates — it equals never-escalate while always-H5 sums
  107.5 s better. Negative result; fresh validation not justified. See
  [output/duration_selection_v4/REPORT.md](../output/duration_selection_v4/REPORT.md).
- **OE-21 OE-20 correction (2026-09-10, zero simulation; CORR-32)**:
  fold-training-only constant corrected (4/17 positives, 5/17 negatives:
  Brier 0.2249 / log loss 0.6535) — the model shows a small unvalidated
  advantage (0.2221/0.6431), reversing "loses to the constant"; policy
  populations separated (4119's always-H5 outcome UNKNOWN); temporal
  features stated untested; 4120/4126 inherited labels justified
  case-specifically, 4119 not equivalent. See
  [output/duration_selection_v5/REPORT.md](../output/duration_selection_v5/REPORT.md).

## History (milestones; evidence preserved, not active instructions)

The owner-authorized [next-experiment plan](NEXT_EXPERIMENT_PLAN.md) organizes the
remaining work into separately gated scenario, maneuver, mechanism, duration,
replication and observation-selection blocks. The owner has requested execution
and findings after every part. Each stage still requires its explicit evidence
gate and bounded registration. Part A is complete with zero simulation and keeps
the original map; see [its report](../output/lifecycle_repair_v1/PART_A.md).
Part B1's [finite-maneuver repair](../archive/circles_era/docs/LIFECYCLE_REPAIR_CHECK.md) stopped after its
first120-second fixture refused at entry; [failure and diagnosis](../output/lifecycle_repair_v1/PART_B1.md)
are preserved. Part B2 is a separately registered [maintainable-readiness correction](../archive/circles_era/docs/LIFECYCLE_MAINTAINABLE_CHECK.md),
with an additional240-second mechanics ceiling before the still-unallocated two2400-second
corridor cases. All physical settings and fixture geometry remain fixed. The
cumulative B2 ceiling including B1 was5160 simulated seconds. B2 subsequently
verified the normal maneuver but its downstream-disturbance vehicle could not
insert; [both outcomes are retained](../output/lifecycle_repair_v1/PART_B2.md).
The [B3 fixture-only correction](../archive/circles_era/docs/LIFECYCLE_FIXTURE_INSERTION_CHECK.md) changes
only that native actor's departure-speed semantics, budgets one additional120s,
and reuses the passing normal case under explicit evidence checks. The wall
controller stays version2; cumulative ceiling including the two full
corridor jobs is5280s. B3's corrected disturbance fixture passed: actual8s HOLD
and1s exit after the acquired constraint appeared; combined with the reused
normal20s HOLD, both mechanics questions are physically verified. The two full
corridor jobs completed under `runs/active/lifecycle_repair_v2`:4107 delivered
12s assembly/20s HOLD/1s exit but fails TTC and absolute-collision criteria;
4108 refused the original cohort and exactly matches damping. See the
[Part B physical report](../output/lifecycle_repair_v2/PART_B.md).
Part C completed the single [acquired cohort eligibility change](../archive/circles_era/docs/COHORT_ELIGIBILITY_CHECK.md),
with verified4107 reuse and one new2400s4108 episode. The upstream4108 cohort
delivered18s assembly/20s HOLD/1s exit but worsened all five continuous costs;
see [Part C](../output/lifecycle_repair_v2/PART_C.md). Part D completed both
[same-prefix HOLD+5 release episodes](../archive/circles_era/docs/DURATION_RELEASE_CHECK.md) under unchanged
admission and exit law. Both deliver5s HOLD/1s exit but remain harmful. In4107,
early release costs28.03s/request versusH20 while improving TTC/source/safety;
in4108 it changes very little. See [Part D](../output/lifecycle_repair_v2/PART_D.md).
The bounded A–D program is closed: five distinct corridor traces in two exposed
seeds, four harmful and one nonbeneficial refusal, zero qualifying or strict
joint successes;12480 allocated simulated seconds including preserved mechanics
failures.642 tests and Ruff pass. Part E replication, Part F fitting/extra data
and Part G prospective efficacy remain gated; no automatic successor is authorized.
The offline F audit and methods/negative-result draft are complete. Acceptance
stays sealed and training paused. Read the [complete stage findings](../output/lifecycle_repair_v2/REPORT.md)
and [E–G decisions](../output/lifecycle_repair_v2/PART_E_F_G.md) before proposing
new work. Do not rerun this plan as if its questions were still unanswered.

Owner clarification, 2026-09-06: CIRCLES actions on on-ramp CAVs are acceptable;
do not introduce a mainline-only damping restriction. Corridor length is a
modifiable scenario design choice, not an immutable reason to abandon a wall.
If insufficient road or premature network termination is the demonstrated
limitation, a separately versioned local network extension is an available
repair. Preserve the shared network and existing results; compare damping and
wall on the identical extended scenario with adequate full-service accounting.
Do not attribute downstream traffic constraints to map length, or assume an
extension establishes benefit. This clarification does not itself start a map
edit, new simulation block, or parameter search.

Owner clarification, 2026-09-06: the contribution is conditional improvement
over a fixed external damping controller using CAV-observable RSU escalation.
A wall may help in free-flow, oscillating or congested traffic. Unresolved
congestion, density thresholds and poor damping performance are not entry
requirements. High-density limitations are a possible finding, not an assumption.

Harm localization and target-speed investigation (2026-09-07, complete): the
[consolidated report](../output/wall_harm_speed_v1/REPORT.md) explains the
4107 duration tradeoff (a lane-acceptance cascade at 637.5 s splits a
congested-attractor pair worth 28.03 s/request), localizes the shared 4108
harm to the assembly/HOLD slowdown dose (merge-approach phase shift and later
queue discharge, not direct braking), aligns both mechanisms with acquired CAV
reports (the downstream queue was measured age 0 before every decision; the
exit feasibility wrongly assumes the 24 m/s wall speed after handback), audits
target-speed coverage (the 24 m/s cap binds in every realized engagement; no
historical study tested this comparison), and executes the one registered
alternative, the uncapped mean-minus-2 rule. Result: in 4108 the wall becomes
indistinguishable from damping (its entire harm was the cap-forced 5.4 m/s
slowdown); in 4107 the harmful realization reproduces within 2 s per vehicle.
Seven delivered wall traces across two exposed seeds now include zero
qualifying benefits. New budget actually used: 5040 simulated seconds
including one preserved mechanics failure. Retaining CIRCLES damping remains
the only justified controller behavior; further speed/duration/braking search
is not justified by this evidence.

Scoped correction (2026-09-07, experiment-history maintenance; no successor
campaign authorized or implied): in the paragraph above, "seven delivered wall
traces" counts the lifecycle/cohort/duration/entry-target subset — six
delivered walls plus the B4108 refusal, which executed no wall; the
natural-panel attempts, gentle-completion episodes and brake/HOLD cells are
separate CIRCLES-era traces outside that count (all also non-qualifying). "The
wall becomes indistinguishable from damping" and "its entire harm was the
cap-forced slowdown" are corrected by the verified corrections referenced
below: the higher-target 4108 wall is nearly neutral in aggregate, not an
exact no-op — braking is a slight improvement (−0.000765 vehicle-seconds per
request), effort and TTC worsen slightly so strict joint non-deterioration
fails, and the entry-target comparison changed assembly and handback together
with maintained speed (a full-maneuver contrast, not a pure slowdown-dose
experiment). The master, versioned evidence record is
[protocols/experiment_history.json](../protocols/experiment_history.json) with
[docs/EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md);
[docs/EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md) is a secondary index.

Observation-to-action investigation (2026-09-07, complete): the
[consolidated report](../output/observation_decision_v1/REPORT.md) records
verified corrections to the target-speed prose (near-neutral, not a no-op;
strict joint failing; trace count six delivered walls plus one refusal across
two seed groups), fixes the two-decision contract with its label vector and
evidence taxonomy, inventories every candidate observation (acquired /
state-informative / feasibility-informative / outcome-associated /
action-validated — nothing reaches the last level), and runs the registered
same-prefix continuation contrast: lowering the maintained target to 22 m/s at
HOLD+5 is deliverable but harmful in both seeds (4107 improves service versus
damping by 1.56 s/request yet worsens effort and TTC; 4108 worsens all four).
The first +2 m/s arm failed on an implementation defect, reclassified and
repaired under hold_target_adjust/2: the delivered 24-to-26 m/s continuations
give the best 4107 service result measured (minus 8.44 s/request versus
damping) while worsening effort/TTC/wait beyond guardrails, and match keep-24
in 4108; both remain harmful. Original-block charge 7320 of 9840 s plus a
separate 5040 s repair budget at its maximum; eight delivered same-prefix
continuation actions, two seed groups, zero qualifying. Corrected dataset,
reports and superseded v1 files are versioned under
output/observation_decision_v1/ (read REPORT_V2 and EVIDENCE_CORRECTIONS_V2).
Retaining CIRCLES damping remains the only policy consistent with the original
criteria across all measured arms; [docs/EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md)
indexes every block with reconciled charges. A zero-simulation TTC diagnosis (output/ttc_diagnosis_v1/REPORT_TTC_V3.md) localizes the raise-26 arm's remaining TTC excess entirely to post-release lane-allocation chains (703/707 family), with acquired queue-formation reports visible only 4-9 s ahead — after the last controllable decision; no intervention is currently supported. A v4 audit corrected the phase accounting (coordination 38/30.5 veh-s; pre-entry 477/412 identical) and withdrew the 4-9 s warning claim (fixed-definition leads 0-3.5 s; unimplemented fusion candidate ~4.5 s) - see output/ttc_diagnosis_v1/REPORT_TTC_V4.md. The v5 fusion test (output/ttc_diagnosis_v1/REPORT_TTC_V5.md) found a retrospectively motivated multi-CAV candidate (ahead CAV hard-brake + same-lane approacher) with earlier leads than the follower-front rule (1-5 s vs 0.8-3.4 s on paired events) at far lower false-positive cost but much lower recall; exploratory signal only, decisions untested. The v6 repair closed the fusion block (output/ttc_diagnosis_v1/REPORT_TTC_V6.md: earlier leads confirmed, coverage penetration-limited) and the joint reassessment (output/joint_design_v1/DECISION.md) proposes the smallest next experiment: one registered continue/release decision at HOLD+20 (delivered 30 s wall), 5040 s, not executed. The hold-extension experiment was then implemented and executed (output/hold_extend_v1/REPORT.md; 6360 s charged vs 4920 authorized, overrun disclosed): 4107 extension aborted protectively at +6 s (lost compact hold; TTC flipped to improvement but 2 new collisions; harmful), 4108 extension inert (identical to raise-H20; harmful). No escalation qualifies; adaptive duration has no supporting evidence in these cases. A repair pass corrected the report's signs (4107-ext: service/braking/effort/wait improve vs damping; TTC +0.020141 fails by 0.000141; 2 collisions fail), diagnosed the vacuous v1 admission (it evaluated the H20 final tick, not the H30 commitment; downstream per-tick checks kept the run bounded), repaired admission as hold_extend/2 (replay accepts both branches), split physical evidence from contract validity, and added persistent budget-guard enforcement (output/hold_extend_v1/REPORT_V2.md). The fusion evaluation was corrected and closed at v7 (output/ttc_diagnosis_v1/REPORT_TTC_V7.md: coverage/timing repairs; M earlier than F on 56/69 strict-advance pairs at 3-11% coverage). The selection-potential oracle shows no qualifying continuation action in either seed, so the next block is an action repair: extend the raise-26 history by +10 s at HOLD+20 (output/joint_design_v1/NEXT_BLOCK_PROPOSAL_V2.md, 4920 s, not executed).

Active owner goal (2026-09-06): execute the history-informed route and find a
physically useful escalation without repeating answered searches. The
[bounded execution contract](../archive/circles_era/docs/OBSERVER_REPLAY_EXECUTION.md) specifies observer
implementation, five exact explanatory replays, and the evidence required for
one separately registered repair. No efficacy success is assumed.

The five explanatory replays and report-aware dissolution check are complete.
The latest owner-authorized step is the [gentle wall check](../archive/circles_era/docs/GENTLE_WALL_CHECK.md):
one wall-only implementation, at most 240 simulated seconds of controlled
mechanics, then two 2400-second existing development cases with verified reused
damping. Saved-trajectory diagnosis accompanies this block. All original benefit
guardrails remain fixed; no automatic variant, fresh-seed expansion or training.

The owner subsequently authorized the separately registered replacement of the
two logging-failed corridor cases. Both full episodes completed: neither gentle
wall meets the unchanged criteria or completes its registered lifecycle. The
[completion report](../output/gentle_wall_completion_v1/REPORT.md) records actual
commands, handback, full physical outcomes and saved-trajectory diagnosis. The
[original stopped block](../output/gentle_wall_v1/REPORT.md) remains intact.

The owner-authorized [braking-limit and HOLD-duration test](../archive/circles_era/docs/BRAKE_HOLD_TEST.md)
is complete; the [report](../output/brake_hold_v1/REPORT.md) records all 12 cells,
eight new full episodes and two mechanics fixtures. Holding lifecycle logic fixed,
the -0.3/-3.0 by H0/H20/H40 factorial finds no qualifying setting. HOLD effects
are large but depend on the braking contract in 4107; 4108 aborts before HOLD.
Neither nominal H40 completes, and -0.3 is neither a guaranteed measured floor
nor consistently better. Do not repeat this factorial or infer an optimal duration.
Only 4107/strong/H20 completes its registered maneuver, and it still fails braking
and TTC guardrails. No successor sweep, fresh seeds or training was started.

The subsequent design question is acquired-report feasibility of the entire maneuver,
including the acceleration ramp, formation, HOLD and exit. A command cap alone
did not remove downstream harms. The saved report audit finds that the current
gentle downstream check already fails at entry in 4108 and at HOLD entry in 4107,
but runtime consults it only at taper. The separately logged ramp estimate does
not enter the reused runway assessment. Reconcile these lifecycle checks before
a repair-specific physical comparison; the conservative distance estimate is not
a validated safety proof or benefit predictor. Seed 4108 never reached HOLD;
the completed factorial now proves same-cap actuator equivalence across all three
durations. Reuse that result and the measured 4107 duration contrasts. Do not start
another full run, train a gate, broaden seeds
or tune thresholds automatically. Any repair must first address a demonstrated
mechanism and pass a separately budgeted mechanics check; feasibility alone is
not incremental benefit and does not qualify escalation.

History review (2026-09-06): the completed natural panel does not qualify wall
benefit. Before fresh-seed replication, complete the current mechanism-to-CAV-
information audit described in the [history-informed recommendation](../output/history_route_audit/DECISION.md).
It reuses prior work and limits the first block to five observer-only replays of
existing arms after an exact registration. Historical
metering gains do not validate the current moving formation. The scientific
sequence below remains the route after its actual prerequisites pass.

The working controller remains adopted and frozen. Implementation validity is
required; standalone superiority to IDM and elimination of congestion are not
prerequisites for wall research. The [ordered tests](../archive/circles_era/docs/ESCALATION_TESTS.md) and
[machine-readable catalog](../protocols/escalation_test_plan_v1.json) specify
the complete construction sequence. No damping retraining is part of this route.

## 1. Register fixed-baseline characterization and opportunity capture

Before running, create one machine-readable protocol with exact model/adapter,
source and scenario hashes; named development cases and exclusions; nominal
penetrations 5%, 10%, 20%, 30%; demand scope; paired native arms; repetitions;
physical metrics; tolerances; stopping rules; maximum trajectory and retry budget.
Use multiple development seeds at each demand, rather than presenting the two
existing single-seed outcomes as population estimates. Do not silently change
the 1 s / 0.5 s / 100 m tested interface or select tolerances to approve its
already observed service/TTC costs.

Specify braking and effort benefit alongside total requested service, source
waiting, TTC, collisions, teleports and incomplete requests. Keep both mean
paired effects and per-case failures. Choose absolute tolerances where relative
changes have unstable small denominators. This document deliberately does not
invent numerical acceptance tolerances or claim their approval; the registered
protocol is the concrete next deliverable.

Audit seed exposure using the archive manifest and seed history. Registry absence
does not prove freshness. The current CLI allows previously exposed/development
cases and blocks sealed acceptance; opening a new prospective block requires
an explicit audited registration and a reviewed execution path, not bypassing
the guard. Acceptance outcomes remain sealed throughout development.

The full evaluation will use the supported `evaluate` command with one new output
directory per arm. Reuse exact native results only after matching all scenario,
simulation and accounting identities; otherwise budget a matched reference.
Report the fixed controller's cross-penetration result honestly. A weak damping
result characterizes the counterfactual rather than rejecting it or authorizing
another learner. A good result does not rule out wall benefit. Repair invalid
implementation/replay, but do not require damping to pass an efficacy gate.
Record measured traffic density separately from nominal demand and keep all
requested service, including source waiting and incomplete trips.

## 2. Establish wall benefit over this same damping baseline

After implementation integrity passes, register a small treatment experiment with
the identical background controller and explicit wall geometry, membership,
assembly/hold/taper/release, safety envelope and abort contract. Establish
feasibility, actual command ownership, and enough physical benefit to justify
prediction. Compare continuing damping with the proposed wall using matched
exogenous futures and the same full service horizon. Characterization can share
these damping control arms. Release must return selected CAVs to the unchanged
CIRCLES controller. There is no congestion filter: test benefits in safety,
smoothness or service at any traffic state under predeclared cost guardrails.
Distinguish initial candidate separation from formed-wall compactness; the old
50 m threshold is a hypothesis, not a demonstrated physical feasibility boundary.

The existing report-only wall interface is a mechanics boundary, not an accepted
treatment. Historical wall labels came from different controllers/information
contracts and cannot be reused as if generated under this baseline.

## 3. Learn CAV-observable escalation only after treatment headroom exists

Each CAV reports its acquired self state and detected neighbors to the RSU.
Build histories with timestamps, coverage, receipt age and missingness. Separate
"not enough members," "unsafe observed geometry" and "not enough information."
Lower penetration affects both available control members and information;
true penetration is not an online feature or an arbitrary wall threshold.

The feasibility gate and incremental-benefit gate must approve the same concrete
wall proposal. Offline paired outcomes may label the difference between a wall
and continued damping; global outcomes and privileged shadows never enter online
observations. Validate feature acquisition, missingness and delayed reporting
before fitting. Then register separate training, calibration and prospective
partitions and compare against always damping and transparent analytical
escalation rules under the same action opportunities and budgets.

Only held-out physical intervention outcomes support the paper's claim that
report-based escalation improves traffic over the adopted damping baseline.
Reward curves, fit accuracy alone, or old wall results do not establish that claim.

<!-- OE34-traffic-question -->
OE-34 supplementary [traffic-state review](../output/observation_relational_v1/review/COLLAPSE_QUESTION_V1.md): saved damping histories show low merge speeds before entry600; collapse incidence/onset and prevention efficacy remain unmeasured. Zero new simulation; full study close-out remains in progress.


<!-- MIG02-status -->
**MIG-02 (2026-09-17):** [Source-isolation verification](../output/research_removal_v1/REPORT.md) passes: 568 local asset hashes, 1020 tests, exact saved replay and archive path/size checks. No further dependencies needed copying. Automatic approval review blocked deletion; all D:/research contents restored, old folder NOT deleted. Scientific route unchanged.
<!-- WM39-mix-retirement -->
WM-39 complete (2026-09-25, zero simulation): the per-lane/per-cell mix layer (traffic_shepherd/spatial_assignment.py, spatial_pipeline.py, spatial_grid.py + their tests) is RETIRED by owner decision and removed (hash-verified backup: D:/research-dense-backups/mix_removal_20260925/; registered in protocols/deleted_evidence.json as NOT MIG-04). Action model is now explicit: actions are chosen PER CAV (native / damping / PI external-U; heterogeneous assignment REQUIRED), and the wall (wall-v2) is a GROUP action whose members are exclusively owned by the wall session from assembly through hold to handback while non-members stay free. mixed_assignment.py is kept (dwell/commitment/native-fallback bookkeeping; the WM-13 starvation fix and the subset-roster wall dispatcher are registered next steps, WM-40). Retired banners added to the WM-15 spatial docs and output/worldmodel_v3_planner/README.md. Suite after: 523 passed / 2 failed (both ledger tests on the WM-38+39 deletions, planner integrates) / 156 skipped; ruff clean.

<!-- WM38-fs-retirement -->
WM-38 complete (2026-09-25, zero simulation): FollowerStopper retired 2026-09-25 by owner decision and removed from the active code (analytical_candidate/analytical_mechanics/fs_close_mechanics, the follower-stopper branches in controller_factory/workflow/mechanics/__main__, and the FS test files); per-CAV actions are now native, damping, PI external-U (wall-v2 stays a gated group action). CV-03..CV-06 records remain historical evidence. Hashed backup: D:/research-dense-backups/fs_removal_20260925/. Full active suite after removal: 601 passed / 1 failed (pre-existing test_evidence_regeneration_v1 failure) / 157 skipped; ruff clean.

