# Project status log

Chronological status blocks formerly at the top of README.md (moved by MIG-03, 2026-09-24).

<!-- WM19-reviewed -->
**WM-19 complete (2026-09-24):** 21 radius-input exports independently verified; 16 adapter tests pass, including actual synthetic mass and hidden-control separation. [Reviewed report](../output/worldmodel_radius_input_v1/REPORT_FINAL.md). No real forecasts/training or WM-16 resumption. Remaining priority: observed/prior speed fusion hides queues (WM-18 occluded detects 1/5). One saved-grid prior-speed test is proposed, not executed; action-response dynamics remain unvalidated. This supersedes earlier WM-19 in-progress statements below.
<!-- /WM19-reviewed -->

<!-- WM18-current -->
WM-18 complete:50m cross-lane multi-neighbor sensing materially improves reconstruction on21 exposed worlds. Occluded countMAE -42.9%, speedRMSE -34.4%, mean absolute total-count error -30.6%;19 completed launches+2 independently recovered outputs. All negatives preserved. **Queue detection remains poor (1/5)**: guessed fast prior mass contaminates slow observed traffic. Historical WM17 truth omitted some no-report lanes; complete-truth scores and exact legacy reconciliation are now separate. [Current final report](../output/radius_sensor_v1/REPORT_FINAL.md).

WM-19 is in progress: owner-authorized conditional world-model input integration using the improved reconstructor, preserving observed/prior provenance. Synthetic mechanics and21 artifact-only exports; no real-data rollout, training, SUMO or sealed data. WM-16 remains paused. This does not declare reliable congestion-state estimation or action-benefit prediction.

<!-- WM17-current-state -->
WM-17 completed its bounded CAV-only t=600 state-constructor comparison on 25 exposed development worlds (zero SUMO/fitting). Motion/neighbor constraints modestly improve 10 m placement and speed, but all arms inherit the wrong 50 m count total, nominal count intervals are uncalibrated, and five slow-queue boundaries are missed. This is diagnostic state evidence, not controller selection or deployment evidence. See [WM-17 close-out](../output/state_constructor_v1/WM17_CLOSEOUT_REPORT.md); the older WM-15 status below is historical architecture context.

<!-- WM15-spatial-correction -->
WM-15 is the current architecture: a shared 10 m × four-group grid for receipt-causal CAV reconstruction, action-conditioned grid prediction, spatial cell→controller proposals and per-CAV dispatch. The [first-milestone report](../output/worldmodel_spatial_v1/FIRST_MILESTONE_REPORT.md) records a working synthetic structural path and the measured bottleneck: saved runs lack full vehicle positions after the 600 s action boundary. No spatial forecast accuracy or selection benefit is validated. The [v1 contract](../output/worldmodel_spatial_v1/SPATIAL_CONTRACT_V1.md) fixes geometry and causal mechanics; WM-14 uniform-arm evaluations remain baseline diagnostics and its draft fitting allocation is unspent. Wall execution remains separate.

<!-- WM14-spatial-supersession -->
WM-14 stopped as a uniform-arm baseline/component program after the spatial correction. Its draft fit allocation was never authorized or spent. Repaired source pins and 150 passing focused tests are preserved; [scope close-out](../output/worldmodel_repair_wm14/SPATIAL_SUPERSESSION.md).

<!-- WM13-review -->
WM-13 independent review qualifies WM-12: useful prototype, but no validated closed-loop safety/selection. Reproduced dispatcher stable-request starvation; planner later-tick labels are static600s outcomes; reconstruction headline is oraclebest-head and sensorablation has unequal membercounts. Next: offline repairs and one same-state operational benchmark before expanded mixed/wall campaigns. No code patch/simulation/training. [Review](../output/worldmodel_review_wm13/REPORT.md).

<!-- WM12-status -->
WM-12 Phase A complete (2026-09-23, zero simulation, 8 workstreams via parallel subagents): every component of the outcome-scored architecture now has acceptance evidence. Per-vehicle car-following dynamics PASSED its gate (2.17 vs 2.33 advected persistence; bounded 60-step rollout); post-switch arm-aware dynamics proved controller identity INFORMATIVE (0.1228 vs 0.1364 m/s, p=0.0002); reconstructor v5 beat v4 on every stored metric (RMSE 2.27, ramp 1.55 via the owner-authorized RSU ramp sensor, ablation-proven -39%); horizon study proved 60 s scoring INSUFFICIENT (top-1 0.56 vs 0.72-0.76 at 600 s) -> planner scoring moved to 600 s with ties-default-native; wall audits: history multiplies episodes ~30x (era-tagged) but wall-v2 engagement is 380 vehicle-seconds and assembly dv needs command/owner inputs - Phase-B wall matrix specified (12 runs); dispatcher built and unit-tested (13 tests) with wall-subset execution stubbed pending a subset-roster wall session. PLANNER VERDICT: SAFE but NOT ADDITIVE - raw imagined argmax mis-ranks vs measured outcomes (18/54 holdout recovery); the tie-to-native rule recovers exactly always-native performance (regret 0.19; fleet 0.534 vs 0.509); 30,566 labeled decisions generated for a future learned proposer. Registered next steps: Phase-B mixed-assignment runs (per-CAV effects are extrapolation until measured), Gate-3 coverage-input fix (degenerate regularizer input identified), command-aware wall features. [Reports](../output/worldmodel_v3_planner/RESULTS.json).

<!-- CV313210-status -->
CV-31 + CV-32 + WM-10 completed (2026-09-22): the coverage extension delivered 15 accepted arms across five fresh cells (7533 safety-closed by a retain-arm collision, evidence preserved) plus the first five layered-verified wall arms on the fixed hold-phase supervisor (zero collisions). New structure discovered: the first retain-best cell ever (7535m2400 - damping wins at mid-demand), and opposite native/PI winners at matched m2700 demand (7531 PI-best vs 7534 native-best). Wall verdict: operable and safe; fully engaged at 7531 and lost to all three controllers; gate correctly suppressed entry at 7535; never beat a cells best controller, never hurt vs retained damping. Selector verdict (WM-10): the enlarged corpus did NOT fix generalization - a fresh never-trained congested cell (7534) still picked PI on 91% of windows (+7.6 s/veh regret), so the native-vs-PI moderator is NOT identifiable from the current lane-split state features at this scale; more same-kind seeds is ruled out as the fix and the registered next step is moderator identification. Always-native remains the correct deployment policy (+0.38 vs selector +7.14 s/veh mean holdout regret). [Reports](../output/worldmodel_v2_full/REHEARSAL_REPORT.md).

<!-- CV31-status -->
CV-31 completed: 15 of 18 full-horizon coverage arms (cells 7531-7534 at m2700, 7535-7536 at m2400, three arms each); 38400 s charged / 38400.0 s actual; per-cell 600 s prefixes verified by the reviewed CV31 oracle; closed cells preserved with their charges. Sign-consistency over cells with all three arms valid (5/6): native beat retained CIRCLES on all five costs in 4; PI beat retained CIRCLES on all five in 3; PI beat native on all five in 0. Per-request service deltas, treatment minus reference (negative = improvement): 7531m2700 native-retain -1.203252, pi-retain -5.022173 | 7532m2700 native-retain -18.432568, pi-retain -17.740674 | 7534m2700 native-retain -20.746110, pi-retain -12.935644 | 7535m2400 native-retain +6.086056, pi-retain +3.524303 | 7536m2400 native-retain -1.248216, pi-retain +0.241475. One scene per cell; no retry, fitting or acceptance access; no switching rule, generalization or world-model claim. [Report](../output/controller_verification_v31/REPORT.md).

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

> Current update CV-16: offline bounded budget-write repair verified (32 scoped tests); no simulation. CV15 remains stopped, 2400s charged/190.5s reached. Controller switching benefit remains untested. See output/controller_verification_v16/REPORT.md.

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
**Active owner goal (2026-09-20):** [Uncertainty-aware traffic regulation from sparse CAV observations](../docs/WORLD_MODEL_RESEARCH_GOAL.md). Verify literature-supported controllers first, then establish state-dependent control opportunity before world-model learning. Walls are optional. Astra plans/reviews; Luna executes bounded tasks. This direction supersedes wall-only planning below; frozen baselines, evidence and safety/information constraints remain preserved. CV-01 initial audit and isolated FollowerStopper equation verification are complete (zero simulation/fitting); full controller verification is ongoing.
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
**OE-51 completed (2026-09-12):** [Early-H5 execution package](../output/oe51_integration_v1/REPORT.md). Mechanics PASSED (canonical v3 17/17, R17 equality VERIFIED) on ONE live 240 s allocation; three early-H5 corridor arms (90026 t*=240, 90031 t*=246, 90037 t*=240) executed to full horizon: ALL THREE HARMFUL under unchanged criteria. 90026: system +4.27, braking -0.124, effort +4.96, TTC -0.368; 90031 (protective abort 9 s, HOLD never reached): system +12.16, braking -0.186, effort -8.64; 90037: system -12.57 PRESERVED AND IMPROVED vs H5-at-600 (-4.05) but effort +7.42 violated (label harmful). Prefix verification passed 3/3; onsets verified; corrected influence margin 156/222.5/151.5 s. Act-earlier timing route CLOSED. Zero simulation beyond authorized blocks; OE-49 STOP and 240 s preserved; oe51_early_h5 mechanics STOP preserved; corridors under successor identity oe51_early_h5_corridors.
<!-- OE51-current-end -->

<!-- OE52-current-start -->
**OE-52 completed (2026-09-12):** [OE-51 numerical/mechanism repair and acceleration-effort diagnosis](docs/oe52_integration_v1/REPORT.md if false else 'output/oe52_integration_v1/REPORT.md'). Three parallel offline branches (zero simulation). Vectors/grading: all 30 cross-check numbers exact at 6 dp; frozen grading reproduced; interval reconciliation 0.0; corrected useful_intervention for 90031 flips True->False (saved OR semantics vs embedded BOTH-references definition; corrected: delivered but not useful). Timing: all six recorded onsets verified correct under frozen definitions; occupancy onset = whole-pre-merge-edge scalar (2,644 m edge), not a band; first spatial signature upstream in the 600-800 m bin 39-63 s before region onsets; slow-vehicle counts identical in both arms at every onset (one delayed non-cohort follower differs); OE-35 local queue candidates identical (merge queue pre-existing); corrected influence-before-development margin 156/222.5/151.5 s. Effort: formula = signed measured a^2 x 0.5 s over ALL live vehicles (HDVs included); reconciliation 7.1e-9; wall episode 0.06-0.23% of each difference; 99.8%+ post-handback, diffuse over 1200+ vehicles; balanced speed-loss/recovery oscillation; 90031 saving is post-abort divergence. All OE-51 physical conclusions survive; act-earlier route remains CLOSED. Zero simulation.
<!-- OE52-current-end -->

# Traffic Shepherd

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


Latest implementation (OE-28): [formation recovery release](../output/recovery_release_v1/REPORT.md). The120s v1 mechanics run failed recovery delivery and safely coasted to handback; v2 is repaired offline, not physically validated. Redundant corridor runs were omitted. No controller-benefit or predictor claim follows.

This project studies when an RSU should escalate from decentralized CAV damping
to a temporary coordinated moving wall at an IDM highway merge. The intended
paper contribution is escalation based on information acquired and reported by
CAVs. We use an externally released, frozen CIRCLES controller as the **working
damping baseline**. Scientific acceptance of this transfer remains outstanding.

Start with [current status](../docs/STATUS.md), the [architecture](../docs/ARCHITECTURE.md),
and the single [scientific route](../docs/ROUTE.md). Earlier learners, negative
results and superseded directions are in the [research archive](../archive/pre_circles/ARCHIVE_INDEX.md).

Latest (2026-09-10): [OE-27](../output/duration_selection_v11/REPORT.md) repairs
OE-26's policy accounting and completes offline verification of the six OE-25
instrumented arms. CAV reports contain short-horizon collision-associated warnings,
but successful warnings in wall arms occur after handback and do not establish
entry/duration selection. Existing H5 evidence contains qualifying cases, while
the two fitted entry selectors still decline every labeled case. No deployable
benefit gate or new intervention is validated. This block used zero new simulation;
training remains paused and acceptance data sealed. Use the
[current route](../docs/ROUTE.md), [open questions](../docs/OPEN_QUESTIONS.md), and
[master ledger](../protocols/experiment_history.json) before planning further work.

The [ordered escalation test plan](../archive/circles_era/docs/ESCALATION_TESTS.md) covers implementation,
spatial formation, wall effects, both gates and prospective evaluation. A wall
may help at any traffic state; unresolved congestion and standalone damping
superiority are not prerequisites. `python -m traffic_shepherd.escalation_study plan`
prints the full test catalog without launching experiments.

A searchable record of everything already tested — results, negative evidence,
corrections, and open questions — is maintained in the
[experiment history](../docs/EXPERIMENT_HISTORY.md) with the machine-readable ledger
[protocols/experiment_history.json](../protocols/experiment_history.json),
[open questions](../docs/OPEN_QUESTIONS.md), and the required
[experiment record template](../docs/EXPERIMENT_RECORD_TEMPLATE.md). Consult these
before proposing any new simulation, and register new experiments in the ledger.
Read-only lookup and consistency validation (never reads sealed outcomes):
`python tools/experiment_history.py search|card|closest|open-questions|pending|summary|validate`.
`verify-sources` additionally compares stored current-era numbers field-by-field
against the registered allowed source analyses (report:
protocols/source_verification_report.json).

## Setup

Use Python 3.10+ and SUMO **1.27.1**, matching the adopted TraCI version. The
existing `rl_driving` environment already contains the verified dependencies.
For a new environment, install `requirements-dev.txt`; active execution needs
NumPy, ONNX Runtime and TraCI, with no RLlib or custom learner dependency.

```powershell
Set-Location D:\research-dense
conda activate rl_driving
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
python -m pip install -r requirements-dev.txt
python -m traffic_shepherd inspect
python -m pytest -q
python -m ruff check traffic_shepherd tests
```

The single configuration is [baselines/circles/adapter.json](../baselines/circles/adapter.json).
It references the byte-identical local network at
`assets/research/highway_merge_shepherd_long/osm.net.xml.gz` and the preserved local
[IDM route/type template](../scenarios/idm/driver_routes.rou.xml). Active execution no
longer needs `D:/research`. MIG-01 preserved required scenario inputs and archived
baseline dependencies locally; see [verification and historical replay paths](../docs/RESEARCH_ASSET_MIGRATION.md).
The map bytes, frozen controller, physical criteria, and old evidence are unchanged.

## Supported entry points

Read-only identity, import, documentation and saved-observation checks:

```powershell
python -m traffic_shepherd inspect
python -m traffic_shepherd.verification --saved-trajectories --archive
```

Add `--archive-hashes` to independently rehash every archived file (about 21 GB).
Ordinary tests do not run SUMO or read sealed acceptance outcomes.

One explicit development evaluation, **only after its evaluation scope is
registered**; these example commands are not an instruction to start a campaign:

```powershell
python -m traffic_shepherd evaluate --seed 911001 --loading 2000 --penetration 0.2 --role development --controller native --output runs/active/native_s911001_p20
python -m traffic_shepherd evaluate --seed 911001 --loading 2000 --penetration 0.2 --role development --controller damping --output runs/active/damping_s911001_p20
```

Both use the fixed 2,400-second horizon, identical exogenous requests and IDM
driver types. `--duration 30` explicitly marks a run as mechanics-only. Existing
output directories are rejected; there is no implicit resume, retry, training,
wall-generation or acceptance command. Use `--help` for all arguments.

The narrowly bounded actuator check has its own entry point:

```powershell
python -m traffic_shepherd.mechanics --output runs/active/ownership_mechanics_new
```

It runs exactly two 10-second synthetic IDM trajectories. The completed cleanup
evidence is linked in [status](../docs/STATUS.md); rerunning is unnecessary unless
the actuator changes.

## Active structure

- `traffic_shepherd/`: sensing, frozen damping, sole command owner, report-only
  RSU interface, provisional wall mechanics, scenarios, evaluation and verification.
- `baselines/circles/`: exact ONNX, MIT license, provenance, adapter and saved replay rows.
- `scenarios/idm/`: preserved IDM type and route definitions.
- `tests/`: active contracts and ported infrastructure regressions.
- `docs/`: architecture, status and one current scientific route.
- `runs/active/`: all new execution artifacts; `output/adoption_cleanup/`: cleanup audit.
- `archive/pre_circles/`: historical code, 1,146 original tests, reports and raw results.

The baseline passes a singleton raw SI `[ego speed, leader speed, bumper gap]`
to ONNX; embedded normalization and the full rule wrapper remain unchanged.
Its output is absolute acceleration, applied at 1 s with 0.5 s physics, 100 m
same-lane sensing and SUMO speedMode31. Native fallback releases the override.
There is no approximate IDM addition or old tight jerk/front/rear projection.

The direct controller reduced severe braking in two exposed p20 cases, with
service and TTC trade-offs. Direct performance at other penetrations remains
untested. No accepted wall, deployable predictor or exact match to a paper's
evaluated checkpoint is implied. Bespoke damping training and wall/predictor
generation remain paused.


<!-- MIG02-status -->
**MIG-02 (2026-09-17):** [Source-isolation verification](../output/research_removal_v1/REPORT.md) passes: 568 local asset hashes, 1020 tests, exact saved replay and archive path/size checks. No further dependencies needed copying. Automatic approval review blocked deletion; all D:/research contents restored, old folder NOT deleted. Scientific route unchanged.
