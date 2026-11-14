<!-- WM39-action-model -->
WM-39 (2026-09-25, owner decision, zero simulation): the per-CAV action model is current policy - actions chosen per CAV (native / damping / PI external-U, heterogeneous assignment required), wall-v2 as a gated GROUP action with exclusive member ownership for the whole episode; the per-lane/per-cell mix layer is retired and removed (backup D:/research-dense-backups/mix_removal_20260925/).

<!-- WM15-first-milestone -->
WM-15 first structural milestone completed (2026-09-23, zero simulation/fitting): CAV-report grid reconstruction, a topology-corrected diagnostic future-grid model, spatial action maps, and per-CAV dispatch connect in a synthetic-only smoke with safe production abstention. Existing CV-17/19/30/31 records omit full vehicle positions after the 600 s action boundary, preventing forecast-error and calibration assessment on the intended target; mixed spatial physical outcomes are also absent. This is a working interface, not a validated traffic selector. [Milestone report](../output/worldmodel_spatial_v1/FIRST_MILESTONE_REPORT.md).

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

# Active research goal — owner decision, 2026-09-20

Build uncertainty-aware highway traffic regulation from sparse CAV observations.
The research question is whether planning over plausible unobserved traffic
states improves physical intervention selection compared with committing to one
reconstructed state. This is a hypothesis, not an established novelty or result.

This owner-requested direction supersedes the requirement that the contribution
must be a moving wall. Preserve the historical CIRCLES baseline and all evidence;
a wall is optional and must earn inclusion through physical evidence. Do not
resume bespoke damping learning or use acceptance outcomes for development.

## Ordered work

1. Verify a bounded shortlist of literature-supported controllers. Separate
   publication evidence, implementation fidelity, original-benchmark replication,
   and efficacy on this project's merge. Begin with exact CIRCLES provenance and
   interface audit, faithful FollowerStopper/PI-saturation candidate screening,
   and target-speed tracking suitability. Review moving-bottleneck literature
   conditionally; do not repeat answered wall parameter sweeps.
2. Establish useful control authority and state-dependent action headroom with
   matched physical outcomes. Compare native IDM and the frozen CIRCLES baseline;
   register each candidate version, scenario, metrics and finite budget first.
3. Freeze the selected controller and action contract. Define CAV-report-only
   histories, coverage, receipt ages and missingness. Inferred states remain
   explicitly probabilistic; simulator truth is offline supervision only.
4. Develop belief reconstruction and action-conditioned stochastic prediction.
   Distinguish hidden-state uncertainty, future driver variability and model
   uncertainty; ensemble disagreement alone is not calibrated breakdown risk.
5. Compare fixed/reactive/direct-predictor, deterministic model and probabilistic
   model planning under identical controller, actions, information and budgets.
   Validate on independently registered scenarios; preserve service, source wait,
   braking, effort, TTC, collisions, teleports and unfinished requests. Assess
   calibration and action regret as well as full-horizon physical outcomes.

## Execution and evidence

Owner update (2026-09-23): use GPT-6 Sol for plans and independent review;
ZCode GLM-5.3-Flash for bounded implementation. Verify the actual executor model
from request/response logs. Delegate independent tasks with disjoint files within actual host
limits. The integrating agent owns budgets, shared records and acceptance gates.

The owner authorizes progress through this staged goal. Before each simulation
block, freeze a small explicit budget, source identities and stopping rules;
record failures and prohibit silent retries. No unlimited overnight sweep is
implied. Do not stop merely to seek repeated permission for already authorized
work; do stop dependent stages when their scientific prerequisites fail.

Current stage: controller evidence/integration audit, stable task CV-01, output
`output/controller_verification_v1/`. Initial allocation is zero simulator seconds
and zero fitting. Controller efficacy and world-model necessity remain unproven.

Planner deliverable: [Astra controller verification plan](../output/controller_verification_v1/ASTRA_PLAN.md).
Proposed ceilings in that plan are planning limits, not launched allocations.
Original-benchmark replication is recorded separately from equation tests and
merge transfer. A missing publication-to-checkpoint bridge does not prevent
testing a transparently identified released artifact, but forbids calling it a
replication of that publication's result.

Useful action headroom and usable causal sensing are prerequisites for the
world-model stage; an already successful observation-based benefit classifier
is not required. Whether belief reconstruction improves decisions is itself
the hypothesis to test. Uncertainty-aware planning must earn its complexity
against simpler methods and may fail honestly.

CV-01 initial-stage close-out: [report](../output/controller_verification_v1/REPORT.md). Next: source-complete PI/target interface and bounded mechanics, not world-model training yet.

CV-02 completed: [source and opt-in adapter report](../output/controller_verification_v2/REPORT.md). Current next stage is a registered 20-second matched FS mechanics check; PI highway estimator and lifecycle remain open.

CV-03 completed: [live FS mechanics](../output/controller_verification_v3/REPORT.md),20s charged/actual,13/13 endpoints and real handback verified. Next: narrow close-leader mechanics and PI highway estimator/lifecycle contract, then matched efficacy characterization. The main goal remains active; no world-model fit yet.
