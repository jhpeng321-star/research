<!-- C4-CLOSEOUT-20261106 -->
**C4 close-out of Nov 4-6 (recorded 2026-11-06; zero new simulation by the integrator):** [sources receipt](../runs/active/C4-V7FT-20261017/C4_CLOSEOUT_20261106_SOURCES.json). **CLEAN1-20261105** stopped at sealed A9/B9/R9 after R u3 and B u9 composite safety pauses (0 collisions) and the owner restart order; 1,214,880 sim-s; u6 B synergy +0.043 CI95 [-0.141, +0.293]. **CLEAN2-20261105** stopped at u18 (lesson-phase underquote, then owner stop); 1,849,200 sim-s; u6 B total +726.4 but synergy -0.034, u12 synergy -0.016. **CLEAN3-20261106**: 12 chain attempts (22,800 sim-s pre-launch), root clean3_r6 launched 12:00, faulted at u1 (G4 frozen-runtime binding) and owner-paused; 120,760 charged incl. a 19,200 zero-physics G4 reservation; G1 NO-GO; 1 of 5 mined walls observable. **S-MINING-20261106**: 472,341 sim-s, 0 admitted G5 moves (ramp role redundant with native merging). **COOP-PROBE-20261106**: 240,078 sim-s; lead wall synergy +336.7 on 3/6 seeds; hp 1/6; less-cooperative HDVs none; WM ranked the winners last. **HICAV-PROBE-20261106**: 149,715 sim-s; 0.6/0.8 share no better; admitted walls span 181-253 m. **NEWMAP-PROBE-20261106**: 148,520 sim-s; lane-drop pairs super-additive (8/34 admitted). **UPPER-BOUND-20261106**: 148,995 sim-s; synergy ceiling ~0 (1/12 strict, 0/12 material); held-out single-CAV total benefit mean +200.9 / median +42.1. **WALLMINE-20261106** (in progress): 147,550 sim-s; 1 new admitted B wall. **PLANFIX-20261105** superseded: 33,600 sim-s, 0 survivors, 566,400 released. **S-LANEMAPPO-20261106**: keep speed-only base; aecd7d70 as extra baseline.
<!-- /C4-CLOSEOUT-20261106 -->

<!-- S-GROUPDIAG-closeout -->
**S-GROUPDIAG (resumed 2026-11-06; zero-simulation analysis):** [Report](../runs/active/C4-V7FT-20261017/CLEAN2_GROUP_COOP_RESEARCH.md), [verification receipt](../runs/active/C4-V7FT-20261017/S_GROUPDIAG_RESUME_RECEIPT_20261106.json). B/u6 benefit+726.399612 veh-s equals best-solo+726.433686 minus0.034074 synergy. One joint macro is offered/executed in seven test states; u6 training has96 joint starts/93 physically joint,78 in practice, zero rejected non-none starts. Eight admitted native examples have positive matched-window synergy, but projected macros lack outcome certification; legacy unequal-horizon arithmetic can be misleading. Reward has no explicit second-helper or best-solo credit. Literature and six ranked levers prioritize executable labels, useful role-aware opportunities, complementary pair actions and measured marginal credit. Original cutoff and u24 criterion retained; no new SUMO/training/charges/remote process interaction.
<!-- /S-GROUPDIAG-closeout -->

<!-- S-DIAG-closeout -->
**S-DIAG-CLEAN1-20261105 (resumed 2026-11-06; zero-simulation analysis):** [Report](../runs/active/C4-V7FT-20261017/CLEAN1_DIAGNOSIS_20261105.md), [evidence/verification receipt](../runs/active/C4-V7FT-20261017/S_DIAG_RESUME_RECEIPT_20261106.json). Preserved u1–u8 work; added final u9, for144 B training receipts and54 matched A practice references. None stays near11%; bank CE2.063→0.402 on three contexts; continuous PPO remains below its0.045 KL cap while bank imitation binds. Offset-only checkpointu6 sensitivity on1,263 active rows raises mean acceleration by0.008456 m/s², including the network pathway: weak/reversed instantaneous actuation. Replicated +0.042733 veh-s is extra benefit over best solo, not total base-relative gain; exact zeros can contain accepted moves/nonzero residuals. Matched total delay remains mixed (u8−1.31%,u9+1.73%); no established synergy/generalization. Durable clean1 STOP preserved; causal single-move benefit and exact post-filter CE identities remain unresolved. Zero new SUMO/training/charges/process changes.
<!-- /S-DIAG-closeout -->

<!-- S-WMCHECK-closeout -->
**S-WMCHECK (2026-11-05, zero-simulation analysis):** [Report](../runs/active/C4-V7FT-20261017/WM_ERROR_CHECK_20261105.md), [preserved PASS receipt](../runs/active/C4-V7FT-20261017/S_WMCHECK_RESUME_RECEIPT_20261105.json). Existing cpuQ warm1/all8 verification reused: 542 matched selected-plan pairs, 32 training seeds, 90/221 positive predictions win; median absolute logged gain error 328.77 veh-s, pred-minus-real bias +207.29 veh-s, Spearman 0.103. Raw/calibrated >=80% accuracy and ECE unavailable due to missing paired probabilities. One family, correlated panels, mixed horizons and availability/selection bias; no new SUMO/training/charges. Resume preserved completed work without remote recollection or duplicate dispatch.
<!-- /S-WMCHECK-closeout -->

Label fix + timing audit (2026-10-03): [FIX1](../runs/active/WM-N1-LABEL-TIME-FIX1-20261003/REPORT.md). The switching-run label lookup is fixed (196 chunk2/3 campaign switching items rebound to their own swnative/swpi labels; 0 excluded), and the corrected TRAIN labels pass every identity gate. The training mapper chose a later decision time than t0 in all 5,480 labelled cells. Owner approved (a) the t0 mapper, (b) excluding 263 base/native_base items from outcome supervision, and (c) the 10 s eval correction window. M1 retrain package: [RETRAIN_READY](../runs/active/WM-N1-LABEL-TIME-FIX1-20261003/RETRAIN_READY.md); GPU50 smoke is the only training-path test. M1 unchanged.

Full TRAIN timing audit (2026-10-03): [stopped at corrected policy gate](../runs/active/WM-N1-LABEL-TIME-FULL-20261003/REPORT.md). With source-cited policy correspondence, 900 labelled items map, but 196 chunk2/3 campaign switching items (13,524 rows) carry true-fleet labels (native_fleet/pi_fleet): suspected label-binding defect. Last-row-per-tau timing not measured; exclude/rebind decision and any further read need owner approval. M1 unchanged.

TRAIN identity diagnostic (2026-10-03): [exact row and policy-name resolution](../runs/active/WM-N1-LABEL-TIME-DIAG-20261003/REPORT.md). Global row69 uses label class damping_fleet versus receipt controller damping; frozen source explains this naming difference. The prior exact-string audit check was overstrict for this row. Full TRAIN timing remains unresolved; another read needs owner approval. M1 unchanged.

WM-N1-LABEL-TIME-20261003: [stopped TRAIN metadata audit](../runs/active/WM-N1-LABEL-TIME-20261003/STOP_REPORT.md), one exact75624-row/ten-column projection, zero physical values/SUMO/training.12 generated cases pass; actual row identity gate refused. Failure preserved; no real-data retry.

Evaluation bridge (2026-10-03): [source foundation review](../runs/active/WM-N1-EVAL-BRIDGE-20261003/SOURCE_REVIEW.md), 47 generated cases covered with failure/repair receipts retained. Real scoring remains disabled pending action/temporal binding, comparator and truth-roster contracts plus explicit data release. Source stays in its managed branch; M1 unchanged.

EFFH-01 (2026-10-03): existing GLM effect-horizon analysis indexed as in-progress/report-only; [independent source review](../runs/active/EFFH-01-REVIEW-20261003/SOURCE_REVIEW.md) supports the finite 300-to-600s warning and corrects extrapolation beyond600. No new simulation/training/outcome access; original numerical/provenance verification remains pending.

<!-- MIG05-closeout -->
**WM-S-TRAINSETUP-20261009:** zero-simulation engineering preparation of mixed
switching/direct and original training, recorded-teacher default and optional
timeline callbacks, strict exclusions and four-stage launch guards. One interrupted
synthetic CPU check and its smaller completed successor are recorded; no GPU or
real-data fit. [Evidence](../_out/TRAINSETUP_READY.md).
**MIG-05 (2026-09-24):** [dataset regeneration](../output/evidence_regeneration_v1/CLOSEOUT.md). 88/88 world-model dataset files match the original hashes. Charged 57,792 s of 602 s prefixes. Whole-run aggregates differ. Original run directories, checkpoints and pre-CIRCLES results remain deleted. No scientific conclusion changed.

<!-- WM20-closeout -->
**WM-20 (2026-09-24):** [100 m radius and detection-first constructor](../output/radius_penetration_v1/REPORT.md). Offline, 21 exposed worlds, zero simulation. Identity reproduces WM-18. 100 m ideal occupied count MAE 0.26329 versus 0.43183 at 50 m. Sensing-only abstains (100 m, 0.872 at 2% to 0.100 at 40%). Memory plus nearest-cell fill overcounts declared cells by about 140–230 vehicles. Conservation counting and a physics fill are the untested successor, not part of this result.

<!-- CV02-closeout -->
**CV-02 (2026-09-20):** [PI source semantics and FS adapter](../output/controller_verification_v2/REPORT.md). Published PI zero-history startup resolved, causal state checks added; opt-in FS bridge preserves sole ownership and multi-CAV telemetry. 52 scoped tests pass; zero simulation/fitting, no physical efficacy claim. Next bounded live mechanics remains unallocated.

<!-- CV01-closeout -->
**CV-01 (2026-09-20):** [Controller-first goal, initial audit and pure-law verification](../output/controller_verification_v1/REPORT.md). Astra plan/review and Luna execution; frozen CIRCLES identity/replay checks pass, publication-checkpoint gap remains; original FollowerStopper pure law implemented and 13 tests pass. No new simulation or physical benefit claim. Controller verification program remains active.

# Experiment history — what has been tested, what it showed, and what it answers

<!-- OE41-current-start -->
**OE-41 completed (2026-09-11):** [Formation-harm investigation](../output/formation_harm_v1/REPORT.md). All32 feasible OE34 formations and64 dependent policy contrasts audited. Initially compact14/15 reach HOLD versus8/17 noncompact;delivered medians18/28s. Assembly has zero excess severe-braking/TTC/collision exposure but creates local motion differences. Seven H5/H20 qualification flips share identical assembly; long formation alone does not explain harm. Targets are source-dependent (3.045–24m/s),not universally24. Two qualifying whole policies never reach HOLD;original labels preserved. Costs appearing after handback do not identify handback as cause. H0 immediate controlled exit at actual readiness is a bounded proposal only (7320s ceiling if separately executed),not launched. No new simulation/fitting or validated predictor; CAV estimator work remains open. V1 independent interval-check boundary error preserved/superseded byV2.
<!-- OE41-current-end -->

<!-- OE40-current-start -->
**OE-40 completed / CORR-44 (2026-09-11):** [Measured wall flow effects](../output/flow_effect_v1/REPORT.md). All114 existing OE34 runs reconstructed with conservation (547314 samples); 32 feasible groups,64 dependent policy contrasts. In all22 actual duration branches, the first downstream cumulative-count difference follows HOLD+5 and actual handback (H20 median76.5s after entry). This is a count-response diagnostic, not onset of all effects. Flow and accumulation change in both directions; early gains can reverse later or fail other guardrails. H20/H5 qualifying counts remain5/32 and8/32 feasible groups. Next: report-only error/coverage audit for anticipated time-to-influence and future merge state; optimal activation and benefit selection remain unvalidated. Full-state reconstruction is complete; CAV measurement validation is not. OE39 equal-final-count/online-measurability inference is superseded in part. Zero new simulation/fitting; early nine read-only extraction chronology disclosed; no successor campaign launched.
<!-- OE40-current-end -->

<!-- OE38-current-start -->
**OE-38 / CORR-42 (2026-09-11):** [OE37 source review](../output/oe37_review_v1/REPORT.md). OE37 contains superseded feature/model claims and ambiguous pooled qualification counts; its RouteD conclusion is not established by the submitted synthesis. OE34 fitted tracked histories/interactions in a300-column combined kernel representation; its negative validation remains. Complete the requested existing-data flow/queue mechanism analysis and report-only measurement/coverage audit before choosing a successor route. Zero simulation/fitting; no new observation success or physical result claimed. OE37 original preserved with review status.
<!-- OE38-current-end -->

<!-- OE36-current-start -->
**OE-36 completed (2026-09-11):** [Extra regulation and sequential second-wall tests](../output/multi_wall_probe_v1/REPORT.md). Two live mechanics and three full2400s corridors completed; three additional policies were exactly source-equivalent and reused. Total7440s charged/completed, no physical retries. All three new arms fail original criteria:90030 relay improves service but violates braking/effort;90045 extension and relay worsen service and remain TTC failures. A40s HOLD and disjoint sequential20s relay are physically delivered; no new qualifying outcome, no simultaneous-wall or predictor validation. Ledger OE-36 preserves all comparisons, aliases, accounting repairs and negative evidence. No successor campaign follows.
<!-- OE36-current-end -->

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


**OE-28 (2026-09-10):** [Recovery release implementation and test](../output/recovery_release_v1/REPORT.md). The single120s fixture completed safely but recovery was refused by a predicted speed/slew conflict. Version2 repairs that constraint offline; no live v2 validation. Both selected corridor cases lack one lane reference and reproduce the original H5 policy across2400 saved ticks, so4800s of redundant simulation was omitted. The campaign is stopped; training remains paused and acceptance sealed.

Latest maintenance entry: **OE-27 / CORR-38 (2026-09-10)**, the
[consolidated evidence and warning analysis](../output/duration_selection_v11/REPORT.md).
Zero new simulation; six-arm replication and both branch-prefix checks completed;
OE-26 totals/count claims corrected. Fixed front/rear report warnings reveal local
hazards after handback, without validated wall-action value. The older narrative
coverage and entry counts below are historical; the machine ledger and ROUTE
contain the current record. No prior negative results were discarded.

Purpose: a searchable account of experiments already performed, their results, and the
questions they answer, so that no test is repeated and useful negative evidence survives.
This file is **evidence, not instructions**; the single current route remains
[ROUTE](ROUTE.md). Machine-readable records with stable IDs live in
[protocols/experiment_history.json](../protocols/experiment_history.json) (schema
`experiment_history/2`, 99 entries; structured cost-difference records for current-era
comparisons, campaign/run/comparison registries, coverage metadata, pending updates,
and correction/migration logs). Future experiments must be registered and closed with
[EXPERIMENT_RECORD_TEMPLATE](EXPERIMENT_RECORD_TEMPLATE.md); open questions are tracked in
[OPEN_QUESTIONS](OPEN_QUESTIONS.md). Read-only lookup and validation:
`python tools/experiment_history.py search|card|closest|open-questions|pending|validate`.

**Scope and freshness (ledger `coverage` block):** families CUR/LC/TH/TA/SW/W5/CFG/DEC/ND/LEG,
cutoff 2026-09-07 (maintenance update 2). The continuation target-adjust evidence is
**incorporated** as TA-01/TA-02/TA-03 with source-verified numbers; the block's remaining
open items are the TTC warning lead/false-alarm quantification, the PART4_V2 raise-row
re-derivation, and raw artifacts (never opened by this ledger, by policy) — see ledger
`pending_updates` PU-1/PU-2 and the `ttc_diagnosis` block.
[docs/EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md) is a secondary short index, not the
master. Evidence-status vocabulary (ledger `conventions`): **consistency_verified**
(stored arithmetic/relationships/paths validated) | **source_values_verified** (stored
values additionally compared field-by-field against the registered source analysis via
`verify-sources`; report in protocols/source_verification_report.json) |
**integrity_receipt_verified** (an identified receipt attests replay/execution integrity) |
**report_only_indexed** (historical families) | **unknown_or_pending**. Raw per-vehicle
artifacts were never independently re-derived by this ledger in any mode.

Reading rules honored by this inventory:

- No sealed raw acceptance outcome was read. Sealed blocks are recorded by permitted
  metadata only and their outcomes are marked uninspected (ledger `sealed_blocks`).
- Previously published historical summaries are kept distinct from any access to sealed
  underlying outcomes.
- Historical documents retain their original text. Where prose disagrees with
  machine-readable evidence, the original text is preserved and a correction entry is
  added (see [Corrections](#corrections-and-reconciliations)); frozen criteria are never
  weakened.
- Zero new simulations were run to produce this record.

---

## 1. Eras at a glance

| Era | Window | Controller family | Baseline | Fate |
|---|---|---|---|---|
| Fairness/medium RL | 2026-08-03..10 | Shepherd MAPPO, learned abstention (Krauss) | paired no-op | frozen out; "do not resume medium optimization" |
| Metastable/Krauss walls | 2026-08-11..24 | health-gated band wall, supervisors sup_v43/sup_idm | no-op / Krauss champion | consolidated as legacy motivation ([RESEARCH_OVERVIEW](../archive/pre_circles/docs/RESEARCH_OVERVIEW.md)) |
| Counterfactual gates | 2026-08-24..27 | gate models on paired no-wall/wall rows | wall-off / damping | predictor failed locked test; estimand reset twice |
| Decentralized damping | 2026-08-27..31 | analytical residual + activation options (IDM) | native IDM abstain | fixed-rule prospective acceptance FAILED; learning abandoned |
| IDM compact wall v4–v5.8 | 2026-08-26..09-02 | latched-cohort compact moving wall (IDM) | analytical damping | treadmill of no-gos; halted by estimand reset |
| Sustained wall (SW) | 2026-09-02..05 | health-gated band-passage wall (IDM) | C / damping arm D | regime-mapped, gate fit development-only; stage 2 paused 32/240; archived at adoption |
| Bespoke native damping | 2026-09-05..06 | sensor-local RL damping (v1/v2/v2.1) | native IDM | NO-GO; owner pivots to external controller |
| **CIRCLES era (current)** | 2026-09-05..07 | frozen CIRCLES damping + RSU moving-wall treatments | frozen CIRCLES damping | active; zero qualifying wall benefits to date |

Everything before the CIRCLES adoption is archived under
[archive/pre_circles](../archive/pre_circles/ARCHIVE_INDEX.md) (59,846 files, manifest-preserved).
It is evidence and provenance, not a resumable route.

---

## 2. Current era (CIRCLES baseline): the central question

**Q: Can an RSU escalate from frozen CIRCLES damping to a temporary coordinated moving
wall and produce qualifying physical benefit?**

**Answer to date: No — zero qualifying benefits in any executed wall action.** Seven
distinct physical traces — **six delivered walls + one refusal** — from the
lifecycle/cohort/duration/entry-target subset (B4107, B4108-refusal, C4108, D4107-H5,
D4108-H5, T4107, T4108) on two exposed seed groups all fail; the natural-panel
attempts (8), gentle-completion episodes (2) and brake/HOLD cells (12) are separate
CIRCLES-era traces and also fail. **"No tested wall qualifies" does not establish "no
wall can qualify"** — it is a statement about the tested designs on two seeds.
Retaining CIRCLES damping remains the only justified *controller behavior to date*
(route statement; see [OPEN_QUESTIONS](OPEN_QUESTIONS.md) §R for the policy/science
split). Further speed/duration/braking search on the swept axes is not justified by
this evidence.

The chain of executed blocks (all seeds 4107/4108, mainline 2700 veh/h unless stated,
p20, 2400 s horizon, decision 600 s):

1. **Adoption** ([PUBLISHED_DAMPING_REUSE_AUDIT](../archive/pre_circles/docs/PUBLISHED_DAMPING_REUSE_AUDIT_2026-09-06.md),
   [PUBLISHED_IDM_FIT_CHECK](../archive/pre_circles/docs/PUBLISHED_IDM_FIT_CHECK_2026-09-06.md)):
   constrained adapter failed at higher penetration; direct-output p20 adopted
   (severe braking −18.76%/−4.63% vs native at 2000/2200 veh/h, with +2.08/+1.47 s
   service and +53/+55.5 vehicle-s TTC<3 trade-offs). Adoption is not scientific
   acceptance; paper-checkpoint identity unverified. (CUR-01)
2. **Characterization** (`escalation_pilot_v1`, 19,200 s): native vs damping on 4107/4108
   × 2000/2700. Damping is better at m2700 (service +11.60 s/req improvement at 4108)
   and worse at m2000/4108; 4 damping collision involvements at 4107/m2700 are a
   retained baseline anomaly. Poor damping results are characterization, not a trigger
   for another learner. (CUR-04)
3. **Natural panel** (`natural_wall_effects_v1`, 33,600 s): 8 wall attempts, 5 complete
   20 s HOLDs, 3 aborts; **0/8 qualifying** under the frozen criteria (benefit ≥1.0
   s/req service or ≥0.02 veh-s braking; cost allowances; zero treatment collisions).
   Assembly-only worsened service +23.15 s/req where the full wall improved it 8.42
   s/req — sensitivity to continued coordination, not a 20 s optimality proof.
   (CUR-09; [REPORT](../output/escalation_execution/REPORT.md))
4. **Observer replay** (12,000 s, all five replays exact): regional reconstruction shows
   4107 approached congested (near-approach reporter mean 12.29 m/s) while 4108 was
   free-flowing (21.60 m/s); downstream queues were measurable from reports before every
   decision. Explanatory only. (CUR-11)
5. **Report-aware dissolution release** (4,800 s): both arms harmful; in 4107 the
   transition never activated; in 4108 it fired for 5 s and still violated the effort
   allowance. (CUR-12)
6. **Gentle wall** (−0.3 m/s² cap, 0.15 m/s³ ramp): first corridor block **invalid**
   (logger crash at 600 s, zero wall commands; preserved); owner-authorized replacement
   (4,800 s) completed both cases — **both harmful** (4107 TTC +0.020495 veh-s/req +
   2 treatment collision involvements 888 s after abort; 4108 braking/effort/service/TTC
   all violated; measured member decel reached −0.8022 m/s² under −0.3-bounded
   commands). (CUR-14 invalid, CUR-15 valid negative)
7. **Lifecycle repair A–G** (12,480 s incl. preserved mechanics failures): finite
   lifecycle wall delivers correctly — 4107 12 s assembly/20 s HOLD/1 s exit (harmful:
   TTC +0.1124 veh-s/req), 4108 refused at entry (exact damping match, nonbeneficial),
   upstream-cohort 4108 delivered 18/20/1 (all five costs worse), H5 release branches
   delivered 5 s HOLD + 1 s exit (4107 H5 costs +28.03 s/req vs H20; 4108 barely
   changes). Parts E/F/G not activated. (LC-05/07/08, LC-11)
8. **Braking/HOLD factorial** (19,440 s fully consumed): all **12 cells harmful**; only
   4107/strong/H20 completes its registered maneuver and still fails braking/TTC
   guardrails; neither nominal H40 completes (actual 27/31 s); −0.3 is not a measured
   floor (measured −0.49/−0.80 under bounded commands). Do not repeat this factorial.
   (LC-10; [REPORT](../output/brake_hold_v1/REPORT.md))
9. **Harm localization + target speed** (Parts 1–4 offline; uncapped experiment 5,040 s):
   the 4107 H5-vs-H20 gap is a lane-acceptance cascade at 637.5 s (a congested-attractor
   pair worth 28.03 s/request); the shared 4108 harm is the assembly/HOLD slowdown dose
   (queue phase shift, later discharge — not direct braking); the 24 m/s cap binds in
   every realized engagement; the one registered alternative (uncapped mean-minus-2)
   made 4108 nearly neutral (nonbeneficial; **fails strict joint non-deterioration** —
   effort and TTC worsen slightly) and left 4107's harmful realization reproducing
   within 2 s per vehicle. (TH-01..07;
   [REPORT](../output/wall_harm_speed_v1/REPORT.md))

**Strongest supporting experiment** for "no qualifying benefit": the convergent set of
CUR-09 + LC-05/07/08 + LC-10 + TH-07 — different actuators (strong/gentle/finite/
uncapped-target), durations (H0–H40), braking caps, cohorts and entry targets, all
failing the *unchanged* criteria on the same two seeds.

**What was never properly tested** under this baseline: any mechanism that addresses the
demonstrated failure mode (merge-saturation attractors under discrete lane-allocation
flips); any fresh seed; assembly-vs-handback isolation for the current actuator; any
penetration other than p20; any action validated across seed groups (zero positive
labels exist — fitting is inadmissible, and that insufficiency is itself the result).

### 2b. Continuation evidence (incorporated 2026-09-07; ledger TA-01..03, source-verified)

The entry-target comparison (TH-07) changed assembly and handback together with
maintained speed — a full-maneuver contrast. The **continuation** arms now vary
maintained speed *separately*: they branch from the executed keep-24 histories at
HOLD start + 5 s with the original 20 s HOLD, scheduled taper/exit and handback
unchanged. All numbers below are cost differences (treatment − reference, negative =
improvement) verified against the registered analyses (ledger `verify-sources`: 11
sources, 200 outcomes, 0 mismatches).

- **Lower-22** (TA-01): 4107 — service −1.562, braking −0.053, wait −0.108 improve;
  effort +0.950 and TTC +0.053 worsen → harmful. 4108 — all four costs worse (+9.456
  service) → harmful. Versus keep-24: lowering is worse in 4107 (+2.591 service) and
  better in 4108 (−3.813). Lowering 24→22 is a *deeper* slowdown; two settings per
  seed, no dose law in either direction.
- **Raise-26** (TA-03): 4107 — service −8.436, braking −0.058, effort −5.745, wait
  −0.134 all improve; **TTC +0.0396 is the only violated guardrail** (56.0 veh-s
  excess vs 28.3 allowance); collisions 0 vs 4 → harmful (TTC only), the strongest
  measured service result in 4107 and −4.282 s/req vs keep-24. 4108 — statistically
  indistinguishable from keep (service Δ −0.009), all four costs worse → harmful.
  The v2 report prose showing effort +5.745/wait +0.134 for this arm is **sign-
  inverted** (ledger CORR-6; superseded by TTC_GROUND_TRUTH_V3 and the registered
  analysis). The original v2 up-arm failure (TA-02) was a preventable configuration
  error, **not** physical infeasibility.
- **Status:** eight delivered same-prefix continuation actions across two seed groups —
  eight harmful, zero qualifying, zero strict-joint successes. "Improving an existing
  wall does not automatically justify assembling it"; a better continuation never
  retroactively justifies the entry decision.

**TTC diagnosis (external audit v3, finalized and incorporated; ledger `ttc_diagnosis`).**
Ground truth independently verified. Localization: coordination-window TTC exposure is
*identical* across raise/keep/damping in each seed (4107: 38 veh-s in [600,633) per the
phase decomposition; 4108: 30.5) — the entire excess is post-release (4107 phase split:
raise 1823 / damping 1767 / keep 1926 veh-s; net raise−damping +56, raise−keep −103).
Known issues and their status: encounter-start phase assignment corrected in the v4
phase module (pre-entry excluded; boundary-crossing encounters split by sample); equal
same-prefix observations show non-selectability for that decision, **not** observation
insufficiency; front-gap vs rear-neighbor feature labeling corrected in the inventory;
warning lead times (4–9 s observed for the dominant events) are real but **false-alarm
rates remain unquantified — pending**. The pre-event reporting signature is **not an
action-validated warning**, and no conclusion that future TTC risk *cannot* be predicted
follows from the involved vehicles being unrequested or HDV at the decision.

**What would merely repeat an answered question**: another speed/duration/braking
setting; re-running the factorial; another cohort rule; renaming an old sweep as a new
route; treating the IDM-era sustained-wall results as if they tested the CIRCLES
actuator.

---

## 3. Damping: adoption, learning, and ownership

- **Bespoke learning is answered (negatively), do not reopen casually.** The
  decentralized line failed mode/gain/offset authority, fresh-seed replication of its
  benefit signal (AUROC 0.507 fresh vs 0.85 development — "it was development-sample
  structure"), all mechanism variants (projected release, modified FollowerStopper,
  projected blend, P50), timing learnability (oracle +0.0478 s exists but is not
  deployably predictable — ridge below shuffled control), and the one prospective-style
  test: the frozen 5–10% rule arm on untouched seeds 4201–4220
  (`acceptance_passes = false`, reward −0.04325, CI [−0.1243, +0.0157], every
  leave-one-seed-out negative). (DEC-01..05)
- **Native damping v1/v2/v2.1**: v1 had an advantage-subtraction defect (terminal −2
  penalty changed advantages by exactly zero); ep20 checkpoint ineligible (6/8 dev cases
  fail; tiny residuals still issue slowDown on 81.16% of decisions); magnitude ablation
  proved an interface discontinuity (effects persist at ±1e-6 m/s²); v2 with explicit
  native mode failed both corridor cases; v2.1 fixed the rebound but pulse effort was
  +18.94% worse. The 2026-08-28 training-contamination smoke is quarantined and must
  never be used. Campaign paused at 36/8/8 (124,800 optimized seconds preserved, no
  accepted checkpoint); the 32-case acceptance battery was never executed and its seed
  block 912001-912032 stays sealed. (ND-01..07)
- **Pivot**: "The owner has clarified that escalation is the principal contribution.
  Reusing an existing trained baseline is preferable to inventing another damping
  reference." (ND-05/06 → CUR-01)
- **Ownership repairs** (implementation, not efficacy): exact numerical replay of
  36,200 decisions (max error 0); cancelling an unfinished command changes physics
  (−0.8 → +2.412 m/s² at 4 s); the 30 s workflow smoke matched the saved prefix to
  1.1e-14 vehicle-s. (CUR-02)

## 4. Wall formation and actual delivery

- Controlled mechanics: dispersed members from 10 to 150 m initial span assemble, hold
  60 s compact, taper and hand back on a clear road (readiness 6–34 s); two fixture
  defects (front_unsafe collision; member_lane_loss fixture invalidity; a real
  stale-report dispatch bug) were fixed by separately registered repairs. Initial span
  is neither a delivery rule nor a physical boundary — the old 50 m threshold is a
  formation-criterion hypothesis, **not** an admission limit. (CUR-05..07)
- Corridor delivery is solved for the finite design: 12/18/20 s assemblies, verified
  20 s HOLDs at half-second endpoints, coast exits, identical-damping handbacks, all
  command contracts within bounds (min measured response −0.30000000000000426 m/s²).
  What is *not* solved: completion of the registered maneuver in every setting
  (4108 gentle never reached HOLD; factorial H40s abort at 27/31 s) and post-handback
  safety (measured decels exceed command bounds under traffic). (LC-03..05, LC-10,
  CUR-15)
- **Feasibility ≠ incremental benefit** — established repeatedly: B4107 delivered
  perfectly and was harmful; C4108's 18 s forecast was delivered to the second and all
  five costs worsened.

## 5. Braking authority and phase-specific limits

- Factorial −0.3 vs −3.0 m/s² × H0/H20/H40: no qualifying cell; gentler is not
  uniformly safer (gentle 4108 measured −0.8022 m/s²; gentle 4107 H20 aborts at release
  with 2 collisions) and stronger is not uniformly better (strong fails TTC guardrails
  where delivered). IDM-era v5.6.2 similarly showed low sustained *assembly* authority
  cannot retain compactness or benefits. Do not repeat this factorial or infer an
  optimal duration. (LC-10; W5-15)
- Phase-specific limits (IDM era): v5.2's +251 paired hard-brake vehicle-steps were
  97.7% post-control (onset excess +248 beyond 60 s; +231 on downstream edge
  1487606178) — delayed downstream divergence, not release-localized; the T880 causal
  diagnosis (sham exact, assembly-only not dominant) blamed HOLD + taper/release
  discharge; the v5.5.1 follower-aware release could not make beneficial cases safe.
  In the current era, harms also concentrate post-handback (all 161.5 gentle-4108 TTC
  vehicle-seconds; C4108's late-service share 63.8%). (W5-06/11/12; LC-09, CUR-15)
- The 4108 exit-feasibility defect: the check evaluates at the 24 m/s wall speed with a
  1-tick horizon while handback re-accelerates members to 27.9–29.8 m/s within 6–10 s;
  runtime consults the downstream predicate only at taper and never consumes the ramp
  estimate at admission. (TH-03; gentle completion entry/exit audit)

## 6. Target speed and slowdown depth — four distinct studies, do not conflate

1. **Krauss 10/20/30% study** (2026-08-27,
   [TERMINAL_WALL_SPEED_SENSITIVITY](../archive/pre_circles/docs/TERMINAL_WALL_SPEED_SENSITIVITY_RESULTS.md)):
   varied *percentage reference-speed reduction caps* (5 dev seeds × 2 steps × 3
   levels); mean requested targets 24.92/22.16/19.59 m/s, realized 20.94/19.35/18.47;
   state-dependent and **nonmonotonic**; 30% reached 75% strict-case safety (below the
   frozen criterion); no level universally safe. **No absolute 24 m/s arm existed
   here.** (W5-04)
2. **IDM fixed-band absolute speeds W10/W15/W18/W21** (2026-09-02, sustained-wall
   actuator): only W10 kept the incidence reduction (0.40 [0.233,0.567]); W15/W18 each
   *induced* 3 collapses; delay fell monotonically with speed. A different actuator and
   baseline; does not transfer. (SW-03)
3. **Current 24 m/s cap**: binds in every tick of all four realized current-wall arms;
   forces −3.68/−5.43 m/s slowdown and 5–11 s extra assembly vs the formation rule's
   own intent. Never previously varied. (TH-04)
4. **Registered uncapped experiment** (`target_speed_v1`, 5,040 s): commanded
   25.681/27.432 m/s; 4108 became nearly neutral (nonbeneficial; effort/TTC worsen
   slightly so strict joint non-deterioration fails); 4107's harmful attractor
   reproduced within 2 s/vehicle. Closing position: no further speed/duration/braking
   search justified. (TH-07)

**Monotonicity warning**: two target settings on two seeds establish no monotone dose
law, no uncontrollability, and no observation insufficiency; the entry-target change
alters assembly and handback together with maintained speed — it is a full-maneuver
contrast, never a pure slowdown-depth effect.

## 7. HOLD duration and release behavior

- Nominal vs delivered must always be separated: H40 nominal never delivered (27/31 s
  actual); B3's constraint-triggered HOLD was 8 s actual vs 20 s nominal; the
  sustained-wall "hard max 900 s" was an absolute cap = 650 s hold (self-declared);
  R1/R1b state-release rules were provably inert on plateau demand (R1 digest-identical
  to the fixed hold on 120/120; R1b fired 0/90).
- Same-prefix continuation contrasts (verified, not reruns): H5 vs H20 in 4107 costs
  +28.03 s/request (mechanism: 637.5 s lane-acceptance cascade; congested-attractor
  pair) while improving TTC/source/collisions; in 4108 it changes almost nothing.
  Duration curves in the IDM era were regime-dependent (2400 shorter-better; 2600
  interior optimum 650 s; 2700 optimum 450 s; WON onset-triggered lost at 2700,
  +3829 s). (LC-08; TH-01; SW-08/10)
- Release behavior: report-aware dissolution and gentle report-aware exit both
  delivered correctly by their contracts and both remained harmful; identical-damping
  handback is verified in every current-era block. (CUR-12/15; LC family)

## 8. Cohort placement, recruitment region, map feasibility

- Removing the 1,900 m recruitment cutoff (report-eligible pairs, shortest predicted
  assembly) delivered the forecast 18 s assembly in 4108 — and worsened all five costs.
  Cohort feasibility is not benefit. (LC-06/07)
- Part A (zero simulation): the original map suffices; no evaluation truncation exists
  (every factorial cell serves every requested trip by 2400 s); corridor length is a
  design choice, and a v5.7 +900 m extension design existed but was never physically
  evaluated (and was self-flagged as confounded). An extension does not itself
  establish benefit. (LC-01; W5-16)

## 9. Assembly-only, sham, and other causal comparisons

- Sham assignment exactly matches continued damping (30 s prefix bridge; sham identity
  3/3 exact in the IDM T880 diagnosis) — observation/assignment alone does nothing.
  (CUR-08; W5-11)
- Assembly-only vs full wall: in the natural panel the full wall improved service 8.42
  s/req where assembly-only worsened it 23.15 s/req (same seed) — the harm/benefit sits
  in continued coordination and release, not assembly alone. The IDM-era T880 diagnosis
  agreed (assembly not dominant; HOLD + taper/release discharge blamed). **But no
  executed experiment isolates the current 12/18 s assembly from the current handback** —
  assembly, first 5 s HOLD, and handback are bundled in every executed 4108 arm.
  (CUR-09; W5-11; TH-02)

## 10. Regional CAV reports and observation studies

- Reports demonstrably support state discrimination and feasibility/execution (the 18 s
  assembly forecast delivered exactly 18 s; downstream queues measured age 0 before
  every decision; regional contrasts separate 4107's congested approach from 4108's
  free-flow). (CUR-11; TH-03; LC-06)
- What reports have never supported: a *benefit* predictor. IDM-era ladder: fair
  single-feature threshold L0' (upstream_inflow_30s or near_merge_count) was not beaten
  by the multivariate L1' (+0.160 [−0.014,+0.328] at λ=0); onset-within-100 s is
  predictable (OOF AUC 0.944; merge_edge_mean_speed 0.832 univariate) but onset
  predictability ≠ benefit (delay-delta R² 0.168); prevention concentrates pre-onset
  (0.16 vs 0.04–0.07) and demand regime, not lead time, flips the delay sign.
  Current era: 65 acquired features, 7 fitting-eligible rows, **zero positive labels**.
  (SW-10; TH-03; LC-11)
- Earlier observation diagnostics: padded-vs-live RSU fields changed nothing (AUROC
  delta exactly 0.000); ranked-neighbor and 134-field signals did not replicate on
  fresh seeds; the shadow-rollout program (10 s exact twins) reached dose AUC 0.978 but
  safety 0.601/benefit 0.677 with no threshold policy meeting 90/50 — confirmatory
  NO-GO. (DEC-08; DEC-02; W5-10)

## 11. Gate fitting, leakage, and held-out results

- Leakage discipline (established 2026-08-24/25 and carried forward): split by seed
  group, never by decision window; health h cannot be the primary outcome because it is
  a policy feature; post-treatment values are labels only; "a random-row split would be
  misleading because seed leakage would make the dataset look larger than it is."
- Negative gate results, all valid: N30 training failed (health features degenerate at
  0 in hyper density); two-head local-v2 failed its one-time locked test on seeds
  201-210 (locked R² all negative; precision 15.8%) — **seeds 201–210 are consumed and
  can never serve as a locked split again**; win-sensitivity relabels are post-hoc and
  do not repair the predictor; the escalation-v2 300-case dataset is invalid as
  efficacy evidence (dose failure: mean delivered ≈4 s of the requested 30 s) though
  valid as a dose audit; v5.4 Gate-B development screens and shadow-v2 confirmatory
  check all failed; validation seeds 4601–4660 and final seeds 5001–5060 were never
  opened and remain sealed. (CFG-04/05/06/08; W5-10)
- The sustained-wall gate fits (gate v1, L0'/L1') are development-only and explicitly
  not frozen: "the gate is close to a loading threshold; the multivariate machinery
  adds no robust value on top of it." (SW-07/10)

## 12. Fixed-band metering vs persistent moving-wall designs

Three actuator families must never be pooled:

- **Band-passage "metering" walls** (LEG-07 legacy campaigns; SW sustained wall):
  membership by distance-to-merge band [200,400] m; IDM or Krauss; damping arm D is
  *harmful at 2200* (24→29 collapses, +1.891 s/veh) and the wall's delay sign flips
  with demand (+2.0–2.8k s at 2200–2400 vs −3.1 to −7.1k s at 2600–2700; boundary
  between 2400 and 2500). Stage 2 (final generation) was paused at 32/240 seeds and
  never produced results; the whole program was archived at adoption.
- **Latched-cohort compact walls** (W5 v4–v5.8): treadmill of no-gos ending in the
  2026-09-02 estimand reset; no collapse-prevention capability was ever demonstrated
  (best: 2 preventions in 179 dose-qualified dev cases).
- **The current CIRCLES-era cohort moving wall** (CUR/LC/TH): the only family with
  evidence under the paper's baseline — and it has zero qualifying benefits.

"Historical metering gains do not validate the current moving formation"
([history route audit](../output/history_route_audit/DECISION.md)); the privileged
multi-feature gate advantage over a fair threshold was only +0.160 s/vehicle
[−0.014, +0.328] in the IDM era.

## 13. Proposed but never implemented (do not silently resume or re-derive)

| Item | Status | Where |
|---|---|---|
| v5.5 density-duration factorial (294 branches) | protocol only, superseded by the mapper's single 880 s question | W5-11 |
| v5.7 extended-zone frozen protocol | frozen, unexecuted, self-flagged confound | W5-16 |
| v5.7.2 upstream-convergent assembly | frozen, never executed, halted by reset | W5-18 |
| v5.8 multiwall (Gates M0–M4) | skeleton + 28 synthetic tests only; no SUMO ever | W5-19 |
| Gate-B observation-v3 repair; exploratory prospective validation (4601–4660) | frozen, never executed; validation seeds sealed | W5-10 |
| v5.5.1 successor (seeds 4407–4430) | never opened | W5-12 |
| Stage 1b p30 stratum (7241–7360); Stage 2 p30 (8241–8360) | registered, never run | SW-07/11 |
| Stage 2 p20 seeds 8033–8240 | paused at 32/240; owner stopped | SW-11 |
| Native damping acceptance battery (912001–912032) | protocol only; sealed | ND-07 |
| 40% terminal-speed variant; S06 extension (57,600 s); predictor-scale program (448 trajectories) | predeclared/registered, never run | W5-04; CUR-09 evidence audit |

## 14. Invalid, interrupted, or incomplete (do not quote as results)

- **Invalid**: gentle_wall_v1 corridor (logger crash at 600 s, zero wall commands,
  1,200 s actual / 5,040 s charged; benefit_label null); B2 developing_constraint
  fixture (insertion impossible; the constraint never existed; B2 whole-stage passing
  status is false); escalation-v2 300-case dataset as efficacy evidence (dose failure);
  the quarantined 2026-08-28 contamination smoke; ND v1 checkpoints (defect);
  TH-05 mechanics fixture (charged 120 s failure, no result consumed).
- **Interrupted/stopped**: B1 (entry refusal, STOP); Stage 2 generation (32/240,
  owner-stopped); native damping campaign (36/8/8 checkpoint pause); formation
  mechanics v1/v2 (STOPs resolved by v2/v3); expC criterion-3 (2400 rows pending,
  never appended).
- **Valid negatives** (keep distinct from the above): natural panel, brake/HOLD
  factorial, lifecycle A–D, dissolution, gentle completion, uncapped-target
  experiment, DEC-04 acceptance, CFG-05 locked test, W5-10 shadow NO-GO, ND-05/06.

---

## Corrections and reconciliations

Original texts are preserved; the following corrections are recorded here and in the
ledger (`corrections` fields). None weakens an original benefit/safety criterion.

1. **Uncapped 4108 wall: nearly neutral, not a no-op.** Registered cost differences
   (treatment − damping, per request, population 1,307; negative = improvement):
   service −0.000383 s (slight improvement), braking **−0.000765 veh-s (slight
   improvement)**, effort +0.001100 m²/s³ (worse), TTC +0.000765 veh-s (worse), wait
   0.0; net arrival difference −0.5 vehicle-second network-wide; but 2 vehicles differ
   ∓75.5 s and effort differs on 665 vehicles (+1.44 total). PART5's "meets strict
   joint non-deterioration" is wrong — `analysis.json`
   `all_continuous_costs_nonworsening = false` — and REPORT's "physical no-op"
   overstates. The label stays **nonbeneficial** (within guardrails, no material
   improvement); T4107 also fails strict joint (service and TTC), consistent with
   harmful. *(A v1 version of this history recorded the braking sign with the wrong
   direction; corrected here and in ledger CORR-1 against
   `runs/active/target_speed_v1/analysis.json`.)*
2. **Signs in PART5's table are mixed.** The 4108 row (and the 4107 source-wait cell)
   display improvements-convention values under a "treatment cost − damping cost"
   header; true treatment-minus-damping is −0.0004/−0.0008/+0.0011/+0.0008. Re-derive
   any sign from `runs/active/target_speed_v1/analysis.json` before quoting.
3. **Trace counting.** "Seven delivered wall traces" = seven distinct physical traces =
   **six delivered walls** (B4107, C4108, D4107-H5, D4108-H5, T4107, T4108) **+ one
   refusal** (B4108, executed no wall, exact damping match). C4107 is exact B4107
   reuse. Labels: 5 harmful delivered, 1 nonbeneficial delivered, 1 nonbeneficial
   refusal; **2 independent seed groups** (rows/branches are not seed groups).
   **Scope:** the seven are the lifecycle/cohort/duration/entry-target *subset* —
   not every CIRCLES-era experiment; the natural-panel attempts, gentle-completion
   episodes and brake/HOLD cells are separate physical evidence outside that count.
4. **Remaining-budget arithmetic.** Target-speed block: 120 + 120 + 2400 + 2400 =
   5,040 s of the 9,840 s ceiling; the correct remainder is 4,800 s = 2 episode slots
   and 0 mechanics slots. PART5 contradicts itself ("1 mechanics fixture slot and 2
   episode slots remain" vs "Budget remaining: 1 episode slot"); REPORT repeats the
   latter. Protocol planned 4,920 s; the extra 120 s is the disclosed TH-05 charged
   failure. Lifecycle A–D: 4×120 + 5×2400 = 12,480 s. Brake/HOLD: 240 + 19,200 =
   19,440 s fully consumed. The 12,480 and 19,440 blocks are separately registered; no
   document sums them.
5. **Monotonicity/purity guard**: no monotone dose law, no pure slowdown-depth effect,
   and no uncontrollability claim may be drawn from two target settings on two seeds;
   the entry-target comparison changed assembly and handback together.
6. **Verifier/scoring failures and their receipts** (re-verification ≠ new physical
   run): TH-05 ImportError (charged 120 s, preserved); TH-06 criterion-formulation
   failure corrected by a zero-simulation offline law recomputation (all 15 checks
   true); corridor STOP on `observation_unavailable` telemetry cleared by
   `resolve-target-speed` re-verification with zero new simulation
   (`verification_resolution.json`); gentle logger failure fixed metadata-only, then a
   new authorized 4,800 s block; brake_hold aggregate-comparison over-strictness
   resolved by tolerance re-check (largest discrepancy 9.75e-9 on 148,761.19);
   formation v2 failures resolved by *new* physical fixtures (v3); adoption runner-smoke
   role guard resolved by metadata (0 s); sequential-timing LOO −0.070 arithmetic
   defect corrected 2026-08-30; timing sweep "28" was a typo for 18; handback
   speed-equality statistic corrected with a later paired physical check (+0.098404 vs
   −0.335789 m/s²).
7. **Archived self-documented drift** (preserved, quoted in ledger entries): terminal
   speed 75%-vs-80% denominator mismatch; PREP3 "5 prevented" vs table 8/3; 2200
   control delay 12.43/12.44 s/veh; T880 taper convenience field zero vs authoritative
   five steps; margin law v1 vs degenerate 32% LOO refit; RESEARCH_OVERVIEW n=200/n=50
   caveat sentence; IDM 2000/2200 no-control cells n=50 vs n=20; PART1's 39,666.5
   interval sum vs 39,663.0 per-vehicle sum and the −126 veh-s vs −0.116 s/request
   insertion mismatch; PART4's duplicated +0.044 hold-error; evidence_audit's pre-freeze
   "TTC allowance 0" vs the frozen 0.02 margin actually applied.
8. **Archive-verification precision.** Adoption-time verification performed ONE
   complete post-suite SHA256 scan of all 59,846 files plus targeted re-verification of
   its sole mismatch ("No new full data scan is claimed"); later era checks traversed
   manifests with `hashes_checked=0` — path/manifest identity checks, **not** rehashes.
   Quote the right evidence level.
9. **Evidence-kind and scope hygiene.** Keep four levels apart: measured effects
   (registered paired differences), temporal localization (window analyses), plausible
   mechanisms (e.g., the 637.5 s cascade — geometry-consistent, not an internal proof),
   and experimentally isolated causes (sham / assembly-only / same-prefix continuation
   arms). Entry-target contrasts are **full-maneuver contrasts** (assembly and handback
   changed with maintained speed), never pure slowdown-dose effects. Failed historical
   prediction studies invalidate the tested feature/action combinations on their
   populations; they do not prove CAV-observation insufficiency under every controller
   or action space.

Ledger `correction_log` (CORR-1..5) and `migration_log` record every change made in
schema v2 with supporting artifact paths; original report texts were never modified.

---

## Making the history actionable

For each major question above: the "what we know" statements cite the strongest
experiment; conflicts are named (e.g., IDM-era W15/W18 induced collapses vs the current
era's cap-forced slowdown harms — different actuators); never-properly-tested items and
genuinely-new vs merely-repeating experiments are consolidated in
[OPEN_QUESTIONS](OPEN_QUESTIONS.md). Before proposing any simulation: consult
[protocols/experiment_history.json](../protocols/experiment_history.json), cite the
closest prior experiment (`python tools/experiment_history.py closest --text "..."`
plus `card <ID>` for the evidence card), state the exact unanswered difference, and
register source, splits, comparators and budget first (template:
[EXPERIMENT_RECORD_TEMPLATE](EXPERIMENT_RECORD_TEMPLATE.md)). Close-out numbers use the
cost-difference convention (treatment − reference; negative = improvement) and are
validated by `python tools/experiment_history.py validate` and
`tests/test_experiment_history_ledger.py`.

## Coverage checklist (what this audit actually inspected)

Inspected in depth (reports, registrations, receipts, machine summaries — not raw
trajectories or sealed outcomes):

- Current era: docs/STATUS, ROUTE, ESCALATION_*, OBSERVER_REPLAY_EXECUTION,
  RSU_DISSOLUTION_CHECK, GENTLE_WALL_CHECK, GATE_FEATURE_CONTRACT,
  OBSERVATION_DECISION_CONTRACT (counting basis only), BRAKE_HOLD_TEST,
  COHORT_ELIGIBILITY_CHECK, DURATION_RELEASE_CHECK, LIFECYCLE_* checks,
  NEXT_EXPERIMENT_PLAN, TARGET_SPEED_CHECK; output/{adoption_cleanup,
  escalation_execution, escalation_test_build, history_route_audit, observer_replay_v1,
  report_dissolution_v1→(via runs), gentle_wall_v1, gentle_wall_completion_v1,
  lifecycle_repair_v1, lifecycle_repair_v2, brake_hold_v1, wall_harm_speed_v1};
  runs/active registrations/summaries/receipts for the corresponding blocks.
- Archived: all 13 sustained-wall stage docs; the v4–v5.8 line (~50 docs); the
  counterfactual gate program (9 docs) + legacy record/overview; the decentralized line
  (~40 docs incl. contamination event); native damping (9 docs); metastable/phase-map
  program (FINDINGS.md, probe designs, git commits e2bc8df..8d993e9);
  PUBLISHED_DAMPING_REUSE_AUDIT / PUBLISHED_IDM_FIT_CHECK.

Known gaps (explicitly not "everything reviewed"):

- Raw per-vehicle trajectory/outcome files were never opened (by design); window-level
  numbers are taken from the registered analyses and were spot-recomputed where cited.
- Stage 2 generation has no results (paused 32/240; Results section unfilled) — no
  Stage 2 population statistic exists anywhere.
- CFG-12 (p30 actionability diagnostic) has no archived results doc; its outcome is
  known only through the estimand-reset diagnosis.
- The legacy stage-2 confirmation / s7 probes / s8 grid have no dedicated report;
  evidence is FINDINGS.md log tails + git commit subjects (cited in LEG-06).
- The 188 archive docs were family-covered via the indexed reports and their cited
  artifacts; per-file line-by-line review of every protocol variant was not performed.
- Budget totals for pre-adoption programs are recorded in branches/seeds and the
  documented simulated-second ceilings (e.g., SW stage branch counts, ND allocations);
  a single global simulated-second grand total across eras is not reconstructible from
  preserved metadata and is deliberately not invented here.
