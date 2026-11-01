# S-RB2-LANE — FINAL delivery / fitted LC NOT ADOPTED, 2026-11-01

Branch `c4/rb2-lane`; engineering base FULL-v4 `22cc32f6` (inherited
round-1 final-delivery HEAD `03c81f91`).

## Round-2 delivery
- **Source fixes ready; exact frozen fitted LC fails free lane-frequency test.**
  Keep fitted LC OFF pending an integrated-source development comparison.
  Implementation commits `a29116e9` + `6b4bfbd5` (plan `73a5685b`);
  conditional evidence `a60a6a62`/`docs/rb2lane/fit_v2`; native free pilot source
  `a60a6a62`, scoring audit `b5e53b3e`; final evidence `docs/rb2lane/free_v1`.
  Numerical evidence commit `b56040d9`; byte-integrity followup `802810b2` (take both).
  It preserves original machine receipt/CSV bytes on Windows/Linux; normalized
  scientific contents and all dynamics are unchanged. `round2_release --verify`
  checks working AND committed artifact hashes against the capsule.
  Final byte audit PASS:24 working/committed artifacts plus the shared result.
  Report `docs/paper_results/RB2_LANE.md`; shared `rb-shared/RB2_LANE_DEVELOPMENT_RESULT.md`.
  Capsule `docs/rb2lane/DELIVERY.json`, SHA256
  `ff42bac3a7567d803257a03256655539088d00023c1df3da200c6ce88480e329`.
- Source comparison complete against read-only SUMO1.27.1 cache
  `E:/Temp/opencode/sumo-1.27.1-rb-lane`. No downloaded source executed.
- Required LC continuation: `E:/agent-work/c4-rb-fit/artifacts/rbfit/lane_raw_c5602102891e`.
  Population SHA256 `b903fa49fe1f3f1a15428ae48de5c8f86863aa5c5821dbb0acc8275c77d4483f`.
- Explicit candidate `configs/wm_phys_rb2_lane.toml`
  differs only by `fitted_lc=true` and `lane_parameter_directory`. Base recipe
  stays OFF. Separate LC causal conditioning attaches AFTER independent IDM;
  native tape saved-ID posteriors never transfer to another export's identities.
  Sampled-profile causal60s replay initializes wishes by ID. No learned corrections.
- Source fixes: nondecision lookahead freeze; documented actionStepLength for
  wishes/decay; source tauRemainder secure speeds for nondefault action steps.
  The legal-decision guard also removes fictitious upstream single-lane ramp
  speed-peak memory before the first legal merge-edge decision (source380).
  No generic changeWish decay exists in this SUMO tag. Full serial negotiation
  and internal critical leaders remain incomplete; see `docs/RB2_LANE_SOURCE_AUDIT.md`.
- 85 core targeted CPU tests PASS (`CPU_FINAL.xml`) plus5 guard/scorer/ramp
  regressions PASS; Ruff PASS. Actual strict-loaded
  checkpoint/recipe four60s native closed-cohort futures (FIT969420/969421,K2)
  PASS finite/conserved/no corrections; sampled LC state remains finite/warm.
  `docs/rb2lane/runtime_v1_SMOKE.json`, source `6b4bfbd5`; this smoke has no arrivals.
- All80 conditional diagnostic episodes COMPLETE/collected: native FIT32 H60/300,
  development12 four nonnative arms H60, K8 persistent profiles, same pooled IDM.
  Native300 precision2.478→13.662%, recall41.489→11.357%; FP56,794→2,496.25
  (−95.6%). Keep-right FP48,368→763.125 (−98.4%) but precision only0.761→1.437%.
- Native300 recorded3,478 vs proposal steps58,237→2,891.25; compressed positive
  runs3,716→706.5. **Proposal-step count proximity is NOT a lane-event-rate match.**
  Upstream lane0 proposed-step rate650.0→12.28 vs recorded11.01/veh-h; fitted
  positive-run rate only0.538. Recall/rate/timing remain inadequate for adoption.
- Native300 fair-K8 event Brier.043621→.002436 vs never-change.002579;
  empirical K8 Brier.002648 (finite-sample penalty). No calibration/CI claim.
- Per-motivation precision/all-event recall contributions: `fit_v2/motivation.csv`;
  per-lane/segment rates: `lane_segment.csv`; timing curves: `timing_10s.csv`.
  Recorded motivation labels do not exist, so true class recalls are unidentified.
- Target reconciliation:4,432/4,439 scored HDV events match independent target
  table, max position difference.005m; seven absent events are the known300.5s
  first-tape-frame boundary. All other events match. Raw receipts retained.
- Exposure: prior rules/morning/handoffs/noon freeze and aggregate evidence read;
  new diagnosis ONLY FIT969400–969431. No reserved outcome files opened for
  round-2 selection. No WMDATA2 access; respect INT exposure quarantine.
- **Entrants-ON native development12 K2/H300 COMPLETE48/48**, both variants warm,
  common IDM/arrival draws; LC block only factor. All finite/conserved/no_slot=0.
  Actual recorded1,554 changes vs default1,986/fitted1,048.5; count bias+36→−42.125
  per world, MAE36→44.625, empirical CRPS33.125→41.3542 (fair29.4167→37.75).
  Mainline upstream/merge/downstream rates: real16.456/16.718/7.893 per veh-h;
  default26.046/13.899/9.017; fitted10.494/8.070/3.849. Under-changing remains.
- `free_v1/RESULTS.json` + lane_rate_10s.csv bind target comparison, all48 futures,
  exact common draws, exposure and source. K2/singleton cells: no CI/qualification.
  This is FULL-v4-derived lane-only evidence, not integrated b2df7458 forecasts.
- Count scorer excludes four initial internal→normal normalizations per variant,
  proven by12 ONE-step prefix replays with identical posterior/arrival hashes;
  physics unchanged. Independent first-frame all-vehicle target audit restored0
  missing events on these12worlds. Raw forecasts retained. Guard/scorer4CPU PASS.
- Resources: local NVIDIA driver unavailable; gpuA GPU99% occupied at admission,
  so all new jobs CPU-only/nice19/single-thread. Every owned worker EXITED; no signals.
- INT composition: apply CAL spread wrapper after LC attachment (or retain its
  subclass); INT already identified this intake requirement. New evidence does
  not alter the tested dynamics. INT's integrated engineering READY signal is
  separate; this adverse component result is shared before final-panel selection.
  Full fivearm/distributional gate and final reserved comparison remain INT-owned.

## Preserved round-1 handoff

# S-RB-INT FINAL — 2026-11-01

Branch `c4/rb-int`. Retain all prior work and immutable jobs. Earlier detail:
`docs/paper_results/WMPHYS_INT_HANDOFF_ARCHIVE_{0034,0515}.md`.
Owner schedule: interim04/08, freeze09:30, evaluation through11:30, report11:45.
FINAL report59lines: `docs/paper_results/WMPHYS_MORNING.md`; delivery/provenance
`docs/rbint/FINAL_DELIVERY.md`. Actual assembly11:46:40 (100s late); no inference rerun.
WM2/A2/FULL-v4 each20complete worlds;B2=19,C2=12;all candidates reserved8 COMPLETE.
FULL-v4 point switchingFAIL;53CPU analysis/orchestration +10final report checks PASS.
Fresh-session recovery: `docs/rbint/RESUME_0627.md`. Original collectors and
completed scores preserved;08:26round recovered after isolated CPU child exit1.
Peer observer retained. WM2-HDV60 now160/160 COMPLETE; no cpuQ dependency.
08interim COMPLETE: `docs/rbint/INTERIM_0800.md`, all8reserved FULL-v4 K8 worlds.
CRPS−393/−3040/−11647;HDV60−15.26m;300srate-distance+5.63pp→point switching FAIL.
Full-world CIs NA/singletons. Freeze `docs/rbint/FREEZE_0930.md`; current publication
`noon/20261101T093000+0800`. Collector receipt `runs/rbint_noon_v7/PROCESS.json`.

## READY FOR EVAL / existing ownership
- FULL-v4-gpu FROZEN/160tasks/20worlds COMPLETE source22cc32f65a80af482a1a277c5f22377fa727d0a3:
  gpuA `/root/rbeval20261031/phys_code_22cc32f65a80/configs/wm_phys.toml`.
  Prior398268/398269/398270+monitor398303 all EXITED; root `/root/rbeval20261031/int/WM-PHYS-FULL-v4-gpu`.
  Canary370.48s reused. Final LANE +move→LC→emit source phase; do not redispatch.
- FULL-v3-gpu OPERATIONALLY RETIRED07:03, sourcef378ac6033079bdd87c0059627e27e394a1d7cbe:
  gpuA `/root/rbeval20261031/phys_code_f378ac603307/configs/wm_phys.toml`.
  Own370037/370038/370039, monitor370088; root `/root/rbeval20261031/int/WM-PHYS-FULL-v3-gpu`.
  Canary223.28s reused; causal motion posterior, learned arrivals, physics-native
  CAVs and fast lane arbitration. Original observed-ID60s endpoints retained.
- FULL-v1 OPERATIONALLY RETIRED07:03 source4015b641; prior333463/333464/333465, monitor335983,
  gpuA `/root/rbeval20261031/int/WM-PHYS-FULL-v1`. Saved-ID crosswalk unverified;
  newer FULL-v3 removes that dependence. Never redispatch either source.
- All8reserved K8 worlds retained for EACH retired source;81/87total completed
  tasks preserved. Exact-own pidfd stop receipts `docs/rbint/SUPERSEDED_STOPPED_0703.md`.
  Six old GPU workers released for latest FULL-v4 throughput; no efficacy-selected stop.
- A2 owned by RUNA; B2 owned by RUNB; C2 handover below. Shared claims bind ownership.
- C2 now exclusively INT-owned on gpuR; original4015b641 source+smoke hashes
  verified; NEW `/root/rbeval20261031/int/WM-PHYS-C2`. CanaryPASS318.98s,reusedONCE;
  EXITED843902/843903/843904+monitor843928 at own cutoff;102tasks/12K8worlds retained.
  All8reserved complete; no redispatch. `docs/rbint/FINAL_DELIVERY.md`.
  `E:/agent-work/rb-shared/evalclaims/WM-PHYS-A2.runa.json`
  `E:/agent-work/rb-shared/evalclaims/WM-PHYS-C2.runa.json`
  `E:/agent-work/rb-shared/evalclaims/WM-PHYS-B2.runb.json`
  Old failed A/C sources and all STOPs retained. No peer process action.
- WM2-HDV60 COMPLETE160/160 gpuR, original WM2 runtime, short120step prefix-matched protocol;
  own812962/812963/812964, monitor814672; root `/root/rbeval20261031/int/WM2-HDV60`.
  All original workers/monitor EXITED; all three COMPLETE markers, no STOP.
  Original WM2 H600 panel/finalizer remain EVAL-owned. SUMOCORE144/144 complete.

## Current candidate engineering
- Merged LANE b8037c1d/aefc4bd9: signed keep-right/cooperative early returns,
  float32 endpoint normalization, urgency/message presence and blending.
  Three-value step output reconciled with INT ramp ownership and fast arbitration.
  Targeted66CPU PASS/1CUDA skip. Full lane precision still poor (default3.006%,
  encoder2.400%; recall37.942%/40.492%, conditional FIT32/fivearms).
- MERGE3ce7aa6a intake: post-movement source phase available; retain INT's
  default-OFF exact merge query switch. Merge/phase/receipt/combined/query suite
  59CPU PASS/1CUDA skip. Multi-step evaluation wiring now explicit below.
- NEW explicit recipe rollout.departure_phase='after_movement_and_lanes' now
  selects inference-only multi-step adapter (shared model default unchanged).
  Carries settings/history/absolute controller clock; repairs cost observer's
  old-road cohort, preserving exit speed/no birth charge and source ordinals.
  Phase/recipe/source suite27CPU PASS;8x60s recorded smoke+fivearm600s canary PASS.
- Complete touched-module CPU gate287PASS/4skips/60.55s (two local CUDA,
  optional SciPy prefix diagnostic, unfetched historical GPU canary).
- CF47b7742c and UNC84ee27b9 already integrated; FULL uses frozen
  FIT shrinkage_v1 population and causal condition_motion, not saved tape IDs.
  Learned strength19.35568 already applied. Fitted LC remains OFF.
- All recipes SUMO IDM/LC2013/highway merge; learned motion/lane/merge/courtesy/
  admission/noise corrections OFF. A encoder+uncertainty, B population,
  C fitted population point, FULL fitted joint posterior+learned Gamma-Poisson arrivals.
- Current SUMO-merge HDV integrator ALREADY uses Euler; CAVs retain registered
  executor plant. Legacy/checkpoint paths default unchanged. No training tonight.
- DECIDEe8302db3 intake: explicit matched --worlds, selected-switch reliability,
  reverse preference directions/ties and coverage visible in CREDIBILITY.md.

## Evidence and next actions
- Recovered100episode factual60s observable targets, committed ee6bb6d2:
  `docs/rbint/hdv60/FACTUAL_065e9853.json`. Extraction not repeated.
  Exact t0 identity/junction carry matching; future targets isolated from forecasts.
  Existing Miniconda endpoint/protocol suite9PASS; paired common-K8 scorer delivered.
- ~04 report:4paired reserved worlds, FULL-v1−WM2 CRPS−785/−3512/−11041veh-s;
  300s rate-distance+1.25pp. CORRECTION05:25: old CIs retracted; unscored same-cell
  peers must not create bootstrap variation. Corrected singleton-stratum CIs NA.
- Existing20min read-only peer observer retained (`runs/rbint_poll`); do not start
  another observer. Both GPUs busy in last audit; preserve live evaluation jobs.
- FULL-v4 canary395510/source22cc32f6 PASS370.48s/reused ONCE; panel now
  LIVE gpuA own398268/398269/398270 +monitor398303; shared exclusive claim.
- NEW paired FULL-v3 endpoints/delay, actual snapshot05:15:55 (path label0545):
  HDV60 mean-world-median error−33.94m,574common endpoints;
  CRPS120/300/600−573/−3359/−11150; rate-distance300+2.5pp (worse point).
  ALL paired95%world CIs NA after scored-world-only fixed-stratum correction:
  four complete worlds are distinct singleton cells. Old0545 report CIs RETRACTED;
  use paired_0525_corrected outputs. NOT switch eligible. No inference repeated.
  Endpoint/protocol/comparator suite18PASS; immutable manifests/attrition retained.
  WM2 full panel160/160tasks fetched read-only. No inference repeated.
- A2 snapshot6complete reserved worlds:ΔCRPS−501/−2591/−9243veh-s,300srate-
  distance−3.75pp. B2 snapshot1world969433:−2492/−8183/−19790,rate-distance−2.5pp.
  All paired world CIs NA/singleton strata; endpoints absent. C2 still pending.
  Reports docs/rbint/variants/{A2,B2}_0528. Different available sets, not rankings.
- Next: collect complete K8 matched worlds, scored-world-only endpoint CIs and
  DECIDE credibility, update ≤60line WMPHYS_MORNING.md at08:00/freeze09:30/final11:45.
- Full-world CI presentation also requires EVERY contributing scored cell to
  have replication; mixed singleton+replicated bootstrap intervals remain only
  conditional diagnostics. `docs/rbint/FULL_WORLD_UNCERTAINTY_0650.md`;9CPU PASS.
- Remaining parity gaps: synchronous LC, incomplete serial negotiation/internal
  junction state and cold lane memory; no full distributional qualification.
- Bounded local noon collector delivered: read-only snapshots,
  complete-K8-only cached scoring, source-specific endpoint/credibility/traffic
  reports,08:00interim/09:30engineering-freeze/11:35final collection by11:45.
  Foreign-work/branch/staged/source guards; commits only own generated files.
  tests/test_rbint_noon.py+endpoint/bootstrap/credibility gate36PASS/1historical skip.
- First complete collection/scoring publication retained at docs/rbint/noon/
  20261101T054124+0800. Local39728 stopped only at Windows allowed-path guard,
  BEFORE staging; all completed analyses preserved. Diagnosed v3 recovery uses
  --resume-publication and never repeats unchanged complete-ensemble scores.

<!-- NOON-COLLECTOR:BEGIN -->
## Final publication
Final report assembled from completed score components and verified frozen-source cache.
Publication `docs/rbint/noon/20261101T113555+0800/PUBLICATION.json`; FINAL_ASSEMBLY.json records actual delivery time.
No remaining collector daemon; no new inference/scoring. Point switching rule FAIL; full-world CIs NA.
FULL-v4 complete20; actual A/B/C coverage in final publication. All original failures/partial transfers retained.
<!-- NOON-COLLECTOR:END -->
