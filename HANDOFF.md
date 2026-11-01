# S-RB-INT handoff — 08:00 interim published, 2026-11-01 08:07 +08

Branch `c4/rb-int`. Retain all prior work and immutable jobs. Earlier detail:
`docs/paper_results/WMPHYS_INT_HANDOFF_ARCHIVE_{0034,0515}.md`.
Owner schedule: interim04/08, freeze09:30, evaluation through11:30, report11:45.
Fresh-session recovery: `docs/rbint/RESUME_0627.md`. Original collectors and
completed scores preserved;08:26round recovered after isolated CPU child exit1.
Peer observer retained. WM2-HDV60 now160/160 COMPLETE; no cpuQ dependency.
08interim COMPLETE: `docs/rbint/INTERIM_0800.md`, all8reserved FULL-v4 K8 worlds.
CRPS−393/−3040/−11647;HDV60−15.26m;300srate-distance+5.63pp→point switching FAIL.
Full-world CIs NA/singletons. Current publication `noon/20261101T075846+0800`.

## READY FOR EVAL / existing ownership
- FULL-v4-gpu LIVE source22cc32f65a80af482a1a277c5f22377fa727d0a3:
  gpuA `/root/rbeval20261031/phys_code_22cc32f65a80/configs/wm_phys.toml`.
  Own398268/398269/398270, monitor398303; root `/root/rbeval20261031/int/WM-PHYS-FULL-v4-gpu`.
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
  LIVE843902/843903/843904+monitor843928; no redispatch. `docs/rbint/C2_LIVE_0645.md`.
  `E:/agent-work/rb-shared/evalclaims/WM-PHYS-A2.runa.json`
  `E:/agent-work/rb-shared/evalclaims/WM-PHYS-C2.runa.json`
  `E:/agent-work/rb-shared/evalclaims/WM-PHYS-B2.runb.json`
  Old failed A/C sources and all STOPs retained. No peer process action.
- WM2-HDV60 LIVE gpuR, original WM2 runtime, short120step prefix-matched protocol;
  own812962/812963/812964, monitor814672; root `/root/rbeval20261031/int/WM2-HDV60`.
  Last05:03 audit45tasks, FULL-v3 34tasks; all workers/monitors live, no STOP.
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
## Bounded noon collector
2026-11-01T09:30:00+08:00; read-only remote collection, no inference dispatch.
Latest publication `docs/rbint/noon/20261101T093000+0800/PUBLICATION.json`; 08interim `docs/rbint/noon/20261101T075846+0800`,09:30freeze `docs/rbint/noon/20261101T093000+0800/FREEZE.json`.
Source22cc32f6 FULL-v4 and prior A/B/C/FULL sources remain immutable.
Scored-world-only fixed-stratum CIs; singleton uncertainty NA; prior CIs retracted.
Existing remote supervisors retain own-PID11:30 deadline. Final collection11:35, report by11:45.
Collector owns ONLY its local CPU analysis children and publication files; foreign work stops it.
<!-- NOON-COLLECTOR:END -->
