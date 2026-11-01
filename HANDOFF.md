# S-RB2-LPHYS — DELIVERED, 2026-11-01 ~13:20 +08

Fresh-session closure 2026-11-02: `docs/rb2lphys/RESUME_VERIFIED_20261102.md`.
Clean recovery at a9e4b65c; artifact/archive lineage verifier PASS. Completed
development evidence and default-OFF recommendation stand. INT acknowledges
engineering intake; its latest handoff records an explicit zero-admission HOLD.
No new fitting, inference, dispatch or process intervention during recovery.

Branch `c4/rb2-lphys`; frozen engineering base22cc32f6, inherited HEAD03c81f91.
**Engineering ready for INT ablation; NOT a qualified congestion/credibility repair.**
Report `docs/paper_results/RB2_LPHYS.md` (≤40lines); exact before/after table
`docs/rb2lphys/comparison/before_after.csv`, full COMPARISON.json/effects/world rows.

## Exact implementation / switch / artifact
- Commits fd768d54/cef6321c/06eaf9b3/1c5722e9/f6cf52ab, already INT-ingested.
- Final numerical delivery5e8e23f7 plus sealed-byte integrity followup21c12039;
  take both evidence commits. Physical code/parameter values remain f6cf52ab.
- Default-OFF `physics_params='learned_v1'`, explicit recipe-relative
  `physics_params_file='artifacts/rb2lphys/learned_v1.json'`.
- `configs/wm_phys_lp.toml` is exactly baseline +two physical fields. For FULL-v5,
  apply those same fields to INT's integrated recipe; preserve its LC/CAL adapters.
- Global first-lane0/mainline-last0/ramp-last0.011075188190315551. Outputs only
  documented SUMO departLane/departSpeed settings, fixed by future/world/source
  ordinal through retries/actions. No rule-form or acceleration-residual learning.
- Canonical artifact SHA256 e1387b5fa606b3fa8c51050dab69c5935cbf0bc1468769e6fb948f0e7d7e366f.
  Evaluated Windows-archive CRLF SHA20a4d7774d365d2ec1915c749f3159d17813b9726504fdac8704738c1e18286a.
  `docs/rb2lphys/DELIVERY_AUDIT_FINAL.json` verifies equal JSON/rule source modulo endings;
  artifact attributes pin LF; sealed results preserve exact raw bytes. Hashes are separated.

## Development result / readiness limits
- GP Bayesian optimisation57trials/16common support draws;1068native prefix births.
  Later development2059birth objective0.08274163→0.08279937 (slightly worse).
- Autonomous12worlds/fivearms/K2/H600 COMPLETE:48tasks/240arm-futures; same K8
  latent plans, first2scored as K2. Every task finite/conserved; CRN/source checked.
- Delay CRPS120/300/600:841.09→841.43 /3705.55→3696.28 /14034.18→14024.29veh-s.
  600s bias12820.87→12816.69;300/600rates94.17/100% UNCHANGED vs real61.67/90%.
- 600s upstream speed/onset/range coverage worsen slightly; lane counts nearly
  unchanged. Tiny delay gains pass the narrow point screen, not the broader goal.
  Keep default OFF; retain explicit FULL-v5-LP comparison. Singleton95%CIs NA.
- Ramp-lane road loss and ramp-source waiting are separately reported; full
  ramp-origin delay remains unscored. These results are FULL-v4-source +LP,
  not integrated FULL-v5 forecasts or final-gate evidence.

## Exposure / ownership / resources
- Fit: native969400–969431240..300s; diagnosis: same32native240..600s;
  selection/autonomous comparison:969420–969431all5arms300..900s. No other new TRAIN tapes.
- No reserved969432–969439 or WMDATA2 fitting/selection. Required prior aggregates
  read; canonical inherited reference bundle has other rows, excluded from scores.
  Existing encoder, FIT shrinkage_v1 joint support/posteriors/prior count and
  causal learned-arrival fit unchanged. No black-box/SUMO/neural training.
- FIT/LANE own IDM/fiveLC settings; CAL owns physical spread. No double-learn.
- 31CPU PASS/1CUDA skip;14remote checks PASS; canaries reused once. `QA.json`.
- Own cpuQ1188078/1197634/1218949, gpuR891804/892284/892295/892594 and coordinator
  893753 plus all its children EXITED. gpuR13:14audit GPU0%/1MiB, load7.99/224.
- Results retained at `/root/autodl-tmp/c4/rb2lphys20261101/lphys_panel`; root-volume
  canaries retained too. No peer process/file action, no deletions/moves.
- observations.py untouched. INT owns final exposure release, FULL-v5/LP freeze,
  integrated smoke, registered replicated evaluation, CIs and teacher authorization.

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
