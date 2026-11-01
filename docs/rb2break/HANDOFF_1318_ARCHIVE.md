# S-RB2-BREAK — archived running notes through13:18, 2026-11-01

Owner: S-RB2-BREAK, branch `c4/rb2-break`; FULL-v4 base22cc32f6.
Development-only plan `docs/rb2break/PLAN.md`; coordination
`E:/agent-work/rb-shared/RB2_BREAK_TO_LANE_INT.md`.
Version `6b2d4c01`: guarded all5-arm H600 inference/one-factor adapter.
Baseline96development tasks fetched with hash-bound archive (a prior scp snapshot
was truncated; retained separately, never scored). Baseline report
`docs/rb2break/BASELINE_DEV.json`:600s bias+13,485.24 =road+2,154.35 +source+11,330.89.
Accounting share84% source is NOT a causal attribution. New runner reproduces
969420/future0 clocks/speed/event metrics exactly (`PARITY_DEV.json`).
No candidate or improvement claimed yet. Mainline-LC/merge-advice screens active.
Reserved969432–969439 forbidden to this workstream. Learned corrections OFF.

## Round2 physical correction (screen pending)
Core correction commit `8addc19e` (subsequent orchestration1c5fe0cf).
`physics.sumo_cf_lane_speed_limits=true` (default OFF) maps inferred reference v0
to CURRENT lane limit for longitudinal SUMO CF. Network ramp approach22.22m/s
was already mapped in insertion/merge advice but missing in CF. Source
`MSCFModel_IDM.cpp:131–133` uses current-lane getVehicleMaxSpeed.
Persistent sampled settings unchanged; mainline/acceleration-lane output unchanged.
Regression+departure+motion-conditioning+guard13PASS (existing local Miniconda).
Source fix is available for integration review; do not claim traffic improvement
until matched development inference completes. No change to frozen recipes yet.

Arrival receipt audit `SOURCE_AUDIT_DEV.json`: mean K8 mainline intensity draw
adds only0.713 expected requests over600s vs causal count/exposure (ramp−0.109).
Jeffreys prior alone adds1 request/entrance in600s. This is not an11kveh-s
direct arrival-count offset. Native mainline admissions average260–290/600s
across these worlds; source service / induced queue dynamics need isolation.
Current own queued workers continue the predeclared remaining6 factors serially,
max2GPU workers, inference only. Immutable roots under `/root/rbeval20261031/rb2break`.
Engineering repair12:24: `queue.py` shadowed stdlib queue when executing run.py
as a script. The CPU correction canary failed before importing Torch/inference;
no result exists. Old sleeping supervisors439275/439291 retired via verified
own pidfds BEFORE dispatching factors. Renamed `factor_queue.py`, module entry
(`-m tools.rb2break.run`) avoids shadowing. Initial6b2d4c01 GPU screens unaffected.
Correction CPU run requires matched same-device baseline; no CPU-vs-GPU ablation
will be interpreted as a one-factor effect. Both failed source/output preserved.

## Completed ablations12:40 (K1,12dev worlds, all5arms)
`LC_COMPLETE.json`: no mainline lateral execution adds11,219.37veh-s at600s
vs matchedfuture0 baseline (source only+26.46, road+11,192.91).300s event rate
96.67→100%; upstream speed error worsens. Do NOT disable mainline LC.
`MERGE_COMPLETE.json`: no merge negotiation/advice adds3,436.34veh-s at600s
(source+1.53, road+3,434.81). Keep lane-end stopping/acceptance in both arms.
Do NOT disable merge speed advice. These interventions don't explain source bias.

Additional optional CAV consistency helper (unit tested, traffic screen pending):
`variants.FULL.match_cav_execution_settings=true` binds admission/initial/lane
settings to the SAME registered motion-fitted CAV artifact already forced by the
executor. No controller/plant/HDV/RNG change. Default OFF. Use separately from
lane-limit flag to preserve one-factor attribution. Tests:14targetedPASS;
separate query/parity suite9PASS/2CUDA skips; CF suite28PASS.

CPU full-matrix pair was operationally retired after12min/no completed task,
exact own pidfds440933/440949; all logs retained. Next SAME-device pair uses
existing independently parity-tested wm_phys_cpu recipe (fast exact merge queries).
Current source/metadata/evidence under docs/rb2break; no reserved or WMDATA2 data.

##13:15 continuation
- `DEPART_FIRST_COMPLETE.json`: one-lane departure reduces600s event rate100→90%
  but increases total bias14,980→44,024veh-s, mostly source waiting. Reject.
- `DEPART_LAST_COMPLETE.json`: last-speed departure shifts source bias+12,339→−33,595
  and road+2,640→+17,686, total+14,980→−15,908;600s CRPS worse16,726→17,851.
  It also disagrees with documented scenario departSpeed=max. Diagnostic only;
  a pooled rate/delay match would be cancellation, not a faithful correction.
- Motion-only birth cohort fa4a2793:300fit drivers/35,707rows;154later-birth
  validation drivers/13,635rows,910births matched/0unmatched. Mean-world physical
  acceleration CRPS0.163291→0.165448m/s²: predeclared motion gateFAIL, no traffic
  inference/adoption. All settings on SAME frozen949-point joint support.
- CPU-fast pair+raw/CAV CPU jobs also exceeded30min without a completed task;
  exact four own pidfds retired, no efficacy read, all roots/logs preserved.
  Move these factors to bounded GPU queue AFTER initial source screens; no
  added GPU concurrency. CPU results are absent and never mixed with CUDA.

## Inherited round1 handoff (provenance; not current BREAK process ownership)

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
