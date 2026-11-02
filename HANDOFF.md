# S-RB2-CAL — partial development delivery, recovered 2026-11-03

## RECOVERED Nov3 03:17 +08 — owner-stopped; cache-only completion

Supersedes historical LIVE collector/CPU assertions below. Fresh-session audit:
`docs/paper_results/rb2cal/RECOVERY_20261103.md`. MAIN registration Nov2 02:28
records owner-stopping CAL448399/448400 for S-COOPMWM priority. Direct read-only
gpuA audit: both ABSENT; local33284 absent; original collector terminal is BLOCKED
after SSH failure. Existing57/96futures fetched/transport-hash checked: six complete
K8devworlds969420–969425; nine other futures retained but excluded from scoring.
No new inference/dispatch, signals or fit. Partial cached evidence is published in
`rb2cal/dev_physical_v1_cpu_partial/RESULTS.json` (all paths below relative to
`docs/paper_results`). Complete6only; baseline matched to same6 saved observations.
Physical v1 paired90%coverage120/300/600=51.7/51.7/50.0%; raw selected0.8calls
win68.8/70.0/85.7% vs forecast96.1/96.3/95.5%, n16/10/14. Still overconfident;
no outcome-coverage qualification. CPU-vs-GPU comparison is descriptive only.
Final receipt `rb2cal/PARTIAL_COMPLETION.json`; concise full-metric report
`rb2cal/DEVELOPMENT_PHYSICAL_V1_CPU_PARTIAL.md`. Recovery evidence813ee355,
auditc6f49148. Verification:57taskhashes/48population-conservation checks PASS;
complete-world support/frozen artifact match/allCIs suppressed PASS. Combined
calibration+recovery20CPU PASS38.84s(one thread); RuffPASS. Physical v1 and original
FULL-v4 probability map stay FROZEN/unqualified; no new map.39missing tasks need
renewed MAIN resource/admission authority and source/claim/deadline reconciliation.
Full dev12 qualification remains incomplete; no replacement collector is running.

Branch `c4/rb2-cal`, base frozen FULL-v4 `22cc32f6`. Design frozen before fit:
`docs/paper_results/RB2_CAL_DESIGN.md`. Physical v1 IMPLEMENTED; parameters:
posterior retention=1, Gamma-rate variance=1.667235195661008,
entrant Dirichlet prior concentration=0 (condition on causal soft driver counts).
All fitted ONLY on native FIT969400–969431; dev outcome diagnostics969420–969431.
Artifact `docs/paper_results/rb2cal/v1/fit/calibration.json`; explicit recipe
`configs/wm_phys_rb2_cal_v1.toml`, switch `variants.FULL.spread_calibration_json`.
Runtime source `src/wms/control/spread_calibration.py`; minimal hooks in
`hidden_state.BirthMix`, `rbint/recipe.py`, identity binding in `rbint/evaluate.py`.
Present drivers stay VID-keyed; one future-owned Dirichlet entrant mixture;
request/admission ordinal coupling and fixed joint physical support preserved.
79 distinct CPU tests PASS (74combined +4finite-collector +1empty-call report);
Ruff PASS. Empty selected sets retain NA scores/0calls rather than fake wins.
No neural training; no per-step noise or output-cost corrections are deployed.

FIT115652 later-prefix motion rows/2460drivers. All32 scalar crossfits retain
posterior=1 and entrant concentration=0; rate variance range1.42916–1.75267.
Count NLL3.746607→3.724368 in fit, but3.746607→3.745985 world-excluded:
**very weak crossfit improvement, no coverage/decision qualification**.
World-excluded count CRPS6.02330→6.05365vehicles (worse); 50/80/90% coverage
51.6/75.0/85.9→62.5/84.4/89.1%; exact discrete nominal masses are in
`v1/COUNT_VALIDATION.json`. These are request counts, not team-delay coverage.
Dev baseline and outcome-scale limitation check:
`docs/paper_results/rb2cal/DEVELOPMENT_BASELINE.md` + adjacent JSON;
proper-score-fitted outcome scales1.498/1.219/1.579 at120/300/600s.
OOF paired90% coverage only71.7/65.8/68.3%; a scalar cannot solve the task.
These output-scale diagnostics ARE NOT simulation or a selectable model.
Reliability SVG/CSV and exact summaries in `rb2cal/dev_fullv4_v1/`.
Code/evidence commit `0711a026` (design `c20b82ac`); probability/audit/collector
commit `0782c9bf`. Later strict malformed-posterior validation adds no change on
the valid frozen artifact; the running physical comparison stays source0711a026.
Dev12-only GPU canary
PASS187.57s, own438985 EXITED, runtime
`/root/rbeval20261031/rb2cal/code_0711a02646bb`, output
`/root/rbeval20261031/rb2cal/dev_v1`; all five arms/H600, world969420/k0.
Shared GPU subsequently99% busy with peer jobs; no GPU panel launched.
NEW CPU canary LIVE own440092/start_ticks888711700, same immutable source,
output `/root/rbeval20261031/rb2cal/dev_v1_cpu`. Separate model identity/device;
do not mix its futures with the GPU canary. Quota audit20cores/3.31used,
19.36GiBfree; CPU continuation will use two nice19/one-thread workers.
Own shared claim `E:/agent-work/rb-shared/evalclaims/S-RB2-CAL-dev-v1.json`.
Fresh physical outcome results not yet available; do not adopt v1 as qualified.

Separate source-bound probability map/API: `tools/rbcal/probability.py`, evidence
`rb2cal/probability_fullv4_v1/`. Lambdas120/300/600=.553247/.355783/.539945.
All-pair OOF Brier .23841/.26875/.24245→.22044/.24498/.22503.
The full-data map implies NO0.8switches. OOF selected calls2/0/1, wins2/NA/0;
**not useful-switch qualification**. Full-v4 map MUST NOT be applied to the new
physical candidate or integrated physics: source fingerprints reject mismatch.
Refit only on new dev12 forecasts after integration. `rb2cal/INTEGRATION.md`
documents APIs, switches, exact exposure limits and final CI requirements.
K8 central intervals remain descriptive: continuous iid range reference7/9,
not90%; saved central/range coverage and proper scores stay separate.

Finite CPU continuation/collection tool `tools/rbcal/finish.py` is prepared;
it waits for successful own CPU canary, launches its two-worker panel ONCE,
then only after all96tasks fetches/validates/scores, refits the probability map
and publishes `rb2cal/COMPLETION.json` + `DEVELOPMENT_PHYSICAL_V1_CPU.md`.
Deadline Nov2 09:00+08; no scientific retries or peer PID control. Collector
LIVE local33284/create_filetime134379810406680562; exclusive receipts in
`runs/rb2cal_completion_cpu_v1/{PROCESS,START,TERMINAL}.json` and `poll.jsonl`.
First poll verified CPU canary alive,0tasks,noSTOP; no GPU continuation.
Until completion no nominal traffic-coverage claim. Integrator can read
`rb2cal/COMPLETION.json` after collection; no second collector is needed.
No reserved raw tasks opened. Owner-required historical aggregate reports were
read (disclosed exposure); they cannot qualify an independent final comparison.
Local NVIDIA driver unavailable; gpuA12:02 audit idle (49139MiB free), cpuQ
single nice19/one-thread fit COMPLETE (own1181880/terminalrc0). Canary launch
rechecked GPU free46119MiB, preserving shared teacher capacity. Existing code/evidence
below is inherited INT history, not this worker's current result.

---

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
