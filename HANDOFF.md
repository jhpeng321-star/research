# S-RB-INT handoff

Updated 2026-11-01 05:05 +08; branch `c4/rb-int`. Owner registration5683/5697/5708/5733/5736.
Full earlier history retained in `docs/paper_results/WMPHYS_INT_HANDOFF_ARCHIVE_0034.md`.

## READY FOR EVAL
- **INT-owned FULL-v3-gpu LIVE** sourcef378ac6033079bdd87c0059627e27e394a1d7cbe,
  `/root/rbeval20261031/phys_code_f378ac603307/configs/wm_phys.toml`.
  Own370037/370038/370039 +monitor370088, root `/root/rbeval20261031/int/WM-PHYS-FULL-v3-gpu`.
  Canary969432/k0 PASS223.28s, reused once; original observed-ID60s endpoints saved.
  No peer redispatch; shared FULL-v3-gpu.claim owns this separate source.
- **WM-PHYS-A2 / RUNA gpuA**: `E:/agent-work/rb-shared/evalclaims/WM-PHYS-A2.runa.json`.
- **WM-PHYS-B2 / RUNB gpuR**: `E:/agent-work/rb-shared/evalclaims/WM-PHYS-B2.runb.json`.
- **WM-PHYS-C2 / RUNA gpuA next**: `E:/agent-work/rb-shared/evalclaims/WM-PHYS-C2.runa.json`.
- NEW ready source **4015b6412cdfa2a94168b7095a59287d0fddcd3b**, staged BOTH hosts
  `/root/rbeval20261031/phys_code_4015b6412cdf`; recipe configs/wm_phys.toml.
  Smoke8x60s PASS, SHA8cfb30c91e6d81c9cb4ef2e1d04b7eb6621c36cfa381388533874828037a90eb.
  Real five-arm600s canary969432/k0 PASS206.03s, FULL settings; controller ABI verified.
- **FULL-v1 INT-owned already running gpuA**: shared exclusive claim; do NOT
  redispatch. Own333463/333464/333465, root `/root/rbeval20261031/int/WM-PHYS-FULL-v1`.
  Monitor335983 (source9f35ba4f) live: exact owned incarnations/18.5GiB guard,
  owner11:30 deadline,04/08 complete-K8-only interim + full frozen scorer once.
  `panel_ops status` is read-only; never repeat canary/continue/monitor dispatch.

## What exists
- NEW default-OFF `sumo_lane_conflict_fast`: filter static present/movable/change
  candidates ONCE before the same stable sequential arbitration. Original loop
  performed scalar checks on every padded slot (millions per600s forecast).
  Candidate order/occupancy/ties are unchanged; CPU geometry/coordinate fixed-point
  parity PASS; synthetic640slot CPU kernel24.0x faster. GPU/rollout gate pending.
- CUDA query parity **6/6 PASS /1.65s** on source d279a502. Shared-GPU synthetic
  kernel timing is0.106s dense vs0.114s query (0.94x): no GPU speed win claimed.
  CPU timing was1.150s vs0.0272s. NEW configs/wm_phys_cpu.toml differs ONLY by
  the exact query switch; CPU full-task throughput probe planned before choosing
  execution hardware, with outcome scores unopened. Protocol/config18CPU PASS.
- Runtime posterior profile predictions now use the ACTIVE CF source lane-cap
  chain and deployed lane/link gap geometry, not the older FIT step approximation.
  FIT selector/shrinkage API and learned population remain unchanged. Overspeed
  and cross-link numeric anchors + caller/core suite **27CPU PASS /7.50s**.
- CF47b7742c/2b898f39 integrated: exact lane-cap/planMove chain, relaxed
  emergency stop fallback and all19 applicable independent idmspec vectors.
  Pure-acceleration diagnostic remains default OFF. **65 CPU PASS /1CUDA skip**;
  both legacy/WM2 default-OFF architecture fingerprints still bit-exact.
- UNC84ee27b9 integrated: canonical causal-motion posterior snapshot binding
  (clears saved IDs/LC), observed-counter snapshot hashes/deep copies, stricter
  identity pairing, source-valid optional LC bounds (LC still OFF). INT generic
  wrapper now uses UNC.with_motion_posteriors instead of constructing a parallel
  adapter. **60 CPU tests PASS /11.68s**; no sampling-scale/noise changes.
- Next FULL recipe now selects causal runtime motion conditioning + physics-native
  law (NOT READY until new smoke). Reuses FIT.condition_motion and exact FIT pure
  interval/IDM functions without SciPy; no saved-driver-ID lookup, no population
  refit, no future frames. All current HDVs use their own pre-t0 evidence, with
  learned shrinkage/prior fallback; entrants use learned population.32CPU PASS.
  FULL-v1's saved-ID crosswalk remains unverified; new recipe removes that reliance.
- LANE16db471a/0a1e7e93 integrated: raw upstream lane origins, collapsed-link
  neighbor distances and±1.45m lane-change position translation. Reconciled the
  new (next_lane,next_s) adapter API with INT ramp ownership/mainline arbitration.
  **68 CPU PASS /1CUDA skip**; both default-OFF synthetic architectures remain
  weights/forward/rollout BIT-EXACT to original fingerprint. New recorded smoke next.
- FIT4aee21a8 integrated: MotionFit.condition_motion generic causal posterior
  API (no saved tape-ID lookup); source weights and shrinkage exports preserved.
  New direction-aware/raw LC fitting tools remain diagnostic; LC stays OFF.
  Producer22CPU PASS; real949-profile no-history fallback API gate PASS in INT.
  Latest exact INT-stage SciPy suite pending next staging.
- MERGEda3607eb integrated: source22.22→31.29 ramp limits/speedFactor, EPS
  constants, physics-native opt-in callback and matched native-relative accel.
  Old learned native head's missing-leader dv gathers unrelated slot0 speed:
  can falsely stall an unled ramp CAV. New callback bypasses that head, no training.
  **59 CPU tests PASS /1CUDA skip**, including INT query parity and old adapters.
  Callback not yet selected by the live4015b641 panel or new recipe.
- NEW default-OFF `PhysicsConfig.sumo_merge_query_fast`: exact one-ego geometry
  per reservation rank (junction_query.py). CPU dense/query geometry+reservation
  bit parity PASS;34tests PASS/1CUDA skip. Synthetic640slot kernel benchmark42.3x
  CPU speedup; recorded FULL60s FIT969420/969421 dense-vs-fast COMPLETE tapes
  bit-exact (positions/speeds/lanes/masks/accel/residual). Small closed-cohort
  rollout speedup only1.00–1.03x; padded/GPU speedup unmeasured. Source5a83518e,
  receipt docs/rbint/QUERY_PARITY_5a83518e.json. NOT in live4015b641 runtime.
- Ported ONLY existing frozen MAPPO `policy_native_lanes` ownership option
  from c4/speedhead bridge (default OFF). Needed by the unchanged registered
  callback; policy acceleration preserved, WM native lanes delegated.
- FULL recipe/entrypoint wired and live: fitted
  shrinkage_v1 joint draws + learned entrance-counter/Gamma-rate/Poisson clocks.
  Reads ONLY hash-bound existing start/history exports; numeric vid_index follows
  recorded identity through packed/recycled slots. Unknown posterior IDs use the
  fitted population; reserved-world conditional-motion refit not yet available.
- FIT50f2cd02 + UNCe038da91 integrated: shrinkage strength19.35568 already
  applied, no double shrinkage. Fitted LCv2 stays OFF (loses event-rate baseline).
- MERGE75a24397 integrated (kernel0090c168): Euler brakeGap with tau, overtake
  time+space checks, unblocked-leader cap, persistent lookahead; ONLY two adapter
  keyword hooks, retaining INT's entrant identity and mainline ownership fixes.
- LANE811cd7e8 merged: finite continuation geometry, stale keep-right gate,
  SUMO numerical epsilon; prior immutable A/B/C runtime remains source-bound.
- CF742922b6 + EVAL scorer/finalizer integrated; earlier branches retained in Git.
- DECIDE16dff9ad merged: tools/wmdecide strict paired0.8 win rule, calibration/
  reliability/coverage reports. **15 CPU tests PASS /4.38s** in INT worktree.
- A: existing motion encoder EXACT, hidden-driver joint draws + Poisson arrivals.
- B: inherited physics population point, Poisson arrivals; not encoder centre.
- C: motion-only pooled_full_v5 point fit (JSON SHA bound), Poisson arrivals.
  CAV plant/settings retained; adverse fitted LC is NOT adopted.
- All variants: cf_model=sumo_idm, lane_model=lc2013, merge_model=sumo;
  no learned acceleration/lane/merge/courtesy/admission/noise corrections or training.
- Source fixes: EPS=.001, POSITION_EPS=.1, passenger emergencyDecel9;
  HDV Euler positions; no missing-leader 250m phantom/no duplicate ramp stop.
- Entrant admission and newborn motion use EXACT same weighted ordinal-keyed draw.
- `tools/rbint/{recipe,evaluate_A,evaluate_B,evaluate_C,evaluate_FULL,panel_ops}.py`.
  Full-population ledger/exit/finite audit attached to every candidate task.

## Verification / receipts
- **~04:00 interim published**: four paired complete reserved K8 worlds
  969432/433/434/436; FULL-v1−WM2 CRPS120/300/600=−785/−3512/−11041veh-s,
  paired95%simultaneous world CIs [−1264,−306]/[−5825,−1200]/[−16268,−5815],
  conditional on saved ensembles/available cases. Rate300 distance difference
  +1.25pp[−4.17,+6.67] (worse point); HDV60 CI still missing. NOT switch eligible.
  Report WMPHYS_MORNING.md58lines; exact audit reports/paired_0400/COMPARISON.json.
- Next-candidate local gate **270 PASS /2CUDA skips /21.53s**. Both CUDA
  kernels separately passed6+4 tests on gpuA. Recorded FULL lane-arbitration
  dense/fast COMPLETE tapes are bit-exact on FIT969420/969421 (source8e964de5),
  all six motion/mask/residual fields; recipe now explicitly enables the fast
  lane arbiter. Small-frame CPU speed1.04–1.06x; full GPU latency still pending.
- Latest source95e81e79 recorded A/B/C/FULL CPU-recipe smoke8x60s PASS; FULL
  causal posteriors condition70/79 and39/44 observed HDVs, remaining9/5 use the
  learned population, saved-ID lookup FALSE. Physics-native CAV law exercised.
- CPU v2 throughput canary STOPPED operationally: exact own356826 absent,
 0tasks/~782wall seconds; no efficacy scores inspected, no automatic retry.
  Receipts docs/rbint/full_v2_cpu; old GPU evaluations unchanged.
- Candidate-only lane arbiter CUDA parity4PASS/1.70s; shared-GPU synthetic
 0.534s→0.0303s percall (17.6x). Recorded full-rollout parity/latency gate next.
- Early matched snapshot02:52: FULL-v1=30tasks/3complete reserved worlds;
  WM2=115tasks/13complete worlds, intersection ONLY969432. Mean five-arm CRPS
  difference120/300/600=+59.93/+635.50/−1671.64veh-s; rate300 distance−7.5pp.
  No world-sampling CI from n=1. `docs/rbint/full_v1/reports/paired_0255` binds
  all inputs; tools/rbint/compare.py uses frozen paired-world/maxstat bootstrap,
  conditional on saved ensembles. Comparator3CPU tests PASS. No eligibility claim.
- Current complete local integration gate **215 PASS /1CUDA skip /25.53s**;
  all touched physics/model/hidden-source/recipe/monitor/credibility modules.
  Runtime posterior + physics-native FULL candidate requires fresh recorded smoke.
- Latest complete touched-module CPU suite **155 PASS /18.70s**. Exact staged
  FIT test file on cpuQ **17 PASS /1.54s** (SciPy from existing environment).
- New closed-cohort A/B/C/FULL smoke **8x60s PASS**, gpuA CPU source0ec484ab;
  `docs/paper_results/WMPHYS_SMOKE_0ec484ab.json`. All finite/conserved/no corrections.
  FULL driver identity and learned-arrival conditioning constructors also pass.
- Real all5-controller600s canary now PASS; canary task reused, not rerun.
- FULL26PASS/24.61s: packed-vs-recorded IDs, prior/posterior/CAV preservation,
  paired replay, source conditioning, changed-start/permutation rejection.
- Default-OFF legacy and WM2 synthetic weights/forward/rollout bit-exact.

## Jobs / coordination
- Old A/C claimed/attempted once by RUNA; workers failed before rollout:
  frozen MAPPO callback uses policy_native_lanes unsupported by base bridge.
  Original remote FAILED.json/logs preserved, no completed task to repeat.
  All failed sources/logs/claims retained. A2/B2/C2 use NEW immutable source.
- RUNA watcher44796 and RUNB watcher33076 retained. Old B gpuR728893 waiting
  for resources at last audit; prerequisite note requests owner-side cooperative
  stop before any old-source inference. INT never signals peers.
- Owner00:44 schedule supersedes05:00 freeze: interim04:00/08:00, freeze FULL
  around09:30, full evaluation09:30–11:30, final report with credibility11:45.
- DECIDE credibility scorer ready; run on complete K8 outputs, not single futures.
- EVAL WM2 gpuR workers706825..706828 and finalizer708801 already live; preserve.
- SUMOCORE144/144 complete; all supervisors exited; compact results available.
  Physical validity FAIL (18/144 initial-movement collision flags retained).
- FIT jobs complete; LANE/MERGE diagnostics retain source-specific clocks.
- Shared helper artifacts consumed; raw truth/assignment/prohibited paths untouched.
- Poll workers about every20min. No peer process action, push, deletion or training.

## Remaining / known gaps
- Evaluation-only endpoint target reader implemented: six converted observable
  files, exact unique t0 state matching then persistent VID; causal carried-lane
  recovery handles parallel junction rows without saved-ID assumptions. Future
  junction/absent positions censored. Actual100episode extraction RECOVERED at
  docs/rbint/hdv60/FACTUAL_065e9853.json; no repeat extraction. Endpoint target/
  protocol/export suite9CPU PASS on existing Miniconda environment at05:05.
- WM2-HDV60 now LIVE on ORIGINAL gpuR runtime: own812962/812963/812964,
  root `/root/rbeval20261031/int/WM2-HDV60`. Canary22.59s; verified original
 600s task's all5arms/3regions/6prefix bins before continuation. No baseline
  source/model mutation or600s rerun. Endpoint monitor19CPU tests PASS; short
  schema explicitly excluded from full-distribution scorer. Factual targets next.
- HDV60_DESIGN.md frozen before endpoint inference/scoring. New short baseline
  wrapper imports ORIGINAL WM2 runtime on gpuR, retains600s padding/1200-step
  RNG plans, executes120steps only. Canary must match original first6speed/
  support bins(all5arms/3regions) before continuation. Separate schema/root;
  future targets never enter forecasts. Endpoint/protocol/ownership25CPU PASS.
- New candidate runner exports60s predicted positions by ORIGINAL observed
  string ID, exact halfstep119, with exit/recycled-slot censoring and explicit
  no-factual-future-input provenance. Endpoint exporter2CPU tests PASS. Existing
  old snapshots remain bin/clock-only; baseline short-horizon/factual scorer next.
- RUNA must retain the published loader module so the verified runner resolves.
- Matched distributional efficacy, HDV60s non-regression and paired CIs pending.
- EVAL endpoint gate request: rb-shared/INT_HDV60_GATE_REQUEST.md; bin/clock-only
  completed tasks cannot supply60s HDV positions. Do not infer non-regression.
- LC2013 is highway subset (synchronous messages; incomplete full negotiation).
- Historical lane reproduction and ramp delay are different panels, not efficacy
  evidence for the integrated source-fixed candidate. Sparse8-world cells cannot
  certify full distributional qualification.
- Morning report must explicitly show CRPS120/300/600, breakdown-distance and
  HDV60s differences WITH paired-world CIs under `Switching-eligibility rule`.

## Fresh-session reconciliation 05:05
- Preserved HEAD065e9853 and recovered untracked HDV60 receipts/100episode targets.
- Read-only audit: WM2-HDV60 workers812962/812963/812964 +monitor814672 LIVE,
  45tasks; FULL-v3 workers370037/370038/370039 +monitor370088 LIVE,34tasks.
  No STOP, duplicate launch or process signal. Both GPUs busy; preserve jobs.
- Existing local20minute handoff observer retained; latest04:45 receipt inspected.
  LANE final aefc4bd9 now ready (three-value step_output and advice feedback);
  MERGE post-movement source phase ready. Both need next-candidate integration
  and fresh smoke. CF unchanged. Noon schedule/final09:30 freeze retained.
