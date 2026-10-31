# Archived integrator and MERGE intake notes, 05:15
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
- LANE b8037c1d/aefc4bd9 integrated for NEXT candidate: signed keep-right,
  cooperative early returns, represented endpoint normalization and speed-message
  presence/blending. Three-value step API reconciled with INT's ramp merge owner
  and exact fast arbitration. Current running immutable sources unchanged.
  Targeted lane/geometry/validation/combined/fast suite66CPU PASS/1CUDA skip;
  new advice feedback still requires recorded smoke before evaluation publication.
## Incoming MERGE handoff (retained verbatim)
# S-RB-MERGE — integration handoff / owned validation complete

Branch `c4/rb-merge`, basef0db3c82. Prior work fully retained; see
`docs/rb_merge/HANDOFF_HISTORY_{0100,0315,FINAL_VALIDATION}.md` and source ledger.

## Status
- **ALL owned inference jobs EXITED. No running dispatch remains.**
  v7fast682380 H600, v8end120750052 H120, v8end600789747 H600,
  v9latest300840607 H300 all32/32 complete. Earlier v3/v4/v5/v6 terminal.
  Do not redispatch completed versions. Old v7 directory is engineering only.
- Hash-verified raw receipts and compact scores under `docs/rb_merge/`.
  v8 H120→H600 saved clocks/events/front traces **32/32 bit-exact PASS**;
  `fullfleet_v8_saved_prefix.json` (not full-trajectory parity).
- Source engineering delivered; **NOT distributionally qualified**. Native-only,
  one point-settings future/world cannot certify rate/timing/action-shift gates.

## Source increments for INT — keep your root handoff / reconciled adapter
- **c8e54017 +d1167648** `model/sumo_departure.step_after_movement`:
  explicit opt-in one-step source order move→LC→emit. Newborns at basePos /
  departSpeed; no birth-tick movement/LC/control. Exact FIFO/ordinal/SourceLedger,
  original padding pool restored, no same-frame exit/birth identity overwrite.
  Returns `source_before`, `source_birth_mask`, `source_exit_mask`,
  `driver_params_after_insertion`; boundary costs use the old-road cohort,
  NOT an entry observer's post-motion snapshot. Reset newborn short/LC history.
  Inferred/fixed settings require a coupled entrant sampler.9 phase tests PASS.
  No shared wms/spec/factory changes; original rollout default stays before.
- **d7be510a /0729e09c** `model/sumo_native` callback + `native_physics_accel`:
  frozen native head's missing-leader dv uses unrelated slot0 speed, causing
  false ramp-CAV standstill. CPU reproduction and FIT969400/CAV187 evidence:
  old s=-1.464/v0 through600, physics-native merges326.5/s52.884. Explicit opt-in,
  same registered executor bounds, no head retraining/modification.
- **3df73a27 /5870b556** network22.22 approach→31.29 accel-lane limits,
  existing inferred v0/referenceLimit factor, weighted BirthMix exact candidate,
  NUMERICAL_EPS=.001/POSITION_EPS=.1 (not SPEED_EPS=.01).
- **9bdc35e5** source low-b IDM minNextSpeed bound; existing frozen encoder
  settings and ordinal-verified newcomer point coupling /600s harness.
- **158d0193** exact selected-ego sequential geometry, O(ramps*slots), full-frame
  heterogeneous/batched bit parity; keep INT's own query fast path if preferred.
- **0090c168** leader time+space/Euler-brakeGap(tau)/unblocked-leader advice.
  Retain INT mainline arbitration; only hooks are `lookahead_speed=state.get(...)`
  and post-motion `out.get('lc2013_lookahead')`.
- Switch `PhysicsConfig.merge_model='sumo'` defaultlegacy; learned merge/courtesy/
  ramp adaptation/admission/motion/noise bypassed. Settings default/motion only.

## Validation outcomes — matched FIT32/native, t0=300, stream190
- **v8end600**, INT4015b641 + source phase/native/frozen encoder, H600:
  ramp mean H120955.17/928.87real, H3002325.14/2272.78,
  H600**4663.06/4670.05veh-s**. W1=155.26/454.97/1553.28.
  H600 tails wrong: predicted q10/50/90=2118/3705/8243 vsreal3058/4513/6103.
  Boundary-augmented t0 **202/202 matched**,0missed/extra; timingMAE43.83s,
  position13.36m. Matched mean is not matched distribution. Report
  `fullfleet_v8end600_boundary_summary.json`; unaugmented report retained.
- **v9latest300**, INT8e964de5 (new CF lane-cap/link chain + LANE raw origins /
  translated positions), same point encoder/post-phase/native, exact CPU flags:
  H120927.56/928.87real, W1=135.83; H300**2187.72/2272.78**, W1=522.28.
  H300 boundary-augmented195/202 t0 matches,7missed,0extra; timingMAE33.55s /
  position12.60m. q10/50/90=977/1785/3795 vsreal1641/1966/3422. Latest geometry
  is NOT an across-metric win over old v8 (H300W1=454.97). No outcome tuning.
  Reports `fullfleet_v9latest300{,_boundary}_summary.json`; H600 not run forv9.
- Both completed panels:32/32 road/source conservation, no lost demand/no_slot,
  no beyond-end merges.1-future/native-only screen, not full EVAL qualification.
- Historical562.4/1265.4 is another panel/clock; matched recorded world clocks
  above are primary. No held/reserved/forbidden tape paths used.

## Target / geometry audits and input contract
- Helper misses2 real first-step300.5s merges:969413/VID144 and969419/VID181.
  Episode/target SHA-bound views audit; `boundary_native_fit32.json`. Original
  helper CSV/old reports unchanged; `--boundary-targets` writes separate reports.
- Recorded t0 audit:4 raw-frame body overlaps (402/405/419/425), **0 in lane-chain
  coordinates**. Only initial projection artifacts proven; later raw overlap
  samples are NOT SUMO collision logs or automatically physically exonerated.
- Reused LANEb192e0c8 extractor / frozen WM2 b8797242…; producer all9 settings
  bit-exact. Source legacy feature/default geometry for encoding, SUMO for rollout.
  Initial HDV settings by public VID /240..300 motion; CAV defaults retained;
  future HDVs frozen zero-history centre. No reimplementation/refit/training.
- Every admission/newborn shares exact ordinals. Entry process requires
  conserve_queue=True/couple_draws=True/row_seeds.31.29 delay reference, no newborn
  road charge, old speed on exit, ramp origin after merge, fractional source wait.

## Verification / scope
- **69 local merge/native/phase/identity/receipt/boundary/prefix tests PASS /7.19s**;
  **58 combined runtime CPU tests PASS** on both staged engine versions.
  Original/augmented targets, source hashes, per-world distributions all retained.
- Committed-byte audit PASS: **9 collections /303 source receipts**, every member
  SHA256 and published collection citation matches Git HEAD. Final check receipt
  `docs/rb_merge/FINAL_VERIFICATION.json` records scope and all terminal own PIDs.
- `docs/rb_merge/SOURCES.md` records public source anchors. CPP read only in NEW
  empty temp dirs, never executed. No GPU/SUMO/training, peer signals, truth or
  assignment reads, newenv, reset/clean/deletion/push. All changes committed.
- Open parity scope: serial LC /complete blocker negotiation /explicit internal
  junction state /calibrated hidden-driver+arrival futures. INT/FIT/UNC/EVAL own
  final FULL qualification. Coordination pointers: `rb-shared/MERGE_*.md`.
