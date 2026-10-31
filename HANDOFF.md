# S-RB-INT handoff

Updated 2026-10-31 23:58 +08:00; branch `c4/rb-int`, base `f0db3c82`.

## Status
- CF742922b6 integrated. Source reconciliations are explicit: public
  config.h.cmake:236/239 gives NUMERICAL_EPS=.001/POSITION_EPS=.1; passenger
  emergencyDecel is max(decel,9), recipe explicitly9. Missing leaders use SUMO's
  1e6 free gap; SUMO merge alone owns ramp stopSpeed (no duplicate CF obstacle).
- Full touched-module CPU gate: **159 PASS /33.34s**, including independent
  source anchors T14-17/T19-20, near-stop T08, Euler position, lane/merge/entries,
  hidden futures, legacy native/lane suites, encoder/no-correction audits.
  Default-OFF two-model weights/forward/rollout remain bit-exact after CF merge.
- FIT11d36f75 merged; completed artifacts preserved. C will use the frozen
  pooled_full_v5/idm_pooled.json motion-only point fit + Poisson arrivals, with
  source LC defaults. Fitted LC lost to its event-rate baseline and is NOT adopted.
  Posterior populations are preserved for later qualified uncertainty, not silently
  substituted into A. FIT reports 15 CPU tests PASS on cpuQ, no live fit job remains.
- CF request published at rb-shared/INT_CF_REQUEST.md (00:14): commit existing
  wiring promptly; reconcile shared source EPS0.001/emergencyDecel9 before parity.
- INT candidate adapter exists at tools/rbint/{recipe,evaluate,smoke,stage}.py.
  A uses hidden driver+Poisson futures; B fixes BOTH present/entrant settings to
  inherited physics population defaults (not learned encoder centre), with Poisson
  arrival uncertainty. Strict source checkpoint and encoder copy remain required.
  Await CF worker committed cf_model wiring before recorded smoke/READY publication.
- Fixed cross-worker identity mismatch: SUMO insertion now consumes BirthMix's
  exact weighted candidate rather than recomputing an unrelated uniform draw.
  Wrapped evaluator entries expose birth ordinals/queue config and observe exact
  fractional-step Poisson waiting. 28 integration/sampler/merge CPU tests PASS.
- Fresh resume preserved HEAD841be6ea and untracked read-only remote.py. No INT
  dispatch exists on gpuA; resource check: 48,510MiB free, 0% GPU, load6.40/208.
- EVAL baseline/finalizer and SUMOCORE production already dispatched by owners;
  preserve their receipts/jobs. RUNB waits for READY FOR EVAL, no candidate launched.
- EVAL scorer/finalizer merged; root handoff retained; README preserves both the
  merge diagnostics and distributional evaluator contracts. CF e9992767 now usable;
  UNC1f918969 usable; FIT remains running and must not be duplicated.
- Read owner authorization at active C4 registration line 5682 and all four WHY reports.
- Integrator worktree initially clean. LANE and MERGE integrated at 23:42.
- Initial poll: `c4-rb-{lane,merge,cf,eval}/HANDOFF.md` absent; SUMOCORE handoff absent.
- CPU: 22 logical processors, initial load 40%; local NVIDIA driver unavailable.
- No training, downloaded code, scientific rollout, or peer-process action performed.
- Resume preserved the previous uncommitted handoff and `tools/rbint/`.
- 23:20 poll: LANE and MERGE have source/interface handoffs, implementation pending;
  CF and EVAL handoffs absent. SUMOCORE says no dispatch yet. Shared README read.
- One central REGISTRATION line added at 23:30; committed local receipt:
  `docs/paper_results/WMPHYS_REGISTRATION_RECEIPT.md`.
- EVAL usable tooling merged from `579a0861`; root handoff conflict resolved by
  retaining this INT handoff and preserving EVAL snapshot at
  `docs/paper_results/WMPHYS_EVAL_HANDOFF_579a0861.md`.
- Merged EVAL CPU suite: `py -3.13 -B -m unittest tools.rbeval.test_core -v`,
  4 passed / 5.03 s, including all 1,800 recorded BIC decisions. EVAL owns its
  existing WM2 baseline dispatch/canary; INT will not duplicate those tasks.

## Integration contract for workers
- Please commit usable increments and state exact commit, switch names, test command, and known gaps here in your own handoff.
- Target A/B: SUMO IDM + LC2013 + SUMO merge/insertion; every learned motion/lane/merge correction OFF.
- A keeps the existing motion-history encoder settings; B uses population defaults, including new entrants.
- Loading the existing checkpoint must preserve encoder weights; switches must not silently substitute an untrained encoder.
- Audit findings: `rollout_mode=driver_params` alone is NOT sufficient: long-history residual lane corrections still run (`wms.py:1438`), learned merge/courtesy/yield/ramp terms run independently, and `accel_noise` can still use learned sigma. The rebuilt path must bypass these explicitly.
- Shared-file changes stay opt-in/default-OFF. Existing checkpoint behavior must remain unchanged without the switches.
- Evaluation must bind a fixed permitted FIT panel before scoring; never open prohibited paths or truth assignments.
- Need finite/conserved 60 s rollouts from two recorded FIT starts before the registered A/B distributional panel.
- INT owns `configs/wm_phys.toml`: explicit A=encoder / B=population recipe,
  learned correction families OFF; currently NOT runnable until worker switches
  land. Planned `PhysicsConfig.cf_model='sumo_idm'`, `merge_model='sumo'`,
  `ModelSpec.lane_model='lc2013'`; adapt exact API if worker implementation differs.
- Smoke identities frozen before predictions: native seeds 969420/969421, t0=300,
  60 s each A/B, authentic pre-t0 history, closed cohort (no future mask/teacher).
  Full-panel arrival/source conservation remains a separate required gate.

## Tests / results
- C point-physics adapter now consumes the exact pooled_full_v5 JSON, identity
  binds its SHA, fixes HDV parameters while retaining registered CAV defaults,
  and gives entrants the same point settings. A/B/C recorded smoke is frozen
  before predictions. Four independent recipe tests PASS /12.51s, including
  actual conserved births and unchanged CAV settings; SciPy fitter tests skip
  locally (worker's full15 CPU tests already passed on cpuQ).
- Added full-population exact admission/exit/source ledger audit to candidate
  tasks, and thin A/B entrypoints using the existing runner CLI. Meaningful
  integration test now runs actual births/queue and native CAV execution; 12
  recipe+audit tests PASS /41.99s. This caught missing CAV callback/cap and an
  incorrect residual-log key before any recorded rollout.
- LANE follow-up merged (right-overtake prevention and completed FIT receipts):
  45 touched-module CPU tests PASS /15.61s. Original incorrect test filename
  produced no tests; corrected command includes test_lc2013_validation.py.
- MERGE follow-up integrated: no duplicate lane-end obstacle; retained INT's
  single post-CAV LC2013 arbitration (removed duplicate worker hook); exposed
  final lane probabilities. 42 touched-module CPU tests PASS /15.05s.
- Resume scorer/finalizer suite: 12 run, 11 PASS / 1 optional-dependency skip,
  15.56s. CF kernel e9992767 + UNC1f918969 integrated; independent CPU suite:
  52 PASS /39.27s (CF numeric parity, hidden futures, old futures, INT guards).
- CF worker has its own uncommitted switch wiring in progress; INT will consume
  that committed increment rather than duplicate shared accel/config edits.
- `tools/rbint/default_parity.py`: captured and independently compared two CPU
  synthetic legacy/default + WM2-architecture cases; weights, forward and rollout
  tensors bit-exact. Baseline: `docs/paper_results/WMPHYS_DEFAULT_OFF_BASE.json`.
- Command: `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m tools.rbint.default_parity --compare docs/paper_results/WMPHYS_DEFAULT_OFF_BASE.json`
- Synthetic long-history fixture explicitly masks all history as missing; this is
  compatibility evidence only, NOT the recorded FIT smoke or an efficacy result.
- `tools/rbint/audit.py`: closed-cohort conservation/finite/zero-residual gate and
  temporary hooks rejecting execution of parameterized corrections (encoder allowed).
  `py -3.13 -m pytest -q tests/test_rbint_audit.py`: 9 passed (5.70 s).
- `tools/rbint/rebuild.py`: strict-loaded source -> opt-in rebuilt architecture;
  exact motion-encoder key/shape/value preservation, unchanged source model and
  caller RNG, explicit unused/initialized-head receipt. No checkpoint rewriting.
  `py -3.13 -m pytest -q tests/test_rbint_rebuild.py`: 5 passed (5.80 s).
- Existing local `py -3.13` has torch `2.15.0+cpu`; default Python 3.14 is not the selected runtime.
- Remote resource check: gpuR RTX4090 24,092 MiB free, 0% utilization; cpuQ load 0.94 / 32 logical CPUs, ~61 GB available. Recheck immediately before dispatch.
- Resume local load 68%; parity check used a single CPU thread. No running INT
  rollout/dispatch receipt exists in this worktree; no scientific job launched.

## Known gaps
- Worker APIs and registered evaluation panel are not yet available at first poll.
- No measured WM-PHYS efficacy; historical WM2 statistics are references, not matched-panel results.
- EVAL freeze now available: `c4-rb-eval/docs/paper_results/RBEVAL_DESIGN.md`,
  seeds 969420–969439, native/ctrl0/mappo/pi/circles, K8, t0=300, H=600.
- Worker request: expose checkpoint-preserving construction/runtime switches; CF
  must bypass parameterized residual/temporal/gate/lane branches, not merely zero
  final acceleration. INT will run `forbid_learned_corrections` on recorded smoke.
- LANE/MERGE ownership needs explicit reconciliation: LANE handoff proposes full
  LC2013 lane model; MERGE owns ramp 2->1. Mainline proposals must survive merge
  resolution and carry LC2013 persistent state across rollout chunks.

## Fresh resume 23:36
- Preserved clean HEAD `6dc3c8a0`; prior registration already exists (no duplicate).
- Polled all six workers plus SUMOCORE/shared helpers. LANE `1b88408e` and
  MERGE `f2355d13` now usable; CF still base/no handoff. UNC contract `7b6f26dc`
  and FIT contract `57f4f3ef` exist, implementations pending. Targets helper DONE:
  observable FIT lane/merge/insertion/ramp/speed tables available in rb-shared.
- Read-only gpuR audit: existing EVAL WM2 workers 706825..706828 running,
  GPU 15,403/24,564 MiB, 100% utilization; canary retained, no duplicate launched.
- Incorporate hidden-state sampling into A when ready; add fitted-physics C with
  FIT parameter JSON. No inference/training launched by resumed INT.
- Morning report must explicitly give paired-world CIs for delay CRPS differences
  at 120/300/600, breakdown-rate distance-to-real difference, and 60s HDV position
  error difference under heading `Switching-eligibility rule` (MAIN23:28).
- LANE integration: 29 CPU tests PASS (LC2013 + independent INT audit/rebuild),
  10.60s; default-OFF two-model weights/forward/rollout remain bit-exact.
  Source handoff preserved below. Conflict was root HANDOFF only.
- MERGE integration: 52 CPU tests PASS (13.28s), independent learned-head guard,
  mainline keep-right, ramp simultaneous occupancy and exact chunk memory included.
  Default-OFF parity remains bit-exact. `sumo_merge` now explicitly invokes
  LC2013 on mainline after CAV motion, keeps ramp proposals out of mainline
  arbitration, then resolves ramp merges against accepted mainline reservations.
  Forward diagnostics and persistent LC messages are retained. Both workers'
  highway-subset limitations still apply; CF implementation remains pending.
## Integrated LANE handoff snapshot (1b88408e)

Branch `c4/rb-lane`, base `f0db3c82`. Owner authorization: REGISTRATION.md:5682,
WM-PHYS REBUILD PROGRAM (2026-10-31 23:12). No training.

## Exists
- Source-backed LC2013 highway subset (SUMO **v1_27_1**) implemented;
  default-off `ModelSpec(lane_model="lc2013")` integrated. WM lane IDs are left=0, right=1,
  acceleration lane=2 (opposite to SUMO's native indexing).
- Public source downloaded for **reading only** into newly created
  `E:/Temp/opencode/sumo-1.27.1-rb-lane`; no downloaded code executed.
- Exact IDM secure gap is NOT the WM's existing braking-distance gap:
  `max(0, v*tau + v*(v-v_leader)/(2*sqrt(a*b)))` (IDM.cpp:218-220).
- cpuQ observable views reachable; only FIT worlds 969400–969431 permitted.
  Read only `episode.json`, `time_s.npy`, `veh_kin.npy`, `veh_mask.npy`,
  `is_cav.npy`, `vid.npy`. Local CPU busy (91%); use cpuQ, one thread per worker.

## Integration contract
- Pure tensor `lane_change` in `src/wms/physics/lc2013.py`, no learned weights.
- Persistent state in `lc2013_*` tensor keys, reset on absence/identity changes
  and observed lane changes. `rollout` carries across steps/chunks and resets
  newborns. Direct step callers carry `physics.lc2013.STATE_KEYS` themselves.
  Numeric current identity uses `vehicle_id` or `vid` when available.
- Existing MOBIL/learned lanes remain default. Integrator must disable learned
  merge decision families when selecting LC2013, or they could overwrite it.
- Only directly comparable encoder IDM settings map; MOBIL politeness/threshold
  do **not** identify LC2013 cooperative/speed-gain settings. LC settings use
  SUMO population defaults unless explicit comparable inferred keys are supplied.
- New `model/lc2013_adapter.py`: forward diagnostics use current snapshot; step
  recomputes after longitudinal motion/CAV execution, preserves fixed CAV lanes,
  and geometrically arbitrates simultaneous moves front-to-back. Ramp end held
  if no merge. Previous LC follower-help messages affect next HDV acceleration.
- Build with `ModelSpec(lane_model="lc2013", lane_head="off")` and the desired
  physics/encoder mode. No new weights. `merge_decision` must remain `hazard`
  (no learned merge module); other learned acceleration corrections must be
  disabled by WM-PHYS integration. Do not mutate `.lane_model` on a live
  hazard_v5/first-passage model with its old modules still active.

## Tests/results/gaps
56 CPU tests passed locally (one CPU thread, 22.64s), including 15 new tests
plus existing lane_rule/wmlane/native_lane suites. Tape validation pending.
Numeric secure-gap, discrete braking inverse,
strategic ramp, overlap/two-sided safety, assertive, persistent speed-gain,
5s keep-right, cooperative clearing, identity/absence reset, invalid settings,
default-off/spec checks, lane-logit ownership, fixed CAV lanes and exact chunk parity.
Source scope: highway, sublane OFF. **Known parity gaps:** synchronous messages
instead of SUMO downstream-first serial changes; full leader-overtake/fall-behind
negotiation and right-overtaking prevention not ported; no roundabouts,
pedestrians, opposite-direction driving or TraCI wishes. Optional route
distances/offset/queued occupation accepted; defaults use scenario ramp end and
continuing mainline. No SUMO or training launched, no tape result yet.
## Integrated MERGE handoff snapshot (f2355d13)

LATEST INTEGRATION INCREMENT: **2977be6d** (source corrections; 28 CPU tests pass).
INT already reconciled mainline ownership: cherry-pick source corrections from
2977be6d while keeping your root HANDOFF and your integrated sumo_merge hooks.
This increment changes junction_merge.py / sumo_insertion.py, not sumo_merge.py.
Current sole owned job: cpuQ **233273**, post-helper FIT32 diagnostic, below.

Base: `f0db3c82`, branch `c4/rb-merge`. Owner registration read (line 5682).

## Interface reserved for S-RB-INT / S-RB-LANE
- Switch: `PhysicsConfig.merge_model = "legacy" | "sumo"` (default legacy).
- New mechanics: `src/wms/physics/junction_merge.py`; new insertion adapter:
  `src/wms/model/sumo_insertion.py`. No new model tensors or training.
- Merge owns lane 2 -> 1 inside model-frame `[0, 75.44]`, lane-end stopping,
  target-gap acceptance and speed adaptation. Lane stream owns mainline changes.
- Planned lane hook: callable `lc2013_lane_logits(kin, present, physics, driver_params)`;
  merge implementation can consume mainline proposals and resolves merge conflicts.
- `merge_model="sumo"` bypasses learned merge/courtesy/ramp adaptation and learned
  admission; exact `conserve_queue` remains mandatory. Driver settings come only
  from the existing motion encoder or population defaults.
- SUMO source downloaded as text into newly created empty directory
  `E:/Temp/opencode/rb-merge-sumo-1271-20261031`; no downloaded code executed.

## Implemented increment (ready for integration review)
- Commits: `a33e0b55` mechanics + adapter, `f2355d13` validator,
  `66495a79` validator persistent-ID alignment. Later increment follows below.
- `junction_merge.py`: exact IDM secure-gap formula, lane-end stopSpeed kernel,
  strategic highway merge, leader adaptation + paired follower help, sequential
  accepted-merge occupancy reservations. Geometry proves no minor-priority ramp
  junction conflict in this network; no blanket mainline yield.
- `sumo_merge.py`: early forward/step adapter. Selecting
  `PhysicsConfig(merge_model="sumo")` bypasses ALL learned motion/lane/merge,
  temporal, courtesy, ramp-adaptation and acceleration-noise emissions. Existing
  checkpoint modules/tensors remain loadable. Optional `baseline_lane_logits`
  accepts absolute mainline proposals; without it uses physics MOBIL.
- `sumo_insertion.py`: source IDM insertionFollowSpeed, departLane free/best_free/
  fixed and departSpeed max/last/given; current snapshot, no artificial extrapolation.
  Uses population settings or exact existing LearnedMix identity-keyed encoder draws.
- Rollout automatically installs SUMO insertion on an existing entry process:
  MUST use `TorchEntryProcessV2(conserve_queue=True, couple_draws=True, row_seeds=...)`.
  Learned admission and learned-entry hazard are turned OFF. Arrival clock, FIFO,
  birth ordinals and SourceLedger remain the existing plumbing.
- Source/parity ledger: `docs/rb_merge/SOURCES.md`. This is a highway subset;
  persistent LC lookahead memory, full overtaking/counter-change negotiation and
  SUMO ballistic advice are outstanding, not claimed exact full SUMO parity.
- CPU command: `py -3.13 -m pytest tests/unit/test_rb_merge.py tests/unit/test_wmfix_admission.py -q`
  with `PYTHONPATH=src;.` and OMP/MKL threads=1. First pass: 20 tests passed.

## Integration reconciliation / validation progress (~23:40)
- Integration reconciliation commit **d6954af3** is REQUIRED after a33e0b55:
  mainline LC2013 hooks + no duplicate lane-end obstacle. CPU suite now **24 passed**.
- INT: keep your own root HANDOFF when merging, this task's handoff is a snapshot.
- S-RB-LANE API now read. SUMO adapter detects `model.lane_model == 'lc2013'`:
  `lc2013_adapter.forward_output/evaluate` owns MAINLINE proposals and persistent
  `lc2013_*` outputs; the post-motion ramp resolver owns 2->1. INT's merged LANE
  rollout carries these outputs. This avoids bypassing mainline LC2013 via our
  early return. No peer files copied or modified.
- Fixed a source-identified double-stop bug: SUMO merge uses IDM::stopSpeed for
  the dead lane; `lead_neighbour` must NOT also invent a leader at the lane end
  (that erroneously adds minGap and pins ramp fronts ~2.6 m short of the end).
- cpuQ own synchronous validation directory `/tmp/rb-merge-f2355d13`.
  No background jobs/PIDs. Local script at 66495a79; initial 60s native 969400
  diagnostic finished in 1.29s. Not an efficacy screen: closed t0 cohort only.
- IMPORTANT: observable `vid.npy` is [time,slot], slots recycle. Validator now
  aligns persistent observable IDs and has a regression test for reordering.
- Shared targets DONE consumed: `rb-shared/targets/{README.md,DONE.md}`; avoid
  re-extracting target tables. Their 300s ramp-delay fleet clocks differ from
  our explicitly named ramp-lane cohort speed-loss diagnostic.
- 23:42 first two native FIT 300s checks complete (969400/969401): 17 recorded
  cohort merges, 17 predicted, 17 same-ID matches. Observed birth admission
  recall 299/371. This is not full-population insertion calibration.
- 23:43:41 +08 owned cpuQ PID **200540** runs ONLY remaining native FIT seeds
  **969402–969431**, 300s, one CPU thread, nice=10. No GPU/SUMO/training.
  Exclusive launch receipt `/tmp/rb-merge-f2355d13/remaining_fit_dispatch.json`;
  log `remaining_fit.log`, result `fit_969402_969431_300.json` in that directory.
  Do NOT duplicate this job. Dispatch code commit `5a7db4ee`.

## Shared reference intake / completed first diagnostic
- PID 200540 exited successfully; all 32 native FIT 300s receipts copied into
  `docs/rb_merge/`. Version **d6954af3**, BEFORE the shared-reference corrections
  below. Summary `fit_native32_pre_helper_summary.json`: cohort real/predicted
  merges 200/125, matched 124; matched time MAE 24.52s, position MAE 17.21m.
  Cohort ramp-lane speed loss mean 240.04 real / 761.28 predicted veh-s.
  This candidate is NOT qualified; first-two-world counts did not generalize.
- The mergespec helper became readable during validation. Consumed spec, test
  matrix and exact source excerpts; no helper job duplicated. Added source-based
  urgency /2 with step floor, corrected follower net-gap units, follower pass/
  eventual-help/fall-behind branches, exact departSpeed last/desired refusal,
  single chosen departLane and base POSITION_EPS=.01 (legacy .1 was wrong).
- **28 CPU tests pass**, including helper T3 numeric insertion vector. These
  changes follow public source, not tuning to the FIT errors. A new versioned
  diagnostic is warranted after the mechanics changed; previous results retained.
- `tools/rbeval/summarize_merge.py` refuses duplicate/prohibited seeds and keeps
  conditional admission recall distinct from autonomous insertion calibration.
- Corrected source version **2977be6d** frozen; own cpuQ PID **233273** launched
  23:54:00 +08, one thread, nice=10, 32 FIT native worlds, 300s cohort diagnostics.
  Exclusive receipt `/tmp/rb-merge-f2355d13/post_helper_fit_dispatch.json` includes
  exact SHA256s of validator and all four mechanics modules. Output
  `post_helper_fit_300.json`; previous PID200540 fully exited before any source
  replacement or this new-version job. CPU load 12.9/32 at launch. No GPU use.

## Status / known gaps
- Fresh-session recovery: HEAD f0db3c82; only this untracked handoff survived.
  No implementation, evaluation receipt, PID or dispatch exists in this worktree.
- Authorization verified at REGISTRATION.md:5683; all four WHY documents read.
- Implementation increment complete; FIT validation next. No scientific efficacy
  or 562 veh-s ramp-delay improvement claimed yet.
- S-RB-LANE and S-RB-INT HANDOFF.md did not exist on initial read (~23:15).
- Runtime recovered: `py -3.13`, torch 2.15.0+cpu. Use one torch thread.
- NVIDIA driver unavailable; CPU load 0% at resume resource check.
- Shared mergespec folder not present on resume; no helper work duplicated.
- Validation will restrict recorded FIT seeds to 969400–969431 and observable fields.
