# S-WMFIX-ES fresh-session audit

- Resumed from clean `89d7e29b`; prior commits/artifacts retained. No push.
- Remote `ps` audit: no `tools.wmfixmerge.es` workers on gpuA or cpuQ.
- gpuA `es_run_combined` is empty; cpuQ combined v1/v2 registries exist.
- cpuQ `es_smoke` failed in inferred_timeline: CTRL0 direct/mixed actions
  require an explicit feedback controller. No combined ES generation completed.
- Prior registry used only every fifth wmdata world and CTRL0; MAIN 13:38
  instead requires all 40 worlds/all arms. Correct before dispatch.
- Prior CPU probe is single-future: Runtime uses FuturesConfig(n_samples=1),
  not K8. The measured 1766.6/1920.0 s and 1372/1775 MB remain valid only
  for that single-future two-arm 120+300 s workload. K8 claim withdrawn.
- Runtime source guard remains active. No SUMO, GPU launch or foreign PID
  signalling. cpuQ cap now 30 CPU workers; gpuA cap 20 CPU threads.
- Next: explicit frozen wmdata feedback adapters; all-world/all-arm registry;
  genuine K8 probe; budget the full-corpus calibration honestly.

## Corrected tooling / live probes
- Explicit CTRL0 uses the original 64f70af4 policy with its residual head zeroed;
  MAPPO binds d9042cc5. PI/CIRCLES use published TimelineControllers, including
  explicit algorithm continuity for entrants. Opt-in designed-arm loading is
  restricted in Runtime to verified wmdata_fit episodes; default loader unchanged.
- Registry now covers 52 worlds / 184 contrasts (both FIT arms too); fitness averages arms within
  world before averaging worlds. Original FIT objective scales remain fixed.
- cpuQ tests: 18 passed (ES + wmdata role); Windows has no torch (17 passed,
  the live-dataclass test requires torch; no Windows environment installed).
- K8 probe launched as eight sticky-future single-thread processes per host:
  cpuQ PIDs 3030648-3030655; gpuA PIDs 193235-193242. Each measures both arms
  at 120/300 s, entry path/mix ON. Dedicated `probe_k8/DISPATCH.json` + PID
  receipts prevent relaunch; gpuA code isolated under `es_cpu_v3`.
- CTRL0/MAPPO 60 s smoke complete, 31.2/31.3 s; frozen tensor checks pass.
- PI/CIRCLES first attempt lacked frozen controller sources; retained failures.
  Sources staged byte-for-byte from WMDATA frozen e3final runtime; new owned
  PIDs 3035044/3035045, `es_canary_deps_v3`; no other processes signalled.

## Full-corpus calibration dispatch step
- Missing adapter/model assets caused the second PI/CIRCLES attempt to fail;
  only adapter.json and model.onnx were copied from frozen e3final. Third
  attempt `es_canary_assets_v3` PASS: CIRCLES 31.9 s, PI 31.2 s, frozen checks.
- Both original FIT arms are bound, not just arm 0: 24 + 160 = 184 contrasts.
  Verified registry `TRAIN_PREPARED_COMBINED_FULL.json` copied cpuQ -> gpuA.
- Canonical physical export/reload smoke PASS on gpuA, under private code tree.
- Expanded ES/objective/role tests: 20 passed on cpuQ, including exact parity
  against the registered torch objective and equal-world aggregation.
- Protocol amendment before any combined fitness: full 52-world/184-contrast
  corpus at existing 120 s training horizon, population 2, sigma .10, lr .01,
  seed 140014, <=8 generations / 12600 s, stale patience 3. Six physics scalars;
  entry scalar frozen because canonical verdict entry config cannot carry it.
- Immutable generation exports bind physical config; all old WM2 tensors are
  bitwise checked on export and canonical reload. Failure receipts stop driver.
- Driver reserves live K8 probe slots before each generation (cpuQ total <=30,
  gpuA <=18 CPU workers), low-priority sweeps; no GPU code path.
- Full driver launched gpuA PID 195604, `es_cpu_v3/es_run_full`; cpuQ workers
  `es_run_full`. Read RUN/ENV/CELL/GEN/ES_FINAL/ES_ABORT before any resume.
- Unchanged-plan preflight on canonical smoke export found missing accept panel,
  C1 cohort/runner and candidate quartet completion. Required input/GPU-slot
  requests are recorded in WMFIX_ES.md; no evaluation result is claimed.

## Scoring identity / bounded receipt continuation
- Canonical accept normally evaluates both checkpoints under one shared physical
  config. That would erase option-(b)'s candidate/reference physical difference.
  `r10_attribution.load_model` now accepts an explicit JSON wrapper metadata flag
  `checkpoint_physics`, default OFF; it uses each checkpoint's serialized physics.
  No verdict metric/threshold changed; all existing TOML/default paths unchanged.
- Real serialization-only smoke PASS: two otherwise identical frozen checkpoints
  with distinct physical scalars keep that distinction under the same scoring
  config. No rollout/fitness or manual calibration value used by this fixture.
- Initial schema-field approach collided with already-staged merge-basis config
  additions; restored those original host fields and used wrapper metadata instead.
  Shared S-WMFIXMERGE files/processes never edited/signalled. No ES sweep failure
  receipt exists; running sweeps retain their already-loaded code and frozen tensors.
- Live driver retains its pre-fix config export function. `es_finish.py` waits for
  FINAL, preserves legacy config JSON, emits schema-correct immutable scoring config,
  reload-verifies selected physical values and writes SCORING_READY + final preflight.
  Immutable .pt checkpoints never modified. Finalizer is bounded to six hours.
- `es_collect.py` only pulls completed JSON receipts, computes actual K8 CPU work
  and N-process panel extrapolation, records every generation's J/source MAE and
  updates <=60-line WMFIX_ES.md. Six-hour cap, exclusive PID file, no dispatches.
  Optional commits inspect status/diff/log and stage only report/receipt files;
  foreign edits defer commits or stop report overwrites. No push.

## Fresh-session live audit — 2026-10-31 16:11 +08
- Clean resume at 30d3bb72; all previous commits/artifacts retained. Read required
  design/review/retarget/feasibility/pilot/runtime/panel code and task receipts.
- gpuA original driver 195604 and finalizer 199849 are live. Local collector
  25836 is live (`--watch --commit`); no replacement watcher/driver dispatched.
- Generation 0 completed all 184 contrasts in 1980.4 s: objective .36886510,
  equal-world TRAIN paired J MAE 237.69286, source contrast MAE 26.62019.
- Generation 1 sweeps are active on cpuQ; no failed-sweep receipt observed.
- WMDATA COMPLETE receipt confirms 40 wmdata_fit worlds/all 5 arms and corpus
  acceptance. Existing combined registry already includes them; no preparation
  or completed K8 probes repeated. No foreign PID signalling/GPU/SUMO launch.
- `es_status.py` adds compact, read-only run summaries from owned receipts and
  receipt-identified ES PIDs; no checkpoint/tape reads, launch or automatic retry.

## ES-specific verdict identity isolation
- Found B reducer did not compare its quartet candidate/config/reference hashes
  to BINDING.json. Added fail-closed checks; 4 regression tests PASS (Windows).
- The original plan's candidate directory is shared with S-WMFIXMERGE. Frozen
  `es_run_full/ES_VERDICT_PLAN.json` now uses private B sources/output only;
  cohorts, base quartet, C1/accept bindings and all verdict thresholds unchanged.
- `quartet_candidate.py --out` supports that private output without editing any
  shared GPU source/process. Private host CLI smoke PASS; no rollout dispatched.
- MAIN GPU-slot handoff: after SCORING_READY, run the quartet with the selected
  FINAL checkpoint/config, `--out es_run_full/quartet_candidate`, then verdict
  with `--plan es_run_full/ES_VERDICT_PLAN.json`. Pre-create output directories;
  never reuse the other candidate's completion. C1/accept inputs still required.
- Ideal row-time extrapolation: A's 36 groups x 3 legs x 4 arms x 300 s is ~8.6 h
  on the combined CPUs; B candidate's 8 x 4 x 600 s is ~1.3 h (K8 throughout).
  This is an estimate, not a benchmark; no timely full A/B CPU completion is
  established. Existing WMFIX_ES.md request for MAIN's GPU slot remains active.
- Actual verdict.py preflight previously checked producer code but not missing
  accept/C1/quartet input files; A could run expensively before failing later.
  Added the existing registered input-presence checks to that preflight too;
  no metric change. Combined verdict/binding regressions: 14 passed (26.0 s).
  Staged only private ES files. ES-plan preflight reports precisely those four
   missing inputs; no producer-closure mismatch and no evaluation launch.

## Fresh-session audit — 2026-10-31 16:52 +08
- Resumed at 069c68ae with five modified files plus GEN_001.json; all retained.
  Read required design/review/feasibility/retarget/pilot and rollout sources first.
- Existing gpuA driver 195604 remains live; generation 1 completed all 184 cells
  per candidate, generation 2 already active on both hosts. No failed sweep.
  No driver, worker or receipt collector redispatched.
- Existing local collector 18420 is live (exclusive resume_audit PID receipt);
  its automatic commits deferred because of the prior session's code edits.
- Pending private-plan selection and collector/report changes verified with
  test_wmfixmerge_es_collect, test_wmfixmerge_es_verdict_binding and
  test_wmfix_verdict: 18 passed in 30.99 s. No Windows environment installed.
- Generation 1 selected g1/c1 by registered road/crossing fitness .33808628
  (baseline .36886510); paired J MAE worsens 237.69 -> 243.87 veh-s. Selection
  remains training-objective-only; neither number constitutes a verdict PASS.
- Private-host preflight had not received the recovered plan-selection fix.
  Staged that single verified file into es_cpu_v3 (SHA256 642e3cdd913e2cab...);
  finalizer invokes it in a new process only after FINAL. Its running process
  and all calibration/other-owner code remain untouched. No dispatch repeated.
- Immutable 17:02 status receipts retained for both hosts: gpuA driver/finalizer
  and local collector live; cpuQ has 30 receipt-identified active ES workers.
  All four registered missing input paths rechecked absent. Final results are
  still pending; existing collector will pull/commit each completed generation.

## Final resume / MAIN 17:53 shutdown — 2026-10-31
- Resumed clean at 29faed51; retained all previous work. Read required design,
  review, feasibility, retarget, pilot and preparation/runtime/panel sources.
- Generation 2 was already complete on arrival (2810.9 s, 184 contrasts per
  candidate). Generation 3 had already started in the frozen session; its partial
  cells are preserved but excluded from selection. No generation was dispatched
  by this resume. The existing driver has no STOP polling, so signals were needed.
- `es_stop.py` wrote owned STOP markers, matched RUN/ENV/finalizer PID receipts
  against exact module/output-root command lines, froze the dispatcher first,
  then stopped only those verified owned PIDs. Immutable STOP_MAIN_1753 receipts
  were retained on both hosts and copied to wmfix_es_receipts.
- gpuA driver 195604 and finalizer 199849 stopped; 30 remaining cpuQ generation-3
  workers stopped. Exact-root /proc scans found zero active owned processes on
  either host. Local collectors 18420/25836 were already absent (CIM checked).
- CPU allocations released; no foreign process signalled, no A/B verdict launch.
  GEN_002.json retained locally; selected candidate remains generation 1/c1.

## Completed-task verification — 2026-11-03 +08
- Resumed clean at 5efa2228; final report, shutdown receipts and registration were
  already committed. Required design/runtime documents read; no work repeated.
- Read-only host process audits find zero matching ES processes; local Python
  collector audit also empty. No dispatch or signal issued by this resume.
- gpuA generation 0–2 summaries and partial g3 counts match the shutdown receipt;
  selected generation_001.pt SHA256 reverified exactly. cpuQ's original run
  directory is now absent; committed receipts retained, no reconstruction launched.
- MAIN stop remains final: no A/B verdict; retain WM2. Fresh audit receipt:
  `wmfix_es_receipts/RESUME_STOP_VERIFIED_20261103.json`. Report remains <=60 lines.
