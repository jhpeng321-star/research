# S-RB-MERGE — full-fleet validation continuation

## Resume / MAIN 00:10 continuation
- Preserved clean HEAD `1d68fd5a` and all prior receipts. Prior PID200540 and
  PID233273 are terminal in the completed-result receipts; no redispatch planned.
- Re-read owner registration and four WHY reports. Latest LANE `127a106f`
  supplies mainline LC2013; INT owns the reconciled post-motion arbitration and
  entrant identity correction. Reuse a hash-bound integrated source snapshot
  for the next diagnostic rather than change shared worker code independently.
- Next increment: full-population native FIT diagnostic with autonomous entries,
  exact due/admitted/pending conservation, ramp-origin (including post-merge)
  speed-loss delay and source waiting separately, plus recorded per-world clocks.
  Historical 562.4 is a reference, not the target of parameter tuning. No training.
- Earlier cohort diagnostics below remain valid for their stated contracts.

## Full-fleet validator increment
- `tools/rbeval/full_fleet.py`: autonomous native CAV executor + population HDVs,
  mainline LC2013 and SUMO merge/insertion using pinned INT `ff8a637f` runtime.
  Each 0.5s chunk carries returned state/history, LC memory and synthetic entrant
  identity. Recycled slots retain the NEW vehicle's origin, not the previous one.
- Exact `SourceLedger` per entrance and road-population conservation checked each
  step. Reported-boundary delay uses 31.29m/s, excludes births and uses prior speed
  on exit; ramp-origin charge continues after merging. Source fractional waiting
  is integrated separately with no extra RNG draws. Future arrivals inferred from
  t0 request counters/backlog only; future recorded states never feed predictions.
- Source snapshot currently has legacy CF (SUMO-IDM kernel is present but CF
  switch wiring pending); selector is reported explicitly. LC memory is cold at
  t0. One future is a diagnostic, not the final INT A/B distributional gate.
- **5 new CPU tests pass** with pinned integrated imports (14.90s), including
  native CAV 4-step-vs-chunk exact parity, fractional waiting, slot reuse/origin,
  birth/exit clock and queue-corruption rejection. Existing merge suite **28 pass**.
- CPU command: `py -3.13 -m pytest -o "pythonpath=E:/agent-work/c4-rb-int/src ." tests/unit/test_rb_merge_full_fleet.py -q`.
- `remote_full_fleet.py` stages a new immutable `/tmp/rb-merge-fullfleet-v1`, runs
  CPU tests remotely before admission, and uses an exclusive dispatch receipt.
  FIT32/native/H300, single CPU nice19, no GPU/SUMO/training. No launch yet.
- Remote staging CPU gate caught missing frozen native-CAV JSON in the source
  archive (local tests had that config). No recorded rollout/job was launched.
  Preserve `/tmp/rb-merge-fullfleet-v1/CPU_TESTS.json`; corrected staging uses a
  NEW `/tmp/rb-merge-fullfleet-v2` and pins both native-head/executor config hashes.
- Corrected remote CPU gate: **5/5 PASS (1.35s)**, cpuQ load13.37/32. No launch
  receipt yet; exclusive dispatch is the next step. No scientific rollout repeated.
- v2 owned PID330030 exited before any rollout: converted FIT views have no
  `entry_backlog.npy`. Failure/launch/source receipts preserved in
  `docs/rb_merge/fullfleet_v2/`; zero completed scientific worlds.
- v3 uses the EXISTING identity-hashed `jobs/s{seed}_native/start.pt` observable
  detector contract (same as WMCALIB): pre-t0 insertion counts plus t0 entrance
  queue infer demand over 300s; pre-t0 observed CAV shares. No request XML/type
  assignments are opened. **All32 source preflights PASS**, **6 CPU tests PASS**
  on cpuQ in1.29s. New `/tmp/rb-merge-fullfleet-v3`; old trees left intact.

LATEST INTEGRATION INCREMENT: **2977be6d** (source corrections; 28 CPU tests pass).
INT already reconciled mainline ownership: cherry-pick source corrections from
2977be6d while keeping your root HANDOFF and your integrated sumo_merge hooks.
This increment changes junction_merge.py / sumo_insertion.py, not sumo_merge.py.
Both owned validation jobs exited; no running dispatch remains.

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
- Implementation + FIT32 diagnostics complete. No full-fleet efficacy or
  historical 562 veh-s ramp-delay qualification claimed.
- S-RB-LANE and S-RB-INT HANDOFF.md did not exist on initial read (~23:15).
- Runtime recovered: `py -3.13`, torch 2.15.0+cpu. Use one torch thread.
- NVIDIA driver unavailable; CPU load 0% at resume resource check.
- Shared mergespec folder not present on resume; no helper work duplicated.
- Validation will restrict recorded FIT seeds to 969400–969431 and observable fields.

## Final corrected-version result (2977be6d)
- Source-hash-verified receipt: `docs/rb_merge/fit_native32_post_helper_summary.json`;
  raw per-world evidence `post_helper_fit_300.json`, launch receipt
  `post_helper_fit_dispatch.json`. PID233273 completed in **127.48 s**, exited.
- All **32 FIT native worlds 969400–969431**, t0=300, horizon=300s. Population
  default settings; autonomous t0 cohort, no future state feed-back, no entrants,
  physics MOBIL mainline (LC2013 integration belongs to INT). Every t0 car native.
- Cohort recorded/predicted merges **200 / 202**, same-ID matches **200**.
  Matched timing MAE **30.605s**, signed bias **-12.275s**; position MAE
  **20.236m**, signed bias **+14.040m**. Two extra predicted merges.
- Named ramp-LANE normalized speed-loss diagnostic: world mean **240.04 real /
  164.00 predicted veh-s**, per-world/per-vehicle quantiles in receipts. This
  excludes entrants and ramp-origin traffic after merging; it is NOT the full
  ramp-ENTRANCE-group delay clock used for historical 562.4/1265.4 values.
- Recorded-state acceptance TP/FP/FN/TN **410/2496/625/123046**. Physical entrance
  acceptance at observed births **4865/6154**; recall-only, no false-admission
  denominator or autonomous count calibration. Full-population NPZ scoring is
  implemented for the integrator's final rollout, with persistent-ID alignment.
- **28 CPU tests passed** (11.18s):
  `py -3.13 -m pytest tests/unit/test_rb_merge.py tests/unit/test_rb_merge_validation.py tests/unit/test_wmfix_admission.py -q`
  using `PYTHONPATH=src;.` and OMP/MKL threads=1.
- NOT distributionally qualified. Timing/position errors remain substantial;
  assess the combined SUMO-IDM + LC2013 + inferred/default settings candidate via
  INT/EVAL's full-fleet distributional gate. No tuning/training performed here.
