# S-RB-MERGE — in progress (2026-10-31)

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
