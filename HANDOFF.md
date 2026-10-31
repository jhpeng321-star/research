# S-RB-MERGE — live integration handoff

Branch `c4/rb-merge`, basef0db3c82. Owner registration5683/MAIN00:10.
Earlier work retained in `docs/rb_merge/HANDOFF_HISTORY_{0100,0315}.md`.

## Status / owned dispatches
- **v7fast PID682380 EXITED,32/32 H600 worlds complete**, source158d0193.
- **v8end120 PID750052 EXITED,32/32 H120 worlds complete**, sourcec8e54017.
- All earlier v3/v4/v5/v6 jobs terminal.
- **v8end600 LIVE,03:26:36 +08**, owned cpuQ PID**789747**, source93693678;
  validator SHA `b7d4721f500e919e025c57abc3bc357c2d87723d4a93cdad5c012783d9d1db26`.
  Receipt `/tmp/rb-merge-fullfleet-v8end600/LAUNCH.json`; do not duplicate or
  modify this snapshot. OneCPU/nice19/CUDA-hidden, load2.11/32 at launch.
  Same source/operator as
  v8end120; **58 CPU tests PASS /2.46s**, all32 source/strict-encoder preflights PASS.
  H600/512slots/FIT969400–969431/native/stream190; explicit `--entry-phase after`.
  Saved clocks/events/front traces will be checked against the completed H120
  prefix, not counted as another independent world/future. Three prefix-audit
  CPU tests PASS (changed contracts/checkpoints/birth/trace rejection).
  Do not relaunch any completed version. Old v7 directory is engineering only.
- Latest integrated screen `/tmp/rb-merge-fullfleet-v9latest300` **LIVE**:
  **03:49:23 +08**, owned cpuQ PID**840607**, source6bb7ce99,
  validator SHA `ae4d2412050de70baa0de57de38f2de6ac853e506e413ed291d9298a8baa1b14`.
  Receipt `/tmp/rb-merge-fullfleet-v9latest300/LAUNCH.json`; oneCPU/nice19,
  load18.03/32; distinct new-source H300 screen, do not duplicate.
  INT**8e964de5** (new CF source lane-cap/link chain + LANE raw
  origins/position-translation API), current owned kernels/phase/native callback,
  same frozen encoder and FIT32/H300/512slots/stream190, CPU exact-fast flags.
  **58 CPU tests PASS /3.13s**, all32 strict input/encoder preflights PASS.
  Producer INT recorded lane-arbitration parity receipt also PASS; runtime
  construction names exact source commits in STAGE.json. v8end600 stays frozen.

## Source increments for INT (keep your own root handoff / adapter)
- **c8e54017** NEW `model/sumo_departure.py::step_after_movement`: source
  MSNet.cpp order is move → LC → removePending → emitVehicles. Explicit opt-in
  one-step API: newborns at basePos/departSpeed, no same-tick motion/LC/control.
  Preserves source generator/FIFO/ordinals/SourceLedger/padding pool. Returns
  `source_before`, `source_birth_mask`, `source_exit_mask`,
  `driver_params_after_insertion`. Use those for boundary costs/next-step params;
  a post-insertion entry-observer snapshot is NOT the old road cohort.
  Same-tick exit/birth cannot overwrite one logged slot; newborn history resets.
  **7 dedicated phase tests PASS**,58 combined runtime tests PASS. No shared
  wms/spec/factory edits. Existing rollout default remains before-phase.
  API follow-up: explicit/inferred driver settings require a coupled entrant
  sampler, preventing newborns from inheriting unrelated departed-slot settings.
  **9 dedicated phase tests PASS /4.34s**; all active screens already provide
  both, so their frozen input/dynamics contracts are unchanged.
- **158d0193** candidate-local sequential geometry: O(ramp_count*slots), bit-exact
  to full-frame geometry with heterogeneous/batched inputs; reservations preserved.
- **9bdc35e5** exact IDM minNextSpeed low-b bound + causal encoder/600s harness.
- **d7be510a /0729e09c** native physics callback and `native_physics_accel` for
  native-relative controls. Frozen native head incorrectly uses slot0 dv without
  a same-lane leader; unrelated mainline traffic can trap an unled ramp CAV.
  CPU reproduction + FIT969400/CAV187: old stays at s=-1.464/v0 through600;
  physics-native merges at326.5/s52.884. No head retrained/modified.
- **3df73a27 /5870b556** local22.22 ramp-approach →31.29 acceleration-lane
  speed limits, inferred v0/referenceLimit factor, exact weighted BirthMix draw,
  source EPS=.001/POSITION_EPS=.1 (not SPEED_EPS=.01).
- **0090c168** leader overtake time+space / Euler brakeGap(tau) / unblocked
  leader advice. Keep INT's post-motion mainline arbitration; only two hooks:
  merge_speed_advice `lookahead_speed=state.get('lc2013_lookahead')`,
  apply_merge `lookahead_speed=out.get('lc2013_lookahead')`.
- `PhysicsConfig.merge_model='sumo'` defaultlegacy; bypass learned merge/courtesy/
  ramp adaptation/admission/motion/noise. Motion-inferred/default settings only.

## Completed FIT32 diagnostics (one future/native only — NOT qualified)
- v6 population/SUMO-CF/physics-native, H300: all200 recorded t0 merges matched,
  202predicted; full-fleet merge mean33.16 vs33.06real. Ramp delay1185.32 vs
  2272.78veh-s real: removed false gridlock but over-clears; W1=1087.46.
- **v7fast frozen motion encoder + latest LANE811cd7e8**, original before-phase:
  H120 mean955.84/928.87 real, W1=137.75;
  H3002451.98/2272.78, W1=497.39;
  H6005026.36/4670.05, W1=1764.62. H600 predicted q10/50/90=2206/3619/7668,
  real3058/4513/6103. All200 t0 merges matched; timingMAE41.98s, position13.48m.
  Broader tails still wrong; multiple source/settings details changed vs v6.
- **v8end120 post-motion phase**, same frozen encoder/native/LC settings:
  mean955.17 vs928.87 real, W1=155.26 (no marginal improvement over v7 H120).
  H120 t0 matches160/193, timeMAE18.55s, position12.34m; censored support differs
  from H600. Prefer source-correct ordering for fidelity, not outcome tuning.
- Both new panels:32/32 exact road/source conservation, no discarded demand or
  slot pressure; no beyond-end merges. Raw corridor-overlap samples are NOT SUMO
  collision counts. Recorded t0 geometry audit:4 raw-frame overlaps in402/405/
  419/425, **zero after connector-length chain mapping**. Projection artifacts.
  That audit proves only initial geometry, not every later overlap sample.
- Historical562.4/1265.4 is a different panel/clock; matched recorded world values
  above are primary. No breakdown/controller-shift/conditional-uncertainty PASS.
- Hash-verified raw `docs/rb_merge/fullfleet_v{7fast,8end120}/`, compact
  `fullfleet_v{7fast,8end120}_summary.json`. Initial audit in v7fast collection.
  JSON receipts byte-preserved by local .gitattributes; old records retained.

## Settings / interfaces / checks
- Reused LANE **b192e0c8** strict extractor, registered WM2 SHA b8797242…;
  producer parity all9 outputs bit-exact. Preserve trained legacy encoder
  feature/default geometry; SUMO geometry only for rollout. Initial HDV settings
  follow public VID /240..300 motion; CAV defaults retained; newcomer HDVs use
  zero-history encoder centre. No new extraction implementation/fit/training.
- `EncodedPointBirths` checks admission/newborn ordinal identity for EVERY birth.
  Entry process requires conserve_queue=True/couple_draws=True/row_seeds.
- Boundary delay:31.29 reference, no newborn road charge, prior speed on exit,
  ramp origin retained after merge, fractional source waiting separate.
- Tools: `remote_full_fleet.py status|poll|trace|initial-geometry|collect --version …`.
  Current runner defaultv8end600/after; direct validator default remains before.
  Single-thread/nice19/CUDAhidden; check load before launch. No GPU/SUMO/training,
  peer signals, truth/assignment paths, newenv, reset/clean/deletion/push.
- Source ledger `docs/rb_merge/SOURCES.md`; public CPP read-only in NEW empty
  temp directory, never executed. Coordination pointers in rb-shared/MERGE_*.md.
- Scope gaps: synchronous LC, complete blocker negotiation, explicit internal
  junction state/actionStep, calibrated hidden-driver/arrival futures; final
  distributional gate belongs to INT/FIT/UNC/EVAL, not this point-settings screen.
