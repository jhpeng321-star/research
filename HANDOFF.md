# S-RB-MERGE — integration handoff / owned validation complete

Branch `c4/rb-merge`, basef0db3c82. Prior work fully retained; see
`docs/rb_merge/HANDOFF_HISTORY_{0100,0315,FINAL_VALIDATION}.md` and source ledger.

## Status
- November 3 subsequent recovery: clean HEAD9a108e72 retained; four final local
  PIDs absent at15:16:15+08. Current registration/INT/LANE/shared receipts still
  confirm closure and no new owned repair. Same-day recovery receipt updated.
- November 3 recovery: clean HEADaa3d5d03 retained; final four local job PIDs
  absent at04:33:45+08. Current INT/LANE/shared receipts confirm original closure,
  frozen MERGE intake and no new owned repair. `docs/rb_merge/RESUME_20261103.md`.
- November 2 recovery: clean HEAD7628781d retained; final four local job PIDs
  absent at21:25:39+08. INT confirms original rebuild closed; round two separately
  owned. LANE requests no MERGE ABI repair. `docs/rb_merge/RESUME_20261102.md`.
- Fresh-session recovery after noon on2026-11-01: clean HEAD3ce7aa6a retained;
  four final local PIDs absent at12:57:50+08. INT confirms MERGE intake and frozen
  FULL-v4 post-motion departure wiring; no new owned integration repair requested.
  Recovery receipt `docs/rb_merge/RESUME_20261101_AFTER_NOON.md`. Prior tests/results
  retained; no dispatch/re-score after the elapsed09:30freeze.
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
