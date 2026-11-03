# S-RB-CF handoff

Branch `c4/rb-cf`, base `f0db3c82` (c4/wmfixmerge). Owner authorization: the
WM-PHYS REBUILD PROGRAM line of the active C4 REGISTRATION (2026-10-31 23:12).
Task S-RB-CF: SUMO 1.27.1 IDM car-following parity + a pure-physics preset.

## Status (updated as work lands; all times 2026-10-31/11-01 +08:00)

- `18291ef5` **Euler position parity** (2026-11-01 ~10:15, post-freeze
  increment): `physics.position_rule` = `'legacy_trapezoid'` (default,
  bit-identical) | `'sumo_euler'` = SUMO's default position update
  `s' = s + v' dt` (`MSVehicle.cpp:4954-4963`, `deltaPos = SPEED2DIST(vNext)`),
  wired through `WmSModel.step` (pivot AND non-pivot longitudinal updates);
  `stop_time_integration` is a trapezoid-only correction (skipped under Euler:
  a clamped SUMO stop does not move).  Preset
  `PURE_SUMO_PHYSICS_EULER` (frozen `PURE_SUMO_PHYSICS` unchanged for
  provenance).  Fixes my earlier WRONG doc claim that the WM trapezoid
  "reproduces the SUMO ballistic position rule" (evidence:
  rb-shared LANE_TO_CF_POSITION_20261101: 2,346,379 FIT intervals, Euler
  residual 3.5e-5 m vs trapezoid 0.039 m).  Tests
  `tests/test_rb_cf_position_rule.py` (6) + full CF/equivalence suite: 92
  passed.  Tape check 60 s (`tape_check_euler60.json`, attribution arms
  `tape_check_legacy_euler60.json`/`tape_check_legacy_pure60.json`): Euler
  position improves HDV speed MAE on BOTH cores vs the frozen trapezoid arms
  (SUMO core -0.032/-0.122 m/s @30/60 s, 21/32 and 19/32 worlds; LEGACY core
  -0.054/-0.096 m/s, 20/32 and 19/32 — core-independent integrator effect);
  gap MAE neutral at matched horizon (+0.08/+0.22 m; an initial -8.4 m gap
  delta was a 120s-vs-60s horizon mismatch, discarded).  **WM2 as-is + Euler
  position ONLY** (`tape_check_wm2_euler60.json`): -0.020/-0.102/-0.106 m/s at
  10/30/60 s (25/32, 27/32, 19/32 worlds), gap -0.26 m — the integrator fix
  helps every stack (WM2 / legacy pure / SUMO pure), no retraining.  Value is
  generator parity + bias removal, not the error score.  Measured per-step
  bias of the
  legacy rule on the 40 native dev tapes (merge zone): braking steps overshoot
  downstream +8.5 cm mean (p99 19.8 cm, n=195,638), accelerating steps
  undershoot -8.5 cm (n=91,762) — both directions align with the
  "too severe / too slow to clear" symptom (see rb-shared
  CF_POSITION_EULER_1015.md).
- `e9992767` **SUMO IDM parity module** `src/wms/physics/idm_sumo.py`: a
  line-annotated tensor port of SUMO 1.27.1 `MSCFModel_IDM` (`_v` iterated safe
  speed with the `TS/0.25` sub-stepping = 2 iterations at 0.5 s; `minNextSpeed`
  with the `min(emergencyDecel, 1.5)` relaxations; `minNextSpeedEmergency`;
  `maxNextSpeed`; the planMove/finalizeSpeed composite incl. the lane-cap
  chain; `stopSpeed` incl. the exact `maximumSafeStopSpeedEuler` + relax-
  emergency fallback; `freeSpeed`; `insertionFollowSpeed` reference for
  S-RB-MERGE).  **Reconciled with the rb-shared `idmspec` reference**
  (`2b898f39`): `NUMERICAL_EPS = 0.001` (src/config.h.cmake:236) and the
  passenger default `emergencyDecel = 9` (not decel; apparentDecel is the one
  that defaults to decel).  All 19 applicable idmspec reference vectors pass
  (`tests/fixtures/rb_cf/sumo_idm_test_vectors.json`; T18 actionStepLength=1
  is the documented ASL gap).
- `742922b6` **Switch** `physics.cf_model = 'sumo_idm'` (default `'legacy'`,
  bit-identical): replaces the legacy textbook-IDM core AND its legacy
  emergency/yield add-ons in `baseline_accel` by the SUMO step.  New config
  keys: `sumo_cf_stepping_s` (0.25), `sumo_cf_emergency_decel_mps2` (None =
  SUMO passenger default 9), `sumo_cf_vtype_max_mps` (None = unbounded),
  `sumo_cf_action_step_s` (None = every step).  Per-driver settings map 1:1
  (encoder `v0/T/min_gap/accel/decel` or population defaults).
- `6b0f4969` **Pure-physics switch** `WmSModel.learned_accel_off` (runtime
  attribute, never in a checkpoint, default OFF = bit-identical): the final HDV
  acceleration IS the physics core; drops residual attention, temporal/slow
  emission, mainline residual, negotiation yield/ramp deltas, merge-core yield,
  follower courtesy, learned mainline yield, ramp speed adaptation, merge
  response, and the plant clamp.  Preset dict
  `wms.physics.idm_sumo.PURE_SUMO_PHYSICS` and `RolloutFlags.pure_physics`
  (`wms.eval.predictor`).
- `c39ce13f` **Teacher-forced tape check** `tools/rb_cf/tape_check.py` +
  results (`tape_check_results.json` 60 s; `tape_check_120s.json` incl. the
  attribution arm): 32 WMDATA dev worlds 969400-969431, native arm, t0=300 s,
  teacher-forced CAVs, entries OFF.  HDV speed/position MAE vs the recording:

  | arm | v MAE 10 s | 30 s | 60 s | gap MAE |
  |---|---|---|---|---|
  | WM2 as-is | **1.00** | 2.43 | 3.61 | 58.7 |
  | WM2 physics core, learned corrections OFF (`legacy_pure`) | 1.04 | **2.35** | **3.00** | 57.1 |
  | pure SUMO IDM, encoder settings (`sumo_enc`) | 1.09 | 2.33 | 3.04 | 57.7 |
  | pure SUMO IDM, population defaults | 1.38 | 2.96 | 3.60 | — |

  Attribution (paired vs WM2): the 30-60 s speed gain comes from turning the
  learned acceleration corrections OFF (`legacy_pure` -0.62 m/s at 60 s, better
  in 23/32; `sumo_enc` -0.57, 18/32) — the SUMO CF core and the legacy core are
  equal within noise on this metric, so the SUMO arm's value is DISTRIBUTIONAL
  parity, not this error score.  WM2's learned stack only helps <=10 s
  (+0.04..0.09).  At 120 s only ~5 t0-roster HDVs remain per world (weak cell;
  pure arms slightly worse there, position divergence through unchanged
  lane/merge code).  Encoder-inferred driver settings are worth 0.3-0.6 m/s
  over population defaults for the pure arm.

## Tests

`python -m pytest tests/test_rb_cf_sumo_idm.py tests/test_rb_cf_switch.py
tests/test_rb_cf_pure_physics.py tests/test_rb_cf_position_rule.py
tests/equivalence -q` (92 + 16 pre-existing skips pass; conda base python
3.13.13 / torch 2.14.1+cpu — note: the `rl_driving` env no longer exists on
this machine, no new env was created). Expected values are derived from an
independent scalar transcription of the SUMO source formulas, hand-derived
anchors, and the idmspec reference vectors.

## How to run the A/B (for S-RB-INT)

```python
from wms.eval.predictor import RolloutFlags, WmsPredictor
from wms.physics.idm_sumo import PURE_SUMO_PHYSICS, PURE_SUMO_PHYSICS_EULER
p = WmsPredictor.from_checkpoint(ckpt, RolloutFlags(pure_physics=True),
                                 physics=PURE_SUMO_PHYSICS)          # frozen recipe
q = WmsPredictor.from_checkpoint(ckpt, RolloutFlags(pure_physics=True),
                                 physics=PURE_SUMO_PHYSICS_EULER)    # + Euler position
```

Checkpoints load unchanged (no new tensors, no key changes); without the
switches every existing result is bit-identical.

## Known gaps

- `actionStepLength > stepLength` holds the last acceleration between action
  steps; helpers exist (`action_step_enabled`/`action_step_update`) but the WM
  rollout does not carry that state (SUMO default does not need it).
- Semi-implicit Euler speed rule is the WM default and matches SUMO; the
  POSITION rule differs by default (`legacy_trapezoid`) — SUMO parity needs
  `position_rule='sumo_euler'` (or `PURE_SUMO_PHYSICS_EULER`).  The frozen
  FULL-v4 panel ran trapezoid (see above for the measured bias).
- Junction/internal-lane link speeds, perception errors, IDMM adaptation,
  startup delay: out of car-following parity scope (S-RB-MERGE / defaults are
  no-ops).
- desSpeed = the driver/road desired speed (v0); vType maxSpeed cap optional.
  The 0.992 `sumo_fidelity` fudge of the legacy core is NOT applied in the
  SUMO arm (SUMO has no such factor).

## Pending

- Cross-check against `E:/agent-work/rb-shared/idmspec`: DONE (see above; all
  reference vectors pass).
- The full-panel S-RB-EVAL distributional scoring (breakdown/delay) runs on the
  integrated WM-PHYS config (S-RB-INT); this stream's parity evidence is the
  tape check above plus the unit parity tests.

## Post-freeze status (2026-11-03 11:45, fresh-session audit; no new runs)

- S-RB-CF deliverables 1-2 are COMPLETE and consumed: all four frozen INT
  recipes set `cf_model="sumo_idm"` with `sumo_cf_emergency_decel_mps2=9.0`
  and learned corrections OFF. No pending CF asks in rb-shared or the
  REGISTRATION tail (program has moved to RB2/S-COOPMWM; v5/teacher frozen
  under the MAIN `S-RB2-INT RESUME` hold).
- GAP (published as
  `rb-shared/CF_TO_INT_POSITION_RULE_UNCONSUMED_20261103.md`): NO frozen
  recipe (v4/lp/v5/v5-LP) sets `physics.position_rule`, so every frozen panel
  ran `legacy_trapezoid`; the committed, default-OFF `sumo_euler` switch
  (18291ef5) and `PURE_SUMO_PHYSICS_EULER` preset are unconsumed. Evidence and
  consumption line are in that note; revision decision belongs to INT/MAIN.
- Tests re-verified green this session: 92 passed, 16 pre-existing skips.
  ENV NOTE: machine default `python` is now 3.14 (no torch); use
  `C:/Users/jpeng/miniconda3/python.exe` (conda base 3.13.13).
- Branch quiescent unless INT opens a recipe revision.
