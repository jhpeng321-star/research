# S-RB-CF handoff

Branch `c4/rb-cf`, base `f0db3c82` (c4/wmfixmerge). Owner authorization: the
WM-PHYS REBUILD PROGRAM line of the active C4 REGISTRATION (2026-10-31 23:12).
Task S-RB-CF: SUMO 1.27.1 IDM car-following parity + a pure-physics preset.

## Status (updated as work lands; all times 2026-10-31/11-01 +08:00)

- `e9992767` **SUMO IDM parity module** `src/wms/physics/idm_sumo.py`: a
  line-annotated tensor port of SUMO 1.27.1 `MSCFModel_IDM` (`_v` iterated safe
  speed with the `TS/0.25` sub-stepping = 2 iterations at 0.5 s; `minNextSpeed`
  with the `min(emergencyDecel, 1.5)` relaxations; `minNextSpeedEmergency`;
  `maxNextSpeed`; the `finalizeSpeed` composite; `stopSpeed` incl. the exact
  `maximumSafeStopSpeedEuler` standing fallback with the action-step headway;
  `freeSpeed`; `insertionFollowSpeed` (reference for S-RB-MERGE)).
  Source anchors in the module docstring; read-only SUMO source copies at
  `E:/Temp/opencode/sumo_src` and `E:/Temp/opencode/sumo-1.27.1-rb-lane`
  (not executed).
- `742922b6` **Switch** `physics.cf_model = 'sumo_idm'` (default `'legacy'`,
  bit-identical): replaces the legacy textbook-IDM core AND its legacy
  emergency/yield add-ons in `baseline_accel` by the SUMO step. New config
  keys: `sumo_cf_stepping_s` (0.25), `sumo_cf_emergency_decel_mps2` (None =
  decel, the SUMO default), `sumo_cf_vtype_max_mps` (None = unbounded),
  `sumo_cf_action_step_s` (None = every step). Per-driver settings map 1:1
  (encoder `v0/T/min_gap/accel/decel` or population defaults).
- `6b0f4969` **Pure-physics switch** `WmSModel.learned_accel_off` (runtime
  attribute, never in a checkpoint, default OFF = bit-identical): the final HDV
  acceleration IS the physics core; drops residual attention, temporal/slow
  emission, mainline residual, negotiation yield/ramp deltas, merge-core yield,
  follower courtesy, learned mainline yield, ramp speed adaptation, merge
  response, and the plant clamp. Preset dict
  `wms.physics.idm_sumo.PURE_SUMO_PHYSICS` (physics-side overrides:
  cf_model=sumo_idm + every learned/hand acceleration-adjacent rule off) and
  `RolloutFlags.pure_physics` (`wms.eval.predictor`).

## Tests

`python -m pytest tests/test_rb_cf_sumo_idm.py tests/test_rb_cf_switch.py
tests/test_rb_cf_pure_physics.py tests/equivalence -q` (conda env
`rl_driving` python; 15 + 5 + 5 + 60 pass). Expected values are derived from an
independent scalar transcription of the SUMO source formulas plus hand-derived
anchors (see module/test docstrings).

## How to run the A/B (for S-RB-INT)

```python
from wms.eval.predictor import RolloutFlags, WmsPredictor
from wms.physics.idm_sumo import PURE_SUMO_PHYSICS
p = WmsPredictor.from_checkpoint(ckpt, RolloutFlags(pure_physics=True),
                                 physics=PURE_SUMO_PHYSICS)
```

Checkpoints load unchanged (no new tensors, no key changes); without the
switches every existing result is bit-identical.

## Known gaps

- `actionStepLength > stepLength` holds the last acceleration between action
  steps; helpers exist (`action_step_enabled`/`action_step_update`) but the WM
  rollout does not carry that state (SUMO default does not need it).
- Semi-implicit Euler speed rule + ballistic position rule assumed (SUMO
  defaults); the WM integrator already matches that combination.
- Junction/internal-lane link speeds, perception errors, IDMM adaptation,
  startup delay: out of car-following parity scope (S-RB-MERGE / defaults are
  no-ops).
- desSpeed = the driver/road desired speed (v0); vType maxSpeed cap optional.
  The 0.992 `sumo_fidelity` fudge of the legacy core is NOT applied in the
  SUMO arm (SUMO has no such factor).

## Pending

- Teacher-forced short-rollout check on FIT tapes 969400-969431: HDV
  speed/gap error of WM2 vs the pure SUMO-IDM arm (running; results below).
- Cross-check against `E:/agent-work/rb-shared/idmspec` when its DONE.md lands.
