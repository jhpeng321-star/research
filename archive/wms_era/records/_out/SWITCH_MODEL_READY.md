# WM-S switch model readiness

**Status: READY — the two remaining CPU implementation items are complete and
covered by targeted synthetic/unit tests.** This is implementation readiness,
not world-model efficacy evidence or a full-pack validation.

## Completed

- Closed-loop CAV rollouts allow post-t0 entrants. Plan field
  `default_entrant_controller` selects their action and defaults to `native`;
  supported values are `native`, `damping`, and `pi-external-u` (with the
  existing `pi` aliases). A damping controller is created when an entrant first
  becomes live and retained for that slot. The selected controller runs at the
  existing 1 s control / 0.5 s physics cadence.
- `wms_eval.indexed_switch.bind_indexed_switch_timeline` joins a switching
  schedule and suffix command JSONL to indexed pack slots. It verifies every
  scheduled CAV against `vehicle_map.json` and the t0 row in `identities.npy`,
  then resolves presence through the tick-local `slot_vehicle.npy` dense-truth
  map. The resulting timeline remains indexed by the stable t0 pack columns.
  Identity, row, schedule, timestamp, and direct-command mismatches fail closed.
  `.npy` arrays are opened with `mmap_mode="r"` and only one dense tick is read
  at a time.
- Existing timeline support remains in place: direct commands retain issued
  acceleration/target speed, legacy fixed-controller windows remain constant
  timelines, and controller switches use the existing one-second action / two
  half-second physics-step cadence.
- Generator schema reference: `_in/SWGEN_READY.md`, including the
  `vehicle_id`/slot identity mapping and timestamp semantics.

## Verification

Interpreter: `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`; PI sources:
`PYTHONPATH=_in/pi_deps;output/wd01/wm_n1`.

Targeted tests:

```powershell
$env:PYTHONPATH = "_in/pi_deps;output\wd01\wm_n1"
& 'C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe' -m pytest -q `
  output/wd01/wm_n1/wms/tests/test_indexed_switch.py `
  output/wd01/wm_n1/wms/tests/test_action_timeline.py `
  output/wd01/wm_n1/wms/tests/test_junction_presence.py
```

Result: **19 passed**. The synthetic indexed-pack fixture verifies identity
binding, dense-slot changes across ticks, direct suffix command alignment, and
fail-closed behavior for an absent t0 identity. Entrant tests verify the plan
default and spawn-local controller lifecycle.

Ruff:

```powershell
& 'C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe' -m ruff check `
  output/wd01/wm_n1/wms_eval/wms_plugin.py `
  output/wd01/wm_n1/wms_eval/indexed_switch.py `
  output/wd01/wm_n1/wms/tests/test_indexed_switch.py `
  output/wd01/wm_n1/wms/tests/test_junction_presence.py
```

Result: **passed**.

No real indexed pack sidecars were available for a pack-specific audit; the
requested synthetic fixture covers the join contract. No SUMO, GPU, remote
machine, sealed data, or full generation was used.

## Next action

Use the binding on an authorized switching evaluation pack when one is staged;
full generation remains separate from this readiness implementation.
