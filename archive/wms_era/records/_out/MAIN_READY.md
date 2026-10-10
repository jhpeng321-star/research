# WM-S integrated branch: READY

Engineering ID: **WMS-MAIN-MERGE-20261009**. All seven requested branches are
integrated on `wms/main`; release suites: **438 passed, 6 skipped**. See
[`MAIN_MERGE.md`](MAIN_MERGE.md) for merge/conflict and failure accounting.

## Optional real-reference verification for the coordinator

This execution did not access external source/data locations. Stage **copies**
of the following approved fixtures inside this workspace (no junction to another
worktree and no sealed/acceptance fixtures):

- `_ext/campaign_chunk4/w8676m2700p20t360__base/` and
  `_ext/campaign_chunk4/w8677m2700p20t360__base/`, including their saved-state
  manifests and matching TRAIN-only report/departure inputs.
- `_in/traffic_shepherd/wall_group.py` and
  `_in/traffic_shepherd/wall_group_v22.py`, the frozen source pair required by
  the wall-executor reference tests. No SUMO execution is involved.

Run from `E:/rd-wt/ws-main` using a fresh basetemp directory:

```powershell
$env:TMP="E:/rd-wt/ws-main/_out"
$env:TEMP=$env:TMP
$env:CUDA_VISIBLE_DEVICES=""
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:WMS_WALL_SOURCE_DIR="E:/rd-wt/ws-main/_in/traffic_shepherd"
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m pytest output/wd01/wm_n1/wms/tests/test_rollout2_parity.py output/wd01/wm_n1/wms/tests/test_wall_exec.py -q --basetemp=_out/pytest_main_real_reference --junitxml=_out/MAIN_REAL_REFERENCE_TESTS.xml
```

Expected coverage with both fixture groups present: **10 tests** (2 real-state
comparisons + 4 standalone wall tests + 4 frozen-source wall parity cases).
Report the actual results; this count is a coverage target, not an executed
result. Preserve any failure receipt and use another unique basetemp for a retry.
No new calibration, dataset evaluation or training follows automatically.

## Integrated live-action input

`cav_controllers.action_timeline` (or `state.action_timeline`) accepts the
`wms_action_timeline/1` mapping emitted by `ActionTimeline.to_dict()`.
`vehicle_ids` identifies its columns; IDs must match the t0 roster. If omitted,
columns must already match the plugin's sorted t0 slots. Partial CAV rosters
retain the fixed plan for other slots. Entrants use
`default_entrant_controller` (default native). Set `cav_mode="closed_loop"` to
execute native, damping or PI; direct actions use their issued acceleration and
target-speed fields. Indexed bindings remain available through
`wms_eval.indexed_switch.bind_indexed_switch_timeline`, which memory-maps identity
and dense slot arrays and fails closed on unmatched identity/command evidence.

Synthetic compatibility/parity tests verify residual, driver_params, hybrid,
history lane heads, receipt physics/projection/caps, entries-v2 and constant
timeline equivalence. Numeric compact timeline packs declare
`controller_timeline_schema="wms_action_timeline/1"` (or their explicit
`controller_vocab`); overlapping unmarked legacy IDs are refused.
