# ENTRY: observed-counter vehicle entries for the torch WM-S rollout
Status: COMPLETE. Branch `wms/entry` (parent `wms/integrate` 678b35a). TRAIN-only,
CPU, no SUMO/GPU. All new features are flag-gated default OFF.

## What was added
- `output/wd01/wm_n1/wms/entries.py` — torch port of the numpy observed-counter
  entry model (`closedloop/rollout_entries.py`): per-entrance rate = mean of the
  most recent 12 ten-second bins of the **t0** counter `[2,30]`; seeded Poisson
  draws per 0.5 s step (common random numbers across batch rows); mainline split
  over lanes 0/1 by the observed t0 lane share, ramp -> lane 2; CAV vs HDV by the
  observed t0 CAV share; insertion at the lane start (`ENTRY_S_M`, s_down frame)
  only with a SUMO-like safe gap (`min_gap + headway*v + closing brake`), else
  FIFO pending (<= 6) retried next step; newcomers take free slots from the
  caller's **padding pool** with vType-default physics (5 m, IDM/MOBIL + residual,
  initial speed = mean of the three rearmost lane vehicles). Only RSU-observable
  t0 quantities are read; no recorded future arrival is ever consulted.
- `wms/model.py` — `WmSModel.rollout(..., entries=...)` inserts due arrivals
  before each step and returns a cumulative `entrant` mask (slots that ever held
  a post-t0 newcomer). `step` routes t0 CAVs to the recorded teacher and entrant
  CAVs (no recorded plan) to the supplied `cav_controller` or native hold-speed.
  `entries=None` is byte-identical to before.
- `wms_eval` — `--entries` flag on `run`; `models.load_model(..., entries=...)`
  turns `open_entries` on for the `wms` spec and for `pure_physics`/A11 baselines
  (fair comparison). `wms_plugin` builds the padding pool, derives the counter
  from `state.counter` or `state.base_run_dir` (RSU-observable departures <= t0),
  and appends entrant ids (`__entry_*`) to the predicted frames. A `physics=`
  override lets the plugin score a checkpoint with its trained physics flags.

Entrants are excluded from the per-car loss (no recorded target; the rollout
`entrant` mask makes this explicit) but remain ordinary present vehicles for
traffic-level metrics.

## CPU check
`python _out/entry_cpu.py --states 3` (read-only: run-1 checkpoint
`E:/research-dense/runs/active/WM-S-TRAIN-20261005/box/full/wm_s.pt`, sha256
`97d8f998…`, == the ws-eval copy; dev states `E:/rd-wt/ws-eval/_out/states_dev.json`;
trained physics `sumo_fidelity=true, sumo_idm=true`). Pooled merge-section means
over 0-300 s, native arm, 3 states (`_out/entry_cpu.json`):

| stream   | queue m | stopped | delay s | mean speed m/s | flow veh/min |
|----------|---------|---------|---------|----------------|--------------|
| recorded | 9.65    | 1.32    | 1858.3  | 9.74           | 42.9         |
| off      | 0.52    | 0.115   | 132.3   | 16.73          | 4.67         |
| on       | 0.30    | 0.138   | 225.4   | 22.01          | 32.07        |

Per-state entries inserted (mainline, ramp): w8676 283 (247,36), w8626 199
(195,4), w8629 190 (152,38); no_slot 0, dropped_pending 0/16/0.

Interpretation: entries feed the bottleneck — merge flow rises 4.7 -> 32 veh/min
(recorded 42.9) and mean delay/throughput move toward recorded. But the run-1
checkpoint still under-predicts the merge queue/stopped population (0.30 vs 9.65
m): the base physics free-flows and the residual gate is weak, so newcomers do
not queue. This confirms RUN1_DIAG's recommendation (entries are necessary to
feed the merge) and adds that entries alone are not sufficient with run-1
weights; the queue lever must be trained/calibrated in run 2. The 3-state check
is a diagnostic, not efficacy evidence.

## Tests
- `output/wd01/wm_n1/wms/tests/test_wms_entries.py` (12 synthetic): counter
  window only (no older/future bin leakage); identical arrivals for the same t0
  counter; padding-slot allocation and pool confinement; safe-gap block/retry;
  determinism + common random numbers across rows; observed CAV share; entrant
  CAV native hold; flag-off byte identity; rollout entrant marking.
- `tests/test_wms_eval_wms_plugin.py`: `--entries` default off, overrides wms and
  pure_physics, counter-driven insertion.
- `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` (rl_driving, torch 2.12.1+cpu):
  127 passed, 2 skipped (both pre-existing TRAIN chunk-4 state skips). `ruff
  check` clean on all changed files.

## Enable
`python -m wms_eval run --model wms --ckpt <wm_s.pt> --states <states.json> --entries ...`
(or `--model pure_physics --entries`). Default OFF reproduces the frozen flags.
