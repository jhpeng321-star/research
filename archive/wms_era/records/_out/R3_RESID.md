# R3 residual / lane-rate / long-horizon selection fixes

**ID:** WMS-R3-RESID-20261009. **Branch:** `wms/r3-resid`.
**Base:** `bf298cf16efef0b3a85e0fe9ffea59b44f2122b4`.
Engineering implementation and synthetic CPU verification only; no research
training, GPU, SUMO, or scientific outcome analysis. No protected datasets were
opened. Scientific ledger/conclusions are unchanged; this file is the engineering
verification record. Initial worktree was clean.

## Flags and defaults

- `--residual-soft-bound-mps2 B`: default absent (`None`, OFF). Apply
  `B*tanh((g*delta_a)/B)` before the existing plant acceleration limits.
- `--residual-soft-bound-neg-mps2 N`: default absent. Requires the positive bound;
  use `N*tanh(x/N)` for negative residuals and `B*tanh(x/B)` otherwise.
  Both bounds must be finite and positive. Reject combination with either
  `--residual-cap-mps2` or `--residual-cap-neg-mps2`.
- `--residual-mag-weight W`: default `0.0` (OFF). Add
  `W*mean((g*delta_a)^2)` over valid supervised HDV transitions. The raw residual
  is measured **before** residual bounding and plant acceleration limits.
  Logged weighted term: `residual_mag`.
- `--lane-rate-weight W`: default `0.0` (OFF). Add
  `W*(mean(predicted_change_probability)-target)^2` over valid HDV transitions
  across the **whole batch/unroll**, without time-decay weighting.
- `--lane-rate-target R`: default `0.0123`, inert with weight zero. Finite `[0,1]`.
  Logged weighted term: `lane_rate`.
- `--select-by-long-horizon`: default `False` (OFF). Requires a validation dataset.
- `--val-every N`: default `200`, inert with selection OFF. Positive optimizer
  step cadence; selection ON also validates at each stage end.

Inspection found that the committed legacy residual-cap helpers already use
`tanh`, contrary to the supplied hard-clamp diagnosis. Their behavior is preserved.
The new bound computes its backward slope via a stable `sech^2` expression,
avoiding the float32 `1-tanh(z)^2 == 0` failure at saturation. An underflow-only
dtype-tiny derivative floor handles extreme inputs. Tests verify the true
nonzero derivative at raw residuals `-26` and `+26` with bound `1.5`.

The old hybrid `--residual-magnitude-weight` identifiability penalty remains a
separate existing term. The new L2 term uses raw `g*delta_a` in every WM-S mode.

## Monitoring, gradients, and selection

Always-on train-batch monitors:

- `predicted_lane_change_rate`
- `raw_residual_abs_mean_mps2`
- `raw_residual_abs_p95_mps2`

The history lane head uses `1-P(stay)`; the absolute-lane head uses
`1-P(current_predicted_lane)` at the start of each rollout step. Masks exclude
CAVs, padding, entrants and invalid recorded targets; a transition also requires
the vehicle to be present before the step. Empty supervision reports zero.
Validation emits `mean_`-prefixed monitors (means of batch statistics, including
batch p95s), consistent with its existing batch-mean loss aggregation.

Both new losses participate in `--loss-share-logging`. The lane-rate flag uses
one additional deterministic no-grad pass with Torch RNG restoration to obtain
the whole-batch mean, then supplies its exact derivative chunk by chunk. It does
not retain a whole-unroll activation graph or substitute chunkwise squared losses.
Residual L2 contributions are normalized by the whole-batch valid HDV count.

Selection evaluates the current stage's longest available validation unroll and
60 seconds when available, deduplicating horizons. Strict validation skips
insufficient-length batches; when no batch reaches the requested stage it falls
back to the longest available duration and records that actual duration.
Only finite `mean_position` scores with valid targets at actual horizons and
stages >=60 seconds can select `wm_s_best.pt`. A newly reached longer horizon
resets the comparison even if its score is worse; shorter horizons never replace
it. At the same horizon, strictly lower scores win; ties keep the earlier model.
`TRAIN_RECEIPT.json.long_horizon_selection` records policy, cadence, best step,
horizon and checkpoint. `wm_s.pt` continues to save the final optimizer state’s
model weights. Checkpoint replay restores the optional soft bounds from config.

`wms.gate_stages --require-terms` supports `residual_mag,lane_rate`; positive
weights in adjacent `train_config.json` or `TRAIN_RECEIPT.json` also require them
on every log row. On-path config is written before optimizer execution.

With all new flags at defaults, config serialization and original losses,
diagnostics and parameter gradients are bit-identical to the pinned base. The
three explicitly requested always-on monitors are additive output fields;
complete log/receipt bytes therefore intentionally differ. Existing braking
parity tests retain exact original-field and gradient assertions.

## Verification and artifacts

Interpreter: `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`, PyTorch
`2.12.1+cpu`. All temporary paths were inside this worktree; CUDA visibility was
disabled and no CUDA context was initialized.

- Initial targeted tests: **113 passed**, `_out/R3_RESID_TARGET.xml`.
- Full WM-S suite: **583 passed, 6 skipped**, 237.54 seconds,
  `_out/R3_RESID_TESTS_FINAL.xml`. Peak process working set:
  **889,430,016 bytes**, below 6 GB. Six skips require missing TRAIN-state or staged
  wall-source fixtures. Existing NumPy read-only and escape-string warnings remain.
- Subsequent checkpoint replay checks: **2 passed**,
  `_out/R3_RESID_REPLAY_FINAL.xml`; includes the newly added soft-bound replay test
  and the existing asymmetric-cap replay regression.
- Ruff: passed for the complete WM-S package plus the modified checkpoint loader,
  using isolated settings matching the repo rules so `output/` is not excluded.
- `git diff --check`: passed.

New tests cover bounds, masks, empty losses, exact penalty values, differentiable
lane probabilities, whole-batch lane-rate gradients across 1/2/4-step BPTT chunks,
default-off loss/gradient parity against pinned committed model/trainer source,
CLI/config wiring, loss shares, gate metadata, strict horizons, mocked periodic
checkpoint selection, final-vs-best weights and receipt persistence. Selection
orchestration uses a mocked update function; it performs no optimizer training.

Failure receipts: the first full-suite attempt hit the tool's 120-second timeout;
a diagnostic run then found an old log-key parity assertion rejecting the new
mandatory monitors (`_out/R3_RESID_DIAGNOSE.xml`). Those assertions were updated
without relaxing loss/gradient checks. A later expanded replay/compatibility
command failed because E: was full; its XML could not be written. After enumerating
reparse points (none), only session-created pytest fixture directories were
removed, recovering about 0.4 GB; XML receipts were retained. The narrower final
replay checks passed. No unrelated files were removed or processes manually killed.

## Exact verification commands

Run from `E:/rd-wt/ws-r3-resid`; use fresh basetemp/XML names for a new attempt:

```powershell
$env:TMP = 'E:/rd-wt/ws-r3-resid/_out'
$env:TEMP = $env:TMP
$env:TMPDIR = $env:TMP
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& 'C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe' -m pytest -q output/wd01/wm_n1/wms/tests --basetemp=_out/pytest_r3_new_attempt --junitxml=_out/R3_RESID_NEW_ATTEMPT.xml
& 'C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe' -m pytest -q output/wd01/wm_n1/wms/tests/test_wms_long_horizon.py::test_checkpoint_replay_restores_soft_bounds_from_config output/wd01/wm_n1/wms/tests/test_wms_brakefix.py::test_eval_plugin_replays_the_asymmetric_cap --basetemp=_out/pytest_r3_replay_new_attempt
& 'C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe' -m ruff check --no-cache --isolated --select E4,E7,E9,F --target-version py310 output/wd01/wm_n1/wms output/wd01/wm_n1/wms_eval/wms_plugin.py
git diff --check
```

## CLI examples (documented, not executed)

Future authorized use with permitted workspace-local TRAIN packs/lists staged
at the shown paths. These examples are not a training allocation. From repo root:

```powershell
$env:PYTHONPATH = 'E:/rd-wt/ws-r3-resid/output/wd01/wm_n1'
# All R3 training switches OFF: numerical legacy path.
& 'C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe' -m wms.cli train --pack _in/pack_train --identities _in/identities_train --window-list _in/windows_train.json --val-window-list _in/windows_val.json --out _out/r3_default_off --device cpu --unroll-s 10 --steps 1 --batch-size 1
# All R3 fixes ON, with whole-batch penalties and 60/120-second selection.
& 'C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe' -m wms.cli train --pack _in/pack_train --identities _in/identities_train --window-list _in/windows_train.json --val-window-list _in/windows_val.json --out _out/r3_fixes_on --device cpu --rollout-mode hybrid --lane-head history --unroll-s 10 --unroll-s 30 --unroll-s 60 --unroll-s 120 --stage-steps 400,400,400,400 --residual-soft-bound-mps2 1.5 --residual-soft-bound-neg-mps2 3.0 --residual-mag-weight 0.001 --lane-rate-weight 1.0 --lane-rate-target 0.0123 --select-by-long-horizon --val-every 100 --loss-share-logging
& 'C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe' -m wms.gate_stages _out/r3_fixes_on/train_log.jsonl --stages 10,30,60,120 --require-terms position,speed,lane,residual_mag,lane_rate
```

Engineering verification is complete. Training efficacy and the supplied MAIN2
diagnostic values were not evaluated in this task.
