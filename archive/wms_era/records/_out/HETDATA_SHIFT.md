# HETDATA t0-shift runbook (WM-S2 VAL/TEST)

Build the **same kind of shifted state sets** the C1 v2 materialiser builds
(`output/wd01/wm_n1/wms_eval/materialize.py`, the `shift_steps` /
`--shift-seconds` path) from the WM-S2 heterogeneous-driver (hetdata) VAL and
TEST packs, then score them with the WM-S2 driver encoder.

The helper is `output/wd01/wm_n1/wms_eval/hetdata_shift.py`. It is CPU-only,
reads the pack with `numpy.load(..., mmap_mode="r")`, processes one manifest
item at a time (selection stores only integer ticks), and stays well under the
6 GB rule. It touches no CAL/HOLDOUT/sealed data.

## 0. Local vs box

* **The local `wms/evalshift` worktree has only the TRAIN mechanics check pack**
  (`E:/rd-wt/ws-hetdata/_out/box_bundle/hetpack_check/pack_hetdata_train`,
  1 item, 121 ticks / 60.5 s). There is no local VAL or TEST pack. The VAL/TEST
  commands below are **for the box**.
* On the box the packs and sidecars live under the per-split hetpack output
  roots, e.g. `hetpack3_VAL/pack_hetdata_val` + `driver_params_hetdata_val` and
  `hetpack3_TEST/pack_hetdata_test` + `driver_params_hetdata_test`
  (`HETPACK_RECEIPT_val.json` records `"pack_dir": "pack_hetdata_val"`).
* The helper must sit next to the box's `driver.py` in the WM-N1 tree
  (`$WMS_ROOT/wms_eval/hetdata_shift.py`). Copy the file there before running.

## 1. Environment (box)

```bash
export WMS_ROOT=/root/autodl-tmp/gpu1/output/wd01/wm_n1   # integrated WM-N1 tree
export PYTHONPATH="$WMS_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/root/autodl-tmp/gpu1/env/rd_gpu/bin/python            # box CPU env
ls "$WMS_ROOT/wms_eval/hetdata_shift.py"                  # helper present?
```

## 2. Materialise the shifted VAL/TEST states

`--shift-to-t0-s 120 --min-t0-s 120 --margin-steps 10` lands t0 at
`tick_start + round(120/0.5) + 10 = tick_start + 250` (125 s), the first BUSY
tick at/after that, with a real 10-step pre-t0 history. `--min-t0-s 120` makes
the run fail closed (item skipped, reported in `skipped`) if a window cannot
reach 120 s. `--horizon-steps 0` is correct for driver recovery (only pre-t0
history + the t0 state are needed); set `--horizon-steps 600` (300 s) if you
also want room reserved for a future horizon.

```bash
PACK_VAL=/root/autodl-tmp/hetpack3_VAL/pack_hetdata_val
PACK_TEST=/root/autodl-tmp/hetpack3_TEST/pack_hetdata_test
OUT=/root/autodl-tmp/wms2_shift
mkdir -p "$OUT"

"$PY" -m wms_eval.hetdata_shift \
  --pack "$PACK_VAL" \
  --out "$OUT/hetdata_val_shift120.json" \
  --shift-to-t0-s 120 --min-t0-s 120 --margin-steps 10 \
  --history-steps 10 --horizon-steps 0 --min-vehicles 1

"$PY" -m wms_eval.hetdata_shift \
  --pack "$PACK_TEST" \
  --out "$OUT/hetdata_test_shift120.json" \
  --shift-to-t0-s 120 --min-t0-s 120 --margin-steps 10 \
  --history-steps 10 --horizon-steps 0 --min-vehicles 1
```

Each command prints a JSON summary (`n_states`, `n_requested`, `n_skipped`,
`skipped[]`). Every written record has the `wms_driver_states/1` schema the
existing `materialise_pack_states` writes: `world`, `arm`, `t0_s`, `tick`,
`tick_start`, `tick_end`, `item_position`, `state{time_s,vehicles[]}`,
`veh_kin`, `veh_mask`, `history{kin,mask,target_mask}`, plus additive
`t0_window_s` / `shift_steps` / `shift_s` / `history_steps` diagnostics.

## 3. Score with the WM-S2 driver encoder

`--sidecar` is the slot-aligned hetpack driver bundle (`driver_params.npy`
`[n_items, k_max, 11]`), so the report is keyed by the original manifest
`item_position` and pack slot exactly as before. Outputs `DRIVER_RECOVERY.json`
+ `DRIVER_PRED.json`. Drop `--sidecar` to get the predictions-only dump.

```bash
CKPT=/root/autodl-tmp/wm_s2_driver/wm_s.pt   # WM-S2 driver_params checkpoint

"$PY" -m wms_eval driver \
  --states "$OUT/hetdata_val_shift120.json" \
  --ckpt "$CKPT" \
  --sidecar /root/autodl-tmp/hetpack3_VAL/driver_params_hetdata_val \
  --out "$OUT/score_val"

"$PY" -m wms_eval driver \
  --states "$OUT/hetdata_test_shift120.json" \
  --ckpt "$CKPT" \
  --sidecar /root/autodl-tmp/hetpack3_TEST/driver_params_hetdata_test \
  --out "$OUT/score_test"
```

### CLI form note

In this tree the first token of `python -m wms_eval` is the dispatch
(`wms_eval/cli.py`), so the working forms are
`python -m wms_eval materialise ...` and `python -m wms_eval driver ...`.
The doubled `python -m wms_eval driver materialise ...` /
`python -m wms_eval driver driver ...` forms fail with
`unrecognized arguments` / missing `--states`.

## 4. Geometry (why this shift)

* **History.** `HISTORY_H = 10` steps (5 s at 0.5 s). The trainer's
  `wms.train._prior_history` uses `lo = tick - HISTORY_H + 1` and the encoder
  consumes exactly that pre-t0 window. The stored hetdata item's `tick` is the
  FIRST NON-EMPTY tick of the window (not `tick_start + 1`), so the stored t0 has
  no real 10-step history behind it. Moving t0 forward inside the same window
  gives the encoder real recorded history.
* **Why t0 >= 120 s.** The run window begins near the origin and the first ticks
  are a cold-start transient. `tick_start + round(120/0.5) + 10-step margin`
  (`tick_start + 250`) lands on a warmed-up, BUSY snapshot: at least 120 s into
  the window, with a full 10-step history before it.
* **Bounds.** For each item the helper chooses the first tick with
  `veh_mask` occupancy `>= --min-vehicles` (BUSY) in
  `[max(tick_start + history_steps - 1, tick_start + ceil(min_t0_s/DT)), tick_end - 1 - horizon_steps]`,
  after `tick_start + round(shift_to_t0_s/DT) + margin_steps`. The shift is
  therefore bounded above by **`tick_end - horizon_steps`** (a valid t0 needs
  `new_tick + horizon_steps <= tick_end - 1`) and below by the history width and
  `--min-t0-s`; an item with no fitting busy tick is skipped and reported, never
  silently placed at a wrong time. `history.kin` is read as
  `kin[max(tick_start, new_tick - history_steps + 1) : new_tick + 1]`, so it
  never crosses the run/window start.

## 5. Local validation (this worktree, CPU, no long jobs)

Executed with the local TRAIN check pack and the local driver checkpoint
`_out/wms_driver_test/wm_s_driver.pt` (`rollout_mode=driver_params`).

```bash
export PYTHONPATH="E:/rd-wt/ws-evalshift/output/wd01/wm_n1"
PY="C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe"
PACK="E:/rd-wt/ws-hetdata/_out/box_bundle/hetpack_check/pack_hetdata_train"
TMP="C:/Users/jpeng/AppData/Local/Temp/opencode/hetdata_shift_val"

# a) real TRAIN pack, shift fitted to its 60.5 s window
"$PY" -m wms_eval.hetdata_shift --pack "$PACK" --out "$TMP/train_shift50.json" --shift-to-t0-s 50
# -> n_states=1, tick=110, t0_window_s=55.0, history.kin=(10,192,7), 49 occupied slots

# b) same pack, default 120 s -> fail closed
"$PY" -m wms_eval.hetdata_shift --pack "$PACK" --out "$TMP/train_shift120.json" --shift-to-t0-s 120
# -> n_states=0, skipped: "window too short: need tick >= 250 ... tick_end-horizon=120"

# c) scorer consumes the shifted output (predictions-only)
"$PY" -m wms_eval driver --states "$TMP/train_shift50.json" \
  --ckpt "E:/rd-wt/ws-evalshift/_out/wms_driver_test/wm_s_driver.pt" --out "$TMP/score"
# -> {"mode":"predictions-only","n_records":1}; DRIVER_PRED.json has 192 slots,
#    9 values + 9 uncertainty each, tick=110, item_position=0
```

The 120 s property itself cannot be shown on the 60.5 s local fixture; it was
exercised on a synthetic 320-tick, 3-item copy of the same pack (temp only),
which produced `new_tick=250`, `t0_window_s=125.0` and `history.kin=(10,192,7)`
for all three items. The box packs are full 1200 s runs, so the real VAL/TEST
commands in section 2 use the 120 s default directly.
