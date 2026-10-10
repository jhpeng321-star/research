# WM-S2 readiness on the WM-S2-HETDATA (heterogeneous-driver) data — CPU-only

Status: **READY (code) — the WM-S2 readiness gaps found in the previous task are
closed at the CPU-code level: dense per-driver truth, a hetpack bundle/dense
loader with rank-correlation and per-class recovery scoring, a TRAIN-scoped
validation holdout, and an offline VAL/TEST scorer. What remains is gpuA
execution only (owner-authorized), in the order given in section 7; no GPU run,
no SUMO and no remote step was performed by this task.**
Workspace: `E:/rd-wt/ws-s2int` (branch `wms/s2int`). CPU only: no GPU, no SUMO,
no SSH, no remote machines; nothing under `E:/research-dense/runs/` was modified;
no CAL/HOLDOUT/sealed/chunk-5 data was opened. The staged `_out/` run-2
artefacts pre-date this task and are carried only by the requested
`wms/trafloss-fix` merge.

## 0. Merge `wms/run3` + WM-S2 CPU speed fix (2026-10-07)

Merged `wms/run3` (run-3 commits `4b3fbef`, `ba860fb`, `da567b3`, `6d69314`;
merge base `9e130ec`) into `wms/s2int`, keeping BOTH feature sets.  Two files
conflicted and were resolved as a union:

* `cli.py`: the unified flag set keeps the WM-S2 `--driver-sidecar`,
  `--val-holdout-*`, `--projection-enabled`/`--projection` surface AND run-3's
  `--loss-share-logging`; `window_list=train_window_list` (the WM-S2
  `--val-holdout-frac`-derived list) is kept.
* `train.py`: the per-stage record keeps run-3's horizon-invariant loss logging
  (traffic logged per chunk, `chunks`, `loss_share`) AND the WM-S2
  `driver_param_stats`/`entrants`/`valid_hdv_slots_t0` fields.

Run-3's fixes are carried in: `wms/stability.py` segmented-prefix-min projection,
`wms/entries.py` idle fast-path, `wms/gate_stages.py` multi-stage gate,
`wms/plan_steps.project()` + `--stage-steps/--max-hours`, and the
`test_run3_gate.py` / `test_run3_vectorization_parity.py` suites.  The WM-S2
modules (`dense_truth.py`, `driver_encoder.py`, `driver_sidecar.py`,
`val_split.py`, `gate_s2.py`) and their tests are preserved.

### WM-S2-only hot paths checked and vectorised

The WM-S2-only paths were checked for the same anti-patterns (per-vehicle Python
loops, `.item()/bool()/.cpu()` syncs inside the rollout).  The driver encoder,
the `driver_params` rollout and `val_split` contain no per-vehicle loop and no
device sync inside the rollout.  Two paths were fixed:

* `WmSTrainer._driver_param_stats` (called once per optimizer step in
  `driver_params` mode) issued a leading `bool(valid.any())` sync plus 3 device
  syncs per parameter (28 syncs/step); it now does one boolean gather and a
  single host transfer.
* `dense_truth.build_run` scattered the per-(tick, vehicle) slot map with a
  Python loop; `_scatter_records` now does one fancy-index scatter.

Both are pinned to the original numerics by
`wms/tests/test_wms_s2_vectorization_parity.py` (driver-stat parity over
float32/float64 and both `[B]`/`[B,K]` mask conventions; dense-truth scatter vs
the reference loop).

### `s2_chain.sh`: explicit budget + multi-stage gate

`output/wd01/wm_n1/wms/s2_chain.sh` now runs a single multi-stage smoke over
every curriculum stage with every loss term active and `--loss-share-logging`,
checked by BOTH `wms.gate_stages` (all six terms, dominance/growth, shares) and
`wms.gate_s2` (per-stage record count + driver-parameter bounds).  The full run
uses an EXPLICIT `--stage-steps 2500,2500,1875,1250,400` (the 10/30/60/120 s
stages match C1 exactly); the smoke only PROJECTS the wall-clock hours via
`wms.plan_steps --stage-steps ... --max-hours 14`, and the chain refuses to start
above `MAX_HOURS=14` (it never silently shrinks the steps).  The separate
400-step reference smoke is dropped (its role is covered by the multi-stage
smoke + `gate_s2`).

### Tests (this task)

```
python -m pytest output/wd01/wm_n1/wms/tests -q          # 198 passed, 2 skipped
python -m pytest tests/test_wms_eval_*.py tests/test_wms_decide.py \
    output/wd01/wm_n1/closedloop/tests -q                # 88 passed
ruff --isolated --select E4,E7,E9,F <changed modules>    # clean
```

The 2 skips are the pre-existing `test_rollout2_parity.py` TRAIN chunk-4 states
absent on this box.  `bash -n s2_chain.sh` is clean and the `plan_steps` explicit
projection was exercised end-to-end on a synthetic smoke log (refusal at the cap).

### Code tar

`_out/code_s2_2a0eeb65aa59.tar` (9.95 MB, `git archive` of commit `2a0eeb65aa59`,
scoped to `output/wd01/wm_n1`), sha256
`21694ac21fe388041fd3010d08d990d145fcfa0e8a97703b8429ca2a2778ccee`
(`_out/code_s2_2a0eeb65aa59.tar.sha256`).  It supersedes `code_s2_5c6e77b.tar`.

## 1. Merge `wms/trafloss-fix` into `wms/s2int`

Merged `wms/trafloss-fix` (`9e130ec`, "fix Run2 traffic-loss blow-up") into
`wms/s2int`. Three files conflicted (`cli.py`, `model.py`, `train.py`); resolved
as a union that keeps the WM-S2 surface and adds the run-2 fixes:

* `cli.py`: one unified flag set — `--entries-v2/--entry-pad`, `--rollout-mode`,
  `--driver-sidecar`, `--projection-enabled` (alias `--projection`),
  `--residual-cap-mps2`, `--lane-logit-cap`, `--stability-*`,
  `--conservation-metric`, `--conservation-weight`, `--traffic-*`,
  `--state-noise-*`. `--projection-enabled` and `--projection` share
  `dest="projection_enabled"`.
* `model.py` / `train.py`: WM-S2 driver-encoder/driver-params rollout + entries +
  traffic loss + projection/stability/conservation together; the traffic-loss fix
  (`traffloss.py`, `test_traffloss_fix.py`) and `gate_run2.py` come in from the
  merge. No duplicate keyword arguments.

Tests (conda `rl_driving`, torch 2.12.1+cpu):

```
python -m pytest output/wd01/wm_n1/wms/tests -q          # 159 passed, 2 skipped
python -m pytest tests/test_wms_eval_wms_plugin.py \
    tests/test_wms_eval_c1.py output/wd01/wm_n1/closedloop/tests -q  # 52 passed
python -m ruff check output/wd01/wm_n1/wms output/wd01/wm_n1/wms_eval  # clean
```

The 2 skips are the pre-existing `test_rollout2_parity.py` TRAIN chunk-4 states
absent on this box.

## 2. Anomaly findings (from the code, not the raw packs)

The box small outputs are under
`E:/research-dense/runs/active/WM-S2-HETDATA-20261006/box/` (read-only). The
builder is `E:/rd-wt/ws-hetpack/hetpack/`; the WM-S2 loader is
`output/wd01/wm_n1/wms/{train,driver_sidecar}.py`.

### (a) `identity_dir` is null for VAL/TEST — is it needed?

**No for the WM-S2 trainer; the null is intended.** The identity sidecar is a
TRAIN-only window-integrity artefact:

* `hetpack/build.py:248-252` builds it only when `split == "TRAIN"`.
* The frozen builder is TRAIN-only by construction:
  `train/identity_sidecar.py` writes `TRAIN_IDENTITY_RECEIPT.json` with
  `split="TRAIN"`, `_metadata_identity` accepts only `TRAIN/CALIBRATION`, and
  `_plan` selects only `rec["split"] == "TRAIN"` (so calling it on a VAL/TEST
  pack would raise, not silently skip).
* The only consumer is `wms.cli train` → `TrajectoryDataset.from_real_packs`
  (`train.py:442`), and `_read_identity_rows` (`train.py:696`) hard-requires
  `receipt["split"] == "TRAIN"`; `validate_train_source` refuses any pack whose
  manifest split is not TRAIN. So the trainer cannot load VAL/TEST packs at all.
* Validation monitoring reuses the **same** TRAIN packs/identities filtered by
  `--val-window-list`; held-out scoring is done offline by `wms_eval`
  (state-file based, no identity sidecar).

Consequence (a real integration gap, but not an identity fix): the hetpack
`windows_val_hetdata.json` / `windows_test_hetdata.json` are scoped to
`pack_hetdata_val` / `pack_hetdata_test`, so they cannot be passed as
`--val-window-list` to the TRAIN corpus (`lpo.window_filter` fails closed on
unmatched keys). Validation on the disjoint VAL worlds must therefore run
through the offline evaluator, not the trainer's monitoring loop. Building
VAL/TEST identities would not help without also weakening the deliberate
TRAIN-only guard, and is **not** done.

### (b) `driver_n_vehicles = 106/34/35` — what it counts, and is truth present?

`driver_n_vehicles` counts distinct `(world, vehicle_id)` pairs that carry truth
**at each run's single packed snapshot tick** (`driver_params.py:158-167,185`).
It is low because the pack snapshot is the run's **first non-empty tick**, not a
busy tick:

* `corpus/pack.py:182-185`: `snap_tick = tick_start + first tick where
  veh_mask.any()`; the manifest item tick is `snap_tick` (`pack.py:270`).
* The identity sidecar reads that same tick (`identity_sidecar.py:338`).
* Measured on the TRAIN sidecar: `tick - tick_start` = 2–16 (i.e. 1.0–8.0 s into
  a 1200 s run, `n_ticks=2401`), and `identities.npy` has only 1–3 non-negative
  slots per row (median 1); `vehicle_map.json` has exactly 106 entries.
* The heterogeneous runs start from an empty network with flow insertion
  (`loading` 2000–2700 veh/h), so at 1–8 s only 1–3 vehicles have entered.

**This finding is now CLOSED by the dense sidecar** (`wms.dense_truth`, section
4a): for any `(run, tick, slot)` the HDV vehicle id and its 11 truth parameters
can be looked up over every tick of every raw run, at ~1 HDV/world only at t0 but
densely after insertion. The paragraph below documents the original limitation.

Therefore **per-driver truth is NOT present for every HDV**: the
`scenario/hetdata_truth.csv` contains one row per scheduled vehicle (order of
~10^3 per run; the TEST receipt records a 219,559-byte truth CSV), but the
driver sidecar only aligns the ~1–3 vehicles present at the near-empty snapshot
tick. This is a data-generation/pack-snapshot choice, not a parsing bug:
`driver_params.load_truth` reads the full table; the alignment is the limit.

Impact for WM-S2: the trainer's `--start-shift-prob 0.75` shifts most windows to
random later (busy) ticks (`train.py:730-748`, identity rows intentionally
unused there), so supervised HDV motion is available. But per-driver
*identification ground truth* is thin (≈1 HDV/world at t0). A future
re-generation should choose a busy snapshot tick (or extend the driver sidecar
over the run) if per-driver recovery is to be scored densely.

## 3. `s2_chain.sh` on the hetpack3 paths + multi-stage smoke gate

`output/wd01/wm_n1/wms/s2_chain.sh` now reads the heterogeneous-driver artefacts
on the box (defaults; all overridable):

```
$HP3/hetpack3_TRAIN/pack_hetdata_train            # TRAIN pack (trainer input)
$HP3/hetpack3_TRAIN/identities_hetdata_train      # TRAIN identities (+ receipt)
$HP3/hetpack3_TRAIN/windows_train_hetdata.json    # TRAIN keep list
$HP3/hetpack3_{TRAIN,VAL,TEST}/driver_params_hetdata_{train,val,test}
$HP3/hetpack3_VAL/pack_hetdata_val, windows_val_hetdata.json
$HP3/hetpack3_TEST/pack_hetdata_test, windows_test_hetdata.json
```

TRAIN pack/identities/windows are hard-required; the VAL/TEST packs, window
lists and driver sidecars are existence-checked as held-out **evaluation**
inputs (the trainer is TRAIN-only). The gate is now a single multi-stage smoke
(`--rollout-mode driver_params --traffic-loss --entries-v2
--projection-enabled --residual-cap-mps2 --lane-logit-cap --stability-weight
--conservation-weight --loss-share-logging`) over every curriculum stage
(default `10,30,60,120,300`), checked by **both** `wms.gate_stages` (all six
terms present and finite, no term >100x the position term, no >10x growth
between stages, `loss_share.jsonl` per stage) and `wms.gate_s2` (every stage
logged exactly `SMOKE_STEPS` records; driver-parameter encoder outputs finite and
inside their hard bounds).  The separate 400-step reference smoke was dropped;
the multi-stage smoke's measured per-stage step time feeds the explicit-budget
projection.  See section 0 for the current chain.

`wms.gate_s2` is covered by `wms/tests/test_wms_gate_s2.py` (6 tests) and by an
end-to-end CPU run of the real `WmSTrainer.fit` over a 2-stage curriculum with
all S2 flags on (log accepted by `gate_s2`).

## 4. Exact launch command (gpuA, owner-authorized only)

```bash
# on gpuA, after uploading code_s2_2a0eeb65aa59.tar (extracted to $W/code):
export W=/root/autodl-tmp/wms_s2_20261006/work
export HP3=/root/autodl-tmp
export PY=/root/autodl-tmp/gpu1/env/rd_gpu/bin/python
bash $W/code/output/wd01/wm_n1/wms/s2_chain.sh
```

Equivalent explicit trainer invocation (what the chain runs for Stage 2):

```bash
$PY -m wms.cli train \
  --pack /root/autodl-tmp/hetpack3_TRAIN/pack_hetdata_train \
  --identities /root/autodl-tmp/hetpack3_TRAIN/identities_hetdata_train \
  --window-list /root/autodl-tmp/hetpack3_TRAIN/windows_train_hetdata.json \
  --val-holdout-frac 0.1 --val-holdout-seed 0 \
  --out $W/full --device cuda --threads 2 --num-workers 5 --batch-size 8 \
  --sumo-fidelity --start-shift-prob 0.75 --truncation-steps 20 \
  --unroll-s 10 --unroll-s 30 --unroll-s 60 --unroll-s 120 --unroll-s 300 \
  --stage-steps 2500,2500,1875,1250,400 \
  --rollout-mode driver_params --traffic-loss --traffic-weight 0.1 \
  --traffic-bin-s 30 --traffic-ramp-start-s 30 --traffic-ramp-end-s 120 \
  --entries-v2 --entry-pad 8 --projection-enabled --projection-min-gap-m 2.0 \
  --residual-cap-mps2 1.5 --lane-logit-cap 1.0 \
  --stability-weight 0.1 --stability-margin 1.0 --stability-probes 1 \
  --conservation-weight 0.1 --loss-share-logging
```

## 5. Estimated GPU time

The chain uses an **explicit** optimizer-step budget
`STAGE_STEPS=2500,2500,1875,1250,400` (the 10/30/60/120 s stages match C1
exactly).  The on-box multi-stage smoke measures the real per-stage step time;
`wms.plan_steps --stage-steps ... --max-hours 14` PROJECTS the wall-clock hours
and the chain **refuses to start** above `MAX_HOURS=14` (GPU shared with other
trainings) rather than shrinking the steps.  With the run-3 CPU fixes applied the
projected time is the C1-measured per-step cost times this budget (~9 h).  The
multi-stage smoke adds a small bounded overhead (5 stages × `SMOKE_STEPS=6`
updates). No GPU run was started by this task.

## 6. New in this task (closes findings (a), (b) and the bundle-loader gap)

### 6a. Dense per-driver truth — `wms/dense_truth.py`, script `wms/build_dense_truth.sh`

Streams each raw run (`run/physics_steps.jsonl.gz` + `run/scenario/hetdata_truth.csv.gz`)
and writes, per run, a `slot_vehicle.npy [n_ticks, k_max]` (`-1` = empty, else a row
into `vehicle_ids.json`) plus `vehicle_params.npy [n_veh, 11]`,
`vehicle_class.npy [n_veh]`, so **any `(run, tick, slot)` resolves to the HDV
vehicle id and its truth parameters**. The per-tick slot pool is the SAME
persistent-pool reuse algorithm as `features.featurizer.parse_run` (continuing
vehicles keep their slot; a freed slot is reused at the lowest free index), so
slot indices match the packed windows. A top-level `DENSE_TRUTH_RECEIPT.json` +
`runs.jsonl` + per-run `SHA256SUMS` bind the output. One run per worker,
`open_memmap` writes, a whole pack is never loaded (memory rule).

### 6b. Slot-aligned bundle / dense loader — `wms/driver_sidecar.py`

`load_driver_bundle(path)` loads a hetpack bundle directory
(`driver_params.npy [n_items, k_max, 11]` + `driver_class.npy`,
`global_positions.npy`, `vehicle_params.json`) or a bare 3-D `.npy`/`.npz`.
`DriverBundle.item_sidecar(i)` returns the slot-aligned `[k_max, 11]`
`DriverSidecar`, and `item_index_of(position)` maps a pack manifest position.
`load_driver_sidecar(path)` now auto-detects a bundle dir / 3-D `.npy` and the
dense-truth dir (`DenseDriverTruth`). `driver_recovery` now also reports
**rank correlation per parameter and a per-anchor-class MAE breakdown**, in
addition to MAE/RMSE/bias and uncertainty calibration.

### 6c. TRAIN-scoped validation holdout — `wms/val_split.py`, `--val-holdout-frac`

Gap (a) is closed without weakening the TRAIN-only guard: `--val-holdout-frac F`
(requires `--window-list`) deterministically holds out ~`F` of the TRAIN **worlds**
from training and derives a TRAIN-scoped validation list that `--val-window-list`
(monitoring) can actually consume (the list keeps the TRAIN pack name, so
`lpo.filter_positions` matches instead of failing closed). `s2_chain.sh` now runs
with `VAL_HOLDOUT_FRAC=0.1` (override or `=0` to disable). The separate VAL/TEST
packs remain offline-evaluation only.

### 6d. Offline VAL/TEST recovery scorer — `wms_eval materialise` / `wms_eval driver`

`python -m wms_eval materialise --pack <pack_hetdata_{val,test}> --out states_*.json`
writes one state record per manifest item (snapshot tick + pre-t0 observed
history) from the memmapped pack arrays (no ids in the pack, so slots are named
`slot<k>`). `python -m wms_eval driver --states ... --ckpt ... --sidecar <bundle|dense> --out ...`
runs the trained `driver_params` encoder on each record and writes
`DRIVER_RECOVERY.json` (per-parameter MAE/RMSE/bias, rank correlation, per-class,
uncertainty calibration). `wms_plugin._load` now restores the checkpoint's
`rollout_mode` so a `driver_params` checkpoint is scored as such. Script
`wms/score_hetdata_recovery.sh` runs both splits end-to-end (CPU).

## 6e. Remaining blockers / limitations (honest)

1. **gpuA execution not performed here:** sections 6a--6d are CPU-integrated and
   unit-tested on synthetic fixtures only; the real-box post-processing, smoke
   gates, training and offline scoring in section 8 have NOT been run (no GPU,
   no remote access, no SUMO in this task).
2. **Dense slot map is a reimplementation** of the featurizer's pool algorithm
   (unit-tested to equal `parse_run` on a synthetic fixture); the box `verify`
   step re-hashes the emitted arrays, but an equality check against the actual
   pack `slot_index` should be run on gpuA before trusting exact slot alignment.
3. **VAL/TEST recovery needs a trained checkpoint** and is scored at each
   manifest item's snapshot tick; a busy-tick scoring sweep over the dense truth
   is possible but is a further offline pass, not part of this task.
4. **Per-driver identification ground truth at t0 is still thin** for the bundle
   path (1--3 HDVs/world); the dense sidecar is the route to dense C2 scoring.

## 7. Ordered box commands (gpuA, owner-authorized only)

Run in this exact order; each step is CPU unless noted. `$W` = work dir, `$C` =
extracted code tree, `$PY` = the rd_gpu python.

```bash
export W=/root/autodl-tmp/wms_s2_20261006/work
export HP3=/root/autodl-tmp
export PY=/root/autodl-tmp/gpu1/env/rd_gpu/bin/python
export C=$W/code/output/wd01/wm_n1

# 1) DENSE-TRUTH POST-PROCESSING (CPU; -> /root/autodl-tmp/dense_truth_20261007)
cd "$C" && RUNS=$HP3/hetdata_runs OUT=$HP3/dense_truth_20261007 WORKERS=4 \
  bash wms/build_dense_truth.sh

# 2) SMOKE GATES + TRAINING (GPU; s2_chain runs stage 1a multi-stage smoke,
#    stage 1b 400-step smoke, then the shared 10/30/60/120/300 s curriculum;
#    VAL_HOLDOUT_FRAC=0.1 holds out ~10% of TRAIN worlds for monitoring)
export SIDECAR=$HP3/hetpack3_TRAIN/driver_params_hetdata_train   # eval-only truth
bash "$C/wms/s2_chain.sh"

# 3) OFFLINE VAL/TEST RECOVERY SCORING (CPU; needs step 2's checkpoint)
CKPT=$W/full/wm_s.pt SIDECAR=$HP3/dense_truth_20261007 OUT=$W/recovery \
  bash "$C/wms/score_hetdata_recovery.sh"
```

Per-split equivalents of step 3 (`materialise` then `driver`) are documented in
`wms/score_hetdata_recovery.sh`. `SIDECAR` may be either the dense-truth
directory or the corresponding `driver_params_hetdata_{val,test}` bundle.
