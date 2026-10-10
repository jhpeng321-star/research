# WM-N1 `train/` — TRAIN piece

Implements the TRAIN part of the frozen contract
[`../INTERFACES.md`](../INTERFACES.md) ("New stub APIs" TRAIN part,
"Config keys", "File ownership"): the packed-corpus dataset with sibling-pair
sampling and the outcome-target mapping, the single-member `Trainer` with
checkpoint/resume, and the training CLI.  CPU-only; no SUMO, no real-data
training, no HOLDOUT/CAL/chunk-5 access.

## Modules

| module | what it does |
| --- | --- |
| `config.TrainConfig` | design §3/§7 defaults; `load`/`save`/`to_dict`; `model_config()`, `seed_for_member()`, `loss_weights()` |
| `dataset.MemmapCorpus` | map-style dataset over a packed memmap; `__len__`, `__getitem__`, `sibling_pairs`, `worlds`; TRAIN-only guard |
| `dataset` helpers | `map_outcome_targets`, `contrast_batch_indices`, `collate`, `is_reference` |
| `trainer.Trainer` | one member; `fit`, `save_checkpoint`, `resume`, `evaluate_cal` |
| `cli` | `python -m train --member k --seed … --config … --corpus … --out …`; `record_seed` |

## Semantics

* **Item.** One t0 snapshot per run: `veh_step [K,16]`, `veh_mask [K]`,
  `plan [K,19,8]`, `wall [4,11]`, `wall_mask [4]`, `counter [2,30]`, the step
  targets `accel/target_mask/lane/arrivals`, and the outcome rows
  `outcome_quantiles [N_TAU,N_OUTCOME_TARGETS]`, `outcome_mask [N_TAU]`,
  `collision [N_TAU]`.
* **Sibling pairs.** `sibling_pairs(i)` returns `(a, b)` = (treatment,
  all-native reference) of the same `pair_key = "<world>@t0=<t0>"`, so the
  comonotone contrast `q_a − q_b` is `V(P) − V(all-native)`
  (`model.losses.contrast_loss`).  `pair_sampling=True` orders pair groups
  contiguously; `contrast_batch_indices` finds the pairs inside a collated
  batch.
* **Outcome mapping (INTERFACES Mismatch 3).**
  `OUTCOME_COLUMN_MAP` maps the v0.4a shard columns
  `total_delay_ref_s → delay_s`, `ttc_lt3_veh_s → ttc_lt3_s`,
  `hard_brake_s → hard_brake_s`; `collision_any → collision`,
  `tau_available → outcome_mask`; missing/NaN values become 0.
* **Split guard.** `split` defaults to `"TRAIN"`; only `TRAIN`/`CALIBRATION`
  (CAL via the explicit `split="CALIBRATION"`/`"CAL"` argument) are opened.
  A pack containing any `HOLDOUT`/`TEST`/`chunk5` item is refused at open time
  (`CorpusRefusal`).
* **Checkpoint.** `member_<k>.pt` holds plain `state_dict` + optimizer + epoch
  + step + member + seed + `config` (JSON) + the batch generator state; a
  `member_<k>.json` sidecar carries the config.  Every `torch.load` passes
  `weights_only=True` explicitly (supported torch >=2.7 defaults to `True`).

## Usage

```
python -m train --member 1 --seed 101 --config cfg.json \
    --corpus PACK --out RUN_DIR --device cpu
```

Member-1-first order is `config.MEMBER_ORDER = (1,2,3,4,5)`
(`REGISTRATION_WM_N1_AMENDMENT_1.md` A1); the seed is written to
`<out>/SEEDS.json` before the first step.

## Tests (CPU, synthetic packed shards)

```
WD_FORBID_D=1 python -m pytest output/wd01/wm_n1/train/tests -q
```

`tests/packsynth.py` builds the minimal pack; it is the byte-format fixture
for the not-yet-implemented CORPUS writer.

## Interface deviations from the stub

* **D1 — config key.** The stub dataclass field `caliber` is implemented as
  `calibrate_on_cal`, the name in the frozen `INTERFACES.md` Config-keys
  table (`calibrate_on_cal=False`).  Reason: the INTERFACES key is the
  contract; `caliber` was a stub typo.
* **D2 — additive config helpers.** `from_dict`, `model_config`,
  `seed_for_member`, `loss_weights` were added; `load`/`save`/`to_dict`
  unchanged.
* **D3 — packed layout is train-owned.** The CORPUS `pack_shards` stub defines
  no byte format, so `dataset.py` fixes the read-side layout (documented in
  its module docstring) and `tests/packsynth.py` builds a matching fixture.
  The real CORPUS writer must emit this layout.
* **D4 — additive dataset helpers.** `pair_keys` and the module-level
  `map_outcome_targets`/`contrast_batch_indices`/`collate`/`is_reference` were
  added; the frozen `MemmapCorpus` signatures are unchanged.
* **D5 — Trainer internals.** `_iter_batches`, `_targets`, `_forward_loss`,
  `_train_step`, `_write_receipt` are private; public signatures unchanged.
  `evaluate_cal` opens CAL only through the explicit CAL guard and returns
  `{"status": "unavailable"}` when no CAL pack is present; it never changes
  hyperparameters.
