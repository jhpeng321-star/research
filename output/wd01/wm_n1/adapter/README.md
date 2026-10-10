# WM-N1 adapter (`output/wd01/wm_n1/adapter`)

Piece **ADAPTER** (INTERFACES.md).  Bridges a trained WM-N1 ensemble to the
two frozen consumers:

* **`evalh`** via the `evalh.protocol.Predictor` protocol
  (`adapter/predictor.py:WmN1Predictor`), and
* **`combine`** via a neural CAL/OOF row table
  (`adapter/export.py:export_cal` / `export_oof`).

No real data, no training, no SUMO.  Everything is CPU-testable with a
randomly initialised tiny ensemble.

## Modules

| module | contents |
| --- | --- |
| `predictor.py` | `WmN1Predictor` (`PredictorBase` subclass) + window/corpus glue |
| `export.py` | `export_cal`, `export_oof`, `ExportGuardError`, split/seal guards |
| `checkpoint.py` | plain-`state_dict` + JSON config ensemble I/O |
| `quantiles.py` | 19 -> 7 quantile-curve interpolation (INTERFACES Mismatch 1) |

`WmN1Predictor` implements the frozen stub signatures:

```python
WmN1Predictor(checkpoint=None, corpus=None, *, device="cuda")
WmN1Predictor.from_checkpoint(checkpoint, corpus=None, *, device)
predict_quantiles(rows) -> np.ndarray [N, 19]
sample_rollouts(rows, horizons_s=None) -> evalh.protocol.RolloutBatch
```

`predict_quantiles` runs the ensemble mixture at each row's own `tau_s`
(nearest `model.constants.TAUS`) and returns the 19 delay quantiles
(`OUTCOME_TARGETS[0] == "delay_s"`).  `sample_rollouts` runs
`model.rollout.rollout` per member, averages the members, and collapses
`[n_samples, n_steps+1, B, K]` to `RolloutBatch.pos/speed [N, H, V]`
(INTERFACES Mismatch 4), with `V` the number of occupied slots in the window's
`veh_mask`.

A file `adapter/predictor.py` also exposes the helper
`predict_quantiles_members(rows, member_idx)` used by the OOF export; it does
not change the protocol surface (the protocol `predict_quantiles(rows)` is
unchanged).

### Corpora and rows

`corpus` resolves a row's six frozen window tensors at its `(world, t0_s)`.
Accepted forms: an object with `window_tensors` / `get_window` / `window`, a
`{world: window}` or `{(world, t0_s): window}` mapping, or a callable
`corpus(world, t0_s)`.  A row may instead carry the window directly in
`row.features["tensors"]`.  An optional `row.features["actions"]` dict/list
overrides the per-CAV action one-hot (`_apply_actions`).  Windows are the
unbatched frozen tensors (`veh_step [K,16]`, `veh_mask [K]`, `plan [K,19,8]`,
`wall [4,11]`, `wall_mask [4]`, `counter [2,30]`); a leading batch axis of
size 1 is stripped.

## Quantile mapping (Mismatch 1)

`quantiles.map_quantiles(src_levels, qvals, dst_levels)` linearly interpolates
the quantile function `Q(p)` per row (`numpy.interp`) and accumulates for
monotonicity.  It is **exact at shared levels** (the 7
`combine.stacking.DEFAULT_QUANTILES` all lie on the 19 neural levels), linear
between them, and clamped outside the source range.

## Checkpoint format

The loader is coded to the `train` INTERFACES format and always calls
`torch.load(..., weights_only=True)` explicitly (supported Torch `>=2.7` now
defaults to `True`, so the explicit value keeps every version identical).
Two layouts are accepted:

* **directory**: `member_<k>.pt` (1-based) plus the config either in
  `config.json` or, as `train.Trainer` writes it, in the `member_<k>.json`
  sidecar / member payload (flat `TrainConfig`).  A `train.Trainer` member file
  nests the weights under `"state_dict"` (with optimizer/epoch/seed for
  resume); a plain `state_dict` is also accepted; and
* **file**: `{"config": <json str>, "members": [state_dict, ...]}`.

The config holds `arch` (`WmN1Config` fields, or flat for a `TrainConfig`),
`seeds`, `n_members`, and the optional OOF bookkeeping `train_worlds` /
`member_train_worlds` (`{member_index_1based: [world, ...]}`).  Write with
`checkpoint.save_ensemble` / `save_ensemble_file`.  The directory loader then
round-trips the exact `members/` tree the BUNDLE queue produces
(`train/tests/test_cross_piece.py`).

## Export guards

* `export_cal` requires rows that **explicitly declare the CALIBRATION
  split** (`combine.guards.guard_cal_only`); TRAIN, TEST and HOLDOUT are
  refused and TEST/HOLDOUT are named explicitly.
* `export_oof` requires **TRAIN rows only** (`guard_train_only`).
* both refuse **chunk-5** world names (`ExportGuardError`).
* **member-level OOF limit.**  The design (WM_N1_DESIGN.md section 3) trains
  every member on the same TRAIN corpus, so no member has a world held out
  unless the checkpoint records `member_train_worlds`.  When that bookkeeping
  is present, each world fold is scored only by members that did not train on
  any of the fold's worlds (`model_kind="oof"`).  When it is absent, the rows
  are tagged `model_kind="full-in-sample"` (NOT OOF) and the result carries
  `member_oof_available=False` plus `limit_note`; the world-grouped OOF source
  remains `combine.oof_tree`.

`export_cal` / `export_oof` both return the row-table + npz paths written in
the exact `combine.oof_tree.OofResult` shape, and additionally return
`levels` and `qvals` so callers can build a `combine.crps.QuantilePred`
directly.

## Tests

```
$env:WD_FORBID_D = "1"
python -m pytest output/wd01/wm_n1/adapter/tests -q
```

Covers: `Predictor` protocol conformance, rollout-shape collapse, 19->7
mapping (exact at shared levels, monotone, interpolated between), CAL export
accepted by `combine.stacking.fit_stack`, OOF member-eligibility, split/seal
refusals, and checkpoint roundtrip.  The suite pins `torch.set_num_threads(1)`
(tiny CPU nets run ~15x faster single-threaded) and finishes in well under a
minute.

### Deviations from the stub

* `__init__.py` additionally re-exports `checkpoint` and `quantiles`.
* `WmN1Predictor` adds the non-protocol helper `predict_quantiles_members`.
* `predictor.py` accepts a pre-built `model.ensemble.WmN1Ensemble` as
  `checkpoint` (in addition to a path), for tests.
