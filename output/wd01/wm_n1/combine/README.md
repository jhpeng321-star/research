# WM-N1 combination layer (`wm_n1/combine`)

Preparation-only tooling for the owner add-on in
[`../WM_N1_DESIGN.md`](../WM_N1_DESIGN.md) section 5 and
[`../REGISTRATION_WM_N1.md`](../REGISTRATION_WM_N1.md) sections 2-3.

It implements the **C** (combined) variant of WM-N1: world-grouped
out-of-fold tree predictions, CAL-only stacking of the neural and tree
predictive CDFs, conformal recalibration, the neural/tree disagreement
signal, and the planner-variant rule.  Everything runs on CPU with
numpy/pandas/scipy: **no SUMO, no model training, no real dataset read**.

## Modules and API

| module | entry points |
| --- | --- |
| `crps.py` | `QuantilePred` (piecewise-linear CDF), `crps_pinball`, `build_cdf_grid`, `crps_cdf`, `pooled_crps_series`, `mixture_quantiles`, `cdf_distance` |
| `guards.py` | `guard_train_only`, `guard_cal_only`, `SplitLeakError` |
| `oof_tree.py` | `build_oof_predictions`, `fit_guarded`, `partition_worlds`, `RowTableDataset`, `row_table_builder`, `OofResult.save` / `load_oof` |
| `stacking.py` | `fit_pool_weight`, `cross_fit_pool`, `conformal_shifts`, `fit_conformal_pool`, `fit_stack`, `StackModel`, `interval_coverage` |
| `disagreement.py` | `disagreement`, `tau_scale`, `spearman_abs_error`, `abstain_curve` |
| `select_variant.py` | `select_planner_variant`, `write_planner_variant`, `verify_planner_variant` |
| `synthetic.py` | `synthetic_rows`, `SyntheticQuantileModel`, `gaussian_pred` |

`build_oof_predictions(rows, model_factory, ds_builder=None, *, k=5, ...)`
takes a row table (`world`, `split`, `tau`, `target`, `policy`, a stable
`row_id`) and a factory returning a `partB_v0.gbm.GbmValueModel`-compatible
object (`fit(ds)`, `predict_quantiles_rows(ds, idx, target, ...)`).  It
returns an `OofResult` whose table holds every row per target with
`fold_id` (the held-out fold, or `-1` for the full-TRAIN model) and
`model_kind` (`"oof"` / `"full"`), plus the `q<level>` quantile columns.
`save()` writes a parquet row table and an npz with the row keys, fold ids,
quantile matrix and fold bookkeeping.

`fit_stack(fn, ft, y, worlds, splits, *, levels, folds=2)` returns a
`StackModel` (pool weight + conformal calibration) and the cross-fit
record.  `disagreement(fn, ft, scale=...)` returns `D_tau` per row.

### CRPS representation

- `crps_pinball` is the pinball-average approximation used by Part B
  (`partB_v0.gbm.crps_from_quantiles`), asserted equal in
  `tests/test_interface.py`.
- `crps_cdf` is `Integral (F(x) - 1{x >= y})^2 dx` by the trapezoid rule on
  a dense common grid (`build_cdf_grid`).  The pool
  `F_C = w*F_N + (1-w)*F_T` is exact in this representation, so the weight
  search and the mixture quantiles both go through the CDF form.
  `mixture_quantiles` inverts the pooled CDF.

## Leakage rules (enforced)

- **OOF tree (TRAIN).**  Every TRAIN row gets a prediction from a model
  fitted on K world-grouped folds; the model that predicted a row was
  fitted on `all TRAIN worlds - that row's world`.  `fit_guarded` raises
  `SplitLeakError` if any non-TRAIN row reaches a fit.
- **Full-TRAIN (CAL/TEST).**  Non-TRAIN rows get the single model fitted
  on all TRAIN worlds; fold models never predict them.
- **CAL-only weighting/calibration.**  Pool weights (world-grouped 2-fold
  cross-fitting) and the per-level conformal shifts are fitted on
  CALIBRATION rows only; `guard_cal_only` raises on any non-CALIBRATION
  row, and names TEST/HOLDOUT explicitly.
- **Sealed evaluation.**  HOLDOUT and chunk-5 TEST are sealed: the layer
  never opens their files and the planner variant is frozen in
  `PLANNER_VARIANT.json` (with sha256) before any TEST read.  The tests
  build synthetic CAL/TEST labels only.
- **Read-only reuse.**  `partB_v0/` is imported read-only (interface test
  only); nothing here edits it.

## Planner variant

`select_planner_variant(crps, coverage)` picks the lowest CAL mean dV CRPS
over the gating contrasts at tau 60/120/300, subject to 10-90 coverage in
[0.75, 0.85]; a tie within 2 % goes to the simpler model (T < C < N).
`write_planner_variant` writes the sidecar with a self-describing sha256.

## CLI (synthetic only)

```
python -m combine version
python -m combine synthetic --out <dir> [--seed 0] [--n-train 8] [--n-cal 4] [--k 5]
```

Run from this directory's parent (`output/wd01/wm_n1`) so `combine` is
importable.  The synthetic pipeline exercises OOF, stacking, the
disagreement signal and the variant rule, and writes
`oof_tree.parquet/.npz`, `abstain_curve.csv`, `PLANNER_VARIANT.json` and
`summary.json` under `--out`.

## Tests

```
python -m pytest output/wd01/wm_n1/combine/tests
```

with `WD_FORBID_D=1` set (the `output/wd01/conftest.py` forbid-D hook).
