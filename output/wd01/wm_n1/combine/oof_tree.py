"""World-grouped out-of-fold tree predictions (WM-N1 design section 5b).

The residual ablation feeds the tree Part B prediction to the neural head.
To keep that feature leakage-free:

  * every TRAIN row receives an OUT-OF-FOLD tree prediction from a model
    fitted on K world-grouped folds -- the model that predicted a row never
    saw any row of that row's world;
  * every non-TRAIN row (CAL / TEST) receives the prediction of the single
    model fitted on ALL TRAIN worlds;
  * a fit never sees a non-TRAIN row (``fit_guarded``).

The row table is a pandas frame with, per row, a stable key, the world id,
the split label, and the tau / target / policy conditioning.  A
``ds_builder(world_ids)`` materialises the dataset a ``GbmValueModel``-
compatible model can ``fit`` and ``predict_quantiles_rows`` on; the
default builder wraps the row table itself (``RowTableDataset``), so the
planner-side tree can be represented by any object exposing
``.rows``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .guards import TRAIN, guard_train_only

DEFAULT_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
FULL_FOLD = -1


class RowTableDataset:
    """Minimal dataset a row-table tree model can fit/predict on."""

    def __init__(self, rows: pd.DataFrame):
        self.rows = rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)


def row_table_builder(rows: pd.DataFrame, world_col: str = "world"):
    """Return ``build(world_ids) -> RowTableDataset`` over a row table."""
    worlds = rows[world_col].astype(str)

    def build(world_ids) -> RowTableDataset:
        keep = {str(w) for w in world_ids}
        return RowTableDataset(rows[worlds.isin(keep)])

    return build


def partition_worlds(worlds, k: int = 5):
    """Deterministic world-grouped fold assignment.

    Returns ``(assign, k_eff)``: ``assign[world] = fold`` for every world,
    folds partition the worlds, and ``k_eff = min(k, n_worlds)`` so a tiny
    world set still gets one fold per world.
    """
    uw = sorted({str(w) for w in worlds})
    if not uw:
        raise ValueError("no worlds to partition")
    k_eff = max(1, min(int(k), len(uw)))
    assign = {w: i % k_eff for i, w in enumerate(uw)}
    return assign, k_eff


def fit_guarded(model_factory, ds, context: str = "oof_tree"):
    """Fit a fresh model on ``ds``; raises if any row is not TRAIN."""
    guard_train_only(ds.rows["split"].to_numpy(), context)
    model = model_factory()
    model.fit(ds)
    return model


def _quantile_columns(quantiles) -> list[str]:
    return [f"q{float(q):g}" for q in quantiles]


@dataclass
class OofResult:
    table: pd.DataFrame
    fold_worlds: dict = field(default_factory=dict)
    fit_worlds: dict = field(default_factory=dict)
    full_fit_worlds: list = field(default_factory=list)
    k: int = 0
    quantiles: tuple = DEFAULT_QUANTILES

    def save(self, prefix) -> dict:
        """Write ``<prefix>.parquet`` and ``<prefix>.npz`` (row keys,
        fold ids, quantile matrix, meta) and return both paths."""
        base = str(prefix)
        pq = f"{base}.parquet"
        npz = f"{base}.npz"
        Path(base).parent.mkdir(parents=True, exist_ok=True)
        self.table.to_parquet(pq, index=False)
        qcols = _quantile_columns(self.quantiles)
        meta = dict(k=int(self.k),
                    fold_worlds={str(a): list(b)
                                 for a, b in self.fold_worlds.items()},
                    fit_worlds={str(a): list(b)
                                for a, b in self.fit_worlds.items()},
                    full_fit_worlds=list(self.full_fit_worlds),
                    quantiles=[float(q) for q in self.quantiles])
        np.savez_compressed(
            npz,
            row_id=self.table["row_id"].to_numpy().astype(str),
            fold_id=self.table["fold_id"].to_numpy().astype(np.int32),
            qvals=self.table[qcols].to_numpy(dtype=np.float64),
            meta=np.array(json.dumps(meta)))
        return dict(parquet=pq, npz=npz)


def build_oof_predictions(
    rows: pd.DataFrame,
    model_factory,
    ds_builder=None,
    *,
    row_key: str = "row_id",
    world_col: str = "world",
    split_col: str = "split",
    tau_col: str = "tau",
    target_col: str = "target",
    policy_col: str = "policy",
    k: int = 5,
    targets=None,
    quantiles=DEFAULT_QUANTILES,
    calibrated: bool = True,
    context: str = "oof_tree",
) -> OofResult:
    """World-grouped K-fold OOF predictions for TRAIN + full-TRAIN for the
    rest.

    ``model_factory()`` must return a ``partB_v0.gbm.GbmValueModel``-
    compatible object exposing ``fit(ds)`` and
    ``predict_quantiles_rows(ds, row_idx, target, calibrated=...)``.
    """
    rows = rows.reset_index(drop=True).copy()
    if row_key not in rows.columns:
        rows[row_key] = np.arange(len(rows))
    if ds_builder is None:
        ds_builder = row_table_builder(rows, world_col)

    splits = rows[split_col].astype(str).to_numpy()
    worlds = rows[world_col].astype(str).to_numpy()
    train_mask = splits == TRAIN
    train_worlds = sorted(set(worlds[train_mask]))
    if not train_worlds:
        raise ValueError("the row table has no TRAIN rows")
    assign, k_eff = partition_worlds(train_worlds, k)

    if targets is None:
        targets = [str(t) for t in
                   dict.fromkeys(rows[target_col].astype(str).tolist())]
    else:
        targets = [str(t) for t in targets]
    quantiles = tuple(sorted(float(q) for q in quantiles))
    qcols = _quantile_columns(quantiles)

    keys = rows[row_key].astype(str).to_numpy()
    key_to_pos = {key: i for i, key in enumerate(keys)}
    n = len(rows)
    pred = {t: np.full((n, len(quantiles)), np.nan) for t in targets}
    fold_of = {t: np.full(n, FULL_FOLD, dtype=np.int64) for t in targets}
    fold_worlds: dict = {}
    fit_worlds: dict = {}

    def _place(tgt, ds_pred, idx, model, fold_id):
        if len(idx) == 0:
            return
        sub_keys = ds_pred.rows[row_key].astype(str).to_numpy()
        gp = np.array([key_to_pos[str(kk)] for kk in sub_keys[idx]])
        got = model.predict_quantiles_rows(ds_pred, idx, tgt,
                                           calibrated=calibrated)
        for j, q in enumerate(quantiles):
            if float(q) in got:
                pred[tgt][gp, j] = np.asarray(got[float(q)],
                                              dtype=np.float64)
        fold_of[tgt][gp] = fold_id

    for f in range(k_eff):
        test_w = [w for w, ff in assign.items() if ff == f]
        fit_w = [w for w in train_worlds if w not in set(test_w)]
        fold_worlds[f] = test_w
        fit_worlds[f] = fit_w
        model = fit_guarded(model_factory, ds_builder(fit_w),
                            f"{context}:fold{f}")
        ds_te = ds_builder(test_w)
        for tgt in targets:
            idx = np.flatnonzero(
                ds_te.rows[target_col].astype(str).to_numpy() == tgt)
            _place(tgt, ds_te, idx, model, f)

    full_fit_worlds: list = []
    non_train = ~train_mask
    if non_train.any():
        non_worlds = sorted(set(worlds[non_train]))
        ds_all = ds_builder(train_worlds)
        model_full = fit_guarded(model_factory, ds_all, f"{context}:full")
        full_fit_worlds = list(train_worlds)
        ds_nt = ds_builder(non_worlds)
        for tgt in targets:
            idx = np.flatnonzero(
                ds_nt.rows[target_col].astype(str).to_numpy() == tgt)
            _place(tgt, ds_nt, idx, model_full, FULL_FOLD)

    out = []
    for tgt in targets:
        for i in range(n):
            rec = {row_key: keys[i], world_col: worlds[i],
                   split_col: splits[i], tau_col: rows[tau_col].iloc[i],
                   target_col: tgt, policy_col: rows[policy_col].iloc[i],
                   "fold_id": int(fold_of[tgt][i]),
                   "model_kind": "oof" if train_mask[i] else "full"}
            for j, col in enumerate(qcols):
                rec[col] = pred[tgt][i, j]
            out.append(rec)
    table = pd.DataFrame(out)
    return OofResult(table=table, fold_worlds=fold_worlds,
                     fit_worlds=fit_worlds, full_fit_worlds=full_fit_worlds,
                     k=k_eff, quantiles=quantiles)


def load_oof(prefix) -> OofResult:
    """Load an ``OofResult`` written by :meth:`OofResult.save`."""
    with np.load(f"{prefix}.npz", allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    table = pd.read_parquet(f"{prefix}.parquet")
    return OofResult(table=table, fold_worlds=meta["fold_worlds"],
                     fit_worlds=meta["fit_worlds"],
                     full_fit_worlds=meta["full_fit_worlds"], k=meta["k"],
                     quantiles=tuple(meta["quantiles"]))
