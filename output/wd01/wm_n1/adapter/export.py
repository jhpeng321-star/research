"""Neural OOF/CAL export for combine/stacking.

`combine.oof_tree.build_oof_predictions` expects a row table with a stable
``row_id`` and ``q<level>`` quantile columns, and ``combine.stacking`` pools at
``combine.stacking.DEFAULT_QUANTILES`` = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9,
0.95) (7 levels, Part B).  The neural head emits 19 levels
(``model.constants.QUANTILE_LEVELS``), so this module maps the neural quantile
curve onto the 7 pooling levels (see ``adapter.quantiles``) before the C
variant can pool.

Guards (owner constraints, 2026-09-30):

* **CAL export** requires rows that *explicitly* declare the CALIBRATION
  split; every other split is refused and TEST/HOLDOUT are named explicitly.
* **TRAIN (OOF) export** requires TRAIN rows only; TEST and HOLDOUT are
  refused.  A TRAIN row may only be scored for export by members that did not
  train on its world (see :func:`export_oof` and ``member_train_worlds``);
  when the checkpoint carries no world-holdout bookkeeping the export records
  the documented member-level OOF limit instead of silently treating
  in-sample predictions as out-of-fold.
* chunk-5 world names are refused by both entry points (the TEST chunk is
  sealed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from combine.guards import guard_cal_only, guard_train_only
from combine.oof_tree import FULL_FOLD, OofResult, partition_worlds
from combine.stacking import DEFAULT_QUANTILES
from evalh.protocol import OutcomeRow

from . import quantiles as Q
from .predictor import eligible_members

#: Documented member-level OOF limit (design section 3: all 5 members train
#: on the same TRAIN corpus, so no member has a world held out unless the
#: checkpoint records ``member_train_worlds``).
MEMBER_OOF_LIMIT_NOTE = (
    "The WM-N1 ensemble trains every member on the same TRAIN corpus "
    "(WM_N1_DESIGN.md section 3), so no member has any TRAIN world held out "
    "unless the checkpoint records per-member world holdouts "
    "('member_train_worlds').  Rows scored without that bookkeeping are "
    "labelled 'full-in-sample', NOT out-of-fold; the tree layer "
    "(combine.oof_tree) remains the world-grouped OOF source.")

CHUNK5_MARKERS = ("chunk5", "chunk-5")


class ExportGuardError(RuntimeError):
    """An export guard refused: a sealed split/world or an OOF violation."""


# ------------------------------------------------------------- row tables --
def _as_row_table(rows) -> pd.DataFrame:
    """Normalise DataFrame / evalh-row inputs to the export row table."""
    if isinstance(rows, pd.DataFrame):
        tab = rows.reset_index(drop=True).copy()
    else:
        recs = []
        for i, r in enumerate(rows):
            feats = getattr(r, "features", None) or {}
            recs.append(dict(
                row_id=feats.get("row_id", getattr(r, "row_id", f"r{i:06d}")),
                world=str(r.world),
                t0_s=float(getattr(r, "t0_s", 0.0)),
                decision_t_s=float(getattr(r, "decision_t_s", 0.0)),
                tau=float(r.tau_s),
                policy=str(r.policy),
                split=feats.get("split", getattr(r, "split", None)),
                target=feats.get("target", "delay"),
                y=float(getattr(r, "y_delay_s", float("nan")))))
        tab = pd.DataFrame(recs)

    if "tau" not in tab.columns and "tau_s" in tab.columns:
        tab["tau"] = tab["tau_s"]
    if "row_id" not in tab.columns:
        tab.insert(0, "row_id", [f"r{i:06d}" for i in range(len(tab))])
    if "t0_s" not in tab.columns:
        tab["t0_s"] = 0.0
    if "decision_t_s" not in tab.columns:
        tab["decision_t_s"] = 0.0
    if "target" not in tab.columns:
        tab["target"] = "delay"
    if "split" not in tab.columns:
        tab["split"] = None
    for col in ("row_id", "world", "split", "target", "policy"):
        tab[col] = tab[col].astype(str)
    tab["tau"] = tab["tau"].astype(float)
    tab["t0_s"] = tab["t0_s"].astype(float)
    tab["decision_t_s"] = tab["decision_t_s"].astype(float)
    return tab.reset_index(drop=True)


def _levels(levels) -> np.ndarray:
    return np.asarray(sorted(float(q) for q in
                             (DEFAULT_QUANTILES if levels is None else levels)),
                      dtype=np.float64)


def _assert_not_chunk5(worlds, context: str) -> None:
    bad = sorted({str(w) for w in worlds
                  if any(m in str(w).lower() for m in CHUNK5_MARKERS)})
    if bad:
        raise ExportGuardError(
            f"{context}: chunk-5 TEST worlds are sealed and must never be "
            f"exported: {bad}")


def _predict_quantiles(predictor, tab: pd.DataFrame,
                       member_idx=None) -> np.ndarray:
    rows = [OutcomeRow(world=str(w), t0_s=float(t0), decision_t_s=float(dec),
                       tau_s=float(tau), policy=str(pol), y_delay_s=float(y))
            for w, t0, dec, tau, pol, y in zip(
                tab["world"], tab["t0_s"], tab["decision_t_s"], tab["tau"],
                tab["policy"], tab.get("y", pd.Series([0.0] * len(tab))))]
    if member_idx is None:
        q = predictor.predict_quantiles(rows)
    else:
        q = predictor.predict_quantiles_members(rows, member_idx)
    return np.asarray(q, dtype=np.float64)


def _write_export(tab: pd.DataFrame, qvals: np.ndarray, levels: np.ndarray,
                  out_prefix, *, fold_id, model_kind, k: int,
                  fold_worlds=None, fit_worlds=None,
                  full_fit_worlds=None) -> dict:
    fold_id = np.asarray(fold_id)
    if fold_id.ndim == 0:
        fold_id = np.full(len(tab), int(fold_id), dtype=np.int64)
    model_kind = np.asarray(model_kind, dtype=object)
    if model_kind.ndim == 0:
        model_kind = np.full(len(tab), str(model_kind), dtype=object)
    out = pd.DataFrame({
        "row_id": tab["row_id"].astype(str),
        "world": tab["world"].astype(str),
        "split": tab["split"].astype(str),
        "tau": tab["tau"].astype(float),
        "target": tab["target"].astype(str),
        "policy": tab["policy"].astype(str),
        "fold_id": fold_id.astype(np.int32),
        "model_kind": model_kind.astype(str),
    })
    for j, lv in enumerate(levels):
        out[f"q{float(lv):g}"] = qvals[:, j]
    result = OofResult(
        table=out, fold_worlds=fold_worlds or {}, fit_worlds=fit_worlds or {},
        full_fit_worlds=list(full_fit_worlds or []), k=int(k),
        quantiles=tuple(float(lv) for lv in levels))
    saved = result.save(out_prefix)
    return dict(saved, n_rows=len(out), levels=[float(lv) for lv in levels],
                qvals=qvals, table=out)


# ------------------------------------------------------------------ CAL ----
def export_cal(predictor, rows, out_prefix, *, levels=None) -> dict:
    """Export CAL neural quantiles as a row table + npz for stacking.

    ``levels`` defaults to ``combine.stacking.DEFAULT_QUANTILES``.  Rows must
    explicitly carry the CALIBRATION split (``combine.guards.guard_cal_only``);
    TEST/HOLDOUT and chunk-5 worlds are refused.
    """
    tab = _as_row_table(rows)
    lv = _levels(levels)
    guard_cal_only(tab["split"].to_numpy(), "adapter.export_cal")
    _assert_not_chunk5(tab["world"], "adapter.export_cal")

    q19 = _predict_quantiles(predictor, tab)
    q = Q.map_quantiles(predictor.quantile_levels, q19, lv)
    result = _write_export(tab, q, lv, out_prefix, fold_id=FULL_FOLD,
                           model_kind="full", k=1)
    result.update(kind="CAL", member_oof_available=False,
                  predictor=getattr(predictor, "name", "wm_n1"))
    return result


# ------------------------------------------------------------------ OOF ----
def export_oof(predictor, rows, out_prefix, *, folds: int = 5,
               levels=None) -> dict:
    """Export TRAIN world-grouped OOF neural predictions.

    Returns the same row-table/npz shape as ``combine.oof_tree.OofResult``.
    Each world fold is scored with the members that did not train on any of
    the fold's worlds (``checkpoint`` ``member_train_worlds``); a member
    subset that is empty yields the documented
    :data:`MEMBER_OOF_LIMIT_NOTE` and in-sample rows tagged
    ``model_kind="full-in-sample"`` rather than being mislabelled OOF.
    """
    tab = _as_row_table(rows)
    lv = _levels(levels)
    guard_train_only(tab["split"].to_numpy(), "adapter.export_oof")
    _assert_not_chunk5(tab["world"], "adapter.export_oof")

    worlds = tab["world"].astype(str).to_numpy()
    uniq_worlds = sorted(set(worlds.tolist()))
    assign, k_eff = partition_worlds(uniq_worlds, folds)
    n_members = len(predictor._require_loaded().members)
    member_worlds = predictor.member_train_worlds
    member_oof = bool(member_worlds)

    q19 = np.full((len(tab), int(predictor.quantile_levels.size)), np.nan)
    fold_of = np.full(len(tab), FULL_FOLD, dtype=np.int64)
    kinds = np.array(["full-in-sample"] * len(tab), dtype=object)
    fold_worlds: dict = {}
    fit_worlds: dict = {}

    for f in range(k_eff):
        te = sorted(w for w, ff in assign.items() if ff == f)
        te_set = set(te)
        idx = np.array([i for i, w in enumerate(worlds) if w in te_set],
                       dtype=int)
        if idx.size == 0:
            continue
        sub = tab.iloc[idx].reset_index(drop=True)
        eligible = eligible_members(member_worlds, te, n_members)
        if eligible:
            q19[idx] = _predict_quantiles(predictor, sub, eligible)
            fold_of[idx] = f
            kinds[idx] = "oof"
            fold_worlds[f] = te
            fit_worlds[f] = [w for w in uniq_worlds if w not in te_set]
        else:
            q19[idx] = _predict_quantiles(predictor, sub)

    missing = np.flatnonzero(np.isnan(q19).any(axis=1))
    if missing.size:
        q19[missing] = _predict_quantiles(
            predictor, tab.iloc[missing].reset_index(drop=True))

    q = Q.map_quantiles(predictor.quantile_levels, q19, lv)
    result = _write_export(
        tab, q, lv, out_prefix, fold_id=fold_of, model_kind=kinds, k=k_eff,
        fold_worlds=fold_worlds, fit_worlds=fit_worlds,
        full_fit_worlds=(uniq_worlds if member_oof else []))
    result.update(kind="TRAIN", member_oof_available=member_oof,
                  folds=k_eff, fold_of=assign,
                  predictor=getattr(predictor, "name", "wm_n1"))
    if not member_oof:
        result["limit_note"] = MEMBER_OOF_LIMIT_NOTE
    return result
