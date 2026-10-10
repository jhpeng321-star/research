"""The v0.4b fit pipeline (registration s.5/s.6): pairs -> direct-dY
quantile-GBM per (tau x variant) -> TRAIN-OOB acceptance + strata.

TRAIN-only: this module never opens a CAL row (cal_guard.
assert_train_only on every row frame it consumes).  The ONE CAL read
lives in cal_read_v04b.py and is double-locked (OWNER_ACK + read-once
marker).

Artifacts (per variant, under the run's results dir):
  model_<variant>.json      chosen grid candidates + conformal shifts +
                            selection records (the fitted pickles stay
                            in the run dir for the guarded CAL read)
  train_oob_acceptance_<variant>.json   A1/A3 (labelled TRAIN-OOB) +
                            s.12.1 stratified tables + aux numbers
  representation_counts.json  the s.12.1 item 4 a-priori counts
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401
from . import reg
from . import acceptance_v04b as AC
from . import descriptors_v04b as DE
from . import pairs_v04b as PB
from .cal_guard import assert_train_only
from .model_v04b import DirectDiffGBM


def log_fn(log):
    def f(msg):
        log(msg)
    return f


def tau_pairs(pairs: pd.DataFrame, tau: float,
              head: str = reg.GATING_HEAD) -> pd.DataFrame:
    return PB.complete_pairs(pairs, tau, head)


def model_pred_fn(model: DirectDiffGBM, rows: pd.DataFrame,
                  variant: str, tau: float,
                  head: str = reg.GATING_HEAD):
    """fn(pairs_rows) -> {q: [n]} calibrated quantile predictions."""
    def f(g: pd.DataFrame) -> dict:
        X = DE.pair_matrix(g, rows, variant)
        return model.predict_quantiles(X, tau, head)
    return f


def oob_pred_fn(model: DirectDiffGBM, tau: float,
                keys: list) -> "object":
    """fn(pairs_rows) -> {q: [n]} world-grouped OOB predictions, looked
    up by (world, arm_source) so contrast subsets stay row-aligned with
    the fit rows of that tau."""
    tau = float(tau)
    oof = model.oof[(tau, reg.GATING_HEAD)]
    pos = {k: i for i, k in enumerate(keys)}

    def f(g: pd.DataFrame) -> dict:
        idx = np.array([pos[(str(w), str(a))] for w, a in
                        zip(g["world"], g["arm_source"])])
        return {q: np.asarray(v, float)[idx] for q, v in oof.items()}
    return f


def fit_variant(pairs: pd.DataFrame, rows: pd.DataFrame, variant: str,
                 log=print, ablation: bool = True, grid=None,
                 cv_folds: int | None = None) -> dict:
    """Fit + TRAIN-OOB acceptance for one sensing variant.  grid/cv_folds
    default to the registered constants (tests may shrink them; the
    shipped fit never does)."""
    assert_train_only(pairs["split"], f"fit_variant({variant})")
    train = pairs[pairs["split"] == "TRAIN"]
    ref = AC.train_references_v04b(train)
    taus = reg.TAUS
    models: dict = {}
    keys: dict = {}
    for tau in taus:
        g = tau_pairs(train, tau)
        if not len(g):
            continue
        X = DE.pair_matrix(g, rows, variant)
        y = {h: g[f"dy_{h}_{int(tau)}"].to_numpy(float) for h in reg.HEADS}
        groups = g["world"].to_numpy()
        headline = (g["contrast"] == reg.HEADLINE_CONTRAST).to_numpy()
        kw = {} if grid is None else dict(grid=grid)
        if cv_folds is not None:
            kw["cv_folds"] = cv_folds
        m = DirectDiffGBM(taus=(tau,), **kw)
        m.fit(X, y, tau, groups, headline, log=log)
        if ablation and int(headline.sum()) >= 4:
            m_abl = DirectDiffGBM(taus=(tau,), **kw)
            m_abl.fit(X[headline], {h: v[headline] for h, v in y.items()},
                      tau, groups[headline],
                      np.ones(int(headline.sum()), bool), log=log)
            models[("ablation", float(tau))] = m_abl
        models[(float(tau))] = m
        keys[float(tau)] = list(zip(g["world"].astype(str),
                                    g["arm_source"].astype(str)))
    # ---- TRAIN-OOB acceptance (labelled in-sample-selection-free) ----
    oob_a3: list = []
    strata: list = []

    def oob_fn(g):
        tau = float(g["tau"].iloc[0])
        return oob_pred_fn(models[tau], tau, keys[tau])(g)

    oob_a1 = AC.a1_records(train, oob_fn)
    for tau in reg.TAUS:
        recs = AC.a3_records(train, tau, oob_fn, ref)
        oob_a3.extend(recs)
        strata.extend(AC.stratified_records(train, tau, oob_fn, ref))
    aux_records = [r for r in oob_a3
                   if r["contrast"] != reg.HEADLINE_CONTRAST]
    a3_head = AC.a3_verdict(oob_a3)
    return dict(
        variant=variant, models=models, references=ref,
        train_oob=dict(a1=oob_a1, a3_records=oob_a3, a3=a3_head,
                       aux_records=aux_records, strata=strata,
                       label="TRAIN-OOB (in-sample-selection-free)"))


def save_artifacts(art: dict, rows: pd.DataFrame, out_dir: Path,
                   log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = art["variant"]
    side = dict(
        schema=reg.SCHEMA, kind="v04b_fit_sidecar", variant=variant,
        registration_lf_sha256=reg.reg_sha(),
        quantiles=list(reg.QUANTILES), taus=list(reg.TAUS),
        chosen={f"{t:.0f}": list(m.chosen[float(t)])
                for t, m in ((k, v) for k, v in art["models"].items()
                             if not isinstance(k, tuple))},
        selection=[rec for k, m in art["models"].items()
                   if not isinstance(k, tuple) for rec in m.selection],
        feature_names=DE.feature_names(variant),
        train_oob=art["train_oob"],
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    p = out_dir / f"fit_sidecar_{variant}.json"
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(side, indent=1, default=_json_default),
                   encoding="utf-8")
    tmp.replace(p)
    with open(out_dir / f"model_{variant}.pkl", "wb") as fh:
        pickle.dump({str(k): v for k, v in art["models"].items()}, fh,
                    protocol=5)
    log(f"v04b[save]: {p.name} + model_{variant}.pkl")
    return dict(sidecar=str(p), model=str(out_dir / f"model_{variant}.pkl"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)
