"""World-grouped cross-validation and signed-residual uncertainty calibration.

Frozen procedure (COOPBANDIT_DESIGN.md):
- five world-grouped outer folds; a world never appears in its own training set;
- inner 4-fold world-grouped signed residuals calibrate the confidence radius;
- radius = finite-sample 95% quantile of the per-world MAX overprediction
  (predicted - true D at the policy horizon) across decision times/options;
- outer-test labels never touch their own fold's radius or model;
- the full-data model is calibrated by grouped 5-fold out-of-fold residuals.
The radius is the model's own CV uncertainty; no behavioural threshold enters.
"""
from dataclasses import dataclass, field
import math
import time

import numpy as np
import pandas as pd

from .loader import Dataset
from .models import fit_option_models, predict_option_models
from .schema import CONF_LEVEL, FOLDS, GROUP_OPTIONS, INNER_FOLDS, POLICY_HORIZON


@dataclass
class CVResult:
    kind: str
    horizon: int
    oof: pd.DataFrame
    fold_of: dict
    fold_radius: dict
    radius_full: float
    models_full: dict
    feature_names: list
    inner_folds: dict = field(default_factory=dict)
    seconds: float = 0.0

    def radius_for_world(self, world):
        return self.fold_radius[int(self.fold_of[str(world)])]


def conformal_quantile(values, level=CONF_LEVEL):
    """Finite-sample (n+1)-order-statistic quantile, clamped to the sample."""
    s = sorted(float(v) for v in values)
    if not s:
        raise ValueError("empty calibration sample")
    k = min(len(s), max(1, math.ceil((len(s) + 1) * level)))
    return s[k - 1]


def inner_world_folds(train_worlds, k=INNER_FOLDS):
    """Deterministic world-grouped inner split: sorted worlds dealt round-robin."""
    ordered = sorted(str(w) for w in train_worlds)
    return {w: i % k for i, w in enumerate(ordered)}


def _design(dataset: Dataset, state_ids, horizon):
    X = dataset.feature_matrix(list(state_ids))
    avail = dataset.states.set_index("state_id").loc[list(state_ids)]
    targets = {o: np.array([dataset.labels_for(s, horizon).get(o, 0.0) for s in state_ids])
               for o in GROUP_OPTIONS}
    availability = {o: avail[o].to_numpy(dtype=bool) for o in GROUP_OPTIONS}
    return X, targets, availability


def _residual_rows(dataset, state_ids, horizon, pred, fold_tag, split_tag):
    rows = []
    for i, sid in enumerate(state_ids):
        world = dataset.world_of([sid])[0]
        for option in dataset.available(sid):
            y = dataset.labels_for(sid, horizon).get(option)
            if y is None:
                continue
            rows.append(dict(state_id=sid, world=world, option=option, y_true=y,
                             y_pred=float(pred[option][i]), residual=float(pred[option][i]) - y,
                             fold=fold_tag, split=split_tag))
    return rows


def _fit_radius(dataset, train_worlds, horizon, kind, fold_tag):
    """Inner grouped CV residuals -> per-world max overprediction -> radius."""
    inner = inner_world_folds(train_worlds)
    residuals = []
    inner_groups = {}
    for j in range(INNER_FOLDS):
        val_worlds = {w for w in train_worlds if inner[str(w)] == j}
        tr_worlds = {w for w in train_worlds if inner[str(w)] != j}
        tr_states = [s for s in dataset.states.state_id if dataset.world_of([s])[0] in tr_worlds]
        va_states = [s for s in dataset.states.state_id if dataset.world_of([s])[0] in val_worlds]
        if not tr_states or not va_states:
            continue
        Xtr, ytr, atr = _design(dataset, tr_states, horizon)
        models = fit_option_models(kind, Xtr, ytr, atr)
        Xva, _, _ = _design(dataset, va_states, horizon)
        residuals += _residual_rows(dataset, va_states, horizon, predict_option_models(models, Xva),
                                    fold_tag, f"inner{j}")
        inner_groups[j] = sorted(val_worlds)
    if not residuals:
        raise ValueError("no inner residuals; dataset too small for calibration")
    per_world = pd.DataFrame(residuals).groupby("world").residual.max()
    return conformal_quantile(per_world.values), inner_groups


def run(dataset: Dataset, horizon=POLICY_HORIZON, kind="gbt", folds=FOLDS):
    """Full grouped-CV sweep: OOF predictions, per-fold radii, full-data model."""
    started = time.time()
    worlds_by_fold = {}
    for w, f in dataset.fold_of.items():
        worlds_by_fold.setdefault(int(f), set()).add(w)
    oof_rows, fold_radius = [], {}
    for f in sorted(worlds_by_fold):
        train_worlds = {w for g, ws in worlds_by_fold.items() if g != f for w in ws}
        test_worlds = worlds_by_fold[f]
        radius_f, inner_groups = _fit_radius(dataset, train_worlds, horizon, kind, f)
        fold_radius[f] = radius_f
        tr_states = [s for s in dataset.states.state_id if dataset.world_of([s])[0] in train_worlds]
        te_states = [s for s in dataset.states.state_id if dataset.world_of([s])[0] in test_worlds]
        Xtr, ytr, atr = _design(dataset, tr_states, horizon)
        models = fit_option_models(kind, Xtr, ytr, atr)
        Xte, _, _ = _design(dataset, te_states, horizon)
        oof_rows += _residual_rows(dataset, te_states, horizon, predict_option_models(models, Xte), f, "outer")
    oof = pd.DataFrame(oof_rows)
    per_world = oof.groupby("world").residual.max()
    radius_full = conformal_quantile(per_world.values)
    X, targets, availability = _design(dataset, list(dataset.states.state_id), horizon)
    models_full = fit_option_models(kind, X, targets, availability)
    return CVResult(kind=kind, horizon=horizon, oof=oof, fold_of=dict(dataset.fold_of),
                    fold_radius=fold_radius, radius_full=radius_full, models_full=models_full,
                    feature_names=list(dataset.feature_names), seconds=time.time() - started)
