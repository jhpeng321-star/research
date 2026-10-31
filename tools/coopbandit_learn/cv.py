"""World-grouped cross-validation and signed-residual uncertainty calibration.

Frozen procedure (COOPBANDIT_DESIGN.md + executor learn.py):
- five world-grouped outer folds (sklearn GroupKFold on world_id, deterministic);
- inner 4-fold world-grouped signed residuals calibrate the confidence radius;
- radius = finite-sample 95% quantile of the per-world MAX overprediction
  (predicted - true D at the policy horizon) across decision times/options;
- outer-test labels never touch their own fold's radius or model;
- the full-data model is calibrated by grouped 5-fold out-of-fold residuals.

Difference from executor learn.py, documented: models are fitted per option ONLY
on states where the option is available, and calibration residuals pool
available options only. learn.py fits all options on all states (unavailable
options are measured exact no-ops with D=0); the availability mask avoids
letting no-op rows influence the fitted response surfaces. The choice rule,
radius definition, folds and hyperparameters are identical.
"""
from dataclasses import dataclass
import math
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .loader import Dataset
from .models import fit_option_models, predict_option_models
from .schema import CONF_LEVEL, FOLDS, INNER_FOLDS, OPTIONS, POLICY_HORIZON


def conformal_quantile(values, level=CONF_LEVEL):
    """Finite-sample (n+1)-order-statistic quantile, clamped to the sample."""
    s = sorted(float(v) for v in values)
    if not s:
        raise ValueError("empty calibration sample")
    k = min(len(s), max(1, math.ceil((len(s) + 1) * level)))
    return s[k - 1]


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
    seconds: float = 0.0

    def radius_for_world(self, world):
        return self.fold_radius[int(self.fold_of[int(world)])]

    def dev_guard(self):
        bad = [int(w) for w in self.fold_of if 981080 <= int(w) <= 981091]
        if bad:
            raise ValueError("live dev world in fit/calibration: " + repr(bad))


def _fold_of(groups, k=FOLDS):
    """Deterministic world->fold map mirroring GroupKFold's assignment."""
    assigner = GroupKFold(n_splits=k)
    order = np.argsort(groups, kind="stable")
    folded = np.zeros(len(groups), dtype=int)
    for fold, (_, test) in enumerate(assigner.split(np.zeros(len(groups)), groups=groups[order])):
        folded[order[test]] = fold
    return {int(g): int(f) for g, f in zip(groups, folded)}


def _design(dataset: Dataset, state_ids, horizon):
    X = dataset.feature_matrix(list(state_ids))
    availability = dataset.availability_map(state_ids)
    targets = {o: np.array([dataset.labels_for(s, horizon).get(o, 0.0) for s in state_ids])
               for o in OPTIONS}
    return X, targets, availability


def _residual_rows(dataset, state_ids, horizon, pred, fold_tag, split_tag):
    rows = []
    avail = dataset.availability_map(state_ids)
    for i, sid in enumerate(state_ids):
        world = int(dataset.world_of([sid])[0])
        for option in OPTIONS:
            if not avail[option][i]:
                continue
            y = dataset.labels_for(sid, horizon).get(option)
            if y is None:
                continue
            rows.append(dict(state_id=sid, world=world, option=option, y_true=y,
                             y_pred=float(pred[option][i]), residual=float(pred[option][i]) - y,
                             fold=fold_tag, split=split_tag))
    return rows


def _fit_radius(dataset, train_states, horizon, kind, fold_tag):
    """Inner grouped CV residuals -> per-world max overprediction -> radius."""
    groups = dataset.world_of(train_states).astype(int)
    inner_assign = _fold_of(np.unique(groups), INNER_FOLDS)
    inner = np.array([inner_assign[g] for g in groups])
    residuals, inner_groups = [], {}
    for j in sorted(set(inner)):
        tr = [s for s, m in zip(train_states, inner != j) if m]
        va = [s for s, m in zip(train_states, inner == j) if m]
        if not tr or not va:
            continue
        Xtr, ytr, atr = _design(dataset, tr, horizon)
        models = fit_option_models(kind, Xtr, ytr, atr)
        Xva, _, _ = _design(dataset, va, horizon)
        residuals += _residual_rows(dataset, va, horizon, predict_option_models(models, Xva),
                                    fold_tag, f"inner{j}")
        inner_groups[j] = sorted(set(dataset.world_of(va).tolist()))
    if not residuals:
        raise ValueError("no inner residuals; dataset too small for calibration")
    per_world = pd.DataFrame(residuals).groupby("world").residual.max()
    return conformal_quantile(per_world.values), inner_groups


def run(dataset: Dataset, horizon=POLICY_HORIZON, kind="gbt", folds=FOLDS):
    """Full grouped-CV sweep: OOF predictions, per-fold radii, full-data model."""
    started = time.time()
    state_ids = list(dataset.states.state_id)
    groups = dataset.world_of(state_ids).astype(int)
    fold_of = _fold_of(np.unique(groups), folds)
    fold_idx = np.array([fold_of[g] for g in groups])
    oof_rows, fold_radius = [], {}
    for f in sorted(set(fold_idx)):
        tr_states = [s for s, m in zip(state_ids, fold_idx != f) if m]
        te_states = [s for s, m in zip(state_ids, fold_idx == f) if m]
        radius_f, _ = _fit_radius(dataset, tr_states, horizon, kind, f)
        fold_radius[f] = radius_f
        Xtr, ytr, atr = _design(dataset, tr_states, horizon)
        models = fit_option_models(kind, Xtr, ytr, atr)
        Xte, _, _ = _design(dataset, te_states, horizon)
        oof_rows += _residual_rows(dataset, te_states, horizon, predict_option_models(models, Xte), f, "outer")
    oof = pd.DataFrame(oof_rows)
    per_world = oof.groupby("world").residual.max()
    radius_full = conformal_quantile(per_world.values)
    X, targets, availability = _design(dataset, state_ids, horizon)
    models_full = fit_option_models(kind, X, targets, availability)
    result = CVResult(kind=kind, horizon=horizon, oof=oof, fold_of=fold_of, fold_radius=fold_radius,
                      radius_full=radius_full, models_full=models_full,
                      feature_names=list(dataset.feature_names), seconds=time.time() - started)
    result.dev_guard()
    return result
