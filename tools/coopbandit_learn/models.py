"""Per-option D(option|observable state) models with frozen hyperparameters.

Design (COOPBANDIT_DESIGN.md): sklearn GradientBoostingRegressor, 100 trees,
depth 2, min_samples_leaf 8, Huber loss, random_state 0; a standardized ridge
linear baseline; one model per group option, fitted only on states where the
option is available.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .schema import OPTIONS


def make_gbt():
    return GradientBoostingRegressor(n_estimators=100, max_depth=2, min_samples_leaf=8,
                                     loss="huber", random_state=0)


def make_linear():
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])


MODELS = {"gbt": make_gbt, "linear": make_linear}


def fit_option_models(kind, X, targets, availability):
    """Fit one model per option; options with no available rows map to None.

    targets: {option: array of D} for every state row (D at the model horizon);
    availability: {option: bool array} marking where the option may be chosen.
    """
    models = {}
    for option in OPTIONS:
        mask = np.asarray(availability[option], dtype=bool)
        if not mask.any():
            models[option] = None
            continue
        est = MODELS[kind]()
        est.fit(np.asarray(X, dtype=float)[mask], np.asarray(targets[option], dtype=float)[mask])
        models[option] = est
    return models


def predict_option_models(models, X):
    """Predict D per option; unavailable/unfitted options yield -inf."""
    X = np.asarray(X, dtype=float)
    out = {}
    for option, est in models.items():
        out[option] = est.predict(X) if est is not None else np.full(len(X), -np.inf)
    return out
