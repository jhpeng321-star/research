"""Export a fitted policy as a stageable JSON artifact + verify runtime parity.

Why JSON rather than the executor's pickle: remote staging copies only .py
text files, so the live-test box needs a text artifact plus runtime.py. The
artifact carries the full-data per-option GBTs (tree arrays), the grouped
5-fold calibration radius, and the feature order. artifact.verify() asserts
runtime predictions match sklearn to 1e-9 on the training design matrix.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .cv import CVResult
from .schema import OPTIONS, POLICY_HORIZON, checked, sha
from .runtime import PolicyRuntime

VERSION = 1
PARITY_TOL = 1e-9


def _tree_spec(tree):
    return dict(feature=[int(v) for v in tree.feature],
                threshold=[float(v) for v in tree.threshold],
                left=[int(v) for v in tree.children_left],
                right=[int(v) for v in tree.children_right],
                value=[float(v[0][0]) for v in tree.value.reshape(tree.value.shape[0], 1, 1)])


def _model_spec(model):
    constant = np.asarray(model.init_.constant_).ravel()
    return dict(init=float(constant[0]), learning_rate=float(model.learning_rate),
                trees=[_tree_spec(est.tree_) for est in model.estimators_.ravel()])


def build(result: CVResult, radius=None, source=None):
    models = {}
    for option in OPTIONS:
        model = result.models_full.get(option)
        if model is not None:
            models[option] = _model_spec(model)
    return dict(version=VERSION, horizon_s=int(result.horizon), radius=float(radius if radius is not None
                                                                              else result.radius_full),
                feature_names=list(result.feature_names), options=list(OPTIONS), models=models,
                created_utc=datetime.now(timezone.utc).isoformat(),
                source_sha256=str(source) if source else None,
                rule="argmax predicted D among available options; execute only if "
                     "predicted D - radius > 0; else O0 (MAPPO). Radius is the finite-sample "
                     "95% quantile of per-world max signed overprediction, world-grouped 5-fold CV.")


def verify(artifact, X, result: CVResult):
    """Runtime-vs-sklearn parity on every option model; raises on mismatch."""
    runtime = PolicyRuntime(artifact)
    X = np.asarray(X, dtype=float)
    worst = 0.0
    for option, model in result.models_full.items():
        if model is None:
            continue
        expected = model.predict(X)
        got = np.array([runtime.predict(dict(zip(runtime.feature_names, row)))[option] for row in X])
        worst = max(worst, float(np.abs(expected - got).max()))
    if worst > PARITY_TOL:
        raise ValueError("runtime parity violated: max abs diff " + repr(worst))
    return worst


def write(artifact, out_path):
    checked(out_path)
    Path(out_path).write_text(json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8")
    return sha(out_path)
