"""Load predictors by ``module:attr`` import path.

The model package (e1) is built in parallel, so the harness names a predictor
on the command line, never importing a concrete model at module import time:

    --predictor  mypkg.mymodel:WMN1Predictor
    --reference  mypkg.hybrid:HybridPredictor

``attr`` may be a dotted path.  If the resolved object is a class it is
instantiated with no arguments; if it is a zero-argument callable it is called
(so a factory works too); otherwise the object itself must be a ``Predictor``.
"""
from __future__ import annotations

import importlib

from .protocol import Predictor


class PredictorLoadError(ValueError):
    """A predictor import path could not be resolved to a Predictor."""


def _resolve(mod_name: str, attr: str):
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def load_object(spec: str):
    """Resolve ``module:attr`` (dotted attr) and call classes/factories.

    Classes are instantiated with no arguments and zero-argument callables
    (factories without a ``predict_quantiles`` attribute) are called.
    """
    if ":" not in spec:
        raise PredictorLoadError(f"spec {spec!r} must be 'module:attr'")
    mod_name, attr = spec.split(":", 1)
    try:
        obj = _resolve(mod_name, attr)
    except (ImportError, AttributeError) as e:
        raise PredictorLoadError(f"cannot load {spec!r}: {e}") from e
    if isinstance(obj, type):
        obj = obj()
    elif callable(obj) and not hasattr(obj, "predict_quantiles"):
        obj = obj()
    return obj


def load_predictor(spec: str) -> Predictor:
    obj = load_object(spec)
    if not (hasattr(obj, "predict_quantiles")
            and hasattr(obj, "sample_rollouts")):
        raise PredictorLoadError(
            f"{spec!r} resolved to {obj!r}, which has no predict_quantiles / "
            "sample_rollouts")
    return obj
