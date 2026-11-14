"""Black-box WM variants for the C1 leave-one-controller-out comparison.

C1 contrasts the *structured* WM-S (controller run as code, only HDV motion
learned) against *black-box* world models that take the controller as an input
feature.  The two black-box variants are authored on their own branches
(``wms/bb`` -> ``wms.blackbox_id``, ``wms/bb2`` -> ``wms.blackbox_traj``) and
are merged after this harness.  This module is the single, clean hook point.

A black-box module must expose one of the names in ``_PREFERRED_NAMES``:

* a factory ``build(checkpoint_path, flags=None)`` returning an object with
  ``predict_states(state, cav_controllers, horizon_s, seed)``; or
* a module-level object (class or instance) with that method.

The returned object is wrapped in :class:`wms_eval.models.PluginModel`, so the
harness transparently supplies the rollout flags and accepts both a
``predict_states`` method and a plain callable.  When the branch is not yet
merged the loader fails closed with an actionable message instead of silently
falling back to the structured model.
"""

from __future__ import annotations

import importlib
from typing import Any

from .models import ModelUnavailable, PluginModel, RolloutFlags

#: model spec -> (module path, preferred attribute names, branch name)
BLACKBOX_SPECS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "blackbox_id": ("wms.blackbox_id", ("build", "MODEL", "model"), "wms/bb"),
    "blackbox_traj": ("wms.blackbox_traj", ("build", "MODEL", "model"), "wms/bb2"),
}


def load_blackbox(spec: str, checkpoint: str, flags: RolloutFlags | None = None) -> PluginModel:
    """Import a black-box variant and wrap it for the harness rollout contract."""
    if spec not in BLACKBOX_SPECS:
        raise ModelUnavailable(f"unknown black-box spec {spec!r}")
    module_name, names, branch = BLACKBOX_SPECS[spec]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ModelUnavailable(
            f"black-box module {module_name!r} is not present; merge branch {branch} "
            f"into wms/c1eval (loader namespace reserved by wms_eval.blackbox)"
        ) from exc
    obj: Any = None
    for name in names:
        obj = getattr(module, name, None)
        if obj is not None:
            break
    if obj is None:
        raise ModelUnavailable(
            f"black-box module {module_name!r} exposes none of {names}"
        )
    if isinstance(obj, type) or (callable(obj) and not hasattr(obj, "predict_states")):
        try:
            obj = obj(checkpoint, flags)
        except TypeError:
            obj = obj(checkpoint)
    return PluginModel(obj, flags or RolloutFlags())
