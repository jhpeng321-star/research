"""Drive the clean ``wms.eval`` with the LEGACY WM-S network (pre-integration adapter).

Until the clean ``wms.model`` / ``wms.data`` packages land, the evaluation backend is built
from the historical ``output/wd01/wm_n1/wms`` package, loaded under the alias
``wms_legacy`` (its top-level name collides with the clean ``wms``).  The legacy package
must therefore be loaded AFTER the clean ``wms`` namespace has been imported.

This module is test support only; it disappears once WP-B/WP-C are merged.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Mapping

REPO = Path(__file__).resolve().parents[2]
WM_N1 = REPO / "output" / "wd01" / "wm_n1"
LEGACY_WMS = REPO / "archive" / "wms_era" / "wm_n1" / "wms"
LEGACY_ALIAS = "wms_legacy"


def ensure_src_on_path() -> None:
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    importlib.import_module("wms")          # claim the name before the legacy path appears


def load_legacy() -> types.ModuleType:
    """The legacy ``wms`` package under the alias ``wms_legacy`` (idempotent)."""
    ensure_src_on_path()
    if LEGACY_ALIAS in sys.modules:
        return sys.modules[LEGACY_ALIAS]
    init = LEGACY_WMS / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        LEGACY_ALIAS, init, submodule_search_locations=[str(init.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[LEGACY_ALIAS] = module
    spec.loader.exec_module(module)
    return module


def _submodule(name: str, **members) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(members)
    sys.modules[name] = module
    return module


def _clean_module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def install_clean_aliases() -> None:
    """Register ``wms.data.*`` / ``wms.model.entries`` names backed by the legacy modules."""
    ensure_src_on_path()
    if "wms.data.history" in sys.modules or _clean_module_exists("wms.data.history"):
        return
    load_legacy()
    history = importlib.import_module(f"{LEGACY_ALIAS}.history")
    model = importlib.import_module(f"{LEGACY_ALIAS}.model")
    presence = importlib.import_module(f"{LEGACY_ALIAS}.presence")
    timeline = importlib.import_module(f"{LEGACY_ALIAS}.action_timeline")
    entries = importlib.import_module(f"{LEGACY_ALIAS}.entries")
    wms = importlib.import_module("wms")
    data = _submodule("wms.data")
    data.__path__ = []
    wms.data = data
    data.history = _submodule(
        "wms.data.history", history_spec=history.history_spec,
        long_history_features=history.long_history_features,
        history_step_features=model.history_step_features)
    data.presence = _submodule(
        "wms.data.presence", junction_carry_presence=presence.junction_carry_presence)
    data.action_timeline = _submodule(
        "wms.data.action_timeline", ActionTimeline=timeline.ActionTimeline,
        CONTROLLERS=timeline.CONTROLLERS)
    model_pkg = sys.modules.get("wms.model")
    if model_pkg is None:
        model_pkg = _submodule("wms.model")
        model_pkg.__path__ = []
        wms.model = model_pkg
    model_pkg.entries = entries
    sys.modules["wms.model.entries"] = entries


def legacy_backend(checkpoint: Path, physics: Mapping | None = None):
    """A ``WmsBackend`` for a structured ``wm_s/1`` checkpoint, mirroring the legacy loader."""
    import torch

    from wms.eval.backend import WmsBackend, default_tools

    install_clean_aliases()
    model_mod = importlib.import_module(f"{LEGACY_ALIAS}.model")
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    config = dict(payload.get("config") or {})
    raw = dict(physics or config.get("physics") or {})
    allowed = set(model_mod.PhysicsConfig.__dataclass_fields__)
    cfg = model_mod.PhysicsConfig(**{k: v for k, v in raw.items() if k in allowed})
    model = model_mod.WmSModel(
        physics=cfg, history_features=int(config.get("history_features", 6)),
        rollout_mode=str(config.get("rollout_mode", "residual")),
        lane_head=str(config.get("lane_head", "off")),
        residual_cap_mps2=config.get("residual_cap_mps2"),
        residual_cap_neg_mps2=config.get("residual_cap_neg_mps2"),
        residual_soft_bound_mps2=config.get("residual_soft_bound_mps2"),
        residual_soft_bound_neg_mps2=config.get("residual_soft_bound_neg_mps2"),
        lane_logit_cap=config.get("lane_logit_cap"),
        history_s=config.get("history_s"),
        history_stride_s=config.get("history_stride_s", 1.0))
    projection = None
    if config.get("projection_enabled"):
        stability = importlib.import_module(f"{LEGACY_ALIAS}.stability")
        projection = stability.FeasibilityProjection(stability.ProjectionConfig(
            min_gap_m=float(config.get("projection_min_gap_m", 2.0))))
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return WmsBackend(model=model, config=config, tools=default_tools(), projection=projection)


def receipt_physics(receipt: Path) -> dict:
    return json.loads(Path(receipt).read_text(encoding="utf-8"))["physics"]
