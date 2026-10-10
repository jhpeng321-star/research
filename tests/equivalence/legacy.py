"""Load the legacy WM-S code under the alias ``wms_legacy``.

Since WP-F the legacy ``wms`` package lives in ``archive/wms_era/wm_n1/wms``; its sibling
packages (``hdv``, ``features``, ...) stay in ``output/wd01/wm_n1`` and must be importable
while the legacy modules load.  The legacy tree is also called ``wms``; it is loaded through
importlib under another name and ``wm_n1`` is removed from ``sys.path`` again, so it never
shadows ``src/wms``.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WM_N1 = REPO / "output" / "wd01" / "wm_n1"                       # hdv/, features/, ... (siblings)
LEGACY_WMS = REPO / "archive" / "wms_era" / "wm_n1" / "wms"        # the moved legacy package
C1_CHECKPOINT = Path("E:/research-dense/runs/active/WM-S-C1FAST-20261006/box/full_struct/wm_s.pt")


def load_legacy():
    """Return the legacy package module (``wms_legacy``)."""
    if "wms_legacy" in sys.modules:
        return sys.modules["wms_legacy"]
    importlib.import_module("wms")          # make sure the new package is cached first
    root = str(WM_N1)
    added = root not in sys.path
    if added:
        sys.path.append(root)
    try:
        spec = importlib.util.spec_from_file_location(
            "wms_legacy", LEGACY_WMS / "__init__.py",
            submodule_search_locations=[str(LEGACY_WMS)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["wms_legacy"] = module
        spec.loader.exec_module(module)
        for name in ("model", "stability", "entries", "physics_torch", "wall_exec",
                     "lane_head", "driver_encoder", "history", "long_horizon", "brakefix",
                     "blackbox_traj"):
            importlib.import_module(f"wms_legacy.{name}")
    finally:
        if added:
            sys.path.remove(root)
    return module


def legacy(name: str):
    load_legacy()
    return importlib.import_module(f"wms_legacy.{name}")
