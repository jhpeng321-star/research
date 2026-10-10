"""Load the FROZEN reference stack from the research-dense tree
(wdpaths.RESEARCH_DENSE_ROOT, default E:/research-dense), read-only.

The workspace ships its own traffic_shepherd copy; the emulators must be
equivalent to the research-dense code that produced chunk 1, so this
shim binds the reference packages explicitly (no sys.path shadowing) and
disables bytecode writing so the read-only tree is never modified.

Import order contract: call load_reference() BEFORE any test imports
traffic_shepherd / output.controller_verification_*.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import wdpaths  # noqa: E402

RESEARCH_DENSE = Path(wdpaths.RESEARCH_DENSE_ROOT)
MODEL_ONNX = RESEARCH_DENSE / "baselines" / "circles" / "model.onnx"
CHUNK1_RUNS = RESEARCH_DENSE / "runs" / "campaign_stage1_chunk1"
CHUNK1_OUT = RESEARCH_DENSE / "output" / "campaign_stage1_chunk1"
WORLD_PLAN = CHUNK1_OUT / "results_plan_wm44" / "WORLD_PLAN.json"

_loaded = False


def load_reference() -> None:
    """Bind traffic_shepherd (research-dense) and the output.* namespace."""
    global _loaded
    if _loaded:
        return
    sys.dont_write_bytecode = True

    ts_spec = importlib.util.spec_from_file_location(
        "traffic_shepherd", RESEARCH_DENSE / "traffic_shepherd" / "__init__.py",
        submodule_search_locations=[str(RESEARCH_DENSE / "traffic_shepherd")])
    ts_pkg = importlib.util.module_from_spec(ts_spec)
    sys.modules["traffic_shepherd"] = ts_pkg
    ts_spec.loader.exec_module(ts_pkg)

    # output.* is a PEP-420 namespace on the reference tree only
    ns_spec = importlib.machinery.ModuleSpec("output", None, is_package=True)
    ns = importlib.util.module_from_spec(ns_spec)
    ns.__path__ = [str(RESEARCH_DENSE / "output")]
    sys.modules["output"] = ns

    for module in ("traffic_shepherd.config", "traffic_shepherd.sensing",
                   "traffic_shepherd.damping", "traffic_shepherd.wall_group",
                   "traffic_shepherd.ownership",
                   "output.controller_verification_v2.pi_saturation_v1",
                   "output.controller_verification_v4.pi_external_u",
                   "output.controller_verification_v7.pi_owner_adapter",
                   "output.controller_verification_v4.pi_external_u"):
        importlib.import_module(module)
    _bind_worktree_wall_group()
    _loaded = True


def _bind_worktree_wall_group() -> None:
    """wall-v2.1 closure law (owner decision 2026-09-30, Option B): the wall
    command law under test is THIS worktree's traffic_shepherd/wall_group.py
    (self-contained module); the rest of the frozen reference stack is
    unchanged.  The emulator parity tests compare against it."""
    src = Path(__file__).resolve().parents[3] / "traffic_shepherd" / "wall_group.py"
    if not src.is_file():
        return
    spec = importlib.util.spec_from_file_location("traffic_shepherd.wall_group", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["traffic_shepherd.wall_group"] = mod
    spec.loader.exec_module(mod)
    setattr(sys.modules["traffic_shepherd"], "wall_group", mod)


def reference():
    """(module,) bundle of the reference implementations after binding."""
    load_reference()
    from traffic_shepherd import config, damping, sensing, wall_group, ownership
    from output.controller_verification_v2 import pi_saturation_v1
    from output.controller_verification_v4 import pi_external_u
    from output.controller_verification_v7 import pi_owner_adapter
    return dict(config=config, damping=damping, sensing=sensing,
                wall_group=wall_group, ownership=ownership,
                pi_saturation_v1=pi_saturation_v1,
                pi_external_u=pi_external_u,
                pi_owner_adapter=pi_owner_adapter)
