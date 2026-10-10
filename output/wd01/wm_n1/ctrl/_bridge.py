"""Read-only import bridge to the verified ``partA_ctrl`` mirrors.

P1 never re-implements a controller: the damping / PI / wall-v2.1 mirrors
are imported from ``output/wd01/partA_ctrl/emulators.py`` exactly as they
were validated (E1-E5) and validated against the logged TRAIN commands
(E4).  ``partA_ctrl`` is a sys.path package (no ``__init__.py``), so this
module pins the two directories it needs onto ``sys.path`` (idempotent)
and imports the mirror module once.  Nothing here writes to disk and no
``partA_ctrl`` file is modified: READ-ONLY contract (INTERFACES_V2.md).

The CIRCLES ONNX payload is resolved from THIS worktree
(``baselines/circles/model.onnx``); its sha256 is pinned by
:data:`ctrl.config.CIRCLES_MODEL_SHA256` and re-checked by
``DampingEmulator.__init__``.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_WD01 = Path(__file__).resolve().parents[2]          # output/wd01
_ROOT = _WD01.parents[1]                             # workspace root
_MODEL_ONNX = _ROOT / "baselines" / "circles" / "model.onnx"

_patched = False


def _ensure_path() -> None:
    """Put output/wd01 and output/wd01/partA_ctrl on sys.path (once)."""
    global _patched
    if _patched:
        return
    for entry in (str(_WD01), str(_WD01 / "partA_ctrl")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    _patched = True


def emulators():
    """The verified ``partA_ctrl.emulators`` module (imported read-only)."""
    _ensure_path()
    return importlib.import_module("partA_ctrl.emulators")


def model_onnx() -> Path:
    """Path of the sha-pinned CIRCLES ONNX payload in this worktree."""
    if not _MODEL_ONNX.is_file():
        raise FileNotFoundError(
            f"CIRCLES model payload missing at {_MODEL_ONNX}; the P1 mover "
            "refuses to approximate it (controllers are exact, never learned)")
    return _MODEL_ONNX
