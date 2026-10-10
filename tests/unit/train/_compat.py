"""Test-time access to the LEGACY WM-S code for the train-layer unit/equivalence tests.

Integration state (WP-I): the sibling packages (``wms.core``, ``wms.model``, ``wms.data``)
now exist for real, so nothing is shimmed any more; the old ``install()`` stand-ins are gone.
What remains is (1) a way to import the legacy modules and (2) ``FlatConfig``, a minimal
stand-in for ``TrainConfig`` honouring the ``to_legacy_flat``/``fingerprint`` contract so
that trainer-level tests can build configs straight from the legacy ``WmSTrainConfig``.

The legacy ``wms`` package was moved to ``archive/wms_era/wm_n1/wms`` (WP-F) and is reachable
through the shim package ``output.wd01.wm_n1.wms`` (dotted path from the repo root). Its
sibling packages (``hdv``, ``features``, ...) live in ``output/wd01/wm_n1``; that directory is
appended (never prepended) to ``sys.path`` so it cannot shadow ``src/wms``.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
WM_N1 = ROOT / "output" / "wd01" / "wm_n1"
LEGACY = "output.wd01.wm_n1.wms"


def install() -> None:
    """Make ``src`` (new package) and the legacy tree importable, new package first."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    importlib.import_module("wms")          # claim the name before the legacy path appears
    if str(WM_N1) not in sys.path:
        sys.path.append(str(WM_N1))


def legacy(name: str):
    """Import ``<legacy wms>.<name>`` (e.g. ``legacy("train")``)."""
    install()
    return importlib.import_module(f"{LEGACY}.{name}")


class FlatConfig:
    """Stand-in honouring the ``TrainConfig`` compatibility contract used by the trainer."""

    def __init__(self, flat: dict) -> None:
        self._flat = json.loads(json.dumps(flat))

    @classmethod
    def from_legacy(cls, legacy_cfg) -> "FlatConfig":
        return cls(legacy_cfg.to_dict())

    @classmethod
    def from_legacy_flat(cls, flat: dict) -> "FlatConfig":
        return cls(flat)

    def to_legacy_flat(self) -> dict:
        return json.loads(json.dumps(self._flat))

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self._flat, sort_keys=True).encode()).hexdigest()
