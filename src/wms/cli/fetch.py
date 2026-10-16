"""``wms fetch``: thin wrapper over ``tools/fetch_data.py`` (list | status | verify | get).

Arguments after ``fetch`` are forwarded verbatim, so the tool's own global options
(``--registry``, ``--cache-dir``, ``--budget-bytes``) work unchanged.
"""
from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Sequence

from ..config.presets import repo_root
from ..core.errors import WmsError


def load_fetch_tool() -> ModuleType:
    """Import ``tools/fetch_data.py`` by path (``tools`` is not an installed package)."""
    path = repo_root() / "tools" / "fetch_data.py"
    if not path.is_file():
        raise WmsError(f"data fetch tool not found at {path}")
    spec = importlib.util.spec_from_file_location("wms_fetch_data_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(forwarded: Sequence[str]) -> int:
    """Run ``tools/fetch_data.py`` with ``forwarded`` arguments; return its exit status."""
    return int(load_fetch_tool().main(list(forwarded)))
