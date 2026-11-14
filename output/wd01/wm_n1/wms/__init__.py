"""Legacy import shim (WP-F, 2026-10-10).

The WM-S-era `wms` package was moved unchanged to `archive/wms_era/wm_n1/wms/`
(see `archive/wms_era/PATH_MAP.json`). This thin shim keeps the historical import
paths (`wms`, `output.wd01.wm_n1.wms`) resolving for replay, legacy tests and
`lpwo` by redirecting the package search path to the archived sources and running the
archived `__init__` in this namespace. It contains no logic of its own.
Do not add code here; new work belongs in `src/wms`.
"""
from pathlib import Path as _Path

_ARCHIVED = _Path(__file__).resolve().parents[4] / "archive" / "wms_era" / "wm_n1" / "wms"
__path__ = [str(_ARCHIVED)]
_init = _ARCHIVED / "__init__.py"
exec(compile(_init.read_text(encoding="utf-8"), str(_init), "exec"))
