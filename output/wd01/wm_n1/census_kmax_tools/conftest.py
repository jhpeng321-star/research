"""Test bootstrap for the K_max census tools (mirrors features/tests)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]        # output/wd01/wm_n1
_TOOLS = Path(__file__).resolve().parent
_FEAT_TESTS = _ROOT / "features" / "tests"
for _p in (_ROOT, _TOOLS, _FEAT_TESTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if __import__("os").environ.get("WD_FORBID_D", "").strip() == "1":
    _WD01 = str(_ROOT.parent)                      # output/wd01
    if _WD01 not in sys.path:
        sys.path.insert(0, _WD01)
    import forbid_d_hook
    forbid_d_hook.install()
