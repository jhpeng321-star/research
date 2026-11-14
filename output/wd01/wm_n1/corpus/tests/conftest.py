"""Test bootstrap for the CORPUS suite (mirrors features/tests/conftest.py)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_WM_N1 = Path(__file__).resolve().parents[2]        # output/wd01/wm_n1
if str(_WM_N1) not in sys.path:
    sys.path.insert(0, str(_WM_N1))
_WD01 = _WM_N1.parent                               # output/wd01
if str(_WD01) not in sys.path:
    sys.path.insert(0, str(_WD01))
_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

if os.environ.get("WD_FORBID_D", "").strip() == "1":
    import forbid_d_hook

    forbid_d_hook.install()
