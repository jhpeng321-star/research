"""Test bootstrap: put the wm_n1 root (for ``model`` and read-only
``features`` imports) and this tests dir on sys.path, and honour WD_FORBID_D.
Mirrors ``model/tests/conftest.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]        # output/wd01/wm_n1
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

if os.environ.get("WD_FORBID_D", "").strip() == "1":
    _WD01 = str(Path(__file__).resolve().parents[3])   # output/wd01
    if _WD01 not in sys.path:
        sys.path.insert(0, _WD01)
    import forbid_d_hook

    forbid_d_hook.install()
