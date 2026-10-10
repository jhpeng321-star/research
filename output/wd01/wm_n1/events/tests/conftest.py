"""Test bootstrap for the ``events`` piece (mirrors model/tests/conftest.py)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

if os.environ.get("WD_FORBID_D", "").strip() == "1":
    _WD01 = str(_ROOT.parent)
    if _WD01 not in sys.path:
        sys.path.insert(0, _WD01)
    import forbid_d_hook

    forbid_d_hook.install()
