"""INC1 test bootstrap: put ``output/wd01`` and the repo root on sys.path."""
from __future__ import annotations

import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_INC1 = _TESTS.parent
for _root in (_INC1.parent.parent, _INC1.parent):  # repo root, output/wd01
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
