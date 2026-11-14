"""wd01-wide pytest conftest.

With WD_FORBID_D=1 in the environment this installs the forbid-D audit
hook (output/wd01/forbid_d_hook.py) BEFORE any test module is imported,
so every wd01 suite run (each <part>/tests dir) fails loudly on any
attempt to open, list, create, rename or delete a drive-D path.  Unset
(default) this file does nothing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if os.environ.get("WD_FORBID_D", "").strip() == "1":
    _WD01 = str(Path(__file__).resolve().parent)
    if _WD01 not in sys.path:
        sys.path.insert(0, _WD01)
    import forbid_d_hook

    forbid_d_hook.install()
