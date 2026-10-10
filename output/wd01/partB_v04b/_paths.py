"""sys.path shims for partB_v04b (import partB_v0 / partB_v0data READ-ONLY).

The sibling packages keep their flat module layout (import v0lib, import
actions_v04 ...), so both package roots and output/wd01 (for wdpaths)
must be on sys.path.  Inserted in a fixed order; nothing is modified on
disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent

for _p in (WD01, HERE, WD01 / "partB_v0data", WD01 / "partB_v0"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
