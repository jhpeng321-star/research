"""WM-N1-INC1 incident simulation package (BUILD ONLY; zero SUMO).

This package builds, but never runs, the WM-N1-INC1 zero-shot incident demo
registered in ``output/wd01/wm_n1/REGISTRATION_WM_N1_INC1.md`` (LF-sha
``d5282ecb...``).  It provides:

* ``seeds_audit``  -- the read-only fresh-seed audit (9601-9620);
* ``plan``         -- the deterministic 20-world WORLD_PLAN builder;
* ``obstacles``    -- the stopped-HDV insertion + free-gap placement;
* ``schedules``    -- the R0-R4 response schedules (CommandOwnership-owned);
* ``gates``        -- authorization / model-lock / budget refusals;
* ``runner``       -- the gated warm-up + per-arm driver and receipts.

Everything is exercised against a fake traci in the tests; no SUMO/TraCI
process is started by this package.  Nothing here edits a frozen controller,
the wall v2.2 driver/patches, or the registration record.
"""
from __future__ import annotations

import sys
from pathlib import Path

#: make ``traffic_shepherd`` (workspace root) and the wd01 packages importable
#: exactly as the existing wd01 suites do, without installing anything.
_INC1 = Path(__file__).resolve().parent
for _root in (_INC1.parent.parent.parent, _INC1.parent):  # workspace, wd01
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

__all__ = ["constants"]
