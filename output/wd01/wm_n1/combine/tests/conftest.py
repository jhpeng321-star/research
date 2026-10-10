"""Path shim for the combine tests.

``import combine`` resolves via the wm_n1 directory; ``partB_v0`` and
``wdpaths`` resolve via output/wd01.  The wd01-wide conftest installs the
forbid-D hook when WD_FORBID_D=1.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMBINE = HERE.parent                 # output/wd01/wm_n1/combine
WM_N1 = COMBINE.parent                # output/wd01/wm_n1
WD01 = WM_N1.parent.parent            # output/wd01

for _p in (str(WM_N1), str(WD01)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
