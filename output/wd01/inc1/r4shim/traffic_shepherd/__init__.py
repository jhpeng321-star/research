"""Package shim: ``traffic_shepherd`` resolves wall law v2.2 FIRST.

``wall_group`` / ``sensing`` come from the PATCHED copies
(``merge_release/copies/post``); every other submodule (``config``,
``mixed_assignment``, ``ownership``, ...) falls through to this workspace's
frozen ``traffic_shepherd/`` tree.  The first ``__path__`` entry wins, so the
v2.2 wall law shadows the pre-v2.2 repo module.
"""
from pathlib import Path

_HERE = Path(__file__).resolve()
_WD01 = _HERE.parents[3]                              # output/wd01
_POST = (_WD01 / "walls_design" / "merge_release" / "copies" / "post")
_REPO = _WD01.parents[1] / "traffic_shepherd"         # workspace root

__path__ = [str(_POST), str(_REPO)]  # noqa: W0212 - package path hook
