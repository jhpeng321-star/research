"""Package shim: INC1 R4 imports the PATCHED ``campaign_walls_v1``.

Resolution order (first wins):
1. ``walls_design/merge_release/copies/post_inc1`` - the patched
   ``wall_driver.py`` (frozen post + ``patches/wall_driver_inc1_hook.diff``);
2. ``walls_design/merge_release/copies/post`` - the frozen v2.2
   ``common.py`` / ``wall_request.py`` (unchanged by the patch).

Never writes to either copy directory.  ``campaign_pilot_v1`` /
``campaign_stage1_v3`` resolve from the live read-only
``E:/research-dense/output`` tree (appended to ``sys.path`` by
``inc1.r4_driver``).
"""
from pathlib import Path

_HERE = Path(__file__).resolve()
_WD01 = _HERE.parents[3]                              # output/wd01
_MERGE = _WD01 / "walls_design" / "merge_release" / "copies"
_POST_INC1 = _MERGE / "post_inc1"
_POST = _MERGE / "post"

__path__ = [str(_POST_INC1), str(_POST)]  # noqa: W0212 - package path hook
