"""Package shim: ``campaign_walls_v1`` resolves the PATCHED copies only.

(common.py / wall_request.py / wall_driver.py from
merge_release/copies/post; the unpatched launch/smoke/plan modules are
not needed by these tests and are deliberately not importable here.)
"""
from pathlib import Path

_POST = Path(__file__).resolve().parents[3] / "copies" / "post"

__path__ = [str(_POST)]  # noqa: W0212 - package path hook
