"""Package shim: ``traffic_shepherd`` resolves law v2.2 FIRST.

``wall_group``/``sensing`` come from the PATCHED copies
(merge_release/copies/post); every other submodule (config,
mixed_assignment, ...) falls through to this workspace's frozen
``traffic_shepherd/`` tree.  Order matters: the first ``__path__`` entry
wins, so the v2.2 wall law shadows the pre-v2.2 repo module.
"""
from pathlib import Path

_POST = Path(__file__).resolve().parents[3] / "copies" / "post"
_REPO = Path(__file__).resolve().parents[7] / "traffic_shepherd"

__path__ = [str(_POST), str(_REPO)]  # noqa: W0212 - package path hook
