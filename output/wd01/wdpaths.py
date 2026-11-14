"""Path roots for WD-01 live code: E:-native, no D: dependence.

The research data used to live behind D: junctions (D:/research-dense,
D:/rd-wt, D:/research-dense-backups -> E:).  The owner will delete those
junctions, so every part resolves its roots here and every place that
OPENS a path read back from a manifest/record/registration file routes it
through remap() (frozen records keep their historical D: strings).

Roots (read once at import, like the rest of the wd01 libs):
  RESEARCH_DENSE_ROOT  env RESEARCH_DENSE_ROOT,          default E:/research-dense
  RD_WT_ROOT           env RD_WT_ROOT,                   default E:/rd-wt
  BACKUPS_ROOT         env RESEARCH_DENSE_BACKUPS_ROOT,  default E:/research-dense-backups

remap(path) rewrites a leading D:/research-dense, D:/research-dense-backups
or D:/rd-wt prefix (any slash style, drive letter case-insensitive, \\?\
extended-length forms too) to the corresponding root and leaves every
other path unchanged.  str in -> str out, Path in -> Path out.

Import from any part via the usual sys.path shim, e.g.:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import wdpaths
"""
from __future__ import annotations

import os
from pathlib import Path

RESEARCH_DENSE_ROOT = os.environ.get("RESEARCH_DENSE_ROOT", "").strip() \
    or "E:/research-dense"
RD_WT_ROOT = os.environ.get("RD_WT_ROOT", "").strip() or "E:/rd-wt"
BACKUPS_ROOT = os.environ.get("RESEARCH_DENSE_BACKUPS_ROOT", "").strip() \
    or "E:/research-dense-backups"

# Mirror the resolved roots into the environment so read-only REFERENCE
# modules that read these variables directly (e.g. the frozen
# hybrid_state_v1/repo.py, whose baked-in default is the retired D:
# junction) resolve to the live E: roots even when the caller exported
# nothing.  Explicit user exports always win (setdefault).
for _name, _val in (("RESEARCH_DENSE_ROOT", RESEARCH_DENSE_ROOT),
                    ("RD_WT_ROOT", RD_WT_ROOT),
                    ("RESEARCH_DENSE_BACKUPS_ROOT", BACKUPS_ROOT)):
    os.environ.setdefault(_name, _val)

# longest prefix first so research-dense-backups wins over research-dense
_PREFIXES = (
    ("research-dense-backups", BACKUPS_ROOT),
    ("research-dense", RESEARCH_DENSE_ROOT),
    ("rd-wt", RD_WT_ROOT),
)
_EXT_PREFIX = "\\\\?\\"          # Windows extended-length prefix


def _remap_str(s: str) -> str:
    ext = s.startswith(_EXT_PREFIX)
    core = s[len(_EXT_PREFIX):] if ext else s
    if len(core) < 2 or core[1] != ":" or core[0].upper() != "D":
        return s
    rest = core[2:]
    if rest and rest[0] not in "/\\":
        return s                             # drive-relative, not ours
    tail = rest.lstrip("/\\")                # first separator is re-derived
    low = tail.lower()
    for name, root in _PREFIXES:
        if low.startswith(name):
            after = tail[len(name):]
            if after and after[0] not in "/\\":
                continue                     # e.g. research-denseX: not ours
            base = root.rstrip("/\\")
            out = (base + after) if after else base
            return _EXT_PREFIX + out if ext else out
    return s


def remap(path):
    """Rewrite a leading D: junction prefix to its E: root; else unchanged.

    Accepts str or any os.PathLike (returns the same kind); raises TypeError
    for anything else.  Never touches paths that do not start with a D:
    drive (any case), including relative paths and other drives.
    """
    if isinstance(path, str):
        return _remap_str(path)
    if isinstance(path, Path):
        return Path(_remap_str(os.fspath(path)))
    if isinstance(path, os.PathLike):
        return _remap_str(os.fspath(path))
    raise TypeError(f"remap: expected str or path-like, got {type(path)!r}")
