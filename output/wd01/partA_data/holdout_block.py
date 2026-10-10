"""HOLDOUT ACCESS BLOCK (WD-01 part 3d, shared by part3/part3d).

The 5 holdout roots (split == "holdout" in partA_data/MANIFEST.json) may
only be loaded through a registered gate:

    assert_holdout_allowed(root, registration_sha256=H, records_dir=D)

where H is the sha256 of an existing REGISTRATION file and D contains a
record JSON ({"sha256": H, "file": <registration path>}) that recorded H
BEFORE the access, and the referenced file still exists with exactly that
hash.  Without H, with a wrong H, or without a matching record, loading a
holdout root raises HoldoutAccessError.

Training/validation roots are unaffected.  The registry directory can be
pointed at a temp dir for tests (env WD01_HOLDOUT_RECORDS or the
records_dir argument).
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

P2C = Path(__file__).resolve().parent
MANIFEST = P2C / "MANIFEST.json"
DEFAULT_RECORDS_DIRS = (P2C / "results" / "holdout_records",
                        P2C.parent / "part3" / "results" / "holdout_records")


class HoldoutAccessError(PermissionError):
    """Raised when a holdout root is accessed without a valid,
    previously recorded registration hash."""


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1)
def _splits():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {name: r["split"] for name, r in man["roots"].items()}


def records_dirs(extra=None):
    dirs = []
    env = os.environ.get("WD01_HOLDOUT_RECORDS", "")
    if env:
        dirs.append(Path(env))
    if extra:
        dirs.append(Path(extra))
    dirs.extend(DEFAULT_RECORDS_DIRS)
    return dirs


def registration_recorded(records_dirs_list, reg_sha):
    """(ok, record_path, reg_file): True iff some record JSON in the
    registry recorded reg_sha for an existing registration file whose
    current hash still matches."""
    for d in records_dirs_list:
        d = Path(d)
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if rec.get("sha256") != reg_sha:
                continue
            reg_file = rec.get("file")
            if reg_file and Path(reg_file).exists() and \
                    sha256_file(reg_file) == reg_sha:
                return True, str(f), str(reg_file)
    return False, None, None


def assert_holdout_allowed(root, registration_sha256=None,
                           records_dirs_list=None):
    """Raise HoldoutAccessError for holdout roots without a valid
    registered hash; no-op for train/validation roots."""
    splits = _splits()
    split = splits.get(root)
    if split != "holdout":
        return
    dirs = records_dirs_list or records_dirs()
    if not registration_sha256:
        raise HoldoutAccessError(
            f"holdout root '{root}' requires registration_sha256 "
            f"(hash of an existing REGISTRATION file recorded in the "
            f"holdout records registry)")
    ok, rec, reg_file = registration_recorded(dirs, registration_sha256)
    if not ok:
        raise HoldoutAccessError(
            f"holdout root '{root}': no holdout-records entry matches "
            f"sha256 {registration_sha256}")
    return
