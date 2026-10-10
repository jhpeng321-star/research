"""Frozen wall-law v2.2 patch-sha verification (INC1 R4 start gate).

Registration ``REGISTRATION_WALL_V22.md`` (LF-sha ``e7fd3536...``) binds the
wall law to the PATCHED v2.2 files, whose LF sha256s are recorded in
``output/wd01/walls_design/merge_release/PATCH_SHAS.json`` and whose frozen
deployed copies are
``output/wd01/walls_design/merge_release/copies/post/``.

INC1's R4 must NOT re-implement that law.  This module is the runner-start
gate: it recomputes the frozen copies' LF sha256 and refuses (loudly) on any
mismatch, so a drifted/tampered patch can never reach the incident runner.
It never edits the frozen driver, patches, copies or the registration.

The deployed ``wall_driver.py`` copy carries the behaviour-neutral v2.2
LOGGING patch (``PATCH_SHAS.json`` ``logging_patch`` section); the pure-law
``post`` sha is retained in the record for transparency.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    from . import constants as C
except ImportError:  # script execution
    import constants as C  # type: ignore


class FreezeError(RuntimeError):
    """A frozen v2.2 patch/copy violated its registered LF sha; refuse to run."""


MERGE_RELEASE_DIR = C.WD01_ROOT / "walls_design" / "merge_release"
PATCH_SHAS_PATH = MERGE_RELEASE_DIR / "PATCH_SHAS.json"
POST_DIR = MERGE_RELEASE_DIR / "copies" / "post"
POST_INC1_DIR = MERGE_RELEASE_DIR / "copies" / "post_inc1"
HOOK_PATCH_PATH = MERGE_RELEASE_DIR / "patches" / "wall_driver_inc1_hook.diff"
FREEZE_ADDENDUM_PATH = MERGE_RELEASE_DIR / "WALL_V22_FREEZE_ADDENDUM_INC1.json"

#: copy filename -> (PATCH_SHAS section, section filename key, sha key)
FROZEN_COPIES = (
    ("common.py", "post", "common.py", "lf_sha256"),
    ("wall_request.py", "post", "wall_request.py", "lf_sha256"),
    ("wall_group.py", "post", "wall_group.py", "lf_sha256"),
    ("wall_driver.py", "logging_patch", "wall_driver.py", "patched_lf_sha256"),
    ("sensing.py", "post", "sensing.py", "lf_sha256"),
)


def _lf_sha256(path: Path) -> str:
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def verify_frozen_v22_patches(*, patch_shas_path: Path | None = None,
                              post_dir: Path | None = None) -> dict:
    """Verify every frozen v2.2 copy against PATCH_SHAS.json; refuse on mismatch.

    Returns the evidence record (per-file expected/actual/match) on success.
    Raises :class:`FreezeError` if the manifest, a copy or a sha is wrong.
    """
    shas_path = Path(patch_shas_path or PATCH_SHAS_PATH)
    copies_dir = Path(post_dir or POST_DIR)
    if not shas_path.is_file():
        raise FreezeError(f"v2.2 PATCH_SHAS.json is missing: {shas_path}")
    try:
        manifest = json.loads(shas_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FreezeError(f"{shas_path} is not valid JSON: {exc}") from exc

    files = []
    for filename, section, key, sha_key in FROZEN_COPIES:
        entry = (manifest.get(section) or {}).get(key) or {}
        expected = entry.get(sha_key)
        copy_path = copies_dir / filename
        if not expected:
            raise FreezeError(
                f"PATCH_SHAS.json has no {section}.{key}.{sha_key} entry")
        if not copy_path.is_file():
            raise FreezeError(f"frozen v2.2 copy is missing: {copy_path}")
        actual = _lf_sha256(copy_path)
        match = actual == expected
        files.append({
            "file": filename,
            "copy": str(copy_path),
            "manifest_section": f"{section}.{key}.{sha_key}",
            "expected_lf_sha256": expected,
            "actual_lf_sha256": actual,
            "match": match,
            "patch": entry.get("patch"),
        })
    mismatches = [row["file"] for row in files if not row["match"]]
    if mismatches:
        raise FreezeError(
            "frozen wall-law v2.2 copy sha mismatch for "
            f"{mismatches}; refusing to start (the registration binds "
            "REGISTRATION_WALL_V22.md e7fd3536 and PATCH_SHAS.json)")
    return {
        "schema": "wm_n1_inc1_wall_v22_freeze_check/1",
        "registration": "walls_design/merge_release/REGISTRATION_WALL_V22.md",
        "registration_lf_sha256_expected": (
            "e7fd353616eea9ef2ab0440c751c8c9211f11a3f5de97f1b911b7a528f6f01e7"),
        "patch_shas_path": str(shas_path),
        "post_dir": str(copies_dir),
        "files": files,
        "all_match": True,
    }


def verify_inc1_hook_patch(*, patch_shas_path: Path | None = None,
                           post_inc1_dir: Path | None = None,
                           hook_patch_path: Path | None = None,
                           addendum_path: Path | None = None) -> dict:
    """Verify the INC1 hook patch provenance (A1.2 / A1.6).

    Checks: the patched ``copies/post_inc1/wall_driver.py`` LF-sha matches the
    ``inc1_hook`` entry in ``PATCH_SHAS.json``, the diff file's LF-normalised
    sha256 matches its recorded ``diff_sha256``, the diff exists, and
    ``WALL_V22_FREEZE_ADDENDUM_INC1.json`` exists (and binds the same sha).
    Raises :class:`FreezeError` on any mismatch.
    """
    shas_path = Path(patch_shas_path or PATCH_SHAS_PATH)
    inc1_dir = Path(post_inc1_dir or POST_INC1_DIR)
    diff_path = Path(hook_patch_path or HOOK_PATCH_PATH)
    addendum = Path(addendum_path or FREEZE_ADDENDUM_PATH)
    if not shas_path.is_file():
        raise FreezeError(f"v2.2 PATCH_SHAS.json is missing: {shas_path}")
    manifest = json.loads(shas_path.read_text(encoding="utf-8"))
    entry = (manifest.get("inc1_hook") or {}).get("wall_driver.py") or {}
    expected = entry.get("patched_lf_sha256")
    if not expected:
        raise FreezeError(
            "PATCH_SHAS.json has no inc1_hook.wall_driver.py.patched_lf_sha256")
    copy_path = inc1_dir / "wall_driver.py"
    if not copy_path.is_file():
        raise FreezeError(f"patched INC1 driver copy is missing: {copy_path}")
    actual = _lf_sha256(copy_path)
    if actual != expected:
        raise FreezeError(
            "patched INC1 driver sha mismatch: "
            f"{actual} != {expected} (refusing to run R4)")
    if not diff_path.is_file():
        raise FreezeError(f"INC1 hook patch is missing: {diff_path}")
    # the recorded ``diff_sha256`` is the LF-normalised sha of the text patch
    # (the repository hashes text with the ``lf_sha256`` convention); a raw
    # hash would break on a ``core.autocrlf=true`` checkout.
    diff_sha = _lf_sha256(diff_path)
    recorded_diff = entry.get("diff_sha256")
    if recorded_diff and diff_sha != recorded_diff:
        raise FreezeError(
            f"INC1 hook patch sha {diff_sha} != recorded {recorded_diff}")
    if not addendum.is_file():
        raise FreezeError(
            "WALL_V22_FREEZE_ADDENDUM_INC1.json is missing: "
            f"{addendum} (A1.6 gate)")
    record = json.loads(addendum.read_text(encoding="utf-8"))
    declared = (record.get("patched") or {}).get("lf_sha256")
    if declared and declared != expected:
        raise FreezeError(
            "freeze addendum patched sha does not match PATCH_SHAS inc1_hook")
    return {
        "schema": "wm_n1_inc1_inc1_hook_freeze_check/1",
        "patch_shas_path": str(shas_path),
        "patched_file": str(copy_path),
        "expected_lf_sha256": expected,
        "actual_lf_sha256": actual,
        "match": True,
        "diff": str(diff_path),
        "diff_sha256": diff_sha,
        "expected_diff_sha256": recorded_diff,
        "addendum": str(addendum),
    }


def main(argv=None) -> int:
    """Print the freeze-check evidence; exit 1 on any mismatch."""
    record = verify_frozen_v22_patches()
    print(json.dumps(record, indent=2))
    print(f"FROZEN v2.2 PATCH SHAs OK ({len(record['files'])} files)")
    hook = verify_inc1_hook_patch()
    print(f"INC1 HOOK PATCH OK ({hook['actual_lf_sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
