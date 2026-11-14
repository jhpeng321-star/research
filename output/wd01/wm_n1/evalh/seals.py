"""Seal enforcement for the WM-N1 evaluation harness.

Hard seals (owner constraints, 2026-09-30):

* **CAL** is read at most once per marker directory (early stopping,
  calibration, blend weights and the variant choice); a second CAL read is
  refused.  REGISTRATION_WM_N1.md:56-58,70-72.
* **chunk-5 TEST** is opened ONLY with ``--final-test`` AND the frozen
  registration hashes below, and only once.  REGISTRATION_WM_N1.md:60-73.
* **HOLDOUT** is ALWAYS refused, via the canonical
  ``E:/research-dense/tools/holdout_guard.py`` (owner ruling, PL-02a).

All hashing is **LF-normalised** (``\\r\\n`` -> ``\\n``) so a CRLF checkout
and an LF checkout of a frozen registration hash identically.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_WD01 = Path(__file__).resolve().parents[2]
if str(_WD01) not in sys.path:
    sys.path.insert(0, str(_WD01))
try:
    import wdpaths  # noqa: E402
    RD = Path(wdpaths.RESEARCH_DENSE_ROOT)
except Exception:                                    # pragma: no cover
    RD = Path(os.environ.get("RESEARCH_DENSE_ROOT", "E:/research-dense"))

WM_N1_DIR = Path(__file__).resolve().parents[1]
REGISTRATION_WM_N1 = WM_N1_DIR / "REGISTRATION_WM_N1.md"
REGISTRATION_CHUNK5 = WM_N1_DIR / "REGISTRATION_CHUNK5_TEST.md"

# Frozen LF-normalised sha256 values from the WM-N1 task freeze.
LF_SHA_CHUNK5 = ("f8c7d760baedeb359c501cd90a3e2a599fb0ddfef390ea555cf304"
                 "c38214b31e")
LF_SHA_WM_N1 = ("666233aa1d1e466750d6d944b9102c84a2f32850ebeb4a97c9f3c9b56"
                "dd93246")

#: Frozen Amendment 6 (state-only global token for the correction head).  The
#: A6 freeze is the ONLY thing that lets ``state_global_token`` default ON: every
#: approval binds this LF-sha and training refuses the flag ON without the file.
LF_SHA_AMENDMENT_6 = ("28dbc184b0178d5f74f681e0701771725fc8abc80e534e1087f0a9"
                      "e49835c948")
AMENDMENT_6_FILENAME = "REGISTRATION_WM_N1_AMENDMENT_6_STATE_TOKEN.md"
AMENDMENT_6_PATH = WM_N1_DIR / AMENDMENT_6_FILENAME

SPLITS = ("train_oof", "cal", "final_test")


class SealError(PermissionError):
    """A sealed split or file was accessed against the registration."""


class HoldoutRefusal(SealError):
    """A HOLDOUT world was requested."""


def lf_normalize(data: bytes) -> bytes:
    """Normalise CRLF (and lone CR) line endings to LF for hashing."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def lf_sha256(path) -> str:
    """LF-normalised sha256 of a file (CRLF-safe)."""
    return hashlib.sha256(lf_normalize(Path(path).read_bytes())).hexdigest()


def _read_bytes(path) -> bytes:
    return Path(path).read_bytes()


# ---------------------------------------------------------- AMENDMENT 6 ------

class Amendment6Error(SealError):
    """Amendment 6 is absent, or its LF-sha256 does not match the freeze."""


def _a6_path(path=None) -> Path:
    return Path(path) if path is not None else AMENDMENT_6_PATH


def amendment_6_lf_sha256(path=None) -> str:
    """Actual LF-normalised sha256 of Amendment 6 (CRLF-safe)."""
    return lf_sha256(_a6_path(path))


def amendment_6_frozen(path=None) -> bool:
    """True iff the frozen A6 file is present and hashes to the freeze."""
    p = _a6_path(path)
    if not p.is_file():
        return False
    try:
        return lf_sha256(p) == LF_SHA_AMENDMENT_6
    except OSError:                                      # pragma: no cover
        return False


def require_frozen_amendment_6(path=None) -> str:
    """Refuse unless the frozen A6 file is present and its LF-sha matches.

    This is the single gate that ties ``state_global_token`` ON to the A6
    freeze (REGISTRATION_WM_N1_AMENDMENT_6_STATE_TOKEN.md): training and the
    approvals both call it, so the flag can never be ON without the record.
    """
    p = _a6_path(path)
    if not p.is_file():
        raise Amendment6Error(
            "Amendment 6 (state-only global token) is not frozen/present at "
            f"{p}; state_global_token may not be ON without the frozen A6 file")
    got = lf_sha256(p)
    if got != LF_SHA_AMENDMENT_6:
        raise Amendment6Error(
            f"Amendment 6 LF-sha256 {got} != frozen {LF_SHA_AMENDMENT_6} "
            f"at {p}")
    return got


# --------------------------------------------------------------- HOLDOUT ----

def load_holdout_guard():
    """Import the canonical holdout guard from ``<RD>/tools``."""
    tools = RD / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    try:
        import holdout_guard
        return holdout_guard
    except Exception:                                # pragma: no cover
        return None


def assert_holdout(worlds, data: dict | None = None) -> None:
    """Refuse HOLDOUT/excluded roots via the canonical guard.

    With ``data=None`` the guard reads
    ``protocols/holdout_exclusions.json``; tests may inject a synthetic
    ``data`` dict.  Also enforces the literal ``HOLDOUT`` name.
    """
    worlds = [str(w) for w in worlds]
    for w in worlds:
        if str(w).split("__", 1)[0].upper() == "HOLDOUT":
            raise HoldoutRefusal(f"HOLDOUT world {w!r} is always refused")
    guard = load_holdout_guard()
    if guard is None:                                # pragma: no cover
        if data is None:
            raise HoldoutRefusal(
                "holdout_guard.py unavailable and no injected exclusions")
        bad = {row["world"] for row in data.get("excluded", [])}
        hit = sorted({str(w).split("__", 1)[0] for w in worlds} & bad)
        if hit:
            raise HoldoutRefusal(f"excluded HOLDOUT roots refused: {hit}")
        return
    try:
        guard.assert_final_holdout_eligible(worlds, data)
    except HoldoutRefusal:
        raise
    except Exception as e:                           # noqa: BLE001
        raise HoldoutRefusal(
            f"holdout_guard refused {worlds}: {e}") from e


# --------------------------------------------------------------- markers ----

def _marker_path(marker_dir, name: str) -> Path:
    return Path(marker_dir) / name


def _write_marker(marker_dir, name: str, payload: dict) -> Path:
    marker_dir = Path(marker_dir)
    marker_dir.mkdir(parents=True, exist_ok=True)
    p = _marker_path(marker_dir, name)
    body = dict(payload)
    body["written_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return p


def _read_marker(marker_dir, name: str) -> dict | None:
    p = _marker_path(marker_dir, name)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                # noqa: BLE001
        return {"corrupt": True}


CAL_MARKER = "SEAL_CAL_READ.json"
TEST_MARKER = "SEAL_FINAL_TEST_READ.json"


def read_once(marker_dir, name: str, payload: dict) -> Path:
    """Write a read-once marker; refuse if it already exists."""
    existing = _read_marker(marker_dir, name)
    if existing is not None:
        raise SealError(
            f"{name} already exists in {marker_dir}: this split is read-once "
            "(already consumed); refusing a second read")
    return _write_marker(marker_dir, name, payload)


# --------------------------------------------------------------- split gate --

def guard_split(split: str, *, marker_dir, final_test: bool = False,
                registration_sha: str | None = None,
                chunk5_sha: str | None = None,
                wmn1_path=REGISTRATION_WM_N1, chunk5_path=REGISTRATION_CHUNK5,
                dry_run: bool = False) -> dict:
    """Resolve a split against the seals; return a receipt dict.

    ``train_oof`` is free.  ``cal`` consumes its read-once marker.
    ``final_test`` additionally needs ``--final-test`` and BOTH frozen
    LF-sha256 values (REGISTRATION_WM_N1.md:60-73).
    """
    if split not in SPLITS:
        raise SealError(f"unknown split {split!r}; expected one of {SPLITS}")

    if split == "train_oof":
        return dict(split=split, allowed=True, sealed=False, marker=None)

    if split == "cal":
        if dry_run:
            if _read_marker(marker_dir, CAL_MARKER) is not None:
                raise SealError(
                    f"CAL already consumed in {marker_dir} (read-once)")
            return dict(split=split, allowed=True, sealed=True, marker=None)
        marker = read_once(marker_dir, CAL_MARKER,
                           dict(split=split, note="CAL read-once"))
        return dict(split=split, allowed=True, sealed=True,
                    marker=str(marker))

    # final_test (chunk 5)
    if not final_test:
        raise SealError(
            "chunk-5 TEST is sealed; pass --final-test to proceed "
            "(REGISTRATION_WM_N1.md:60-73)")
    want_chunk5 = chunk5_sha or lf_sha256(chunk5_path)
    want_wmn1 = registration_sha or lf_sha256(wmn1_path)
    if want_chunk5 != LF_SHA_CHUNK5:
        raise SealError(
            f"chunk-5 registration LF-sha256 mismatch: got {want_chunk5!r}, "
            f"expected {LF_SHA_CHUNK5!r}")
    if want_wmn1 != LF_SHA_WM_N1:
        raise SealError(
            f"REGISTRATION_WM_N1.md LF-sha256 mismatch: got {want_wmn1!r}, "
            f"expected {LF_SHA_WM_N1!r}")
    if dry_run:
        if _read_marker(marker_dir, TEST_MARKER) is not None:
            raise SealError(
                f"chunk-5 TEST already consumed in {marker_dir} (read-once)")
        return dict(split=split, allowed=True, sealed=True, marker=None,
                    chunk5_lf_sha256=want_chunk5, wmn1_lf_sha256=want_wmn1)
    marker = read_once(marker_dir, TEST_MARKER,
                       dict(split=split, chunk5_lf_sha256=want_chunk5,
                            wmn1_lf_sha256=want_wmn1,
                            note="chunk-5 final TEST read-once"))
    return dict(split=split, allowed=True, sealed=True, marker=str(marker),
                chunk5_lf_sha256=want_chunk5, wmn1_lf_sha256=want_wmn1)
