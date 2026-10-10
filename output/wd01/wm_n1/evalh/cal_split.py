"""Amendment 4 CAL split: the single CAL-tune / CAL-cal loader and filter.

``REGISTRATION_WM_N1_AMENDMENT_4_M1_TUNING.md`` (A4.2) froze a MANDATORY
world-level split of the CAL worlds into two halves, materialised ONCE (before
any training) as ``wm_n1/CAL_SPLIT_A4.json``:

* **cal_tune** -- used ONLY by the member-1-first improvement loop's selection
  metric (and by any quick-tune scoring);
* **cal_cal**  -- used ONLY by the registered early stopping, calibration,
  blend weights and variant choice.

This module is the *single* loader/filter for that split.  ``train``, ``evalh``
and ``combine`` all resolve their CAL worlds here, so no path can silently
consult the other half: :func:`resolve_worlds` / :func:`require_worlds` raise
:class:`CalSplitError` the moment a caller hands them a world from the wrong
half (or a world that is not in the split at all).

The split names are world NAME metadata only (from the chunk WORLD_PLANs); no
outcome is read by this module.  Never open real CAL/HOLDOUT/chunk-5 files in
tests -- point ``path=`` at a synthetic split file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CAL_SPLIT_SCHEMA = "wm_n1_cal_split_a4/1"
CAL_SPLIT_FILENAME = "CAL_SPLIT_A4.json"
AMENDMENT_4_FILENAME = "REGISTRATION_WM_N1_AMENDMENT_4_M1_TUNING.md"

#: Frozen LF-normalised sha256 values (A4_FREEZE.json; CRLF-safe).
LF_SHA_AMENDMENT_4 = (
    "d392b30030f2b04ddb5db25c681cf92bca80fe1fc4e77d234303f7bbb810822f")
LF_SHA_CAL_SPLIT_A4 = (
    "bc4e32f2b6efcf0e0165cb175a0b3ffaec0aaa6f419076463f9242ad90820d46")

WM_N1_DIR = Path(__file__).resolve().parents[1]
CAL_SPLIT_PATH = WM_N1_DIR / CAL_SPLIT_FILENAME
AMENDMENT_4_PATH = WM_N1_DIR / AMENDMENT_4_FILENAME

#: Canonical half names and their JSON keys.
CAL_TUNE = "cal_tune"
CAL_CAL = "cal_cal"
HALVES = (CAL_TUNE, CAL_CAL)

#: Which callers may read which half (documentation + tests).
SELECTION_HALF = CAL_TUNE
CALIBRATION_HALF = CAL_CAL


class CalSplitError(RuntimeError):
    """A CAL half was requested with worlds from the other half (or unknown)."""


def _lf_sha(data: bytes) -> str:
    return hashlib.sha256(
        data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _resolve_path(path) -> Path:
    return Path(path) if path is not None else CAL_SPLIT_PATH


def _as_world(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("world", ""))
    world = getattr(item, "world", None)
    if world is None:
        raise CalSplitError(
            f"cannot read a world name from {type(item).__name__!r}; pass "
            "strings or dicts carrying a 'world' key")
    return str(world)


def load_cal_split(path=None) -> dict:
    """Load and structurally validate ``CAL_SPLIT_A4.json``.

    Refuses a missing/ill-formed split, a split whose two halves overlap, or a
    split with an empty half: the halves are disjoint and together cover the
    CAL set by construction.
    """
    p = _resolve_path(path)
    if not p.is_file():
        raise CalSplitError(f"CAL split not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                # noqa: BLE001
        raise CalSplitError(f"CAL split is not JSON: {p}: {e}") from e
    if not isinstance(data, dict):
        raise CalSplitError(f"CAL split is not an object: {p}")
    schema = data.get("schema")
    if schema != CAL_SPLIT_SCHEMA:
        raise CalSplitError(
            f"CAL split schema {schema!r} != {CAL_SPLIT_SCHEMA!r}")
    tune = [str(w) for w in (data.get(CAL_TUNE) or [])]
    cal = [str(w) for w in (data.get(CAL_CAL) or [])]
    if not tune or not cal:
        raise CalSplitError(
            f"CAL split {p} has an empty half: cal_tune={len(tune)} "
            f"cal_cal={len(cal)}")
    if len(set(tune)) != len(tune) or len(set(cal)) != len(cal):
        raise CalSplitError(f"CAL split {p} repeats a world name")
    overlap = sorted(set(tune) & set(cal))
    if overlap:
        raise CalSplitError(
            f"CAL split {p} halves overlap: {overlap}")
    return dict(data, path=str(p), cal_tune=tune, cal_cal=cal)


def half_worlds(half: str, *, path=None) -> list:
    """The canonical world names of one half (validated against the split)."""
    if half not in HALVES:
        raise CalSplitError(f"unknown CAL half {half!r}; expected {HALVES}")
    data = load_cal_split(path)
    return list(data[half])


def half_set(half: str, *, path=None) -> frozenset:
    return frozenset(half_worlds(half, path=path))


def world_half(world, *, path=None) -> str | None:
    """Which half ``world`` belongs to, or ``None`` if it is in neither."""
    w = _as_world(world)
    data = load_cal_split(path)
    if w in set(data[CAL_TUNE]):
        return CAL_TUNE
    if w in set(data[CAL_CAL]):
        return CAL_CAL
    return None


def require_worlds(worlds, half: str, *, path=None, context: str = "") -> list:
    """Refuse unless every ``world`` belongs to ``half``.

    This is the guard each path calls before it scores: any world from the
    other half (or unknown to the split) raises :class:`CalSplitError`, so the
    selection path can never read CAL-cal and the calibration path can never
    read CAL-tune.
    """
    if half not in HALVES:
        raise CalSplitError(f"unknown CAL half {half!r}; expected {HALVES}")
    want = half_set(half, path=path)
    other = CAL_CAL if half == CAL_TUNE else CAL_TUNE
    other_set = half_set(other, path=path)
    vals = [_as_world(w) for w in worlds]
    wrong = sorted({w for w in vals if w in other_set})
    unknown = sorted({w for w in vals if w not in want and w not in other_set})
    if wrong or unknown:
        where = f"{context}: " if context else ""
        raise CalSplitError(
            f"{where}{half} path received world(s) from the other half "
            f"{wrong} or unknown to the CAL split {unknown}; {half} may read "
            f"ONLY {half} (amendment A4.2)")
    return vals


def filter_worlds(worlds, half: str, *, path=None) -> list:
    """Keep only the worlds of ``half``; refuse a world outside the split.

    Used to restrict a mixed CAL set to one half (the other half is dropped,
    never scored).  A world in neither half is a sealed/foreign world and is
    refused.
    """
    if half not in HALVES:
        raise CalSplitError(f"unknown CAL half {half!r}; expected {HALVES}")
    want = half_set(half, path=path)
    other = half_set(CAL_CAL if half == CAL_TUNE else CAL_TUNE, path=path)
    out, unknown = [], []
    for w in worlds:
        name = _as_world(w)
        if name in want:
            out.append(name)
        elif name not in other:
            unknown.append(name)
    if unknown:
        raise CalSplitError(
            f"{half} filter received world(s) outside the CAL split: "
            f"{sorted(set(unknown))}")
    return out


def resolve_worlds(half: str, supplied=None, *, path=None,
                   context: str = "") -> list:
    """Resolve the worlds a path may use for ``half``.

    With ``supplied=None`` the canonical half from ``CAL_SPLIT_A4.json`` is
    returned.  With an explicit ``supplied`` list, every world must belong to
    ``half`` (otherwise :class:`CalSplitError`), so asking a path for the wrong
    half is refused rather than silently accepted.
    """
    if supplied is None:
        return half_worlds(half, path=path)
    return require_worlds(list(supplied), half, path=path, context=context)


def verify_cal_split_sha(path=None) -> str:
    """LF-sha256 of the split file, checking it against the frozen value."""
    p = _resolve_path(path)
    if not p.is_file():
        raise CalSplitError(f"CAL split not found: {p}")
    got = _lf_sha(p.read_bytes())
    if got != LF_SHA_CAL_SPLIT_A4:
        raise CalSplitError(
            f"{p} LF-sha256 {got} != frozen {LF_SHA_CAL_SPLIT_A4}")
    return got


__all__ = [
    "CAL_SPLIT_SCHEMA", "CAL_SPLIT_FILENAME", "AMENDMENT_4_FILENAME",
    "LF_SHA_AMENDMENT_4", "LF_SHA_CAL_SPLIT_A4", "CAL_SPLIT_PATH",
    "AMENDMENT_4_PATH", "CAL_TUNE", "CAL_CAL", "HALVES", "SELECTION_HALF",
    "CALIBRATION_HALF", "CalSplitError", "load_cal_split", "half_worlds",
    "half_set", "world_half", "require_worlds", "filter_worlds",
    "resolve_worlds", "verify_cal_split_sha",
]
