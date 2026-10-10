"""Comparison helpers + tolerance table for the golden reference artifacts.

Import this from the rewritten code's tests:

    from golden_compare import compare_json, compare_npz, TOLERANCES
    compare_npz(my_arrays, "forward_c1.npz")        # loads the stored golden itself

``actual`` can be a path or an in-memory dict (npz) / object (json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"

# file -> absolute tolerance for floating point values (0.0 == exact).
# Integers, bools, strings, and index/selection data are always compared exactly.
TOLERANCES = {
    "selection.json": 0.0,
    "batches_u10_histon": 0.0, "batches_u10_histoff": 0.0,
    "batches_u60_histon": 0.0, "batches_u60_histoff": 0.0,
    "forward_c1.npz": 1e-6, "forward_r3.npz": 1e-6,
    "forward_c1.json": 1e-6, "forward_r3.json": 1e-6,
    "r3_init_state.npz": 0.0,           # fresh init with torch.manual_seed(101)
    "losses_r3.json": 1e-5,
    "train3_r3.json": 1e-5,
    "train3_final_params.npz": 1e-5,
    "eval_rollouts.json": 1e-5,
    "eval_pred_frames.npz": 1e-5,
}
# keys that are volatile and never compared
IGNORED_KEYS = {"seconds", "timings_s", "git_head", "generated", "torch", "numpy"}
# JSON keys holding bitwise parameter hashes: compared only when strict_hashes=True
# (the CURRENT code must match them; a rewrite cannot be expected to be bit-identical
# unless it reproduces the exact floating-point op order).
HASH_KEYS = {"checksum", "sha256_float32", "init_checksum"}


def load_npz(name_or_path) -> dict:
    p = Path(name_or_path)
    if not p.is_absolute():
        p = DATA / p
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_json(name_or_path):
    p = Path(name_or_path)
    if not p.is_absolute():
        p = DATA / p
    return json.loads(p.read_text(encoding="utf-8"))


def compare_arrays(actual: dict, expected: dict, atol: float, label: str = "") -> list[str]:
    errs = []
    if set(actual) != set(expected):
        errs.append(f"{label}: key mismatch missing={sorted(set(expected) - set(actual))[:5]} "
                    f"extra={sorted(set(actual) - set(expected))[:5]}")
    for k in sorted(set(actual) & set(expected)):
        a, e = np.asarray(actual[k]), np.asarray(expected[k])
        if a.shape != e.shape:
            errs.append(f"{label}:{k}: shape {a.shape} != {e.shape}")
            continue
        if e.dtype.kind in "fc":
            if not np.allclose(a.astype(np.float64), e.astype(np.float64), rtol=0.0,
                               atol=atol, equal_nan=True):
                diff = float(np.nanmax(np.abs(a.astype(np.float64) - e.astype(np.float64))))
                errs.append(f"{label}:{k}: max|diff|={diff:.3e} > atol {atol:g}")
        elif not np.array_equal(a, e):
            errs.append(f"{label}:{k}: integer/bool/string arrays differ")
    return errs


def _cmp(a, e, atol, path, errs, strict_hashes):
    if isinstance(e, dict):
        if not isinstance(a, dict):
            errs.append(f"{path}: expected object")
            return
        ek = {k for k in e if k not in IGNORED_KEYS}
        ak = {k for k in a if k not in IGNORED_KEYS}
        if ek != ak:
            errs.append(f"{path}: keys differ missing={sorted(ek - ak)[:5]} extra={sorted(ak - ek)[:5]}")
        for k in sorted(ek & ak):
            if k in HASH_KEYS and not strict_hashes:
                continue
            _cmp(a[k], e[k], atol, f"{path}.{k}", errs, strict_hashes)
    elif isinstance(e, list):
        if not isinstance(a, list) or len(a) != len(e):
            errs.append(f"{path}: list length {len(a) if isinstance(a, list) else '?'} != {len(e)}")
            return
        for i, (x, y) in enumerate(zip(a, e)):
            _cmp(x, y, atol, f"{path}[{i}]", errs, strict_hashes)
    elif isinstance(e, bool) or e is None or isinstance(e, str):
        if a != e:
            errs.append(f"{path}: {a!r} != {e!r}")
    elif isinstance(e, int):
        if a != e:
            errs.append(f"{path}: {a!r} != {e!r}")
    elif isinstance(e, float):
        if a is None or not isinstance(a, (int, float)):
            errs.append(f"{path}: {a!r} != {e!r}")
        elif abs(float(a) - e) > atol and not (e != e and a != a):
            errs.append(f"{path}: {a!r} vs {e!r} (atol {atol:g})")
    else:
        errs.append(f"{path}: unsupported type {type(e)}")


def compare_json(actual, expected_name: str, atol: float | None = None,
                 strict_hashes: bool = False) -> list[str]:
    """Return a list of mismatch descriptions (empty == match)."""
    expected = load_json(expected_name)
    if not isinstance(actual, (dict, list)):
        actual = load_json(actual)
    tol = TOLERANCES.get(expected_name, 0.0) if atol is None else atol
    errs: list[str] = []
    _cmp(actual, expected, tol, expected_name, errs, strict_hashes)
    return errs


def compare_npz(actual, expected_name: str, atol: float | None = None) -> list[str]:
    expected = load_npz(expected_name)
    if not isinstance(actual, dict):
        actual = load_npz(actual)
    tol = TOLERANCES.get(expected_name, 0.0) if atol is None else atol
    return compare_arrays(actual, expected, tol, expected_name)
