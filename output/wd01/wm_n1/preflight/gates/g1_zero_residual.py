"""G1 ``zero-residual``.

With the HDV residual forced to 0 and arrivals OFF, the Idea A rollout
(P3 + P1 + baseline + validity) must reproduce the Part A v2g rollout on the
same J4 TRAIN window: per vehicle per tick, positions and speeds within
``1e-6`` (or an explained, documented tolerance).

Entry point (P3): ``rollout2.equivalence.zero_residual_report()`` returning::

    {"ok": bool, "windows": [...], "max_position_error_m": float,
     "max_speed_error_mps": float, "integrator": "v2g",
     "tolerance": 1e-6, "tolerance_explained": "..."}
"""
from __future__ import annotations

from ..imports import PieceMissing
from ..result import fail, ok

NAME = "zero-residual"
TITLE = "G1 zero-residual (Idea A == Part A v2g, residual 0, arrivals off)"
ENTRY_POINTS = ("rollout2.equivalence.zero_residual_report",)
BASE_TOLERANCE = 1e-6


def evaluate(report, *, base_tolerance: float = BASE_TOLERANCE):
    """Pure check over a v2g-equivalence report -> (passed, detail, checks)."""
    if not isinstance(report, dict):
        return False, f"report is {type(report).__name__}, expected dict", {}
    tol = float(report.get("tolerance", base_tolerance))
    pos = float(report.get("max_position_error_m", float("inf")))
    spd = float(report.get("max_speed_error_mps", float("inf")))
    windows = report.get("windows") or []
    integrator = report.get("integrator", "v2g")
    checks = {"tolerance_m": tol, "max_position_error_m": pos,
              "max_speed_error_mps": spd, "n_windows": len(windows),
              "integrator": integrator,
              "tolerance_explained": bool(report.get("tolerance_explained"))}
    if integrator not in ("v2g", "partA_v2g", "parta_v2g"):
        return False, f"integrator mode {integrator!r} is not the v2g baseline", checks
    if not windows:
        return False, "no J4 TRAIN window reported", checks
    if tol > base_tolerance and not report.get("tolerance_explained"):
        return False, (f"tolerance {tol:g} > {base_tolerance:g} without an "
                       "explanation"), checks
    passed = bool(report.get("ok")) and pos <= tol and spd <= tol
    detail = (f"pos<= {pos:.3g} m, speed<= {spd:.3g} m/s, tol {tol:g}, "
              f"{len(windows)} window(s)")
    if not passed:
        detail = "residual-off rollout diverges from v2g: " + detail
    return passed, detail, checks


def run(settings, resolve):
    try:
        fn = resolve(ENTRY_POINTS[0])
    except PieceMissing as e:
        return fail(NAME, TITLE, str(e), ENTRY_POINTS, error="piece-missing")
    try:
        report = fn()
    except Exception as e:                                # noqa: BLE001
        return fail(NAME, TITLE, f"{ENTRY_POINTS[0]} raised "
                    f"{type(e).__name__}: {e}", ENTRY_POINTS, error="gate-error")
    passed, detail, checks = evaluate(report)
    return (ok if passed else fail)(NAME, TITLE, detail, ENTRY_POINTS, checks)
