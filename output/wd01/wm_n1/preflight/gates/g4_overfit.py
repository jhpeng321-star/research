"""G4 ``overfit``.

A single member must drive the total loss (including the contrast and rollout
terms) near zero on 2 tiny TRAIN windows, on CPU.  The report shows the
threshold and the loss curve.

Entry point (P7): ``train.idea_a.overfit_report()`` returning::

    {"ok": bool, "device": "cpu", "n_windows": 2, "threshold": 1e-2,
     "final_loss": float, "loss_curve": [float, ...]}
"""
from __future__ import annotations

import math

from ..imports import PieceMissing
from ..result import fail, ok

NAME = "overfit"
TITLE = "G4 overfit (1 member drives total loss near 0 on 2 tiny TRAIN windows)"
ENTRY_POINTS = ("train.idea_a.overfit_report",)
MIN_WINDOWS = 2


def evaluate(report):
    if not isinstance(report, dict):
        return False, f"report is {type(report).__name__}, expected dict", {}
    curve = report.get("loss_curve") or []
    try:
        final = float(report.get("final_loss", float("inf")))
        threshold = float(report.get("threshold", float("-inf")))
    except (TypeError, ValueError):
        final, threshold = float("inf"), float("-inf")
    n_windows = int(report.get("n_windows", 0))
    device = str(report.get("device", "")).lower()
    finite = all(isinstance(x, (int, float)) and math.isfinite(x)
                 for x in curve)
    checks = {"device": device, "n_windows": n_windows,
              "threshold": threshold, "final_loss": final,
              "curve_len": len(curve), "curve_finite": bool(finite)}
    problems = []
    if device and device != "cpu":
        problems.append(f"device {device!r} is not cpu")
    if n_windows < MIN_WINDOWS:
        problems.append(f"only {n_windows} window(s) < {MIN_WINDOWS}")
    if not curve:
        problems.append("empty loss curve")
    elif not finite:
        problems.append("loss curve has non-finite values")
    if final > threshold:
        problems.append(f"final loss {final:g} > threshold {threshold:g}")
    passed = bool(report.get("ok")) and not problems
    detail = (f"final {final:g} <= threshold {threshold:g} on {n_windows} "
              "windows (cpu)") if passed else "; ".join(problems)
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
