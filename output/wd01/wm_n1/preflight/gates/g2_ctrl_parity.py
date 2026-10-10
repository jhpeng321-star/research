"""G2 ``ctrl-parity``.

The P1 CAV mover must reproduce the recorded SUMO CAV trajectories (teacher
data) on >= 2 TRAIN windows for each of native, damping, PI and wall v2.1 (the
J4 law), and must support BOTH wall v2.1 (J4 parity) and v2.2 (the new rule).

Entry point (P1): ``ctrl.parity.strategy_parity_report()`` returning::

    {"ok": bool, "tolerance": 1e-3, "windows_per_strategy": 2,
     "strategies": {"native": {"max_error": ..., "n_windows": ...}, ...},
     "new_rule": "wall-v2.2"}
"""
from __future__ import annotations

from ..imports import PieceMissing
from ..result import fail, ok

NAME = "ctrl-parity"
TITLE = "G2 ctrl-parity (P1 replays recorded CAV motion, native/damping/PI/wall)"
ENTRY_POINTS = ("ctrl.parity.strategy_parity_report",)
REQUIRED_STRATEGIES = ("native", "damping", "pi", "wall-v2.1")
NEW_RULE_STRATEGIES = ("wall-v2.2",)
DEFAULT_TOLERANCE = 1e-3
MIN_TRAIN_WINDOWS = 2


def _max_error(entry) -> float:
    if isinstance(entry, dict):
        for key in ("max_error", "max_abs_error", "error"):
            if key in entry:
                return float(entry[key])
        return float("inf")
    return float(entry)


def evaluate(report, *, tolerance: float = DEFAULT_TOLERANCE,
             min_windows: int = MIN_TRAIN_WINDOWS):
    if not isinstance(report, dict):
        return False, f"report is {type(report).__name__}, expected dict", {}
    strategies = report.get("strategies") or {}
    tol = float(report.get("tolerance", tolerance))
    checks = {"tolerance": tol, "strategies": sorted(strategies)}
    missing = [s for s in REQUIRED_STRATEGIES if s not in strategies]
    if missing:
        return False, f"missing strategy parity reports: {missing}", checks
    checks["new_rule"] = "wall-v2.2" in strategies
    bad = []
    for s in REQUIRED_STRATEGIES:
        entry = strategies[s]
        n = int(entry.get("n_windows", 0)) if isinstance(entry, dict) else 0
        err = _max_error(entry)
        checks[f"{s}_max_error"] = err
        checks[f"{s}_n_windows"] = n
        if n < min_windows:
            bad.append(f"{s}: only {n} TRAIN window(s) < {min_windows}")
        if err > tol:
            bad.append(f"{s}: error {err:g} > tol {tol:g}")
    if not checks["new_rule"]:
        bad.append("wall-v2.2 (the new rule) is not supported by P1")
    passed = bool(report.get("ok")) and not bad
    detail = "all strategies within tolerance on >= 2 windows" if passed \
        else "; ".join(bad)
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
