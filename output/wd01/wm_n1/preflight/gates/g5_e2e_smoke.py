"""G5 ``e2e-smoke``.

A 60 s J4 TRAIN world through the GPU-queue commands: convert -> timing (30 s)
-> train (a few steps) -> checkpoint -> resume -> eval, all on CPU.  Also runs
the torch >=2.7,<2.13 compatibility check (static API scan + a
``torch.__version__``-gated op test; see :mod:`preflight.torch_compat`).

Entry point (bundle e2e): ``bundle.e2e.run_e2e_smoke()`` returning::

    {"ok": bool, "device": "cpu", "stages": ["convert", "timing", "train",
     "checkpoint", "resume", "eval"], "timing_seconds": 30.0,
     "train_steps": int, "n_worlds": int}
"""
from __future__ import annotations

from .. import torch_compat
from ..imports import PieceMissing
from ..result import fail, ok

NAME = "e2e-smoke"
TITLE = ("G5 e2e-smoke (convert->timing->train->ckpt->resume->eval, CPU, "
         "torch>=2.7,<2.13)")
ENTRY_POINTS = ("bundle.e2e.run_e2e_smoke",)
STAGES = ("convert", "timing", "train", "checkpoint", "resume", "eval")
MIN_TIMING_SECONDS = 30.0


def evaluate(report, torch_check):
    if not isinstance(report, dict):
        return False, f"report is {type(report).__name__}, expected dict", {}
    stages = [str(s) for s in (report.get("stages") or [])]
    device = str(report.get("device", "")).lower()
    try:
        timing_s = float(report.get("timing_seconds", -1.0))
    except (TypeError, ValueError):
        timing_s = -1.0
    missing = [s for s in STAGES if s not in stages]
    checks = {"device": device, "stages": stages, "missing_stages": missing,
              "timing_seconds": timing_s, "torch": torch_check}
    problems = []
    if device and device != "cpu":
        problems.append(f"device {device!r} is not cpu")
    if missing:
        problems.append(f"missing queue stage(s): {missing}")
    if timing_s < MIN_TIMING_SECONDS:
        problems.append(f"timing stage {timing_s:g}s < {MIN_TIMING_SECONDS:g}s")
    if not report.get("ok"):
        problems.append("e2e report ok is false")
    if not torch_check.get("ok"):
        problems.append("torch>=2.7,<2.13 compatibility check failed")
    passed = not problems
    detail = ("queue e2e stages complete on CPU; torch scan clean ("
              f"version gate: {torch_check.get('version_gate', {}).get('status')})"
              ) if passed else "; ".join(problems)
    return passed, detail, checks


def run(settings, resolve):
    try:
        torch_check = torch_compat.scan_and_check(settings.wm_n1_dir)
    except Exception as e:                                # noqa: BLE001
        torch_check = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    try:
        fn = resolve(ENTRY_POINTS[0])
    except PieceMissing as e:
        return fail(NAME, TITLE, str(e), ENTRY_POINTS,
                    checks={"torch": torch_check}, error="piece-missing")
    try:
        report = fn()
    except Exception as e:                                # noqa: BLE001
        return fail(NAME, TITLE, f"{ENTRY_POINTS[0]} raised "
                    f"{type(e).__name__}: {e}", ENTRY_POINTS,
                    checks={"torch": torch_check}, error="gate-error")
    passed, detail, checks = evaluate(report, torch_check)
    return (ok if passed else fail)(NAME, TITLE, detail, ENTRY_POINTS, checks)
