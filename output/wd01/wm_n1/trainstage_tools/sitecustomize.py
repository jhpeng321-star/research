"""Fail-closed sitecustomize bootstrap for the pinned TRAIN-stage lineage.

Runs before any user import when ``WM_TRAINSTAGE_REQUIRE_GUARD=1``; the
guard is installed and self-checked before numpy/pandas/pyarrow or any
stage module can load, and every spawned worker that inherits the pinned
environment re-attests through this same path.  Failure exits immediately;
there is no repair path.
"""
import os
import sys

WM_TRAINSTAGE_BOOT_PID = None
if os.environ.get("WM_TRAINSTAGE_REQUIRE_GUARD") == "1":
    try:
        import train_stage_guard
        train_stage_guard.install_and_selfcheck()
        WM_TRAINSTAGE_BOOT_PID = os.getpid()
    except BaseException as exc:
        try:
            sys.stderr.write("[wm-trainstage] startup refused: %r\n" % (exc,))
            sys.stderr.flush()
        finally:
            os._exit(87)
