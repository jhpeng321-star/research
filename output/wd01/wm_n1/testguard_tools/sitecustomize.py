"""Fail-closed sitecustomize bootstrap for a pinned guarded Python lineage."""
import os
import sys

WM_TESTGUARD_BOOT_PID = None
if os.environ.get("WM_TESTGUARD_REQUIRE_GUARD") == "1":
    try:
        import wm_testguard
        wm_testguard.install_and_selfcheck()
        WM_TESTGUARD_BOOT_PID = os.getpid()
    except BaseException as exc:
        try:
            sys.stderr.write("[wm-testguard] startup refused: %r\n" % (exc,))
            sys.stderr.flush()
        finally:
            os._exit(87)
