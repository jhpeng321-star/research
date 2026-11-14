"""Test-only Windows stand-in for ``fcntl.flock`` (HARDEN-3 CPU rehearsal).

The live hosts are Linux and import the real ``fcntl``; this module is installed
into ``sys.modules`` ONLY when ``fcntl`` is missing (local Windows CPU tests and
the end-to-end rehearsal). Exclusive locks use ``msvcrt.locking`` on byte 0 of
the claim file, so they serialize separate handles across threads and processes
like ``flock`` does on Linux. Shared locks are treated as exclusive (stricter).
"""
import os
import sys
import time

LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 4, 8


def _fd(f):
    return f if isinstance(f, int) else f.fileno()


def flock(f, op):
    import msvcrt
    fd = _fd(f)
    os.lseek(fd, 0, os.SEEK_SET)
    if op & LOCK_UN:
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    while True:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            if op & LOCK_NB:
                raise BlockingIOError('lock held')
            time.sleep(.005)


def install():
    """Install only when the platform lacks fcntl; returns True if installed."""
    try:
        import fcntl  # noqa: F401
        return False
    except ModuleNotFoundError:
        sys.modules['fcntl'] = sys.modules[__name__]
        return True
