"""lower_priority_below_normal on 64-bit Windows (FEAT1 blocker 2).

The classic defect: ``GetCurrentProcess`` returns a 64-bit pseudo-handle
(-1) which the default ``c_int`` restype truncates, so ``SetPriorityClass``
fails with ``GetLastError: 6`` and the helper always returned False on
64-bit Python.  The fix prefers psutil and otherwise declares the proper
``c_void_p`` restype.
"""
import os

from corpus import _util as U

BELOW_NORMAL = 0x00004000


def test_returns_false_off_windows():
    if os.name != "nt":
        assert U.lower_priority_below_normal() is False


def test_sets_and_restores_below_normal():
    if os.name != "nt":
        return
    before = U.current_priority_class()
    assert before != 0
    try:
        assert U.lower_priority_below_normal() is True
        assert U.current_priority_class() == BELOW_NORMAL
    finally:
        _restore(before)
    assert U.current_priority_class() == before


def _restore(priority: int) -> None:
    if priority == BELOW_NORMAL or priority == 0:
        return
    try:
        import psutil
    except ImportError:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p,
                                              ctypes.c_uint32)
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), priority)
        return
    psutil.Process().nice(priority)
