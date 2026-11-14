"""Free-memory gate for long featurize-all runs.

The owner requires the featurize-all driver to check available memory before it
starts and between worker batches, and to pause (not crash) while the box is
under pressure.  The gate is probe-injectable so tests can mock it.

Defaults: **10 GB** free virtual memory (``FREE_VIRTUAL_GB``); RAM is optional.
On Windows the probe uses ``GlobalMemoryStatusEx``; elsewhere it falls back to
``os.sysconf`` when available.
"""
from __future__ import annotations

import time

DEFAULT_PAUSE_S = 30.0
DEFAULT_MAX_WAIT_S = 3600.0
FREE_VIRTUAL_GB = 10.0
FREE_RAM_GB = 0.0
_GIB = 1024 ** 3

# Dynamic worker gate (FEAT1 v2 speed-up, owner decision via main): a run
# above the classic 6-worker hard max (e.g. 12 workers) drops to
# ``DYNAMIC_FALLBACK_WORKERS`` for the affected batch while the box is under
# memory pressure (free virtual < 8 GB or free RAM < 6 GB).
DYNAMIC_VIRTUAL_GB = 8.0
DYNAMIC_RAM_GB = 6.0
DYNAMIC_FALLBACK_WORKERS = 8


class MemoryGateError(RuntimeError):
    """The free-memory gate could not be satisfied before the deadline."""


def _windows_probe() -> dict:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        raise OSError("GlobalMemoryStatusEx failed")
    return dict(
        free_virtual_gb=stat.ullAvailVirtual / _GIB,
        free_ram_gb=stat.ullAvailPhys / _GIB,
        total_ram_gb=stat.ullTotalPhys / _GIB,
        free_pagefile_gb=stat.ullAvailPageFile / _GIB,
    )


def _posix_probe() -> dict:
    import os

    free_ram_gb = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") \
        / _GIB
    total_ram_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") \
        / _GIB
    return dict(free_virtual_gb=free_ram_gb, free_ram_gb=free_ram_gb,
                total_ram_gb=total_ram_gb, free_pagefile_gb=free_ram_gb)


def free_memory(probe=None) -> dict:
    """Return ``{"free_virtual_gb", "free_ram_gb", ...}`` (probe-injectable)."""
    if probe is not None:
        return dict(probe())
    import os

    if os.name == "nt":
        return _windows_probe()
    try:
        return _posix_probe()
    except (ValueError, OSError, AttributeError):  # pragma: no cover
        return dict(free_virtual_gb=float("inf"), free_ram_gb=float("inf"),
                    total_ram_gb=float("inf"), free_pagefile_gb=float("inf"))


def _satisfied(sample: dict, min_free_virtual_gb, min_free_ram_gb) -> bool:
    if min_free_virtual_gb is not None \
            and sample.get("free_virtual_gb", 0.0) < min_free_virtual_gb:
        return False
    if min_free_ram_gb is not None \
            and sample.get("free_ram_gb", 0.0) < min_free_ram_gb:
        return False
    return True


def require_memory(min_free_virtual_gb=None, min_free_ram_gb=None, *,
                   probe=None, pause_s: float = DEFAULT_PAUSE_S,
                   max_wait_s: float = DEFAULT_MAX_WAIT_S,
                   sleep=time.sleep) -> dict:
    """Block (pausing) until the memory thresholds are met, else raise.

    ``min_free_virtual_gb`` defaults to :data:`FREE_VIRTUAL_GB` (10 GB) and
    ``min_free_ram_gb`` defaults to :data:`FREE_RAM_GB` (disabled).  A
    ``MemoryGateError`` is raised only after ``max_wait_s`` of waiting.
    """
    if min_free_virtual_gb is None:
        min_free_virtual_gb = FREE_VIRTUAL_GB
    if min_free_ram_gb is None:
        min_free_ram_gb = FREE_RAM_GB

    deadline = time.monotonic() + max(0.0, float(max_wait_s))
    while True:
        sample = free_memory(probe)
        sample["min_free_virtual_gb"] = float(min_free_virtual_gb)
        sample["min_free_ram_gb"] = float(min_free_ram_gb)
        if _satisfied(sample, min_free_virtual_gb, min_free_ram_gb):
            sample["waited"] = False
            return sample
        if time.monotonic() >= deadline:
            raise MemoryGateError(
                f"free-virtual-memory gate not met: "
                f"{sample.get('free_virtual_gb'):.2f} GB free < "
                f"{min_free_virtual_gb:.2f} GB required after "
                f"{max_wait_s:.0f}s")
        sample["waited"] = True
        sleep(pause_s)


def wait_for_memory(min_free_virtual_gb=None, min_free_ram_gb=None, *,
                    probe=None, pause_s: float = DEFAULT_PAUSE_S,
                    max_wait_s: float = DEFAULT_MAX_WAIT_S, sleep=time.sleep
                    ) -> dict:
    """Between-batch alias of :func:`require_memory` (pause while low)."""
    return require_memory(min_free_virtual_gb, min_free_ram_gb, probe=probe,
                          pause_s=pause_s, max_wait_s=max_wait_s, sleep=sleep)


def dynamic_worker_gate(sample, *, workers,
                        threshold_virtual_gb=DYNAMIC_VIRTUAL_GB,
                        threshold_ram_gb=DYNAMIC_RAM_GB,
                        fallback=DYNAMIC_FALLBACK_WORKERS):
    """Return ``(effective_workers, dropped)`` for one memory sample.

    Runs at or below ``fallback`` workers are returned unchanged.  Above it,
    a sample below either threshold (free virtual < 8 GB or free RAM < 6 GB)
    drops the batch to ``fallback`` workers; the caller must log the drop.
    """
    workers = int(workers)
    fallback = int(fallback)
    if workers <= fallback:
        return workers, False
    low_virtual = (float(sample.get("free_virtual_gb", 0.0))
                   < float(threshold_virtual_gb))
    low_ram = float(sample.get("free_ram_gb", 0.0)) < float(threshold_ram_gb)
    if low_virtual or low_ram:
        return fallback, True
    return workers, False
