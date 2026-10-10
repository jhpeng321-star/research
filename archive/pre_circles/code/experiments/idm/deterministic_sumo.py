"""Race-free SUMO launching (Stage 1b determinism gate, A1/A2).

Observed failure mode (2026-09-03): under heavy concurrent launches
(8+ workers per chain, sometimes two chains), SUMO occasionally died with
'tcpip::Socket::accept() Unable to create listening socket: Address already
in use'. Root cause: ``env.sumo_utils.find_free_port`` binds an ephemeral
port, CLOSES the socket, and returns the port; SUMO later binds it. Two
workers can draw the same port within that window (Windows glibc-style
ephemeral-port reuse is not serialized), and one SUMO loses the bind race.

Fix (new code, no frozen-file edits): patch ``env.sumo_utils.start_sumo``
with a version that holds an inter-process machine lock (``msvcrt.locking``
on a lock file in the probe directory) across the whole
find-free-port -> traci.start window. ``traci.start`` returns only after
SUMO has bound the port and accepted the connection, so holding the lock
until then makes the reservation effective. This is installed at import
time (``install()``) and must be imported before ``env.core`` (which takes a
module-level reference to the original function); the generation runner
imports this module first for this reason.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

_LOCK_PATH = Path(os.environ.get("ZCODE_SUMO_LOCK", str(Path(__file__).parents[2] / "runs" / "probe" / ".sumo_launch.lock")))


class _InterprocessLaunchLock:
    def __enter__(self) -> "_InterprocessLaunchLock":
        _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(_LOCK_PATH, "a+")
        self._handle.seek(0)
        if self._handle.tell() == 0:
            self._handle.write("\0")
            self._handle.flush()
        import msvcrt

        msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
        return self

    def __exit__(self, *_exc: Any) -> None:
        import msvcrt

        try:
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._handle.close()


def _guarded_start_sumo(
    sumo_binary: str,
    sumo_config: str,
    project_root: Path,
    port: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
    max_retries: int = 16,
    seed: Optional[int] = None,
) -> int:
    import socket

    import traci

    from env.sumo_utils import close_sumo

    sumo_cmd = [sumo_binary, "-c", str(project_root / sumo_config)]
    if seed is not None:
        sumo_cmd += ["--seed", str(seed)]
    if extra_args:
        sumo_cmd.extend(extra_args)

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        close_sumo()
        with _InterprocessLaunchLock():
            use_port = port if (port is not None and attempt == 0) else _find_free_port_locked()
            try:
                traci.start(sumo_cmd, port=use_port)
                return use_port
            except Exception as exc:  # noqa: BLE001 - retry on TraCI/port races
                last_err = exc
                close_sumo()
                port = None
    raise RuntimeError(f"Failed to start SUMO after {max_retries} tries") from last_err


def _find_free_port_locked() -> int:
    # Called while the launch lock is held: the bind-to-0 probe is still
    # enough, and the SUMO bind happens before the lock is released.
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def install() -> None:
    """Replace env.sumo_utils.start_sumo with the guarded version."""
    import env.sumo_utils as su

    su.start_sumo = _guarded_start_sumo  # type: ignore[assignment]


install()
