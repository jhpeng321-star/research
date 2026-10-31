"""Explicit SUMO lifecycle; byte-zero launch lock replaces historical monkeypatch.

Extracted from env/sumo_utils.py and experiments/idm/deterministic_sumo.py.
Only startup is serialized. Independent simulations use separate TraCI ports
and independent Python processes. This module never changes vehicle safety.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import socket

from .research_assets import resolve_research_path, sumo_input_overrides

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def launch_lock(path=None):
    path = Path(path) if path is not None else ROOT / "runs/active/.sumo_launch.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        # Never append on subsequent entries: all processes lock the same byte.
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def close_sumo(api=None):
    if api is None:
        import traci as api
    if api.isLoaded():
        api.close()


def start_sumo(sumo_binary, scene, *, seed, config=None, api=None, lock_path=None,
               extra_cli=None, no_step_log=True, command_record=None):
    from traffic_shepherd.config import load_config
    config = config if config is not None else load_config()
    if api is None:
        import traci as api
    if api.isLoaded():
        raise RuntimeError("a TraCI connection is already active in this process")
    scene = resolve_research_path(scene)
    command = [str(sumo_binary), "-c", str(Path(scene).resolve()), "--seed", str(seed),
               "--step-length", str(config.physics_dt_s), "--end", str(config.episode_seconds + 1),
               "--no-step-log", "true" if no_step_log else "false",
               "--duration-log.disable", "true"]
    command.extend(sumo_input_overrides(scene))
    if extra_cli:
        command.extend(str(a) for a in extra_cli)
    if command_record is not None:
        command_record.append(command)
    with launch_lock(lock_path):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        # Keep the interprocess lock until SUMO binds and TraCI connects.
        # No silent simulation retry: callers retain their interrupted-run ledger.
        try:
            api.start(command, port=port)
        except Exception:
            # The connection was absent before this launch; clean only a partial
            # connection created by our own failed api.start call.
            if api.isLoaded():
                api.close()
            raise
    return port
