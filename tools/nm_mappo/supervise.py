"""N7 supervisor (Linux training box): registered smoke / production / resume of the wz3to2
speed-only MAPPO, wall cap, exact own-PID receipts.  Adapted from S-MAPPOSPEED supervise.py.

    setsid nohup nice -n 19 python -u tools/nm_mappo/supervise.py {smoke|production|resume} RUN_DIR

Started by ``launch.py smoke|run|resume``.  Re-verifies PLAN.json <-> REGISTERED.json, the
config and init sha256, the optional code-bundle SOURCE_MANIFEST.json, nice 19 and the smoke
gate; raises the open-file limit (the v5/S-MAPPOSPEED requirement: >= 65536); runs the exact
planned trainer argv with one numerical thread per worker and CUDA hidden.  ``RUN_DIR/STOP``
or the registered wall cap (minus 120 s) writes ``<stage>/STOP_REQUESTED`` (clean stop at the
next episode / minibatch, resumable); at the cap itself only the PIDs recorded under OUR trainer
(pid + start ticks, pidfd) are killed.  Never signals by name.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_launch():
    import importlib.util
    if "nm_mappo_launch" in sys.modules:
        return sys.modules["nm_mappo_launch"]
    spec = importlib.util.spec_from_file_location("nm_mappo_launch", HERE / "launch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nm_mappo_launch"] = module
    spec.loader.exec_module(module)
    return module


L = _load_launch()

STOP_LEAD_S = 120


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def identity(pid: int) -> dict:
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 2:].split()
    return {"pid": pid, "start_ticks": int(fields[19]), "ppid": int(fields[1]),
            "session": int(fields[3]), "state": fields[0]}


def descendants(pid: int) -> list[dict]:
    found = []
    try:
        ids = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except FileNotFoundError:
        return found
    for text in ids:
        try:
            child = identity(int(text))
            found.append(child)
            found.extend(descendants(child["pid"]))
        except FileNotFoundError:
            pass
    return found


def _syscall(number, *args):
    import platform
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("pidfd fallback requires Linux x86_64/aarch64")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        error = ctypes.get_errno()
        if error == errno.ESRCH:
            raise ProcessLookupError(error, os.strerror(error))
        raise OSError(error, os.strerror(error))
    return result


def pidfd_open(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    return _syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_send_signal(fd, sig):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(fd, sig)
    return _syscall(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))


def exact_signal(record: dict, sig) -> None:
    """pidfd pins the incarnation; never signal a reused PID or a process by name."""
    try:
        fd = pidfd_open(record["pid"])
    except ProcessLookupError:
        return
    try:
        if identity(record["pid"])["start_ticks"] == record["start_ticks"]:
            pidfd_send_signal(fd, sig)
    except (FileNotFoundError, ProcessLookupError):
        pass
    finally:
        os.close(fd)


def raise_fd_limit():
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(65536, hard) if hard != resource.RLIM_INFINITY else 65536
    if target < 65536:
        raise RuntimeError(f"open-file hard limit {hard} is below the v5 launcher requirement")
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, target), hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)


def verify_bundle() -> int:
    manifest = L.CODE / "SOURCE_MANIFEST.json"
    if not manifest.exists():
        return 0
    entries = json.loads(manifest.read_text())
    for name, digest in entries.items():
        if L.sha256_text_lf(L.CODE / name) != digest:
            raise SystemExit(f"code bundle drift: {name}")
    return len(entries)


def trainer_env(plan: dict) -> dict:
    env = dict(os.environ, **plan["env"])
    env["PYTHONPATH"] = f"{L.CODE / 'src'}:{L.CODE}"
    for key in ("C4_EXECUTION_PARITY", "C4_POLICY_STOCHASTIC", "C4_POLICY_ACCEL_NOISE"):
        env.pop(key, None)                   # standard MAPPO execution, as the original
    if plan.get("sumo_home"):
        env["SUMO_HOME"] = plan["sumo_home"]
        env["PATH"] = f"{Path(plan['sumo_home']) / 'bin'}:{Path(plan['python']).parent}:{env['PATH']}"
    return env


def main() -> int:
    stage, run_dir = sys.argv[1], Path(sys.argv[2])
    assert stage in ("smoke", "production", "resume"), stage
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19, "start with nice -n 19"
    fd_limits = raise_fd_limit()
    checked = L.verify_plan(run_dir)
    plan = checked["plan"]
    files = verify_bundle()
    target = "smoke" if stage == "smoke" else "production"
    out = run_dir / target
    if stage == "production":
        ok, why = L.smoke_passed(run_dir)
        assert ok, f"smoke gate: {why}"
        assert not out.exists(), "fresh production only (use resume)"
    elif stage == "smoke":
        assert not out.exists(), "fresh smoke only"
    else:
        assert (out / "checkpoints").is_dir(), "resume needs production checkpoints"
    assert not (run_dir / "STOP").exists(), "remove RUN_DIR/STOP deliberately before (re)starting"
    out.mkdir(exist_ok=stage == "resume")
    start_path = run_dir / "WALL_START.json"
    if not start_path.exists():
        L.write_json_new(start_path, {"unix": time.time(), "utc": utc()})
    deadline = json.loads(start_path.read_text())["unix"] + int(plan["wall_cap_s"])
    assert time.time() < deadline - STOP_LEAD_S, "registered wall cap already reached"
    argv = list(plan["argv"][target])
    if stage == "resume":
        argv = L.resume_argv(argv)
    log_name = f"train_{stage}_{int(time.time())}.log"
    with (out / log_name).open("x") as log:
        trainer = subprocess.Popen(argv, cwd=L.CODE, env=trainer_env(plan), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        owner = identity(trainer.pid)
        L.write_json_new(out / f"OWNERSHIP_{stage}_{owner['pid']}.json", {
            "stage": stage, "supervisor": identity(os.getpid()), "trainer": owner, "command": argv,
            "started_utc": utc(), "deadline_unix": deadline, "plan_sha256": checked["plan_sha256"],
            "registration_sha256": checked["registered"].get("registration_sha256"),
            "bundle_files_verified": files, "nice": 19, "nofile_limits": fd_limits})
        known = {(owner["pid"], owner["start_ticks"]): owner}
        stop_sent = False
        with (out / f"OWN_PIDS_{stage}_{owner['pid']}.jsonl").open("x") as pidlog:
            while trainer.poll() is None:
                for child in descendants(trainer.pid):
                    key = (child["pid"], child["start_ticks"])
                    if key not in known:
                        known[key] = child
                        pidlog.write(json.dumps({"utc": utc(), **child}) + "\n")
                        pidlog.flush()
                if not stop_sent and ((run_dir / "STOP").exists()
                                      or time.time() >= deadline - STOP_LEAD_S):
                    with (out / "STOP_REQUESTED").open("a") as fh:
                        fh.write("supervisor: STOP or registered wall cap\n")
                    stop_sent = True
                if time.time() >= deadline:
                    for child in reversed(list(known.values())):
                        exact_signal(child, signal.SIGKILL)
                    break
                time.sleep(2)
            rc = trainer.wait()
    if rc not in (0, 3):
        for child in reversed(list(known.values())):
            exact_signal(child, signal.SIGTERM)
    L.write_json_new(out / f"EXIT_{stage}_{owner['pid']}.json", {
        "rc": rc, "finished_utc": utc(), "stop_sent": stop_sent,
        "wall_s_since_start": time.time() - json.loads(start_path.read_text())["unix"]})
    return rc


if __name__ == "__main__":
    sys.exit(main())
