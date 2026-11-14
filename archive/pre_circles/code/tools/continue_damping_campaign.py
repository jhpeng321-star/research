"""Serialize only the registered damping stages; stop on any failed stage.

Can adopt one already running Windows trainer by verified PID and command.
Does not launch acceptance, candidate selection changes, walls, or predictors.
"""

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.idm.train_native_damping import atomic_json, exclusive_run
from idm.damping_campaign import CHECKPOINT_EPISODES, OPTIMIZER_SEEDS

ROOT = Path(__file__).resolve().parents[1]
MODULE = "experiments.idm.train_damping_campaign"
STAGES = tuple((seed, episode) for seed in OPTIMIZER_SEEDS for episode in CHECKPOINT_EPISODES)


def read(path):
    return json.loads(path.read_text())


def stage_complete(root, seed, episode):
    path = root / f"lineage_{seed}" / "stage_summary.json"
    evaluation = root / "development" / f"optimizer_{seed}_episode_{episode:03}" / "summary.json"
    if not path.exists() or not evaluation.exists():
        return False
    stage, record = read(path), read(evaluation)
    return (
        stage.get("status") == "stage_complete"
        and stage.get("optimizer_seed") == seed
        and stage.get("optimized_episodes", -1) >= episode
        and record.get("optimizer_seed") == seed
        and record.get("optimized_global_steps") == episode * 2400
    )


def verify_sources(root):
    for name, expected in read(root / "manifest.json")["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"registered training source changed: {name}")


def verify_command(command, root, initialization, stage):
    args = [token.strip('"') for token in shlex.split(command, posix=False)]

    def value(flag):
        if args.count(flag) != 1:
            raise ValueError(f"foreign trainer command is missing/duplicates {flag}")
        return args[args.index(flag) + 1]

    if (
        Path(args[0]).resolve() != Path(sys.executable).resolve()
        or value("-m") != MODULE
        or Path(value("--output")).resolve() != root
        or Path(value("--initialization")).resolve() != initialization
        or int(value("--optimizer-seed")) != stage[0]
        or int(value("--through-episode")) != stage[1]
    ):
        raise ValueError("PID does not identify the expected campaign trainer")


def wait_existing(pid, root, initialization, stage, heartbeat):
    if os.name != "nt":
        raise RuntimeError("foreign-process adoption is implemented for Windows only")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        raise RuntimeError(
            "requested trainer PID is missing or inaccessible; inspect before resuming"
        )
    try:
        # PID is numeric; no user strings or paths are interpolated into shell code.
        info = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                f"Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}' | ConvertTo-Json -Compress",
            ],
            check=True,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        process = json.loads(info.stdout)
        verify_command(process["CommandLine"], root, initialization, stage)
        heartbeat("waiting_existing", stage, pid)
        while True:
            result = kernel.WaitForSingleObject(handle, 30000)
            if result == 0:
                code = wintypes.DWORD()
                if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                    raise OSError(ctypes.get_last_error(), "cannot read trainer exit code")
                return code.value
            if result != 258:  # WAIT_TIMEOUT keeps this exact process handle alive.
                raise OSError(ctypes.get_last_error(), "trainer process wait failed")
            heartbeat("waiting_existing", stage, pid)
    finally:
        kernel.CloseHandle(handle)


def run_stage(root, initialization, stage, heartbeat):
    seed, episode = stage
    command = [
        sys.executable,
        "-m",
        MODULE,
        "--output",
        str(root),
        "--initialization",
        str(initialization),
        "--optimizer-seed",
        str(seed),
        "--through-episode",
        str(episode),
        "--resume",
    ]
    with subprocess.Popen(
        command, cwd=ROOT, creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    ) as child:
        heartbeat("running_child", stage, child.pid)
        while True:
            try:
                return child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                heartbeat("running_child", stage, child.pid)


def continue_campaign(root, initialization, wait_pid, heartbeat):
    verify_sources(root)
    active = read(root / "active_stage.json")
    if wait_pid is not None:
        stage = (active["optimizer_seed"], active["through_episode"])
        if (
            active.get("pid") != wait_pid
            or active.get("status") != "running"
            or stage not in STAGES
        ):
            raise ValueError("requested adoption disagrees with active trainer state")
        code = wait_existing(wait_pid, root, initialization, stage, heartbeat)
        if (
            code
            or read(root / "active_stage.json").get("status") != "complete"
            or not stage_complete(root, *stage)
        ):
            raise RuntimeError(f"adopted training stage failed or is incomplete (exit {code})")
    elif active.get("status") != "complete":
        raise RuntimeError(
            "active stage is not complete; verify/adopt its live PID or inspect failure"
        )
    for stage in STAGES:
        if stage_complete(root, *stage):
            continue
        verify_sources(root)
        print(json.dumps(dict(event="starting_registered_stage", stage=stage)), flush=True)
        code = run_stage(root, initialization, stage, heartbeat)
        if (
            code
            or read(root / "active_stage.json").get("status") != "complete"
            or not stage_complete(root, *stage)
        ):
            raise RuntimeError(f"training stage {stage} failed or is incomplete (exit {code})")
    heartbeat("all_registered_stages_complete", None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    root, initialization = args.campaign.resolve(), args.initialization.resolve()
    if not root.is_dir() or not initialization.is_dir():
        raise ValueError("existing campaign and initialization directories are required")
    directory = root / "continuation"
    with exclusive_run(directory):

        def heartbeat(status, stage, child_pid):
            atomic_json(
                directory / "state.json",
                dict(
                    status=status,
                    stage=stage,
                    coordinator_pid=os.getpid(),
                    trainer_pid=child_pid,
                    updated_at_utc=datetime.now(timezone.utc).isoformat(),
                    accepted_checkpoint=False,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                ),
            )

        try:
            continue_campaign(root, initialization, args.wait_pid, heartbeat)
        except BaseException as exc:
            heartbeat("failed", None, None)
            atomic_json(directory / "failure.json", dict(error=repr(exc)))
            raise


if __name__ == "__main__":
    main()
