"""Stop verified Windows trainers after their next atomic complete checkpoint."""

from concurrent.futures import ThreadPoolExecutor
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from experiments.idm.train_native_damping import atomic_json, exclusive_run
from tools.continue_damping_campaign import read, verify_command

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "runs/rllib/native_damping_campaign_v1"
INITIALIZATION = ROOT / "runs/rllib/native_damping_v2_credit_repair_o71"
OUTPUT = CAMPAIGN / "parallel_execution/checkpoint_safe_pause"


def process_info(pid):
    command = (f"Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}' | "
               "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                            capture_output=True, text=True, check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    return json.loads(result.stdout) if result.stdout.strip() else None


def pause_worker(seed):
    state = read(CAMPAIGN / "parallel_execution" / f"optimizer_{seed}.json")
    worker, pid = Path(state["worker_root"]), state["trainer_pid"]
    info = process_info(pid)
    if info is None:
        raise RuntimeError(f"trainer {seed} is already terminal; inspect before acting")
    verify_command(info["CommandLine"], worker, INITIALIZATION, tuple(state["stage"]))
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtSuspendProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtSuspendProcess.restype = ctypes.c_long
    handle = kernel.OpenProcess(0x100000 | 0x1000 | 0x800 | 0x1, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), "cannot open verified trainer")
    directory = worker / f"lineage_{seed}"
    latest = max(directory.glob("trainer_*.pt"))
    begin = int(latest.stem.rsplit("_", 1)[1])
    target = directory / f"trainer_{begin+1:03}.pt"
    record = dict(optimizer_seed=seed, pid=pid, previous_checkpoint=str(latest),
                  target_checkpoint=str(target), target_episode=begin+1,
                  status="waiting_next_complete_checkpoint", accepted_checkpoint=False)
    status_path = OUTPUT / f"optimizer_{seed}.json"
    atomic_json(status_path, record)
    print(json.dumps(record), flush=True)
    heartbeat = time.monotonic()
    try:
        while not target.exists():
            if kernel.WaitForSingleObject(handle, 0) == 0:
                raise RuntimeError("trainer exited before next checkpoint; inspect failure")
            if time.monotonic()-heartbeat >= 30:
                atomic_json(status_path, {**record, "updated_at_utc": datetime.now(timezone.utc).isoformat()})
                heartbeat = time.monotonic()
            time.sleep(0.01)
        # The trainer publishes .pt by atomic rename after torch.save completes.
        if ntdll.NtSuspendProcess(handle) < 0:
            raise RuntimeError("could not suspend trainer at checkpoint")
        record["status"] = "suspended_for_checkpoint_verification"
        atomic_json(status_path, record)
        payload = torch.load(target, map_location="cpu", weights_only=True)
        episode = begin+1
        trajectory_path = directory / f"policy_{episode-1:03}.json"
        trajectory = read(trajectory_path)
        if (payload["iteration"] != episode or payload["global_steps"] != episode*2400
                or payload["identity"] != {**read(worker/"manifest.json"), "optimizer_seed":seed}
                or payload["history"][-1]["summary"] != trajectory["summary"]
                or not trajectory["summary"]["team_terminal"]
                or len(trajectory["intervals"]) != 2400
                or not trajectory_path.with_suffix(".npz").exists()):
            raise RuntimeError("checkpoint verification failed; trainer stays suspended")
        if not all(torch.isfinite(value).all() for part in ("actor", "critic") for value in payload[part].values()):
            raise RuntimeError("checkpoint has nonfinite weights; trainer stays suspended")
        partial = []
        for path in directory.glob("policy_*.started.json"):
            if not path.with_name(path.name.replace(".started.json", ".json")).exists():
                partial.append(dict(path=str(path), allocation=read(path)))
        if not kernel.TerminateProcess(handle, 0):
            raise OSError(ctypes.get_last_error(), "could not stop verified checkpointed trainer")
        kernel.WaitForSingleObject(handle, 5000)
        record.update(status="stopped_at_verified_checkpoint", checkpoint_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                      optimized_episodes=episode, optimized_global_steps=episode*2400,
                      incomplete_started_rollouts=partial,
                      reason="owner-authorized repair pause; not a completed campaign stage")
        atomic_json(status_path, record)
        print(json.dumps(record), flush=True)
    finally:
        kernel.CloseHandle(handle)


def main():
    if os.name != "nt":
        raise RuntimeError("Windows process suspension required")
    with exclusive_run(OUTPUT):
        torch.set_num_threads(1)
        atomic_json(OUTPUT/"manifest.json", dict(reason="owner-authorized native-mode repair",
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    coordinator_already_stopped=True, seeds=[71,72,73]))
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(pause_worker, (71,72,73)))
        atomic_json(OUTPUT/"summary.json", dict(status="all_trainers_stopped_at_verified_checkpoints",
                    workers=[read(OUTPUT/f"optimizer_{s}.json") for s in (71,72,73)],
                    accepted_checkpoint=False))


if __name__ == "__main__":
    main()
