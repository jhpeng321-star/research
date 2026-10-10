"""One separately recorded recovery of the failed pre-step SUMO startup.

Controller/reward/checkpoint remain frozen. The failed allocation is retained
and charged in full; this recovery uses one of the two reserved rollout slots.
"""

from contextlib import contextmanager
from pathlib import Path
import hashlib
import os

from experiments.idm import deterministic_sumo
from experiments.idm.evaluate_damping_v2_check import verify_sources, read
from experiments.idm.train_native_damping import atomic_json, exclusive_run, save_rollout
from env.damping_v2_rollout import collect_episode
from idm.damping_scenario import request_schedule

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "runs/probe/native_selectable_damping_v2_check"


@contextmanager
def fixed_launch_lock():
    """Lock byte zero in binary mode; never append into another writer's lock."""
    import msvcrt

    path = ROOT / "runs/probe/.damping_v2_sumo_launch.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    corridor = CHECK / "corridor"
    failures = [r for r in read(corridor / "executions.json") if r["exit_code"]]
    if failures != [dict(seed=911003, arm="native", exit_code=1)]:
        raise RuntimeError("recovery only admits the specifically inspected single startup failure")
    log = (corridor / "native_911003.log").read_text(encoding="utf-8")
    if "PermissionError: [Errno 13] Permission denied" not in log or "collector seed=" in log:
        raise RuntimeError("failure is not the verified pre-step lock failure")
    if (corridor / "native_911003.json").exists():
        raise RuntimeError("original allocation unexpectedly complete")
    output = corridor / "startup_recovery_01"
    with exclusive_run(output):
        if (output / "manifest.json").exists():
            raise RuntimeError("recovery already allocated; no automatic repeat")
        registration = read(corridor / "manifest.json")
        verify_sources(CHECK, registration)
        record = dict(
            schema="damping_v2_launch_recovery/1",
            pid=os.getpid(),
            original_failed_allocation=str(corridor / "native_911003.started.json"),
            reason="binary byte-zero lock replaces legacy append/text startup lock in this process only",
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            controller_and_reward_unchanged=True,
            allocated_global_steps=2400,
            total_allocated_corridor_steps=16800,
            preregistered_corridor_cap=19200,
            total_allocated_repair_steps=20400,
            preregistered_repair_cap=22800,
        )
        atomic_json(output / "manifest.json", record)
        (output / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
        deterministic_sumo._InterprocessLaunchLock = fixed_launch_lock
        scene = corridor / "scene_911003/scenario.sumocfg"
        requests = request_schedule(911003, 2200, role="development")
        result = collect_episode(
            scene,
            requests,
            seed=911003,
            sumo_binary="C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe",
        )
        save_rollout(output / "native_911003.json", result)
        verify_sources(CHECK, registration)
        atomic_json(
            output / "summary.json",
            dict(
                record,
                status="complete",
                actual_global_steps=2400,
                total_successful_repair_steps=18000,
            ),
        )


if __name__ == "__main__":
    main()
