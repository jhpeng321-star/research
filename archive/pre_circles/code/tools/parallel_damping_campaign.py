"""Run the three registered optimizer lineages in isolated output directories.

Changes scheduling only. Never changes the frozen trainer or generates new
reference/acceptance cases. Existing canonical training is adopted by PID.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Event

from experiments.idm.train_damping_campaign import initialize
from experiments.idm.train_native_damping import atomic_json, digest, exclusive_run
from idm.damping_campaign import CHECKPOINT_EPISODES, DEVELOPMENT_CASES, TRAINING_CASES
from idm.damping_selection import assess_candidate, select_candidate
from idm.damping_scenario import request_schedule
from tools.continue_damping_campaign import (
    MODULE, ROOT, read, stage_complete, verify_sources, wait_existing,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_exact(source, target):
    """Resume-safe copying of immutable completed files, never overwrite a difference."""
    source, target = Path(source), Path(target)
    expected = sha(source)
    if target.exists():
        if sha(target) != expected:
            raise ValueError(f"conflicting existing file: {target}")
        return expected
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream, source.open("rb") as origin:
        shutil.copyfileobj(origin, stream)
    if sha(target) != expected:
        raise ValueError(f"copy verification failed: {target}")
    return expected


def cached_paths(root):
    paths = []
    expected = {}
    for role, cases in (("training", TRAINING_CASES), ("development", DEVELOPMENT_CASES)):
        for seed, _ in cases:
            paths.append(root / "references" / f"{role}_{seed}.json")
        for seed, loading in cases:
            requests = request_schedule(seed, loading, role=role)
            folder = root / "scenarios" / f"{role}_{seed}"
            identity = hashlib.sha256(
                json.dumps([asdict(r) for r in requests], sort_keys=True).encode()
                + (folder / "scenario.sumocfg").read_bytes()
                + (folder / "requests.rou.xml").read_bytes()
            ).hexdigest()
            expected[seed] = (identity, len(requests))
    paths.extend(root / "analytical" / f"development_{seed}.json" for seed, _ in DEVELOPMENT_CASES)
    # Missing cache is an error, not permission to collect another reference.
    for path in paths:
        row = read(path)
        summary = row["summary"]
        seed = int(path.stem.rsplit("_", 1)[1])
        if (summary["seed"] != seed or
                (summary["scenario_identity"], summary["population"]) != expected[seed]):
            raise ValueError(f"cached reference identity mismatch: {path}")
        if not summary["team_terminal"] or summary["global_control_steps"] != 2400:
            raise ValueError(f"incomplete cached reference: {path}")
        if not path.with_suffix(".npz").is_file():
            raise ValueError(f"missing cached arrays: {path}")
    return [p for path in paths for p in (path, path.with_suffix(".npz"))]


def prepare(root, initialization, sumo):
    verify_sources(root)
    manifest = read(root / "manifest.json")
    cache = cached_paths(root)
    workers = {71: root, **{s: root / "parallel_workers" / f"optimizer_{s}" for s in (72, 73)}}
    hashes = {str(path.relative_to(root)): sha(path) for path in cache}
    for seed in (72, 73):
        worker = workers[seed]
        with exclusive_run(worker):
            identity, _ = initialize(worker, initialization, sumo)
            if identity != manifest:
                raise ValueError("isolated worker changes registered campaign identity")
            for path in cache:
                copy_exact(path, worker / path.relative_to(root))
    registration = dict(
        schema="native_damping_parallel_scheduling/1", campaign=digest(manifest),
        worker_roots={str(k): str(v) for k, v in workers.items()},
        cached_files_sha256=hashes, wrapper_sha256=sha(__file__),
        stages=list(CHECKPOINT_EPISODES), maximum_concurrent_lineages=3,
        additional_simulation_budget=0, acceptance_authorized=False,
    )
    path = root / "parallel_execution/manifest.json"
    if path.exists() and read(path) != registration:
        raise ValueError("parallel scheduling registration changed")
    atomic_json(path, registration)
    copy_exact(Path(__file__), path.parent / "source_snapshot.py")
    return workers, registration


def run_stage(root, initialization, stage, heartbeat, log_dir):
    seed, episode = stage
    command = [sys.executable, "-m", MODULE, "--output", str(root),
               "--initialization", str(initialization), "--optimizer-seed", str(seed),
               "--through-episode", str(episode), "--resume"]
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"optimizer_{seed}_through_{episode:03}.log").open("ab", buffering=0) as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0) as child:
            while True:
                heartbeat("running", stage, child.pid)
                try:
                    return child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass


def run_worker(root, initialization, seed, adopted_pid, heartbeat, log_dir, failed):
    verify_sources(root)
    if adopted_pid is not None:
        active = read(root / "active_stage.json")
        stage = (active["optimizer_seed"], active["through_episode"])
        if active.get("pid") != adopted_pid or active.get("status") != "running" or stage[0] != seed or stage[1] not in CHECKPOINT_EPISODES:
            raise ValueError("adopted PID disagrees with active lineage")
        code = wait_existing(adopted_pid, root, initialization, stage, heartbeat)
        if code or read(root / "active_stage.json").get("status") != "complete" or not stage_complete(root, *stage):
            raise RuntimeError("adopted stage failed or is incomplete")
    for episode in CHECKPOINT_EPISODES:
        stage = (seed, episode)
        if stage_complete(root, *stage):
            continue
        if failed.is_set():
            raise RuntimeError("another worker failed; no further stages launched")
        verify_sources(root)
        code = run_stage(root, initialization, stage, heartbeat, log_dir)
        if code or read(root / "active_stage.json").get("status") != "complete" or not stage_complete(root, *stage):
            raise RuntimeError(f"stage {stage} failed or is incomplete")
    heartbeat("lineage_complete", (seed, 100), None)


def consolidate(root, workers):
    """Only after every worker is terminal, copy completed immutable evidence."""
    with exclusive_run(root):
        canonical = read(root / "manifest.json")
        for seed, worker in workers.items():
            if read(worker / "manifest.json") != canonical or read(worker / "active_stage.json").get("status") != "complete":
                raise ValueError("worker identity or terminal state mismatch")
            if not all(stage_complete(worker, seed, p) for p in CHECKPOINT_EPISODES):
                raise RuntimeError("all registered worker evaluations are required")
        copied = {}
        for seed in (72, 73):
            worker = workers[seed]
            folders = [worker / f"lineage_{seed}"] + [worker / "development" / f"optimizer_{seed}_episode_{p:03}" for p in CHECKPOINT_EPISODES]
            for folder in folders:
                for path in folder.rglob("*"):
                    if path.is_file():
                        target = root / path.relative_to(worker)
                        copied[str(target.relative_to(root))] = copy_exact(path, target)
        records = []
        for seed in (71, 72, 73):
            for episode in CHECKPOINT_EPISODES:
                record = read(root / "development" / f"optimizer_{seed}_episode_{episode:03}" / "summary.json")
                if record["campaign"] != digest(canonical) or record["assessment"] != assess_candidate(record["cases"]):
                    raise ValueError("consolidated assessment or identity mismatch")
                records.append({**record, "checkpoint_path": str(root / f"lineage_{seed}" / f"trainer_{episode:03}.pt")})
        chosen = select_candidate(records)
        atomic_json(root / "development_selection.json", dict(
            candidates=records, selected=None if chosen is None else {k:chosen[k] for k in ("optimizer_seed", "optimized_global_steps", "checkpoint_path")},
            accepted_checkpoint=False, acceptance_evaluation_authorized=False,
        ))
        atomic_json(root / "parallel_execution/consolidation.json", dict(
            copied_sha256=copied, completed_evaluations=len(records), accepted_checkpoint=False,
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--adopt-pid", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root, initialization = args.campaign.resolve(), args.initialization.resolve()
    directory = root / "parallel_execution"
    with exclusive_run(directory):
        workers, registration = prepare(root, initialization, shutil.which("sumo"))
        if args.prepare_only:
            print(json.dumps(dict(status="prepared", **registration)), flush=True)
            return
        # Old coordinator must be stopped before replacing its scheduling ownership.
        with exclusive_run(root / "continuation"):
            failed = Event()

            def heartbeat(seed, status, stage, pid):
                atomic_json(directory / f"optimizer_{seed}.json", dict(
                    status=status, stage=stage, trainer_pid=pid, coordinator_pid=os.getpid(),
                    updated_at_utc=datetime.now(timezone.utc).isoformat(),
                    worker_root=str(workers[seed]), accepted_checkpoint=False,
                ))

            errors = []
            atomic_json(directory / "state.json", dict(status="running", coordinator_pid=os.getpid(), accepted_checkpoint=False))
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {}
                for seed, worker in workers.items():
                    if seed != 71 and failed.wait(3):
                        break
                    future = pool.submit(run_worker, worker, initialization, seed,
                                         args.adopt_pid if seed == 71 else None,
                                         lambda status, stage, pid, s=seed: heartbeat(s, status, stage, pid),
                                         directory / "logs", failed)
                    futures[future] = seed
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        failed.set()
                        errors.append(dict(optimizer_seed=seed, error=repr(exc)))
                        heartbeat(seed, "failed", None, None)
                        atomic_json(directory / "failures.json", errors)
            if errors:
                atomic_json(directory / "state.json", dict(status="failed", errors=errors, accepted_checkpoint=False))
                raise RuntimeError("parallel campaign stopped; see failures.json")
            consolidate(root, workers)
            atomic_json(directory / "state.json", dict(status="all_registered_stages_complete", accepted_checkpoint=False))


if __name__ == "__main__":
    main()
