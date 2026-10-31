"""Real SUMO evaluator for S-SUMOES-RUN (runs on cpuQ next to the frozen E3 runtime).

``evaluate_many`` takes already-AUTHORISED requests (the driver's BudgetLedger charged
them), appends durable fsynced accounting rows BEFORE any worker starts, splits the
jobs over <= ``workers`` owned nice-19 worker processes (tools/sumoes/remote/worker.py),
waits, and returns per-job results. Any worker failure / STOP marker -> MechanicalFailure
after all owned workers have exited (no retries; failed bytes kept).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .driver import MechanicalFailure

CHARGE = 900


def theta_sha(values) -> str:
    return hashlib.sha256(json.dumps([float(v) for v in values]).encode()).hexdigest()


def _append_fsync(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class SumoRuntimeEvaluator:
    def __init__(self, own: str | Path, *, cap_sim_s: int, workers: int = 24,
                 python: str = sys.executable, poll_s: float = 10.0,
                 launcher: list[str] | None = None, design: dict | None = None) -> None:
        if not 1 <= workers <= 24:
            raise ValueError("cpuQ registration allows 1..24 SUMO workers")
        self.own = Path(own)
        self.cap_sim_s = int(cap_sim_s)
        self.workers, self.python, self.poll_s = workers, python, poll_s
        self.launcher = launcher if launcher is not None else ["nice", "-n", "19"]
        self.design = design if design is not None else json.loads(
            (self.own / "design.json").read_text())
        self.by_seed = {int(w["seed"]): w for w in self.design["worlds"]}
        self.batch = 0
        for name in ("jobs", "thetas", "batches", "locks", "logs"):
            (self.own / name).mkdir(parents=True, exist_ok=True)

    # -- accounting -------------------------------------------------------------------
    def ledger_rows(self) -> list[dict]:
        path = self.own / "accounting.jsonl"
        if not path.exists():
            return []
        return [json.loads(s) for s in path.read_text().splitlines() if s]

    def charge(self, job_ids: list[str]) -> None:
        rows = self.ledger_rows()
        known = {r["id"] for r in rows}
        if known & set(job_ids) or len(set(job_ids)) != len(job_ids):
            raise MechanicalFailure("job id already charged; never retry")
        total = sum(r["sim_s"] for r in rows) + CHARGE * len(job_ids)
        if total > self.cap_sim_s:
            raise MechanicalFailure(f"charging would reach {total} > cap {self.cap_sim_s}")
        now = time.time()
        _append_fsync(self.own / "accounting.jsonl",
                      [dict(id=j, sim_s=CHARGE, unix=now) for j in job_ids])

    # -- theta files ------------------------------------------------------------------
    def theta_file(self, theta) -> tuple[str, str]:
        values = [float(v) for v in theta.detach().cpu().double().flatten().tolist()]
        if len(values) != 137:
            raise ValueError("theta must have 137 scalars")
        digest = theta_sha(values)
        path = self.own / "thetas" / f"{digest}.json"
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(theta=values, sha256=digest)) + "\n")
            os.replace(tmp, path)
        return str(path), digest

    # -- dispatch ---------------------------------------------------------------------
    def evaluate(self, theta, world_seed: int, update: int, tag: str) -> dict:
        return self.evaluate_many([dict(theta=theta, seed=world_seed, update=update,
                                        tag=tag)])[0]

    def evaluate_many(self, requests: list[dict]) -> list[dict]:
        if (self.own / "STOP").exists():
            raise MechanicalFailure("STOP marker present; dispatch refused")
        jobs = []
        for request in requests:
            seed = int(request["seed"])
            world = self.by_seed[seed]
            arm = request.get("arm", "ours")
            job = dict(id=request["tag"], seed=seed, arm=arm,
                       penetration=world["penetration"], demand=world["demand"],
                       tag=request["tag"], update=request["update"])
            if arm == "ours":
                job["theta_file"], job["theta_sha256"] = self.theta_file(request["theta"])
            elif arm != "ctrl0":
                raise ValueError(f"unknown arm {arm}")
            if (self.own / "jobs" / job["id"]).exists():
                raise MechanicalFailure(f"{job['id']} already exists; never retry")
            jobs.append(job)
        self.charge([j["id"] for j in jobs])
        self.batch += 1
        stamp = f"b{int(time.time())}_{self.batch:03d}"
        folder = self.own / "batches" / stamp
        folder.mkdir(parents=True)
        lanes = min(self.workers, len(jobs))
        # Longest-first is unknowable; round-robin keeps each worker's list short.
        split = [jobs[i::lanes] for i in range(lanes)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", C4_POLICY_STOCHASTIC="0",
                   C4_POLICY_ACCEL_NOISE="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        procs = []
        for index, chunk in enumerate(split):
            spec = folder / f"{stamp}_w{index:02d}.json"
            spec.write_text(json.dumps(chunk, indent=1) + "\n")
            log = (folder / f"{spec.stem}.log").open("x")
            proc = subprocess.Popen(
                self.launcher + [self.python, str(self.own / "pytools/tools/sumoes/remote/worker.py"),
                                 str(spec)], stdout=log, stderr=subprocess.STDOUT, env=env,
                cwd=str(self.own))
            procs.append((proc, spec, log))
        with (folder / "PIDS.json").open("x") as handle:
            json.dump([dict(pid=p.pid, spec=str(s)) for p, s, _ in procs], handle)
        while any(p.poll() is None for p, _, _ in procs):
            time.sleep(self.poll_s)
        for _, _, log in procs:
            log.close()
        bad = [str(s) for p, s, _ in procs if p.returncode != 0]
        if bad or (self.own / "STOP").exists():
            raise MechanicalFailure(f"worker failure in {bad or 'STOP marker'}; dispatch stopped")
        return [self._result(job) for job in jobs]

    def _result(self, job: dict) -> dict:
        path = self.own / "jobs" / job["id"] / "result.json"
        if not path.exists():
            raise MechanicalFailure(f"{job['id']}: missing result.json")
        result = json.loads(path.read_text())
        metrics = result["metrics"]
        clean = (result.get("ok") is True and result.get("actual_sim_s") == 900
                 and not any(metrics[k] for k in ("collisions", "teleports",
                                                   "dropped_vehicle_steps"))
                 and all(v == 0 for v in result["lane_audit"].values()))
        return dict(job=job["id"], J600=float(result["J600"]),
                    actual_sim_s=float(result["actual_sim_s"]), safety_clean=clean,
                    initial_state_sha256=metrics["initial_state_sha256"],
                    head_stats=result.get("head_stats"), wall_s=result.get("wall_s"))
