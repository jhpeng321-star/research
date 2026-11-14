"""WM-N1-REPL-CL-20261005 Part C launcher: policies x 30 TEST_REPL worlds.

DO NOT run without the coordinator's go (the coordinator launches the
campaign). Usage (from any cwd):

    python E:/rd-wt/pl01-a11/_out/run_closed_loop.py --workers 4 \
        --policies pl01_a11,fleet_wm,always_native,always_pi,random [--threads 4]
        [--dry-run] [--policy-module mypkg.walls_policy] [--worlds w1,w2]

* worlds: ``output/campaign_stage1_v3/results_plan_repl/WORLD_PLAN.json``;
* restores ``sealed/repl_20261005/runs/<world>__base`` (only the EXISTENCE of
  ``saved_state/STATE_MANIFEST.json`` and ``RUN_RECEIPT.json`` is checked --
  a world whose base is not there yet is skipped and picked up by a later
  invocation);
* writes ``sealed/repl_20261005/closed_loop/<policy>/<world>/`` (one child
  process per run: ``closedloop/runner.py``);
* every launch is preceded by a ``start`` row and followed by an ``end`` row in
  ``runs/active/WM-N1-REPL-CL-20261005/RUNS_C.jsonl``;
* 0 retries: a run with a complete receipt is skipped; a run with ANY earlier
  start row and no complete receipt (failed / interrupted) is never relaunched
  and its directory is kept;
* this launcher reads only receipts' ``exit_status`` -- never outcome values.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("WM_N1_CODE_ROOT") or "E:/research-dense")
WT = Path(__file__).resolve().parents[1]                 # E:/rd-wt/pl01-a11
RUNNER = WT / "output" / "wd01" / "wm_n1" / "closedloop" / "runner.py"
DEFAULT_PLAN = REPO / "output/campaign_stage1_v3/results_plan_repl/WORLD_PLAN.json"
DEFAULT_BASES = REPO / "sealed/repl_20261005/runs"
DEFAULT_OUT = REPO / "sealed/repl_20261005/closed_loop"
EXP = REPO / "runs/active/WM-N1-REPL-CL-20261005"
DEFAULT_RUNS_LOG = EXP / "RUNS_C.jsonl"
DEFAULT_REGISTRATION = EXP / "REGISTRATION.md"
DEFAULT_POLICIES = "pl01_a11,fleet_wm,always_native,always_pi,random"
PYTHON = os.environ.get("WM_N1_PYTHON") or sys.executable
SCHEMA = "wm_n1_repl_runs_c/1"
_lock = threading.Lock()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, rec: dict) -> None:
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def receipt_status(target: Path):
    p = target / "RUN_RECEIPT.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("exit_status")
    except (OSError, json.JSONDecodeError):
        return "unreadable"


def prior_starts(runs_log: Path) -> set:
    out = set()
    if runs_log.is_file():
        for line in runs_log.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "start":
                out.add(rec["job"])
    return out


def base_ready(bases: Path, world: str) -> bool:
    d = bases / f"{world}__base"
    return (d / "saved_state" / "STATE_MANIFEST.json").is_file() and (d / "RUN_RECEIPT.json").is_file()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--policies", default=DEFAULT_POLICIES)
    ap.add_argument("--threads", type=int, default=4, help="torch threads per run")
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--bases-root", type=Path, default=DEFAULT_BASES)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--runs-log", type=Path, default=DEFAULT_RUNS_LOG)
    ap.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    ap.add_argument("--horizon-s", type=float, default=600.0)
    ap.add_argument("--worlds", default=None, help="comma list (default: all plan worlds)")
    ap.add_argument("--policy-module", action="append", default=[])
    ap.add_argument("--code-root", type=Path, default=REPO)
    ap.add_argument("--native-reference", choices=("zero", "head"), default=None)
    ap.add_argument("--max-decision-units", type=int, default=None)
    ap.add_argument("--open-road", action="store_true")
    ap.add_argument("--python", default=PYTHON)
    ap.add_argument("--device", default=None, help="cpu|cuda (HDV net)")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-sha256", default=None)
    ap.add_argument("--climatology", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    policies = [p.strip() for p in a.policies.split(",") if p.strip()]
    plan = json.loads(a.plan.read_text(encoding="utf-8"))
    worlds = [w["world"] for w in plan["worlds"]]
    if a.worlds:
        keep = {w.strip() for w in a.worlds.split(",")}
        worlds = [w for w in worlds if w in keep]
    starts = prior_starts(a.runs_log)
    jobs, skipped = [], {"complete": 0, "prior_attempt": 0, "no_base": 0}
    for world in worlds:
        if not base_ready(a.bases_root, world):
            skipped["no_base"] += len(policies)
            continue
        for pol in policies:
            job = f"{world}__cl_{pol}"
            target = a.out_root / pol / world
            st = receipt_status(target)
            if st == "complete":
                skipped["complete"] += 1
            elif job in starts or target.exists():
                skipped["prior_attempt"] += 1          # 0 retries; kept
            else:
                jobs.append({"job": job, "world": world, "policy": pol, "target": target})
    print(json.dumps({"to_launch": len(jobs), "skipped": skipped, "policies": policies,
                      "workers": a.workers}), flush=True)
    if a.dry_run:
        for j in jobs:
            print(j["job"])
        return 0
    logs = a.out_root / "_logs"
    logs.mkdir(parents=True, exist_ok=True)

    def run(job):
        cmd = [a.python, str(RUNNER), "--world", job["world"], "--policy", job["policy"],
               "--plan", str(a.plan), "--bases-root", str(a.bases_root),
               "--out", str(job["target"]), "--horizon-s", str(a.horizon_s),
               "--registration", str(a.registration), "--threads", str(a.threads),
               "--code-root", str(a.code_root)]
        for mod in a.policy_module:
            cmd += ["--policy-module", mod]
        if a.native_reference is not None:
            cmd += ["--native-reference", a.native_reference]
        if a.open_road:
            cmd += ["--open-road"]
        if a.max_decision_units is not None:
            cmd += ["--max-decision-units", str(a.max_decision_units)]
        for flag, val in (("--device", a.device), ("--model-path", a.model_path),
                          ("--model-sha256", a.model_sha256), ("--climatology", a.climatology)):
            if val:
                cmd += [flag, str(val)]
        env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
        start = utcnow()
        append_jsonl(a.runs_log, {"schema": SCHEMA, "kind": "start", "job": job["job"],
                                  "world": job["world"], "policy": job["policy"],
                                  "out_dir": str(job["target"]), "start_utc": start,
                                  "cmd": cmd})
        t = time.time()
        with open(logs / f"{job['job']}.out.log", "a", encoding="utf-8") as out, \
                open(logs / f"{job['job']}.err.log", "a", encoding="utf-8") as err:
            try:
                proc = subprocess.Popen(cmd, cwd=str(WT), stdout=out, stderr=err, env=env)
                append_jsonl(a.runs_log, {"schema": SCHEMA, "kind": "pid", "job": job["job"],
                                          "pid": proc.pid})
                rc = proc.wait()
            except OSError as exc:
                err.write(f"spawn error: {exc!r}\n")
                rc = -1
        status = receipt_status(job["target"])
        append_jsonl(a.runs_log, {"schema": SCHEMA, "kind": "end", "job": job["job"],
                                  "world": job["world"], "policy": job["policy"],
                                  "start_utc": start, "end_utc": utcnow(), "rc": rc,
                                  "wall_s": round(time.time() - t, 1),
                                  "status": "complete" if (rc == 0 and status == "complete")
                                  else "failed"})
        print(f"{job['job']}: rc={rc} status={status}", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        rcs = list(pool.map(run, jobs))
    print(json.dumps({"launched": len(jobs), "failed": sum(1 for r in rcs if r != 0)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
