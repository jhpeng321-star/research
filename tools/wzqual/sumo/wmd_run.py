"""Run WMQUAL-B prefix jobs with a hard sim-s cap and a precharge ledger (task WM-D, phase wm_wz_data).

    python wmd_run.py JOBS.jsonl --workers 12 [--cap 100000]

* every job is precharged (t_end sim-s) in ledger/CHARGES.jsonl BEFORE its process starts; a job whose
  precharge would exceed the cap is refused (REFUSED_BUDGET.json receipt, never launched);
* a job whose out_dir already holds RESULT.json is skipped (never re-run / overwritten);
* own child PIDs are appended to PIDS.txt; nothing else is ever signalled.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
LEDGER = BASE / "ledger" / "CHARGES.jsonl"
CAP = 100000.0
PHASE = "wm_wz_data"


def charged(ledger: Path) -> float:
    if not ledger.exists():
        return 0.0
    return sum(float(json.loads(x)["planned_sim_s"]) for x in ledger.read_text().splitlines() if x.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cap", type=float, default=CAP)
    args = ap.parse_args()
    assert args.cap <= CAP
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(x) for x in Path(args.jobs).read_text().splitlines() if x.strip()]
    queue = [j for j in jobs if not (Path(j["out_dir"]) / "RESULT.json").exists()]
    running = {}
    pids = HERE / "PIDS.txt"
    refused = 0
    total = charged(LEDGER)
    while queue or running:
        while queue and len(running) < args.workers:
            j = queue.pop(0)
            p = float(j["t_end"])
            Path(j["out_dir"]).mkdir(parents=True, exist_ok=True)
            if total + p > args.cap + 1e-6:
                (Path(j["out_dir"]) / "REFUSED_BUDGET.json").write_text(json.dumps(
                    dict(job_id=j["job_id"], charged=total, planned=p, cap=args.cap)))
                refused += 1
                continue
            jf = Path(j["out_dir"]) / "JOB.json"
            jf.write_text(json.dumps(j))
            with LEDGER.open("a") as fh:
                fh.write(json.dumps(dict(phase=PHASE, task="WM-D", job_id=j["job_id"], host="cpuV",
                                         planned_sim_s=p, time=time.time())) + "\n")
            total += p
            proc = subprocess.Popen(["nice", "-n", "19", sys.executable, str(HERE / "wmd_prefix.py"), str(jf)],
                                    stdout=subprocess.DEVNULL,
                                    stderr=open(Path(j["out_dir"]) / "stderr.txt", "w"))
            with pids.open("a") as fh:
                fh.write(f"{proc.pid} {j['job_id']} {time.time():.0f}\n")
            running[proc.pid] = (proc, j)
        time.sleep(0.5)
        for pid, (proc, j) in list(running.items()):
            if proc.poll() is not None:
                del running[pid]
    print(json.dumps(dict(jobs=len(jobs), refused=refused, charged=charged(LEDGER))))


if __name__ == "__main__":
    main()
