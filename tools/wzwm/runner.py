"""Job runner with a hard sim-s cap and a precharge ledger (PLANGEN pg_run.py semantics).

    python -m tools.wzwm run JOBS.jsonl --root ROOT --cap SIM_S [--workers 20] [--core nm2|fake|PATH]
                                                     [--phase wmc] [--host cpuV]

* every job is precharged (planned sim-s) in ROOT/ledger/CHARGES.jsonl BEFORE its process starts; a job
  whose precharge would exceed the cap is refused (REFUSED_BUDGET.json receipt, never launched);
* a job whose out_dir already holds RESULT.json is skipped (never re-run / overwritten);
* own child PIDs are appended to ROOT/PIDS.txt; nothing else is ever signalled;
* core 'nm2' = the vendored, sha-pinned nm2_core.py (SUMO; cpuV only); 'fake' = tools.wzwm.fakes
  (CPU dry run). inprocess=True (tests / dry runs with the fake core only) runs jobs in this process.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import geometry as G
from . import n3

REPO = Path(__file__).resolve().parents[2]


def ledger_path(root):
    return Path(root) / 'ledger' / 'CHARGES.jsonl'


def charged(root):
    p = ledger_path(root)
    if not p.exists():
        return 0.0
    return sum(float(json.loads(x)['planned_sim_s']) for x in p.read_text().splitlines() if x.strip())


def planned(job):
    return float(job['t_end']) if job['kind'] == 'native' else float(job['H'])


def read_jobs(path):
    return [json.loads(x) for x in G.check_path(path).read_text().splitlines() if x.strip()]


def core_cmd(core, job_file):
    if core == 'nm2':
        n3.verify()
        cmd = [sys.executable, str(n3.CORE), str(job_file)]
    elif core == 'fake':
        cmd = [sys.executable, '-m', 'tools.wzwm.fakes', str(job_file)]
    else:
        cmd = [sys.executable, str(G.check_path(core)), str(job_file)]
    if shutil.which('nice'):
        cmd = ['nice', '-n', '19'] + cmd
    return cmd


def run(jobs, root, cap, *, workers=20, core='nm2', phase='wmc', host='cpuV', inprocess=False, world=None):
    root = G.check_path(Path(root))
    led = ledger_path(root)
    led.parent.mkdir(parents=True, exist_ok=True)
    for j in jobs:
        out = Path(j['out_dir']).resolve()
        if root.resolve() not in out.parents:
            raise ValueError(f"job {j['job_id']} writes outside the run root: {out}")
    queue = [j for j in jobs if not (Path(j['out_dir']) / 'RESULT.json').exists()]
    skipped = len(jobs) - len(queue)
    total = charged(root)
    refused = launched = 0
    running = {}
    pids = root / 'PIDS.txt'
    while queue or running:
        while queue and len(running) < max(1, int(workers)):
            j = queue.pop(0)
            p = planned(j)
            out = Path(j['out_dir'])
            out.mkdir(parents=True, exist_ok=True)
            if total + p > float(cap) + 1e-6:
                (out / 'REFUSED_BUDGET.json').write_text(json.dumps(dict(job_id=j['job_id'], charged=total, planned=p,
                                                                         cap=float(cap))))
                refused += 1
                continue
            jf = out / 'JOB.json'
            jf.write_text(json.dumps(j))
            with led.open('a') as fh:
                fh.write(json.dumps(dict(phase=phase, job_id=j['job_id'], host=host, planned_sim_s=p,
                                         time=time.time())) + '\n')
            total += p
            launched += 1
            if inprocess:
                if core != 'fake':
                    raise ValueError('inprocess execution is only allowed with the fake core')
                from . import fakes
                try:
                    res = fakes.run_job(j, world)
                except Exception as exc:
                    res = dict(status='failed', job=j, error=repr(exc))
                fakes.write_result(j, res)
                continue
            err = open(out / 'stderr.txt', 'w')
            proc = subprocess.Popen(core_cmd(core, jf), cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=err)
            with pids.open('a') as fh:
                fh.write(f"{proc.pid} {j['job_id']} {time.time():.0f}\n")
            running[proc.pid] = (proc, j, err)
        time.sleep(0.0 if inprocess else 0.3)
        for pid, (proc, j, err) in list(running.items()):
            if proc.poll() is not None:
                err.close()
                del running[pid]
    return dict(jobs=len(jobs), skipped_existing=skipped, launched=launched, refused=refused, charged=charged(root),
                cap=float(cap))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wzwm run')
    ap.add_argument('jobs')
    ap.add_argument('--root', required=True)
    ap.add_argument('--cap', type=float, required=True)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--core', default='nm2')
    ap.add_argument('--phase', default='wmc')
    ap.add_argument('--host', default='cpuV')
    a = ap.parse_args(argv)
    print(json.dumps(run(read_jobs(a.jobs), a.root, a.cap, workers=a.workers, core=a.core, phase=a.phase,
                         host=a.host)))


if __name__ == '__main__':
    main()
