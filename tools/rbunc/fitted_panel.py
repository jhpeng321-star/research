"""Four disjoint owned CPU workers, durable terminal receipt, no retries."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fit-directory')
    p.add_argument('--lane-directory')
    args = p.parse_args()
    assert args.out.is_dir() and os.getloadavg()[0] < 24
    workers = []
    extra = []
    for key in ('fit_directory', 'lane_directory'):
        if getattr(args, key) is not None:
            extra += ['--' + key.replace('_', '-'), getattr(args, key)]
    for index in range(4):
        log = (args.out/f'probe_worker{index}.log').open('x')
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).with_name('fitted_probe.py')),
            '--out', str(args.out), '--worker', str(index), *extra], stdout=log, stderr=subprocess.STDOUT)
        with (args.out/f'probe_worker{index}_PID.json').open('x') as f:
            json.dump(dict(pid=child.pid, parent=os.getpid(), partition=index), f)
        workers.append((child, log))
    results = []
    for child, log in workers:
        results.append(dict(pid=child.pid, rc=child.wait()))
        log.close()
    with (args.out/'probe_TERMINAL.json').open('x') as f:
        json.dump(dict(results=results, end_unix=time.time()), f)
    assert all(r['rc'] == 0 for r in results), results


if __name__ == '__main__':
    main()
