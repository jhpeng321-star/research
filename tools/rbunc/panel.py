"""Own at most four single-thread CPU FIT probes; preserves canary and all logs."""
from pathlib import Path
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--horizon', type=int, default=120)
    args = p.parse_args()
    os.nice(19)
    assert args.out.is_dir() and os.getloadavg()[0] < 16
    jobs = []
    running_canary = False
    for seed in range(969400, 969432):
        target = args.out/f'w{seed}_h{args.horizon}.json'
        if target.exists():
            old = json.loads(target.read_text())
            assert old['receipt']['world'] == seed and old['horizon'] == args.horizon
        else:
            if seed == 969400 and (args.out/f'w{seed}_h{args.horizon}.log').exists():
                running_canary = True  # owned canary continues, never duplicated
                continue
            assert not (args.out/f'w{seed}_h{args.horizon}.log').exists(), 'preserve existing attempt'
            jobs.append(seed)
    receipt = dict(pid=os.getpid(), worlds=jobs, workers=8, horizon=args.horizon,
        running_canary_preserved=running_canary,
        source=str(Path(__file__).resolve()), start_unix=time.time())
    with (args.out/'PANEL_DISPATCH.json').open('x') as f:
        json.dump(receipt, f, indent=1)
    def run(seed):
        while os.getloadavg()[0] > 24:
            time.sleep(10)
        with (args.out/f'w{seed}_h{args.horizon}.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).with_name('fit_probe.py')),
                '--out', str(args.out), '--world', str(seed), '--horizon', str(args.horizon)],
                stdout=log, stderr=subprocess.STDOUT)
            with (args.out/f'PID_{seed}.json').open('x') as f:
                json.dump(dict(pid=child.pid, parent=os.getpid(), seed=seed), f)
            rc = child.wait()
        print('EXIT', seed, rc, flush=True)
        return seed, rc
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, jobs))
    with (args.out/'PANEL_TERMINAL.json').open('x') as f:
        json.dump(dict(results=results, completed_unix=time.time()), f, indent=1)
    assert all(rc == 0 for _, rc in results)


if __name__ == '__main__':
    main()
