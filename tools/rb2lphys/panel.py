"""Bounded two-worker development panel, preserving and reusing its own canaries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--off-canary', type=Path, required=True)
    p.add_argument('--lp-canary', type=Path, required=True)
    p.add_argument('--futures', type=int, default=2)
    args = p.parse_args()
    assert args.out.is_dir() and not (args.out/'PANEL.json').exists()
    assert args.futures == 2
    stage = Path(__file__).resolve().parents[2]
    start = time.time()
    # Canary tasks reused exactly once; never re-run completed frozen tasks.
    canaries = {'off': args.off_canary, 'learned_v1': args.lp_canary}
    for params, root in canaries.items():
        while not (root/'task/LP_COMPLETE.json').exists():
            process = json.loads((root/'PROCESS.json').read_text())
            stat = Path(f"/proc/{process['pid']}/stat")
            if not stat.exists() or stat.read_text().split()[2] == 'Z':
                raise RuntimeError('canary failed: '+str(root))
            time.sleep(10)
        record = json.loads((root/'task/LP_COMPLETE.json').read_text())
        assert record['world'] == 969420 and record['K'] == 2 and record['params'] == params
    records = {}
    def worker(params):
        rows = [dict(world=969420, directory=str(canaries[params]/'task'), reused_canary=True)]
        for world in range(969421,969432):
            while os.getloadavg()[0] > 24:time.sleep(30)
            if (args.out/'STOP').exists():raise RuntimeError('own cooperative stop')
            task = args.out/f'{params}_{world}'
            command = [sys.executable, '-B', '-m', 'tools.rb2lphys.evaluate',
                '--world', str(world), '--futures','2','--params',params,'--out',str(task)]
            with (args.out/f'{params}_{world}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=stage, stdout=log, stderr=subprocess.STDOUT)
                own = dict(owner='S-RB2-LPHYS', pid=child.pid,
                    start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21], command=command)
                (args.out/f'{params}_{world}_PROCESS.json').write_text(json.dumps(own,indent=2)+'\n')
                result = child.wait()
            if result or not (task/'LP_COMPLETE.json').exists():
                raise RuntimeError(f'own task failed{result}: {task}')
            rows.append(dict(world=world,directory=str(task),reused_canary=False))
            print('PANEL DONE', params, world, flush=True)
        records[params] = rows
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, params) for params in canaries]
        for job in jobs:job.result()
    result = dict(schema='rb2lphys.panel.v1', worlds=list(range(969420,969432)), K=2,
        arms=['native','ctrl0','mappo','pi','circles'], horizons=[120,300,600],
        records=records, started_unix=start, finished_unix=time.time(),
        scope='development only; fixed engineering source; not FULL-v5 qualification')
    with (args.out/'PANEL.json').open('x') as f:json.dump(result,f,indent=2)
    print('PANEL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
