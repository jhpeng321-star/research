"""Finite own-dev collector with one-use CPU continuation; no retries/PID signals."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.rbint.local_resources import child_incarnation
from tools.rbint.remote import ssh
from tools.wmdecide.report import write_json

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/root/rbeval20261031/rb2cal/dev_v1_cpu'


def remote_status():
    return json.loads(ssh('gpuA', f'''
from pathlib import Path
import json
root=Path({REMOTE!r})
spec=json.loads((root/'CANARY.json').read_text())
assert spec['owner']=='S-RB2-CAL' and spec['worlds']==list(range(969420,969432))
assert spec['source']['commit']=='0711a02646bb014a785e190f24e0838460ed8e23'
count=sum((root/('shard'+str(i))/('w'+str(s)+'_k'+str(k)+'.json')).is_file() for i in range(2) for s in range(969420,969432) for k in range(8))
process=spec['process'];path=Path('/proc/'+str(process['pid'])+'/stat')
try:
 fields=path.read_text().split();alive=fields[2]!='Z' and fields[21]==process['start_ticks']
except FileNotFoundError:alive=False
print(json.dumps(dict(tasks=count,stops=[str(p) for p in root.glob('shard*/STOP')],source=spec['source'],
 canary_alive=alive,panel_started=(root/'PANEL.json').exists())))
''', timeout=60))


def work(out, deadline, *, continue_cpu=False):
    receipt = dict(start_unix=time.time(), deadline_unix=deadline, remote=REMOTE,
                   collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(out/'START.json', receipt)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    try:
        while time.time() < deadline:
            status = remote_status()
            with (out/'poll.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(unix=time.time(), **status))+'\n')
            if status['stops']:
                raise RuntimeError('producer terminal STOP; preserve run, no scientific retry')
            if continue_cpu and not status.get('panel_started', True) and not status.get('canary_alive', True):
                if status['tasks'] != 1:
                    raise RuntimeError('canary exited without exactly one successful task; no retry')
                command = [sys.executable, '-B', '-m', 'tools.rbcal.gpu_ops', 'panel',
                    '--runtime', '/root/rbeval20261031/rb2cal/code_0711a02646bb',
                    '--version', 'dev_v1_cpu', '--device', 'cpu']
                with (out/'panel_dispatch.log').open('x', encoding='utf-8') as log:
                    launched = subprocess.run(command, cwd=ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT, timeout=180)
                write_json(out/'PANEL_DISPATCH.json', dict(command=command, rc=launched.returncode))
                if launched.returncode:
                    raise RuntimeError('one-use CPU panel launch blocked; preserve receipts, no retry')
            if status['tasks'] == 96:
                break
            if status['tasks'] > 96:
                raise RuntimeError('duplicate development tasks')
            time.sleep(min(300, max(0, deadline-time.time())))
        else:
            raise TimeoutError('development run incomplete at registered collection deadline')
        commands = [
            ['-m', 'tools.rbcal.gpu_ops', 'fetch', '--version', 'dev_v1_cpu'],
            ['-m', 'tools.rbcal.outcomes', '--source', 'runs/rb2cal_dev_v1_cpu_fetched',
             '--snapshot', 'runs/rb2cal_dev_physical_v1_cpu_snapshot', '--physical-only',
             '--out', 'docs/paper_results/rb2cal/dev_physical_v1_cpu'],
            ['-m', 'tools.rbcal.probability', '--report', 'docs/paper_results/rb2cal/dev_physical_v1_cpu/physical',
             '--out', 'docs/paper_results/rb2cal/probability_physical_v1_cpu'],
            ['-m', 'tools.rbcal.summarize', '--results', 'docs/paper_results/rb2cal/dev_physical_v1_cpu/RESULTS.json',
             '--fit', 'docs/paper_results/rb2cal/v1/fit', '--out', 'docs/paper_results/rb2cal/DEVELOPMENT_PHYSICAL_V1_CPU.md']]
        for i, command in enumerate(commands):
            with (out/f'analysis{i}.log').open('x', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, '-B', *command], cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            write_json(out/f'analysis{i}.json', dict(command=command, rc=result.returncode))
            if result.returncode:
                raise RuntimeError(f'analysis step {i} failed; outputs retained, no retry')
        receipt.update(state='COMPLETE', results='docs/paper_results/rb2cal/DEVELOPMENT_PHYSICAL_V1_CPU.md',
            probability='docs/paper_results/rb2cal/probability_physical_v1_cpu/summary.json',
            outcome_qualification=False, source=status['source'])
        target = ROOT/'docs/paper_results/rb2cal/COMPLETION.json'
        write_json(target, receipt)
    except Exception as error:
        receipt.update(state='BLOCKED', error=repr(error))
    receipt['end_unix'] = time.time()
    write_json(out/'TERMINAL.json', receipt)
    print(json.dumps(receipt), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--deadline', default='2026-11-02T09:00:00+08:00')
    p.add_argument('--work', action='store_true')
    p.add_argument('--continue-cpu', action='store_true', help='one-use CPU continuation after the successful canary')
    args = p.parse_args()
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    if not time.time() < deadline <= datetime.fromisoformat('2026-11-02T09:00:00+08:00').timestamp():
        raise ValueError('bounded collection before owner final delivery required')
    if args.work:
        if not args.out.is_dir():
            raise ValueError('exclusive dispatch directory required')
        work(args.out, deadline, continue_cpu=args.continue_cpu)
    else:
        if not args.out.parent.is_dir() or args.out.exists():
            raise ValueError('exclusive new claim directory required')
        args.out.mkdir()
        command = [sys.executable, '-B', '-u', '-m', 'tools.rbcal.finish', '--work',
                   '--out', str(args.out.resolve()), '--deadline', args.deadline]
        if args.continue_cpu:
            command += ['--continue-cpu']
        with (args.out/'collector.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        receipt = dict(pid=child.pid, command=command, deadline_unix=deadline,
                       source=REMOTE, **child_incarnation(child))
        write_json(args.out/'PROCESS.json', receipt)
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
