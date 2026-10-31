"""Single owned CPU validation job; exclusive receipt prevents duplicate launches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--first-seed', type=int, default=969402)
    p.add_argument('--label', default='remaining_fit')
    args = p.parse_args()
    work = Path(args.work).resolve()
    if not work.is_dir():
        p.error('existing task-owned work directory required')
    if not 969400 <= args.first_seed <= 969431 or not args.label.replace('_', '').isalnum():
        p.error('invalid FIT range or receipt label')
    receipt = work / f'{args.label}_dispatch.json'
    # Reserve before Popen, so a repeated invocation cannot duplicate the job.
    with receipt.open('x') as f:
        json.dump({'state': 'reserved', 'owner': 'S-RB-MERGE'}, f)
    script = Path(__file__).with_name('validate_merge.py')
    command = [sys.executable, str(script), '--root', args.root, '--seeds',
               *map(str, range(args.first_seed, 969432)), '--horizon', '300',
               '--target-csv', str(script.with_name('ramp_delay_world.csv')),
               '--output', str(work / f'{args.label}_300.json'), '--quiet']
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (work / f'{args.label}.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True, env=env,
                                 preexec_fn=lambda: os.nice(10))
    record = dict(owner='S-RB-MERGE', pid=child.pid, state='launched', command=command,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  validator_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                  mechanics_sha256={str(path.relative_to(script.parents[2])): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (script.parents[2] / 'src/wms/physics/junction_merge.py',
                                   script.parents[2] / 'src/wms/model/sumo_insertion.py',
                                   script.parents[2] / 'src/wms/model/sumo_merge.py',
                                   script.parents[2] / 'src/wms/physics/neighbours.py')},
                  scope=f'FIT native seeds {args.first_seed}-969431; no training/SUMO')
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
