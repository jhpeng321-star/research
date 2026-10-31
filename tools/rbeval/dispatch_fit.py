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
    args = p.parse_args()
    work = Path(args.work).resolve()
    if not work.is_dir():
        p.error('existing task-owned work directory required')
    receipt = work / 'remaining_fit_dispatch.json'
    # Reserve before Popen, so a repeated invocation cannot duplicate the job.
    with receipt.open('x') as f:
        json.dump({'state': 'reserved', 'owner': 'S-RB-MERGE'}, f)
    script = Path(__file__).with_name('validate_merge.py')
    command = [sys.executable, str(script), '--root', args.root, '--seeds',
               *map(str, range(969402, 969432)), '--horizon', '300',
               '--target-csv', str(script.with_name('ramp_delay_world.csv')),
               '--output', str(work / 'fit_969402_969431_300.json'), '--quiet']
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (work / 'remaining_fit.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True, env=env,
                                 preexec_fn=lambda: os.nice(10))
    record = dict(owner='S-RB-MERGE', pid=child.pid, state='launched', command=command,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  validator_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                  scope='remaining FIT native seeds 969402-969431; first two already complete; no training/SUMO')
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
