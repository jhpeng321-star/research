"""Exclusive owner2240 same-phase handoff; never dispatch a fresh-zero pilot."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from tools.candB.pilot_handoff import alive, rows, sample_gpu, may_parallel, write, validate_first_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exploratory-owner-20261030', action='store_true', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    runtime = Path.cwd()
    write(root / 'continue_OWNERSHIP.json', dict(pid=os.getpid(),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        utc=datetime.now(timezone.utc).isoformat(), runtime=str(runtime)))
    dry_owner = json.loads((root / 'dry_OWNERSHIP.json').read_text())
    supervisor = json.loads((root / 'OWNERSHIP.json').read_text())
    if (root / 'pilot_OWNERSHIP.json').exists() or (root / 'handoff_OWNERSHIP.json').exists():
        raise ValueError('another dispatcher exists; refuse duplicate continuation')
    (root / 'STOP').touch(exist_ok=True)
    dry = root / 'dry_EXPLORATORY'
    while alive(dry_owner):
        if (root / 'continue_STOP').exists():
            return 1
        baseline = dry / 'frozen_scoring_baseline.json'
        # The existing CLI checks STOP only at update boundaries. Wait30s after
        # durable baseline completion to allow its first update to enter.
        if rows(dry / 'updates.jsonl') or (baseline.is_file() and time.time()-baseline.stat().st_mtime > 30):
            if not (dry / 'STOP').exists():
                (dry / 'STOP').touch()
                write(root / 'continue_BOUNDARY_STOP.json', dict(epoch=time.time(),
                    reason='owner2240 preserve same phase and frozen baseline',
                    completed_rows=len(rows(dry / 'updates.jsonl'))))
        time.sleep(1)
    while alive(supervisor):
        time.sleep(1)
    first = validate_first_update(dry)
    write(root / 'continue_FIRST_UPDATE_VALIDATED.json', dict(row=first, fresh_zero_restart=False))
    last_query = rows(root / 'GPU_ADMISSION.jsonl')[-1]['t']
    child = None
    while True:
        if (root / 'continue_STOP').exists():
            if child:
                (root / 'pilot_samephase_EXPLORATORY' / 'STOP').touch()
            return 1
        if child and child.poll() is not None:
            write(root / 'continue_EXIT.json', dict(returncode=child.returncode))
            return child.returncode
        if time.time()-last_query < 600:
            time.sleep(min(2, 600-(time.time()-last_query)))
            continue
        sample = sample_gpu()
        last_query = sample['t']
        with (root / 'continue_GPU.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(label='EXPLORATORY', **sample))+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        if child is not None or not may_parallel(sample):
            continue
        output = root / 'pilot_samephase_EXPLORATORY'
        env = dict(os.environ, RUNTIME=str(runtime), PYTHON=args.python,
                   PILOT_ADMISSION_FREE_MIB=str(sample['free_MiB']), PILOT_ADMISSION_EPOCH=str(sample['t']))
        command = ['bash', 'tools/candB/pilot_continue_launch.sh', str(output), str(dry),
                   '--exploratory-owner-20261030']
        with (root / 'continue.log').open('x') as log:
            child = subprocess.Popen(command, cwd=runtime, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, env=env)
        write(root / 'continue_trainer_OWNERSHIP.json', dict(pid=child.pid,
            start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19],
            argv=command, output=str(output), runtime=str(runtime),
            utc=datetime.now(timezone.utc).isoformat(), admission=sample))


if __name__ == '__main__':
    raise SystemExit(main())
