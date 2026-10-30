"""One-update dry handoff, exclusive dispatch, and >=600s GPU monitoring."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET


def write(path, doc):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(dict(label='EXPLORATORY', **doc), stream, indent=1, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def alive(doc):
    path = Path('/proc') / str(doc['pid']) / 'stat'
    if not path.is_file():
        return False
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
        return fields[19] == str(doc['start_ticks']) and fields[0] != 'Z'
    except FileNotFoundError:
        return False


def rows(path):
    if not path.is_file():
        return []
    # Never admit a partially written final line.
    data = path.read_text()
    return [json.loads(line) for line in data.splitlines(keepends=True) if line.endswith('\n')]


def validate_first_update(stage):
    updates = rows(stage / 'updates.jsonl')
    if len(updates) != 1 or updates[0].get('phase_update') != 1:
        raise ValueError('handoff requires exactly one completed dry update')
    row = updates[0]
    if row.get('label') != 'EXPLORATORY':
        raise ValueError('missing exploratory dry stamp')
    for key in ('actor_loss', 'critic_loss', 'actor_grad_norm', 'critic_grad_norm'):
        if not math.isfinite(row[key]):
            raise ValueError('nonfinite dry ' + key)
    score = row['independent_J']
    for key in ('raw_dJ_vs_CTRL0_dense', 'raw_dJ_vs_CTRL0_long'):
        if not math.isfinite(score[key]):
            raise ValueError('nonfinite dry ' + key)
    baseline = json.loads((stage / 'frozen_scoring_baseline.json').read_text())
    if any(baseline['independent'][key] != 0. for key in
           ('raw_dJ_vs_CTRL0_dense', 'raw_dJ_vs_CTRL0_long')):
        raise ValueError('dry phase baseline is not zero == CTRL0')
    summary = json.loads((stage / 'RUN_SUMMARY.json').read_text())
    if summary.get('error') != 'RuntimeError: own-run STOP requested; checkpoint saved':
        raise ValueError('dry did not exit through the owned boundary STOP')
    if not (stage / 'policy_latest.pt').is_file():
        raise ValueError('missing boundary checkpoint')
    return row


def sample_gpu():
    doc = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True))
    gpu = doc.find('gpu')
    def memory(name):
        return int(gpu.findtext('fb_memory_usage/' + name).split()[0])
    return dict(t=time.time(), total_MiB=memory('total'), free_MiB=memory('free'),
                processes=[dict(pid=int(p.findtext('pid')), used_memory=p.findtext('used_memory'))
                           for p in gpu.findall('processes/process_info')])


def may_parallel(sample, own_pid=None):
    peers = [p for p in sample['processes'] if p['pid'] != own_pid]
    needed = 43000 if own_pid is None else .48 * sample['total_MiB'] + 3304
    return not peers and sample['free_MiB'] >= needed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exploratory-owner-20261030', action='store_true', required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--python', required=True)
    a = p.parse_args()
    root = a.root.resolve()
    owner = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                 utc=datetime.now(timezone.utc).isoformat(), runtime=str(a.runtime), dry_root=str(root))
    write(root / 'handoff_OWNERSHIP.json', owner)  # Exclusive lease: never overwrite/re-dispatch.
    dry_owner = json.loads((root / 'dry_OWNERSHIP.json').read_text())
    supervisor = json.loads((root / 'OWNERSHIP.json').read_text())
    if not alive(dry_owner) or not alive(supervisor) or (root / 'pilot_OWNERSHIP.json').exists():
        raise ValueError('original live chain identity mismatch or pilot already dispatched')
    # The original supervisor must never auto-dispatch its dry2 pilot.
    (root / 'STOP').touch(exist_ok=True)
    dry = root / 'dry_EXPLORATORY'
    baseline = dry / 'frozen_scoring_baseline.json'
    while alive(dry_owner):
        if (root / 'handoff_STOP').exists():
            return 1
        completed = rows(dry / 'updates.jsonl')
        # Arm STOP once baseline receipts have been durable for >3s: the first
        # expensive update is in flight, and finishes before STOP is inspected.
        # This avoids racing update2 in an external polling controller.
        if completed or (baseline.is_file() and time.time() - baseline.stat().st_mtime > 3):
            if not (dry / 'STOP').exists():
                (dry / 'STOP').touch()
                write(root / 'dry_BOUNDARY_STOP.json', dict(epoch=time.time(),
                    completed_rows=len(completed), reason='owner22:30 one-update boundary STOP'))
        time.sleep(1)
    while alive(supervisor):
        time.sleep(1)
    first = validate_first_update(dry)
    write(root / 'dry_FIRST_UPDATE_VALIDATED.json', dict(row=first, fresh_pilot_from_dry=False))
    samples = root / 'handoff_GPU.jsonl'
    last_query = rows(root / 'GPU_ADMISSION.jsonl')[-1]['t']
    child = None
    parallel_written = False
    while True:
        if (root / 'handoff_STOP').exists():
            if child:
                (root / 'pilot_EXPLORATORY' / 'STOP').touch()
            return 1
        if child and child.poll() is not None:
            write(root / 'handoff_pilot_EXIT.json', dict(returncode=child.returncode))
            return child.returncode
        if time.time() - last_query < 600:
            time.sleep(min(5, 600 - (time.time() - last_query)))
            continue
        sample = sample_gpu()
        last_query = sample['t']
        with samples.open('a') as stream:
            stream.write(json.dumps(dict(label='EXPLORATORY', **sample)) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        output = root / 'pilot_EXPLORATORY'
        if child is None:
            if sample['free_MiB'] < 20500:
                continue
            env = dict(os.environ, RUNTIME=str(a.runtime), PYTHON=a.python,
                       PILOT_ADMISSION_FREE_MIB=str(sample['free_MiB']),
                       PILOT_ADMISSION_EPOCH=str(sample['t']))
            command = ['bash', 'tools/candB/pilot_recipe_launch.sh', str(output),
                       '--exploratory-owner-20261030']
            log = (root / 'pilot.log').open('x')
            child = subprocess.Popen(command, cwd=a.runtime, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, env=env)
            log.close()
            ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            write(root / 'pilot_OWNERSHIP.json', dict(pid=child.pid, start_ticks=ticks,
                argv=command, output=str(output), runtime=str(a.runtime),
                utc=datetime.now(timezone.utc).isoformat()))
            # No trainer.update occurs until its long frozen baseline has completed.
            if may_parallel(sample):
                for _ in range(60):
                    if output.is_dir() or child.poll() is not None:
                        break
                    time.sleep(1)
                if output.is_dir():
                    write(output / 'TWO_FORWARD_WORKERS.json', dict(forward_workers=2, admission=sample))
                    parallel_written = True
        elif not parallel_written and may_parallel(sample, child.pid):
            write(output / 'TWO_FORWARD_WORKERS.json', dict(forward_workers=2, admission=sample))
            parallel_written = True


if __name__ == '__main__':
    raise SystemExit(main())
