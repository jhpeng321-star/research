"""Finite-path, read-only progress for the already-dispatched ranking shards."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

BASE = Path('/root/autodl-tmp/c4')
RESUME = BASE / 'candB_rankcheck_82dbd7bc_resume_20261030'


def inspect(directory):
    ownership = directory / 'OWNERSHIP.json'
    if not ownership.exists():
        return dict(directory=str(directory), dispatched=False)
    owned = json.loads(ownership.read_text())
    try:
        tick = Path(f'/proc/{owned["pid"]}/stat').read_text().split(') ')[1].split()[19]
    except FileNotFoundError:
        tick = None
    result = dict(directory=str(directory), pid=owned['pid'], start_ticks=owned['start_ticks'],
                  live=tick == str(owned['start_ticks']))
    raw = directory / 'RAW.json'
    if raw.exists():
        try:
            doc = json.loads(raw.read_text())
            result.update(completed_arms=dict(Counter(arm for row in doc['rows'] for arm in row['arms'])),
                          plan_indices=[row.get('plan_index', i + doc.get('start_index', 0))
                                        for i, row in enumerate(doc['rows'])],
                          shard_finished=doc.get('shard_finished', False),
                          mtime=raw.stat().st_mtime)
        except json.JSONDecodeError:
            result['raw_status'] = 'write in progress; retain last collected complete receipt'
    log = directory / 'run.log'
    if log.exists():
        result['log_tail'] = log.read_text()[-1800:].splitlines()[-3:]
    return result


host = sys.argv[1]
assert host in ('A', 'R')
if host == 'A':
    paths = [BASE / 'candB_rankcheck_2dcb6af5_host_20261030/measurement_CIRCLES',
             BASE / 'candB_rankcheck_3cd6481e_plan_20261030/measurement_MAPPO_native_lanes+MAPPO_speed_partial',
             BASE / 'candB_rankcheck_f396955a_tail_20261030/measurement_MAPPO_native_lanes_tail',
             BASE / 'candB_rankcheck_f396955a_tail_20261030/measurement_MAPPO_speed_partial_tail']
else:
    paths = [RESUME / 'measurement_PI_native_cuda_12_16_a2',
             RESUME / 'measurement_PI_native_cpu_11_12_a2',
             *[RESUME / f'measurement_native_cpu_{begin}_{end}'
               for begin, end in ((0, 3), (3, 6), (6, 9), (9, 11))]]
result = dict(time=datetime.now().astimezone().isoformat(), host=host,
              gpu_memory=subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                                  '--format=csv,noheader'], text=True).strip(),
              jobs=[inspect(path) for path in paths], read_only=True)
if len(sys.argv) == 3:
    assert re.fullmatch(r'SNAPSHOT_[A-Z0-9_]+\.json', sys.argv[2])
    root = RESUME if host == 'R' else BASE / 'candB_rankcheck_2dcb6af5_host_20261030'
    with (root / sys.argv[2]).open('x') as stream:
        json.dump(result, stream, indent=1)
print(json.dumps(result, indent=1))
