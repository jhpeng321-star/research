"""Bounded read-only polling of this task's named receipts."""
import json
import sys
import time

from tools.longint.resume_c3 import ARM, OUT, REL


def watch(wait=0):
    assert 0 <= wait <= 30
    time.sleep(wait)
    status = json.loads((REL / 'AUTONOMY_STATUS.json').read_text())
    result = {k: status.get(k) for k in ('t', 'status', 'charged', 'reason', 'exit')}
    state = json.loads((ARM / 'loop/LOOP14_STATE.json').read_text())
    result['phases'] = {k: v['status'] for k, v in state['cycles'][-1]['phases'].items()}
    policy = state['cycles'][-1]['phases']['policy']
    directory = ARM / 'loop/policy' / ('c03.a' + str(policy['attempt']))
    peaks = directory / 'cuda_peaks.jsonl'
    if peaks.exists():
        rows = [json.loads(x) for x in peaks.read_text().splitlines()]
        result['completed_retry_updates'] = len(rows)
        result['last_update'] = rows[-1] if rows else None
    memory = OUT / 'GPU_MEMORY.jsonl'
    if memory.exists():
        rows = [json.loads(x) for x in memory.read_text().splitlines()]
        result['gpu_sample'] = rows[-1] if rows else None
        result['process_peak_mib'] = max((r['used_mib'] for r in rows), default=0)
    journal = ARM / 'loop/worlds_c03.json'
    if journal.exists():
        doc = json.loads(journal.read_text())
        result['worlds_done'] = len(doc['worlds'])
        result['guard'] = doc.get('native_smoke')
    if status['status'].endswith('STOPPED') or status['status'] == 'OPERATIONAL_PAUSED':
        result['error'] = policy.get('error')
    print(json.dumps(result))


if __name__ == '__main__':
    watch(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
