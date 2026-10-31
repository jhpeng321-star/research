"""Read-only proof that HOLD has admitted no further audit SUMO."""
import json
from pathlib import Path
import sys
import time

root = Path('/root/coopaudit20261031')
hold = json.loads((root/'ARTIFACT_HOLD.json').read_text())
worker = hold['worker']
fields = (Path('/proc')/str(worker['pid'])/'stat').read_text().split()
assert fields[21] == worker['proc_start_ticks'] and fields[2] == 'T'
assert hold['active'] is True and (root/'YIELD').exists()
completed = sorted(p.parent.name for p in (root/'jobs').glob('main_*/result.json'))
started = sorted(p.parent.name for p in (root/'jobs').glob('main_*/START.json'))
assert completed == started == hold['completed_main']
ledger = [json.loads(line) for line in (root/'accounting.jsonl').read_text().splitlines()]
charged = sum(r['sim_s'] for r in ledger)
actual = sum(json.loads(p.read_text())['actual_sim_s'] for p in (root/'jobs').glob('*/result.json'))
actual += sum(json.loads(p.read_text())['actual_sim_s'] for p in (root/'jobs').glob('*/failure.json'))
assert actual == hold['actual_sim_s'] and charged == hold['charged_sim_s']
cpu = json.loads((root/'CPU_TESTS.json').read_text())
assert cpu['ok'] and cpu['tests'] == 21
sys.path.insert(0, str(root/('code_'+cpu['version'])))
from tools.coopaudit.release import require_release
try:
    require_release(root)
except ValueError as error:
    assert 'HOLD active' in str(error)
else:
    raise AssertionError('held launch unexpectedly authorised')
print(json.dumps(dict(ok=True, phase='ARTIFACT_HOLD', unix=time.time(),
                      worker_pid=worker['pid'], worker_state=fields[2],
                      completed_main=completed, actual_sim_s=actual,
                      charged_sim_s=charged, additional_sim_s_since_hold=0,
                      cpu=cpu, release_issued=(root/'ARTIFACT_RELEASE.json').exists()), indent=2))
