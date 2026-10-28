"""Bounded read-only receipt summary; no model or simulator execution."""
from pathlib import Path
import hashlib
import json
import subprocess

R = Path('/root/autodl-tmp/c4/restart20261026')
O = R / 'arm/loop/c4prep'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


result = {'task': 'S-C4PREP-1', 'runtime': json.loads((O/'CURRENT_RUNTIME.json').read_text())}
pre = json.loads((O/'INPUT_PREFLIGHT_1.json').read_text())
result['input_preflight'] = {'sha256': sha(O/'INPUT_PREFLIGHT_1.json'), 'ready': pre['ready'],
    'episodes': sum(len(c['episodes']) for c in pre['cycles']),
    'pairs': sum(c['pairs'] for c in pre['cycles']),
    'missing_clock_episodes': sum(bool(e['missing_observable_arrays']) for c in pre['cycles'] for e in c['episodes']),
    'actors_verified': {str(c['cycle']): c['actor_verified'] for c in pre['cycles']}, 'blockers': pre['blockers']}
tag = result['runtime']['commit'][:12]
result['cpu_tests'] = json.loads((O/f'CPU_TESTS_{tag}.json').read_text())
result['cpu_tests']['log_sha256'] = sha(O/f'CPU_TESTS_{tag}.log')
result['cpu_tests']['summary'] = (O/f'CPU_TESTS_{tag}.log').read_text().strip().splitlines()[-1]
result['configs'] = {}
for mode in ('ON', 'OFF'):
    p = O/f'CONFIG_{mode}.json'
    doc = json.loads(p.read_text())
    result['configs'][mode] = {'path': str(p), 'sha256': sha(p), 'ready': doc['ready']}
result['gate_complete'] = (R/'arm/loop/checks/c03.a2/HORIZON_GATE.json').is_file()
result['autonomy_status'] = json.loads((R/'AUTONOMY_STATUS.json').read_text())
state = json.loads((R/'arm/loop/LOOP14_STATE.json').read_text())
result['charged'] = state['sim_s_charged']
result['stop_driver_preserved'] = (R/'arm/loop/STOP_DRIVER').is_file()
result['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader'],text=True).strip()
print(json.dumps(result, indent=1))
