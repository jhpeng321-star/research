"""Read canonical policy arguments/config without dispatching shared phases."""
from pathlib import Path
import json
R = Path('/root/autodl-tmp/c4/restart20261026')
for rel in ('arm/CONFIG_C3_LONGINT.json', 'arm/CONFIG.json'):
    p = R / rel
    if p.is_file():
        d = json.loads(p.read_text())
        print(rel, json.dumps({k: d[k] for k in ('policy', 'looplong', 'probe') if k in d}))
p = R / 'arm/loop/longint/c3resume/r2/RECOVERY.json'
d = json.loads(p.read_text())
print('recovery_keys', list(d))
for k in ('args', 'argv', 'command', 'policy_command'):
    if k in d:
        print(k, d[k])
for rel in ('arm/loop/policy/c03.a6.log', 'arm/loop/policy/c03.a6/RUN_SUMMARY.json'):
    p = R / rel
    if p.is_file():
        if p.suffix == '.log':
            with p.open() as stream:
                print('trainer_command', stream.readline())
        else:
            d = json.loads(p.read_text())
            print('boundaries', {k: v for k, v in d.items() if k in ('looplong_initial_boundary', 'looplong_final_boundary', 'real_credit')})
print('gate_done', (R/'arm/loop/checks/c03.a2/HORIZON_GATE.json').is_file())
