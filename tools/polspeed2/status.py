"""Read only this task's logs plus nvidia-smi; suitable for >=60s SSH polls."""
from pathlib import Path
import json
import subprocess

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
for filename in ('qualification.log', 'qualification3.log', 'suite.log', 'remaining.log'):
    path = base / filename
    print('\n###', path, flush=True)
    if path.exists():
        lines = path.read_text(errors='replace').splitlines()
        for line in lines[-60:]:
            if line.startswith('{'):
                try:
                    row = json.loads(line)
                    print({k: row[k] for k in ('name', 'returncode', 's_per_update', 'peak_owned_gpu_mib') if k in row},
                          'parity', {k: row.get('parity', {}).get(k) for k in ('bitwise_identical', 'n_diff')})
                    continue
                except json.JSONDecodeError:
                    pass
            print(line)
for out in sorted((base/'out').glob('*')):
    print('\n###', out.name, flush=True)
    if (out/'RESULT.json').exists():
        result = json.loads((out/'RESULT.json').read_text())
        print({k: result.get(k) for k in ('returncode', 's_per_update', 'peak_owned_gpu_mib')},
              'parity', {k: result.get('parity', {}).get(k) for k in ('compared_leaves', 'bitwise_identical', 'n_diff')})
        continue
    for filename in ('console.log', 'memory_execution.jsonl', 'RESULT.json'):
        path = out / filename
        if not path.exists():
            path = base/'launch'/out.name/filename
        if path.exists():
            lines = path.read_text(errors='replace').splitlines()
            print(filename, '\n'.join(lines[-5:]), flush=True)
    for log in out.glob('future_workers/*/worker*.log'):
        if log.stat().st_size:
            print(log, log.read_text(errors='replace')[-10000:], flush=True)
subprocess.run(['nvidia-smi'])
