"""Read only this task's logs plus nvidia-smi; suitable for >=60s SSH polls."""
from pathlib import Path
import subprocess

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
for filename in ('qualification.log', 'suite.log'):
    path = base / filename
    print('\n###', path, flush=True)
    if path.exists():
        lines = path.read_text(errors='replace').splitlines()
        print('\n'.join(lines[-100:]))
for out in sorted((base/'out').glob('*')):
    print('\n###', out.name, flush=True)
    for filename in ('console.log', 'memory_execution.jsonl', 'RESULT.json'):
        path = out / filename
        if not path.exists():
            path = base/'launch'/out.name/filename
        if path.exists():
            lines = path.read_text(errors='replace').splitlines()
            print(filename, '\n'.join(lines[-12:]), flush=True)
    for log in out.glob('future_workers/*/worker*.log'):
        if log.stat().st_size:
            print(log, log.read_text(errors='replace')[-10000:], flush=True)
subprocess.run(['nvidia-smi'])
