"""Continue only unmeasured recipes; preserve all completed outputs."""
from pathlib import Path
import json
import subprocess
import sys

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
assert base.is_dir() and not (base/'STOP').exists()
variants = ['six_N2', 'six_N4', 'six_N8', 'nine_N4', 'nine_N8']
remaining = []
for name in variants:
    path = base/'out'/name
    if path.exists():
        assert 's_per_update' in json.loads((path/'RESULT.json').read_text())
    else:
        remaining.append(name)
assert remaining
with (base/'remaining.log').open('w') as log:
    process = subprocess.Popen([sys.executable, '-u', str(base/'runtime/tools/polspeed2/benchmark.py'),
        *remaining], cwd=base, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
(base/'COORDINATOR_PID').write_text(str(process.pid))
print('coordinator', process.pid, 'recipes', remaining)
