"""Continue only unmeasured recipes; preserve all completed outputs."""
from pathlib import Path
import json
import subprocess
import sys
import time
from benchmark import memory_failure

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
assert base.is_dir() and not (base/'STOP').exists()
if '--wait' in sys.argv:
    pid = int((base/'COORDINATOR_PID').read_text())
    print('waiting for owned coordinator', pid, flush=True)
    stat = Path(f'/proc/{pid}/stat')
    while True:
        try:
            state = stat.read_text().rsplit(')', 1)[1].split()[0]
        except FileNotFoundError:
            break
        if state == 'Z':
            break
        if (base/'STOP').exists():
            raise SystemExit(0)
        time.sleep(5)
variants = ['six_N2', 'six_N4', 'six_N8', 'nine_N4', 'nine_N8', 'nine_N8_gate', 'nine_N8_full', 'nine_N4_s8']
remaining = []
for name in variants:
    path = base/'out'/name
    if path.exists():
        result = json.loads((path/'RESULT.json').read_text())
        assert 's_per_update' in result or memory_failure(path)
    else:
        remaining.append(name)
assert remaining
with (base/'remaining.log').open('w') as log:
    process = subprocess.Popen([sys.executable, '-u', str(base/'runtime/tools/polspeed2/benchmark.py'),
        *remaining], cwd=base, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
(base/'COORDINATOR_PID').write_text(str(process.pid))
print('coordinator', process.pid, 'recipes', remaining)
