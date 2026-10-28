"""Preserve the failed CLI-argument preflight and start the corrected benchmark."""
from pathlib import Path
import json
import subprocess
import sys

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
out = base/'out'
assert out.is_dir() and not (base/'failed_argparse').exists()
for directory in out.iterdir():
    result = json.loads((directory/'RESULT.json').read_text())
    assert result['returncode'] == 2 and 's_per_update' not in result
    assert 'invalid choice: 1 (choose from 2)' in (directory/'console.log').read_text()
out.rename(base/'failed_argparse')
(base/'suite.log').rename(base/'suite_argparse.log')
with (base/'suite.log').open('w') as log:
    process = subprocess.Popen([sys.executable, '-u', str(base/'runtime/tools/polspeed2/benchmark.py'),
        'nine_N2', 'six_serial', 'six_N2', 'six_N4', 'six_N8'], cwd=base,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(base/'COORDINATOR_PID').write_text(str(process.pid))
print('coordinator', process.pid)
