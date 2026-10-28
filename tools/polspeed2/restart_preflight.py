"""Preserve the failed output-directory preflight, then launch corrected argv."""
from pathlib import Path
import json
import subprocess
import sys

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
reason = sys.argv[1] if len(sys.argv) > 1 else 'existing_output'
expected = {'existing_output': 'FileExistsError', 'starts_cwd': 'configs/c4_policy14_starts.toml'}[reason]
out = base/'out'
assert out.is_dir() and not (base/f'failed_{reason}').exists()
for directory in out.iterdir():
    result = json.loads((directory/'RESULT.json').read_text())
    assert result['returncode'] == 1 and 's_per_update' not in result
    assert expected in (directory/'console.log').read_text()
out.rename(base/f'failed_{reason}')
if (base/'launch').exists():
    (base/'launch').rename(base/f'failed_{reason}_launch')
(base/'suite.log').rename(base/f'suite_{reason}.log')
with (base/'suite.log').open('w') as log:
    process = subprocess.Popen([sys.executable, '-u', str(base/'runtime/tools/polspeed2/benchmark.py'),
        'nine_N2', 'six_serial', 'six_N2', 'six_N4', 'six_N8'], cwd=base,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(base/'COORDINATOR_PID').write_text(str(process.pid))
print('coordinator', process.pid)
