"""Archive an owned conservative whole-device STOP and re-budget owned PIDs."""
from pathlib import Path
import json
import os
import subprocess
import sys

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
old = base/'out/nine_N2'
result = json.loads((old/'RESULT.json').read_text())
assert 's_per_update' not in result and result['peak_total_gpu_fraction'] > .60
assert 'future worker exited (0)' in (old/'console.log').read_text()
assert (base/'STOP').exists()
archive = base/'failed_shared_monitor'
archive.mkdir(exist_ok=False)
old.rename(archive/'nine_N2')
(base/'launch/nine_N2').rename(archive/'launch_nine_N2')
(base/'STOP').rename(archive/'STOP')
(base/'suite.log').rename(archive/'suite.log')
env = dict(os.environ, POLSPEED_STARTUP_SUMMARY=str(archive/'nine_N2/RUN_SUMMARY.json'))
with (base/'suite.log').open('w') as log:
    process = subprocess.Popen([sys.executable, '-u', str(base/'runtime/tools/polspeed2/benchmark.py'),
        'nine_N2', 'six_serial', 'six_N2', 'six_N4', 'six_N8'], cwd=base, env=env,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(base/'COORDINATOR_PID').write_text(str(process.pid))
print('coordinator', process.pid)
