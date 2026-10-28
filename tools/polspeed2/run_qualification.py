"""Detached, logged qualification; SSH disconnect cannot orphan test workers."""
from pathlib import Path
import json
import os
import subprocess
import sys

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
runtime = base/'runtime'
assert base.is_dir() and not (base/'STOP').exists()
env = dict(os.environ, PYTHONPATH=f'{runtime}/src:{runtime}',
           C4_ALLOWED_ROOTS=str(base), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
with (base/'qualification3.log').open('w') as log:
    process = subprocess.Popen([sys.executable, '-m', 'pytest',
        'tests/unit/test_polspeed2_parallel.py', 'tests/unit/test_auditfix_sixrow.py',
        '--basetemp='+str(base/'qualification3_tmp'), '-q'], cwd=runtime, env=env,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(base/'QUALIFICATION_PID.json').write_text(json.dumps(dict(pid=process.pid)))
print('qualification pid', process.pid)
