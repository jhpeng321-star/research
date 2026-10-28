"""Background coordinator entry point; all artifacts stay under BASE."""
from pathlib import Path
import os
import subprocess
import sys

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
runtime = base / 'runtime'
env = dict(os.environ, PYTHONPATH=f'{runtime}/src:{runtime}',
           C4_ALLOWED_ROOTS=str(base), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
with (base/'qualification.log').open('w') as stream:
    result = subprocess.run([sys.executable, '-m', 'pytest',
        'tests/unit/test_polspeed2_parallel.py', 'tests/unit/test_auditfix_sixrow.py', '-q'],
        cwd=runtime, env=env, stdout=stream, stderr=subprocess.STDOUT)
print('qualification exit', result.returncode, flush=True)
if result.returncode == 0 and not (base/'STOP').exists():
    subprocess.run([sys.executable, str(runtime/'tools/polspeed2/benchmark.py'),
        'nine_N2', 'six_serial', 'six_N2', 'six_N4', 'six_N8'], env=env, cwd=base, check=True)
