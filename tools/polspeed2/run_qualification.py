"""Detached, logged qualification; SSH disconnect cannot orphan test workers."""
from pathlib import Path
import json
import os
import subprocess
import sys
import argparse

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
runtime = base/'runtime'
assert base.is_dir() and not (base/'STOP').exists()
parser = argparse.ArgumentParser()
parser.add_argument('--tag', choices=('qualification3', 'qualification4', 'qualification5'), default='qualification3')
parser.add_argument('--gated-only', action='store_true')
args = parser.parse_args()
env = dict(os.environ, PYTHONPATH=f'{runtime}/src:{runtime}',
           C4_ALLOWED_ROOTS=str(base), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
tests = (['tests/unit/test_polspeed2_parallel.py::test_gated_vjps_preserve_full_update[cuda]']
         if args.gated_only else ['tests/unit/test_polspeed2_parallel.py', 'tests/unit/test_auditfix_sixrow.py',
                                  'tests/unit/test_polspeed2_report.py'])
with (base/(args.tag+'.log')).open('w') as log:
    process = subprocess.Popen([sys.executable, '-m', 'pytest',
        *tests, '--basetemp='+str(base/(args.tag+'_tmp')), '-q'], cwd=runtime, env=env,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(base/'QUALIFICATION_PID.json').write_text(json.dumps(dict(pid=process.pid)))
print('qualification pid', process.pid)
