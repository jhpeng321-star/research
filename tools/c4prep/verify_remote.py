"""CPU-only verification in the owned immutable runtime; short foreground call."""
from pathlib import Path
import json
import os
import subprocess

R = Path('/root/autodl-tmp/c4/restart20261026')
OUT = R / 'arm/loop/c4prep'
CODE = Path(json.loads((OUT / 'CURRENT_RUNTIME.json').read_text())['runtime'])
assert CODE.is_dir()
OUT.mkdir(exist_ok=True)
env = {**os.environ, 'PYTHONPATH': f'{CODE}/src:{CODE}', 'C4_ALLOWED_ROOTS': '/root/autodl-tmp/c4',
       'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
args = ['/root/miniconda3/bin/python', '-B', '-m', 'pytest', '-q',
        'tests/unit/test_c4prep.py', '--basetemp=' + str(CODE / 'pytest-c4prep')]
tag = json.loads((CODE / 'SOURCE.json').read_text())['commit'][:12]
with (OUT / f'CPU_TESTS_{tag}.log').open('x') as log:
    result = subprocess.run(args, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=25)
with (OUT / f'CPU_TESTS_{tag}.json').open('x') as stream:
    json.dump({'exit': result.returncode, 'argv': args}, stream)
print((OUT / f'CPU_TESTS_{tag}.log').read_text())
