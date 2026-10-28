"""Run focused tests on the staged, CPU-only task runtime."""
import sys
from remote import ssh

ssh(sys.argv[1], '''
import os,subprocess
from pathlib import Path
p=Path.home()/'entrant/runtime';assert p.is_dir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(p/'src'),OMP_NUM_THREADS='1')
subprocess.run([str(Path.home()/'miniconda3/bin/python'),'-m','pytest',
    'tests/unit/test_entrant.py','-q'],cwd=p,env=env,check=True)
''')
