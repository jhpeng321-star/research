"""Run once on gpuA; prepare only the task-owned output/runtime."""
from pathlib import Path
import subprocess
import tarfile

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
assert base.parent.is_dir()
base.mkdir(exist_ok=True)
runtime = base / 'runtime'
runtime.mkdir(exist_ok=True)
with tarfile.open(base.parent / 'polspeed2_source.tar.gz') as archive:
    archive.extractall(runtime, filter='data')
subprocess.run(['/root/miniconda3/bin/python', '-m', 'pip', 'install', 'cloudpickle>=3.1,<4'], check=True)
print('prepared', runtime, flush=True)
