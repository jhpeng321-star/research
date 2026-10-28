"""Run once on gpuA; prepare only the task-owned output/runtime."""
from pathlib import Path
import subprocess
import tarfile
import sys

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
assert base.parent.is_dir()
base.mkdir(exist_ok=True)
runtime = base / 'runtime'
runtime.mkdir(exist_ok=True)
with tarfile.open(base.parent / 'polspeed2_source.tar.gz') as archive:
    archive.extractall(runtime, filter='data')
subprocess.run(['/root/miniconda3/bin/python', '-m', 'pip', 'install', 'cloudpickle>=3.1,<4'], check=True)
print('prepared', runtime, flush=True)
if '--start' in sys.argv:
    with (base / 'suite.log').open('w') as stream:
        process = subprocess.Popen(['/root/miniconda3/bin/python', '-u',
            str(runtime/'tools/polspeed2/qualify_and_run.py')], cwd=base,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
    (base/'COORDINATOR_PID').write_text(str(process.pid))
    print('coordinator', process.pid, flush=True)
