"""Task-owned SSH transport with explicit path denylist and no shell data quoting."""
from pathlib import Path
import base64
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')


def ssh(source, host='cpuQ', data=None, python='python3'):
    assert host in ('cpuQ', 'gpuR')
    assert python in ('python3', '/home/ubuntu/miniconda3/bin/python')
    payload = base64.b64encode(source.encode()).decode()
    command = python + " -c 'import base64;exec(base64.b64decode(\"" + payload + "\"))'"
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, command],
                       input=data, capture_output=True, timeout=300)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace') + p.stdout.decode(errors='replace'))
    return p.stdout


def inspect(paths):
    for p in paths:
        assert not any(x in p.lower() for x in BAD), p
    source = '''
from pathlib import Path
import json, os, shutil
for name in PATHS:
 p=Path(name).expanduser()
 print('FILE',str(p))
 if p.is_dir():
  print(json.dumps([f.name for f in p.iterdir() if not any(x in f.name.lower() for x in BAD)]))
 elif p.exists():
  print(p.read_text())
 else: print('MISSING')
'''
    return ssh('PATHS='+repr(paths)+'\nBAD='+repr(BAD)+'\n'+source).decode()


if __name__ == '__main__':
    print(inspect(sys.argv[1:]))
