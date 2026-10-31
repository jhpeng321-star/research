"""Read-only, named-root prerequisite probe. No inference or simulator imports."""
from pathlib import Path
import argparse
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DENIED = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
          'hetdata_truth', 'vtype')


def allowed(path):
    text = str(path).lower()
    if any(x in text for x in DENIED):
        raise ValueError('excluded input path')
    return path


def ssh(host, code, python='/root/miniconda3/bin/python'):
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
         python + ' -B -'], input=code,
        text=True, encoding='utf-8', capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('host', choices=('gpuA', 'gpuR', 'cpuQ', 'cpuX'))
    parser.add_argument('--python', default='/root/miniconda3/bin/python')
    parser.add_argument('--file')
    parser.add_argument('--directory')
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--npz')
    parser.add_argument('--runtime', action='store_true')
    args = parser.parse_args()
    if args.runtime:
        code = '''import sys,os,inspect,json
sys.path[:0]=['/root/autodl-tmp/c4/speedhead20261030/tools/speedhead']
import wm_gpu_wmcalib as w
print('frozen',w.frozen.__file__)
print('Entries',inspect.getfile(w.frozen.Entries))
print(inspect.getsource(w.frozen.Entries))
print('config',w.frozen.WM_CONFIG, w.frozen.WM_CONFIG.exists())
print('candidate',w.frozen.CANDIDATE,w.frozen.CANDIDATE.exists())
print('runtime',w.frozen.FIX)
from wms.model.wms import WmSModel
print(inspect.getsource(WmSModel._exec_cap))
'''
    elif args.checkpoint:
        code = '''import torch,json,hashlib
from pathlib import Path
p=Path('/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt')
x=torch.load(p,map_location='cpu',weights_only=False)
print('SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
print('KEYS',list(x))
for k,v in x.items():
 if k in ('format','config'):
  print(k, str(v)[:24000])
'''
    elif args.npz:
        path = allowed(args.npz)
        code = f'''import numpy as np,json
with np.load({path!r},allow_pickle=False) as x:
 print(json.dumps({{k:{{'shape':list(x[k].shape),'dtype':str(x[k].dtype)}} for k in x.files}},indent=2))
'''
    elif args.directory:
        path = allowed(args.directory)
        code = f'''from pathlib import Path
import json
p=Path({path!r})
denied={DENIED!r}
print(json.dumps(sorted(q.name for q in p.iterdir() if not any(x in q.name.lower() for x in denied)),indent=2) if p.is_dir() else 'MISSING')
'''
    elif args.file:
        path = allowed(args.file)
        code = f'from pathlib import Path; print(Path({path!r}).read_text())'
    else:
        code = '''import os,json,subprocess
from pathlib import Path
paths = [Path('/root/wmcalib'), Path('/root/autodl-tmp/c4/wmcombined_wm2'),
         Path('/root/autodl-tmp/c4/speedhead20261030'),
         Path('/root/autodl-tmp/c4/wmdata20261030')]
denied = ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')
print(json.dumps({str(p): sorted(q.name for q in p.iterdir()
    if not any(x in q.name.lower() for x in denied)) if p.is_dir() else None for p in paths},indent=2))
print('load',os.getloadavg(),'cpus',os.cpu_count())
print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader'],text=True))
'''
    print(ssh(args.host, code, args.python))


if __name__ == '__main__':
    main()
