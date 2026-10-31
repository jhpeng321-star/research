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


def ssh(host, code):
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
         '/root/miniconda3/bin/python -B -'], input=code,
        text=True, encoding='utf-8', capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('host', choices=('gpuA', 'gpuR'))
    parser.add_argument('--file')
    parser.add_argument('--directory')
    parser.add_argument('--checkpoint', action='store_true')
    args = parser.parse_args()
    if args.checkpoint:
        code = '''import torch,json,hashlib
from pathlib import Path
p=Path('/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt')
x=torch.load(p,map_location='cpu',weights_only=False)
print('SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
print('KEYS',list(x))
for k,v in x.items():
 if k not in ('state_dict','model','model_state_dict','optimizer','optimizer_state_dict','rng_state'):
  print(k, str(v)[:24000])
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
    print(ssh(args.host, code))


if __name__ == '__main__':
    main()
