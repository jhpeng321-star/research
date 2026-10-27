"""Sequential, bounded SSH operations for S-LONGLOSSTEST-1."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = '/root/autodl-tmp/c4/longloss20261027'
OLD = '/root/autodl-tmp/c4/wmdeep20261027'


def remote(host, source, timeout=120):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
                        '/root/miniconda3/bin/python -B -'], input=source,
                       text=True, encoding='utf-8', capture_output=True, timeout=timeout)
    print(p.stdout, end='')
    print(p.stderr, end='')
    p.check_returncode()


def inspect():
    remote('gpuR', f'''import pathlib,json,shutil
p=pathlib.Path({OLD!r})
print('DISK_GIB',shutil.disk_usage(p).free/2**30)
for n in ['DESIGN.json','TRAINING.json','RESULTS.json']:
 print(n,(p/n).read_text())
for n in ['views/loop14_pairs/C4_EPISODES_MANIFEST.json']:
 d=json.loads((p/n).read_text());print(n,'META',{{k:v for k,v in d.items() if k!='episodes'}},'EXAMPLE',d['episodes'][0])
print('R_TO_A',__import__('subprocess').run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','gpuA','true']).returncode)
''')
    remote('gpuA', '''import pathlib,json
p=pathlib.Path('/root/autodl-tmp/c4/restart20261026/arm/loop')
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')
for name in ['data','returns']:
 q=p/name
 if q.is_dir():print(name,[x.name for x in q.iterdir() if not any(t in str(x).lower() for t in bad)])
q=p/'LOOP14_STATE.json'
if q.is_file():
 d=json.loads(q.read_text())
 for c in d['cycles']:
  print('CYCLE',c['cycle'],'CONVERT',json.dumps(c['phases'].get('convert',{})))
''')


def state():
    remote('gpuR', f'''import pathlib,json,shutil,subprocess
p=pathlib.Path({OUT!r}); old=pathlib.Path({OLD!r})
print('LONGLOSS_EXISTS',p.exists(),'DISK_GIB',shutil.disk_usage(old).free/2**30)
if p.exists():
 print('CONTENTS',[x.name for x in p.iterdir()])
 for n in ['LAUNCH.json','RUNNING.json','DONE.json','STATE.json']:
  q=p/n
  if q.is_file():print(n,q.read_text())
print('OLD_CONTENTS',[x.name for x in old.iterdir() if not any(t in x.name.lower() for t in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1'))])
print('GPU',subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout)
''')


def sources():
    remote('gpuR', f'''from pathlib import Path
p=Path({OLD!r})
for n in ['job.py','offline_chain.py','deep_config.json','SPLIT.json']:
 print('FILE',n,(p/n).read_text())
''')


def contracts():
    remote('gpuR', f'''from pathlib import Path
import json
p=Path({OLD!r})
print('CONFIG',(p/'runtime/configs/c4_wm_loop14_r16_ft.toml').read_text())
for name in ['native_v5','loop14_pairs']:
 d=json.loads((p/'views'/name/'C4_EPISODES_MANIFEST.json').read_text())
 print(name,{{k:v for k,v in d.items() if k!='episodes'}},'ROLES',sorted(set(e.get('split_role','') for e in d['episodes'])))
''')
    remote('gpuA', '''from pathlib import Path
import json
p=Path('/root/autodl-tmp/c4/restart20261026/arm/loop/returns')
for c in ['c00.a1','c01.a1','c02.a1']:
 d=json.loads((p/('pairing_'+c+'.json')).read_text())
 print(c,{k:v for k,v in d.items() if k!='pairs'},'FIRST_PAIR',d['pairs'][0])
''')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['inspect', 'state', 'sources', 'contracts'])
    args = ap.parse_args()
    globals()[args.action]()
