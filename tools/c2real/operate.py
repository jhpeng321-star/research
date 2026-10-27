"""Explicit task-only transport; all numerical analysis runs on gpuA CPU."""
import argparse
import base64
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/root/autodl-tmp/c4/restart20261026/c2real'
REGISTER = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')


def execute(source, timeout=110):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'gpuA',
                             '/root/miniconda3/bin/python -B -'], input=source,
                            text=True, encoding='utf-8', capture_output=True, timeout=timeout)
    print(result.stdout, end='')
    print(result.stderr, end='')
    result.check_returncode()


def stage():
    paths = {'analyze.py': ROOT/'tools/c2real/analyze.py',
             'coopbehav.py': Path('E:/agent-work/c4-coopmeas/traffic_shepherd/c4/coopbehav.py'),
             'thresholds.json': Path('E:/agent-work/c4-coopmeas/evidence_coopmeas/thresholds.json')}
    if (ROOT/'tools/c2real/imagine.py').exists():
        paths['imagine.py'] = ROOT/'tools/c2real/imagine.py'
    paths['verify.py'] = ROOT/'tools/c2real/verify.py'
    paths['coverage.py'] = ROOT/'tools/c2real/coverage.py'
    payload = {n: base64.b64encode(p.read_bytes()).decode() for n, p in paths.items()}
    reg = REGISTER.read_text(encoding='utf-8').split('## S-C2REAL-1', 1)[1]
    execute(f'''import pathlib,base64,hashlib,json,os
os.nice(19)
r=pathlib.Path({REMOTE!r});assert r.parent.is_dir();r.mkdir(exist_ok=True)
for n,b in {payload!r}.items():
 p=r/n;data=base64.b64decode(b)
 if p.exists() and p.read_bytes()!=data:
  old=r/(n+'.'+hashlib.sha256(p.read_bytes()).hexdigest()+'.before')
  if not old.exists():old.write_bytes(p.read_bytes())
 p.write_bytes(data)
(r/'AUTHORITY.md').write_text('## S-C2REAL-1'+{reg!r})
assert hashlib.sha256((r/'thresholds.json').read_bytes()).hexdigest().startswith('9119cd77')
print(json.dumps({{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in {list(payload)!r}}}))
''')


def run():
    execute(f'''import os,subprocess,pathlib,json
r=pathlib.Path({REMOTE!r});assert r.is_dir()
env={{**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}}
with (r/'analysis.log').open('ab') as log:
 p=subprocess.Popen(['/usr/bin/nice','-n','19','/root/miniconda3/bin/python','-B',str(r/'analyze.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
f=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()
receipt={{'pid':p.pid,'start_ticks':f[19],'task':'S-C2REAL-1 analysis','sid':os.getsid(p.pid)}}
(r/'PID.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
(r/('PID.'+str(p.pid)+'.json')).write_text(json.dumps(receipt))
''')


def status():
    execute(f'''import pathlib,json,os
os.nice(19);r=pathlib.Path({REMOTE!r})
for n in ['PID.json','RESULTS.json']:
 p=r/n
 if p.exists():print(n,p.read_text())
p=r/'analysis.log'
if p.exists():print('LOG',p.read_text()[-2000:])
''')


def fetch():
    names = ['C2REAL.csv', 'RESULTS.json', 'CENSUS.json', 'INPUTS.json', 'VERIFIED.json',
              'AUTHORITY.md', 'coopbehav.py', 'thresholds.json', 'COVERAGE.json']
    code = f'''import pathlib,base64,json,os
os.nice(19);r=pathlib.Path({REMOTE!r});print(json.dumps({{n:base64.b64encode((r/n).read_bytes()).decode() for n in {names!r} if (r/n).exists()}}))'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'gpuA',
                             '/root/miniconda3/bin/python -B -'], input=code,
                            text=True, encoding='utf-8', capture_output=True, check=True)
    dest = ROOT/'tools/c2real/evidence'
    dest.mkdir(exist_ok=True)
    for name, b64 in json.loads(result.stdout).items():
        (dest/name).write_bytes(base64.b64decode(b64))
    paper = Path('E:/research-dense/docs/paper_results')
    assert paper.is_dir()
    (paper/'C2REAL.csv').write_bytes((dest/'C2REAL.csv').read_bytes())
    print('Fetched task numerical evidence to tools/c2real/evidence')


def verify():
    execute(f'''import os,subprocess,pathlib
r=pathlib.Path({REMOTE!r});assert r.is_dir()
env={{**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}}
p=subprocess.run(['/usr/bin/nice','-n','19','/root/miniconda3/bin/python','-B',str(r/'verify.py')],env=env,capture_output=True,text=True)
print(p.stdout);print(p.stderr);raise SystemExit(p.returncode)
''')


def coverage():
    execute(f'''import os,subprocess,pathlib
r=pathlib.Path({REMOTE!r});assert r.is_dir()
env={{**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}}
p=subprocess.run(['/usr/bin/nice','-n','19','/root/miniconda3/bin/python','-B',str(r/'coverage.py')],env=env,capture_output=True,text=True)
print(p.stdout);print(p.stderr);raise SystemExit(p.returncode)
''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stage', 'run', 'status', 'fetch', 'verify', 'coverage'])
    globals()[parser.parse_args().action]()
