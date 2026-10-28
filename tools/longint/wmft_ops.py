"""Explicit S-C3WMFT file transport and short remote status calls."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from tools.longint.operate import ROOT, RUNTIME, RELEASE, remote

OUT = RELEASE + '/arm/loop/longint/c3wmft'
SCRIPT = RUNTIME + '/tools/longint/wmft_c3.py'
REGISTRATION = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')


def stage():
    data = (ROOT / 'tools/longint/wmft_c3.py').read_bytes().replace(b'\r\n', b'\n')
    # Verify the production scientific code/recipe; never replace it for this task.
    names = ['scripts/c4/loop14.py', 'scripts/c4/loop14_long.py',
             'tools/longint/gate_shard.py',
             'configs/c4_wm_loop14_r16_ft.toml']
    expected = {n: hashlib.sha256((ROOT / n).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                for n in names}
    remote(f'''import pathlib,hashlib,sys,json
r=pathlib.Path({RUNTIME!r})
for name,digest in {expected!r}.items():
 p=r/name; assert hashlib.sha256(p.read_bytes()).hexdigest()==digest, name
gate=hashlib.sha256((r/'tools/coopw/horizon_gate.py').read_bytes()).hexdigest()
assert gate.startswith('e9aac75f'), gate
p=pathlib.Path({SCRIPT!r}); assert p.parent.is_dir() and not p.exists()
data=sys.stdin.buffer.read({len(data)}); assert hashlib.sha256(data).hexdigest()=={hashlib.sha256(data).hexdigest()!r}
p.write_bytes(data)
print(json.dumps({{'staged':str(p),'sha256':hashlib.sha256(data).hexdigest(),'scientific_sources_verified':{expected!r}}}))
''', data)


def execute(mode):
    digest = hashlib.sha256(REGISTRATION.read_bytes()).hexdigest()
    remote(f'''import subprocess
args=['/root/miniconda3/bin/python','-B',{SCRIPT!r},{mode!r}]
if {mode!r}=='prepare': args.append({digest!r})
subprocess.run(args,check=True)
''', timeout=50)


def status():
    remote(f'''import pathlib,json,subprocess
out=pathlib.Path({OUT!r}); r=pathlib.Path({RELEASE!r})
for name in ('PID.json','STATUS.json','RESULTS.json','ERROR.json','CHILDREN.jsonl'):
 p=out/name
 if p.is_file(): print(name,p.read_text())
for name in ('driver.log','horizon.log'):
 p=out/name
 if p.is_file(): print(name,p.read_text()[-3500:])
s=json.loads((r/'arm/loop/LOOP14_STATE.json').read_text())
ph=s['cycles'][-1]['phases']['wm_ft']; print('wm_phase',json.dumps(ph))
attempt=ph['attempt']; name='c03'+('' if attempt<=1 else '.a'+str(attempt))
d=r/'arm/loop/wm'/name
p=d/'train_log.jsonl'
if p.is_file():
 lines=p.read_text().splitlines(); print('training_rows',len(lines)); print('last_training',lines[-1])
p=d.with_name(d.name+'.log')
if p.is_file(): print('training_log_tail',p.read_text()[-2500:])
print('gpu',subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True))
''', timeout=40)


def poll(wait):
    time.sleep(wait)
    remote(f'''import pathlib,json,datetime
o=pathlib.Path({OUT!r}); r=pathlib.Path({RELEASE!r})
s=json.loads((o/'STATUS.json').read_text()); state=json.loads((r/'arm/loop/LOOP14_STATE.json').read_text())
p=state['cycles'][-1]['phases']; a=p['wm_ft']['attempt']; n='c03'+('' if a<=1 else '.a'+str(a))
f=r/'arm/loop/wm'/n/'train_log.jsonl'
rows=[json.loads(line) for line in f.read_text().splitlines()] if f.is_file() else []
doc={{'now':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':s['status'],
 'phases':{{k:v['status'] for k,v in p.items()}},'attempt':a,'rows':len(rows),
 'last':{{k:v for k,v in (rows[-1] if rows else {{}}).items() if k in ('step','loss','grad_norm','elapsed_s','horizon_s')}}}}
for name in ('PID.json','c03_FIRST_FINITE.json','c03_TRAIN_VERIFIED.json','RESULTS.json','ERROR.json'):
 f=o/name
 if f.is_file():
  d=json.loads(f.read_text())
  if name=='c03_FIRST_FINITE.json': d.pop('first_rows',None)
  doc[name]=d
f=o/'PID.json'; d=json.loads(f.read_text()); proc=pathlib.Path('/proc')/str(d['pid'])
doc['coordinator_live']=proc.exists() and (proc/'stat').read_text().split()[21]==d['start_ticks']
g=r/'arm/loop/checks/c03/gate_shards/rows'
if g.is_dir(): doc['gate_groups']=len(list(g.glob('*.json')))
print(json.dumps(doc))
''', timeout=30)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['stage', 'prepare', 'launch', 'status', 'poll'])
    ap.add_argument('--wait', type=int, default=0)
    args = ap.parse_args()
    action = args.action
    if action == 'stage':
        stage()
    elif action == 'status':
        status()
    elif action == 'poll':
        poll(args.wait)
    else:
        execute(action)
