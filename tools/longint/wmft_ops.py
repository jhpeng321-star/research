"""Explicit S-C3WMFT file transport and short remote status calls."""
import argparse
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time

from tools.longint.operate import ROOT, RUNTIME, RELEASE, remote

OUT = RELEASE + '/arm/loop/longint/c3wmft'
SCRIPT = RUNTIME + '/tools/longint/wmft_c3.py'
REGISTRATION = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')


def stage(repair=False):
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
p=pathlib.Path({SCRIPT!r}); assert p.parent.is_dir()
if {repair!r}:
 old=p.read_bytes(); digest=hashlib.sha256(old).hexdigest()
 assert digest=='60e5221a8f16ed63ac412ae9232227a62d8aca60577201cda33d7c8d9cc249b0'
 backup=p.with_name(p.name+'.'+digest+'.before'); assert not backup.exists(); backup.write_bytes(old)
else: assert not p.exists()
data=sys.stdin.buffer.read({len(data)}); assert hashlib.sha256(data).hexdigest()=={hashlib.sha256(data).hexdigest()!r}
p.write_bytes(data)
print(json.dumps({{'staged':str(p),'sha256':hashlib.sha256(data).hexdigest(),'scientific_sources_verified':{expected!r}}}))
''', data)


def execute(mode):
    digest = hashlib.sha256(REGISTRATION.read_bytes()).hexdigest()
    retry = mode in ('repair', 'launch-retry')
    command = 'launch' if mode == 'launch-retry' else mode
    remote(f'''import subprocess,os
args=['/root/miniconda3/bin/python','-B',{SCRIPT!r},{command!r}]
if {command!r} in ('prepare','repair'): args.append({digest!r})
env={{**os.environ}}
if {retry!r}: env['C3WMFT_RETRY']='1'
subprocess.run(args,check=True,env=env)
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
 lines=p.read_text().splitlines(); print('training_rows',len(lines)); print('last_training',lines[-1] if lines else None)
p=d.with_name(d.name+'.log')
if p.is_file(): print('training_log_tail',p.read_text()[-2500:])
print('gpu',subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True))
''', timeout=40)


def poll(wait):
    time.sleep(wait)
    remote(f'''import pathlib,json,datetime
o=pathlib.Path({OUT!r}); r=pathlib.Path({RELEASE!r})
if (o/'retry1/PID.json').exists(): o=o/'retry1'
s=json.loads((o/'STATUS.json').read_text()); state=json.loads((r/'arm/loop/LOOP14_STATE.json').read_text())
p=state['cycles'][-1]['phases']; a=p['wm_ft']['attempt']; n='c03'+('' if a<=1 else '.a'+str(a))
f=r/'arm/loop/wm'/n/'train_log.jsonl'
rows=[json.loads(line) for line in f.read_text().splitlines()] if f.is_file() else []
doc={{'now':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':s['status'],
 'attempt':a,'rows':len(rows),
 'last':{{k:v for k,v in (rows[-1] if rows else {{}}).items() if k in ('step','loss','grad_norm','elapsed_s','horizon_s')}}}}
for name in ('c03_TRAIN_VERIFIED.json','RESULTS.json','ERROR.json'):
 f=o/name
 if f.is_file():
  d=json.loads(f.read_text())
  doc[name]=d
f=o/'PID.json'; d=json.loads(f.read_text()); proc=pathlib.Path('/proc')/str(d['pid'])
doc['coordinator_live']=proc.exists() and (proc/'stat').read_text().split()[21]==d['start_ticks']
g=r/'arm/loop/checks/c03/gate_shards/rows'
if g.is_dir(): doc['gate_groups']=len(list(g.glob('*.json')))
print(json.dumps(doc))
''', timeout=30)


def collect():
    """Fetch bounded final numerical receipts and independently verify the stop."""
    code = f'''import pathlib,json,hashlib,base64,math,datetime
r=pathlib.Path({RELEASE!r}); o=pathlib.Path({OUT!r}); final=o/'retry1'
s=json.loads((r/'arm/loop/LOOP14_STATE.json').read_text()); p=s['cycles'][-1]['phases']
result=json.loads((final/'RESULTS.json').read_text())
assert json.loads((r/'AUTONOMY_STATUS.json').read_text())['status']=='WAIT_MAIN_C4'
assert s['cycles'][-1]['cycle']==3 and s['sim_s_charged']==264544
assert p['wm_ft']['status']==p['check']['status']=='done'
assert (r/'arm/loop/STOP_DRIVER').is_file()
ft=p['wm_ft']['receipt']; check=p['check']['receipt']; out=pathlib.Path(ft['out'])
rows=[json.loads(line) for line in (out/'train_log.jsonl').read_text().splitlines()]
assert len(rows)==ft['train_log_rows'] and 0<len(rows)<=40
assert [row['step'] for row in rows]==list(range(1,len(rows)+1))
assert all(row.get('finite') is not False and math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in rows)
wm_sha=hashlib.sha256(pathlib.Path(ft['candidate']).read_bytes()).hexdigest()
assert wm_sha==ft['candidate_sha256']==result['wm3_sha256']
assert ft['init_from_sha256']=='b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
for directory in (o,final):
 d=json.loads((directory/'PID.json').read_text()); proc=pathlib.Path('/proc')/str(d['pid'])
 assert not proc.exists() or (proc/'stat').read_text().split()[21]!=d['start_ticks']
 for line in (directory/'CHILDREN.jsonl').read_text().splitlines():
  d=json.loads(line); proc=pathlib.Path('/proc')/str(d['pid'])
  assert not proc.exists() or (proc/'stat').read_text().split()[21]!=d['start_ticks']
paths={{'initial_AUTHORITY.json':o/'AUTHORITY.json','initial_PID.json':o/'PID.json',
 'initial_ERROR.json':o/'ERROR.json','initial_FAILED_FT_ATTEMPT1.json':o/'FAILED_FT_ATTEMPT1.json',
 'retry_AUTHORITY.json':final/'AUTHORITY.json','retry_PID.json':final/'PID.json',
 'retry_STATUS.json':final/'STATUS.json','retry_TRAIN_VERIFIED.json':final/(out.name+'_TRAIN_VERIFIED.json'),
 'RESULTS.json':final/'RESULTS.json','ACCEPT.json':pathlib.Path(check['out'])/'ACCEPT.json'}}
gp=pathlib.Path(check['out'])/'HORIZON_GATE.json'
if gp.is_file():
 gate=json.loads(gp.read_text()); assert gate['wm_sha256']==wm_sha
 assert gate['threshold_source']['sha256']=='c479b2dec8716adeb1d518710f75f3352612f33f2fb7b58622f400ed462c8966'
 paths['HORIZON_GATE.json']=gp
verified={{'task':'S-C3WMFT','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'wm3_sha256':wm_sha,'training_rows':len(rows),'optimization_metrics_finite':True,
 'cuda_peak_mb':max(row.get('cuda_peak_mb',0) for row in rows),
 'acceptance':check,'trusted_horizon_s':result.get('trusted_horizon_s'),
 'charged':264544,'stop_driver_preserved':True,'cycle4_absent':True,'owned_processes_exited':True,
 'status':'WAIT_MAIN_C4','files':{{name:{{'remote_path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}} for name,path in paths.items()}}}}
vp=final/'FINAL_VERIFIED.json'; assert not vp.exists(); vp.write_text(json.dumps(verified,indent=2))
paths['FINAL_VERIFIED.json']=vp
print(json.dumps({{name:{{'data':base64.b64encode(path.read_bytes()).decode(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}} for name,path in paths.items()}}))
'''
    with contextlib.redirect_stdout(io.StringIO()):
        payload = json.loads(remote(code, timeout=40))
    directory = ROOT / 'tools/longint/receipts_c3wmft'
    assert directory.is_dir()
    for name, item in payload.items():
        assert '/' not in name and '\\' not in name
        data = base64.b64decode(item['data'])
        assert hashlib.sha256(data).hexdigest() == item['sha256']
        target = directory / name
        assert not target.exists()
        target.write_bytes(data)
    print(json.dumps({'files': len(payload), 'directory': str(directory),
                      'verified': json.loads((directory / 'FINAL_VERIFIED.json').read_bytes())}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['stage', 'prepare', 'launch', 'status', 'poll',
                                     'repair-stage', 'repair', 'launch-retry', 'collect'])
    ap.add_argument('--wait', type=int, default=0)
    args = ap.parse_args()
    action = args.action
    if action == 'stage':
        stage()
    elif action == 'repair-stage':
        stage(repair=True)
    elif action == 'status':
        status()
    elif action == 'poll':
        poll(args.wait)
    elif action == 'collect':
        collect()
    else:
        execute(action)
