"""Short explicit SSH operations for the owner-requested HOLD-credit continuation."""
import argparse
import json
from pathlib import Path
from tools.coopbandit import ops as ssh

LOCAL = Path(__file__).resolve().parent
ROOT = '/home/ubuntu/coopfree20261101_build5'


def gate():
    sources = {name: (LOCAL/name).read_text(encoding='utf-8') for name in
               ('coopfree_hold_gate.py', 'coopfree_batch.py')}
    return ssh.remote('cpuQ', f'''import pathlib,json,os,subprocess,sys
r=pathlib.Path({ROOT!r});assert r.is_dir()
assert not (r/'HOLD_GATE_DISPATCH.json').exists(),'already dispatched'
for name,text in json.loads({json.dumps(sources)!r}).items():
 p=r/'pytools/tools'/name;assert p.parent.is_dir()
 if p.exists(): assert p.read_text()==text
 else: p.write_text(text)
sys.path.insert(0,str(r/'pytools'))
from tools.coopfree.storage import write
from tools.coopfree_batch import identity
identity(pathlib.Path('/proc/339173/stat').read_text())
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
log=(r/'HOLD_GATE.log').open('x')
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopfree_hold_gate'],env=env,cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
receipt=dict(pid=p.pid,ticks=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],root=str(r),new_physics=0)
write(r/'HOLD_GATE_DISPATCH.json',receipt)
print(json.dumps(receipt))
''', timeout=20)


def status():
    return ssh.remote('cpuQ', f'''import pathlib,json,os,shutil
r=pathlib.Path({ROOT!r});read=lambda p:json.loads(p.read_text())
names=['HOLD_GATE_DISPATCH.json','HOLD_GATE_PARKED.json','BATCH_RESUMED.json','STOP.json','HOLD_STOP.json','HOLD_DISPATCH.json','HOLD_RETIRED_LEGACY.json','HOLD_COMPLETE.json','UPDATE_001.json']
out={{n:read(r/n) for n in names if (r/n).exists() and n!='UPDATE_001.json'}}
out['legacy_update_exists']=(r/'UPDATE_001.json').exists()
p=pathlib.Path('/proc/339173/stat')
out['supervisor_stat']=p.read_text().split()[2:3] if p.exists() else None
if (r/'HOLD_DISPATCH.json').exists():
 d=read(r/'HOLD_DISPATCH.json');p=pathlib.Path('/proc/'+str(d['pid'])+'/stat')
 out['hold_alive']=p.exists() and p.read_text().split()[21]==d['start_ticks'] and p.read_text().split()[2]!='Z'
if (r/'HOLD_PLAN_001.json').exists():
 plan=read(r/'HOLD_PLAN_001.json')
 out['hold_plan']=dict(tasks=len(plan['tasks']),baselines=len(set(plan['baseline_ids'])),exact_credit_windows=len(plan['exact_indices']),baseline_cost=plan['hold_baseline_cost'])
 out['hold_tasks_complete']=sum((r/'jobs'/t['id']/'RESULT.json').exists() and (r/'jobs'/t['id']/'CLOSE.json').exists() for t in plan['tasks'])
out['hold_update_exists']=(r/'HOLD_UPDATE_001.json').exists()
rows=[json.loads(s) for s in (r/'CHARGES.jsonl').read_text().splitlines()]
out['charged']=sum(x['sim_s'] for x in rows)
out['load']=os.getloadavg()
out['disk_free']=shutil.disk_usage(r).free
out['batch_complete']=(r/'BATCH_COMPLETE.json').exists()
out['sidecar_state']=pathlib.Path('/proc/436993/stat').read_text().split()[2] if pathlib.Path('/proc/436993/stat').exists() else None
out['unfinished']=[p.name for p in (r/'jobs').iterdir() if p.is_dir() and (p/'START.json').exists() and not (p/'CLOSE.json').exists()]
print(json.dumps(out))
''', timeout=20)


def relocate():
    return ssh.remote('cpuQ', f'''import pathlib,json,shutil,hashlib,sys,os
r=pathlib.Path({ROOT!r});assert r.is_dir()
dest=pathlib.Path('/dev/shm/coopfree20261101_observable');assert dest.parent.is_dir()
assert not (r/'TRACE_RELOCATION.json').exists(),'one-shot relocation'
dest.mkdir(exist_ok=False)
assert shutil.disk_usage(dest).free>4*1024**3
sys.path.insert(0,str(r/'pytools'))
from tools.coopfree.storage import write
rows=[];size=0
for job in sorted((r/'jobs').iterdir()):
 p=job/'observable.jsonl.gz'
 if not p.is_file() or p.is_symlink() or not (job/'CLOSE.json').exists() or not (job/'RESULT.json').exists(): continue
 result=json.loads((job/'RESULT.json').read_text())
 digest=hashlib.sha256(p.read_bytes()).hexdigest()
 assert digest==result['trace_sha256']
 target=dest/(job.name+'.jsonl.gz');n=p.stat().st_size
 shutil.move(str(p),str(target));p.symlink_to(target)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
 rows.append(dict(id=job.name,bytes=n,sha256=digest,target=str(target)));size+=n
 if size>=600*1024**2: break
write(r/'TRACE_RELOCATION.json',dict(bytes=size,files=rows,reason='unchanged2GiB admission blocked at1.974GiB; preserve content and logical paths; no trace deletion',disk_free=shutil.disk_usage(r).free))
print(json.dumps(dict(bytes=size,files=len(rows),disk_free=shutil.disk_usage(r).free)))
''', timeout=45)


def stage_cpu():
    sources = {name: (LOCAL/name).read_text(encoding='utf-8') for name in
               ('coopfree_hold.py', 'test_coopfree_hold.py')}
    design = (LOCAL.parent/'docs/paper_results/COOPFREE_DESIGN.md').read_text(encoding='utf-8')
    return ssh.remote('cpuQ', f'''import pathlib,json,hashlib,sys,os,unittest,time
from unittest.mock import patch
r=pathlib.Path({ROOT!r});assert r.is_dir()
for name,text in json.loads({json.dumps(sources)!r}).items():
 p=r/'pytools/tools'/name;assert p.parent.is_dir()
 if p.exists(): assert p.read_text()==text,'immutable HOLD source'
 else: p.write_text(text)
p=r/'HOLD_DESIGN.md';text={design!a}
if p.exists(): assert p.read_text()==text
else: p.write_text(text)
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(r/'pytools'))
from tools.leverprobe.backend import setup
setup('/home/ubuntu/e3final','/home/ubuntu/mappospeed/production2/policy_latest.pt')
os.environ['COOPFREE_CPU_CHECKPOINT']='/home/ubuntu/mappospeed/production2/policy_latest.pt'
from tools.coopfree.storage import write
begin=time.monotonic()
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(n) for n in ('tools.coopfree.test_cpu','tools.test_coopfree_hold')])
with patch('subprocess.Popen',side_effect=AssertionError('SUMO forbidden during CPU gate')):
 result=unittest.TextTestRunner(verbosity=2).run(suite)
receipt=dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,wall_s=time.monotonic()-begin,
 sources={{name:hashlib.sha256((r/'pytools/tools'/name).read_bytes()).hexdigest() for name in {list(sources)!r}}},design_sha256=hashlib.sha256((r/'HOLD_DESIGN.md').read_bytes()).hexdigest())
if receipt['pass_'] and not receipt['skips']: write(r/'HOLD_CPU_APPROVED.json',receipt)
print(json.dumps(receipt));assert receipt['pass_'] and not receipt['skips']
''', timeout=25)


def launch():
    import hashlib
    from tools.coopfree.protocol import checked
    registry = checked(Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')).read_text(encoding='utf-8')
    line = next(s for s in registry.splitlines() if s.startswith('- S-COOPFREE OWNER01:48 HOLD CONTINUATION REGISTERED'))
    expected = {name: hashlib.sha256((LOCAL/name).read_text(encoding='utf-8').encode()).hexdigest()
                for name in ('coopfree_hold.py', 'test_coopfree_hold.py')}
    design_sha = hashlib.sha256((LOCAL.parent/'docs/paper_results/COOPFREE_DESIGN.md').read_text(encoding='utf-8').encode()).hexdigest()
    return ssh.remote('cpuQ', f'''import pathlib,json,hashlib,sys,subprocess,os
r=pathlib.Path({ROOT!r});assert r.is_dir()
assert not (r/'HOLD_DISPATCH.json').exists(),'already dispatched; inspect, never duplicate'
assert not (r/'STOP.json').exists() and not (r/'HOLD_STOP.json').exists()
sys.path.insert(0,str(r/'pytools'))
from tools.coopfree.storage import read,write
from tools.coopfree.runner import verify
verify(r)
cert=read(r/'HOLD_CPU_APPROVED.json');assert cert['pass_'] and cert['tests']==22 and cert['sim_s']==0 and not cert['skips']
assert cert['sources']==json.loads({json.dumps(expected)!r})
assert cert['design_sha256']=={design_sha!r}
assert (r/'HOLD_GATE_PARKED.json').exists()
for name,expected in cert['sources'].items(): assert hashlib.sha256((r/'pytools/tools'/name).read_bytes()).hexdigest()==expected
write(r/'HOLD_AUTHORITY.json',dict(registration_line={line!a},design_sha256=cert['design_sha256'],original_cap=2000000))
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
log=(r/'HOLD_SUPERVISOR.log').open('x')
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopfree_hold'],cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=p.pid,start_ticks=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],root=str(r),host='cpuQ',workers=4,cap=2000000,design_sha256=cert['design_sha256'])
write(r/'HOLD_DISPATCH.json',receipt);print(json.dumps(receipt))
''', timeout=20)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('gate','status','relocate','stage-cpu','launch'));a=p.parse_args()
    print({'gate':gate,'status':status,'relocate':relocate,'stage-cpu':stage_cpu,'launch':launch}[a.mode]())
