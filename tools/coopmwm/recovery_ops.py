"""Explicit incident recovery: register first; qualify; dispatch once; observe."""
import argparse
import json
from . import ops
from .storage import read,write
from .protocol import REGISTRY,sha,checked
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_recoverycpu1'
OVERLAYS=('incident_recovery','warm_health','warm_runner','quota_admission','plan_client',
    'exact_runtime','exact_grade','warm_protocol','capacity_broker','warm_broker','test_incident_recovery_cpu')


def register_plan():
    from .incident_recovery import recovery_budget,episode_plan
    from .warm_protocol import batch_worlds,worlds
    old=read(ops.EVIDENCE/'MAIN0508_REPAIR_REGISTRATION.json')
    text=checked(REGISTRY).read_text(encoding='utf-8')
    assert '2026-11-02 09:58 MAIN:' in text and '2026-11-02 10:11 MAIN:' in text
    target=ops.EVIDENCE/'MAIN1011_RECOVERY_PLAN.json';assert not target.exists(),'adopt existing plan'
    state=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time,hashlib
r=pathlib.Path({ROOT!r});jobs={{}};checkpoints={{}}
for arm,u in [('B',3),('R',2)]:
 for p in (r/'jobs').glob(f'warm_u{{u:03}}_{{arm}}_*'):
  jobs[p.name]=dict(closed=(p/'RESULT.json').exists(),claimed=(p/'CLAIM.json').exists())
 checkpoints[arm]=hashlib.sha256((r/f'{{arm}}_checkpoint_{{u-1:03}}.pt').read_bytes()).hexdigest()
assert not list(r.glob('PAUSE_*.json')),'health pause requires diagnosis'
stop=r/'WARM_GUIDED_STOP_V2.json';v=json.loads(stop.read_text())
assert v['error']=="ValueError('GPU failed; no inference retry')"
print(json.dumps(dict(unix=time.time(),jobs=jobs,checkpoints=checkpoints,stop=v,stop_sha256=hashlib.sha256(stop.read_bytes()).hexdigest())))''',timeout=40))
    plans={a:episode_plan(batch_worlds(u),worlds('train'),state['jobs'],a,u) for a,u in [('B',3),('R',2)]}
    assert sum(v['additional_sim_s'] for v in plans.values())==25200
    b=recovery_budget(old['budget'])
    value=dict(budget=b,plans=plans,state=state,resume=dict(B=2,R=1),checkpoints=state['checkpoints'],
        authority='MAIN09:58 reserve21000; MAIN10:11 B24/R24 primary A24 secondary A48',
        invariant='same warm memory/fastWM/H300/K8Hfinal/MPPI/Thompson/exact grades/trust/health; completed episodes and original Adam/RNG/library adopted; no original claimed job or interrupted inference replay',
        recovery='same-cell ascending unused REGISTERED TRAIN worlds replace21interrupted jobs;10unclaimed R precharges adopted; original25,200 failed charges retained; retired-job quota/ticket reservations excluded only from the replacement wave, all originals preserved',
        final='A24 new96episode+384branch jobs460800; keep673precomputed MAPPO/native/A48/reuse receipts opaque; final primary B24-A24-MAPPO/R24/native; A48 secondary; sealed analysis only after B24/R24',
        mirror='OWNER activation approved only after genuine null-fixture trajectory parity; currently OFF: existing probes fail absolute parity, no activation claimed',
        new_physics_dispatched=0)
    write(target,value);h=sha(target)
    line='\nS-COOPMWM MAIN0958/1011 RECOVERY AND FIXED-ENDPOINT AMENDMENT REGISTERED planSHA'+h+': '+json.dumps({k:v for k,v in value.items() if k!='state'},sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'MAIN1011_RECOVERY_REGISTRATION.json',dict(plan_sha256=h,line=line))
    print(json.dumps(dict(registered=True,plan_sha256=h,budget=b,replacements={a:len(v['replaced']) for a,v in plans.items()})))


def cpu():
    prior=read(ops.EVIDENCE/'CPU_coopmwm20261102_incidentcpu2.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['root']!r});files={{}}
for n,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 files[n.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=40))
    for n in OVERLAYS:files['tools/coopmwm/'+n+'.py']=checked(ops.LOCAL/'tools/coopmwm'/(n+'.py')).read_text(encoding='utf-8')
    staged=json.loads(ops.stage('cpuQ',SOURCE,files))
    out=ops.ssh.remote('cpuQ',f'''import sys,os,json,unittest,time
from unittest.mock import patch
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',COOPMWM_COST_SOURCE={SOURCE!r}+'/FROZEN_COSTS.json',COOPMWM_MOTION_SOURCE={SOURCE!r}+'/FROZEN_MOTION_SOURCE.py')
import torch;torch.set_num_threads(1)
names=('test_incident_recovery_cpu','test_incident_cpu','test_warm_health_cpu','test_quota_admission_cpu','test_tail_runtime_cpu','test_parallel_groups_cpu','test_capacity_broker_cpu','test_warm_broker_cpu','test_exact_grade_cpu','test_awr_cpu','test_acceleration_cpu')
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in names);start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('no physics/inference during CPU qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,root={SOURCE!r},wall_s=time.monotonic()-start)))''',timeout=90)
    print(out);v=json.loads(out.strip().splitlines()[-1]);v['source_hashes']=staged['hashes']
    write(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'),v)
    assert v['pass_'] and v['skips']==v['sim_s']==0


def bind():
    plan=read(ops.EVIDENCE/'MAIN1011_RECOVERY_PLAN.json')
    reg=read(ops.EVIDENCE/'MAIN1011_RECOVERY_REGISTRATION.json')
    cpu=read(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    value=dict(plan,cpu=cpu,source_root=SOURCE,amendment_sha256=reg['plan_sha256'])
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib,time
r=pathlib.Path({ROOT!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.incident_recovery import retired_key
v=json.loads({json.dumps(value)!r});assert not (r/'INCIDENT_RECOVERY_BINDING.json').exists(),'adopt existing binding'
for n,h in v['cpu']['source_hashes'].items():assert hashlib.sha256((pathlib.Path(v['source_root'])/n).read_bytes()).hexdigest()==h
for n in ('TAIL_B_DISPATCH.json','QUOTA_R_DISPATCH.json','TAIL_DRAIN_BROKER_DISPATCH.json','EXACT_GRADER_DISPATCH_V2.json'):
 d=json.loads((r/n).read_text());assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 p=pathlib.Path('/proc',str(d['pid']),'stat');s=p.read_text().split() if p.exists() else None
 assert not s or s[21]==d['start_ticks'] and s[2]=='Z','existing owner still live'
assert hashlib.sha256((r/'WARM_GUIDED_STOP_V2.json').read_bytes()).hexdigest()==v['state']['stop_sha256']
failed=[x['failed_id'] for a in v['plans'].values() for x in a['replaced']]
retired=[];rtickets=[]
for req,resp in [('requests','responses'),('centre_requests','centre_responses')]:
 for p in (r/req).glob('*.json'):
  record=json.loads(p.read_text())
  if not retired_key(record,failed):continue
  retired.append(dict(key=p.stem,kind=req,responded=(r/resp/p.name).exists()))
  if req=='centre_requests':rtickets.append(p.stem)
  # Existing healthy assigned searches finish once; do not deliver or resubmit
  # retired requests through a replacement worker. Their original claims remain.
with (r/'AUTHORITY.md').open('a') as f:f.write({reg['line']!r})
immutable(r/'INCIDENT_RECOVERY_BINDING.json',v)
immutable(r/'INCIDENT_RETIRED_JOBS.json',dict(jobs=failed,R_ticket_keys=rtickets,query_keys=[x['key'] for x in retired],disabled_slots=['gpuA@0'],amendment_sha256=v['amendment_sha256']))
immutable(r/'INCIDENT_GUIDED_RELEASE.json',dict(original_stop_sha256=v['state']['stop_sha256'],amendment_sha256=v['amendment_sha256']))
old=json.loads((r/'WARM_REMOTE_BINDING.json').read_text())
immutable(r/'INCIDENT_BROKER_CONFIG.json',dict(amendment_sha256=v['amendment_sha256'],source_root=v['source_root'],deadline_unix=old['deadline_unix'],receipt_tag='_I1',pipeline_centre_fairness=True))
print(json.dumps(dict(bound=True,unix=time.time(),retired_queries=len(retired),retired_R_tickets=len(rtickets),budget=v['budget'])))''',timeout=45)
    v=json.loads(out);write(ops.EVIDENCE/'INCIDENT_RECOVERY_BOUND.json',v);print(out)


def dispatch(component):
    assert component in ('broker','grader','B','R')
    dest='INCIDENT_'+component.upper()+'_DISPATCH.json'
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,subprocess,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.incident_recovery import binding
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running
b=binding(r);assert_running(r,'B');assert not (r/{dest!r}).exists(),'adopt existing owner'
if {component!r} in ('B','R'):
 from tools.coopmwm.exact_runtime import assert_live
 assert_live(r)
 d=json.loads((r/'INCIDENT_BROKER_DISPATCH.json').read_text());st=pathlib.Path('/proc',str(d['pid']),'stat').read_text().split();assert st[21]==d['start_ticks'] and st[2]!='Z'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
with (r/{('INCIDENT_'+component.upper()+'.log')!r}).open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.incident_recovery',str(r),'--component',{component!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source=str(s),root=str(r),unix=time.time(),component={component!r})
immutable(r/{dest!r},v);print(json.dumps(v))''',timeout=45)
    v=json.loads(out);write(ops.EVIDENCE/dest,v);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register_plan','cpu','bind','dispatch'))
    p.add_argument('--component',choices=('broker','grader','B','R'));a=p.parse_args()
    dispatch(a.component) if a.mode=='dispatch' else globals()[a.mode]()
