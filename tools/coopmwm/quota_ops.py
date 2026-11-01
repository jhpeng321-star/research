"""MAIN05:48 outcome-blind admission registration and boundary handover."""
import argparse
import json
from . import ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_quotacpu4'


def preregister():
    path=ops.EVIDENCE/'MAIN0548_ADMISSION_PLAN.json'
    if not path.exists():
        write(path,dict(authority='MAIN2026-11-02 05:48, full290line brief',
            admission='per update Nexec10: if executed+unresolved potentially executable queries<10, eligible decisions query at p=.95 and wait even without a free slot; otherwise p=.95 only with a free slot, else0; serialize reservations; record the predicate and realised probability',
            broker='preserve quota-blocking requests; adopt all existing assigned inference without recomputation; ordinary overflow declines remain explicit',
            first_update='retain all24 current B episodes; apply u1 even if fewer than10 starts with explicit engineering deviation, no extra episodes; later shortage is governed by two-consecutive-update health pause',
            pipeline='independent R(u) consumes final B(u) quota while B(u+1) collects; exact count/regime/group-size/duration matching; no library for R',
            handover='register and CPU qualify first; freeze only receipt-owned B/R supervisor parents before optimizer boundary if necessary; existing child SUMO and GPU work continues; wait all24 B RESULTs, then terminate old parents by verified identity and adopt complete results with checkpoint0 Adam/RNG; no physics replay',
            invariants='same warm fast-path MPPI, K4H300/K8Hfinal, populations, beta/exact-only grading, guards, endpoints, original GPU deadline; no learned proposer activation',
            budget=dict(cap=7500000,allocated=7495040,free=4960,additional_sim_s=0),
            no_outcome_selection=True))
    h=sha(path)
    line='\nS-COOPMWM MAIN0548 QUOTA-AWARE ADMISSION REGISTERED amendmentSHA'+h+': '+json.dumps(read(path),sort_keys=True)+'\n'
    registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line)
    reply=ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});p=r/'AUTHORITY.md'
if {h!r} not in p.read_text():
 with p.open('a') as f:f.write({line!r})
print(json.dumps(dict(registered=True,amendment_sha256={h!r},new_sumo_sim_s=0)))''',timeout=40)
    value=json.loads(reply);write(ops.EVIDENCE/'MAIN0548_ADMISSION_PREREGISTRATION.json',value);print(reply)


def freeze():
    reg=read(ops.EVIDENCE/'MAIN0548_ADMISSION_PREREGISTRATION.json')
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,os,signal,time,sys
r=pathlib.Path({ROOT!r});assert {reg['amendment_sha256']!r} in (r/'AUTHORITY.md').read_text()
sys.path.insert(0,'/home/ubuntu/coopmwm20261102_capacitycpu3/pytools')
from tools.coopmwm.warm_spool import immutable
assert not (r/'B_UPDATE_001.json').exists(),'boundary passed; inspect before controlling owners'
assert not list((r/'jobs').glob('warm_u001_R_*/CLAIM.json')),'R already collecting'
owners={{}}
for arm in ('B','R'):
 d=json.loads((r/f'WARM_{{arm}}_DISPATCH_V6.json').read_text());p=pathlib.Path('/proc',str(d['pid']))
 assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 st=(p/'stat').read_text().split();assert st[21]==d['start_ticks'] and st[2] in ('S','R','T')
 cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 assert 'tools.coopmwm.capacity_recovery' in cmd and '--component '+arm in cmd and str(r) in cmd
 os.kill(d['pid'],signal.SIGSTOP);owners[arm]=dict(d,command=cmd)
immutable(r/'MAIN0548_FROZEN_PARENTS.json',dict(owners=owners,unix=time.time(),amendment_sha256={reg['amendment_sha256']!r},children_signalled=False))
print(json.dumps(dict(frozen_parents=owners,children_signalled=False,new_sumo_sim_s=0)))''',timeout=40)
    v=json.loads(out);write(ops.EVIDENCE/'MAIN0548_FROZEN_PARENTS.json',v);print(out)


def qualify():
    prior=read(ops.EVIDENCE/'CPU_coopmwm20261102_quotacpu3.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['root']!r});files={{}}
for name,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 files[name.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=40))
    names=('quota_ops','quota_runtime','quota_transition','quota_admission','test_quota_admission_cpu','plan_client','capacity_broker','warm_runner')
    for name in names:files['tools/coopmwm/'+name+'.py']=(ops.LOCAL/'tools/coopmwm'/(name+'.py')).read_text(encoding='utf-8')
    staged=json.loads(ops.stage('cpuQ',SOURCE,files))
    out=ops.ssh.remote('cpuQ',f'''import sys,unittest,os,json,time
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',COOPMWM_COST_SOURCE={SOURCE!r}+'/FROZEN_COSTS.json',COOPMWM_MOTION_SOURCE={SOURCE!r}+'/FROZEN_MOTION_SOURCE.py')
import torch;torch.set_num_threads(1)
from unittest.mock import patch
start=time.monotonic()
names=['tools.coopnative.test_cpu']+['tools.coopmwm.'+n for n in ('test_cpu','test_volume_cpu','test_acceleration_cpu','test_warm_cpu','test_mppi_search_cpu','test_branch_eval_cpu','test_mppi_runtime_cpu','test_warm_protocol_cpu','test_awr_cpu','test_warm_health_cpu','test_warm_transport_cpu','test_warm_spool_cpu','test_warm_broker_cpu','test_committed_cpu','test_overlap_cpu','test_exact_grade_cpu','test_plan_proposer_cpu','test_capacity_broker_cpu','test_capacity_recovery_cpu','test_quota_admission_cpu')]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
with patch('subprocess.Popen',side_effect=AssertionError('no SUMO in qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=1).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,root={SOURCE!r},wall_s=time.monotonic()-start)))''',timeout=90)
    print(out);v=json.loads(out.strip().splitlines()[-1]);v['source_hashes']=staged['hashes']
    write(ops.EVIDENCE/'CPU_coopmwm20261102_quotacpu4.json',v);assert v['pass_'] and not v['skips']


def register():
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_quotacpu3.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    reg=read(ops.EVIDENCE/'MAIN0548_ADMISSION_PREREGISTRATION.json')
    v=dict(cpu=cpu,source_root=SOURCE,plan_amendment_sha256=reg['amendment_sha256'],
        new_sumo_sim_s=0,gpu_source_changes=False,library_and_search_unchanged=True,
        migration='original B/R parents frozen by identity; children finish once; all24 closed B results adopted, checkpoint0 Adam/RNG preserved; broker claims/forecasts retained; grader stays original',
        boundary_drain='until all24 old B episodes close, adopt every already assigned search but decline NEW unassigned searches with logged zero-compute admission; finish the four pending queries once instead of repeatedly extending the old-client wave; release drain only at completed-result handover; no extra SUMO')
    path=ops.EVIDENCE/'MAIN0548_RUNTIME_AMENDMENT.json';write(path,v);h=sha(path)
    line='\nS-COOPMWM MAIN0548 QUOTA RUNTIME QUALIFIED REGISTERED amendmentSHA'+h+': '+json.dumps({k:x for k,x in v.items() if k!='cpu'},sort_keys=True)+'\n'
    registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line)
    reply=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
for name,h in json.loads({json.dumps(cpu['source_hashes'])!r}).items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==h
if {h!r} not in (r/'AUTHORITY.md').read_text():
 with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
immutable(r/'MAIN0548_RUNTIME_BINDING.json',json.loads({json.dumps(dict(v,amendment_sha256=h))!r}))
old=json.loads((r/'WARM_REMOTE_BINDING.json').read_text())
immutable(r/'QUOTA_BROKER_CONFIG.json',dict(amendment_sha256={h!r},source_root=str(s),deadline_unix=old['deadline_unix'],receipt_tag='_Q1',drain_until_handover=True))
print(json.dumps(dict(registered=True,amendment_sha256={h!r},new_sumo_sim_s=0)))''',timeout=40)
    write(ops.EVIDENCE/'MAIN0548_RUNTIME_REGISTRATION.json',dict(v,amendment_sha256=h,remote=json.loads(reply)));print(reply)


def deploy_broker():
    assert read(ops.EVIDENCE/'MAIN0548_RUNTIME_REGISTRATION.json')['amendment_sha256']
    out=ops.ssh.remote('cpuQ',f'''import sys,json
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from tools.coopmwm.quota_transition import replace_broker
print(json.dumps(replace_broker({ROOT!r})))''',timeout=45)
    v=json.loads(out);write(ops.EVIDENCE/'QUOTA_broker_DISPATCH.json',v);print(out)


def repair_register():
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_quotacpu4.json');assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
    old=read(ops.EVIDENCE/'MAIN0548_RUNTIME_REGISTRATION.json')
    v=dict(cpu=cpu,source_root=SOURCE,plan_amendment_sha256=old['plan_amendment_sha256'],
        diagnosis='identity poll observed empty cmdline after verified TERM/CONT of old supervisor; Linux clears argv before proc disappears; all24 B results complete, no optimizer/new B/R dispatch occurred',
        fix='same boot and start ticks remain mandatory; empty cmdline is observed as exiting without further signals; disappearing proc files count as exited; no relaxed nonempty-command identity checks',
        new_sumo_sim_s=0,budget_unchanged=True,previous_runtime=old['amendment_sha256'])
    path=ops.EVIDENCE/'MAIN0548_PARENT_EXIT_AMENDMENT.json';write(path,v);h=sha(path)
    line='\nS-COOPMWM MAIN0548 PARENT-EXIT ENGINEERING REPAIR REGISTERED amendmentSHA'+h+': '+json.dumps({k:x for k,x in v.items() if k!='cpu'},sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path[:0]=[str(s/'pytools'),'/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.quota_transition import completed
failure=json.loads((r/'MAIN0548_TRANSITION_FAILED.json').read_text());assert failure['error']=="AssertionError('command identity changed')"
assert completed(r) and not (r/'B_UPDATE_001.json').exists()
for a in ('B','R'):assert not (r/f'QUOTA_{{a}}_DISPATCH.json').exists()
for name,h in json.loads({json.dumps(cpu['source_hashes'])!r}).items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==h
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
immutable(r/'MAIN0548_RUNTIME_BINDING_V2.json',json.loads({json.dumps(dict(v,amendment_sha256=h))!r}))
from tools.coopmwm.quota_runtime import install
from tools.coopmwm.accel_learner import load
wr,_=install(r)
for a in ('B','R'):
 b=wr.verify(r,a);_,_,_,payload=load(r/f'{{a}}_checkpoint_000.pt');assert payload['update']==0
print(json.dumps(dict(registered=True,amendment_sha256={h!r},all24_adopted_preflight=True,u0_Adam_RNG=True,new_sumo_sim_s=0)))''',timeout=45)
    write(ops.EVIDENCE/'MAIN0548_PARENT_EXIT_REGISTRATION.json',dict(v,amendment_sha256=h,remote=json.loads(out)));print(out)


def handover():
    assert read(ops.EVIDENCE/'MAIN0548_PARENT_EXIT_REGISTRATION.json')['remote']['all24_adopted_preflight']
    out=ops.ssh.remote('cpuQ',f'''import sys,json
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from tools.coopmwm.quota_transition import handover
print(json.dumps(handover({ROOT!r})))''',timeout=45)
    owners=json.loads(out)
    for arm,owner in owners.items():write(ops.EVIDENCE/f'QUOTA_{arm}_DISPATCH.json',owner)
    print(out)


def dispatch_watch():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,sys,json,os,subprocess,time
sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.quota_runtime import binding
from tools.coopmwm.warm_spool import immutable
r=pathlib.Path({ROOT!r});binding(r)
assert not (r/'MAIN0548_TRANSITION_DISPATCH.json').exists(),'adopt existing watcher'
d=json.loads((r/'QUOTA_broker_DISPATCH.json').read_text());st=pathlib.Path('/proc',str(d['pid']),'stat').read_text().split();assert st[21]==d['start_ticks'] and st[2] in ('S','R')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH={SOURCE!r}+'/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
with (r/'MAIN0548_TRANSITION.log').open('x') as log:
 p=subprocess.Popen([{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.quota_transition','watch',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()
v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),source={SOURCE!r},unix=time.time())
immutable(r/'MAIN0548_TRANSITION_DISPATCH.json',v);print(json.dumps(v))''',timeout=40)
    v=json.loads(out);write(ops.EVIDENCE/'MAIN0548_TRANSITION_DISPATCH.json',v);print(out)


def status():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,time,os,sys
r=pathlib.Path({ROOT!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.quota_transition import completed
names=['QUOTA_broker_DISPATCH.json','QUOTA_B_DISPATCH.json','QUOTA_R_DISPATCH.json','MAIN0548_TRANSITION_DISPATCH.json','EXACT_GRADER_DISPATCH_V2.json']
owners={{}}
for name in names:
 if not (r/name).exists():continue
 d=json.loads((r/name).read_text());s=pathlib.Path('/proc',str(d['pid']),'stat');st=s.read_text().split() if s.exists() else None
 assert not st or st[21]==d['start_ticks'];assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 owners[name]=dict(owner=d,state=st[2] if st else 'exited')
receipts={{}}
for name in ('MAIN0548_BROKER_HANDOVER.json','MAIN0548_HANDOVER_COMPLETE.json','MAIN0548_TRAINING_CONTINUED.json','MAIN0548_TRANSITION_FAILED.json','WARM_GUIDED_STOP_V2.json','WARM_B_FAILED_Q1.json','WARM_R_FAILED_Q1.json'):
 if (r/name).exists():receipts[name]=json.loads((r/name).read_text())
for p in r.glob('PAUSE_*.json'):receipts[p.name]=json.loads(p.read_text())
pending=[]
for p in (r/'requests').glob('*.json'):
 if (r/'responses'/p.name).exists():continue
 v=json.loads(p.read_text());a=r/'broker_assignments'/p.name
 pending.append(dict(key=v['state_id'],world=v['world_id'],time_s=v['time_s'],group_size=len(v['group']),age_s=time.time()-p.stat().st_mtime,assignment=json.loads(a.read_text()) if a.exists() else None))
print(json.dumps(dict(unix=time.time(),owners=owners,receipts=receipts,pending=pending,all24closed=completed(r),load=os.getloadavg(),updates={{a:max([0]+[int(p.stem.rsplit('_',1)[1]) for p in r.glob(a+'_UPDATE_*.json')]) for a in ('A','B','R')}},logs={{name:(r/name).read_text()[-5000:] for name in ('MAIN0548_TRANSITION.log','QUOTA_B.log','QUOTA_R.log','QUOTA_broker.log') if (r/name).exists()}})))''',timeout=45)
    v=json.loads(out);write(ops.EVIDENCE/f'MAIN0548_STATUS_{int(v["unix"])}.json',v)
    for name,receipt in v['receipts'].items():
        if name.startswith('MAIN0548_') and not (ops.EVIDENCE/name).exists():write(ops.EVIDENCE/name,receipt)
    for name,owner in v['owners'].items():
        if name.startswith('QUOTA_') and not (ops.EVIDENCE/name).exists():write(ops.EVIDENCE/name,owner['owner'])
    print(out);return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('preregister','freeze','qualify','register','deploy_broker','repair_register','handover','dispatch_watch','status'));a=p.parse_args();globals()[a.mode]()
