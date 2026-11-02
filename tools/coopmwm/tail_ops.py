"""Explicit qualification/registration/one-use boundary deployment for MAIN08:18."""
import argparse
import json
from . import ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_tailcpu3'
OVERLAYS=('quota_admission','parallel_groups','plan_client','accel_runner','warm_runner',
    'tail_runtime','tail_transition','tail_ops','test_parallel_groups_cpu','test_tail_runtime_cpu')


def qualify():
    prior=read(ops.EVIDENCE/'CPU_coopmwm20261102_quotacpu4.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['root']!r});files={{}}
for name,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 files[name.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=40))
    for name in OVERLAYS:files['tools/coopmwm/'+name+'.py']=checked(ops.LOCAL/'tools/coopmwm'/(name+'.py')).read_text(encoding='utf-8')
    staged=json.loads(ops.stage('cpuQ',SOURCE,files))
    out=ops.ssh.remote('cpuQ',f'''import sys,os,json,unittest,time
from unittest.mock import patch
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',COOPMWM_COST_SOURCE={SOURCE!r}+'/FROZEN_COSTS.json',COOPMWM_MOTION_SOURCE={SOURCE!r}+'/FROZEN_MOTION_SOURCE.py')
import torch;torch.set_num_threads(1)
names=['tools.coopmwm.'+n for n in ('test_parallel_groups_cpu','test_tail_runtime_cpu','test_quota_admission_cpu','test_capacity_broker_cpu','test_acceleration_cpu','test_awr_cpu','test_exact_grade_cpu','test_committed_cpu','test_overlap_cpu','test_mppi_search_cpu','test_mppi_runtime_cpu','test_warm_health_cpu','test_warm_broker_cpu','test_warm_transport_cpu')]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names);start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('no SUMO/inference during qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,root={SOURCE!r},wall_s=time.monotonic()-start)))''',timeout=90)
    print(out);v=json.loads(out.strip().splitlines()[-1]);v['source_hashes']=staged['hashes']
    write(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'),v)
    assert v['pass_'] and v['sim_s']==v['skips']==0


def register():
    cpu=read(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
    value=dict(source_root=SOURCE,cpu=cpu,new_sumo_sim_s=0,
        rule='while quota unmet, admitted decision window publishes all eligible disjoint groups before waiting; same single Bernoulli/window; stable first accepted in inherited spatial group order, one plan-start/tick; no score/completion-order ranking across groups; all unexecuted forecasts logged',
        accounting='one possibly executable window reservation for a parallel group batch; all batch queries protected from capacity decline; original .95/free-slot quota predicate; after quota executes, original single-group nonblocking path',
        invariants='same WM/search/CRN/warm history/current-policy centre/library/K4H300/K8Hfinal/topk/Thompson/beta; exact grader one query draw and one beta draw per window; R unchanged clean matching; cap7500000/allocation7495040/free4960/deadline unchanged',
        supervision='qualified source-bound remote transition freezes only original receipt-owned B parent after all24claims; old SUMO/GPU/grade/broker/R continue; adopt all24completed results, checkpoint/Adam/RNG/trust/library/diagnostic-selection RNG; no completed optimizer/physics/query reruns',
        reporting='per-update query/accepted/executed, collection wall including adopted claim times, completed-inference busy fractions per slot with pending censored')
    target=ops.EVIDENCE/'TAIL_MAIN0818_AMENDMENT.json';assert not target.exists();write(target,value);h=sha(target)
    line='\nS-COOPMWM MAIN0818 QUOTA-TAIL GPU FAN-OUT ENGINEERING REGISTERED amendmentSHA'+h+': '+json.dumps({k:v for k,v in value.items() if k!='cpu'},sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    binding=dict(source_root=SOURCE,cpu=cpu,amendment_sha256=h)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib
r=pathlib.Path({ROOT!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.warm_health import assert_running
from tools.coopmwm.warm_spool import immutable
assert_running(r,'B')
for name,h in json.loads({json.dumps(cpu['source_hashes'])!r}).items():assert hashlib.sha256((pathlib.Path({SOURCE!r})/name).read_bytes()).hexdigest()==h
assert not (r/'TAIL_PLAN_BINDING.json').exists(),'adopt existing registration'
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
immutable(r/'TAIL_PLAN_BINDING.json',json.loads({json.dumps(binding)!r}))
print(json.dumps(dict(registered=True,amendment_sha256={h!r},new_sumo_sim_s=0)))''',timeout=40)
    write(ops.EVIDENCE/'TAIL_MAIN0818_REGISTRATION.json',dict(binding=binding,remote=json.loads(out)));print(out)


def dispatch():
    reg=read(ops.EVIDENCE/'TAIL_MAIN0818_REGISTRATION.json')
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,subprocess,time
r=pathlib.Path({ROOT!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.warm_health import assert_running
from tools.coopmwm.warm_spool import immutable
assert_running(r,'B');assert {reg['binding']['amendment_sha256']!r} in (r/'AUTHORITY.md').read_text()
assert not (r/'TAIL_TRANSITION_DISPATCH.json').exists(),'adopt original transition; never duplicate'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH={SOURCE!r}+'/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
with (r/'TAIL_TRANSITION.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.tail_ops','serve'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();v=dict(pid=p.pid,start_ticks=s[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),source={SOURCE!r},unix=time.time())
immutable(r/'TAIL_TRANSITION_DISPATCH.json',v);print(json.dumps(v))''',timeout=40)
    v=json.loads(out);write(ops.EVIDENCE/'TAIL_TRANSITION_DISPATCH.json',v);print(out)


def serve():
    from .tail_transition import watch
    print(json.dumps(watch(ROOT,'TAIL_PLAN_BINDING_V2.json')),flush=True)


def repair_transition():
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_tailcpu3.json')
    assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
    value=dict(cpu=cpu,source_root=SOURCE,new_sumo_sim_s=0,
        fix='restore_progress uses existing accel_storage.read for lossless compressed RESULTs; adopt original frozen parent and complete current wave; preserve original CPU2 receipts and failed read-only preflight',
        invariants='same B-u1 checkpoint/Adam/RNG/trust/library and prior diagnostic selection; no SUMO/GPU/optimizer replay, no change to search/exact grading/arms/quotas/budget/deadline; replace only held transition3874249, retain broker3893211/grader/R/GPU')
    target=ops.EVIDENCE/'TAIL_COMPRESSED_RESTORE_AMENDMENT.json';assert not target.exists();write(target,value);h=sha(target)
    line='\nS-COOPMWM TAIL COMPRESSED-RESULT RESTORE REPAIR REGISTERED amendmentSHA'+h+': '+json.dumps({k:v for k,v in value.items() if k!='cpu'},sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    binding=dict(source_root=SOURCE,cpu=cpu,amendment_sha256=h)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,subprocess,time,signal,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running
assert_running(r,'B');assert not (r/'TAIL_TRANSITION_DISPATCH_V2.json').exists(),'adopt existing repair owner'
assert not (r/'TAIL_B_DISPATCH.json').exists() and not (r/'TAIL_HANDOVER_COMPLETE.json').exists(),'inspect completed handover'
for name,h in json.loads({json.dumps(cpu['source_hashes'])!r}).items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==h
old=json.loads((r/'TAIL_TRANSITION_DISPATCH.json').read_text());p=pathlib.Path('/proc',str(old['pid']))
assert old['pid']==3874249 and old['start_ticks']=='59377221' and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
stat=(p/'stat').read_text().split();assert stat[21]==old['start_ticks'] and stat[2]=='T'
assert (p/'cmdline').read_bytes().rstrip(bytes([0])).decode().split(chr(0))==['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.tail_ops','serve']
paths=[x.split(b'=',1)[1].decode() for x in (p/'environ').read_bytes().split(bytes([0])) if x.startswith(b'PYTHONPATH=')];assert paths[0].split(':')[0]==old['source']+'/pytools'
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
immutable(r/'TAIL_PLAN_BINDING_V2.json',json.loads({json.dumps(binding)!r}))
os.kill(old['pid'],signal.SIGTERM);os.kill(old['pid'],signal.SIGCONT)
deadline=time.monotonic()+15
while (p/'stat').exists():
 try:fields=(p/'stat').read_text().split()
 except FileNotFoundError:break
 assert fields[21]==old['start_ticks']
 if fields[2]=='Z':break
 if time.monotonic()>deadline:raise ValueError('old held controller did not exit; no duplicate')
 time.sleep(.1)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
with (r/'TAIL_TRANSITION_V2.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.tail_ops','serve'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
fields=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();v=dict(pid=p.pid,start_ticks=fields[21],boot_id=old['boot_id'],root=str(r),source=str(s),unix=time.time(),previous=old)
immutable(r/'TAIL_TRANSITION_DISPATCH_V2.json',v);print(json.dumps(v))''',timeout=45)
    v=json.loads(out);write(ops.EVIDENCE/'TAIL_TRANSITION_DISPATCH_V2.json',v);print(out)


def hold_transition():
    """Hold only the receipt-owned handover controller on a proven preflight fault."""
    line='\nS-COOPMWM TAIL RESTORE ENGINEERING HOLD REGISTERED: real CPU preflight found compressed RESULT marker read with plain JSON in restore_progress before any new B launch; freeze ONLY owned transition3874249/start59377221 after boot/start/exact serve argv/source-environment verification; old B parent remains frozen, existing SUMO/GPU/broker/grader/R continue; decode with existing lossless reader, CPU requalify, then replace controller/adopt same complete wave; zero new SUMO.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,os,signal,time
r=pathlib.Path({ROOT!r});d=json.loads((r/'TAIL_TRANSITION_DISPATCH.json').read_text())
assert not (r/'TAIL_B_DISPATCH.json').exists() and not (r/'TAIL_HANDOVER_COMPLETE.json').exists(),'inspect existing handover before control'
assert d['pid']==3874249 and d['start_ticks']=='59377221'
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
p=pathlib.Path('/proc',str(d['pid']));s=(p/'stat').read_text().split();assert s[21]==d['start_ticks']
assert (p/'cmdline').read_bytes().rstrip(bytes([0])).decode().split(chr(0))==['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.tail_ops','serve']
paths=[x.split(b'=',1)[1].decode() for x in (p/'environ').read_bytes().split(bytes([0])) if x.startswith(b'PYTHONPATH=')];assert paths[0].split(':')[0]==d['source']+'/pytools'
assert s[2] not in ('Z','T');os.kill(d['pid'],signal.SIGSTOP)
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
value=dict(unix=time.time(),owner=d,reason='real CPU preflight: compressed RESULT markers require accel_storage.read',new_sumo_sim_s=0)
with (r/'TAIL_TRANSITION_ENGINEERING_HOLD.json').open('x') as f:json.dump(value,f)
print(json.dumps(value))''',timeout=40)
    v=json.loads(out);write(ops.EVIDENCE/'TAIL_TRANSITION_ENGINEERING_HOLD.json',v);print(out)


def preflight():
    """Read-only real checkpoint/state restore; never enqueue work or optimize."""
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,time,hashlib
r=pathlib.Path({ROOT!r});sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import torch;torch.set_num_threads(1)
from tools.coopmwm.tail_runtime import restore_progress
from tools.coopmwm.warm_protocol import batch_worlds,SEED
from tools.coopmwm.accel_learner import load
from tools.coopmwm.protocol import digest
from tools.coopmwm.quota_transition import state
frozen=json.loads((r/'TAIL_PARENT_FROZEN.json').read_text());assert state(frozen['owner'],'tools.coopmwm.quota_runtime')=='T'
resume=frozen['resume_update'];p=r/f'B_checkpoint_{{resume:03}}.pt';assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen['checkpoint_sha256']
previous=json.loads((r/'MAIN0508_REPAIR_BINDING_V2.json').read_text())
worlds=lambda u:previous['repair_plan']['worlds'] if u==1 else batch_worlds(u)
start=time.monotonic();policy,opt,rng,payload=load(p);v=restore_progress(r,'B',resume,worlds,SEED)
assert payload['update']==resume and torch.equal(rng.get_state(),payload['generator'])
assert v['beta']==json.loads((r/f'B_UPDATE_{{resume:03}}.json').read_text())['trust']
assert v['learned']==json.loads((r/f'B_SEARCH_LEARNING_{{resume:03}}.json').read_text())
print(json.dumps(dict(unix=time.time(),pass_=True,resume_update=resume,adopt_update=frozen['adopt_update'],checkpoint_sha256=frozen['checkpoint_sha256'],rng_exact=True,optimizer_parameters=len(opt.state),diagnostic_history_records=len(v['running']),diagnostic_count=v['exact_count'],library_entries=len(v['learned']['library']['entries']),state_sha256=digest(v),wall_s=time.monotonic()-start,new_sumo_sim_s=0,queued_jobs=0)))''',timeout=60)
    v=json.loads(out);write(ops.EVIDENCE/f'TAIL_PREFLIGHT_{int(v["unix"])}.json',v);print(out);return v


def status():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,time,os,sys
r=pathlib.Path({ROOT!r});owners={{}}
sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.quota_admission import state as quota_state
for name in ('TAIL_TRANSITION_DISPATCH.json','TAIL_TRANSITION_DISPATCH_V2.json','TAIL_B_DISPATCH.json','QUOTA_B_DISPATCH.json','QUOTA_R_DISPATCH.json','TAIL_DRAIN_BROKER_DISPATCH.json','CENTRE_FAIR_BROKER_DISPATCH.json','EXACT_GRADER_DISPATCH_V2.json'):
 if not (r/name).exists():continue
 d=json.loads((r/name).read_text());assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 p=pathlib.Path('/proc',str(d['pid']),'stat');s=p.read_text().split() if p.exists() else None;assert not s or s[21]==d['start_ticks']
 owners[name]=dict(owner=d,state=s[2] if s else 'exited')
receipts={{p.name:json.loads(p.read_text()) for name in ('TAIL_PARENT_FROZEN.json','TAIL_HANDOVER_COMPLETE.json','TAIL_TRANSITION_FAILED.json','WARM_B_FAILED_T1.json','WARM_GUIDED_STOP_V2.json','TAIL_DRAIN_BROKER_HANDOVER.json') for p in [r/name] if p.exists()}}
updates={{a:max([0]+[int(p.stem.rsplit('_',1)[1]) for p in r.glob(a+'_UPDATE_[0-9][0-9][0-9].json')]) for a in ('A','B','R')}}
u=updates['B']+1;jobs=list((r/'jobs').glob(f'warm_u{{u:03}}_B_*'))
ru=updates['R']+1;rjobs=list((r/'jobs').glob(f'warm_u{{ru:03}}_R_*'))
batch_records=[json.loads(p.read_text()) for p in (r/'quota_requests'/f'u{{u:03}}').glob('*_REQUEST.json')]
parallel=[v for v in batch_records if 'records' in v]
pending=[]
for p in (r/'requests').glob('*.json'):
 if (r/'responses'/p.name).exists():continue
 v=json.loads(p.read_text());a=r/'broker_assignments'/p.name
 pending.append(dict(key=p.stem,world=v['world_id'],time_s=v['time_s'],group_size=len(v['group']),assignment=json.loads(a.read_text()) if a.exists() else None))
print(json.dumps(dict(unix=time.time(),owners=owners,receipts=receipts,updates=updates,current_B=dict(update=u,claimed=sum((p/'CLAIM.json').exists() for p in jobs),closed=sum((p/'RESULT.json').exists() for p in jobs)),current_R=dict(update=ru,claimed=sum((p/'CLAIM.json').exists() for p in rjobs),closed=sum((p/'RESULT.json').exists() for p in rjobs)),quota=quota_state(r,u),parallel_groups=dict(windows=len(parallel),queries=sum(len(v['records']) for v in parallel)),pending=pending,load=os.getloadavg(),logs={{n:(r/n).read_text()[-2000:] for n in ('TAIL_TRANSITION.log','TAIL_TRANSITION_V2.log','TAIL_B.log','TAIL_DRAIN_BROKER.log') if (r/n).exists()}})))''',timeout=40)
    v=json.loads(out);write(ops.EVIDENCE/f'TAIL_STATUS_{int(v["unix"])}.json',v)
    adopted={name:x['owner'] for name,x in v['owners'].items() if name=='TAIL_B_DISPATCH.json'}
    adopted.update({name:x for name,x in v['receipts'].items() if name in ('TAIL_HANDOVER_COMPLETE.json','TAIL_DRAIN_BROKER_HANDOVER.json')})
    for name,value in adopted.items():
        target=ops.EVIDENCE/name
        if target.exists():assert read(target)==value,'remote immutable receipt changed'
        else:write(target,value)
    print(json.dumps(dict(unix=v['unix'],owners={k:dict(pid=x['owner']['pid'],state=x['state']) for k,x in v['owners'].items()},
        updates=v['updates'],current_B=v['current_B'],current_R=v['current_R'],quota=v['quota'],parallel_groups=v['parallel_groups'],
        pending=dict(count=len(v['pending']),queued=sum(x['assignment'] is None for x in v['pending']),
            assigned=[x for x in v['pending'] if x['assignment'] is not None]),
        receipt_names=sorted(v['receipts']),transition_failure=v['receipts'].get('TAIL_TRANSITION_FAILED.json'),
        training_failure=v['receipts'].get('WARM_B_FAILED_T1.json'),stop=v['receipts'].get('WARM_GUIDED_STOP_V2.json'),logs=v['logs'])))
    return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('qualify','register','dispatch','serve','status','preflight','hold_transition','repair_transition'))
    globals()[p.parse_args().mode]()
