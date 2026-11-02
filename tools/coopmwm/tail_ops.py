"""Explicit qualification/registration/one-use boundary deployment for MAIN08:18."""
import argparse
import json
from . import ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_tailcpu2'
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
    print(json.dumps(watch(ROOT)),flush=True)


def status():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,time,os
r=pathlib.Path({ROOT!r});owners={{}}
for name in ('TAIL_TRANSITION_DISPATCH.json','TAIL_B_DISPATCH.json','QUOTA_B_DISPATCH.json','QUOTA_R_DISPATCH.json','CENTRE_FAIR_BROKER_DISPATCH.json','EXACT_GRADER_DISPATCH_V2.json'):
 if not (r/name).exists():continue
 d=json.loads((r/name).read_text());assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 p=pathlib.Path('/proc',str(d['pid']),'stat');s=p.read_text().split() if p.exists() else None;assert not s or s[21]==d['start_ticks']
 owners[name]=dict(owner=d,state=s[2] if s else 'exited')
receipts={{p.name:json.loads(p.read_text()) for name in ('TAIL_PARENT_FROZEN.json','TAIL_HANDOVER_COMPLETE.json','TAIL_TRANSITION_FAILED.json','WARM_B_FAILED_T1.json','WARM_GUIDED_STOP_V2.json') for p in [r/name] if p.exists()}}
updates={{a:max([0]+[int(p.stem.rsplit('_',1)[1]) for p in r.glob(a+'_UPDATE_[0-9][0-9][0-9].json')]) for a in ('A','B','R')}}
u=updates['B']+1;jobs=list((r/'jobs').glob(f'warm_u{{u:03}}_B_*'))
print(json.dumps(dict(unix=time.time(),owners=owners,receipts=receipts,updates=updates,current_B=dict(update=u,claimed=sum((p/'CLAIM.json').exists() for p in jobs),closed=sum((p/'RESULT.json').exists() for p in jobs)),load=os.getloadavg(),logs={{n:(r/n).read_text()[-2000:] for n in ('TAIL_TRANSITION.log','TAIL_B.log') if (r/n).exists()}})))''',timeout=40)
    v=json.loads(out);write(ops.EVIDENCE/f'TAIL_STATUS_{int(v["unix"])}.json',v);print(out);return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('qualify','register','dispatch','serve','status'))
    globals()[p.parse_args().mode]()
