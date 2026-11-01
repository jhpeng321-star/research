"""Qualify/register/release only the diagnosed MAIN05:08 JSON-publication fault."""
import argparse
import json
from pathlib import Path
from . import ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_capacitycpu2'


def cpu():
    prior=read(ops.EVIDENCE/'CPU_coopmwm20261102_exactcpu5.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['root']!r});files={{}}
for name,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 files[name.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=40))
    names=('capacity_broker','capacity_recovery','capacity_repair_ops','test_capacity_broker_cpu','test_capacity_recovery_cpu',
        'plan_client','accel_runner','warm_runner','warm_health','exact_runtime')
    for name in names:files['tools/coopmwm/'+name+'.py']=(ops.LOCAL/'tools/coopmwm'/(name+'.py')).read_text(encoding='utf-8')
    staged=json.loads(ops.stage('cpuQ',SOURCE,files))
    out=ops.ssh.remote('cpuQ',f'''import sys,unittest,os,json,time
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',COOPMWM_COST_SOURCE={SOURCE!r}+'/FROZEN_COSTS.json',COOPMWM_MOTION_SOURCE={SOURCE!r}+'/FROZEN_MOTION_SOURCE.py')
import torch;torch.set_num_threads(1)
from unittest.mock import patch
start=time.monotonic()
names=['tools.coopnative.test_cpu']+['tools.coopmwm.'+n for n in ('test_cpu','test_volume_cpu','test_acceleration_cpu','test_warm_cpu','test_mppi_search_cpu','test_branch_eval_cpu','test_mppi_runtime_cpu','test_warm_protocol_cpu','test_awr_cpu','test_warm_health_cpu','test_warm_transport_cpu','test_warm_spool_cpu','test_warm_broker_cpu','test_committed_cpu','test_overlap_cpu','test_exact_grade_cpu','test_plan_proposer_cpu','test_capacity_broker_cpu','test_capacity_recovery_cpu')]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
with patch('subprocess.Popen',side_effect=AssertionError('physics forbidden in CPU qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,root={SOURCE!r},wall_s=time.monotonic()-start)))''',timeout=90)
    print(out);v=json.loads(out.strip().splitlines()[-1]);v['source_hashes']=staged['hashes']
    write(ops.EVIDENCE/'CPU_coopmwm20261102_capacitycpu2.json',v);assert v['pass_'] and not v['skips']


def register():
    from .capacity_recovery import replacement_plan,repair_budget
    from .warm_protocol import batch_worlds,worlds,budget
    target=ops.EVIDENCE/'MAIN0508_REPAIR_REGISTRATION.json';assert not target.exists()
    c=read(ops.EVIDENCE/'CPU_coopmwm20261102_capacitycpu2.json');assert c['pass_'] and c['skips']==c['sim_s']==0
    state=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time,hashlib
r=pathlib.Path({ROOT!r});owners={{}}
for name in ('WARM_B_DISPATCH_V4.json','WARM_R_DISPATCH_V4.json','EXACT_GRADER_DISPATCH.json','CAPACITY_BROKER_DISPATCH.json'):
 d=json.loads((r/name).read_text());assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
 p=pathlib.Path('/proc',str(d['pid']),'stat');st=p.read_text().split() if p.exists() else None
 assert not st or (st[21]==d['start_ticks'] and st[2]=='Z'),'previous owner alive; never duplicate'
 owners[name]=d
assert not list(r.glob('PAUSE_*.json')),'scientific health pause is not released'
stop=r/'WARM_GUIDED_STOP.json';v=json.loads(stop.read_text())
assert v['source']=='capacity broker control' and v['error']=="JSONDecodeError('Expecting value: line 1 column 1 (char 0)')"
jobs=[]
for p in (r/'jobs').glob('warm_u001_B_*'):
 assert not (p/'RESULT.json').exists(),'completed work must be reused'
 claim=p/'CLAIM.json';jobs.append(dict(seed=int(p.name.rsplit('_',1)[1]),claimed=claim.exists()))
print(json.dumps(dict(owners=owners,jobs=jobs,stop=v,stop_sha256=hashlib.sha256(stop.read_bytes()).hexdigest(),unix=time.time())))''',timeout=40))
    plan=replacement_plan(batch_worlds(1),worlds('train'),{j['seed'] for j in state['jobs'] if j['claimed']})
    b=repair_budget(budget(),plan['additional_sim_s']);assert plan['additional_sim_s']==21600 and len(plan['precharged_unattempted'])==6
    value=dict(cpu=c,source_root=SOURCE,state=state,repair_plan=plan,budget=b,
        diagnosis='legacy request create-before-fill read race in MAIN0508 broker; CPU1 did not test that race; all18attempted incomplete B episodes superseded, charges retained;6neverclaimed jobs adopted once;no optimizer update occurred',
        fix='atomic new requests; legacy incomplete JSON waits <=30s metadata grace thenSTOP; prior broker admission receipts adopted; first engineering STOP preservedSHA; fresh faults use STOP_V2',
        runtime='load existing B/R checkpoint0 with Adam/RNG; B24/R4/grader4; exact grades asynchronous; R waits exact Bquota, then shares lower-priority centre extraction; unchanged MPPI/warm/H300/K8/Nexec10/health/endpoints',
        spending='18 fresh same-cell registered TRAIN worlds21600 charged before collection; six existing unattempted precharges adopted; no failed world repeated; total7495040/7500000/free4960; all other phase liabilities retained',
        gpu='adopt original four GPU owners and assigned in-flight forecasts, no inference retry/new parity run; unassigned requests from superseded episodes receive explicit no-compute retirement responses')
    path=ops.EVIDENCE/'MAIN0508_REPAIR_AMENDMENT.json';write(path,value);h=sha(path)
    line='\nS-COOPMWM MAIN0508 JSON-PUBLICATION FAULT REPAIR REGISTERED amendmentSHA'+h+': '+json.dumps({k:v for k,v in value.items() if k not in ('cpu','state')},sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    reply=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
for name,h in json.loads({json.dumps(c['source_hashes'])!r}).items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==h
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
v=json.loads({json.dumps(dict(value,amendment_sha256=h))!r})
immutable(r/'MAIN0508_REPAIR_BINDING.json',v)
immutable(r/'MAIN0508_GUIDED_RELEASE.json',dict(original_stop_sha256={state['stop_sha256']!r},amendment_sha256={h!r},no_outcome_selection=True))
old=json.loads((r/'WARM_REMOTE_BINDING.json').read_text())
immutable(r/'CAPACITY_BROKER_CONFIG_V2.json',dict(amendment_sha256={h!r},source_root=str(s),deadline_unix=old['deadline_unix'],receipt_tag='_V2'))
from tools.coopmwm.protocol import digest
failed={{w['failed']['seed'] for w in v['repair_plan']['replaced']}}
retired=[]
for p in (r/'requests').glob('*.json'):
 record=json.loads(p.read_text())
 if record.get('world_id') not in failed or (r/'responses'/p.name).exists() or (r/'broker_assignments'/p.name).exists():continue
 immutable(r/'responses'/p.name,dict(accepted=False,reason='superseded infrastructure-failed episode; no model evaluation',record_sha256=digest(record),
  candidates_searched=0,imagined_futures=0,query_admission=dict(wm_admission_probability=0.,amendment_sha256={h!r},superseded_episode=True)))
 retired.append(p.name)
immutable(r/'MAIN0508_RETIRED_UNASSIGNED.json',dict(requests=retired,amendment_sha256={h!r},new_sumo_sim_s=0))
print(json.dumps(dict(registered=True,amendment_sha256={h!r},budget=v['budget'])))''',timeout=40))
    write(target,dict(value,amendment_sha256=h,remote=reply));print(json.dumps(reply))


DISPATCHES=dict(broker='CAPACITY_BROKER_DISPATCH_V2.json',grader='EXACT_GRADER_DISPATCH_V2.json',B='WARM_B_DISPATCH_V5.json',R='WARM_R_DISPATCH_V5.json')


def dispatch(component):
    reg=read(ops.EVIDENCE/'MAIN0508_REPAIR_REGISTRATION.json');dest=DISPATCHES[component]
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,subprocess,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.capacity_recovery import binding
from tools.coopmwm.warm_health import assert_running
from tools.coopmwm.warm_spool import immutable
b=binding(r);assert_running(r,'B');assert not (r/{dest!r}).exists(),'adopt existing new owner'
if {component!r} in ('B','R'):
 from tools.coopmwm.exact_runtime import assert_live
 assert_live(r)
 p=r/'CAPACITY_BROKER_DISPATCH_V2.json';d=json.loads(p.read_text());st=pathlib.Path('/proc',str(d['pid']),'stat').read_text().split();assert st[21]==d['start_ticks']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
with (r/{('CAPACITY_REPAIR_'+component+'.log')!r}).open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.capacity_recovery',str(r),'--component',{component!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()
v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source=str(s),root=str(r),component={component!r})
immutable(r/{dest!r},v);print(json.dumps(v))''',timeout=40))
    write(ops.EVIDENCE/dest,value);print(json.dumps(value))


def status():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time,os
r=pathlib.Path({ROOT!r});owners={{}}
for component,name in json.loads({json.dumps(DISPATCHES)!r}).items():
 if component in ('B','R') and (r/f'WARM_{{component}}_DISPATCH_V6.json').exists():name=f'WARM_{{component}}_DISPATCH_V6.json'
 p=r/name
 if not p.exists():continue
 d=json.loads(p.read_text());st=pathlib.Path('/proc',str(d['pid']),'stat');s=st.read_text().split() if st.exists() else None
 assert not s or s[21]==d['start_ticks'];assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
 owners[component]=dict(owner=d,state=s[2] if s else 'exited')
logs={{}}
for a in ('broker','grader','B','R'):
 p=r/('CAPACITY_REPAIR_'+a+('_V6' if a in ('B','R') and (r/f'WARM_{{a}}_DISPATCH_V6.json').exists() else '')+'.log')
 if p.exists():logs[a]=p.read_text()[-5000:]
print(json.dumps(dict(unix=time.time(),owners=owners,logs=logs,stop=json.loads((r/'WARM_GUIDED_STOP_V2.json').read_text()) if (r/'WARM_GUIDED_STOP_V2.json').exists() else None,
declines=len(list((r/'capacity_declines').glob('*.json'))),load=os.getloadavg(),updates={{a:max([0]+[int(p.stem.rsplit('_',1)[1]) for p in r.glob(a+'_UPDATE_*.json')]) for a in ('A','B','R')}})))''',timeout=40))
    write(ops.EVIDENCE/f'CAPACITY_REPAIR_STATUS_{int(value["unix"])}.json',value);print(json.dumps(value));return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('cpu','register','dispatch','status'));p.add_argument('--component',choices=tuple(DISPATCHES));a=p.parse_args()
    dispatch(a.component) if a.mode=='dispatch' else globals()[a.mode]()
