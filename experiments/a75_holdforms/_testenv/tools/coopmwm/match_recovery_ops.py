"""MAIN14:47 scoped matching recovery; explicit one-use operations."""
import argparse
import json
from . import ops
from .storage import read, write
from .protocol import REGISTRY, sha, checked
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_matchcpu1'
OVERLAYS=('r_matching','test_r_matching_cpu','plan_client','warm_runner','warm_health',
          'test_warm_health_cpu','capacity_broker','exact_runtime','incident_recovery')


def register():
    target=ops.EVIDENCE/'MAIN1447_MATCH_AMENDMENT.json'
    assert not target.exists(),'adopt existing registration'
    v=dict(authority='MAIN2026-11-02 14:47',cap=9500000,new_sumo_sim_s=0,
        rule='R best effort: same regime then nearest group size within +/-1, then stable ticket id/member order; fallback regime allowed; duration retains registered ticket bucket; record all expected/actual strata, shortfalls and excesses; never assert on mismatch',
        recovery='adopt ALL24 completed R-u2 results and original R-u1 Adam/RNG; preserve B-u3 checkpoint/library and all charges; no episode/query replay',
        stop='release ONLY exact SHA-bound matching AssertionError V3 with canonical authority; fresh faults V4 remain fatal',
        boundary='R resumes through u3 then holds at fixed checkpoint for common u4 verification switch; B remains at u3 pending real smoke/matched activation; no outcome selection',
        services='restore only verified-exited broker/grader; retain existing four GPU processes/assignments, original deadline and observers',
        invariants='SUMO-only grades/reward; K4H300/K8Hfinal warm MPPI; random sampler/no library unchanged; disk guard unchanged; MAIN relay archive authority separate')
    write(target,v);h=sha(target)
    line='\nS-COOPMWM MAIN1447 BEST-EFFORT R MATCHING ENGINEERING REGISTERED amendmentSHA'+h+': '+json.dumps(v,sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'MAIN1447_MATCH_REGISTRATION.json',dict(amendment_sha256=h,line=line))
    print(json.dumps(dict(registered=True,amendment_sha256=h,new_sumo_sim_s=0)))


def cpu():
    prior=read(ops.EVIDENCE/'CPU_coopmwm20261102_recoverycpu1.json')
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
names=('test_r_matching_cpu','test_incident_recovery_cpu','test_incident_cpu','test_warm_health_cpu','test_quota_admission_cpu','test_tail_runtime_cpu','test_parallel_groups_cpu','test_capacity_broker_cpu','test_warm_broker_cpu','test_exact_grade_cpu','test_awr_cpu','test_acceleration_cpu')
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in names);start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('no physics/inference during CPU qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=1).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,root={SOURCE!r},wall_s=time.monotonic()-start)))''',timeout=90)
    print(out);v=json.loads(out.strip().splitlines()[-1]);v['source_hashes']=staged['hashes']
    write(ops.EVIDENCE/'CPU_coopmwm20261102_matchcpu1.json',v)
    assert v['pass_'] and v['skips']==v['sim_s']==0


def bind():
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_matchcpu1.json')
    reg=read(ops.EVIDENCE/'MAIN1447_MATCH_REGISTRATION.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    payload=dict(cpu=cpu,reg=reg,source=SOURCE)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
v=json.loads({json.dumps(payload)!r})
assert not (r/'MATCH_RECOVERY_BINDING.json').exists(),'adopt existing binding'
for n,h in v['cpu']['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h
for name in ('INCIDENT_R_DISPATCH.json','INCIDENT_BROKER_DISPATCH.json','INCIDENT_GRADER_DISPATCH.json'):
 d=json.loads((r/name).read_text());assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 p=pathlib.Path('/proc',str(d['pid']),'stat');st=p.read_text().split() if p.exists() else None
 assert not st or st[21]==d['start_ticks'] and st[2]=='Z','existing owner live'
stop=r/'WARM_GUIDED_STOP_V3.json';assert json.loads(stop.read_text())['error']=="AssertionError('R count/regime/group-size/duration gate')"
assert not list(r.glob('PAUSE_*.json'))
b=json.loads((r/'INCIDENT_RECOVERY_BINDING.json').read_text())
for w in b['plans']['R']['worlds']:assert (r/'jobs'/f'warm_u002_R_{{w["seed"]}}'/'RESULT.json').exists()
assert not (r/'R_UPDATE_002.json').exists()
assert (r/'B_UPDATE_003.json').exists() and not (r/'B_UPDATE_004.json').exists()
b.update(cpu=v['cpu'],source_root=str(s),amendment_sha256=v['reg']['amendment_sha256'],receipt_suffix='_M1',stop_after_update={{'R':3}},history_plans={{'B':{{'3':b['plans']['B']['worlds']}}}})
b['resume']['B']=3;b['checkpoints']['B']=hashlib.sha256((r/'B_checkpoint_003.pt').read_bytes()).hexdigest()
with (r/'AUTHORITY.md').open('a') as f:f.write(v['reg']['line'])
immutable(r/'MATCH_RECOVERY_BINDING.json',b)
immutable(r/'MATCH_GUIDED_RELEASE.json',dict(original_stop_sha256=hashlib.sha256(stop.read_bytes()).hexdigest(),amendment_sha256=b['amendment_sha256']))
old=json.loads((r/'INCIDENT_BROKER_CONFIG.json').read_text())
immutable(r/'MATCH_BROKER_CONFIG.json',dict(old,amendment_sha256=b['amendment_sha256'],source_root=str(s),receipt_tag='_M1'))
print(json.dumps(dict(bound=True,unix=time.time(),resume=b['resume'],stop_after_update=b['stop_after_update'],budget=b['budget'],new_sumo_sim_s=0)))''',timeout=45)
    write(ops.EVIDENCE/'MAIN1447_MATCH_BOUND.json',json.loads(out));print(out)


def dispatch(component):
    assert component in ('broker','grader','R'),'B held at u3 for verification boundary'
    dest='MATCH_'+component.upper()+'_DISPATCH.json'
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,subprocess,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.incident_recovery import binding
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running
b=binding(r);assert_running(r,'R');assert not (r/{dest!r}).exists(),'adopt existing owner'
old=json.loads((r/{('INCIDENT_'+component.upper()+'_DISPATCH.json')!r}).read_text())
assert old['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
p=pathlib.Path('/proc',str(old['pid']),'stat');st=p.read_text().split() if p.exists() else None
assert not st or st[21]==old['start_ticks'] and st[2]=='Z','old owner live'
if {component!r}=='R':
 from tools.coopmwm.exact_runtime import assert_live
 assert_live(r)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
with (r/{('MATCH_'+component.upper()+'.log')!r}).open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.incident_recovery',str(r),'--component',{component!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source=str(s),root=str(r),unix=time.time(),component={component!r})
immutable(r/{dest!r},v);print(json.dumps(v))''',timeout=45)
    write(ops.EVIDENCE/dest,json.loads(out));print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','cpu','bind','dispatch'))
    p.add_argument('--component',choices=('broker','grader','R'));a=p.parse_args()
    dispatch(a.component) if a.mode=='dispatch' else globals()[a.mode]()
