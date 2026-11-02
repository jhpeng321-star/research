"""Register the 9.5M verification allocation, qualify and launch ONE real smoke."""
import argparse
import copy
import json
from . import ops
from .storage import read,write
from .protocol import REGISTRY,checked,sha
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_verifycpu11'


def register():
    target=ops.EVIDENCE/'SUMO_VERIFICATION_PLAN.json'
    assert not target.exists(),'adopt existing registration'
    b=copy.deepcopy(read(ops.EVIDENCE/'MAIN1011_RECOVERY_PLAN.json')['budget'])
    b['total']=9500000
    b['phases']['sumo_verification']=2800000
    b['phases']['exact_plan_grading']-=328840
    b['allocated']=sum(b['phases'].values());b['unallocated']=b['total']-b['allocated']
    assert b['allocated']==9500000 and b['unallocated']==0
    v=dict(authority='OWNER10:20/10:22 and MAIN10:30/14:24/14:47',budget=b,
        selection='B top4 WM candidates and own-policy branch; R4 random candidates and own-policy branch; same common update4; beta BEFORE branches; exact lowest-delay winner only when strictly better than own policy; grade reused, no duplicate grader run',
        quote='remaining B21+R21 updates x minimum10 x max6000 =2520000 sim-s; 2800000 verification phase covers that lower-bound plan plus smoke/rejected decisions/extra plans up to phase cap. Variable acceptance is not guaranteed; phase/cap blockers STOP and report, never reduce horizon or bypass accounting',
        funding='owner cap+2000000, prior spare471160, replace328840 of unspent separate exact grading with reused verifier grades; remaining exact grade431160 (already charged16530 retained); no discovery transfer',
        smoke_rule='first B-u3 completed replacement episode in registered ascending seed order, first accepted forecast with exactly4 topk; no exact grades/outcomes consulted; new-ID replay through t0+10, five exact branches plus policy continuation, all charged in sumo_verification; never retry physics',
        invariants='K4H300/K8Hfinal, warm memory, search, trust, PPO/AWR, health pauses unchanged; primaryA24/B24/R24, secondaryA48; fixed sealed test plan retained; no outcome opening')
    write(target,v);h=sha(target)
    line='\nS-COOPMWM SUMO VERIFICATION ALLOCATION/COMMON-U4 REGISTERED planSHA'+h+': '+json.dumps(v,sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'SUMO_VERIFICATION_REGISTRATION.json',dict(amendment_sha256=h,line=line))
    print(json.dumps(dict(registered=True,amendment_sha256=h,budget=b)))


def cpu():
    from . import verify_ops as v
    v.SOURCE=SOURCE
    v.OVERLAYS=tuple(dict.fromkeys(v.OVERLAYS+('warm_protocol','warm_runner','warm_health',
        'r_matching','test_r_matching_cpu','test_warm_health_cpu','incident_recovery',
        'capacity_broker','verification_smoke')))
    v.cpu()


def register_gate():
    target=ops.EVIDENCE/'VERIFICATION_SMOKE_GATE_AMENDMENT.json';assert not target.exists()
    v=dict(diagnosis='all five real branches completed prefix/CRN checks; candidates rejected and policy continued; helper wrongly required acceptance',
        correction='validate live continuation against chosen candidate when accepted or policy branch when rejected; SHA/prefix/CRN/no-new-plan checks retained; no outcome-conditioned repeat',
        recovery='CPU qualify then recheck original completed TRAIN live/branch receipts ONLY; zero new SUMO/inference/charges; retain initial gate failure and all source/dispatch receipts')
    write(target,v);h=sha(target)
    line='\nS-COOPMWM OUTCOME-INDEPENDENT VERIFICATION SMOKE GATE REGISTERED amendmentSHA'+h+': '+json.dumps(v,sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'VERIFICATION_SMOKE_GATE_REGISTRATION.json',dict(amendment_sha256=h,line=line))
    print(json.dumps(dict(registered=True,amendment_sha256=h,new_sumo_sim_s=0)))


def recheck():
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_verifycpu11.json')
    reg=read(ops.EVIDENCE/'VERIFICATION_SMOKE_GATE_REGISTRATION.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.accel_storage import read
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.verification_grades import record_smoke
cpu=json.loads({json.dumps(cpu)!r})
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h
v=read(r/'VERIFICATION_RUNTIME_BINDING.json');spec=v['smoke']
live=read(r/'jobs'/spec['id']/'RESULT.json')
paths=list((r/'sumo_verification_smoke'/'verifications').glob('*.json'));assert len(paths)==1
receipt=read(paths[0])
with (r/'AUTHORITY.md').open('a') as f:f.write({reg['line']!r})
gate=record_smoke(r/'sumo_verification_smoke',receipt,live)
record_smoke(r/'sumo_verification',receipt,live)
passed=dict(gate,branch_count=5,charged_original_sim_s=spec['quote_sim_s'],recheck_new_sim_s=0,source_root=str(s),amendment_sha256={reg['amendment_sha256']!r})
immutable(r/'VERIFICATION_SMOKE_PASSED.json',passed)
print(json.dumps(passed))''',timeout=45)
    write(ops.EVIDENCE/'VERIFICATION_SMOKE_PASSED.json',json.loads(out));print(out)


def bind():
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_verifycpu10.json')
    reg=read(ops.EVIDENCE/'SUMO_VERIFICATION_REGISTRATION.json')
    plan=read(ops.EVIDENCE/'SUMO_VERIFICATION_PLAN.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    payload=dict(cpu=cpu,source_root=SOURCE,amendment_sha256=reg['amendment_sha256'],budget=plan['budget'])
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.accel_storage import read
from tools.coopmwm.exact_grade import train_noise_seed
v=json.loads({json.dumps(payload)!r});assert not (r/'VERIFICATION_RUNTIME_BINDING.json').exists(),'adopt existing binding'
for n,h in v['cpu']['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h
old=read(r/'INCIDENT_RECOVERY_BINDING.json');found=None
for w in old['plans']['B']['worlds']:
 ident='warm_u003_B_'+str(w['seed']);ep=read(r/'jobs'/ident/'RESULT.json')
 for step in ep['steps']:
  p=step.get('proposal') or {{}}
  if p.get('accepted') and len(p.get('topk',[]))==4:
   t=int(step['time_s']);stop=(t//10+1)*10;end=min(t+max(c['duration_s'] for c in p['topk'])+120,1200)
   found=dict(id='verification_smoke_u003_B_'+str(w['seed'])+'_t'+str(t),original_episode=ident,time_s=t,stop=stop,noise_seed=train_noise_seed(w['seed'],3,20261101),original_result_sha256=hashlib.sha256((r/'jobs'/ident/'RESULT.json').read_bytes()).hexdigest(),quote_sim_s=stop+5*end)
   break
 if found:break
assert found is not None
v['smoke']=found;v['common_update']=4
with (r/'AUTHORITY.md').open('a') as f:f.write({reg['line']!r})
immutable(r/'VERIFICATION_RUNTIME_BINDING.json',v)
print(json.dumps(dict(bound=True,unix=time.time(),smoke=found,budget=v['budget'],new_physics=0)))''',timeout=45)
    write(ops.EVIDENCE/'SUMO_VERIFICATION_BOUND.json',json.loads(out));print(out)


def smoke():
    dest='VERIFICATION_SMOKE_DISPATCH.json'
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,os,sys,subprocess,time,shutil
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running
assert_running(r,'B');assert not (r/{dest!r}).exists() and not (r/'VERIFICATION_SMOKE_STARTED.json').exists()
assert shutil.disk_usage(r).free>=2*1024**3
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
with (r/'VERIFICATION_SMOKE.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.verification_smoke',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source=str(s),root=str(r),unix=time.time())
immutable(r/{dest!r},v);print(json.dumps(v))''',timeout=40)
    write(ops.EVIDENCE/dest,json.loads(out));print(out)


def status():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time,shutil
r=pathlib.Path({ROOT!r});names=('VERIFICATION_SMOKE_DISPATCH.json','VERIFICATION_SMOKE_STARTED.json','VERIFICATION_SMOKE_PASSED.json','WARM_GUIDED_STOP_V4.json')
v=dict(unix=time.time(),receipts={{n:json.loads((r/n).read_text()) for n in names if (r/n).exists()}},log=(r/'VERIFICATION_SMOKE.log').read_text()[-4000:] if (r/'VERIFICATION_SMOKE.log').exists() else None,disk_free=shutil.disk_usage(r).free)
d=v['receipts'].get('VERIFICATION_SMOKE_DISPATCH.json');p=pathlib.Path('/proc',str(d['pid']),'stat') if d else None
v['state']=p.read_text().split()[2] if p and p.exists() else 'exited'
print(json.dumps(v))''',timeout=35))
    write(ops.EVIDENCE/f'VERIFICATION_SMOKE_STATUS_{int(value["unix"])}.json',value)
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','cpu','bind','smoke','status','register_gate','recheck'))
    globals()[p.parse_args().mode]()
