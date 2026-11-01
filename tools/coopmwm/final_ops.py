"""Register/dispatch outcome-blind final precompute; expose metadata only."""
import argparse
import json
import time
from . import ops,exact_ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha,digest
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_finalcpu1'
DISCOVERY='/home/ubuntu/coopmwm20261102_finaldiscovery1'


def qualify():
    target=ops.EVIDENCE/'FINAL_CPU_QUALIFICATION.json';assert not target.exists()
    original=read(ops.EVIDENCE/'CPU_coopmwm20261102_exactcpu5.json')
    frozen=json.loads(ops.stage_frozen('cpuQ',SOURCE,original))
    files={'tools/coopmwm/'+n:checked(ops.LOCAL/'tools/coopmwm'/n).read_text(encoding='utf-8')
           for n in ('final_runtime.py','test_final_runtime_cpu.py')}
    extra=json.loads(ops.stage('cpuQ',SOURCE,files))
    hashes=dict(frozen['hashes'],**extra['hashes'])
    output=ops.ssh.remote('cpuQ',f'''import os,sys,time,unittest,json
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
import torch;torch.set_num_threads(1)
from unittest.mock import patch
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in ('test_final_runtime_cpu','test_branch_eval_cpu'))
start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('SUMO forbidden in final CPU qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
from tools.coopmwm.final_runtime import make_job,specs
from tools.coopmwm.accel_storage import read
from pathlib import Path
r=Path({ROOT!r});b=read(r/'WARM_BINDING.json')
# True-native uses its own actor state schema; MAPPO uses the zero adapter.
for arm in ('native','mappo'):
 j=make_job(r,b,specs((arm,))[0]);assert j['deterministic'] and not j['guided']
 if arm=='native':assert j['native_reference'] and j['authority']==b['authority']
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),wall_s=time.monotonic()-start,sim_s=0)))''',timeout=90)
    print(output);value=json.loads(output.strip().splitlines()[-1])
    value.update(source_root=SOURCE,source_hashes=hashes,original_cpu=original['root'])
    write(target,value)
    assert value['pass_'] and value['skips']==value['sim_s']==0


def register():
    from .branch_eval import build_plan,plan_sha
    from .warm_protocol import budget
    target=ops.EVIDENCE/'FINAL_PRECOMPUTE_REGISTRATION.json';assert not target.exists()
    cpu=read(ops.EVIDENCE/'FINAL_CPU_QUALIFICATION.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    plan=build_plan(times=[300,500,700,900],later_arms=[])
    for a in ('A','B'):plan['arm_execution'][a]='fixed update48 checkpoint; deterministic; no guidance'
    plan['plan_sha256']=plan_sha(plan)
    amendment=dict(source_root=SOURCE,source_hashes=cpu['source_hashes'],cpu=cpu,
        owner_addendum='MAIN04:23',budget=budget(),branch_plan=plan,workers=24,
        scheduling='cpuQ only, nice19+LinuxSCHED_IDLE for every worker/SUMO descendant; no cross-host helper activation; pause new submissions while exact grades pending/running, load>=28, MemAvailable<8GiB, RAMfree<8GiB or diskfree<2GiB',
        precompute='after A48 complete: MAPPO/native/A48 episodes96each, Abranches384; first one-world/one-branch600s MAPPO-reuse equivalence; mismatch STOP, no extra allocation',
        outcomes='during all training no metrics/analysis/summaries; explicit metadata+SHA only; worker-only prefix equivalence validation; no protected directory traversal',
        precompute_sim_s=691800,extra_allocation_sim_s=0,
        continuation='wait fixedA48/B48/R24, then B/R96episodes, B384branches, registered paired episode/bootstrap+branch/Holm analyses once; detector-first discovery remains separate reserved phase',
        checkpoint_selection='none; A/B48,R24 fixed; never use dev/final outcomes',
        output_storage='new unclaimed job symlinks into /dev/shm/coopmwm20261102_final1; explicit allowed-artifact hashes only; no moves/deletions',
        failure='one-use supervisor/claims, complete precharge before each job, no physics retries/refunds; preserve private traceback; safe public error type only')
    p=ops.EVIDENCE/'FINAL_PRECOMPUTE_AMENDMENT.json';write(p,amendment);h=sha(p)
    line='\nS-COOPMWM MAIN0423 OUTCOME-BLIND FINAL PRECOMPUTE REGISTERED amendmentSHA'+h+': fixedA48 completion gate; cpuQ nice19/SCHED_IDLE up to24 idle workers, exact grading and B/R priority; MAPPO/native/A48 episodes and Abranches once, first600s MAPPO-reuse exact check; 691800sim-s WITHIN existing final_test/paired_branch/branch_reuse_equivalence allocations, no new allocation. Public receipts hashes/status only; no outcomes/analysis before ALL fixedA48/B48/R24 complete; final continuation consumes remaining registered B/R episode/Bbranch allocations once, no Rbranch/nativebranch, no checkpoint selection, no retries/refunds/deletions; immutable CPU-qualified source.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    config=dict(amendment,amendment_sha256=h)
    reply=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({ROOT!r});assert r.is_dir();sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.warm_spool import immutable
assert not (r/'FINAL_PRECOMPUTE_DISPATCH.json').exists()
assert not (r/'FINAL_PRECOMPUTE_STARTED.json').exists()
for name in ('FINAL_TEST_STARTED.json','FINAL_TEST_COMPLETE.json','FINAL_OUTCOMES_OPENED.json'):
 assert not (r/name).exists(),'existing evaluation must be reconciled before dispatch'
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
immutable(r/'FINAL_PRECOMPUTE_CONFIG.json',json.loads({json.dumps(config)!r}))
print('registered; no physics')''',timeout=40)
    write(target,dict(amendment_sha256=h,line=line,reply=reply,new_sumo_sim_s=0))
    print(json.dumps(dict(registered=True,amendment_sha256=h,precompute_sim_s=691800,
                         allocation=budget()['allocated'],cap=budget()['total'])))


def dispatch():
    target=ops.EVIDENCE/'FINAL_PRECOMPUTE_DISPATCH.json';assert not target.exists()
    reg=read(ops.EVIDENCE/'FINAL_PRECOMPUTE_REGISTRATION.json')
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,sys,os,json,time,subprocess,hashlib
r=pathlib.Path({ROOT!r});source=pathlib.Path({SOURCE!r});assert r.is_dir() and source.is_dir()
sys.path.insert(0,str(source/'pytools'))
from tools.coopmwm.warm_spool import immutable
c=json.loads((r/'FINAL_PRECOMPUTE_CONFIG.json').read_text());assert c['amendment_sha256']=={reg['amendment_sha256']!r}
for n,h in c['source_hashes'].items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h,n
assert not (r/'FINAL_PRECOMPUTE_DISPATCH.json').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',
 PYTHONPATH=str(source/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
with (r/'FINAL_PRECOMPUTE.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.final_runtime',str(r),str(r/'FINAL_PRECOMPUTE_CONFIG.json')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()
v=dict(pid=p.pid,start_ticks=s[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source=str(source),root=str(r),amendment_sha256=c['amendment_sha256'])
immutable(r/'FINAL_PRECOMPUTE_DISPATCH.json',v);print(json.dumps(v))''',timeout=40))
    write(target,value);print(json.dumps(value))


def status():
    # Never read jobs/RESULT, private log, outcome files or analysis output here.
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time
r=pathlib.Path({ROOT!r});d=json.loads((r/'FINAL_PRECOMPUTE_DISPATCH.json').read_text())
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
p=pathlib.Path('/proc',str(d['pid']),'stat');s=p.read_text().split() if p.exists() else None
assert not s or s[21]==d['start_ticks']
allowed=('FINAL_PRECOMPUTE_STARTED.json','FINAL_PRECOMPUTE_FIXED_A.json','FINAL_MAPPO_REUSE_PASS.json','FINAL_PRECOMPUTE_COMPLETE.json','FINAL_PRECOMPUTE_FAILED.json','FINAL_DISCOVERY_STARTED.json','FINAL_DISCOVERY_FAILED.json')
v=dict(unix=time.time(),owner=d,state=s[2] if s else 'exited',receipts={{n:json.loads((r/n).read_text()) for n in allowed if (r/n).exists()}},completed_jobs=len(list((r/'final_receipts').glob('*.json'))),outcomes_read=False)
latest=sorted(r.glob('FINAL_PRECOMPUTE_STATUS_*.json'))
if latest:v['latest_status']=json.loads(latest[-1].read_text())
print(json.dumps(v))''',timeout=40))
    write(ops.EVIDENCE/f'FINAL_PRECOMPUTE_STATUS_{int(value["unix"])}.json',value)
    print(json.dumps(value));return value


def discovery():
    target=ops.EVIDENCE/'FINAL_DISCOVERY_DISPATCH.json';assert not target.exists()
    cpu=read(ops.EVIDENCE/'FINAL_CPU_QUALIFICATION.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    frozen=json.loads(ops.stage_frozen('cpuQ',DISCOVERY,dict(root=cpu['source_root'],source_hashes=cpu['source_hashes'])))
    files={'tools/coopmwm/'+n:checked(ops.LOCAL/'tools/coopmwm'/n).read_text(encoding='utf-8')
           for n in ('final_discovery.py','test_final_discovery_cpu.py')}
    staged=json.loads(ops.stage('cpuQ',DISCOVERY,files))
    hashes=dict(frozen['hashes'],**staged['hashes'])
    output=ops.ssh.remote('cpuQ',f'''import os,sys,time,json,unittest
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
sys.path[:0]=[{DISCOVERY!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from unittest.mock import patch
start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('SUMO forbidden during discovery CPU tests')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.test_final_discovery_cpu'))
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,wall_s=time.monotonic()-start)))''',timeout=60)
    print(output);qual=json.loads(output.strip().splitlines()[-1])
    write(ops.EVIDENCE/'FINAL_DISCOVERY_CPU.json',dict(qual,source_root=DISCOVERY,source_hashes=hashes))
    assert qual['pass_'] and qual['skips']==qual['sim_s']==0
    config=dict(source_root=DISCOVERY,source_hashes=hashes,qualification=qual,
        opening_gate='all fixedA48/B48/R24 + FINAL_OUTCOMES_OPENED + FINAL_REGISTERED_EVALUATION_COMPLETE',
        executor='existing immutable accel_runner.exact_plan n+2 full-prefix reruns, H60/120/300/600; B48 recorded raw1s acceleration; MAPPO member reversion; no WM or truth input',
        group_rule='unchanged observable100m components2..6 containing detector members; no new detector thresholds',
        duration_rule='round detector measured duration up to registered grid30/60/90/120/150/180/240/300; unknown end uses registered60s',
        sampling='fixed panel order then event time/group, no gain-based selection; remaining registered discovery200000 only, preserve existing TRAIN charges; incomplete n+2 budget -> uncovered, never partial group',
        priority='nice19; submit2twinworkers only under existing grade/resource idle admission; no physics retry, no new allocation')
    path=ops.EVIDENCE/'FINAL_DISCOVERY_AMENDMENT.json';write(path,config);h=sha(path)
    config['amendment_sha256']=h
    line='\nS-COOPMWM FINAL DETECTOR-FIRST DISCOVERY CONTINUATION REGISTERED amendmentSHA'+h+': after fixedA48/B48/R24 and registered final outcome opening/analysis only; unchanged COOPBEHAV detectors, existing100m components2..6 containing observed event members (no legacy pair-only gate), registered duration grid; immutable n+2 exact executor/prefix gates, remaining discovery200000 only, all previous charges retained, insufficient whole-group budget recorded uncovered; fixed panel/time/group order, no outcome selection, no new behavioural threshold, nice19/grade-resource priority, no retries/refunds/new allocation.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,sys,os,json,time,subprocess,hashlib
r=pathlib.Path({ROOT!r});source=pathlib.Path({DISCOVERY!r});assert r.is_dir() and source.is_dir()
sys.path.insert(0,str(source/'pytools'))
from tools.coopmwm.warm_spool import immutable
c=json.loads({json.dumps(config)!r})
for n,h in c['source_hashes'].items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h,n
assert not (r/'FINAL_DISCOVERY_DISPATCH.json').exists()
assert not (r/'FINAL_DISCOVERY_STARTED.json').exists()
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
immutable(r/'FINAL_DISCOVERY_CONFIG.json',c)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',
 PYTHONPATH=str(source/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
with (r/'FINAL_DISCOVERY.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.final_discovery',str(r),str(r/'FINAL_DISCOVERY_CONFIG.json')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()
v=dict(pid=p.pid,start_ticks=s[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source=str(source),root=str(r),amendment_sha256=c['amendment_sha256'])
immutable(r/'FINAL_DISCOVERY_DISPATCH.json',v);print(json.dumps(v))''',timeout=40))
    write(target,dict(value,line=line));print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('qualify','register','dispatch','status','discovery'))
    globals()[p.parse_args().mode]()
