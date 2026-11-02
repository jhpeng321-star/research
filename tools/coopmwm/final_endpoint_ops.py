"""Qualified replacement of waiting final hooks; no precomputed outcome reads."""
import argparse
import json
from . import ops
from .storage import read,write
from .protocol import REGISTRY,checked,sha
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_finalendpoint3'


def cpu():
    prior=read(ops.EVIDENCE/'FINAL_REPLAY_CPU.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['source_root']!r});files={{}}
for n,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;files[n.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=45))
    for n in ('final_runtime','final_discovery','final_endpoint','test_final_endpoint_cpu','warm_protocol'):
        files['tools/coopmwm/'+n+'.py']=checked(ops.LOCAL/'tools/coopmwm'/(n+'.py')).read_text(encoding='utf-8')
    staged=json.loads(ops.stage('cpuQ',SOURCE,files))
    out=ops.ssh.remote('cpuQ',f'''import os,sys,time,json,unittest
from unittest.mock import patch
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',COOPMWM_COST_SOURCE={SOURCE!r}+'/FROZEN_COSTS.json',COOPMWM_MOTION_SOURCE={SOURCE!r}+'/FROZEN_MOTION_SOURCE.py')
import torch;torch.set_num_threads(1)
names=('test_final_endpoint_cpu','test_final_discovery_cpu','test_final_runtime_cpu','test_branch_eval_cpu','test_acceleration_cpu','test_committed_cpu','test_exact_grade_cpu','test_diagnostic_repair_cpu');start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('no SUMO/inference in endpoint qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=1).run(unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in names))
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,wall_s=time.monotonic()-start)))''',timeout=90)
    print(out);v=json.loads(out.strip().splitlines()[-1]);v.update(source_root=SOURCE,source_hashes=staged['hashes'])
    write(ops.EVIDENCE/'FINAL_ENDPOINT_CPU_V3.json',v)
    assert v['pass_'] and v['skips']==v['sim_s']==0


def register():
    cpu=read(ops.EVIDENCE/'FINAL_ENDPOINT_CPU.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    target=ops.EVIDENCE/'FINAL_ENDPOINT_AMENDMENT.json';assert not target.exists()
    v=dict(qualification=cpu,authority='MAIN10:11 primaryA24/B24/R24, secondaryA48; original frozen-policy precompute permission',
        source_root=SOURCE,source_hashes=cpu['source_hashes'],
        invariants='retain673opaque MAPPO/native/A48/reuse jobs, never rerun; newA24 IDs fixedfinal_A24/fixedbranch_A24, same policy armA with fixedcheckpoint24; B/Rfixed24; original panel/metric/Holm/bootstrap/safety unchanged',
        opening='no outcomes opened before B24 AND R24; A48 completion certifies existing fixed A24 checkpoint; secondaryA48 labels separate; detector-first twins on B24 only after registered final analysis',
        budget='no new allocation: A24episodes+branches460800 already allocated by MAIN1011; discovery150200remaining full n+2 charge guard; every original failed charge retained',
        handover='ONLY verified WAITING final3316828 and discovery3604109, no children/no outcome opening; one detached final+discovery coordinator; never duplicate a reserved/attempted final job')
    write(target,v);h=sha(target)
    line='\nS-COOPMWM FIXED24 FINAL HOOK MIGRATION REGISTERED amendmentSHA'+h+': '+json.dumps({k:x for k,x in v.items() if k not in ('qualification','source_hashes')},sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'FINAL_ENDPOINT_REGISTRATION.json',dict(v,amendment_sha256=h,line=line))
    print(json.dumps(dict(registered=True,amendment_sha256=h,tests=cpu['tests'],new_sumo_sim_s=0)))


def replace():
    reg=read(ops.EVIDENCE/'FINAL_ENDPOINT_REGISTRATION.json')
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,os,sys,signal,time,subprocess
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.protocol import sha
v=json.loads({json.dumps(reg)!r})
for n,h in v['source_hashes'].items():assert sha(s/n)==h
assert not (r/'FINAL_ENDPOINT_DISPATCH.json').exists(),'adopt existing owner'
assert not (r/'FINAL_OUTCOMES_OPENED.json').exists() and not (r/'FINAL_DISCOVERY_STARTED.json').exists()
complete=json.loads((r/'FINAL_PRECOMPUTE_COMPLETE.json').read_text());assert complete['completed_jobs']==673 and complete['outcomes_closed']
assert json.loads((r/'FINAL_MAPPO_REUSE_PASS.json').read_text())['pass_']
owners=[]
for name,module in [('FINAL_PRECOMPUTE_DISPATCH.json','tools.coopmwm.final_runtime'),('FINAL_DISCOVERY_DISPATCH_V2.json','tools.coopmwm.final_discovery')]:
 d=json.loads((r/name).read_text());assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 p=pathlib.Path('/proc',str(d['pid']));st=(p/'stat').read_text().split();args=[a.decode() for a in (p/'cmdline').read_bytes().split(bytes([0])) if a]
 assert st[21]==d['start_ticks'] and st[2]=='S' and module in args and str(r) in args
 children=[]
 for child in (p/'task'/str(d['pid'])/'children').read_text().split():
  cp=pathlib.Path('/proc',child);ca=[a.decode() for a in (cp/'cmdline').read_bytes().split(bytes([0])) if a]
  assert len(ca)==4 and ca[1:3]==['-B','-c'] and ca[3].startswith('from multiprocessing.resource_tracker import main;main('),'active worker child; do not replace'
  children.append(dict(pid=int(child),argv=ca,role='owned multiprocessing bookkeeping only; no physics'))
 owners.append(dict(receipt=name,owner=d,argv=args,bookkeeping_children=children))
with (r/'AUTHORITY.md').open('a') as f:f.write(v['line']+'\\nFinal hook preflight refinement: only an owned multiprocessing.resource_tracker child is allowed; no episode/branch worker children. Resource tracker exits with original parent pipe; no extra signal.\\n')
old=json.loads((r/'FINAL_PRECOMPUTE_CONFIG.json').read_text())
old.update(source_root=str(s),source_hashes=v['source_hashes'],amendment_sha256=v['amendment_sha256'],budget=json.loads((r/'VERIFICATION_CONTINUATION_BINDING.json').read_text())['budget'])
immutable(r/'FINAL_ENDPOINT_CONFIG.json',old)
immutable(r/'FINAL_ENDPOINT_HANDOVER.json',dict(unix=time.time(),owners=owners,outcomes_closed=True,signal='SIGTERM',new_sumo_sim_s=0))
for item in owners:os.kill(item['owner']['pid'],signal.SIGTERM)
for _ in range(100):
 if all(not pathlib.Path('/proc',str(x['owner']['pid']),'stat').exists() or pathlib.Path('/proc',str(x['owner']['pid']),'stat').read_text().split()[2]=='Z' for x in owners):break
 time.sleep(.1)
else:raise ValueError('old final owners not exited; no replacement launch')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime');env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
with (r/'FINAL_ENDPOINT.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.final_endpoint',str(r),str(r/'FINAL_ENDPOINT_CONFIG.json')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();d=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),source=str(s),unix=time.time(),outcomes_closed=True)
immutable(r/'FINAL_ENDPOINT_DISPATCH.json',d);print(json.dumps(d))''',timeout=50)
    write(ops.EVIDENCE/'FINAL_ENDPOINT_DISPATCH.json',json.loads(out));print(out)


def hold():
    target=ops.EVIDENCE/'FINAL_ENDPOINT_SPAWN_DIAGNOSIS.json'
    assert not target.exists(),'adopt existing hold'
    v=dict(diagnosis='spawned final workers do not inherit coordinator monkeypatch of budget(); A24-tagged specs lacked explicit checkpoint update, so children load A48; B would look for B48. Discovered by code inspection before outcome opening',
        action='freeze ONLY current final coordinator parent by verified PID/boot/start/source; let already-reserved isolated workers finish without outcome reads; preserve all charges/receipts; B/R/GPU/grader continue',
        repair='carry checkpoint_update explicitly in every policy job spec; spawned make_job consumes it and publishes checkpoint SHA; CPU regression required',
        budget='replacement final physics above50000sim-s needs MAIN allocation decision; no automatic reserve transfer or rerun')
    write(target,v);h=sha(target)
    line='\nS-COOPMWM FINAL SPAWN CHECKPOINT FAULT REGISTERED amendmentSHA'+h+': '+json.dumps(v,sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,time,os,signal
r=pathlib.Path({ROOT!r});d=json.loads((r/'FINAL_ENDPOINT_DISPATCH.json').read_text());assert d['pid']==625637 and d['start_ticks']=='61917851'
assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
p=pathlib.Path('/proc',str(d['pid']));st=(p/'stat').read_text().split();args=[a.decode() for a in (p/'cmdline').read_bytes().split(bytes([0])) if a]
assert st[21]==d['start_ticks'] and st[2] in ('S','R') and args[-3:]==['tools.coopmwm.final_endpoint',str(r),str(r/'FINAL_ENDPOINT_CONFIG.json')]
assert not (r/'FINAL_OUTCOMES_OPENED.json').exists()
v=dict(unix=time.time(),owner=d,amendment_sha256={h!r},signal='SIGSTOP',reason='spawn checkpoint fault; no new final reservations; reserved workers finish; B/R unaffected',outcomes_closed=True)
with (r/'FINAL_ENDPOINT_SPAWN_HOLD.json').open('x') as f:json.dump(v,f)
os.kill(d['pid'],signal.SIGSTOP)
print(json.dumps(v))''',timeout=40)
    write(ops.EVIDENCE/'FINAL_ENDPOINT_SPAWN_HOLD.json',json.loads(out));print(out)


def request_recovery():
    cpu=read(ops.EVIDENCE/'FINAL_ENDPOINT_CPU_V3.json')
    census=read(ops.EVIDENCE/'FINAL_ENDPOINT_QUARANTINE_CENSUS.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    assert census['quarantined_sim_s']==375800 and not census['outcomes_opened']
    v=dict(qualification=cpu,quarantine=census,
        repair='explicit checkpoint_update serialized in every policy spec; missing key fails closed; make_job loads/publishes exact checkpoint SHA; primaryA24v2 fresh IDs; separate V2attempt markers; no previous artifact overwritten',
        budget_request=dict(total=9500000,additional_final=375800,proposed_transfer_from_exact_plan_grading=375800,
            proposed_final_test_increase=115200,proposed_paired_branch_increase=260600,proposed_exact_grade_allocation=55360),
        authority_required='MAIN allocation approval because amount exceeds delegated50000; no replacement physics dispatched',
        training='B/Ru4+verification/broker/GPU/grader/observer continue; no impact on fixed checkpoint rule; original resource deadline unchanged')
    target=ops.EVIDENCE/'FINAL_ENDPOINT_SPAWN_REPAIR_QUALIFIED.json';assert not target.exists();write(target,v)
    line='\nS-COOPMWM FINAL SPAWN REPAIR QUALIFIED / MAIN ALLOCATION REQUEST evidenceSHA'+sha(target)+': '+json.dumps({k:x for k,x in v.items() if k not in ('qualification','quarantine')},sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    print(json.dumps(dict(qualified=True,tests=cpu['tests'],new_sumo_sim_s=0,budget_request=v['budget_request'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('cpu','register','replace','hold','request_recovery'))
    globals()[p.parse_args().mode]()
