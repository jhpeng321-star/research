"""MAIN15:56 final-only recovery; metadata-only census and one-use handover."""
import argparse
import copy
import json
from . import ops
from .storage import read,write
from .protocol import REGISTRY,checked,sha
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_finalendpoint4'


def census():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,time,hashlib
r=pathlib.Path({ROOT!r})
assert not (r/'FINAL_OUTCOMES_OPENED.json').exists()
names=('FINAL_ENDPOINT_DISPATCH.json','FINAL_ENDPOINT_SPAWN_HOLD.json','FINAL_PRECOMPUTE_COMPLETE.json','FINAL_ENDPOINT_RECOVERY_DISPATCH_V2.json')
receipts={{n:json.loads((r/n).read_text()) for n in names if (r/n).exists()}}
d=receipts['FINAL_ENDPOINT_DISPATCH.json'];p=pathlib.Path('/proc',str(d['pid']))
st=(p/'stat').read_text().split();args=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip() and st[21]==d['start_ticks']
children=[]
for child in (p/'task'/str(d['pid'])/'children').read_text().split():
 q=pathlib.Path('/proc',child);children.append(dict(pid=int(child),argv=[x.decode() for x in (q/'cmdline').read_bytes().split(bytes([0])) if x]))
ledger=[json.loads(x) for x in (r/'budget.jsonl').read_text().splitlines()] if (r/'budget.jsonl').exists() else None
from collections import Counter
phase=Counter()
if ledger is None:
 import sys;sys.path.insert(0,str(pathlib.Path(d['source'])/'pytools'))
 from tools.coopmwm.accel_storage import ledger as load
 ledger=load(r)
for x in ledger:phase[x['phase']]+=x['sim_s']
binding=json.loads((r/'VERIFICATION_CONTINUATION_BINDING.json').read_text())
print(json.dumps(dict(unix=time.time(),receipts=receipts,state=st[2],argv=args,children=children,phase_charges=dict(phase),budget=binding['budget'],binding_sha256=hashlib.sha256((r/'VERIFICATION_CONTINUATION_BINDING.json').read_bytes()).hexdigest(),outcomes_closed=True)))''',timeout=45)
    v=json.loads(out);write(ops.EVIDENCE/f'FINAL_RECOVERY_CENSUS_{int(v["unix"])}.json',v);print(out)


def register():
    target=ops.EVIDENCE/'FINAL_RECOVERY_MAIN1556_PLAN.json';assert not target.exists(),'adopt existing registration'
    b=copy.deepcopy(read(ops.EVIDENCE/'VERIFICATION_U4_REPAIR_PLAN.json')['budget'])
    b['phases']['exact_plan_grading']-=375800
    b['phases']['final_test']+=115200
    b['phases']['paired_branch']+=260600
    assert b['phases']['exact_plan_grading']==55360
    assert sum(b['phases'].values())==b['allocated']==b['total']==9500000
    v=dict(authority='MAIN ADDENDUM2026-11-02 15:56; approved375800 final-only allocation from freed B25-48 training/grading',budget=b,
        funding='remaining separate exact-plan-grading allocation431160->55360 (freed extended-B grading, now verification-grade reuse); final_test576000->691200; paired_branch1036800->1297400; train2840400/discovery150200/verification2800000 unchanged; all prior charges retained; no double-counting already-released B25-48 training',
        quarantine='386 invalid A24-tagged completed jobs/375800sim-s retained and never reused; original673opaque MAPPO/native/A48/reuse jobs adopted',
        execution='fresh A24v2 identities; serialized checkpoint_update and checkpoint SHA; fixed primaryA24/B24/R24; secondaryA48; no outcome opening beforeB24 AND R24',
        accounting='final-only explicit approved budget override required: source3 otherwise reads older immutable training binding and its old final phase caps; tiny source4 adapter requalification before handover; training binding and all training/GPU owners retained',
        handover='verify held625637 boot/start/argv, no physics children, no outcome opening; terminate held parent only, never resume its old code; one new detached owner; one-use config/dispatch/log; adopt any completed dispatch after transport error')
    write(target,v);h=sha(target)
    line='\nS-COOPMWM MAIN1556 APPROVED FINAL RECOVERY REGISTERED planSHA'+h+': '+json.dumps(v,sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'FINAL_RECOVERY_MAIN1556_REGISTRATION.json',dict(amendment_sha256=h,line=line))
    print(json.dumps(dict(registered=True,amendment_sha256=h,budget=b,new_sumo_sim_s=0)))


def cpu():
    prior=read(ops.EVIDENCE/'FINAL_ENDPOINT_CPU_V3.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['source_root']!r});files={{}}
for n,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;files[n.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=45))
    for n in ('final_endpoint','warm_protocol','test_final_endpoint_cpu'):
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
    target=ops.EVIDENCE/'FINAL_RECOVERY_CPU_V4.json';assert not target.exists();write(target,v)
    assert v['pass_'] and v['skips']==v['sim_s']==0


def replace():
    cpu=read(ops.EVIDENCE/'FINAL_RECOVERY_CPU_V4.json')
    reg=read(ops.EVIDENCE/'FINAL_RECOVERY_MAIN1556_REGISTRATION.json')
    plan=read(ops.EVIDENCE/'FINAL_RECOVERY_MAIN1556_PLAN.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,os,sys,signal,time,subprocess,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.accel_storage import ledger
from tools.coopmwm.final_endpoint import configure
from tools.coopmwm import final_runtime as f
cpu=json.loads({json.dumps(cpu)!r});v=json.loads({json.dumps(plan)!r});reg=json.loads({json.dumps(reg)!r})
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h
assert not (r/'FINAL_ENDPOINT_RECOVERY_DISPATCH_V2.json').exists(),'adopt existing owner'
assert not (r/'FINAL_ENDPOINT_HANDOVER_V2.json').exists(),'adopt interrupted handover; do not repeat signals'
assert not (r/'FINAL_OUTCOMES_OPENED.json').exists()
assert json.loads((r/'FINAL_PRECOMPUTE_COMPLETE.json').read_text())['completed_jobs']==673
assert json.loads((r/'FINAL_MAPPO_REUSE_PASS.json').read_text())['pass_']
d=json.loads((r/'FINAL_ENDPOINT_DISPATCH.json').read_text());assert d['pid']==625637 and d['start_ticks']=='61917851'
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert d['boot_id']==boot
p=pathlib.Path('/proc',str(d['pid']));st=(p/'stat').read_text().split();args=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
assert st[21]==d['start_ticks'] and st[2]=='T' and args[-3:]==['tools.coopmwm.final_endpoint',str(r),str(r/'FINAL_ENDPOINT_CONFIG.json')]
charges=ledger(r);invalid=[x for x in charges if x['id'].startswith(('fixedfinal_A24_','fixedbranch_A24_'))]
assert len(invalid)==386 and sum(x['sim_s'] for x in invalid)==375800
for x in invalid:
 q=r/'jobs'/x['id'];assert (q/'RESULT.json').is_file() and (q/'CLOSE.json').is_file() and (r/'final_receipts'/(x['id']+'.json')).is_file()
children=[]
for child in (p/'task'/str(d['pid'])/'children').read_text().split():
 cp=pathlib.Path('/proc',child);cs=(cp/'stat').read_text().split();ca=[x.decode() for x in (cp/'cmdline').read_bytes().split(bytes([0])) if x]
 assert ca[0]==args[0] and ca[1:3]==['-B','-c']
 tracker=len(ca)==4 and ca[3].startswith('from multiprocessing.resource_tracker import main;main(')
 pool=len(ca)==5 and ca[3].startswith('from multiprocessing.spawn import spawn_main; spawn_main(') and ca[4]=='--multiprocessing-fork'
 assert tracker or pool
 assert not (cp/'task'/child/'children').read_text().strip(),'active physics descendant; no handover'
 children.append(dict(pid=int(child),start_ticks=cs[21],argv=ca,role='tracker' if tracker else 'closed-job idle pool'))
config=json.loads((r/'FINAL_ENDPOINT_CONFIG.json').read_text())
config.update(source_root=str(s),source_hashes=cpu['source_hashes'],amendment_sha256=reg['amendment_sha256'],budget=v['budget'],receipt_suffix='_V2')
with (r/'AUTHORITY.md').open('a') as h:h.write(reg['line']+'\\nFinal-only accounting adapter:89CPU PASS/0skips/zero physics; old held pool has386closed receipts and no physics descendants; terminate only verified idle pool PIDs and SIGKILL held parent without resuming stale code.\\n')
configure(r,config)
new=f.specs(('A',))+f.specs(('A',),branches=True)
assert len(new)==480 and sum(x['sim_s'] for x in new)==460800
assert all(x['checkpoint_update']==24 and '_A24v2_' in x['id'] for x in new)
existing={{x['id'] for x in charges}}
assert all(x['id'] not in existing and not (r/'jobs'/x['id']).exists() and not (f.RAM/x['id']).exists() for x in new)
for phase in ('final_test','paired_branch'):
 spent=sum(x['sim_s'] for x in charges if x['phase']==phase)
 assert spent+sum(x['sim_s'] for x in new if x['phase']==phase)<=v['budget']['phases'][phase]
assert sum(x['sim_s'] for x in charges if x['phase']=='exact_plan_grading')<=55360
immutable(r/'FINAL_RECOVERY_ACCOUNTING_BINDING.json',dict(amendment_sha256=reg['amendment_sha256'],budget=v['budget'],qualification_sha256={sha(ops.EVIDENCE/'FINAL_RECOVERY_CPU_V4.json')!r},scope='final-only coordinator; original training bindings retained'))
immutable(r/'FINAL_ENDPOINT_CONFIG_V2.json',config)
immutable(r/'FINAL_ENDPOINT_HANDOVER_V2.json',dict(unix=time.time(),owner=d,children=children,quarantined_jobs=386,quarantined_sim_s=375800,signals='SIGTERM verified idle pool; SIGKILL held parent; never SIGCONT',outcomes_closed=True))
for c in children:
 if c['role']=='tracker':continue
 cp=pathlib.Path('/proc',str(c['pid']));assert (cp/'stat').read_text().split()[21]==c['start_ticks']
 os.kill(c['pid'],signal.SIGTERM)
assert (p/'stat').read_text().split()[21]==d['start_ticks']
os.kill(d['pid'],signal.SIGKILL)
for _ in range(100):
 if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
 time.sleep(.1)
else:raise ValueError('old final parent not exited; no new dispatch')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime');env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
with (r/'FINAL_ENDPOINT_V2.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.final_endpoint',str(r),str(r/'FINAL_ENDPOINT_CONFIG_V2.json')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(proc.pid),'stat').read_text().split()
dispatch=dict(pid=proc.pid,start_ticks=st[21],boot_id=boot,root=str(r),source=str(s),unix=time.time(),outcomes_closed=True,amendment_sha256=reg['amendment_sha256'],cpu_tests=cpu['tests'])
immutable(r/'FINAL_ENDPOINT_RECOVERY_DISPATCH_V2.json',dispatch)
print(json.dumps(dispatch))''',timeout=50)
    write(ops.EVIDENCE/'FINAL_ENDPOINT_RECOVERY_DISPATCH_V2.json',json.loads(out));print(out)


def status():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,time,hashlib
r=pathlib.Path({ROOT!r});name='FINAL_ENDPOINT_RECOVERY_DISPATCH_V2.json'
d=json.loads((r/name).read_text());p=pathlib.Path('/proc',str(d['pid']))
st=(p/'stat').read_text().split() if p.exists() else None
assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if st:assert st[21]==d['start_ticks']
names=('FINAL_ENDPOINT_STARTED_V2.json','FINAL_ENDPOINT_FAILED_V2.json','FINAL_OUTCOMES_OPENED.json','FINAL_REGISTERED_EVALUATION_COMPLETE.json')
receipts={{n:json.loads((r/n).read_text()) for n in names if (r/n).exists()}}
paths=sorted(r.glob('FINAL_PRECOMPUTE_STATUS_*.json'));latest=json.loads(paths[-1].read_text()) if paths else None
metas=list((r/'final_receipts').glob('fixedfinal_A24v2_*.json'))+list((r/'final_receipts').glob('fixedbranch_A24v2_*.json'))
sample=json.loads(metas[0].read_text()) if metas else None
if sample:assert sample['checkpoint_update']==24 and sample['checkpoint_sha256']==hashlib.sha256((r/'A_checkpoint_024.pt').read_bytes()).hexdigest()
print(json.dumps(dict(unix=time.time(),owner=d,state=st[2] if st else 'exited',receipts=receipts,latest=latest,A24v2_completed=len(metas),sample_checkpoint_pass=sample is not None,outcomes_closed=not (r/'FINAL_OUTCOMES_OPENED.json').exists())))''',timeout=40)
    v=json.loads(out);write(ops.EVIDENCE/f'FINAL_RECOVERY_STATUS_{int(v["unix"])}.json',v);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('census','register','cpu','replace','status'))
    globals()[p.parse_args().mode]()
