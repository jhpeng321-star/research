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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('census','register','cpu'))
    globals()[p.parse_args().mode]()
