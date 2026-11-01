"""Bind the qualified replay guard to the still-waiting final discovery hook."""
import argparse
import json
from . import ops
from .protocol import REGISTRY,checked,sha
from .storage import read,write
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_finaldiscovery2'
FIX='/home/ubuntu/coopmwm20261102_diagrepaircpu1'


def prepare():
    original=read(ops.EVIDENCE/'FINAL_DISCOVERY_CPU.json')
    qualified=read(ops.EVIDENCE/'CPU_coopmwm20261102_diagrepaircpu1.json')
    baseline=read(ops.EVIDENCE/'CPU_coopmwm20261102_quotacpu4.json')
    name='pytools/tools/coopmwm/acceleration.py'
    assert original['source_hashes'][name]==baseline['source_hashes'][name],'final source differs; do not assume shared qualification'
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({original['source_root']!r});files={{}}
for name,h in json.loads({json.dumps(original['source_hashes'])!r}).items():
 p=r/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;files[name.removeprefix('pytools/')]=p.read_text()
q=pathlib.Path({FIX!r})
for name in ('acceleration','diagnostic_repair','test_diagnostic_repair_cpu'):
 path='pytools/tools/coopmwm/'+name+'.py';p=q/path
 assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads({json.dumps(qualified['source_hashes'])!r})[path]
 files[path.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=45))
    staged=json.loads(ops.stage('cpuQ',SOURCE,files))
    out=ops.ssh.remote('cpuQ',f'''import os,sys,time,json,unittest
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',COOPMWM_COST_SOURCE={SOURCE!r}+'/FROZEN_COSTS.json',COOPMWM_MOTION_SOURCE={SOURCE!r}+'/FROZEN_MOTION_SOURCE.py')
import torch;torch.set_num_threads(1)
from unittest.mock import patch
start=time.monotonic();names=('test_final_discovery_cpu','test_final_runtime_cpu','test_branch_eval_cpu','test_acceleration_cpu','test_committed_cpu','test_exact_grade_cpu','test_diagnostic_repair_cpu')
with patch('subprocess.Popen',side_effect=AssertionError('no SUMO in final replay qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=1).run(unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in names))
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,wall_s=time.monotonic()-start)))''',timeout=90)
    print(out);qual=json.loads(out.strip().splitlines()[-1]);write(ops.EVIDENCE/'FINAL_REPLAY_CPU.json',dict(qual,source_root=SOURCE,source_hashes=staged['hashes']))
    assert qual['pass_'] and qual['skips']==qual['sim_s']==0
    v=dict(source_root=SOURCE,source_hashes=staged['hashes'],qualification=qual,
        numerical_change='ONLY qualified acceleration.requests post-branch recorded-replay departed-ID guard; normal control/common-roster exact parity retained',
        preserved='unchanged final_discovery and final_runtime sources, detectors, panel, opening/endpoints, sampling, n+2 cost/reserve and no-retry rule; no precompute/analysis change',
        owner_handover='replace only verified waiting final-discovery owner before FINAL_DISCOVERY_STARTED; original receipts preserved; no outcome reads or new physics/allocation',
        repair_amendment=read(ops.EVIDENCE/'DIAGNOSTIC_REPAIR_REGISTRATION.json')['amendment_sha256'])
    path=ops.EVIDENCE/'FINAL_REPLAY_AMENDMENT.json';write(path,v);h=sha(path)
    line='\nS-COOPMWM FINAL DIAGNOSTIC DEPARTED-ROSTER GUARD REGISTERED amendmentSHA'+h+': '+json.dumps({k:x for k,x in v.items() if k not in ('source_hashes','qualification')},sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
assert not (r/'FINAL_DISCOVERY_STARTED.json').exists(),'final physics already started'
old=json.loads((r/'FINAL_DISCOVERY_CONFIG.json').read_text())
for name,h in json.loads({json.dumps(staged['hashes'])!r}).items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==h
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
old.update(source_root=str(s),source_hashes=json.loads({json.dumps(staged['hashes'])!r}),qualification=json.loads({json.dumps(qual)!r}),amendment_sha256={h!r})
immutable(r/'FINAL_DISCOVERY_CONFIG_V2.json',old)
print(json.dumps(dict(registered=True,amendment_sha256={h!r},new_sumo_sim_s=0,outcomes_opened=False)))''',timeout=40)
    write(ops.EVIDENCE/'FINAL_REPLAY_REGISTRATION.json',dict(v,amendment_sha256=h,remote=json.loads(out)));print(out)


def replace():
    reg=read(ops.EVIDENCE/'FINAL_REPLAY_REGISTRATION.json')
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,subprocess,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,{FIX!r}+'/pytools')
from tools.coopmwm.quota_transition import terminate
from tools.coopmwm.warm_spool import immutable
assert {reg['amendment_sha256']!r} in (r/'AUTHORITY.md').read_text()
assert not (r/'FINAL_DISCOVERY_STARTED.json').exists() and not (r/'B_checkpoint_048.pt').exists(),'must still be waiting'
assert not (r/'FINAL_DISCOVERY_DISPATCH_V2.json').exists(),'adopt new owner'
old=json.loads((r/'FINAL_DISCOVERY_DISPATCH.json').read_text());terminate(old,'tools.coopmwm.final_discovery')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime');env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
with (r/'FINAL_DISCOVERY_V2.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.final_discovery',str(r),str(r/'FINAL_DISCOVERY_CONFIG_V2.json')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()
v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),source=str(s),unix=time.time(),amendment_sha256={reg['amendment_sha256']!r})
immutable(r/'FINAL_DISCOVERY_DISPATCH_V2.json',v);print(json.dumps(v))''',timeout=45)
    v=json.loads(out);write(ops.EVIDENCE/'FINAL_DISCOVERY_DISPATCH_V2.json',v);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','replace'));a=p.parse_args();globals()[a.mode]()
