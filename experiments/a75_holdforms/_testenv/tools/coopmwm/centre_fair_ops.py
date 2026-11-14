"""One-use compute-only broker handover; all durable inference claims adopted."""
import argparse
import json
from . import ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_centrefaircpu1'


def qualify():
    prior=read(ops.EVIDENCE/'CPU_coopmwm20261102_quotacpu4.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['root']!r});files={{}}
for name,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 files[name.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=40))
    for name in ('capacity_broker','centre_fair_ops','test_centre_fair_cpu'):
        files['tools/coopmwm/'+name+'.py']=checked(ops.LOCAL/'tools/coopmwm'/(name+'.py')).read_text(encoding='utf-8')
    staged=json.loads(ops.stage('cpuQ',SOURCE,files))
    out=ops.ssh.remote('cpuQ',f'''import sys,os,json,unittest,time
from unittest.mock import patch
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import torch;torch.set_num_threads(1)
names=['tools.coopmwm.'+n for n in ('test_centre_fair_cpu','test_capacity_broker_cpu','test_warm_broker_cpu','test_warm_transport_cpu','test_quota_admission_cpu')]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('no SUMO/inference during qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,root={SOURCE!r},wall_s=time.monotonic()-start)))''',timeout=60)
    print(out);v=json.loads(out.strip().splitlines()[-1]);v['source_hashes']=staged['hashes']
    write(ops.EVIDENCE/'CPU_coopmwm20261102_centrefaircpu1.json',v)
    assert v['pass_'] and v['sim_s']==v['skips']==0


def register():
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_centrefaircpu1.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    v=dict(source_root=SOURCE,cpu=cpu,new_sumo_sim_s=0,
        diagnosis='current progress1793574892: R21/24closed but three nuisance-centre queries wait3234..3293s; static plan-first broker ordering starves the registered R(u) pipeline behind B(u+1)',
        rule='per slot: existing assigned claims first ALWAYS; among NEW unassigned work, at most one centre helper between plan searches when both queues nonempty; stable original dense/time/state ordering within each queue; do not idle if either queue empty',
        invariants='same requests/stateIDs/CRN/models/MPPI populations/K4H300/K8Hfinal; R only K1H300 current-policy centre with zero candidate scoring/library; no horizons/arms/endpoints/quotas/trust/sampling/budget changes; original GPU deadline unchanged',
        handover='terminate only identity-verified owned QUOTA broker3514272; detached source-bound replacement takes exclusive existing broker lock and adopts every assigned forecast/claim once; never signal SUMO/GPU/training/grader/observers')
    target=ops.EVIDENCE/'CENTRE_FAIR_AMENDMENT.json';write(target,v);h=sha(target)
    line='\nS-COOPMWM MAIN0518 PIPELINED-R CENTRE STARVATION ENGINEERING REPAIR REGISTERED amendmentSHA'+h+': '+json.dumps({k:x for k,x in v.items() if k!='cpu'},sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib,sys
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running
assert_running(r,'B')
for name,h in json.loads({json.dumps(cpu['source_hashes'])!r}).items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==h
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
old=json.loads((r/'QUOTA_BROKER_CONFIG.json').read_text())
config=dict(old,source_root=str(s),source_hashes=json.loads({json.dumps(cpu['source_hashes'])!r}),amendment_sha256={h!r},receipt_tag='_CF1',pipeline_centre_fairness=True)
immutable(r/'CENTRE_FAIR_CONFIG.json',config)
print(json.dumps(dict(registered=True,amendment_sha256={h!r},deadline_unix=config['deadline_unix'],new_sumo_sim_s=0)))''',timeout=40)
    write(ops.EVIDENCE/'CENTRE_FAIR_REGISTRATION.json',dict(amendment_sha256=h,remote=json.loads(out)));print(out)


def deploy():
    reg=read(ops.EVIDENCE/'CENTRE_FAIR_REGISTRATION.json')
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,subprocess,time
r=pathlib.Path({ROOT!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running
from tools.coopmwm.quota_transition import terminate
assert_running(r,'B');assert {reg['amendment_sha256']!r} in (r/'AUTHORITY.md').read_text()
assert not (r/'CENTRE_FAIR_BROKER_DISPATCH.json').exists(),'adopt existing owner, never duplicate'
old=json.loads((r/'QUOTA_broker_DISPATCH.json').read_text());assert old['pid']==3514272 and old['start_ticks']=='58480523'
pending={{p.stem:json.loads(p.read_text()) for p in (r/'broker_assignments').glob('*.json') if not (r/('responses' if json.loads(p.read_text())['kind']=='plan' else 'centre_responses')/p.name).exists()}}
terminate(old,'tools.coopmwm.quota_runtime')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH={SOURCE!r}+'/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
with (r/'CENTRE_FAIR_BROKER.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.centre_fair_ops','serve'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()
v=dict(pid=p.pid,start_ticks=stat[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),source={SOURCE!r},unix=time.time(),component='broker')
immutable(r/'CENTRE_FAIR_BROKER_DISPATCH.json',v)
immutable(r/'CENTRE_FAIR_HANDOVER.json',dict(previous=old,current=v,assigned_adopted=pending,inference_replay=False,new_sumo_sim_s=0))
print(json.dumps(v))''',timeout=45)
    v=json.loads(out);write(ops.EVIDENCE/'CENTRE_FAIR_BROKER_DISPATCH.json',v);print(out)


def serve():
    import hashlib
    from pathlib import Path
    from .capacity_broker import run
    r=Path(ROOT);config=json.loads((r/'CENTRE_FAIR_CONFIG.json').read_text())
    for name,h in config['source_hashes'].items():assert hashlib.sha256((Path(config['source_root'])/name).read_bytes()).hexdigest()==h
    run(r,r/'CENTRE_FAIR_CONFIG.json')


def status():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,time
r=pathlib.Path({ROOT!r});d=json.loads((r/'CENTRE_FAIR_BROKER_DISPATCH.json').read_text())
assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
p=pathlib.Path('/proc',str(d['pid']),'stat');s=p.read_text().split() if p.exists() else None
assert not s or s[21]==d['start_ticks']
receipts={{p.name:json.loads(p.read_text()) for pattern in ('CENTRE_FAIR_HANDOVER.json','CAPACITY_BROKER_OWNER_CF1.json','WARM_GUIDED_STOP_V2.json') for p in r.glob(pattern)}}
centre=[]
for p in (r/'centre_requests').glob('*.json'):
 a=r/'broker_assignments'/p.name;c=r/'broker_completed'/p.name
 centre.append(dict(key=p.stem,age_s=time.time()-p.stat().st_mtime,assigned=json.loads(a.read_text()) if a.exists() else None,completed=json.loads(c.read_text()) if c.exists() else None,response=(r/'centre_responses'/p.name).exists()))
print(json.dumps(dict(unix=time.time(),owner=d,state=s[2] if s else 'exited',receipts=receipts,centre_requests=centre,log=(r/'CENTRE_FAIR_BROKER.log').read_text()[-3000:])))''',timeout=40)
    v=json.loads(out);write(ops.EVIDENCE/f'CENTRE_FAIR_STATUS_{int(v["unix"])}.json',v);print(out);return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('qualify','register','deploy','serve','status'))
    globals()[p.parse_args().mode]()
