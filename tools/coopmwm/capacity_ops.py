"""Source-bound MAIN05:08 admission repair, explicit one-use broker handover."""
import argparse
import json
import time
from pathlib import Path
from . import ops
from .storage import read,write
from .protocol import sha,checked,REGISTRY
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_capacitycpu1'


def cpu():
    prior=read(ops.EVIDENCE/'CPU_coopmwm20261102_exactcpu5.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['root']!r});files={{}}
for name,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 files[name.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=40))
    for name in ('capacity_broker.py','capacity_ops.py','test_capacity_broker_cpu.py','plan_client.py'):
        files['tools/coopmwm/'+name]=(ops.LOCAL/'tools/coopmwm'/name).read_text(encoding='utf-8')
    staged=json.loads(ops.stage('cpuQ',SOURCE,files))
    out=ops.ssh.remote('cpuQ',f'''import sys,unittest,os,json,time
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import torch;torch.set_num_threads(1)
from unittest.mock import patch
start=time.monotonic()
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in
 ('test_capacity_broker_cpu','test_warm_broker_cpu','test_warm_transport_cpu','test_volume_cpu'))
with patch('subprocess.Popen',side_effect=AssertionError('no physics during qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,root={SOURCE!r},wall_s=time.monotonic()-start)))''',timeout=70)
    print(out);v=json.loads(out.strip().splitlines()[-1]);v['source_hashes']=staged['hashes']
    write(ops.EVIDENCE/'CPU_coopmwm20261102_capacitycpu1.json',v)
    assert v['pass_'] and not v['skips']


def register():
    target=ops.EVIDENCE/'MAIN0508_CAPACITY_REGISTRATION.json';assert not target.exists()
    c=read(ops.EVIDENCE/'CPU_coopmwm20261102_capacitycpu1.json');assert c['pass_'] and c['sim_s']==c['skips']==0
    value=dict(cpu=c,cap=7500000,allocation=7473440,new_sumo_sim_s=0,
        reason='observed first8 episodes blocked by queue before Nexec10;16reserved jobs unclaimed;R waits final matching quota',
        live='replace only verified transport broker; adopt assigned durable GPU requests; immediately decline unassigned excess capacity with explicit zero-compute admission receipt; expose actual4 slots to legacy clients',
        future='nonblocking client queries only with capacity observed free, otherwise own action/p0; broker resolves races; B workers24 at next safe episode-supervisor boundary; existing wave retained',
        R='keep exact count/regime/size/duration matching, never fabricate final B quota; centre requests may use spare slots at lower priority once quota exists; R not described as GPU-free',
        unchanged='K4H300/K8Hfinal,MPPI/population,weights,warm memory,Nexec10,exact-only grades,health guards,endpoints48/48/24,budget,original deadline; no inference/physics replay or outcome tuning',
        legacy_logging='historical Bernoulli query p unchanged in existing closures; additional query_admission gives deterministic WM-capacity gate; capacity declines are not WM evaluations')
    path=ops.EVIDENCE/'MAIN0508_CAPACITY_AMENDMENT.json';write(path,value);h=sha(path)
    line='\nS-COOPMWM MAIN0508 NONBLOCKING CAPACITY ENGINEERING REGISTERED amendmentSHA'+h+': '+json.dumps({k:v for k,v in value.items() if k!='cpu'},sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    remote=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
for name,h in json.loads({json.dumps(c['source_hashes'])!r}).items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==h
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
b=json.loads((r/'WARM_REMOTE_BINDING.json').read_text())
v=dict(amendment_sha256={h!r},source_root=str(s),deadline_unix=b['deadline_unix'])
immutable(r/'CAPACITY_BROKER_CONFIG.json',v)
print(json.dumps(v))''',timeout=40))
    write(target,dict(value,amendment_sha256=h,config=remote));print(json.dumps(dict(registered=h,cpu_tests=c['tests'])))


def handover():
    reg=read(ops.EVIDENCE/'MAIN0508_CAPACITY_REGISTRATION.json')
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,os,signal,time,subprocess,sys,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running
assert_running(r,'B')
assert not (r/'CAPACITY_BROKER_DISPATCH.json').exists(),'adopt existing handover'
for name,h in json.loads({json.dumps(reg['cpu']['source_hashes'])!r}).items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==h
config=json.loads((r/'CAPACITY_BROKER_CONFIG.json').read_text());assert config['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
d=json.loads((r/'WARM_REMOTE_BROKER_DISPATCH.json').read_text())
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
p=pathlib.Path('/proc')/str(d['pid']);st=(p/'stat').read_text().split()
assert st[21]==d['start_ticks'] and b'tools.coopmwm.warm_broker' in (p/'cmdline').read_bytes()
os.kill(d['pid'],signal.SIGSTOP)
snapshot={{q.name:json.loads(q.read_text()) for q in (r/'broker_assignments').glob('*.json') if not (r/'responses'/q.name).exists()}}
immutable(r/'CAPACITY_HANDOVER_SNAPSHOT.json',dict(old_owner=d,pending=snapshot,unix=time.time(),new_sumo_sim_s=0))
# Terminate ONLY the verified CPU transport owner; GPU processes and SUMO are untouched.
os.kill(d['pid'],signal.SIGTERM);os.kill(d['pid'],signal.SIGCONT)
for _ in range(100):
 if not (p/'stat').exists() or (p/'stat').read_text().split()[2]=='Z':break
 time.sleep(.1)
else:raise ValueError('old broker did not exit; do not duplicate')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
with (r/'CAPACITY_BROKER.log').open('x') as log:
 child=subprocess.Popen([{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.capacity_broker',str(r),str(r/'CAPACITY_BROKER_CONFIG.json')],
  env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(child.pid),'stat').read_text().split()
v=dict(pid=child.pid,start_ticks=st[21],boot_id=d['boot_id'],source=str(s),unix=time.time(),old_owner=d,adopted_requests=list(snapshot))
immutable(r/'CAPACITY_BROKER_DISPATCH.json',v);print(json.dumps(v))''',timeout=50))
    write(ops.EVIDENCE/'CAPACITY_BROKER_DISPATCH.json',value);print(json.dumps(value))


def status():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time,os
r=pathlib.Path({ROOT!r});p=r/'CAPACITY_BROKER_DISPATCH.json';d=json.loads(p.read_text()) if p.exists() else None
st=None
if d:
 assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
 p=pathlib.Path('/proc',str(d['pid']),'stat');st=p.read_text().split() if p.exists() else None
 assert not st or st[21]==d['start_ticks']
print(json.dumps(dict(unix=time.time(),owner=d,state=st[2] if st else 'exited',declines=len(list((r/'capacity_declines').glob('*.json'))),
 admissions=[json.loads(p.read_text()) for p in (r/'capacity_admissions').glob('*.json')],
 log=(r/'CAPACITY_BROKER.log').read_text()[-4000:] if (r/'CAPACITY_BROKER.log').exists() else None,
 stop=json.loads((r/'WARM_GUIDED_STOP.json').read_text()) if (r/'WARM_GUIDED_STOP.json').exists() else None)))''',timeout=40))
    write(ops.EVIDENCE/f'CAPACITY_STATUS_{int(value["unix"])}.json',value);print(json.dumps(value));return value


def failure_status():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time
r=pathlib.Path({ROOT!r});owners=[]
for name in ('WARM_B_DISPATCH_V4.json','WARM_R_DISPATCH_V4.json','EXACT_GRADER_DISPATCH.json'):
 d=json.loads((r/name).read_text());p=pathlib.Path('/proc',str(d['pid']),'stat');st=p.read_text().split() if p.exists() else None
 assert not st or st[21]==d['start_ticks'];owners.append(dict(name=name,owner=d,state=st[2] if st else 'exited'))
jobs=[]
for p in (r/'jobs').glob('warm_u001_B_*'):
 c=p/'CLAIM.json';d=json.loads(c.read_text()) if c.exists() else None
 stat=pathlib.Path('/proc',str(d['pid']),'stat') if d else None
 jobs.append(dict(id=p.name,claimed=bool(d),closed=(p/'RESULT.json').exists(),pid=d['pid'] if d else None,
  state=stat.read_text().split()[2] if stat and stat.exists() else 'exited',failure=(p/'FAILED.json').read_text()[-1000:] if (p/'FAILED.json').exists() else None))
logs={{n:(r/n).read_text()[-6000:] for n in ('WARM_B_V4.log','WARM_R_V4.log','CAPACITY_BROKER.log') if (r/n).exists()}}
print(json.dumps(dict(unix=time.time(),owners=owners,jobs=jobs,logs=logs)))''',timeout=40))
    write(ops.EVIDENCE/f'CAPACITY_FAILURE_{int(value["unix"])}.json',value);print(json.dumps(value));return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('cpu','register','handover','status','failure_status'));a=p.parse_args();globals()[a.mode]()
