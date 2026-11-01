"""One-use, source-bound A-only continuation; frozen A implementation is reused."""
import argparse
import json
from . import ops
from .protocol import REGISTRY, checked, sha
from .storage import read, write

ROOT='/home/ubuntu/coopmwm20261101_accel1'
SOURCE='/home/ubuntu/coopmwm20261101_warmA1'


def prepare():
    code=checked(ops.LOCAL/'tools/coopmwm/warm_a.py').read_text(encoding='utf-8')
    staged=json.loads(ops.stage('cpuQ',SOURCE,{'warm_a.py':code}))
    out=ops.ssh.remote('cpuQ',f'''import pathlib,sys,os,json,importlib.util
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[{ROOT!r}+'/../coopmwm20261101_acceltrain1/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
spec=importlib.util.spec_from_file_location('warm_a',{SOURCE!r}+'/warm_a.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if True:
 r=pathlib.Path({SOURCE!r})/'adoption_cpu_fixture';key='u001_A_986000';p=r/'jobs'/key;p.mkdir(parents=True)
 assert m.adoptable(r,key,{{key:dict(sim_s=1200)}})=='unattempted'
 (p/'CLAIM.json').write_text('{{}}')
 try:m.adoptable(r,key,{{key:dict(sim_s=1200)}})
 except AssertionError:pass
 else:raise AssertionError('attempted physics admitted')
 (p/'RESULT.json').write_text('{{}}');(p/'CLOSE.json').write_text('{{}}')
 assert m.adoptable(r,key,{{}})=='complete'
print(json.dumps(dict(pass_=True,tests=3,sim_s=0,source={staged!r})))
''',timeout=30)
    v=json.loads(out);write(ops.EVIDENCE/'WARM_A_CPU.json',v)
    amendment=dict(owner='22:08 A unaffected may continue',source=staged,qualification=v,
        reuse='original A u0 checkpoint/Adam/RNG,12 completed u1 results +12 never-claimed charged reservations',
        max_new_sim_s=100800,training_new=86400,check_new=14400,stop_update=4,
        full_allocation_sim_s=4309640,cap_sim_s=6000000,
        accounting='original4280840 +28800 superseded cold B u1 reservations; A existing charges reused; all51reservations retained',
        B_R='held; no cold resume; fresh u0 after qualification',new_sumo_sim_s=0)
    write(ops.EVIDENCE/'WARM_A_AMENDMENT.json',amendment);h=sha(ops.EVIDENCE/'WARM_A_AMENDMENT.json')
    line=f'\nS-COOPMWM OWNER22:08 A-ONLY CONTINUATION REGISTERED amendmentSHA{h}: original A u0/Adam/RNG and12 completed u1 episodes reused;12 unclaimed A reservations adopted once without recharge; only A advances throughu4 then pause; newphysics maximum100800sim-s (86400train+14400check). Cold B/R stay held; all872840currentcharges retained. Full allocation4309640=4280840+28800supersededcoldB, cap6000000/free1690360; no sealed access.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'WARM_A_REGISTRATION.json',dict(line=line,amendment_sha256=h));print(json.dumps(amendment))


def dispatch():
    amendment=read(ops.EVIDENCE/'WARM_A_AMENDMENT.json');reg=read(ops.EVIDENCE/'WARM_A_REGISTRATION.json')
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,subprocess,os,hashlib,time
r=pathlib.Path({ROOT!r});source=pathlib.Path({SOURCE!r});assert r.is_dir() and source.is_dir()
target=r/'A_WARM_CONTINUATION_DISPATCH.json';assert not target.exists(),'one-use dispatch'
d=json.loads((r/'TRAIN_DISPATCH.json').read_text());s=(pathlib.Path('/proc')/str(d['pid'])/'stat').read_text().split()
assert s[21]==d['start_ticks'] and s[2]=='T'
for n,h in {amendment['source']['hashes']!r}.items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h
for n,v in [('A_WARM_AMENDMENT.json',{amendment!r}),('A_WARM_REGISTRATION.json',{reg!r})]:
 with (r/n).open('x') as f:json.dump(v,f,sort_keys=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=d['source']+'/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',COOPMWM_COST_SOURCE=d['source']+'/FROZEN_COSTS.json')
with (r/'A_WARM_CONTINUATION.log').open('x') as log:
 p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u',str(source/'warm_a.py'),str(r)],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text().split()
v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source=str(source),root=str(r),unix=time.time())
with target.open('x') as f:json.dump(v,f,sort_keys=True)
print(json.dumps(v))
''',timeout=25))
    write(ops.EVIDENCE/'WARM_A_DISPATCH.json',value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','dispatch'));a=p.parse_args()
    {'prepare':prepare,'dispatch':dispatch}[a.mode]()
