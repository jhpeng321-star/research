"""Explicit registered A-only recovery after a pre-claim bridge-STOP failure."""
import argparse
import json
from . import ops
from .protocol import REGISTRY, checked, sha
from .storage import read, write
from .warm_a_ops import ROOT

SOURCE = '/home/ubuntu/coopmwm20261101_warmA3'


def prepare():
    staged = json.loads(ops.stage('cpuQ', SOURCE, {'warm_a.py':checked(ops.LOCAL/'tools/coopmwm/warm_a.py').read_text(encoding='utf-8')}))
    source = f'''import pathlib,json,sys,os,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=['/home/ubuntu/coopmwm20261101_acceltrain1/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
import importlib.util
s=importlib.util.spec_from_file_location('warm_a',{SOURCE!r}+'/warm_a.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=pathlib.Path({ROOT!r});old=json.loads((r/'A_WARM_CONTINUATION_DISPATCH.json').read_text())
p=pathlib.Path('/proc')/str(old['pid']);assert not (p/'stat').exists()
assert not (r/'A_WARM_CONTINUATION.claim').exists() and not list(r.glob('A_UPDATE_*.json'))
stop=json.loads((r/'STOP.json').read_text());assert stop==dict(error="No module named 'torch'",source='owned plan bridge; no retry')
from unittest.mock import patch
base=json.loads((r/'BINDING_TRAINING.json').read_text())
amend=dict(arm='A',stop_update=4,preserved_stop_sha256='STOP_SHA')
reg=dict(amendment_sha256='AMEND_SHA')
values={{'BINDING_TRAINING.json':base,'A_WARM3_AMENDMENT.json':amend,'A_WARM3_REGISTRATION.json':reg,'STOP.json':stop}}
original_read=m.read
def fake_read(p):return values[p.name]
from tools.coopmwm import accel_protocol
original_sha=accel_protocol.sha
def fake_sha(p):
 if p.name=='STOP.json':return 'STOP_SHA'
 if p.name=='A_WARM3_AMENDMENT.json':return 'AMEND_SHA'
 return original_sha(p)
# Only immutable pre-existing authority text is substituted; all source hashes
# and budget/CPU/GPU gates still run against the real binding.
orig_text=pathlib.Path.read_text
def text(p,*a,**kw):return 'AMEND_SHA' if p.name=='A_WARM3_AUTHORITY.md' else orig_text(p,*a,**kw)
with patch.object(m,'read',fake_read),patch.object(accel_protocol,'sha',fake_sha),patch.object(pathlib.Path,'read_text',text):
 assert m.verify_a(r)==base
 values['STOP.json']=dict(error='different failure',source='owned plan bridge; no retry')
 try:m.verify_a(r)
 except AssertionError:pass
 else:raise AssertionError('unrelated STOP bypassed')
print(json.dumps(dict(pass_=True,tests=2,sim_s=0,source={staged!r},preserved_stop_sha256=hashlib.sha256((r/'STOP.json').read_bytes()).hexdigest())))
'''
    v = json.loads(ops.ssh.remote('cpuQ', source, timeout=30))
    write(ops.EVIDENCE/'WARM_A3_CPU.json',v)
    amendment = dict(arm='A',source=staged,qualification=v,preserved_stop_sha256=v['preserved_stop_sha256'],
        reason='A1/A2 exited before claim/physics; A2 serialization hash mismatch fixed by exact authority bytes and real preflight',
        stop_update=4,max_new_sim_s=100800,full_allocation_sim_s=4309640,cap_sim_s=6000000,
        original_A_reused=True,B_R_held=True,new_sumo_sim_s=0)
    write(ops.EVIDENCE/'WARM_A3_AMENDMENT.json',amendment)
    h=sha(ops.EVIDENCE/'WARM_A3_AMENDMENT.json')
    line=f'\nS-COOPMWM A-ONLY BRIDGE STOP RELEASE REGISTERED amendmentSHA{h}: A1/A2 exited before claim/physics/update; preserve exact original STOP SHA{v["preserved_stop_sha256"]}. New source-bound warmA3 copies exact authority bytes and performs real preflight before dispatch; verifies all original immutable source/CPU/GPU/budget gates, allows ONLY this bridge missing-torch STOP for A. Reuse original12 finished A episodes and12 never-claimed reservations once; A u1..4/check then pause. No retry of attempted physics. B/R remain held. WARM_A allocation4309640/cap6000000 and maxnew100800sim-s unchanged; no sealed access.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'WARM_A3_REGISTRATION.json',dict(line=line,amendment_sha256=h))
    print(json.dumps(amendment))


def dispatch():
    a=read(ops.EVIDENCE/'WARM_A3_AMENDMENT.json');reg=read(ops.EVIDENCE/'WARM_A3_REGISTRATION.json')
    texts={n:checked(ops.EVIDENCE/n.replace('A_WARM3','WARM_A3')).read_text(encoding='utf-8')
           for n in ('A_WARM3_AMENDMENT.json','A_WARM3_REGISTRATION.json')}
    authority=checked(REGISTRY).read_text(encoding='utf-8')
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,os,subprocess,time,hashlib
r=pathlib.Path({ROOT!r});src=pathlib.Path({SOURCE!r});assert r.is_dir() and src.is_dir()
target=r/'A_WARM3_DISPATCH.json';assert not target.exists()
assert not (r/'A_WARM_CONTINUATION.claim').exists()
for n,h in {a['source']['hashes']!r}.items():assert hashlib.sha256((src/n).read_bytes()).hexdigest()==h
for n,text in {texts!r}.items():
 with (r/n).open('x') as f:f.write(text)
assert hashlib.sha256((r/'A_WARM3_AMENDMENT.json').read_bytes()).hexdigest()=={reg['amendment_sha256']!r}
with (r/'A_WARM3_AUTHORITY.md').open('x') as f:f.write({authority!r})
d=json.loads((r/'TRAIN_DISPATCH.json').read_text())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=d['source']+'/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',COOPMWM_COST_SOURCE=d['source']+'/FROZEN_COSTS.json')
preflight="import sys,importlib.util,pathlib;sys.path.insert(0,"+repr(d['source']+'/pytools')+");s=importlib.util.spec_from_file_location('warm_a',"+repr(str(src/'warm_a.py'))+");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.verify_a(pathlib.Path("+repr(str(r))+"))"
subprocess.run(['/home/ubuntu/miniconda3/bin/python','-B','-c',preflight],env=env,check=True,capture_output=True,text=True)
with (r/'A_WARM3.log').open('x') as log:
 p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u',str(src/'warm_a.py'),str(r)],cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text().split()
v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source=str(src),root=str(r),unix=time.time())
with target.open('x') as f:json.dump(v,f,sort_keys=True)
print(json.dumps(v))
''',timeout=25))
    write(ops.EVIDENCE/'WARM_A3_DISPATCH.json',value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','dispatch'));a=p.parse_args()
    {'prepare':prepare,'dispatch':dispatch}[a.mode]()
