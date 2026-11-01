"""Owned safe-boundary adoption; original immutable runtime and charges survive."""
import argparse
import json
from . import ops,deploy
from .protocol import authority,budget,REGISTRY,checked,sha
from .storage import read,write

SOURCE_ROOT='/home/ubuntu/coopmwm20261101_v3'


def quiesce():
    out=ops.ssh.remote('cpuQ',f'''import os,signal,pathlib,json,time
r=pathlib.Path({deploy.ROOT!r});assert r.is_dir()
def read(p):return json.loads(p.read_text())
d=read(r/'TRAIN_DISPATCH.json');p=pathlib.Path('/proc')/str(d['pid'])
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
assert (p/'stat').read_text().split()[21]==d['start_ticks']
assert 'tools.coopmwm.runner' in (p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
assert not (r/'QUIESCE_1742.json').exists(),'one-use owned pause'
os.kill(d['pid'],signal.SIGSTOP)
try:
 charges=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
 incomplete=[x['id'] for x in charges if not all((r/'jobs'/x['id']/f).exists() for f in ('RESULT.json','CLOSE.json'))]
 assert not incomplete,'active/queued physics; resume old supervisor and retry later'
 children=[]
 for child in pathlib.Path('/proc').iterdir():
  if not child.name.isdigit():continue
  try:
   stat=(child/'stat').read_text().split()
   if int(stat[3])==d['pid']:
    cmd=(child/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
    assert 'resource_tracker' in cmd,'owned process-pool still active; retry later'
    children.append(dict(pid=int(child.name),cmd=cmd,start_ticks=stat[21]))
  except (FileNotFoundError,ProcessLookupError):pass
 value=dict(owner=d,children=children,jobs_closed=len(charges),charged_sim_s=sum(x['sim_s'] for x in charges),
  reason='17:42 amendment: adopt complete initial batch and E2 queue, prevent superseded seed3',unix=time.time())
 with (r/'QUIESCE_1742.json').open('x') as f:json.dump(value,f,sort_keys=True)
 os.kill(d['pid'],signal.SIGTERM);os.kill(d['pid'],signal.SIGCONT)
 time.sleep(1)
 assert not (p/'stat').exists() or (p/'stat').read_text().split()[2]=='Z'
 print(json.dumps(value))
except BaseException:
 if not (r/'QUIESCE_1742.json').exists():os.kill(d['pid'],signal.SIGCONT)
 raise
''',timeout=20)
    value=json.loads(out);write(ops.EVIDENCE/'QUIESCE_1742.json',value);print(out)


def bind(cpu_name):
    cpu=read(ops.EVIDENCE/cpu_name)
    assert cpu['pass_'] and not cpu['skips'] and cpu['sim_s']==0
    staged=json.loads(ops.stage_frozen('cpuQ',SOURCE_ROOT,cpu))
    assert staged['hashes']==cpu['source_hashes']
    text=checked(REGISTRY).read_text(encoding='utf-8')
    assert 'S-COOPMWM 17:42 RANDOM-GUIDE AMENDMENT REGISTERED' in text
    auth=authority(text)
    amendment=read(ops.EVIDENCE/'BUDGET_AMENDMENT_1742.json')
    out=ops.ssh.remote('cpuQ',f'''import sys,pathlib,json,hashlib
r=pathlib.Path({deploy.ROOT!r});s=pathlib.Path({SOURCE_ROOT!r});assert r.is_dir() and s.is_dir()
sys.path[:0]=[str(s/'pytools'),'/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from tools.coopmwm.storage import read,write,reconciled
assert (r/'QUIESCE_1742.json').exists()
old=read(r/'BINDING.json');b=dict(old)
b.update(cpu=json.loads({json.dumps(cpu)!r}),budget=json.loads({json.dumps(budget())!r}),
 authority=json.loads({json.dumps(auth)!r}),amendment=json.loads({json.dumps(amendment)!r}),
 parent_binding_sha256=hashlib.sha256((r/'BINDING.json').read_bytes()).hexdigest())
b['hashes']=dict(old['hashes'])
for name,h in json.loads({json.dumps(staged['hashes'])!r}).items():b['hashes']['../'+s.name+'/'+name]=h
with (r/'AUTHORITY_1742.md').open('x') as f:f.write({ascii(text)})
b['hashes']['AUTHORITY_1742.md']=hashlib.sha256((r/'AUTHORITY_1742.md').read_bytes()).hexdigest()
write(r/'BINDING_1742.json',b)
from tools.coopmwm.runner import verify
verify(r);charged=reconciled(r)
print(json.dumps(dict(bound=True,root=str(r),source_root=str(s),charged_sim_s=charged,
 binding_sha256=hashlib.sha256((r/'BINDING_1742.json').read_bytes()).hexdigest(),new_sumo_sim_s=0)))
''',timeout=45)
    value=json.loads(out);write(ops.EVIDENCE/'RUNTIME_BINDING_1742.json',value);print(out)


def launch():
    out=ops.ssh.remote('cpuQ',f'''import os,sys,pathlib,json,subprocess,fcntl
r=pathlib.Path({deploy.ROOT!r});s=pathlib.Path({SOURCE_ROOT!r});assert r.is_dir() and s.is_dir()
sys.path[:0]=[str(s/'pytools'),'/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from tools.coopmwm.runner import verify
from tools.coopmwm.storage import read,write,reconciled
verify(r);reconciled(r)
assert not (r/'TRAIN_DISPATCH_1742.json').exists(),'one-use amended dispatch'
old=read(r/'TRAIN_DISPATCH.json');p=pathlib.Path('/proc')/str(old['pid'])
assert not (p/'stat').exists() or (p/'stat').read_text().split()[2]=='Z'
claim=(r/'SUPERVISOR.claim').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',
 CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
log=(r/'TRAIN_1742.log').open('x');claim.close()
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.runner',str(r)],
 cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
d=dict(host='cpuQ',root=str(r),source_root=str(s),pid=p.pid,
 start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],
 boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),amendment='17:42')
write(r/'TRAIN_DISPATCH_1742.json',d);print(json.dumps(d))
''',timeout=30)
    value=json.loads(out);write(ops.EVIDENCE/'TRAIN_DISPATCH_1742.json',value);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('quiesce','bind','launch'))
    p.add_argument('--cpu',default='CPU_coopmwm20261101_cpu16.json');a=p.parse_args()
    if a.mode=='quiesce':quiesce()
    elif a.mode=='bind':bind(a.cpu)
    else:launch()
