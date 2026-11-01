"""Explicit immutable bind/launch/status operations; no implicit simulation."""
import argparse
import json
from pathlib import Path
from . import ops
from .protocol import authority,budget,digest,sha,worlds,REGISTRY,checked
from .storage import read,write

ROOT='/home/ubuntu/coopmwm20261101_v2'
GPU_ROOT='/root/coopmwm20261101_v2'
EVIDENCE=ops.EVIDENCE


def bind(cpu_name):
    cpu=read(EVIDENCE/cpu_name)
    assert cpu['pass_'] and not cpu['skips'] and cpu['sim_s']==0
    staged=json.loads(ops.stage_frozen('cpuQ',ROOT,cpu))
    assert staged['hashes']==cpu['source_hashes'],'CPU-tested source/design binding required'
    text=checked(REGISTRY).read_text(encoding='utf-8');auth=authority(text)
    assert 'S-COOPMWM 17:12 EXPLORATION AMENDMENT REGISTERED' in text
    assert 'S-COOPMWM 17:27 BUCKET TRUST AMENDMENT REGISTERED' in text
    hosts=[read(EVIDENCE/f'FINAL_EXPOSURE_{h}.json') for h in ('cpuQ','gpuA','gpuR')]
    protocols=read(EVIDENCE/'FINAL_PROTOCOL_AUDIT.json')
    assert all(h['pass_'] for h in hosts) and protocols['pass_']
    cache=read(ops.LOCAL/'docs/paper_results/coopnative/MAPPO_DEV.json')
    assert len(cache['records'])==24 and cache['new_sim_s']==0
    exposure=dict(pass_=True,complete_all_hosts=True,cpuP='MAIN 17:07 excluded/unreachable',
        admitted_hosts=hosts,protocols=protocols)
    binding=dict(runtime='/home/ubuntu/e3final',checkpoint='/home/ubuntu/mappospeed/production2/policy_00169.pt',
        authority=auth,budget=budget(),cpu=cpu,exposure=exposure,workers=4,load_max=20,
        final_panel_sha256=digest(worlds('final_test')),hashes=staged['hashes'])
    out=ops.ssh.remote('cpuQ',f'''import os,sys,pathlib,json,hashlib
root=pathlib.Path({ROOT!r});assert root.is_dir()
os.nice(19);os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(root/'pytools'),'/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from tools.coopmwm.storage import write,read
from tools.leverprobe.backend import setup
binding=json.loads({json.dumps(binding)!r})
setup(binding['runtime'],binding['checkpoint'])
for name,value in [('AUTHORITY.md', {ascii(text)}),('MAPPO_DEV.json',json.loads({json.dumps(cache)!r}))]:
 p=root/name
 if p.exists():assert (p.read_text()==value if isinstance(value,str) else read(p)==value)
 elif isinstance(value,str):
  with p.open('x') as stream:stream.write(value)
 else:write(p,value)
 binding['hashes'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
binding['hashes']['../e3final/runtime/SOURCE_MANIFEST.json']=hashlib.sha256(pathlib.Path('/home/ubuntu/e3final/runtime/SOURCE_MANIFEST.json').read_bytes()).hexdigest()
binding['hashes']['../mappospeed/production2/policy_00169.pt']=hashlib.sha256(pathlib.Path(binding['checkpoint']).read_bytes()).hexdigest()
if (root/'BINDING.json').exists():assert read(root/'BINDING.json')==binding
else:write(root/'BINDING.json',binding)
from tools.coopmwm.runner import verify
verify(root)
print(json.dumps(dict(bound=True,root=str(root),workers=binding['workers'],binding_sha256=hashlib.sha256((root/'BINDING.json').read_bytes()).hexdigest(),sim_s=0)))
''',timeout=60)
    result=json.loads(out);write(EVIDENCE/'RUNTIME_BINDING.json',result);print(out)


def bind_gpu(cpu_name):
    cpu=read(EVIDENCE/cpu_name)
    staged=json.loads(ops.stage_frozen('gpuA',GPU_ROOT,cpu))
    assert staged['hashes']==cpu['source_hashes']
    out=ops.ssh.remote('gpuA',f'''import pathlib,sys,json,hashlib
root=pathlib.Path({GPU_ROOT!r});assert root.is_dir()
sys.path.insert(0,str(root/'pytools'))
from tools.coopmwm.storage import write,read
identity=dict(hashes=json.loads({json.dumps(staged['hashes'])!r}),backend='FULL-v4',source='22cc32f6',host='gpuA',memory_cap_GiB=8)
if (root/'GPU_BINDING.json').exists():assert read(root/'GPU_BINDING.json')==identity
else:write(root/'GPU_BINDING.json',identity)
print(json.dumps(dict(bound=True,root=str(root),binding_sha256=hashlib.sha256((root/'GPU_BINDING.json').read_bytes()).hexdigest(),sim_s=0)))
''',timeout=60)
    result=json.loads(out);write(EVIDENCE/'GPU_BINDING.json',result);print(out)


def launch(smoke=False):
    name='SMOKE_DISPATCH.json' if smoke else 'TRAIN_DISPATCH.json'
    out=ops.ssh.remote('cpuQ',f'''import os,sys,json,pathlib,subprocess
root=pathlib.Path({ROOT!r});assert root.is_dir()
sys.path[:0]=[str(root/'pytools'),'/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from tools.coopmwm.runner import verify
from tools.coopmwm.storage import read,write
verify(root)
assert not (root/{name!r}).exists(),'one-use dispatch; inspect owned receipts'
if not {smoke!r}:assert read(root/'SMOKE.json')['pass_']
import fcntl
claim=(root/'SUPERVISOR.claim').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,PYTHONPATH=str(root/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',
 CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
command=['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.runner',str(root)]+(['--smoke-only'] if {smoke!r} else [])
log=(root/('SMOKE.log' if {smoke!r} else 'TRAIN.log')).open('x')
claim.close()
p=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
receipt=dict(host='cpuQ',root=str(root),pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],
 boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),smoke={smoke!r})
write(root/{name!r},receipt);print(json.dumps(receipt))
''',timeout=45)
    result=json.loads(out);write(EVIDENCE/name,result);print(out)


def status():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,os,time
r=pathlib.Path({ROOT!r});assert r.is_dir()
def read(p):return json.loads(p.read_text())
owners=[]
for name in ('SMOKE_DISPATCH.json','TRAIN_DISPATCH.json'):
 if not (r/name).exists():continue
 d=read(r/name);p=pathlib.Path('/proc')/str(d['pid']);alive=False
 if (p/'stat').exists():
  stat=(p/'stat').read_text().split();alive=stat[21]==d['start_ticks'] and stat[2]!='Z'
 owners.append(dict(d,alive=alive))
charges=[json.loads(line) for line in (r/'CHARGES.jsonl').read_text().splitlines()] if (r/'CHARGES.jsonl').exists() else []
finished=[c['id'] for c in charges if (r/'jobs'/c['id']/'RESULT.json').exists() and (r/'jobs'/c['id']/'CLOSE.json').exists()]
receipts={{p.name:read(p) for pattern in ('SMOKE.json','STOP.json','CREDIT_CHECK.json','S*_CHECK_*.json','S*_DEV_*.json','S*_UPDATE_*.json','*TEST_COMPLETE.json','COMPLETE.json') for p in r.glob(pattern)}}
pending=[p.name for p in (r/'requests').glob('*.json') if not (r/'responses'/p.name).exists()]
print(json.dumps(dict(unix=time.time(),owners=owners,charged_sim_s=sum(c['sim_s'] for c in charges),jobs=len(charges),finished=len(finished),pending_WM=pending,
 load=os.getloadavg(),receipts=receipts,log_tail=(r/('TRAIN.log' if (r/'TRAIN.log').exists() else 'SMOKE.log')).read_text()[-12000:] if owners else '')))
''',timeout=40)
    result=json.loads(out)
    write(EVIDENCE/f'STATUS_{int(result["unix"])}.json',result)
    for name,receipt in result['receipts'].items():
        p=EVIDENCE/'runtime'/name
        if p.exists():assert read(p)==receipt,'immutable receipt changed'
        else:write(p,receipt)
    summary={k:v for k,v in result.items() if k not in ('receipts',)}
    summary['receipts']=list(result['receipts'])
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('bind','bind-gpu','smoke','train','status'))
    p.add_argument('--cpu',default='CPU_coopmwm20261101_cpu14.json');a=p.parse_args()
    if a.mode=='bind':bind(a.cpu)
    elif a.mode=='bind-gpu':bind_gpu(a.cpu)
    elif a.mode=='status':status()
    else:launch(a.mode=='smoke')
