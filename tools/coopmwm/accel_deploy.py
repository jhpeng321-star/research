"""Explicit acceleration staging, registered smoke, and identity-only status."""
import argparse
import json
from . import ops
from .storage import read,write
from .protocol import REGISTRY,checked,sha,authority
from .accel_protocol import budget
from .plans import SEARCH
from .plan_bridge import ROOT


def smoke_register(cpu_name):
    cpu=read(ops.EVIDENCE/cpu_name);assert cpu['pass_'] and not cpu['skips']
    parity={h:read(ops.EVIDENCE/f'PLAN_PROBE2_{h}_PARITY.json') for h in ('gpuA','gpuR')}
    assert all(v['pass_'] and v['max_cost_error']<=1e-3 for v in parity.values())
    receipt=dict(cpu=cpu_name,cpu_sha256=sha(ops.EVIDENCE/cpu_name),gpu_parity=parity,
        purpose='zero-update one1200s episode each A/B/R, no guidance or optimizer',
        total_sim_s=3600,budget=budget(),new_sumo_sim_s=0)
    write(ops.EVIDENCE/'ACCEL_SMOKE_AMENDMENT.json',receipt)
    h=sha(ops.EVIDENCE/'ACCEL_SMOKE_AMENDMENT.json')
    line=(f'\nS-COOPMWM ACCELERATION SMOKE REGISTERED BEFORE PHYSICS: sourceCPU{cpu_name}/SHA{receipt["cpu_sha256"]}, '
        f'actual MAPPO acceleration command CPU parityPASS; fixedfull1600/fourcandidate K8 GPU parityBOTH exact0veh-s, '
        f'large-action sensitivity1281.163veh-s. One1200s zero-update deterministic episode eachA/B/R, '
        f'precharge3600 in existing mechanics allocation, bitwiseMAPPOdevcache reused/noMAPPO rerun. '
        f'All optimizer updates and guided training stayHELD pending final throughput/numeric quota; '
        f'smoke-only binding enforces this. NoGPU inference added; GPU probes finish their original registered search. '
        f'Policy continuation decoder preserves nongroup residuals and returns to own continuation after commitment; '
        f'WM continuation is current residual held against dynamic MAPPO, disclosed approximation/cold memory, not future policy truth. '
        f'CPU fixtures verify background/postcommit decoder; duplicate CEM IDs isolated without adding variance floor. '
        f'Budgetallocated{budget()["allocated"]}/6000000 including811640oldcharges unchanged. AmendmentSHA{h}.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'ACCEL_SMOKE_REGISTRATION.json',dict(line=line,amendment_sha256=h));print(line.strip())


def bind_smoke(cpu_name):
    cpu=read(ops.EVIDENCE/cpu_name)
    reg=read(ops.EVIDENCE/'ACCEL_SMOKE_REGISTRATION.json')
    assert read(ops.EVIDENCE/'ACCEL_SMOKE_AMENDMENT.json')['cpu']==cpu_name
    staged=json.loads(ops.stage_frozen('cpuQ',ROOT,cpu));assert staged['hashes']==cpu['source_hashes']
    text=checked(REGISTRY).read_text(encoding='utf-8');assert reg['amendment_sha256'] in text
    b=dict(runtime='/home/ubuntu/e3final',checkpoint='/home/ubuntu/mappospeed/production2/policy_00169.pt',
        authority=authority(text),budget=budget(),search=SEARCH,cpu=cpu,workers=12,load_max=20,
        hashes=staged['hashes'],amendment_sha256=reg['amendment_sha256'],N_exec=None,eta_primary_hours=None,
        accel_bounds=[-3.,2.6],gpu_deadline=read(ops.EVIDENCE/'DUAL_1755_DISPATCH.json')['unix']+86400,
        smoke_only=True,gpu_qualification={h:read(ops.EVIDENCE/f'PLAN_PROBE2_{h}_PARITY.json') for h in ('gpuA','gpuR')})
    out=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib
r=pathlib.Path({ROOT!r});assert r.is_dir();sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.storage import write,read
b=json.loads({json.dumps(b)!r})
for name in ('requests','responses'):(r/name).mkdir(exist_ok=True)
with (r/'AUTHORITY.md').open('x') as f:f.write({ascii(text)})
write(r/'MAPPO_DEV.json',read(pathlib.Path('/home/ubuntu/coopmwm20261101_v2/MAPPO_DEV.json')))
controls=[read(p) for p in pathlib.Path('/home/ubuntu/coopmwm20261101_v2/jobs').glob('control_mappo_*/RESULT.json')];assert len(controls)==12
write(r/'MAPPO_CHECK.json',dict(records=controls,new_sim_s=0))
write(r/'WORKERS.json',dict(workers=['gpuA','gpuR']))
for p in (r/'MAPPO_DEV.json',r/'MAPPO_CHECK.json',pathlib.Path(b['checkpoint']),pathlib.Path('/home/ubuntu/e3final/runtime/SOURCE_MANIFEST.json')):b['hashes'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
write(r/'BINDING.json',b)
from tools.coopmwm.accel_runner import verify
verify(r)
print(json.dumps(dict(root=str(r),binding_sha256=hashlib.sha256((r/'BINDING.json').read_bytes()).hexdigest(),new_sumo_sim_s=0)))
''',timeout=35))
    write(ops.EVIDENCE/'ACCEL_SMOKE_BINDING.json',out);print(json.dumps(out))


def smoke():
    d=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,subprocess,os,time
r=pathlib.Path({ROOT!r});assert r.is_dir();sys.path[:0]=[str(r/'pytools'),'/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from tools.coopmwm.storage import write
from tools.coopmwm.accel_runner import verify
verify(r);assert not (r/'SMOKE_DISPATCH.json').exists()
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,PYTHONPATH=str(r/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
with (r/'SMOKE.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.accel_runner',str(r),'--smoke-only'],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
d=dict(pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),root=str(r))
write(r/'SMOKE_DISPATCH.json',d);print(json.dumps(d))
''',timeout=30))
    write(ops.EVIDENCE/'ACCEL_SMOKE_DISPATCH.json',d);print(json.dumps(d))


def status():
    v=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time
r=pathlib.Path({ROOT!r});assert r.is_dir()
receipts={{p.name:json.loads(p.read_text()) for pattern in ('SMOKE.json','STOP.json','TRAINING_STARTED.json','*_UPDATE_*.json','*_VIABILITY_*.json','*_CHECK_*.json','*_DEV_*.json','ACCEL_EXACT_PLAN_*.json','*COMPLETE.json') for p in r.glob(pattern)}}
charges=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()] if (r/'CHARGES.jsonl').exists() else []
owners=[]
for name in ('SMOKE_DISPATCH.json','TRAIN_DISPATCH.json'):
 if not (r/name).exists():continue
 d=json.loads((r/name).read_text());p=pathlib.Path('/proc')/str(d['pid']);s=(p/'stat').read_text().split() if (p/'stat').exists() else None
 assert not s or (s[21]==d['start_ticks'] and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id'])
 owners.append(dict(d,state=s[2] if s else 'exited'))
print(json.dumps(dict(unix=time.time(),charged_sim_s=811640+sum(c['sim_s'] for c in charges),jobs=len(charges),closed=sum((r/'jobs'/c['id']/'CLOSE.json').exists() for c in charges),owners=owners,receipts=receipts,log=(r/('TRAIN.log' if (r/'TRAIN.log').exists() else 'SMOKE.log')).read_text()[-5000:])))
''',timeout=35))
    for name,value in v['receipts'].items():
        p=ops.EVIDENCE/'accel_runtime'/name
        if p.exists():assert read(p)==value
        else:write(p,value)
    write(ops.EVIDENCE/f'ACCEL_STATUS_{int(v["unix"])}.json',v)
    print(json.dumps(dict(v,receipts=list(v['receipts']))));return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke-register','bind-smoke','smoke','status'));p.add_argument('--cpu',default='CPU_coopmwm20261101_cpu31.json');a=p.parse_args()
    if a.mode=='smoke-register':smoke_register(a.cpu)
    elif a.mode=='bind-smoke':bind_smoke(a.cpu)
    elif a.mode=='smoke':smoke()
    else:status()
