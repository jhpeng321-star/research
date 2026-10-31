"""Prospective immutable main deployment and own-job operations."""
import argparse
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from .cpuops import ENV as CPU_ENV, REMOTE as STEP0
from .design import BUDGET, sha
from .ops import EVIDENCE, REG, ssh
from .storage import canonical, write

CPU = '/home/ubuntu/c1switch20261031/main_v4'
GPU = '/root/c1switch20261031/main_v4'
BB = Path('E:/research-dense/runs/active/WM-S-C1FAST-20261006/box/full_bb_traj/wm_s.pt')
GPU_ENV = dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               C4_ALLOWED_ROOTS='/root/autodl-tmp/c4:/root/c1switch20261031',
               PYTHONPATH='/root/autodl-tmp/c4/speedhead20261030/pythondeps')


def bundle():
    paths = sorted(Path('tools/c1switch').glob('*.py')) + [
        Path('tests/test_c1switch.py'),Path('tests/test_c1switch_observable.py'),
        Path('tests/test_c1switch_feedback.py')]
    sources = {p.as_posix():sha(p) for p in paths}
    engine = Path('traffic_shepherd/c4/branching.py')
    sources['engine/branching.py'] = sha(engine)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as tar:
        for p in paths:tar.add(p,arcname=p.as_posix())
        tar.add(engine,arcname='engine/branching.py')
    return sources,base64.b64encode(buf.getvalue()).decode()


def cpu():
    sources,raw = bundle()
    version = canonical(sources)[:12]
    results = {}
    for host in ('cpuQ','gpuR'):
        parent = '/home/ubuntu/c1switch20261031' if host == 'cpuQ' else '/root/c1switch20261031'
        root = parent+'/maincheck_'+version
        staged = ssh(host,f'''import base64,hashlib,io,json,tarfile
from pathlib import Path
root=Path({root!r});assert root.parent.is_dir()
if not root.exists():
 root.mkdir()
 with tarfile.open(fileobj=io.BytesIO(base64.b64decode({raw!r})),mode='r:gz') as t:t.extractall(root,filter='data')
sources={sources!r}
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in sources.items())
print(json.dumps(dict(root=str(root),complete=(root/'CPU.json').exists())))
''')
        if host == 'gpuR' and not staged['complete']:
            assert sha(BB) == json.loads((EVIDENCE/'MODEL_SELECTION.json').read_text())['bb_checkpoint_sha256']
            subprocess.run(['scp','-o','BatchMode=yes',str(BB),f'gpuR:{root}/bb.pt'],check=True,timeout=90)
            physics = json.loads(BB.with_name('TRAIN_RECEIPT.json').read_text())['physics']
            ssh(host,f'''import json
from pathlib import Path
p=Path({root!r})/'physics.json'
assert not p.exists()
p.write_text(json.dumps({physics!r}))
print(json.dumps(dict(physics_source_sha256={sha(BB.with_name('TRAIN_RECEIPT.json'))!r})))
''')
        env = dict(CPU_ENV) if host == 'cpuQ' else dict(GPU_ENV,CUDA_VISIBLE_DEVICES='',
            C1SWITCH_REAL_CPU='1',C1SWITCH_TEST_BUILD=root)
        result = ssh(host,f'''import contextlib,io,json,os,sys
from pathlib import Path
root=Path({root!r})
if (root/'CPU.json').exists():print((root/'CPU.json').read_text());raise SystemExit
os.nice(19);os.environ.update({env!r});sys.path.insert(0,str(root))
if {host!r}=='cpuQ':
 from tools.c1switch.sumo import setup
 setup()
 import traci
 def forbidden(*a,**kw):raise AssertionError('CPU check attempted SUMO')
 traci.start=forbidden;traci.simulationStep=forbidden
 from traffic_shepherd import simulation
 simulation.start_sumo=forbidden
else:
 sys.path[:0]=['/root/autodl-tmp/c4/speedhead20261030/e3final/runtime/src','/root/autodl-tmp/c4/speedhead20261030/e3final/runtime']
 sys.path.insert(0,'/root/autodl-tmp/c4/speedhead20261030/tools/speedhead')
 import wm_worker
import pytest
log=io.StringIO()
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
 code=pytest.main(['-q',str(root/'tests/test_c1switch.py'),str(root/'tests/test_c1switch_observable.py'),str(root/'tests/test_c1switch_feedback.py')])
out=dict(returncode=code,output=log.getvalue(),root=str(root),sources={sources!r},new_sim_s=0,GPU_s=0)
(root/'CPU.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
''',timeout=120)
        results[host] = result
        print(json.dumps({k:v for k,v in result.items() if k != 'sources'},indent=2))
    write(EVIDENCE/'MAIN_CPU.json',results)


def stage():
    sources,raw = bundle()
    proof = json.loads((EVIDENCE/'MAIN_CPU.json').read_text())
    if any(p['returncode'] or p['sources'] != sources for p in proof.values()):
        raise ValueError('current sources need both host CPU gates')
    design = sha('docs/paper_results/C1SWITCH_DESIGN.md')
    if not any('S-C1SWITCH' in row and design in row and '478410' in row
               for row in REG.read_text(encoding='utf-8').splitlines()):
        raise ValueError('current design/per-arm budget not registered')
    selection = json.loads((EVIDENCE/'MODEL_SELECTION.json').read_text())
    bindings = {arm:dict(selection,model=arm,checkpoint_path=GPU+'/bb.pt',
        adapter_sha256=sources['tools/c1switch/adapter.py'],gpu_cap_s=10800 if arm=='wm' else 3600,
        allocator_limit_gib=2.5,entries=False,mix_active=False) for arm in ('wm','bb')}
    out = {}
    for host,root in [('cpuQ',CPU),('gpuR',GPU)]:
        out[host] = ssh(host,f'''import base64,hashlib,io,json,tarfile
from pathlib import Path
root=Path({root!r});assert root.parent.is_dir();assert not root.exists(),'existing main deployment; reconcile'
if {host!r}=='cpuQ':
 old=Path({STEP0!r})
 assert json.loads((old/'STEP0_GATE.json').read_text())['pass_']
 assert json.loads((old/'TRUNK_GATE.json').read_text())['pass_']
 assert sum(json.loads(x)['sim_s'] for x in (old/'accounting.jsonl').read_text().splitlines())==47010
root.mkdir();stage=root/'pytools';stage.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({raw!r})),mode='r:gz') as t:t.extractall(stage,filter='data')
sources={sources!r};assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==h for n,h in sources.items())
(root/'STAGED.json').write_text(json.dumps(sources,indent=2))
for n in ['jobs','tasks','rpc','rpc/wm','rpc/bb']:(root/n).mkdir(exist_ok=True)
binding=dict(main_authorized=True,total_sim_s=478410,step0_sim_s=47010,per_arm={BUDGET!r},
 design_sha256={design!r},sources=sources,root=str(root))
(root/'BINDING.json').write_text(json.dumps(binding,indent=2))
for arm,b in {bindings!r}.items():(root/(arm+'_BINDING.json')).write_text(json.dumps(b,indent=2))
if {host!r}=='cpuQ':
 (root/'accounting.jsonl').write_text(json.dumps(dict(id='retained_step0_47010',sim_s=47010,stage='step0'))+'\\n')
print(json.dumps(binding))
''')
    # Use the exact checkpoint/physics already validated in the GPU CPU gate.
    tested = proof['gpuR']['root']
    ssh('gpuR',f'''import hashlib,json,shutil
from pathlib import Path
root=Path({GPU!r});tested=Path({tested!r})
assert hashlib.sha256((tested/'bb.pt').read_bytes()).hexdigest()=={selection['bb_checkpoint_sha256']!r}
for name in ['bb.pt','physics.json']:shutil.copyfile(tested/name,root/name)
print(json.dumps(dict(bb_sha256=hashlib.sha256((root/'bb.pt').read_bytes()).hexdigest())))
''')
    write(EVIDENCE/'MAIN_STAGE.json',out)
    print(json.dumps(dict(cpu=CPU,gpu=GPU,design_sha256=design,source_binding=canonical(sources))))


def launch_gpu():
    result = ssh('gpuR',f'''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path({GPU!r});stage=root/'pytools'
assert not (root/'GPU_DISPATCH.json').exists(),'existing own GPU dispatch'
sources=json.loads((root/'STAGED.json').read_text())
assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==h for n,h in sources.items())
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=7168,free
env=dict(os.environ);env.update({GPU_ENV!r},CUDA_VISIBLE_DEVICES='0')
out=dict(host='gpuR',free_mib=free,allocator_total_gib=5,unix=time.time(),models={{}})
for arm in ('wm','bb'):
 with (root/(arm+'_gpu.log')).open('x') as log:
  p=subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch.rpc','--root',str(root/'rpc'/arm),
   '--adapter','tools.c1switch.adapter:create','--binding',str(root/(arm+'_BINDING.json'))],cwd=stage,
   env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
 out['models'][arm]=dict(pid=p.pid,cap_s=10800 if arm=='wm' else 3600)
(root/'GPU_DISPATCH.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
''')
    write(EVIDENCE/'GPU_DISPATCH.json',result)
    print(json.dumps(result,indent=2))


def relay():
    root = EVIDENCE/'relay_main_v4'
    if root.exists():
        raise ValueError('existing local relay namespace; inspect own PID before continuation')
    root.mkdir()
    with (root/'relay.log').open('x') as log:
        p = subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch.transport','--root',str(root.resolve())],
                             stdout=log,stderr=subprocess.STDOUT)
    result = dict(pid=p.pid,root=str(root.resolve()),unix=time.time())
    write(root/'DISPATCH.json',result)
    print(json.dumps(result,indent=2))


def launch(phase):
    if phase == 'main':
        gate = json.loads((EVIDENCE/'MAIN_CANARY.json').read_text())
        if not gate['pass_']:
            raise ValueError('one-world canary required')
    env = dict(CPU_ENV,C1SWITCH_ENGINE_ROOT=CPU)
    result = ssh('cpuQ',f'''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path({CPU!r});stage=root/'pytools';phase={phase!r}
assert not (root/'STOP').exists()
assert not (root/(phase+'_DISPATCH.json')).exists(),'existing dispatch'
assert os.getloadavg()[0]<28
sources=json.loads((root/'STAGED.json').read_text())
assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==h for n,h in sources.items())
if phase=='main':assert (root/'canary/COMPLETE.json').exists()
env=dict(os.environ);env.update({env!r})
with (root/(phase+'_dispatcher.log')).open('x') as log:
 p=subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch.dispatch',phase],cwd=stage,env=env,
  stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
out=dict(pid=p.pid,phase=phase,unix=time.time(),host='cpuQ',max_episode_workers=2 if phase=='canary' else 12,
 max_sumo_workers_including_branches=4 if phase=='canary' else 24)
(root/(phase+'_DISPATCH.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out))
''')
    write(EVIDENCE/(phase.upper()+'_DISPATCH.json'),result)
    print(json.dumps(result,indent=2))


def progress():
    out = {}
    out['cpuQ'] = ssh('cpuQ',f'''import json,os
from pathlib import Path
root=Path({CPU!r});out=dict(load=os.getloadavg(),exists=root.exists())
if root.exists():
 for name in ['canary_DISPATCH.json','main_DISPATCH.json','canary/STATUS.json','canary/COMPLETE.json','main/STATUS.json','main/COMPLETE.json','STOP']:
  p=root/name
  if p.exists():out[name]=json.loads(p.read_text())
 out['charged_sim_s']=sum(json.loads(x)['sim_s'] for x in (root/'accounting.jsonl').read_text().splitlines())
 out['actual_sim_s_closed']=47010+sum(json.loads(p.read_text())['actual_sim_s'] for p in root.glob('jobs/*/CLOSE.json'))
 out['completed_episodes']=[p.parent.name for p in root.glob('jobs/*_episode/RESULT.json')]
 out['closed_branches']=len(list(root.glob('jobs/*_j*/RESULT.json')))
 out['requests']={{a:len(list((root/'rpc'/a).glob('*.request.json'))) for a in ('wm','bb')}}
 if (root/'STOP').exists():
  out['logs']={{str(p.relative_to(root)):p.read_text() for p in root.glob('canary/*.log')}}
print(json.dumps(out))
''')
    out['gpuR'] = ssh('gpuR',f'''import json,subprocess
from pathlib import Path
root=Path({GPU!r});out={{}}
if root.exists():
 for arm in ('wm','bb'):
  row={{}}
  for name in ['READY.json','STOP','GPU_STARTED.json','GPU_FINISHED.json']:
   p=root/'rpc'/arm/name
   if p.exists():row[name]=json.loads(p.read_text())
  row['results']=len(list((root/'rpc'/arm).glob('*.result.json')))
  if (root/'rpc'/arm/'STOP').exists():row['log']=(root/(arm+'_gpu.log')).read_text()
  out[arm]=row
out['gpu_memory']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used','--format=csv,noheader,nounits'],text=True)
print(json.dumps(out))
''')
    write(EVIDENCE/'MAIN_PROGRESS.json',out)
    # Full attestations are retained in evidence; terminal output is progress only.
    display = json.loads(json.dumps(out))
    for row in display['gpuR'].values():
        if isinstance(row,dict) and 'READY.json' in row:
            row['READY.json'].pop('attestation',None)
    print(json.dumps(display,indent=2))
    return out


def canary_gate():
    out = progress()
    cpu = out['cpuQ']
    if 'STOP' in cpu or 'canary/COMPLETE.json' not in cpu:
        raise ValueError('canary is incomplete or stopped')
    result = ssh('cpuQ',f'''import json
from pathlib import Path
root=Path({CPU!r});results={{}}
for arm in ('wm','bb'):
 p=root/'jobs'/(arm+'_s969112_episode');r=json.loads((p/'RESULT.json').read_text())
 assert len(r['decisions'])==10 and len([d for d in r['decisions'] if 'branches' in d])==3
 assert r['J600']['collisions']==0 and r['J600']['teleports']==0
 assert json.loads((p/'CLOSE.json').read_text())==dict(actual_sim_s=900.,charged_sim_s=900.)
 for d in r['decisions']:
  if 'branches' in d:
   assert len(d['branches'])==5
   for b in d['branches'].values():assert b['prefix']==d['prefix']
 results[arm]=dict(J600=r['J600'],schedule=r['schedule'],switches=r['switching_frequency'],measured=3)
closes=[json.loads(p.read_text()) for p in root.glob('jobs/*/CLOSE.json')]
charged=sum(json.loads(x)['sim_s'] for x in (root/'accounting.jsonl').read_text().splitlines())
actual=47010+sum(x['actual_sim_s'] for x in closes)
assert actual==charged
out=dict(pass_=True,world=969112,results=results,charged_sim_s=charged,actual_sim_s=actual)
(root/'CANARY_GATE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
''')
    write(EVIDENCE/'MAIN_CANARY.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('cpu','stage','launch_gpu','relay','canary','main','progress','canary_gate'))
    args = p.parse_args()
    if args.mode in ('canary','main'):
        launch(args.mode)
    else:
        globals()[args.mode]()
