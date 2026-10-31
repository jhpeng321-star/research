"""MAIN23:25 recovery operations; preserve every original deployment/STOP."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .ops import EVIDENCE, REG, ssh
from .storage import canonical, write

CPU = '/home/ubuntu/c1switch20261031/recovery_v8'
GPU = '/root/c1switch20261031/recovery_v8'
GPU_HOST = 'gpuR'
PREFIX = 'V9'


def preflight():
    from .resume import amended_plan
    result = amended_plan(json.loads((EVIDENCE/'V7_INTERRUPTION_AUDIT.json').read_text()))
    write(EVIDENCE/'V8_PLAN.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('complete_jobs','partial_jobs')},indent=2))


def inspect():
    out = {}
    for host in ('cpuQ', 'gpuA', 'gpuR'):
        out[host] = ssh(host, '''import json,os,subprocess
from pathlib import Path
out=dict(load=os.getloadavg())
if HOST != 'cpuQ':
 out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 out['base_exists']=Path('/root/c1switch20261031').exists()
 out['wm_source_exists']=Path('/root/c1switch20261031/main_v7').exists()
 out['inference_base_exists']=Path('/root/autodl-tmp/c4/speedhead20261030').exists()
out['recovery_exists']=Path('/home/ubuntu/c1switch20261031/recovery_v8' if HOST=='cpuQ' else '/root/c1switch20261031/recovery_v8').exists()
print(json.dumps(out))
'''.replace('HOST', repr(host)))
    write(EVIDENCE/'V8_INSPECT.json', out)
    print(json.dumps(out, indent=2))


def dependencies():
    expected = ssh('gpuR', '''import json
from pathlib import Path
r=Path('/root/c1switch20261031/main_v7')
b=json.loads((r/'rpc/wm/READY.json').read_text())['attestation']
print(json.dumps(b))
''')
    got = ssh('gpuA', f'''import hashlib,json
from pathlib import Path
expected={expected!r};out={{}}
for name,h in expected['sources'].items():
 p=Path(name);out[name]=dict(exists=p.is_file(),match=p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h)
print(json.dumps(out))
''')
    result = dict(expected=expected, gpuA=got, pass_=all(x['match'] for x in got.values()))
    write(EVIDENCE/'V8_DEPENDENCIES.json', result)
    print(json.dumps(dict(pass_=result['pass_'], differences={k:v for k,v in got.items() if not v['match']}), indent=2))


def stage():
    from .continuation import FROZEN
    from .design import sha
    from .mainops import BB, bundle
    from .resume import TOTAL
    sources,raw = bundle()
    design = sha('docs/paper_results/C1SWITCH_DESIGN.md')
    registration = REG.read_text(encoding='utf-8')
    assert '23:25 MAIN AMENDMENT S-C1SWITCH recovery' in registration
    assert any(design in line and str(TOTAL) in line for line in registration.splitlines())
    admission = json.loads((EVIDENCE/'V8_PLAN.json').read_text())
    assert admission['projected_total_sim_s']==TOTAL
    prior = admission['prior_gpu_s']
    # Shared-cap partition includes load, cross-host parity and all idle time.
    allocations = dict(bb=350,wm=4380)
    assert prior+sum(allocations.values())<=14400
    gpu_data = ssh('gpuR',f'''import hashlib,json
from pathlib import Path
r=Path('/root/c1switch20261031/main_v7');sources={sources!r}
old=json.loads((r/'STAGED.json').read_text())
assert all(old['tools/c1switch/'+n]==sources['tools/c1switch/'+n] for n in {FROZEN!r})
assert old['engine/branching.py']==sources['engine/branching.py']
out={{}}
for a in ('wm','bb'):
 b=json.loads((r/(a+'_BINDING.json')).read_text());parity=None
 for p in (r/'rpc'/a).glob('*.request.json'):
  req=json.loads(p.read_text());rp=p.with_name(p.name.replace('.request.','.result.'))
  if req['id'] in (a+'_s969112_j0','bb_reuse_v7_j0') and rp.exists() and 'validation_seed' not in req:
   parity=dict(belief=req['belief'],samples=json.loads(rp.read_text())['samples'])
 assert parity is not None,(a,'anonymous parity reference missing')
 b.update(checkpoint_path={GPU!r}+'/bb.pt',shared_gpu_prior_s={prior!r},shared_allocations_s={allocations!r},recovery_allocation_s={allocations!r}[a])
 out[a]=dict(binding=b,parity=parity)
out['physics']=json.loads((r/'physics.json').read_text())
print(json.dumps(out))
''')
    results = {}
    for host,target in [('cpuQ',CPU),('gpuA',GPU)]:
        results[host] = ssh(host,f'''import base64,hashlib,io,json,tarfile
from pathlib import Path
root=Path({target!r});assert not root.exists(),'existing recovery; inspect only'
assert root.parent.parent.is_dir()
root.parent.mkdir(exist_ok=True);root.mkdir();stage=root/'pytools';stage.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({raw!r})),mode='r:gz') as t:t.extractall(stage,filter='data')
sources={sources!r};assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==h for n,h in sources.items())
for n in ['jobs','tasks','prior_jobs','specs','cache/wm','cache/bb','rpc/wm','rpc/bb']:(root/n).mkdir(parents=True,exist_ok=True)
binding=dict(main_authorized=True,authorization='MAIN23:25',total_sim_s={TOTAL},design_sha256={design!r},sources=sources,
 retained_superseded_sim_s=12450,retained_abandoned_sim_s=11100,
 per_arm={{a:r['projected_total_sim_s'] for a,r in {admission['per_arm']!r}.items()}})
(root/'BINDING.json').write_text(json.dumps(binding,indent=2));(root/'STAGED.json').write_text(json.dumps(sources,indent=2))
for a in ('wm','bb'):
 (root/(a+'_BINDING.json')).write_text(json.dumps({gpu_data!r}[a]['binding'],indent=2))
 (root/'rpc'/a/'PARITY_INPUT.json').write_text(json.dumps({gpu_data!r}[a]['parity']))
if {host!r}=='gpuA':(root/'physics.json').write_text(json.dumps({gpu_data!r}['physics']))
print(json.dumps(dict(root=str(root),source_sha256={canonical(sources)!r})))
''',timeout=90)
    subprocess.run(['scp','-o','BatchMode=yes',str(BB),'gpuA:'+GPU+'/bb.pt'],check=True,timeout=90)
    write(EVIDENCE/'V8_STAGE.json',results)
    import_receipts()


def import_receipts():
    results = json.loads((EVIDENCE/'V8_STAGE.json').read_text()) if (EVIDENCE/'V8_STAGE.json').exists() else {}
    audit = json.loads((EVIDENCE/'V7_INTERRUPTION_AUDIT.json').read_text())
    results['import'] = ssh('cpuQ',f'''import hashlib,json,shutil,sys
from pathlib import Path
root=Path({CPU!r});old=root.parent/'main_v7';sys.path.insert(0,str(root/'pytools'))
from tools.c1switch.resume import imported_rows
audit={audit!r};assert hashlib.sha256((old/'accounting.jsonl').read_bytes()).hexdigest()==audit['cpuQ']['accounting_sha256']
assert not any(Path('/proc',pid).exists() for pid in audit['cpuQ']['own_pids'])
assert not (root/'accounting.jsonl').exists(),'existing import; inspect only'
shutil.copyfile(old/'accounting.jsonl',root/'PRIOR_ACCOUNTING.jsonl')
out=imported_rows(audit);(root/'accounting.jsonl').write_text(''.join(json.dumps(x)+'\\n' for x in out))
for job in audit['cpuQ']['jobs']:
 dest=root/('jobs' if job['result_present'] else 'prior_jobs')/job['id'];dest.mkdir()
 for p in (old/'jobs'/job['id']).glob('*.json'):shutil.copyfile(p,dest/p.name)
cached={{}}
for a in ('wm','bb'):
 count=0
 for p in (old/'rpc'/a).glob('*.request.json'):
  req=json.loads(p.read_text());rp=p.with_name(p.name.replace('.request.','.result.'))
  if not req['id'].startswith(a+'_s') or not rp.exists():continue
  result=json.loads(rp.read_text());assert result['request_sha256']==req['request_sha256'] and result['binding']==req['binding']
  (root/'cache'/a/(req['id']+'.json')).write_text(json.dumps(dict(belief=req['belief'],samples=result['samples'],original_request_sha256=req['request_sha256'])))
  count+=1
 cached[a]=count
print(json.dumps(dict(charged_sim_s=sum(x['sim_s'] for x in out),cached=cached,complete_jobs=sum(x['result_present'] for x in audit['cpuQ']['jobs']))))
''',timeout=90)
    write(EVIDENCE/'V8_STAGE.json',results)
    print(json.dumps(results,indent=2))


def cpu():
    from .mainops import CPU_ENV, GPU_ENV
    if (EVIDENCE/'V8_CPU.json').exists():
        write(EVIDENCE/f'V8_CPU_PRIOR_{time.time_ns()}.json',json.loads((EVIDENCE/'V8_CPU.json').read_text()))
    results = {}
    for host,target in [('cpuQ',CPU),(GPU_HOST,GPU)]:
        env = dict(CPU_ENV) if host=='cpuQ' else dict(GPU_ENV,CUDA_VISIBLE_DEVICES='',C1SWITCH_REAL_CPU='1',C1SWITCH_TEST_BUILD=target,
            PYTHONPATH=GPU+'/pydeps:'+GPU_ENV['PYTHONPATH'])
        results[host] = ssh(host,f'''import contextlib,io,json,os,sys
from pathlib import Path
r=Path({target!r});os.nice(19);os.environ.update({env!r});sys.path.insert(0,str(r/'pytools'))
if (r/'CPU.json').exists():(r/'CPU.json').rename(r/('CPU_PRIOR_'+str(__import__('time').time_ns())+'.json'))
if {host!r}=='cpuQ':
 from tools.c1switch.sumo import setup
 setup()
 import traci
 def forbidden(*a,**kw):raise AssertionError('CPU gate attempted SUMO')
 traci.start=forbidden;traci.simulationStep=forbidden
 from traffic_shepherd import simulation
 simulation.start_sumo=forbidden
else:
 sys.path.insert(0,str(r/'pydeps'))
 sys.path[:0]=['/root/autodl-tmp/c4/speedhead20261030/e3final/runtime/src','/root/autodl-tmp/c4/speedhead20261030/e3final/runtime','/root/autodl-tmp/c4/speedhead20261030/tools/speedhead']
 import wm_worker
import pytest
log=io.StringIO()
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
 code=pytest.main(['-q']+[str(r/'pytools/tests'/n) for n in ['test_c1switch.py','test_c1switch_observable.py','test_c1switch_feedback.py','test_c1switch_merge_fast.py','test_c1switch_analysis.py','test_c1switch_continuation.py','test_c1switch_resume.py','test_c1switch_recovery.py']])
out=dict(returncode=code,output=log.getvalue(),new_sim_s=0,new_gpu_s=0)
(r/'CPU.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
''',timeout=120)
    write(EVIDENCE/(PREFIX+'_CPU.json'),results)
    print(json.dumps(results,indent=2))


def dependency_runtime():
    version = ssh('gpuR',"import importlib.metadata,json; print(json.dumps(importlib.metadata.version('onnxruntime')))")
    result = ssh('gpuA',f'''import json,subprocess,sys
from pathlib import Path
r=Path({GPU!r});assert r.is_dir();target=r/'pydeps';assert not target.exists()
p=subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(target),{'onnxruntime=='+version!r}],capture_output=True,text=True)
print(json.dumps(dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,version={version!r})))
''',timeout=120)
    write(EVIDENCE/'V8_RUNTIME_DEPENDENCY.json',result)
    print(json.dumps(result,indent=2))


def launch(host, name, code, gpu=False):
    from .mainops import CPU_ENV, GPU_ENV
    root = CPU if host=='cpuQ' else GPU
    env = dict(CPU_ENV,C1SWITCH_ENGINE_ROOT=CPU,C1SWITCH_MAIN_ROOT=CPU) if host=='cpuQ' else dict(GPU_ENV,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=GPU+'/pydeps:'+GPU_ENV['PYTHONPATH'])
    result = ssh(host,f'''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
r=Path({root!r});stage=r/'pytools';assert not (r/'STOP').exists()
assert json.loads((r/'CPU.json').read_text())['returncode']==0
assert not (r/({name!r}+'_DISPATCH.json')).exists(),'already dispatched: inspect only'
sources=json.loads((r/'STAGED.json').read_text());assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==h for n,h in sources.items())
if {gpu!r}:
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=7168,free
env=dict(os.environ);env.update({env!r})
with (r/({name!r}+'.log')).open('x') as log:
 p=subprocess.Popen([sys.executable,'-B','-u','-c',{code!r}],cwd=stage,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
out=dict(host={host!r},pid=p.pid,unix=time.time(),name={name!r})
(r/({name!r}+'_DISPATCH.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out))
''')
    write(EVIDENCE/(PREFIX+'_'+name.upper()+'_DISPATCH.json'),result)
    print(json.dumps(result,indent=2))


def wm_gpu():
    launch(GPU_HOST,'wm_gpu',f"from pathlib import Path; from tools.c1switch.resume import gpu_serve; gpu_serve(Path({GPU!r})/'rpc/wm',Path({GPU!r})/'wm_BINDING.json')",True)


def bb_gpu():
    launch(GPU_HOST,'bb_gpu',f"from pathlib import Path; from tools.c1switch.resume import gpu_serve; gpu_serve(Path({GPU!r})/'rpc/bb',Path({GPU!r})/'bb_BINDING.json')",True)


def gate():
    result = ssh(GPU_HOST,f'''import json
from pathlib import Path
r=Path({GPU!r});out={{}}
for a in ('wm','bb'):
 p=r/'rpc'/a
 assert not (p/'STOP').exists()
 parity=json.loads((p/'PARITY.json').read_text());assert parity['pass_']
 ready=json.loads((p/'READY.json').read_text());assert Path('/proc',str(ready['pid'])).exists()
 out[a]=dict(parity=parity,attestation=ready['attestation'],pid=ready['pid'])
print(json.dumps(out))
''')
    expected = json.loads((EVIDENCE/'V8_DEPENDENCIES.json').read_text())['expected']
    for row in result.values():
        for key in ('sources','wm_config_sha256','candidate_sha256','mappo_sha256'):
            assert row['attestation'][key]==expected[key],key
    ssh('cpuQ',f'''import json
from pathlib import Path
p=Path({CPU!r})/'PARITY_GATE.json';assert not p.exists();p.write_text(json.dumps(dict(pass_=True,models={result!r})))
print(json.dumps(dict(pass_=True)))
''')
    write(EVIDENCE/(PREFIX+'_GPU_GATE.json'),dict(pass_=True,models=result))
    print(json.dumps(dict(pass_=True,pids={a:r['pid'] for a,r in result.items()}),indent=2))


def relay():
    root = EVIDENCE/'relay_recovery_v8'
    assert not root.exists(),'existing relay; inspect own PID'
    root.mkdir()
    env = dict(os.environ,C1SWITCH_CPU_RPC=CPU+'/rpc',C1SWITCH_GPU_RPC=GPU+'/rpc',C1SWITCH_GPU_HOST=GPU_HOST)
    with (root/'relay.log').open('x') as log:
        p = subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch.transport','--root',str(root.resolve())],env=env,stdout=log,stderr=subprocess.STDOUT)
    out = dict(pid=p.pid,root=str(root.resolve()),unix=time.time())
    write(root/'DISPATCH.json',out)
    print(json.dumps(out,indent=2))


def main_panel():
    ssh('cpuQ',f'''import json
from pathlib import Path
r=Path({CPU!r});assert json.loads((r/'PARITY_GATE.json').read_text())['pass_']
print(json.dumps(dict(pass_=True)))
''')
    launch('cpuQ','recovery',"from tools.c1switch.resume import main; import sys; sys.argv=['resume','supervisor']; main()")


def progress():
    result = {}
    for host,target in [('cpuQ',CPU),(GPU_HOST,GPU)]:
        result[host] = ssh(host,f'''import json,os,time
from pathlib import Path
r=Path({target!r});out=dict(unix=time.time(),load=os.getloadavg())
out['receipts']={{str(p.relative_to(r)):json.loads(p.read_text()) for n in ['STOP','*_DISPATCH.json','recovery/STATUS.json','recovery/COMPLETE.json','rpc/*/GPU_STARTED.json','rpc/*/GPU_FINISHED.json','rpc/*/STOP','rpc/*/PARITY.json'] for p in r.glob(n)}}
out['own_pids']={{str(x['pid']):Path('/proc',str(x['pid'])).exists() for x in out['receipts'].values() if isinstance(x,dict) and 'pid' in x}}
out['logs']={{p.name:p.read_text()[-4000:] for p in r.glob('*.log') if 'Traceback' in p.read_text()}}
if {host!r}=='cpuQ':
 out['charged_sim_s']=sum(json.loads(x)['sim_s'] for x in (r/'accounting.jsonl').read_text().splitlines())
 out['actual_closed_sim_s']=70560+sum(json.loads(p.read_text())['actual_sim_s'] for p in r.glob('jobs/*/CLOSE.json'))
 out['episodes']=len(list(r.glob('jobs/*_episode/RESULT.json')))
 out['physics']=len(list(r.glob('jobs/*_episode/PHYSICS.json')))
 out['branches']=len(list(r.glob('jobs/*_j*/RESULT.json')))
 out['picks']={{a:len(list(r.glob('jobs/'+a+'_*_episode/pick_j*.json'))) for a in ('wm','bb')}}
 out['failures']={{str(p.relative_to(r)):p.read_text()[-4000:] for p in (r/'recovery').glob('*.log') if 'Traceback' in p.read_text()}}
print(json.dumps(out))
''',timeout=90)
    write(EVIDENCE/(PREFIX+'_PROGRESS.json'),result)
    display = json.loads(json.dumps(result))
    for row in display.values():
        status = row['receipts'].get('recovery/STATUS.json')
        if status:
            status['done_count'] = len(status.pop('done'))
    print(json.dumps(display,indent=2))


def relocate():
    from .design import sha
    from .mainops import BB, bundle
    sources,raw = bundle()
    design = sha('docs/paper_results/C1SWITCH_DESIGN.md')
    assert any(design in line and 'ORIGINAL-HOST FALLBACK' in line for line in REG.read_text(encoding='utf-8').splitlines())
    old = ssh('gpuA',f'''import json
from pathlib import Path
r=Path({GPU!r});out={{}}
for a in ('wm','bb'):
 finish=json.loads((r/'rpc'/a/'GPU_FINISHED.json').read_text());assert not Path('/proc',str(finish['pid'])).exists()
 out[a]=dict(finish=finish,binding=json.loads((r/(a+'_BINDING.json')).read_text()),parity=json.loads((r/'rpc'/a/'PARITY_INPUT.json').read_text()))
assert json.loads((r/'STAGED.json').read_text())['tools/c1switch/resume.py']=={sources['tools/c1switch/resume.py']!r}
out['physics']=json.loads((r/'physics.json').read_text());print(json.dumps(out))
''')
    prior = json.loads((EVIDENCE/'V8_PLAN.json').read_text())['prior_gpu_s']+sum(old[a]['finish']['elapsed_s'] for a in ('wm','bb'))
    allocations = dict(wm=4470,bb=200)
    assert prior+sum(allocations.values())<=14400
    for arm in ('wm','bb'):
        old[arm]['binding'].update(shared_gpu_prior_s=prior,shared_allocations_s=allocations,recovery_allocation_s=allocations[arm])
    staged = ssh(GPU_HOST,f'''import base64,hashlib,io,json,tarfile
from pathlib import Path
r=Path({GPU!r});assert r.parent.is_dir();assert not r.exists(),'existing fallback; inspect only'
r.mkdir();s=r/'pytools';s.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({raw!r})),mode='r:gz') as t:t.extractall(s,filter='data')
sources={sources!r};assert all(hashlib.sha256((s/n).read_bytes()).hexdigest()==h for n,h in sources.items())
(r/'STAGED.json').write_text(json.dumps(sources,indent=2))
for a in ('wm','bb'):
 (r/'rpc'/a).mkdir(parents=True)
 (r/(a+'_BINDING.json')).write_text(json.dumps({old!r}[a]['binding'],indent=2))
 (r/'rpc'/a/'PARITY_INPUT.json').write_text(json.dumps({old!r}[a]['parity']))
(r/'physics.json').write_text(json.dumps({old!r}['physics']))
print(json.dumps(dict(root=str(r),source_sha256={canonical(sources)!r},prior_gpu_s={prior!r},allocations={allocations!r})))
''',timeout=90)
    subprocess.run(['scp','-o','BatchMode=yes',str(BB),GPU_HOST+':'+GPU+'/bb.pt'],check=True,timeout=90)
    updated = ssh('cpuQ',f'''import json
from pathlib import Path
r=Path({CPU!r});assert not (r/'recovery_DISPATCH.json').exists();assert not (r/'STOP').exists()
for a in ('wm','bb'):
 p=r/(a+'_BINDING.json');p.rename(r/(a+'_GPUA_BINDING.json'))
 p.write_text(json.dumps({old!r}[a]['binding'],indent=2))
print(json.dumps(dict(updated=True,charged_sim_s=sum(json.loads(x)['sim_s'] for x in (r/'accounting.jsonl').read_text().splitlines()))))
''')
    write(EVIDENCE/'V9_STAGE.json',dict(gpu=staged,cpu=updated,gpuA_finish={a:old[a]['finish'] for a in ('wm','bb')}))
    print(json.dumps(dict(gpu=staged,cpu=updated),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('inspect','dependencies','preflight','stage','import_receipts','cpu','dependency_runtime','wm_gpu','bb_gpu','gate','relay','main_panel','progress','relocate'))
    globals()[parser.parse_args().mode]()
