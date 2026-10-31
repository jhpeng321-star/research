"""Immutable shared-cap recovery staging, dispatch and receipt inspection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .mainops import BB, CPU_ENV, GPU_ENV, bundle
from .ops import EVIDENCE, REG, ssh
from .shared import CPU, GPU, TOTAL, DROPPED
from .storage import canonical, write

OLD_CPU='/home/ubuntu/c1switch20261031/recovery_v8'
OLD_GPU='/root/c1switch20261031/recovery_v8'


def preflight():
    audit=json.loads((EVIDENCE/'V9_FINAL_AUDIT.json').read_text())
    per_arm=dict(wm=175560,bb=180540,random=5400,native=14400,ctrl0=14400,mappo=14400,pi=14400,circles=14400)
    for j in audit['cpuQ']['jobs']:
        if not j['result_present'] and not j['physics_present']:
            per_arm[j['id'].split('_')[0]]+=j['close']['actual_sim_s']
    assert 47010+12450+sum(per_arm.values())==TOTAL
    out=dict(total_sim_s=TOTAL,per_arm=per_arm,prior_gpu_s=audit['summary']['prior_gpu_s'],
             cap_s=int(audit['summary']['remaining_gpu_s'])-3,dropped_random_seeds=list(DROPPED))
    write(EVIDENCE/'V10_PLAN.json',out);print(json.dumps(out,indent=2))


def stage():
    from .continuation import FROZEN
    from .design import sha
    audit=json.loads((EVIDENCE/'V9_FINAL_AUDIT.json').read_text())
    assert audit['summary']['projected_total_sim_s']==TOTAL
    assert audit['summary']['dropped_random_seeds']==list(DROPPED)
    design=sha('docs/paper_results/C1SWITCH_DESIGN.md')
    assert any(design in line and 'SHARED-CAP' in line for line in REG.read_text(encoding='utf-8').splitlines())
    prior=audit['summary']['prior_gpu_s'];cap=int(14400-prior)-3
    sources,raw=bundle()
    gpu_data=ssh('gpuR',f'''import json
from pathlib import Path
r=Path({OLD_GPU!r});old=json.loads((r/'STAGED.json').read_text());sources={sources!r}
assert all(old['tools/c1switch/'+n]==sources['tools/c1switch/'+n] for n in {FROZEN!r})
out={{}}
for a in ('wm','bb'):
 finish=json.loads((r/'rpc'/a/'GPU_FINISHED.json').read_text());assert not Path('/proc',str(finish['pid'])).exists()
 b=json.loads((r/(a+'_BINDING.json')).read_text());b['checkpoint_path']={GPU!r}+'/bb.pt'
 out[a]=dict(binding=b,attestation=json.loads((r/'rpc'/a/'READY.json').read_text())['attestation'])
out['physics']=json.loads((r/'physics.json').read_text());print(json.dumps(out))
''')
    per_arm=dict(wm=175560,bb=180540,random=5400,native=14400,ctrl0=14400,mappo=14400,pi=14400,circles=14400)
    partial={j['id'] for j in audit['cpuQ']['jobs'] if not j['result_present'] and not j['physics_present']}
    for j in audit['cpuQ']['jobs']:
        if j['id'] in partial:per_arm[j['id'].split('_')[0]]+=j['close']['actual_sim_s']
    assert 47010+12450+sum(per_arm.values())==TOTAL
    results={}
    for host,target in [('cpuQ',CPU),('gpuR',GPU)]:
        results[host]=ssh(host,f'''import base64,hashlib,io,json,tarfile
from pathlib import Path
r=Path({target!r});assert r.parent.is_dir();assert not r.exists(),'existing shared stage; inspect only'
r.mkdir();s=r/'pytools';s.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({raw!r})),mode='r:gz') as t:t.extractall(s,filter='data')
sources={sources!r};assert all(hashlib.sha256((s/n).read_bytes()).hexdigest()==h for n,h in sources.items())
(r/'STAGED.json').write_text(json.dumps(sources,indent=2))
for n in ['jobs','prior_jobs','tasks','specs','cache/wm','cache/bb','rpc/wm','rpc/bb']:(r/n).mkdir(parents=True,exist_ok=True)
binding=dict(authorization='MAIN23:25',main_authorized=True,total_sim_s={TOTAL},per_arm={per_arm!r},sources=sources,
 retained_superseded_sim_s=12450,retained_abandoned_sim_s=17820,dropped_random_seeds={DROPPED!r},design_sha256={design!r})
(r/'BINDING.json').write_text(json.dumps(binding,indent=2))
(r/'SHARED_BUDGET.json').write_text(json.dumps(dict(prior_gpu_s={prior!r},cap_s={cap!r},owner_cap_s=14400),indent=2))
for a in ('wm','bb'):
 (r/(a+'_BINDING.json')).write_text(json.dumps({gpu_data!r}[a]['binding'],indent=2))
 (r/(a+'_ATTESTATION.json')).write_text(json.dumps({gpu_data!r}[a]['attestation'],indent=2))
(r/'physics.json').write_text(json.dumps({gpu_data!r}['physics']))
print(json.dumps(dict(root=str(r),sources_sha256={canonical(sources)!r},shared_gpu_cap_s={cap},prior_gpu_s={prior!r},per_arm={per_arm!r})))
''',timeout=90)
    subprocess.run(['scp','-o','BatchMode=yes',str(BB),'gpuR:'+GPU+'/bb.pt'],check=True,timeout=90)
    results['import']=ssh('cpuQ',f'''import json,shutil,sys
from pathlib import Path
r=Path({CPU!r});old=Path({OLD_CPU!r});audit={audit!r};partial={partial!r}
assert not any(Path('/proc',pid).exists() for pid in audit['cpuQ']['own_pids'])
rows=[json.loads(x) for x in (old/'accounting.jsonl').read_text().splitlines()];assert rows==audit['cpuQ']['accounting']
out=[]
for row in rows:
 row=dict(row)
 if any(row['id'].startswith(ident+'_') for ident in partial):
  arm,rest=row['id'].split('_',1);row['id']=arm+'_abandoned_v8_'+rest
 out.append(row)
assert len({{x['id'] for x in out}})==len(out)
shutil.copyfile(old/'accounting.jsonl',r/'PRIOR_ACCOUNTING.jsonl')
(r/'accounting.jsonl').write_text(''.join(json.dumps(x)+'\\n' for x in out))
for p in (old/'prior_jobs').iterdir():shutil.copytree(p,r/'prior_jobs'/p.name)
for j in audit['cpuQ']['jobs']:
 parent='prior_jobs' if j['id'] in partial else 'jobs'
 shutil.copytree(old/'jobs'/j['id'],r/parent/j['id'],dirs_exist_ok=True)
for p in (old/'specs').glob('*.json'):
 spec=json.loads(p.read_text());ep=r/'jobs'/f"{{spec['arm']}}_s{{spec['world']['seed']}}_episode"
 if (ep/'PHYSICS.json').exists():shutil.copyfile(p,r/'specs'/p.name)
for a in ('wm','bb'):
 for p in (old/'cache'/a).glob('*.json'):shutil.copyfile(p,r/'cache'/a/p.name)
 for p in (old/'rpc'/a).glob('*.request.json'):
  req=json.loads(p.read_text());rp=p.with_name(p.name.replace('.request.','.result.'))
  if not rp.exists():continue
  result=json.loads(rp.read_text());assert result['request_sha256']==req['request_sha256'] and result['binding']==req['binding']
  target=r/'cache'/a/(req['id']+'.json');record=dict(belief=req['belief'],samples=result['samples'],original_request_sha256=req['request_sha256'])
  if target.exists():
   existing=json.loads(target.read_text());assert all(existing[k]==record[k] for k in ('belief','samples'))
  else:target.write_text(json.dumps(record))
print(json.dumps(dict(charged_sim_s=sum(x['sim_s'] for x in out),complete_physics=11,complete_branches=231,
 cached={{a:len(list((r/'cache'/a).glob('*.json'))) for a in ('wm','bb')}})))
''',timeout=90)
    write(EVIDENCE/'V10_STAGE.json',results)
    print(json.dumps(results,indent=2))


def cpu():
    results={}
    for host,target in [('cpuQ',CPU),('gpuR',GPU)]:
        env=dict(CPU_ENV,C1SWITCH_ENGINE_ROOT=CPU,C1SWITCH_MAIN_ROOT=CPU) if host=='cpuQ' else dict(GPU_ENV,CUDA_VISIBLE_DEVICES='',C1SWITCH_REAL_CPU='1',C1SWITCH_TEST_BUILD=target)
        results[host]=ssh(host,f'''import contextlib,io,json,os,sys
from pathlib import Path
r=Path({target!r});os.nice(19);os.environ.update({env!r});sys.path.insert(0,str(r/'pytools'))
if {host!r}=='cpuQ':
 from tools.c1switch.sumo import setup
 setup()
 import traci
 def forbidden(*a,**kw):raise AssertionError('CPU gate attempted SUMO')
 traci.start=forbidden;traci.simulationStep=forbidden
else:
 sys.path[:0]=['/root/autodl-tmp/c4/speedhead20261030/e3final/runtime/src','/root/autodl-tmp/c4/speedhead20261030/e3final/runtime','/root/autodl-tmp/c4/speedhead20261030/tools/speedhead']
 import wm_worker
import pytest
log=io.StringIO()
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
 code=pytest.main(['-q']+[str(p) for p in sorted((r/'pytools/tests').glob('test_c1switch*.py'))])
out=dict(returncode=code,output=log.getvalue(),new_sim_s=0,new_gpu_s=0)
(r/'CPU.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
''',timeout=120)
    write(EVIDENCE/'V10_CPU.json',results);print(json.dumps(results,indent=2))


def launch(host,name,code):
    target=CPU if host=='cpuQ' else GPU
    env=dict(CPU_ENV,C1SWITCH_MAIN_ROOT=CPU,C1SWITCH_ENGINE_ROOT=CPU) if host=='cpuQ' else dict(GPU_ENV,CUDA_VISIBLE_DEVICES='0')
    result=ssh(host,f'''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
r=Path({target!r});assert not (r/'STOP').exists();assert not (r/({name!r}+'_DISPATCH.json')).exists()
assert json.loads((r/'CPU.json').read_text())['returncode']==0
sources=json.loads((r/'STAGED.json').read_text());assert all(hashlib.sha256((r/'pytools'/n).read_bytes()).hexdigest()==h for n,h in sources.items())
if {host!r}=='gpuR':
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=7168,free
env=dict(os.environ);env.update({env!r})
with (r/({name!r}+'.log')).open('x') as log:
 p=subprocess.Popen([sys.executable,'-B','-u','-c',{code!r}],cwd=r/'pytools',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
out=dict(pid=p.pid,host={host!r},unix=time.time());(r/({name!r}+'_DISPATCH.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out))
''')
    write(EVIDENCE/('V10_'+name.upper()+'_DISPATCH.json'),result);print(json.dumps(result,indent=2))


def main_panel():launch('cpuQ','recovery','from tools.c1switch.shared import supervisor; supervisor()')
def gpu():launch('gpuR','gpu','from tools.c1switch.shared import serve; serve()')


def relay():
    root=EVIDENCE/'relay_shared_v10a';assert not root.exists();root.mkdir()
    env=dict(os.environ,C1SWITCH_CPU_RPC=CPU+'/rpc',C1SWITCH_GPU_RPC=GPU+'/rpc',C1SWITCH_GPU_HOST='gpuR')
    with (root/'relay.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch.transport','--root',str(root.resolve())],env=env,stdout=log,stderr=subprocess.STDOUT)
    out=dict(pid=p.pid,root=str(root.resolve()),unix=time.time());write(root/'DISPATCH.json',out);print(json.dumps(out,indent=2))


def progress():
    out={}
    for host,target in [('cpuQ',CPU),('gpuR',GPU)]:
        out[host]=ssh(host,f'''import json,os,time
from pathlib import Path
r=Path({target!r});out=dict(unix=time.time(),load=os.getloadavg())
out['receipts']={{str(p.relative_to(r)):json.loads(p.read_text()) for n in ['STOP','*_DISPATCH.json','GPU_STARTED.json','GPU_FINISHED.json','recovery/STATUS.json','recovery/COMPLETE.json','rpc/*/STOP'] for p in r.glob(n)}}
out['own_pids']={{str(x['pid']):Path('/proc',str(x['pid'])).exists() for x in out['receipts'].values() if 'pid' in x}}
out['rpc']={{a:dict(requests=len(list((r/'rpc'/a).glob('*.request.json'))),results=len(list((r/'rpc'/a).glob('*.result.json')))) for a in ('wm','bb')}}
if {host!r}=='cpuQ':
 out['charged_sim_s']=sum(json.loads(x)['sim_s'] for x in (r/'accounting.jsonl').read_text().splitlines())
 out['actual_closed_sim_s']=77280+sum(json.loads(p.read_text())['actual_sim_s'] for p in r.glob('jobs/*/CLOSE.json'))
 out['complete_episodes']=len(list(r.glob('jobs/*_episode/RESULT.json')))
 out['complete_physics']=sum((p/'RESULT.json').exists() or (p/'PHYSICS.json').exists() for p in (r/'jobs').glob('*_episode'))
 out['complete_branches']=len(list(r.glob('jobs/*_j*/RESULT.json')))
 out['picks']={{a:len(list(r.glob('jobs/'+a+'_*_episode/pick_j*.json'))) for a in ('wm','bb')}}
 out['failures']={{p.name:p.read_text()[-2000:] for p in (r/'recovery').glob('*.log') if 'Traceback' in p.read_text()}}
else:
 out['log_tail']=(r/'gpu.log').read_text()[-3000:] if (r/'gpu.log').exists() else ''
print(json.dumps(out))
''',timeout=90)
    write(EVIDENCE/'V10_PROGRESS.json',out)
    display=json.loads(json.dumps(out))
    status=display['cpuQ']['receipts'].get('recovery/STATUS.json')
    if status:status['active_count']=len(status.pop('active'))
    print(json.dumps(display,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('preflight','stage','cpu','relay','main_panel','gpu','progress'))
    globals()[p.parse_args().mode]()
