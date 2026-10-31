"""MAIN20:52 immutable continuation; old STOPs are never removed or bypassed."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .design import BUDGET, CONTROLLERS, sha
from .mainops import bundle, CPU_ENV, GPU_ENV, CPU_V4, GPU_V4
from .ops import EVIDENCE, REG, ssh
from .storage import canonical, write

CPU = '/home/ubuntu/c1switch20261031/main_v7'
GPU = '/root/c1switch20261031/main_v7'
TOTAL = 490860
SUPERSEDED = 12450
FROZEN = ('adapter.py','feedback.py','observable.py','merge_fast.py','rpc.py',
          'model.py','core.py','episode.py','main_backend.py','sumo.py','rerun.py')


def imported_ledger(rows):
    """Preserve the original debit while excluding superseded WM from arm caps."""
    if sum(r['sim_s'] for r in rows) != 71910:
        raise ValueError('original paired canary ledger incomplete')
    if sum(r['sim_s'] for r in rows if r['id'].startswith('wm_')) != SUPERSEDED:
        raise ValueError('superseded WM debit mismatch')
    out = [dict(r,id='superseded_v4_'+r['id']) if r['id'].startswith('wm_') else dict(r) for r in rows]
    if len({r['id'] for r in out}) != len(out):
        raise ValueError('duplicate retained reservation')
    return out


def reuse_decision(old, costs):
    from .core import choose
    pick = choose(costs,old['incumbent'])
    if pick != old['pick']:
        raise ValueError('BB corrected anonymous pick changed; reuse forbidden')
    return pick


def stage():
    sources,raw = bundle()
    proof = json.loads((EVIDENCE/'MAIN_CPU.json').read_text())
    assert all(p['returncode']==0 and p['sources']==sources for p in proof.values())
    design = sha('docs/paper_results/C1SWITCH_DESIGN.md')
    registration = REG.read_text(encoding='utf-8')
    assert '20:52 MAIN AUTHORISES S-C1SWITCH continuation' in registration
    assert any(design in r and '490860' in r and 'S-C1SWITCH' in r for r in registration.splitlines())
    gpu = ssh('gpuR',f'''import json,math
from pathlib import Path
old=Path('/root/c1switch20261031/main_v6');sources={sources!r}
original=json.loads((old/'STAGED.json').read_text())
assert all(original['tools/c1switch/'+n]==sources['tools/c1switch/'+n] for n in {FROZEN!r})
assert original['engine/branching.py']==sources['engine/branching.py']
out={{}}
for a in ('wm','bb'):
 rows=[]
 for v in (4,5,6):
  p=old.parent/('main_v'+str(v))/'rpc'/a/'GPU_FINISHED.json'
  if p.exists():
   r=json.loads(p.read_text());assert not Path('/proc',str(r['pid'])).exists();rows.append(r)
 prior=2+math.ceil(max(r.get('cumulative_gpu_s',r['elapsed_s']) for r in rows))
 assert prior==({{'wm':2557,'bb':1664}}[a])
 b=json.loads((old/(a+'_BINDING.json')).read_text())
 b.update(prior_gpu_s=prior,checkpoint_path={GPU!r}+'/bb.pt')
 out[a]=b
print(json.dumps(out))
''')
    results = {}
    for host,target in [('cpuQ',CPU),('gpuR',GPU)]:
        results[host] = ssh(host,f'''import base64,hashlib,io,json,shutil,tarfile
from pathlib import Path
root=Path({target!r});assert root.parent.is_dir();assert not root.exists(),'existing continuation; inspect receipts'
root.mkdir();stage=root/'pytools';stage.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({raw!r})),mode='r:gz') as t:t.extractall(stage,filter='data')
sources={sources!r};assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==h for n,h in sources.items())
for n in ['jobs','tasks','rpc','rpc/wm','rpc/bb']:(root/n).mkdir(exist_ok=True)
binding=dict(main_authorized=True,authorization='MAIN20:52',total_sim_s={TOTAL},step0_sim_s=47010,
 retained_superseded_sim_s={SUPERSEDED},per_arm={BUDGET!r},canary_arms=['wm'],design_sha256={design!r},sources=sources)
(root/'BINDING.json').write_text(json.dumps(binding,indent=2));(root/'STAGED.json').write_text(json.dumps(sources,indent=2))
for a,b in {gpu!r}.items():(root/(a+'_BINDING.json')).write_text(json.dumps(b,indent=2))
if {host!r}=='gpuR':
 old=root.parent/'main_v6'
 for n in ['bb.pt','physics.json']:shutil.copyfile(old/n,root/n)
print(json.dumps(dict(root=str(root),binding=binding)))
''')
    results['import'] = ssh('cpuQ',f'''import json,shutil,sys
from pathlib import Path
root=Path({CPU!r});old=Path({CPU_V4!r});sys.path.insert(0,str(root/'pytools'))
from tools.c1switch.continuation import imported_ledger
assert json.loads((old/'CANARY_GATE.json').read_text())['pass_']
rows=[json.loads(x) for x in (old/'accounting.jsonl').read_text().splitlines()]
out=imported_ledger(rows)
shutil.copyfile(old/'accounting.jsonl',root/'PRIOR_V4_ACCOUNTING.jsonl')
(root/'accounting.jsonl').write_text(''.join(json.dumps(x)+'\\n' for x in out))
for p in old.glob('jobs/bb_*/*.json'):
 dest=root/'jobs'/p.parent.name/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
assert len(list(root.glob('jobs/bb_*/CLOSE.json')))==16
print(json.dumps(dict(charged_sim_s=sum(x['sim_s'] for x in out),bb_episodes=1,bb_branches=15,superseded_WM={SUPERSEDED})))
''')
    write(EVIDENCE/'V7_STAGE.json',results)
    print(json.dumps(dict(cpu=CPU,gpu=GPU,source_binding=canonical(sources),imported=results['import']),indent=2))


def launch_worker(host, name, code, *, gpu=False):
    root = GPU if host=='gpuR' else CPU
    env = dict(GPU_ENV,CUDA_VISIBLE_DEVICES='0') if gpu else dict(CPU_ENV,C1SWITCH_ENGINE_ROOT=CPU,C1SWITCH_MAIN_ROOT=CPU)
    result = ssh(host,f'''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path({root!r});stage=root/'pytools';name={name!r}
assert not (root/'STOP').exists()
assert not (root/(name+'_DISPATCH.json')).exists(),'already dispatched'
sources=json.loads((root/'STAGED.json').read_text());assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==h for n,h in sources.items())
env=dict(os.environ);env.update({env!r})
with (root/(name+'.log')).open('x') as log:
 p=subprocess.Popen([sys.executable,'-B','-u','-c',{code!r}],cwd=stage,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
out=dict(pid=p.pid,host={host!r},unix=time.time(),name=name)
(root/(name+'_DISPATCH.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out))
''')
    write(EVIDENCE/('V7_'+name.upper()+'_DISPATCH.json'),result)
    print(json.dumps(result,indent=2))


def gpu(arm):
    # Admission only observes memory; it never signals another process.
    admission = ssh('gpuR',f'''import json,subprocess
from pathlib import Path
r=Path({GPU!r});assert not (r/'rpc'/{arm!r}/'STOP').exists()
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=7168,free
print(json.dumps(dict(free_mib=free)))
''')
    print(json.dumps(admission))
    launch_worker('gpuR',arm+'_gpu',f"from pathlib import Path; from tools.c1switch.rpc import serve; serve(Path({GPU!r})/'rpc'/{arm!r},'tools.c1switch.adapter:create',Path({GPU!r})/({arm!r}+'_BINDING.json'),'cuda')",gpu=True)


def relay():
    root = EVIDENCE/'relay_main_v7'
    assert not root.exists(),'existing relay; inspect owned process'
    root.mkdir()
    env = dict(os.environ,C1SWITCH_CPU_RPC=CPU+'/rpc',C1SWITCH_GPU_RPC=GPU+'/rpc')
    with (root/'relay.log').open('x') as log:
        p = subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch.transport','--root',str(root.resolve())],env=env,stdout=log,stderr=subprocess.STDOUT)
    out = dict(pid=p.pid,root=str(root.resolve()),unix=time.time())
    write(root/'DISPATCH.json',out)
    print(json.dumps(out,indent=2))


def parity_worker(arm):
    # Original arbitrary draw is solely a diagnostic, never a live seed override.
    from .rpc import FilePredictor
    root = Path(GPU)
    try:
        oldroot = Path(GPU_V4)/'rpc'/arm
        p = next(p for p in oldroot.glob('*.request.json') if json.loads(p.read_text())['id'].endswith('_j0'))
        req = json.loads(p.read_text())
        old = json.loads(p.with_name(p.name.replace('.request.','.result.')).read_text())
        predictor = FilePredictor(root/'rpc'/arm,json.loads((root/(arm+'_BINDING.json')).read_text()),timeout_s=600)
        predictor.predict(req['belief'],'parity_v4_v7_'+arm,validation_seed=int(canonical(req['belief'])[:8],16)&0x7fffffff)
        p = next(p for p in (root/'rpc'/arm).glob('*.request.json') if json.loads(p.read_text())['id']=='parity_v4_v7_'+arm)
        got = json.loads(p.with_name(p.name.replace('.request.','.result.')).read_text())
        assert got['samples']==old['samples'],'v6 original-draw GPU parity failed'
        write(root/(arm+'_PARITY.json'),dict(pass_=True,bitwise=True,model=arm,samples=got['samples'],new_sim_s=0))
    except BaseException as exc:
        write(root/'rpc'/arm/'STOP',dict(reason='v7 parity',error=str(exc)))
        raise


def reuse_worker():
    from .observable import anonymous_payload
    from .rpc import FilePredictor
    root = Path(CPU)
    try:
        path = root/'jobs/bb_s969112_episode'
        old = json.loads((path/'RESULT.json').read_text());new = copy.deepcopy(old)
        requests = {}
        for p in (Path(CPU_V4)/'rpc/bb').glob('*.request.json'):
            r = json.loads(p.read_text());requests[int(r['id'].rsplit('_j',1)[1])] = r['belief']
        assert set(requests)==set(range(10))
        predictor = FilePredictor(root/'rpc/bb',json.loads((root/'bb_BINDING.json').read_text()),timeout_s=600)
        out = dict(pass_=True,new_sim_s=0,decisions=[])
        for j in range(10):
            payload = anonymous_payload(requests[j])
            costs = predictor.predict(payload,'bb_reuse_v7_j'+str(j))
            pick = reuse_decision(old['decisions'][j],costs)
            out['decisions'].append(dict(j=j,pick=pick,equal=True))
            new['decisions'][j].update(predicted_costs=costs,belief_sha256=canonical(payload))
            write(root/'BB_REUSE_PROGRESS.json',out)
        new['prior_result_sha256'] = canonical(old)
        new['observable_revalidation'] = 'all10 corrected anonymous BB picks identical; full physics reused'
        write(path/'SUPERSEDED_METADATA.json',old)
        write(path/'RESULT.json',new)
        for d in new['decisions']:
            write(path/('pick_j'+str(d['j'])+'.json'),{k:v for k,v in d.items() if k!='branches'})
            if 'branches' in d:write(path/('measurement_j'+str(d['j'])+'.json'),d)
        write(root/'BB_REUSE.json',out)
    except BaseException as exc:
        write(root/'STOP',dict(reason='BB reuse gate',error=str(exc)))
        raise


def launch(phase):
    if phase=='main':
        gates = ssh('cpuQ',f'''import json
from pathlib import Path
r=Path({CPU!r});print(json.dumps({{n:json.loads((r/n).read_text()) for n in ['BB_REUSE.json','CANARY_GATE.json','PARITY.json']}}))
''')
        assert all(g['pass_'] for g in gates.values())
    launch_worker('cpuQ',phase,"from tools.c1switch.dispatch import main; main("+repr(phase)+")")


def gate():
    parity = ssh('gpuR',f'''import json
from pathlib import Path
r=Path({GPU!r});print(json.dumps({{a:json.loads((r/(a+'_PARITY.json')).read_text()) for a in ('wm','bb')}}))
''')
    assert all(r['pass_'] for r in parity.values())
    result = ssh('cpuQ',f'''import json,sys
from pathlib import Path
root=Path({CPU!r});sys.path.insert(0,str(root/'pytools'))
from tools.c1switch.storage import write
assert not (root/'STOP').exists()
assert json.loads((root/'BB_REUSE.json').read_text())['pass_']
assert (root/'canary/COMPLETE.json').exists()
out={{}}
for a in ('wm','bb'):
 p=root/'jobs'/(a+'_s969112_episode');r=json.loads((p/'RESULT.json').read_text())
 assert len(r['decisions'])==10 and len([d for d in r['decisions'] if 'branches' in d])==3
 assert r['J600']['collisions']==0 and r['J600']['teleports']==0
 assert json.loads((p/'CLOSE.json').read_text())==dict(actual_sim_s=900.,charged_sim_s=900.)
 for d in r['decisions']:
  if 'branches' in d:
   assert set(d['branches'])==set({CONTROLLERS!r})
   for b in d['branches'].values():assert b['prefix']==d['prefix']
 out[a]=dict(schedule=r['schedule'],J600=r['J600'])
charged=sum(json.loads(x)['sim_s'] for x in (root/'accounting.jsonl').read_text().splitlines())
actual=47010+12450+sum(json.loads(p.read_text())['actual_sim_s'] for p in root.glob('jobs/*/CLOSE.json'))
assert charged==actual==84360
result=dict(pass_=True,charged_sim_s=charged,actual_sim_s=actual,results=out,superseded_WM=12450)
write(root/'CANARY_GATE.json',result);write(root/'PARITY.json',dict(pass_=True,models={parity!r}));print(json.dumps(result))
''')
    write(EVIDENCE/'V7_GPU_PARITY.json',dict(pass_=True,models=parity))
    write(EVIDENCE/'V7_CANARY_GATE.json',result)
    print(json.dumps(result,indent=2))


def progress():
    result = {}
    for host,target in [('cpuQ',CPU),('gpuR',GPU)]:
        result[host] = ssh(host,f'''import json,os
from pathlib import Path
r=Path({target!r});out=dict(load=os.getloadavg(),exists=r.exists())
if r.exists():
 out['receipts']={{str(p.relative_to(r)):json.loads(p.read_text()) for n in ['STOP','*_DISPATCH.json','*PARITY.json','BB_REUSE*.json','CANARY_GATE.json','canary/STATUS.json','canary/COMPLETE.json','main/STATUS.json','main/COMPLETE.json','rpc/*/GPU_STARTED.json','rpc/*/GPU_FINISHED.json','rpc/*/STOP'] for p in r.glob(n)}}
 out['own_pids']={{str(v['pid']):Path('/proc',str(v['pid'])).exists() for v in out['receipts'].values() if isinstance(v,dict) and 'pid' in v}}
 if (r/'accounting.jsonl').exists():
  out['charged_sim_s']=sum(json.loads(x)['sim_s'] for x in (r/'accounting.jsonl').read_text().splitlines())
  out['actual_closed_sim_s']=59460+sum(json.loads(p.read_text())['actual_sim_s'] for p in r.glob('jobs/*/CLOSE.json'))
  out['episodes']=[p.parent.name for p in r.glob('jobs/*_episode/RESULT.json')]
  out['branches']=len(list(r.glob('jobs/*_j*/RESULT.json')))
 out['rpc']={{a:dict(requests=len(list((r/'rpc'/a).glob('*.request.json'))),results=len(list((r/'rpc'/a).glob('*.result.json')))) for a in ('wm','bb')}}
print(json.dumps(out))
''')
    write(EVIDENCE/'V7_PROGRESS.json',result)
    display = copy.deepcopy(result)
    for r in display.values():
        for k,v in r.get('receipts',{}).items():
            if k.endswith('PARITY.json'):v.pop('samples',None)
    print(json.dumps(display,indent=2))
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('stage','wm_gpu','bb_gpu','relay','wm_parity','bb_parity','bb_reuse','canary','gate','main','progress'))
    mode = p.parse_args().mode
    if mode in ('wm_gpu','bb_gpu'):gpu(mode[:2])
    elif mode in ('wm_parity','bb_parity'):
        arm = mode[:2]
        launch_worker('gpuR',mode,f'from tools.c1switch.continuation import parity_worker; parity_worker({arm!r})')
    elif mode=='bb_reuse':launch_worker('cpuQ',mode,'from tools.c1switch.continuation import reuse_worker; reuse_worker()')
    elif mode in ('canary','main'):launch(mode)
    else:globals()[mode]()
