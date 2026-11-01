"""Owner18:54 guidance-volume amendment; explicit owned recovery operations."""
import argparse
import json
from . import ops, deploy
from .storage import write


def hold():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import os,signal,pathlib,json,time
r=pathlib.Path({deploy.ROOT!r});assert r.is_dir()
d=json.loads((r/'TRAIN_DISPATCH_1742.json').read_text());p=pathlib.Path('/proc')/str(d['pid'])
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
assert (p/'stat').read_text().split()[21]==d['start_ticks']
assert 'tools.coopmwm.runner' in (p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
receipt=r/'HOLD_1854.json'
if receipt.exists():
 v=json.loads(receipt.read_text());assert (p/'stat').read_text().split()[2]=='T'
else:
 os.kill(d['pid'],signal.SIGSTOP)
 charges=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
 incomplete=[x['id'] for x in charges if not all((r/'jobs'/x['id']/f).exists() for f in ('RESULT.json','CLOSE.json'))]
 v=dict(owner=d,unix=time.time(),reason='OWNER18:54 restart B/R at MAPPO update0 after batching and execution gating qualification',
  arms_held=['B','R'],shared_supervisor_paused=True,incomplete_jobs=incomplete,
  charged_sim_s=sum(x['sim_s'] for x in charges),jobs=len(charges),new_sumo_sim_s=0,
  existing_updates=sorted(x.name for x in r.glob('S*_UPDATE_*.json')))
 with receipt.open('x') as f:json.dump(v,f,sort_keys=True)
print(json.dumps(v))
''',timeout=25))
    write(ops.EVIDENCE/'HOLD_1854.json',value)
    print(json.dumps(value))


def inspect_runtime():
    print(ops.ssh.remote('gpuA', '''import pathlib
r=pathlib.Path('/root/rbeval20261031/phys_code_22cc32f65a80/tools/rbint')
for p in (r/'departure.py',r/'recipe.py',r.parent.parent/'src/wms/model/sumo_departure.py'):
 print(str(p));print(p.read_text())
''', timeout=25))


def bench_inputs():
    from .protocol import REGISTRY,checked,digest
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({deploy.ROOT!r});sys.path.insert(0,'/home/ubuntu/coopmwm20261101_v3/pytools')
from tools.coopmwm.wm import proposal_candidates
from tools.coopmwm.protocol import digest
records=[]
for p in sorted((r/'jobs').glob('s0_u001_B_*/RESULT.json')):
 e=json.loads(p.read_text())
 for s in e['steps']:
  if len(s['group'])!=2:continue
  v=dict(world_id=e['world']['seed'],time_s=s['time_s'],rows=s['rows'],targets=s['targets'],group=s['group'],mode='proposal',setting=0)
  v['state_id']=digest(dict(volume_qualification=v))
  if len(proposal_candidates(v))>2:records.append(v)
  if len(records)==4:break
 if len(records)==4:break
assert len(records)==4
print(json.dumps(dict(records=records,selection='first four chronological available windows in original TRAIN B-u1; no outcome selection',new_sumo_sim_s=0)))
''',timeout=35))
    write(ops.EVIDENCE/'VOLUME_INPUTS_1854.json',value)
    line=('\nS-COOPMWM 18:54 BATCH QUALIFICATION REGISTERED: owner guidance-volume amendment; '
          'B/R held, old jobs/charges preserved; no physics. Reuse inherited FULL-v4 futures/executors; '
          'fuse 2/4 observable windows x K8 and K4, once-per-worker single-future parity <=1e-3 veh-s; '
          'both GPUs, max65% memory/process, four preselected TRAIN geometries, two-hour infrastructure cap; '
          'retain failures; choose K4 only if aggregate measured capacity<300queries/hour at K8; '
          'report K8/K4 choice+acceptance agreement; input_sha256='+digest(value)+'.\n')
    reg=checked(REGISTRY)
    with reg.open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'VOLUME_BENCH_REGISTRATION_1854.json',dict(line=line,input_sha256=digest(value),new_sumo_sim_s=0))
    print(json.dumps(dict(registered=True,windows=len(value['records']),input_sha256=digest(value))))


def bench_launch(host):
    from .storage import read
    root='/root/coopmwm20261101_volumeprobe6'
    assert not (ops.EVIDENCE/f'VOLUME_PROBE6_{host}_DISPATCH.json').exists()
    files=ops.sources()
    files['FROZEN_COSTS.json']=ops.ssh.remote('gpuA',"import pathlib;print(pathlib.Path('/root/coopmwm20261101_dual2/FROZEN_COSTS.json').read_text())",timeout=20).strip()
    files['INPUTS.json']=json.dumps(read(ops.EVIDENCE/'VOLUME_INPUTS_1854.json'))
    files['CAPACITY_REFERENCE.json']=ops.ssh.remote(host,"import pathlib;print(pathlib.Path('/root/coopmwm20261101_volumeprobe2/FUSED_2_K8.json').read_text())",timeout=25).strip()
    ops.stage(host,root,files)
    value=json.loads(ops.ssh.remote(host,f'''import os,pathlib,json,subprocess,sys,time
r=pathlib.Path({root!r});assert r.is_dir();assert not (r/'DISPATCH.json').exists()
gpu=subprocess.check_output(['nvidia-smi'],text=True)
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>16000
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (r/'PROBE.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.volume_probe',str(r)],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
v=dict(pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),host={host!r},unix=time.time(),nvidia_smi=gpu,new_sumo_sim_s=0)
with (r/'DISPATCH.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))
''',timeout=30))
    write(ops.EVIDENCE/f'VOLUME_PROBE6_{host}_DISPATCH.json',value);print(json.dumps(value))


def capacity_register():
    from .protocol import REGISTRY,checked
    line=('\nS-COOPMWM 18:54 BATCH CAPACITY ENGINEERING REGISTERED: H300 proposal preparation '
          'restores inherited WMTEACH H300/960 entrant capacity instead of COOPMWM H600/1600; '
          'H600 diagnostics retain1600. Four-window1600 OOM on gpuR preserved; '
          'prospective960 qualification uses same four registered TRAIN inputs and original2-windowK8 futures '
          'as parity reference (<=1e-3veh-s), then4windowsK4/K8. No SUMO, no training or outcome-based selection; '
          'max65%memory, original service deadline retained.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'VOLUME_CAPACITY_REGISTRATION_1854.json',dict(line=line,new_sumo_sim_s=0))
    print(line.strip())


def trim_register():
    from .protocol import REGISTRY,checked
    line=('\nS-COOPMWM 18:54 MASKED-SLOT QUALIFICATION REGISTERED:960-slot source-capacity candidate '
          'FAILED parity (max3.50637veh-s), rejected before training; retain original1600 source/hidden draws. '
          'Next engineering probe trims only universally absent TRAILING slots from each fused model.rollout(n_steps=1), '
          'then restores the original full slot state/history BEFORE the inherited insertion/observer; '
          'no roster reordering, source capacity, RNG, controller, model or grading change. '
          'Same registered2/4-windowK8/K4 inputs and frozen1600 K8 reference; <=1e-3veh-s parity required. '
          'No physics; per-worker parity once; original GPU deadline/cap retained.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'VOLUME_TRIM_REGISTRATION_1854.json',dict(line=line,new_sumo_sim_s=0));print(line.strip())


def quiesce():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,os,signal,time
r=pathlib.Path({deploy.ROOT!r});assert r.is_dir()
d=json.loads((r/'TRAIN_DISPATCH_1742.json').read_text());p=pathlib.Path('/proc')/str(d['pid'])
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
assert (p/'stat').read_text().split()[21]==d['start_ticks']
assert (p/'stat').read_text().split()[2]=='T' and (r/'HOLD_1854.json').exists()
assert not (r/'QUIESCE_1854.json').exists()
charges=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
incomplete=[x['id'] for x in charges if not all((r/'jobs'/x['id']/f).exists() for f in ('RESULT.json','CLOSE.json'))]
assert all(x.startswith('s0_dev004_A_') and not (r/'jobs'/x/'CLAIM.json').exists() for x in incomplete),('active physical jobs block adoption',incomplete)
children=[]
for child in pathlib.Path('/proc').iterdir():
 if not child.name.isdigit():continue
 try:
  stat=(child/'stat').read_text().split()
  if int(stat[3])!=d['pid']:continue
  cmd=(child/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
  assert 'multiprocessing' in cmd,cmd
  children.append(dict(pid=int(child.name),start_ticks=stat[21],cmd=cmd))
 except FileNotFoundError:pass
v=dict(owner=d,children=children,unix=time.time(),jobs=len(charges),charged_sim_s=sum(x['sim_s'] for x in charges),unstarted_precharged_jobs=incomplete,
 reason='adopt closed A jobs/checkpoints; keep B-u1..3 as superseded engineering; new namespace required',new_sumo_sim_s=0)
with (r/'QUIESCE_1854.json').open('x') as f:json.dump(v,f,sort_keys=True)
os.kill(d['pid'],signal.SIGTERM);os.kill(d['pid'],signal.SIGCONT)
for c in children:
 cp=pathlib.Path('/proc')/str(c['pid'])
 if (cp/'stat').exists() and (cp/'stat').read_text().split()[21]==c['start_ticks']:os.kill(c['pid'],signal.SIGTERM)
time.sleep(.5)
assert not (p/'stat').exists() or (p/'stat').read_text().split()[2]=='Z'
print(json.dumps(v))
''',timeout=25))
    write(ops.EVIDENCE/'QUIESCE_1854.json',value);print(json.dumps(value))


def register_A(cpu_name):
    from .storage import read
    from .protocol import budget,sha,REGISTRY,checked
    import time
    cpu=read(ops.EVIDENCE/cpu_name);assert cpu['pass_'] and not cpu['skips'] and not cpu['sim_s']
    value=dict(owner='18:54',unix=time.time(),scope='A continuation only; B/R remain held pending volume qualification',
        budget=budget(),allocated_sim_s=sum(budget()['phases'].values()),
        preserved='all A checkpoints/finished episodes; old B-u1..3 superseded engineering,86400sim-s retained; unstarted Adev jobs adopted without recharge',
        quiesce=read(ops.EVIDENCE/'QUIESCE_1854.json'),cpu_sha256=sha(ops.EVIDENCE/cpu_name),
        design_sha256=sha(ops.LOCAL/'docs/paper_results/COOPMWM_DESIGN.md'),
        final_rule='fixed primaryA/Bupdate24 then once-only989600..989635; R/seed2 follow; no selection',
        release_guided=False,new_sumo_sim_s=0)
    dest=ops.EVIDENCE/'AMENDMENT_1854_A.json';write(dest,value)
    h=sha(dest);line=('\nS-COOPMWM 18:54 A-ONLY CONTINUATION REGISTERED: amendment_sha256='+h+
        '; allocated5,005,200/cap6,000,000/unallocated994,800sim-s incl supersededB86,400; '
        'A adopts completed updates/episodes and only never-CLAIMed precharged Adev jobs; no repeats/recharge. '
        'B/R HELD, new u0 namespaces only after measured batched GPU qualification and numeric quota/sampling registration; '
        'primary-first/checkpoint/test/seed2 rules unchanged. CPU='+cpu_name+' design_sha256='+value['design_sha256']+'.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'REGISTRATION_1854_A.json',dict(line=line,amendment_sha256=h))
    print(json.dumps(dict(registered=True,amendment_sha256=h,allocated_sim_s=value['allocated_sim_s'])))


def bind_A(cpu_name):
    from .storage import read
    from .protocol import budget,REGISTRY,checked,sha,canonical
    import hashlib
    cpu=read(ops.EVIDENCE/cpu_name);assert cpu['pass_'] and not cpu['skips'] and not cpu['sim_s']
    source='/home/ubuntu/coopmwm20261101_volumeA1'
    staged=json.loads(ops.stage_frozen('cpuQ',source,cpu))
    registry=checked(REGISTRY).read_text(encoding='utf-8');am=read(ops.EVIDENCE/'AMENDMENT_1854_A.json')
    h=hashlib.sha256((canonical(am)+'\n').encode()).hexdigest()
    if h not in registry:
        line='\nS-COOPMWM 18:54 A-ONLY REGISTRATION BYTE-ENCODING CORRIGENDUM: normalized Linux LF amendment_sha256='+h+'; same JSON plan/budget, prior Windows-CRLF hash retained; before physics.\n'
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
        write(ops.EVIDENCE/'REGISTRATION_1854_A_ENCODING.json',dict(line=line,amendment_sha256=h,new_sumo_sim_s=0))
        registry+=line
    v=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib
r=pathlib.Path({deploy.ROOT!r});s=pathlib.Path({source!r});sys.path[:0]=[str(s/'pytools'),'/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from tools.coopmwm.storage import read,write
assert (r/'QUIESCE_1854.json').exists()
b=read(r/'BINDING_1742.json');b.update(budget=json.loads({json.dumps(budget())!r}),cpu=json.loads({json.dumps(cpu)!r}),
 volume_amendment=True,volume=dict(release_guided=False),amendment_sha256={h!r},amendment_file='AMENDMENT_1854_A.json',authority_file='AUTHORITY_1854_A.md',
 unstarted_precharged_jobs=read(r/'QUIESCE_1854.json')['unstarted_precharged_jobs'])
for name,h in json.loads({json.dumps(staged['hashes'])!r}).items():b['hashes']['../'+s.name+'/'+name]=h
am=json.loads({json.dumps(am)!r})
if (r/'AMENDMENT_1854_A.json').exists():assert read(r/'AMENDMENT_1854_A.json')==am
else:write(r/'AMENDMENT_1854_A.json',am)
assert hashlib.sha256((r/'AMENDMENT_1854_A.json').read_bytes()).hexdigest()=={h!r}
with (r/'AUTHORITY_1854_A.md').open('x') as f:f.write({ascii(registry)})
write(r/'BINDING_1854_A.json',b)
from tools.coopmwm.runner import verify
verify(r)
print(json.dumps(dict(bound=True,root=str(r),source_root=str(s),binding_sha256=hashlib.sha256((r/'BINDING_1854_A.json').read_bytes()).hexdigest(),new_sumo_sim_s=0)))
''',timeout=45))
    write(ops.EVIDENCE/'BINDING_1854_A.json',v);print(json.dumps(v))


def launch_A():
    from .storage import read
    d=read(ops.EVIDENCE/'BINDING_1854_A.json');source=d['source_root']
    v=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,os,sys,subprocess,json,fcntl,time
r=pathlib.Path({deploy.ROOT!r});s=pathlib.Path({source!r});assert r.is_dir() and s.is_dir()
sys.path[:0]=[str(s/'pytools'),'/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from tools.coopmwm.runner import verify
from tools.coopmwm.storage import read,write
verify(r);assert not (r/'A_DISPATCH_1854.json').exists()
old=read(r/'TRAIN_DISPATCH_1742.json');p=pathlib.Path('/proc')/str(old['pid'])
assert not (p/'stat').exists() or (p/'stat').read_text().split()[2]=='Z'
claim=(r/'SUPERVISOR.claim').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB);claim.close()
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
with (r/'A_1854.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.volume_runner',str(r),'--only-A'],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
v=dict(host='cpuQ',pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_root=str(s),unix=time.time(),scope='A only; B/R held')
write(r/'A_DISPATCH_1854.json',v);print(json.dumps(v))
''',timeout=30))
    write(ops.EVIDENCE/'A_DISPATCH_1854.json',v);print(json.dumps(v))


def bench_status(host):
    from .storage import read
    tag=next(t for t in ('VOLUME_PROBE6','VOLUME_PROBE5','VOLUME_PROBE4','VOLUME_PROBE3','VOLUME_PROBE2','VOLUME_PROBE1') if (ops.EVIDENCE/f'{t}_{host}_DISPATCH.json').exists())
    d=read(ops.EVIDENCE/f'{tag}_{host}_DISPATCH.json')
    v=json.loads(ops.ssh.remote(host,f'''import pathlib,json
r=pathlib.Path({d['root']!r});out={{n:json.loads((r/n).read_text()) for n in ('COMPLETE.json','FAILED.json','TIMING_2_K8.json','TIMING_4_K8.json','TIMING_4_K4.json') if (r/n).exists()}}
out['log']=(r/'PROBE.log').read_text()[-12000:]
import subprocess
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True)
print(json.dumps(out))
''',timeout=25))
    if 'COMPLETE.json' in v or 'FAILED.json' in v:write(ops.EVIDENCE/f'{tag}_{host}_RESULT.json',v)
    print(json.dumps(v))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('hold','inspect','bench-inputs','bench-launch','bench-status','capacity-register','trim-register','quiesce','register-A','bind-A','launch-A'));p.add_argument('--host',default='gpuA');p.add_argument('--cpu',default='CPU_coopmwm20261101_cpu25.json');a=p.parse_args()
    if a.mode=='hold':hold()
    elif a.mode=='inspect':inspect_runtime()
    elif a.mode=='bench-inputs':bench_inputs()
    elif a.mode=='capacity-register':capacity_register()
    elif a.mode=='trim-register':trim_register()
    elif a.mode=='quiesce':quiesce()
    elif a.mode=='register-A':register_A(a.cpu)
    elif a.mode=='bind-A':bind_A(a.cpu)
    elif a.mode=='launch-A':launch_A()
    elif a.mode=='bench-launch':bench_launch(a.host)
    else:bench_status(a.host)
