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
    root='/root/coopmwm20261101_volumeprobe1'
    assert not (ops.EVIDENCE/f'VOLUME_PROBE1_{host}_DISPATCH.json').exists()
    files=ops.sources()
    files['FROZEN_COSTS.json']=ops.ssh.remote('gpuA',"import pathlib;print(pathlib.Path('/root/coopmwm20261101_dual2/FROZEN_COSTS.json').read_text())",timeout=20).strip()
    files['INPUTS.json']=json.dumps(read(ops.EVIDENCE/'VOLUME_INPUTS_1854.json'))
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
    write(ops.EVIDENCE/f'VOLUME_PROBE1_{host}_DISPATCH.json',value);print(json.dumps(value))


def bench_status(host):
    from .storage import read
    d=read(ops.EVIDENCE/f'VOLUME_PROBE1_{host}_DISPATCH.json')
    v=json.loads(ops.ssh.remote(host,f'''import pathlib,json
r=pathlib.Path({d['root']!r});out={{n:json.loads((r/n).read_text()) for n in ('COMPLETE.json','FAILED.json','TIMING_2_K8.json','TIMING_4_K8.json','TIMING_4_K4.json') if (r/n).exists()}}
out['log']=(r/'PROBE.log').read_text()[-12000:]
print(json.dumps(out))
''',timeout=25))
    if 'COMPLETE.json' in v or 'FAILED.json' in v:write(ops.EVIDENCE/f'VOLUME_PROBE1_{host}_RESULT.json',v)
    print(json.dumps(v))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('hold','inspect','bench-inputs','bench-launch','bench-status'));p.add_argument('--host',default='gpuA');a=p.parse_args()
    if a.mode=='hold':hold()
    elif a.mode=='inspect':inspect_runtime()
    elif a.mode=='bench-inputs':bench_inputs()
    elif a.mode=='bench-launch':bench_launch(a.host)
    else:bench_status(a.host)
