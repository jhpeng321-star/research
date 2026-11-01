"""Explicit WMSPEED2 CPU-only staging and receipt retrieval."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from .ops import ssh, stage_frozen

LOCAL = Path(__file__).resolve().parents[2]
READONLY = Path('E:/agent-work/c4-coopnative/docs/paper_results/coopmwm')
ROOT = '/root/coopmwm20261101_wmspeed2'


def launch(tag, compact=False, mode='profile', H=60, K=2, width=4, lines=False, stage=False, fixture=0, host='gpuA', gpu=False, group_size=0, duration=60, warm=False, experiment='actor_graphs'):
    cpu = json.loads((READONLY/'CPU_coopmwm20261101_cpu36.json').read_text())
    if stage:stage_frozen(host, ROOT, cpu)
    files = {name: (LOCAL/'tools/coopmwm'/name).read_text() for name in
             ('speed2.py','speed2_runtime.py','fused.py','plan_wm.py','plan_service.py','test_speed2_cpu.py','plans.py','speed2_memory.py','accel_diagnostic.py','speed2_gpu.py')}
    fixture_text = (READONLY/'VOLUME_INPUTS_1854.json').read_text()
    source = f'''import pathlib,json,hashlib,subprocess,os,sys,time
root=pathlib.Path({ROOT!r});assert root.parent.is_dir()
if {gpu!r}:
 inventory=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 assert not active,'GPU compute owner present: '+active
 assert all(int(row.split(',')[0])<5 for row in inventory.strip().splitlines()),inventory
root.mkdir(exist_ok=True)
snapshot=root
hashes=json.loads({json.dumps(cpu['source_hashes'])!r})
for name,h in hashes.items():
 if not name.startswith('pytools/tools/') and name!='FROZEN_COSTS.json':continue
 p=snapshot/name;assert p.is_file(),str(p)
 data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==h,name
 q=root/name;q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():assert q.read_bytes()==data
 else:q.write_bytes(data)
files=json.loads({json.dumps(files)!r})
run=root/{tag!r};assert not run.exists();run.mkdir()
overlay=run/'pytools/tools/coopmwm';overlay.mkdir(parents=True)
for p in (root/'pytools/tools/coopmwm').glob('*.py'):(overlay/p.name).write_bytes(p.read_bytes())
for name,text in files.items():(overlay/name).write_text(text)
(run/'INPUTS.json').write_text({fixture_text!r})
(run/'FROZEN_COSTS.json').write_bytes((root/'FROZEN_COSTS.json').read_bytes())
manifest=dict(qualified_snapshot_sha256={hashlib.sha256((READONLY/'CPU_coopmwm20261101_cpu36.json').read_bytes()).hexdigest()!r},
 fixture_sha256=hashlib.sha256((run/'INPUTS.json').read_bytes()).hexdigest(),
 overlay_sha256={{name:hashlib.sha256((overlay/name).read_bytes()).hexdigest() for name in files}},
  frozen_sha256=hashes,CUDA_VISIBLE_DEVICES='0' if {gpu!r} else '',new_sumo_sim_s=0)
(run/'MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if {gpu!r} else '',SPEED2_DEVICE='cuda' if {gpu!r} else 'cpu',SPEED2_GROUP_SIZE={str(group_size)!r},SPEED2_DURATION={str(duration)!r},OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(run/'pytools')+':'+str(root/'pytools'))
env['SPEED2_WARM']='1' if {warm!r} else '0'
env['SPEED2_EXPERIMENT']={experiment!r}
with (run/'RUN.log').open('x') as log:
 p=subprocess.Popen([sys.executable,'-B','-u','-m','tools.coopmwm.speed2',{mode!r},'--root',str(run),'--H',{str(H)!r},'--K',{str(K)!r},'--width',{str(width)!r},'--fixture-index',{str(fixture)!r}]+({compact!r} and ['--speed2'] or [])+({lines!r} and ['--lines'] or []),env=env,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
owner=dict(pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],root=str(run),host={host!r},CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'],new_sumo_sim_s=0)
(run/'OWNER.json').write_text(json.dumps(owner));print(json.dumps(owner))
'''
    print(ssh.remote(host, source, timeout=45))


def status(tag, host='gpuA'):
    print(ssh.remote(host, f'''import pathlib,json
r=pathlib.Path({ROOT!r})/{tag!r}
print((r/'RUN.log').read_text()[-16000:])
if (r/'BASELINE.json').exists():print((r/'BASELINE.json').read_text())
if (r/'PARITY.json').exists():
 d=json.loads((r/'PARITY.json').read_text());print(json.dumps({{k:v for k,v in d.items() if k not in ('reference','optimized')}}))
if (r/'TESTS.json').exists():print((r/'TESTS.json').read_text())
if (r/'QUALIFICATION.json').exists():print((r/'QUALIFICATION.json').read_text())
''', timeout=30))


def wait(tags, seconds):
    print(ssh.remote('gpuA', f'''import pathlib,json,time
root=pathlib.Path({ROOT!r});pending=set({tags!r});deadline=time.monotonic()+{seconds!r}
while pending and time.monotonic()<deadline:
 for tag in sorted(pending):
  r=root/tag
  if (r/'PARITY.json').exists():
   d=json.loads((r/'PARITY.json').read_text())
   print(json.dumps(dict(tag=tag,**{{k:v for k,v in d.items() if k not in ('reference','optimized')}})),flush=True)
   pending.remove(tag)
  elif (r/'FAILED.json').exists() or 'Traceback (most recent call last)' in (r/'RUN.log').read_text():
   raise RuntimeError(tag+': '+(r/'RUN.log').read_text()[-4000:])
  else:
   owner=json.loads((r/'OWNER.json').read_text());p=pathlib.Path('/proc')/str(owner['pid'])/'stat'
   if not p.exists():raise RuntimeError(tag+' exited without PARITY.json: '+(r/'RUN.log').read_text()[:2000])
   stat=p.read_text().split();assert stat[21]==owner['start_ticks'],'owned PID identity changed'
   if stat[2]=='Z':raise RuntimeError(tag+' ended without PARITY.json')
 if pending:time.sleep(10)
print(json.dumps(dict(pending=sorted(pending))))
''', timeout=seconds+30))


def collect(tag, host):
    from .storage import write
    value=json.loads(ssh.remote(host,f'''import pathlib,json
r=pathlib.Path({ROOT!r})/{tag!r};out={{'host':{host!r},'tag':{tag!r},'root':str(r)}}
for name in ('OWNER','ENVIRONMENT','PARITY','BASELINE','TESTS','QUALIFICATION','GPU_PROFILE','EXPERIMENT','FAILED'):
 p=r/(name+'.json')
 if p.exists():out[name]=json.loads(p.read_text())
m=json.loads((r/'MANIFEST.json').read_text()) if (r/'MANIFEST.json').exists() else {{'legacy_run_no_manifest':True}}
out['manifest']={{k:v for k,v in m.items() if k!='frozen_sha256'}}
out['log_tail']=(r/'RUN.log').read_text()[-2000:]
o=out['OWNER'];p=pathlib.Path('/proc')/str(o['pid'])/'stat'
out['owned_process_alive']=p.exists() and p.read_text().split()[21]==o['start_ticks'] and p.read_text().split()[2]!='Z'
print(json.dumps(out))
''',timeout=30))
    target=LOCAL/'docs/paper_results/coopmwm'/f'WMSPEED2_{host}_{tag}.json'
    if target.exists() and json.loads(target.read_text())!=value:
        i=1
        while target.with_name(target.stem+f'_snapshot{i}.json').exists():i+=1
        target=target.with_name(target.stem+f'_snapshot{i}.json')
    if not target.exists():write(target,value)
    print(json.dumps(dict(receipt=str(target),alive=value['owned_process_alive'],
        result={k:v for k,v in value.get('PARITY',value.get('EXPERIMENT',{})).items() if k not in ('reference','optimized')},
        failure=value.get('FAILED',{}).get('traceback','')[-1800:])))


def stage_release(tag,host):
    """Stage only; bindings, worker admission and broker swap remain owner actions."""
    import subprocess
    subprocess.run(['git','diff','--exit-code','--','tools/coopmwm'],cwd=LOCAL,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=LOCAL,text=True).strip()
    cpu=json.loads((READONLY/'CPU_coopmwm20261101_cpu36.json').read_text())
    names=('fused.py','plan_wm.py','plan_service.py','plans.py','accel_diagnostic.py',
           'speed2_runtime.py','speed2_memory.py','speed2.py','test_speed2_cpu.py')
    files={name:(LOCAL/'tools/coopmwm'/name).read_text() for name in names}
    inputs=(READONLY/'VOLUME_INPUTS_1854.json').read_text()
    value=json.loads(ssh.remote(host,f'''import pathlib,json,hashlib
base=pathlib.Path({ROOT!r});assert base.is_dir()
root=base/('release_'+{tag!r});assert not root.exists(),'immutable release already exists'
hashes=json.loads({json.dumps(cpu['source_hashes'])!r})
selected={{}}
for name,h in hashes.items():
 if name!='FROZEN_COSTS.json' and not name.startswith('pytools/tools/'):continue
 p=base/name;data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==h,name
 selected[name]=data
for name,text in json.loads({json.dumps(files)!r}).items():selected['pytools/tools/coopmwm/'+name]=text.encode()
selected['INPUTS.json']={inputs!r}.encode()
root.mkdir()
for name,data in selected.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
manifest=dict(root=str(root),source_commit={commit!r},qualified_base='CPU36',
 hashes={{name:hashlib.sha256(data).hexdigest() for name,data in selected.items()}},
 staged_only=True,new_sumo_sim_s=0)
text=json.dumps(manifest,sort_keys=True,indent=2)
(root/'SOURCE_MANIFEST.json').write_text(text)
print(json.dumps(dict(manifest=manifest,manifest_sha256=hashlib.sha256(text.encode()).hexdigest())))
''',timeout=45))
    target=LOCAL/'docs/paper_results/coopmwm'/f'WMSPEED2_{host}_release_{tag}.json'
    from .storage import write
    write(target,value)
    print(json.dumps(dict(receipt=str(target),root=value['manifest']['root'],manifest_sha256=value['manifest_sha256'])))


def trace_summary(tag,host):
    value=json.loads(ssh.remote(host,f'''import pathlib,json,collections,hashlib
r=pathlib.Path({ROOT!r})/{tag!r};p=r/'GPU_TRACE.json'
events=json.loads(p.read_text())['traceEvents'];kernels=collections.Counter();copies=collections.Counter()
for e in events:
 if e.get('ph')!='X':continue
 if e.get('cat')=='kernel':kernels[e['name']]+=e['dur']
 if e.get('cat')=='gpu_memcpy':copies[e['name']]+=e['dur']
total=sum(kernels.values())
print(json.dumps(dict(host={host!r},tag={tag!r},trace_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
 kernel_total_ms=total/1000,kernel_top=[dict(name=k,ms=v/1000,percent=100*v/total) for k,v in kernels.most_common(20)],
 copies_ms={{k:v/1000 for k,v in copies.items()}},new_sumo_sim_s=0)))
''',timeout=30))
    from .storage import write
    target=LOCAL/'docs/paper_results/coopmwm'/f'WMSPEED2_{host}_{tag}_kernels.json'
    write(target,value)
    print(json.dumps(value,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('launch', 'status','wait','collect','stage-release','trace-summary'))
    p.add_argument('tag')
    p.add_argument('--compact', action='store_true')
    p.add_argument('--job-mode',choices=('profile','parity','tests','qualification','gpu-profile','gpu-experiment','table-test'),default='profile')
    p.add_argument('--H',type=int,default=60)
    p.add_argument('--K',type=int,default=2)
    p.add_argument('--width',type=int,default=4)
    p.add_argument('--lines',action='store_true')
    p.add_argument('--stage',action='store_true',help='stage the frozen CPU36 snapshot on first use')
    p.add_argument('--fixture-index',type=int,choices=range(4),default=0)
    p.add_argument('--wait-seconds',type=int,default=600)
    p.add_argument('--host',choices=('gpuA','gpuR'),default='gpuA')
    p.add_argument('--gpu',action='store_true')
    p.add_argument('--group-size',type=int,choices=(0,2,3,4),default=0)
    p.add_argument('--duration',type=int,choices=(60,120,300),default=60)
    p.add_argument('--warm',action='store_true')
    p.add_argument('--experiment',choices=('actor_graphs','tables','wide_batch','births','births_cpu','transfers'),default='actor_graphs')
    a = p.parse_args()
    assert all(re.fullmatch('[A-Za-z0-9_-]+',tag) for tag in a.tag.split(','))
    if a.mode == 'launch':launch(a.tag, a.compact, a.job_mode, a.H,a.K,a.width,a.lines,a.stage,a.fixture_index,a.host,a.gpu,a.group_size,a.duration,a.warm,a.experiment)
    elif a.mode == 'status':status(a.tag,a.host)
    elif a.mode == 'collect':
        for tag in a.tag.split(','):collect(tag,a.host)
    elif a.mode == 'stage-release':stage_release(a.tag,a.host)
    elif a.mode == 'trace-summary':trace_summary(a.tag,a.host)
    else:wait(a.tag.split(','),a.wait_seconds)
