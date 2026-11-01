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


def launch(tag, compact=False, mode='profile', H=60, K=2, width=4, lines=False, stage=False, fixture=0):
    cpu = json.loads((READONLY/'CPU_coopmwm20261101_cpu36.json').read_text())
    if stage:stage_frozen('gpuA', ROOT, cpu)
    files = {name: (LOCAL/'tools/coopmwm'/name).read_text() for name in
             ('speed2.py','speed2_runtime.py','fused.py','plan_wm.py','plan_service.py','test_speed2_cpu.py')}
    fixture_text = (READONLY/'VOLUME_INPUTS_1854.json').read_text()
    source = f'''import pathlib,json,hashlib,subprocess,os,sys,time
root=pathlib.Path({ROOT!r});assert root.parent.is_dir()
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
 frozen_sha256=hashes,CUDA_VISIBLE_DEVICES='',new_sumo_sim_s=0)
(run/'MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(run/'pytools')+':'+str(root/'pytools'))
with (run/'RUN.log').open('x') as log:
 p=subprocess.Popen([sys.executable,'-B','-u','-m','tools.coopmwm.speed2',{mode!r},'--root',str(run),'--H',{str(H)!r},'--K',{str(K)!r},'--width',{str(width)!r},'--fixture-index',{str(fixture)!r}]+({compact!r} and ['--speed2'] or [])+({lines!r} and ['--lines'] or []),env=env,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
owner=dict(pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],root=str(run),CUDA_VISIBLE_DEVICES='',new_sumo_sim_s=0)
(run/'OWNER.json').write_text(json.dumps(owner));print(json.dumps(owner))
'''
    print(ssh.remote('gpuA', source, timeout=45))


def status(tag):
    print(ssh.remote('gpuA', f'''import pathlib,json
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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('launch', 'status','wait'))
    p.add_argument('tag')
    p.add_argument('--compact', action='store_true')
    p.add_argument('--job-mode',choices=('profile','parity','tests','qualification'),default='profile')
    p.add_argument('--H',type=int,default=60)
    p.add_argument('--K',type=int,default=2)
    p.add_argument('--width',type=int,default=4)
    p.add_argument('--lines',action='store_true')
    p.add_argument('--stage',action='store_true',help='stage the frozen CPU36 snapshot on first use')
    p.add_argument('--fixture-index',type=int,choices=range(4),default=0)
    p.add_argument('--wait-seconds',type=int,default=600)
    a = p.parse_args()
    assert all(re.fullmatch('[A-Za-z0-9_-]+',tag) for tag in a.tag.split(','))
    if a.mode == 'launch':launch(a.tag, a.compact, a.job_mode, a.H,a.K,a.width,a.lines,a.stage,a.fixture_index)
    elif a.mode == 'status':status(a.tag)
    else:wait(a.tag.split(','),a.wait_seconds)
