"""Outcome-blind Linux qualification and registration for MAIN20:33 B-REPLAY."""
import argparse
import base64
import json
import zlib
from . import ops
from .protocol import checked
from .storage import read,write
BASE='/home/ubuntu/coopmwm20261103_nichenullcpu7'

SOURCE='/home/ubuntu/coopmwm20261103_replayreleasecpu6'


def release_status():
    """No jobs: verify delegated readiness and that its commit was integrated."""
    import subprocess
    p=ops.LOCAL.parent/'c4-harden-int/HARDEN_INT_READY.json'
    if not p.exists():
        value=dict(ready=False,reason='delegated HARDEN_INT_READY.json absent',
            qualification_dispatched=False,new_sumo_sim_s=0)
    else:
        ready=read(p)
        commit=ready['commit']
        integrated=subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],
            cwd=ops.LOCAL,check=False).returncode==0
        value=dict(ready=True,release=ready,integrated=integrated,
            qualification_dispatched=False,new_sumo_sim_s=0)
    return value


def query_diagnosis():
    from .ls_collect_ops import COLLECTION,ROOT
    print(ops.ssh.remote('cpuQ',f'''import pathlib,json,time
c=pathlib.Path({COLLECTION!r});r=pathlib.Path({ROOT!r});out=[]
for p in (c/'broker_assignments').glob('*.json'):
 a=json.loads(p.read_text());q=json.loads((c/'requests'/p.name).read_text())
 out.append(dict(key=p.stem,slot=a['slot'],world_id=q['world_id'],time_s=q['time_s'],group_size=len(q['group']),has_response=(r/'responses'/p.name).exists()))
print(json.dumps(dict(unix=time.time(),assigned=out)))''',timeout=30))


def qualify():
    release=release_status()
    if not release['ready'] or not release['integrated']:
        print(json.dumps(release));return
    # Complete current package: selective overlays previously omitted late imports.
    files={p.stem:checked(p).read_text(encoding='utf-8')
        for p in (ops.LOCAL/'tools/coopmwm').glob('*.py')}
    for name, text in files.items():
        compile(text, name+'.py', 'exec')
    payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    # cpuQ's older plant package lacks these qualified WM-only command helpers.
    # Copy just Python source from the pinned FULL-v4 producer into this isolated
    # CPU snapshot; never overwrite either runtime or serving GPU files.
    aux=json.loads(ops.ssh.remote('gpuA', '''from pathlib import Path
import json,os,re
r=Path('/root/rbeval20261031/phys_code_22cc32f65a80/src')
forbidden=re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vehicle_params|driver_params',re.I)
files={}
for parent,dirs,names in os.walk(r/'wms'):
 dirs[:]=[d for d in dirs if not forbidden.search(str(Path(parent)/d)) and d!='__pycache__']
 for n in names:
  p=Path(parent)/n
  if n.endswith('.py') and not forbidden.search(str(p)):files[str(p.relative_to(r))]=p.read_text()
assert 'wms/control/alwayscmd.py' in files and 'wms/control/bridge.py' in files
print(json.dumps(files))''',timeout=30))
    aux_payload=base64.b64encode(zlib.compress(json.dumps(aux).encode())).decode()
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,os,time,shutil,hashlib,subprocess,zlib,base64
r=pathlib.Path({SOURCE!r});base=pathlib.Path({BASE!r});assert r.parent.is_dir()
files=json.loads(zlib.decompress(base64.b64decode({payload!r})))
if not r.exists():
 r.mkdir()
 shutil.copytree(base/'pytools',r/'pytools',ignore=shutil.ignore_patterns('__pycache__'))
 for name in ('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py','FROZEN_GEOMETRY_SOURCE.py','FROZEN_NETWORK.net.xml.gz','FROZEN_PRE_NICHE_SEARCH.py'):
  shutil.copy2(base/name,r/name)
 for n,text in files.items():(r/'pytools/tools/coopmwm'/(n+'.py')).write_text(text)
 aux=json.loads(zlib.decompress(base64.b64decode({aux_payload!r})))
 for n,text in aux.items():
  p=r/'WM_PACKAGE'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 # Operator payload tests read repository-relative pure geometry/network files.
 # Link only the existing plant source package, preserving its import semantics.
 plant=pathlib.Path('/home/ubuntu/e3final/runtime/traffic_shepherd');assert plant.is_dir()
 (r/'pytools/traffic_shepherd').symlink_to(plant,target_is_directory=True)
 net=r/'pytools/assets/research/highway_merge_shepherd_long/osm.net.xml.gz'
 net.parent.mkdir(parents=True);shutil.copy2(r/'FROZEN_NETWORK.net.xml.gz',net)
 hashes={{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'pytools').rglob('*.py')}}
 for p in (r/'WM_PACKAGE').rglob('*.py'):hashes[str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest()
 for n in ('__init__.py','c4/__init__.py','c4/geometry.py'):
  p=r/'pytools/traffic_shepherd'/n;hashes[str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest()
 hashes[str(net.relative_to(r))]=hashlib.sha256(net.read_bytes()).hexdigest()
 with (r/'LS_SOURCE_HASHES.json').open('w') as f:json.dump(hashes,f)
 tests=[n for n in files if n.startswith('test_') and n.endswith('cpu')]
 with (r/'LS_RELEASE_TESTS.json').open('w') as f:json.dump(sorted(set(tests)),f)
 # Seven delegated integration rehearsals are already receipted. Do not repeat
 # their subprocess jobs in this physics-forbidden CPU qualification.
 with (r/'LS_CPU_SELECTION.json').open('x') as f:json.dump(dict(pytest_args=['-k','not RehearsalSmoke'],reason='delegated seven-scenario rehearsals adopted from HARDEN_INT_READY'),f)
 with (r/'DELEGATED_RELEASE.json').open('x') as f:json.dump({release!r},f)
for n,text in files.items():assert (r/'pytools/tools/coopmwm'/(n+'.py')).read_text()==text,n
assert json.loads((r/'DELEGATED_RELEASE.json').read_text())=={release!r}
done=r/'LS_CPU_RESULT.json';dispatch=r/'LS_CPU_DISPATCH.json'
if done.exists():print(done.read_text())
elif dispatch.exists():print(dispatch.read_text())
else:
 env=dict(os.environ,PYTHONPATH=str(r/'pytools')+':'+str(r/'WM_PACKAGE')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',COOPMWM_NICHE_BASELINE=str(r/'FROZEN_PRE_NICHE_SEARCH.py'),COOPMWM_NULLFIX_GEOMETRY=str(r/'FROZEN_GEOMETRY_SOURCE.py'),COOPMWM_NULLFIX_NETWORK=str(r/'FROZEN_NETWORK.net.xml.gz'))
 with (r/'LS_CPU.log').open('x') as log:
  code='from tools.coopmwm.nullfix_geometry import install;install('+repr(str(r/'FROZEN_GEOMETRY_SOURCE.py'))+');from tools.coopmwm.ls_cpu_test import run;run('+repr(str(r))+')'
  proc=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-c',code],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d=dict(pid=proc.pid,start_ticks=pathlib.Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
 with dispatch.open('x') as f:json.dump(d,f)
 print(json.dumps(d))''',timeout=40)
    print(out)


def cpu_result():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({SOURCE!r});p=r/'LS_CPU_RESULT.json'
if p.exists():print(p.read_text())
else:print(json.dumps(dict(pending=True,log=(r/'LS_CPU.log').read_text()[-5000:])))''',timeout=30)
    v=json.loads(out)
    if not v.get('pending'):
        p=ops.EVIDENCE/('REPLAY_CPU_'+SOURCE.rsplit('/',1)[1]+'.json')
        if not p.exists():write(p,v)
        print(json.dumps({k:v[k] for k in ('pass_','tests','skips','sim_s','wall_s','root')}))
        if not v['pass_']:print(ops.ssh.remote('cpuQ',f'from pathlib import Path\nprint(Path({SOURCE!r},"LS_CPU.log").read_text()[-16000:])',timeout=30))
    else:print(out)


def layout_check():
    """Read-only live TRAIN compatibility: adopted safety overlays + branch tapes."""
    from .ls_collect_ops import ROOT, COLLECTION
    out=ops.ssh.remote('cpuQ',f'''from pathlib import Path
import sys,json,time,hashlib
s=Path({SOURCE!r});sys.path[:0]=[str(s/'pytools'),str(s/'WM_PACKAGE'),'/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
cpu=json.loads((s/'LS_CPU_RESULT.json').read_text());assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
from tools.coopmwm import replay_bank as rb,bank_safety
r=Path({ROOT!r});src={COLLECTION!r}
decisions=rb.bank_decisions(r,'B',sources=[src]);conf=rb.confirmations(r,'B')
assert conf==bank_safety.confirmations(r,'B')
# Validate real context/action tapes without creating/freezing a temperature file
# or running an optimizer. Unit weight is diagnostic only, never used for training.
rb.temperature=lambda *a,**k:dict(temperature_veh_s=None,rule='read_only_layout_check')
pairs,summary=rb.bank_pairs(r,'B',20261101,sources=[src])
result=dict(unix=time.time(),pass_=True,source_root=str(s),decisions=len(decisions),admitted=sum(x['admitted'] for x in conf.values()),imitation_pairs=len(pairs),summary=summary,new_sumo_sim_s=0,new_reserved_sim_s=0,optimizer_applied=False,temperature_written=False)
print(json.dumps(result))''',timeout=60)
    value=json.loads(out)
    path=ops.EVIDENCE/f'B_REPLAY_LS_LAYOUT_{__import__("time").time_ns()}.json'
    write(path,value)
    print(json.dumps(dict(receipt=str(path),**value)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['qualify','cpu_result','query_diagnosis','release_status','layout_check'])
    a=p.parse_args()
    if a.action=='release_status':print(json.dumps(release_status()))
    else:globals()[a.action]()
