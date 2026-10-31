"""Task-local, short CPU-only SSH operations; never inspect assignment/truth files."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'tools/ensprobe/evidence'
REG = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vehicle_params|driver_params|assignment|requests\.rou|driver_routes', re.I)
SEEDS = list(range(969040, 969045))
RNG_SEEDS = list(range(969086, 969094))
REMOTE='/home/ubuntu/ensprobe20261031'
ENV=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
         C4_POLICY_STOCHASTIC='0',C4_POLICY_ACCEL_NOISE='0',PYTHONDONTWRITEBYTECODE='1',
         PATH='/home/ubuntu/miniconda3/bin:/usr/local/bin:/usr/bin:/bin')


def write(name, obj):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE/name).write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')


def ssh(code, timeout=90, host='cpuQ'):
    python = '/home/ubuntu/miniconda3/bin/python' if host == 'cpuQ' else '/root/miniconda3/bin/python'
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', host, python+' -'],
                       input=code, text=True, encoding='utf-8', capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stderr+p.stdout)
    return json.loads(p.stdout)


def inspect():
    code = '''import json,os,hashlib
from pathlib import Path
r=Path('/home/ubuntu/e3final/runtime')
m=r/'SOURCE_MANIFEST.json'
out=dict(load=os.getloadavg(),cpus=os.cpu_count(),manifest_sha256=hashlib.sha256(m.read_bytes()).hexdigest())
names=['traffic_shepherd/scenarios.py','traffic_shepherd/c4/population.py','traffic_shepherd/simulation.py',
 'traffic_shepherd/c4/env.py','traffic_shepherd/c4/seeds.py','traffic_shepherd/seeds.py',
 'traffic_shepherd/metrics.py','traffic_shepherd/config.py','baselines/circles/adapter.json',
 'traffic_shepherd/c4/evalprep.py','src/wms/control/deploy.py']
manifest=json.loads(m.read_text());out['sources']={}
for name in names:
 p=r/name;raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 if name in manifest:assert digest==manifest[name]
 out['sources'][name]=dict(sha256=digest,in_source_manifest=name in manifest,text=raw.decode())
w=Path('/home/ubuntu/wmdata20261031/tools')
out['wrappers']={}
for name in ['wmdata/worker.py','speedhead/observer.py','ctrlarm/zero.py']:
 p=w/name
 if p.is_file():out['wrappers'][name]=p.read_text()
out['wmdata_names']=[p.name for p in (w/'wmdata').iterdir()] if (w/'wmdata').exists() else []
import sys,xml.etree.ElementTree as ET,gzip
sys.path[:0]=[str(r),str(r/'src')]
from traffic_shepherd.config import load_config
c=load_config();np=c.network_path
with (gzip.open(np,'rb') if str(np).endswith('.gz') else np.open('rb')) as f:net=ET.parse(f).getroot()
out['network']=dict(path=str(np),junctions=[j.attrib for j in net.findall('junction')],params=[p.attrib for p in net.iter('param')])
print(json.dumps(out))
'''
    out = ssh(code)
    write('runtime_sources.json', out)
    for name,row in out['sources'].items():
        (EVIDENCE/(name.replace('/','__')+'.txt')).write_text(row['text'].replace('\r',''),encoding='utf-8')
    for name,text in out['wrappers'].items():
        (EVIDENCE/(name.replace('/','__')+'.txt')).write_text(text.replace('\r',''),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('sources','wrappers')}, indent=2))


def audit():
    candidates=SEEDS+RNG_SEEDS
    conflicts=[]
    umbrellas={(965184,969999),(966000,969999),(966000,969699)}
    text=REG.read_text(encoding='utf-8')
    for n,line in enumerate(text.splitlines(),1):
        if 'S-ENSPROBE' in line:continue
        tokens={int(x) for x in re.findall(r'(?<![\w.])(?:[sw])?(\d{6})(?![\w.])',line)}
        ranges=[]
        for a,b in re.findall(r'(\d{6})\s*(?:-|–|—|\.\.|锟\?)\s*(\d{3,6})',line):
            lo=int(a);hi=int(b) if len(b)==6 else int(a[:-len(b)]+b)
            if (lo,hi) not in umbrellas and any(lo<=s<=hi for s in candidates):ranges.append([lo,hi])
        if tokens.intersection(candidates) or ranges:conflicts.append(dict(line=n,ranges=ranges,text=line))
    hosts={}
    for host,base in [('cpuQ','/home/ubuntu'),('gpuR','/root'),('gpuA','/root')]:
        hosts[host]=ssh(fr'''import os,re,json,sys
from pathlib import Path
bad=re.compile({BAD.pattern!r},re.I);hits=[];count=0
for here,dirs,files in os.walk({base!r},followlinks=False):
 dirs[:]=[n for n in dirs if not bad.search(str(Path(here)/n)) and not n.startswith('.') and n not in ('miniconda3','miniconda','ensprobe20261031')]
 for name in dirs+files:
  p=str(Path(here)/name)
  if bad.search(p):continue
  count+=1
  if any(int(x) in {candidates!r} for x in re.findall(r'(?<!\d)(96\d{{4}})(?!\d)',name)):hits.append(p)
sys.path.insert(0,{base!r}+'/e3final/runtime')
guards=[]
if Path({base!r}+'/e3final/runtime').is_dir():
 from traffic_shepherd.seeds import require_development_case
 from traffic_shepherd.c4.seeds import check_case
 for s,d in zip({candidates!r},[2400,2700,2400,2700,2700]+[2400]*8):
  require_development_case(s,d,'exposed');check_case(s,d);guards.append(s)
print(json.dumps(dict(hits=hits,safe_names_checked=count,guards=guards)))
''',host=host,timeout=120)
    out=dict(seeds=SEEDS,rng_seeds=RNG_SEEDS,registry_sha256=hashlib.sha256(text.encode()).hexdigest(),registry_conflicts=conflicts,hosts=hosts)
    out['pass']=not conflicts and not any(v['hits'] for v in hosts.values()) and hosts['cpuQ']['guards']==candidates
    write('exposure_'+str(SEEDS[0])+'.json',out)
    print(json.dumps(out,indent=2))


def stage():
    exposure=json.loads((EVIDENCE/'exposure_969040.json').read_text())
    assert exposure['pass']
    assert 'S-ENSPROBE EXACT PLAN REGISTERED BEFORE SUMO' in REG.read_text(encoding='utf-8')
    paths=[ROOT/'tools/ensprobe/worker.py',ROOT/'tests/test_ensprobe.py',ROOT/'docs/paper_results/ENSPROBE_DESIGN.md']
    sources={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as tar:
        for p in paths:tar.add(p,arcname=p.relative_to(ROOT).as_posix())
    raw=base64.b64encode(buf.getvalue()).decode()
    out=ssh(f'''import base64,io,json,tarfile,hashlib
from pathlib import Path
root=Path({REMOTE!r});assert root.parent.is_dir();assert not root.exists(),'existing task: reconcile, never restage'
root.mkdir();stage=root/'code';stage.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({raw!r})),mode='r:gz') as t:t.extractall(stage)
sources={sources!r}
assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==h for n,h in sources.items())
binding=dict(sources=sources,exposure={exposure!r},budget_sim_s=49800,cap_sim_s=50000,workers=1)
if root.name in ('recovery_v2','recovery_v3','recovery_v4'):
 original=root.parent if root.name=='recovery_v2' else root.parent/('recovery_v3' if root.name=='recovery_v4' else 'recovery_v2')
 assert (original/'STOP').exists()
 failure=json.loads((original/'FAILURE.json').read_text());assert failure['actual_sim_s']==(69 if root.name=='recovery_v4' else 0)
 rows=[json.loads(s) for s in (original/'accounting.jsonl').read_text().splitlines()]
 assert sum(r['sim_s'] for r in rows)==(900 if root.name=='recovery_v2' else 1800)
 binding['recovery']=dict(original_root=str(original),prior_actual_sim_s=300 if root.name=='recovery_v2' else 1200,unspent_branch_id=failure['id'])
 if root.name in ('recovery_v3','recovery_v4'):
  assert failure['id']=='w969040_native_r969087'
  previous=root.parent/'recovery_v2'
  binding['recovery'].update(reconstruct_first=True,previous_root=str(previous),retained_result_sha256=hashlib.sha256((previous/'w969040/w969040_native_r969086_RESULT.json').read_bytes()).hexdigest())
 if root.name=='recovery_v4':
  binding['budget_sim_s']=49869
  binding['recovery'].update(prior_actual_sim_s=1269,aborted_prefix_sim_s=69)
  rows.append(dict(id='retained_scheduling_partial69',sim_s=69,actual_sim_s=69,reason='own cooperative spare-CPU admission repair; all ticks retained'))
 with (root/'accounting.jsonl').open('x') as f:
  for row in rows:f.write(json.dumps(row)+'\\n')
(root/'BINDING.json').write_text(json.dumps(binding,indent=2))
print(json.dumps(dict(root=str(root),sources=sources,new_sim_s=0)))
''')
    write((Path(REMOTE).name.replace('recovery_','')+'_' if 'recovery_' in REMOTE else '')+'stage.json',out);print(json.dumps(out,indent=2))


def cpu():
    out=ssh(f'''import os,sys,json,contextlib,io
from pathlib import Path
root=Path({REMOTE!r});os.environ.update({ENV!r});os.nice(19);sys.path.insert(0,str(root/'code'))
from tools.ensprobe.worker import setup,MAPPO
receipt=setup()
import traci
def forbidden(*a,**kw):raise AssertionError('CPU gate attempted SUMO')
traci.start=forbidden;traci.simulationStep=forbidden
from wms.control.policy import load_policy_fn
from wms.control.lanes import LaneDwell
exe=load_policy_fn(MAPPO,device='cpu');dwell=LaneDwell(exe.lane_rules.dwell_s,1.)
import pytest
log=io.StringIO()
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):rc=pytest.main(['-q',str(root/'code/tests/test_ensprobe.py')])
receipt.update(returncode=rc,tests=log.getvalue(),controller_class=type(exe).__name__)
with (root/'CPU.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
''',timeout=120)
    write((Path(REMOTE).name.replace('recovery_','')+'_' if 'recovery_' in REMOTE else '')+'cpu.json',out);print(json.dumps(out,indent=2))


def launch():
    out=ssh(f'''import os,sys,json,subprocess,time,hashlib
from pathlib import Path
root=Path({REMOTE!r});assert not any((root/n).exists() for n in ['LAUNCH.json','STOP'])
assert json.loads((root/'CPU.json').read_text())['returncode']==0
binding=json.loads((root/'BINDING.json').read_text());assert binding['exposure']['pass']
assert all(hashlib.sha256((root/'code'/n).read_bytes()).hexdigest()==h for n,h in binding['sources'].items())
assert os.getloadavg()[0]<=20
env=dict(os.environ);env.update({ENV!r})
with (root/'worker.log').open('x') as log:
 p=subprocess.Popen([sys.executable,'-B','-u','-m','tools.ensprobe.worker'],cwd=root/'code',env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
 receipt=dict(pid=p.pid,unix=time.time(),root=str(root),workers=1,budget_sim_s=binding['budget_sim_s'],cap_sim_s=50000)
 with (root/'LAUNCH.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
''')
    write((Path(REMOTE).name.replace('recovery_','')+'_' if 'recovery_' in REMOTE else '')+'launch.json',out);print(json.dumps(out,indent=2))


def progress():
    out=ssh(f'''import json,os
from pathlib import Path
root=Path({REMOTE!r});out=dict(load=os.getloadavg())
for n in ['LAUNCH.json','CPU.json','FAILURE.json','COMPLETE.json']:
 p=root/n
 if p.exists():out[n]=json.loads(p.read_text())
if 'LAUNCH.json' in out:out['own_pid_exists']=Path('/proc',str(out['LAUNCH.json']['pid'])).exists()
out['branches']=len(list(root.glob('w*/*_RESULT.json')));out['gates']=len(list(root.glob('w*/*_GATE.json')))
out['trunks']=len(list(root.glob('w*/TRUNK.json')))
p=root/'accounting.jsonl'
out['charged_sim_s']=sum(json.loads(s)['sim_s'] for s in p.read_text().splitlines()) if p.exists() else 0
out['log_tail']=(root/'worker.log').read_text()[-6000:] if (root/'worker.log').exists() else ''
print(json.dumps(out))
''')
    write((Path(REMOTE).name.replace('recovery_','')+'_' if 'recovery_' in REMOTE else '')+'progress.json',out);print(json.dumps(out,indent=2))


def diagnose():
    """New owned process, zero physics steps; inspect only physical queue getters."""
    out=ssh(f'''import os,sys,json,contextlib,io
from pathlib import Path
root=Path({REMOTE!r});assert (root/'STOP').exists()
os.environ.update({ENV!r});os.nice(19);sys.path[:0]=[str(root/'code'),'/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
import traci
from traffic_shepherd.simulation import start_sumo
from traffic_shepherd.config import load_config
world=root/'w969040';trunk=json.loads((world/'TRUNK.json').read_text())['physical']
scenes=list((world/'scenarios').glob('*/scenario.sumocfg'));assert len(scenes)==1
def forbidden(*a,**kw):raise AssertionError('zero-step diagnosis attempted simulationStep')
traci.simulationStep=forbidden
with contextlib.redirect_stdout(io.StringIO()):
 start_sumo('sumo',scenes[0],seed=969086,config=load_config(),api=traci,
            extra_cli=['--save-state.precision','17','--save-state.rng','false'])
try:
 traci.simulation.loadState(str(world/'physical_state.xml'))
 pending=sorted(traci.simulation.getPendingVehicles())
 queues={{l:traci.lane.getLastStepHaltingNumber(l) for l in traci.lane.getIDList()}}
 out=dict(new_sim_s=0,pid=os.getpid(),time=traci.simulation.getTime(),
          trunk_pending=trunk['pending'],loaded_pending=pending,
          trunk_source_queue=trunk['source_queue'],lane_queue_differences={{l:[trunk['lane_queues'][l],n] for l,n in queues.items() if trunk['lane_queues'][l]!=n}},
          lane_vehicle_lists={{l:list(traci.lane.getLastStepVehicleIDs(l)) for l in queues if trunk['lane_queues'][l]!=queues[l]}})
 with (root/'QUEUE_DIAGNOSIS.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out))
finally:traci.close()
''')
    write('queue_diagnosis.json',out);print(json.dumps(out,indent=2))


def waiting():
    out=ssh(f'''import os,sys,json,contextlib,io
from pathlib import Path
root=Path({REMOTE!r});assert (root/'STOP').exists();os.environ.update({ENV!r});os.nice(19)
sys.path[:0]=['/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
import traci
from traffic_shepherd.simulation import start_sumo
from traffic_shepherd.config import load_config
world=root/'w969040';trunk=json.loads((world/'TRUNK.json').read_text())['physical']
scene=list((world/'scenarios').glob('*/scenario.sumocfg'))[0]
def forbidden(*a,**kw):raise AssertionError('zero-step queue diagnosis attempted physics')
traci.simulationStep=forbidden
with contextlib.redirect_stdout(io.StringIO()):start_sumo('sumo',scene,seed=969086,config=load_config(),api=traci,extra_cli=['--save-state.rng','false'])
try:
 traci.simulation.loadState(str(world/'physical_state.xml'));rows=[]
 for vid in trunk['pending']:
  try:rows.append(dict(id=vid,lane=traci.vehicle.getLaneID(vid),route=list(traci.vehicle.getRoute(vid)),departure=traci.vehicle.getDeparture(vid)))
  except traci.TraCIException as e:rows.append(dict(id=vid,error=str(e)))
 out=dict(new_sim_s=0,pid=os.getpid(),waiting=rows)
 with (root/'WAITING_DIAGNOSIS.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out))
finally:traci.close()
''')
    write('waiting_diagnosis.json',out);print(json.dumps(out,indent=2))


def collect():
    out=ssh(f'''import json,hashlib
from pathlib import Path
root=Path({REMOTE!r});assert (root/'COMPLETE.json').exists(),'complete panel only'
out=dict(results=[],gates=[],trunks=[],receipts={{}},hashes={{}})
for p in sorted(root.glob('w*/*_RESULT.json')):out['results'].append(json.loads(p.read_text()));out['hashes'][str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in sorted(root.glob('w*/*_GATE.json')):out['gates'].append(dict(path=str(p.relative_to(root)),**json.loads(p.read_text())));out['hashes'][str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in sorted(root.glob('w*/TRUNK.json')):out['trunks'].append(json.loads(p.read_text()));out['hashes'][str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['BINDING.json','LAUNCH.json','CPU.json','PREFLIGHT.json','COMPLETE.json','UNSPENT_RESERVATION_RESUMED.json']:
 p=root/name
 if p.exists():out['receipts'][name]=json.loads(p.read_text());out['hashes'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
out['accounting']=[json.loads(s) for s in (root/'accounting.jsonl').read_text().splitlines()]
out['hashes']['accounting.jsonl']=hashlib.sha256((root/'accounting.jsonl').read_bytes()).hexdigest()
print(json.dumps(out))
''')
    write('collection.json',out)
    print(json.dumps(dict(branches=len(out['results']),trunks=len(out['trunks']),terminal=out['receipts']['COMPLETE.json']),indent=2))


def scheduling():
    out=ssh(f'''import os,json
from pathlib import Path
root=Path({REMOTE!r});own=json.loads((root/'LAUNCH.json').read_text())['pid'];peers=[];owned=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace');fields=(p/'stat').read_text().rsplit(')',1)[1].split()
  if fields[0]=='R' and any(s in cmd.lower() for s in ('tools.leverprobe','tools.c1switch','tools.c1closed','tools.wmvalue')):peers.append(dict(pid=int(p.name),command=cmd[:250]))
  if int(p.name)==own or int(fields[1])==own:owned.append(dict(pid=int(p.name),state=fields[0],cpu_ticks=int(fields[11])+int(fields[12]),command=cmd[:250]))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
print(json.dumps(dict(load=os.getloadavg(),running_priority_peers=peers,owned=owned)))
''')
    write('scheduling.json',out);print(json.dumps(out,indent=2))


def scheduling_stop():
    out=ssh(f'''import json
from pathlib import Path
root=Path({REMOTE!r});assert root.name=='recovery_v3'
pid=json.loads((root/'LAUNCH.json').read_text())['pid']
assert b'tools.ensprobe.worker' in Path('/proc',str(pid),'cmdline').read_bytes()
with (root/'STOP').open('x') as f:f.write('OWN cooperative scheduling stop: yield rule unnecessarily waits for one peer on32cores; retain all actual ticks\\n')
print(json.dumps(dict(own_pid=pid,control=str(root/'STOP'),signals=0)))
''')
    write('scheduling_stop.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inspect','audit','stage','cpu','launch','progress','diagnose','waiting','v2_stage','v2_cpu','v2_launch','v2_progress','v2_collect','v3_stage','v3_cpu','v3_launch','v3_progress','v3_collect','v3_scheduling','v3_scheduling_stop','v4_stage','v4_cpu','v4_launch','v4_progress','v4_collect']);a=p.parse_args()
    if a.action.startswith(('v2_','v3_','v4_')):
        REMOTE+='/recovery_'+a.action[:2];ENV['ENSPROBE_ROOT']=REMOTE;a.action=a.action[3:]
    globals()[a.action]()
