"""Short, explicit, write-once host operations; no automatic redispatch or peer signals."""
import argparse
import base64
import io
import json
from pathlib import Path
import tarfile

from tools.c1switch.design import checked, sha
from tools.c1switch.ops import ssh as frozen_ssh
from tools.c1switch.storage import canonical, write

from .design import CONTROLLERS, CPU_ARMS, DESIGN, GPU_SECONDS, PROTOCOL, REGISTRATION, budget
from .gate import approval

EVIDENCE = Path('tools/c1switch2/evidence')
TESTS = ['tests/test_c1switch2.py', 'tests/test_c1switch2_wall.py',
         'tests/test_c1switch2_integrations.py', 'tests/test_switchable_wall.py',
         'tests/test_c1switch.py', 'tests/test_c1switch_feedback.py', 'tests/test_c1switch_merge_fast.py']


def ssh(host, code, timeout=60):
    compile(code, '<c1switch2-remote>', 'exec')
    return frozen_ssh(host, code, timeout=timeout)


def status(host):
    parent = '/home/ubuntu' if host == 'cpuQ' else '/root'
    return ssh(host, f'''import json,os
from pathlib import Path
roots=[]
for entry in os.scandir({parent!r}):
 if entry.name.startswith('c1switch2_run_') and entry.is_dir(follow_symlinks=False):
  root=Path(entry.path);row=dict(root=str(root))
  for name in ('CANARY_DISPATCH_ONCE.json','DISPATCH_CANARY.json','CANARY.json','CANARY_EXIT.json',
               'CPU_DISPATCH_ONCE.json','DISPATCH_CPU.json','STOP'):
   p=root/name
   if p.exists():
    data=json.loads(p.read_text())
    row[name]=({{k:data[k] for k in ('native_repeat','six_way_repeat','planned_sim_s')}}
               if name=='CANARY.json' else data)
  if 'DISPATCH_CANARY.json' in row:
   pid=row['DISPATCH_CANARY.json']['pid'];row['own_pid_exists']=Path('/proc',str(pid)).exists()
  if 'DISPATCH_CPU.json' in row:
   pid=row['DISPATCH_CPU.json']['pid'];proc=Path('/proc',str(pid))
   row['cpu_pid_state']=proc.joinpath('stat').read_text().split()[2] if proc.exists() else 'EXITED'
   row['cpu_complete']=(root/'supervisor'/'COMPLETE.json').exists()
  roots.append(row)
print(json.dumps(dict(host={host!r},load=os.getloadavg(),roots=roots,new_sim_s=0,new_gpu_s=0)))
''')


def env(host, root):
    common = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', C1SWITCH_ENGINE_ROOT=root)
    if host == 'cpuQ':
        common.update(C1SWITCH_RUNTIME='/home/ubuntu/e3final',
            C1SWITCH_WRAPPERS='/home/ubuntu/wmdata20261031',
            C1SWITCH_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt')
    return common


def zero_physics_predecessor(host, root, text):
    parent = '/home/ubuntu' if host == 'cpuQ' else '/root'
    if (Path(root).parent.as_posix() != parent or not Path(root).name.startswith('c1switch2_run_') or
            not any('S-C1SWITCH2 ZERO-PHYSICS STARTUP CORRECTION REGISTERED' in line and root in line
                    for line in text.splitlines())):
        raise ValueError('explicit registered zero-physics predecessor required')
    return ssh(host, f'''import hashlib,json
from pathlib import Path
root=Path({root!r});dispatch=json.loads((root/'DISPATCH_CANARY.json').read_text())
proc=Path('/proc',str(dispatch['pid']))
assert not proc.exists() or proc.joinpath('stat').read_text().split()[2]=='Z','predecessor still alive'
ledger=root/'accounting.jsonl';raw=ledger.read_bytes() if ledger.exists() else b''
assert not raw,'predecessor charged physics; zero-physics correction forbidden'
assert not list((root/'jobs').iterdir()),'predecessor constructed jobs; reconcile explicitly'
assert "ModuleNotFoundError: No module named 'traffic_shepherd.seeds'" in (root/'CANARY.log').read_text()
out=dict(root=str(root),pid=dispatch['pid'],status='EXITED_BEFORE_SETUP_AND_PHYSICS',charged_sim_s=0,actual_sim_s=0,ledger_sha256=hashlib.sha256(raw).hexdigest())
path=root/'STARTUP_STOP.json'
if not path.exists():
 with path.open('x') as stream:json.dump(out,stream,indent=2)
else:assert json.loads(path.read_text())==out
print(json.dumps(out))
''')


def stage(host, predecessor=None, cpu_only=False):
    registration_bytes = checked(REGISTRATION).read_bytes()
    text = registration_bytes.decode('utf-8')
    digest = sha(DESIGN)
    if not approval(text) or not any('S-C1SWITCH2 DESIGN REGISTERED ' in line and digest in line
                                   for line in text.splitlines()):
        raise ValueError('current design/approval not registered')
    state = status(host)
    old = None
    dispatched = [r for r in state['roots'] if 'DISPATCH_CANARY.json' in r or 'CANARY_DISPATCH_ONCE.json' in r]
    if cpu_only:
        from .gate import cpu_authority
        if not cpu_authority(text) or any('CPU_DISPATCH_ONCE.json' in r for r in state['roots']):
            raise ValueError('CPU authority absent or subset already claimed; inspect/reuse')
        if host == 'cpuQ' and not any(r.get('CANARY.json', {}).get('six_way_repeat') == 'PASS'
                                     for r in state['roots']):
            raise ValueError('completed mechanics receipt required')
    elif dispatched:
        if len(dispatched) != 1 or dispatched[0]['root'] != predecessor:
            raise ValueError('existing canary dispatch: inspect and reuse; never duplicate')
        old = zero_physics_predecessor(host, predecessor, text)
    files = sorted(Path('tools/c1switch').glob('*.py'))+sorted(Path('tools/c1switch2').glob('*.py'))
    files += [Path(p) for p in TESTS+[DESIGN, 'docs/paper_results/c1switch2/PANEL_v2.json',
              'traffic_shepherd/c4/switchable_wall.py', 'src/wms/control/wall.py']]
    staged = {p.as_posix(): p for p in files}
    staged['engine/branching.py'] = Path('traffic_shepherd/c4/branching.py')
    for peer, local in [('selmappo', 'E:/agent-work/c4-selmappo/tools/selmappo'),
                        ('selmappo2', 'E:/agent-work/c4-selmappo2/tools/selmappo2')]:
        # Only source modules in the explicitly supplied interface directories.
        for path in Path(local).glob('*.py'):
            staged['peer/'+peer+'/'+path.name] = checked(path)
    sources = {name: sha(path) for name, path in staged.items()}
    identity = canonical(dict(sources=sources, registration=sha(REGISTRATION)))[:16]
    root = ('/home/ubuntu' if host == 'cpuQ' else '/root')+'/c1switch2_run_'+identity
    binding = dict(protocol=PROTOCOL, cpu_host=host, include_phys=False, sim_s=budget(),
        library=list(CONTROLLERS), gpu_seconds={a: GPU_SECONDS[a] for a in budget()},
        design_sha256=digest, registration_sha256=sha(REGISTRATION), sources=sources,
        prediction_objective='on_road_fleet_delay',
        runtime=dict(entry='tools.c1switch2.runtime:create', sha256=sha('tools/c1switch2/runtime.py')),
        predictors={a:dict(entry='tools.c1switch2.rpc:'+a, sha256=sha('tools/c1switch2/rpc.py'))
                    for a in ('wm', 'bb')},
        inference_specs={a:dict(status='PENDING') for a in ('wm', 'bb')},
        selectors={a:dict(entry='tools.c1switch2.selectors:'+entry,
                          sha256=sha('tools/c1switch2/selectors.py'))
                   for a, entry in [('fleet_ppo', 'fleet'), ('cav_mappo', 'per_cav')]},
         selector_specs={a: dict(status='PENDING') for a in ('fleet_ppo', 'cav_mappo')})
    if cpu_only:
        binding.update(dispatch_phase='cpu', enabled_arms=list(CPU_ARMS))
    if old is not None:
        binding['startup_predecessor'] = old
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for name, path in staged.items():
            tar.add(path, arcname=name)
    raw = base64.b64encode(buf.getvalue()).decode()
    result = ssh(host, f'''import base64,hashlib,io,json,tarfile
from pathlib import Path
root=Path({root!r});assert root.parent.is_dir();assert not root.exists(),'immutable stage exists'
root.mkdir();stage=root/'pytools';stage.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({raw!r})),mode='r:gz') as tar:tar.extractall(stage,filter='data')
sources={sources!r}
assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==v for n,v in sources.items())
(root/'STAGED.json').write_text(json.dumps(sources,indent=2))
(root/'BINDING.json').write_text(json.dumps({binding!r},indent=2))
(root/'REGISTRATION.md').write_bytes(base64.b64decode({base64.b64encode(registration_bytes).decode()!r}))
assert hashlib.sha256((root/'REGISTRATION.md').read_bytes()).hexdigest()=={binding['registration_sha256']!r}
for name in ('jobs','specs','tasks','logs'): (root/name).mkdir()
print(json.dumps(dict(host={host!r},root=str(root),binding_sha256={canonical(binding)!r},files=len(sources),new_sim_s=0,new_gpu_s=0)))
''', timeout=90)
    write(EVIDENCE/('STAGE_'+host+'_'+identity+'.json'), result)
    print(json.dumps(result))


def preflight(host, root):
    result = ssh(host, f'''import contextlib,io,json,os,sys,sysconfig
from pathlib import Path
root=Path({root!r});stage=root/'pytools';assert stage.is_dir()
assert not (root/'PREFLIGHT.json').exists(),'preserve prior preflight'
os.nice(19);os.environ.update({env(host, root)!r})
os.environ['PATH']=str(Path(sysconfig.get_paths()['purelib'])/'sumo/bin')+':'+os.environ['PATH']
os.environ['C1SWITCH2_SELECTOR_SOURCE']=str(stage/'peer/selmappo')
os.environ['C1SWITCH2_CAV_SOURCE']=str(stage/'peer/selmappo2')
sys.path.insert(0,str(stage));os.chdir(stage)
log=io.StringIO()
try:
 with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  from tools.c1switch2.runtime import Runtime
  from tools.c1switch2.runner import backend
  from tools.c1switch2.design import assigned,worlds
  assert 'traffic_shepherd.seeds' not in sys.modules,'fresh-process regression was masked'
  scratch=root/'CPU_CONSTRUCTION';scratch.mkdir();(scratch/'jobs').mkdir()
  original=Runtime.create_backend
  Runtime.create_backend=lambda self,*a,**k:dict(inputs=self.inputs)
  try:
   world=next(w for w in worlds() if assigned(w['seed'])=={host!r})
   constructed=backend(scratch,json.loads((root/'BINDING.json').read_text()),'native',world,'cpu_only')
  finally:Runtime.create_backend=original
  inputs=constructed['inputs']
  assert not (scratch/'accounting.jsonl').exists(),'construction-only CPU probe charged physics'
  from tools.c1switch.library import install_wall_extension
  install_wall_extension()
  from tools.c1switch2.design import worlds,canary_world,guard
  for world in worlds()+[canary_world()]:guard(world)
  import subprocess,traci
  def forbidden(*a,**k):raise AssertionError('CPU preflight attempted process or physics')
  subprocess.Popen=forbidden;traci.start=forbidden;traci.simulationStep=forbidden
  import pytest
  code=pytest.main(['-q','-p','no:cacheprovider','-c','/dev/null']+[str(stage/p) for p in {TESTS!r}])
 out=dict(status='PASS' if code==0 else 'FAIL',returncode=int(code),inputs=inputs,guards=17)
except BaseException as exc:
 out=dict(status='FAIL',error=repr(exc))
out.update(host={host!r},root=str(root),output=log.getvalue(),new_sim_s=0,new_gpu_s=0)
(root/'PREFLIGHT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
''', timeout=90)
    write(EVIDENCE/('PREFLIGHT_'+host+'_'+Path(root).name+'.json'), result)
    print(json.dumps(result))


def probe(host, root, model, steps=120):
    if host != 'gpuR':
        raise ValueError('frozen original-host CPU model probe runs on gpuR')
    result = ssh(host, f'''import contextlib,io,json,os,sys
from pathlib import Path
root=Path({root!r});stage=root/'pytools';assert stage.is_dir()
output=root/{('MODEL_CPU_'+model+'_steps'+str(steps)+'.json')!r};assert not output.exists(),'preserve prior model CPU receipt'
os.nice(19);os.environ.update({env(host, root)!r})
sys.path[:0]=[str(stage),'/root/autodl-tmp/c4/speedhead20261030/pythondeps'];os.chdir(stage)
log=io.StringIO()
try:
 with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  from tools.c1switch2.hostcheck import probe
  out=probe({model!r},'/root/c1switch20261031/main_v6/bb.pt',steps={steps!r})
except BaseException as exc:
 import traceback
 out=dict(status='FAIL',error=repr(exc),traceback=traceback.format_exc(),new_sim_s=0,new_gpu_s=0)
out['output']=log.getvalue();output.write_text(json.dumps(out,indent=2));print(json.dumps(out))
''', timeout=90)
    write(EVIDENCE/('MODEL_CPU_'+model+'_steps'+str(steps)+'_'+Path(root).name+'.json'), result)
    summary = {k: v for k, v in result.items() if k not in ('attestation', 'output')}
    print(json.dumps(summary, indent=2))


def canary(host, root, predecessor=None):
    from .audit import registration_audit
    if host != 'cpuQ' or not root.startswith('/home/ubuntu/c1switch2_run_'):
        raise ValueError('canary belongs to the cpuQ task namespace')
    text = checked(REGISTRATION).read_text(encoding='utf-8')
    audit = registration_audit(REGISTRATION)
    if not approval(text) or audit['hits']:
        raise ValueError('current approval/registry not admitted')
    exposures = {}
    for h in ('cpuQ', 'gpuR', 'gpuA'):
        path = EVIDENCE/('EXPOSURE_'+h+'_v3.json')
        receipt = json.loads(path.read_text())
        if not receipt['pass']:
            raise ValueError('retained exposure gate failed')
        exposures[h] = sha(path)
    current = status(host)
    old = None
    dispatched = [r for r in current['roots'] if 'CANARY_DISPATCH_ONCE.json' in r or 'DISPATCH_CANARY.json' in r]
    if dispatched:
        if len(dispatched) != 1 or dispatched[0]['root'] != predecessor:
            raise ValueError('canary previously claimed: inspect/reuse, never duplicate')
        old = zero_physics_predecessor(host, predecessor, text)
    result = ssh(host, f'''import hashlib,json,os,subprocess,sys,sysconfig,time
from pathlib import Path
root=Path({root!r});stage=root/'pytools';assert root.is_dir()
binding=json.loads((root/'BINDING.json').read_text())
assert binding.get('startup_predecessor')=={old!r},'predecessor lineage mismatch'
assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==v for n,v in binding['sources'].items())
preflight=json.loads((root/'PREFLIGHT.json').read_text());assert preflight['status']=='PASS'
assert os.getloadavg()[0]<28,'yield to host pressure'
assert not (root/'STOP').exists();assert not (root/'CANARY_DISPATCH_ONCE.json').exists()
for sibling in root.parent.iterdir():
 if sibling.name.startswith('c1switch2_run_') and sibling.is_dir():
  if str(sibling)!={predecessor!r}:
   assert not (sibling/'CANARY_DISPATCH_ONCE.json').exists(),'another task canary claimed'
sys.path.insert(0,str(stage));os.chdir(stage)
from tools.c1switch.storage import canonical
from tools.c1switch2.gate import REQUIRED,admission
evidence={{}}
needed=('frozen_v7_sources','seed_guards','exposure_cpuQ','exposure_gpuR','exposure_gpuA',
        'registry_intervals','wall_sumo','wall_model','cpu_tests')
for key in REQUIRED:
 evidence[key]=dict(status='PASS' if key in needed else 'PENDING',binding_sha256=canonical(binding),
  source='PREFLIGHT.json + retained exposure receipts + current registry audit; mechanics only')
evidence['exposure_receipt_sha256']={exposures!r}
evidence['current_registry_audit']={audit!r}
assert admission((root/'REGISTRATION.md').read_text(),binding,evidence,phase='canary')['dispatch_authorized']
with (root/'GATES.json').open('x') as stream:json.dump(evidence,stream,indent=2)
with (root/'CANARY_DISPATCH_ONCE.json').open('x') as stream:
 json.dump(dict(cap_sim_s=4680,unix=time.time(),phase='mechanics_only',gpu_s=0),stream);stream.flush();os.fsync(stream.fileno())
environment=os.environ.copy();environment.update({env(host, root)!r})
environment['PATH']=str(Path(sysconfig.get_paths()['purelib'])/'sumo/bin')+':'+environment['PATH']
environment['PYTHONPATH']=str(stage)
with (root/'CANARY.log').open('x') as log:
 process=subprocess.Popen([sys.executable,'-B','-m','tools.c1switch2.runner','canary','--root',str(root)],
   cwd=stage,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
out=dict(host={host!r},root=str(root),pid=process.pid,phase='mechanics_only',planned_sim_s=4680,new_gpu_s=0,unix=time.time())
with (root/'DISPATCH_CANARY.json').open('x') as stream:json.dump(out,stream,indent=2)
print(json.dumps(out))
''', timeout=45)
    write(EVIDENCE/('DISPATCH_CANARY_'+Path(root).name+'.json'), result)
    print(json.dumps(result))


def collect(host, root):
    result = ssh(host, f'''import hashlib,json,time
from pathlib import Path
root=Path({root!r});assert root.name.startswith('c1switch2_run_')
out=dict(host={host!r},root=str(root),unix=time.time())
dispatch=json.loads((root/'DISPATCH_CANARY.json').read_text());pid=dispatch['pid']
proc=Path('/proc',str(pid));out['pid']=pid
out['pid_state']=proc.joinpath('stat').read_text().split()[2] if proc.exists() else 'EXITED'
ledger=root/'accounting.jsonl'
rows=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
out['charged_sim_s']=sum(r['sim_s'] for r in rows)
out['closed_jobs']=[];out['closed_actual_sim_s']=0
for name in ('native_reference','native_repeat',*['branch_'+c for c in {CONTROLLERS!r}],
             'six_way_reference','six_way_repeat'):
 close=root/'jobs'/('canary__'+name)/'CLOSE.json'
 if close.exists():
  row=json.loads(close.read_text());out['closed_jobs'].append(name);out['closed_actual_sim_s']+=row['actual_sim_s']
gate=root/'CANARY.json'
if gate.exists():
 row=json.loads(gate.read_text());out['canary']=dict(native_repeat=row['native_repeat'],six_way_repeat=row['six_way_repeat'],planned_sim_s=row['planned_sim_s'])
 out['canary_sha256']=hashlib.sha256(gate.read_bytes()).hexdigest()
log=root/'CANARY.log'
if log.exists():out['log']=log.read_text()
out['new_gpu_s']=0
print(json.dumps(out))
''', timeout=30)
    path = EVIDENCE/('CANARY_PROGRESS_'+str(int(result['unix']))+'.json')
    write(path, result)
    print(json.dumps(result, indent=2))


def export_canary(host, root):
    """Export completed public mechanics/accounting receipts, never simulator inputs."""
    result = ssh(host, f'''import base64,hashlib,json
from pathlib import Path
root=Path({root!r});assert root.name.startswith('c1switch2_run_')
path=root/'CANARY.json';row=json.loads(path.read_text())
assert row['native_repeat']==row['six_way_repeat']=='PASS'
ledger=[json.loads(line) for line in (root/'accounting.jsonl').read_text().splitlines()]
closed=[]
for name in ('native_reference','native_repeat',*['branch_'+c for c in {CONTROLLERS!r}],
             'six_way_reference','six_way_repeat'):
 closed.append(json.loads((root/'jobs'/('canary__'+name)/'CLOSE.json').read_text()))
charged=sum(r['sim_s'] for r in ledger);actual=sum(r['actual_sim_s'] for r in closed)
assert charged==actual==4680
print(json.dumps(dict(root=str(root),canary_base64=base64.b64encode(path.read_bytes()).decode(),ledger=ledger,closed=closed,
 canary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),charged_sim_s=charged,actual_sim_s=actual,new_gpu_s=0)))
''', timeout=30)
    name = Path(root).name
    path = EVIDENCE/('CANARY_'+name+'.json')
    ledger_path = EVIDENCE/('CANARY_ACCOUNTING_'+name+'.json')
    if path.exists() or ledger_path.exists():
        raise ValueError('preserve exported mechanics receipts')
    with path.open('xb') as stream:
        stream.write(base64.b64decode(result['canary_base64']))
    if sha(path) != result['canary_sha256']:
        raise ValueError('exported mechanics receipt bytes changed')
    write(ledger_path, {k: v for k, v in result.items() if k != 'canary_base64'})
    print(json.dumps({k: v for k, v in result.items() if k not in ('canary_base64', 'ledger', 'closed')}, indent=2))


def archive_binding(host, root):
    result = ssh(host, f'''import hashlib,json
from pathlib import Path
root=Path({root!r});assert root.name.startswith('c1switch2_run_')
out={{name:json.loads((root/name).read_text()) for name in ('BINDING.json','GATES.json','STAGED.json','PREFLIGHT.json')}}
binding=out['BINDING.json']
assert hashlib.sha256((root/'REGISTRATION.md').read_bytes()).hexdigest()==binding['registration_sha256']
assert all(hashlib.sha256((root/'pytools'/name).read_bytes()).hexdigest()==value for name,value in binding['sources'].items())
out['canary_sha256']=hashlib.sha256((root/'CANARY.json').read_bytes()).hexdigest()
out['root']=str(root);print(json.dumps(out))
''', timeout=30)
    path = EVIDENCE/('CANARY_BINDING_'+Path(root).name+'.json')
    if path.exists():
        raise ValueError('preserve archived binding')
    write(path, result)
    print(json.dumps(dict(path=str(path), sha256=sha(path), canary_sha256=result['canary_sha256'])))


def cpu_dispatch(host, root, workers):
    """Claim the seven-arm panel once per host; completed canary is evidence only."""
    from .audit import registration_audit
    from .gate import cpu_authority
    text = checked(REGISTRATION).read_text(encoding='utf-8')
    audit = registration_audit(REGISTRATION)
    if not approval(text) or not cpu_authority(text) or audit['hits']:
        raise ValueError('current CPU authority/registry not admitted')
    if not 1 <= workers <= (14 if host == 'cpuQ' else 4):
        raise ValueError('worker ceiling')
    name = 'c1switch2_run_4ad4ece01d5a9ac1'
    mechanics_path = EVIDENCE/('CANARY_'+name+'.json')
    archive = json.loads((EVIDENCE/('CANARY_BINDING_'+name+'.json')).read_text())
    mechanics = json.loads(mechanics_path.read_text())
    if (sha(mechanics_path) != archive['canary_sha256'] or
            mechanics['native_repeat'] != 'PASS' or mechanics['six_way_repeat'] != 'PASS'):
        raise ValueError('retained mechanics evidence mismatch')
    frozen = {k: v for k, v in archive['BINDING.json']['sources'].items()
              if k.startswith('tools/c1switch/') or k == 'engine/branching.py'}
    exposures = {}
    for h in ('cpuQ', 'gpuR', 'gpuA'):
        path = EVIDENCE/('EXPOSURE_'+h+'_v3.json')
        if not json.loads(path.read_text())['pass']:
            raise ValueError('exposure gate failed')
        exposures[h] = sha(path)
    result = ssh(host, f'''import hashlib,json,os,subprocess,sys,sysconfig,time
from pathlib import Path
root=Path({root!r});parent=Path({'/home/ubuntu' if host == 'cpuQ' else '/root'!r})
assert root.parent==parent and root.name.startswith('c1switch2_run_') and root.is_dir()
stage=root/'pytools';binding=json.loads((root/'BINDING.json').read_text())
assert binding['dispatch_phase']=='cpu' and binding['enabled_arms']=={list(CPU_ARMS)!r}
assert binding['cpu_host']=={host!r}
assert hashlib.sha256((root/'REGISTRATION.md').read_bytes()).hexdigest()=={sha(REGISTRATION)!r}==binding['registration_sha256'],'registration changed; restage before compute'
assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==v for n,v in binding['sources'].items())
assert all(binding['sources'][n]==v for n,v in {frozen!r}.items()),'v7 frozen sources changed'
assert json.loads((root/'PREFLIGHT.json').read_text())['status']=='PASS'
assert not (root/'STOP').exists() and not (root/'accounting.jsonl').exists()
assert not list((root/'jobs').iterdir())
assert os.getloadavg()[0] < {28 if host == 'cpuQ' else 20},'yield to host pressure'
for entry in os.scandir(parent):
 if entry.name.startswith('c1switch2_run_') and entry.is_dir(follow_symlinks=False):
  assert not Path(entry.path,'CPU_DISPATCH_ONCE.json').exists(),'CPU panel previously claimed; never retry'
sys.path.insert(0,str(stage));os.chdir(stage)
from tools.c1switch.storage import canonical
from tools.c1switch2.gate import REQUIRED,CPU_REQUIRED,admission
from tools.c1switch2.design import local_budget
evidence={{k:dict(status='PASS' if k in CPU_REQUIRED else 'PENDING',binding_sha256=canonical(binding),
 source='PREFLIGHT.json; frozen canary; retained preallocation exposure; current registry') for k in REQUIRED}}
evidence['mechanics_sha256']={sha(mechanics_path)!r}
evidence['exposure_receipt_sha256']={exposures!r};evidence['current_registry_audit']={audit!r}
assert admission((root/'REGISTRATION.md').read_text(),binding,evidence)['dispatch_authorized']
with (root/'GATES.json').open('x') as stream:json.dump(evidence,stream,indent=2)
planned=sum(local_budget({host!r})[a] for a in binding['enabled_arms'])
claim=dict(host={host!r},root=str(root),phase='cpu',enabled_arms=binding['enabled_arms'],planned_sim_s=planned,gpu_s=0,unix=time.time())
with (root/'CPU_DISPATCH_ONCE.json').open('x') as stream:
 json.dump(claim,stream,indent=2);stream.flush();os.fsync(stream.fileno())
environment=os.environ.copy();environment.update({env(host, root)!r})
environment['PATH']=str(Path(sysconfig.get_paths()['purelib'])/'sumo/bin')+':'+environment['PATH']
environment['PYTHONPATH']=str(stage)
with (root/'CPU.log').open('x') as log:
 process=subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch2.supervisor','--root',str(root),'--host',{host!r},'--workers',{str(workers)!r}],cwd=stage,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
out=dict(claim,pid=process.pid,workers={workers!r},binding_sha256=canonical(binding))
with (root/'DISPATCH_CPU.json').open('x') as stream:json.dump(out,stream,indent=2)
print(json.dumps(out))
''', timeout=45)
    write(EVIDENCE/('DISPATCH_CPU_'+host+'_'+Path(root).name+'.json'), result)
    print(json.dumps(result, indent=2))


def cpu_collect(host, root):
    result = ssh(host, f'''import json,time
from pathlib import Path
root=Path({root!r});assert root.name.startswith('c1switch2_run_')
dispatch=json.loads((root/'DISPATCH_CPU.json').read_text());proc=Path('/proc',str(dispatch['pid']))
ledger=root/'accounting.jsonl';rows=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
out=dict(host={host!r},root=str(root),unix=time.time(),pid=dispatch['pid'],pid_state=proc.joinpath('stat').read_text().split()[2] if proc.exists() else 'EXITED',charged_sim_s=sum(r['sim_s'] for r in rows),closed_actual_sim_s=0,results=[],accounting=rows,new_gpu_s=0)
out['complete']=(root/'supervisor'/'COMPLETE.json').exists()
out['stop']=json.loads((root/'STOP').read_text()) if (root/'STOP').exists() else None
for job in sorted((root/'jobs').iterdir()):
 close=job/'CLOSE.json'
 if close.exists():out['closed_actual_sim_s']+=json.loads(close.read_text())['actual_sim_s']
 result=job/'PHYSICS.json'
 if result.exists():out['results'].append(json.loads(result.read_text()))
if out['stop']:
 out['supervisor_log']=(root/'CPU.log').read_text()
 failed=root/'logs'/(out['stop'].get('id','')+'.log')
 if failed.is_file():out['failed_log']=failed.read_text()
print(json.dumps(out))
''', timeout=45)
    path = EVIDENCE/('CPU_PROGRESS_'+host+'_'+str(int(result['unix']))+'.json')
    write(path, result)
    print(json.dumps({k: (len(v) if k in ('results', 'accounting') else v)
                      for k, v in result.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('status', 'stage', 'preflight', 'probe', 'canary', 'collect',
                                          'export-canary', 'archive-binding', 'cpu-dispatch', 'cpu-collect'))
    parser.add_argument('--host', choices=('cpuQ', 'gpuR'), required=True)
    parser.add_argument('--root')
    parser.add_argument('--model', choices=('wm', 'bb'))
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--predecessor')
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'status':
        print(json.dumps(status(args.host), indent=2))
    elif args.command == 'stage':
        stage(args.host, args.predecessor, args.cpu_only)
    elif args.command == 'preflight':
        preflight(args.host, args.root)
    elif args.command == 'canary':
        canary(args.host, args.root, args.predecessor)
    elif args.command == 'collect':
        collect(args.host, args.root)
    elif args.command == 'export-canary':
        export_canary(args.host, args.root)
    elif args.command == 'archive-binding':
        archive_binding(args.host, args.root)
    elif args.command == 'cpu-dispatch':
        cpu_dispatch(args.host, args.root, args.workers)
    elif args.command == 'cpu-collect':
        cpu_collect(args.host, args.root)
    else:
        probe(args.host, args.root, args.model, args.steps)


if __name__ == '__main__':
    main()
