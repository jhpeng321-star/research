"""Short, explicit, write-once host operations; no automatic redispatch or peer signals."""
import argparse
import base64
import io
import json
from pathlib import Path
import tarfile

from tools.c1switch.design import checked, sha
from tools.c1switch.ops import ssh
from tools.c1switch.storage import canonical, write

from .design import CONTROLLERS, DESIGN, GPU_SECONDS, PROTOCOL, REGISTRATION, budget
from .gate import approval

EVIDENCE = Path('tools/c1switch2/evidence')
TESTS = ['tests/test_c1switch2.py', 'tests/test_c1switch2_wall.py',
         'tests/test_c1switch2_integrations.py', 'tests/test_switchable_wall.py',
         'tests/test_c1switch.py', 'tests/test_c1switch_feedback.py', 'tests/test_c1switch_merge_fast.py']


def status(host):
    parent = '/home/ubuntu' if host == 'cpuQ' else '/root'
    return ssh(host, f'''import json,os
from pathlib import Path
roots=[]
for entry in os.scandir({parent!r}):
 if entry.name.startswith('c1switch2_run_') and entry.is_dir(follow_symlinks=False):
  root=Path(entry.path);row=dict(root=str(root))
  for name in ('CANARY_DISPATCH_ONCE.json','DISPATCH_CANARY.json','CANARY.json','CANARY_EXIT.json','STOP'):
   p=root/name
   if p.exists():row[name]=json.loads(p.read_text())
  if 'DISPATCH_CANARY.json' in row:
   pid=row['DISPATCH_CANARY.json']['pid'];row['own_pid_exists']=Path('/proc',str(pid)).exists()
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


def stage(host):
    registration_bytes = checked(REGISTRATION).read_bytes()
    text = registration_bytes.decode('utf-8')
    digest = sha(DESIGN)
    if not approval(text) or not any('S-C1SWITCH2 DESIGN REGISTERED ' in line and digest in line
                                   for line in text.splitlines()):
        raise ValueError('current design/approval not registered')
    state = status(host)
    if any('DISPATCH_CANARY.json' in r for r in state['roots']):
        raise ValueError('existing canary dispatch: inspect and reuse; never duplicate')
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
        selector_specs={a: dict(status='PENDING') for a in ('fleet_ppo', 'cav_mappo')})
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
for name in ('jobs','specs','tasks'): (root/name).mkdir()
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
sys.path.insert(0,str(stage));os.chdir(stage)
log=io.StringIO()
try:
 with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  from tools.c1switch.sumo import setup
  inputs=setup()
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


def canary(host, root):
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
    if any('CANARY_DISPATCH_ONCE.json' in r or 'DISPATCH_CANARY.json' in r for r in current['roots']):
        raise ValueError('canary previously claimed: inspect/reuse, never duplicate')
    result = ssh(host, f'''import hashlib,json,os,subprocess,sys,sysconfig,time
from pathlib import Path
root=Path({root!r});stage=root/'pytools';assert root.is_dir()
binding=json.loads((root/'BINDING.json').read_text())
assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==v for n,v in binding['sources'].items())
preflight=json.loads((root/'PREFLIGHT.json').read_text());assert preflight['status']=='PASS'
assert os.getloadavg()[0]<28,'yield to host pressure'
assert not (root/'STOP').exists();assert not (root/'CANARY_DISPATCH_ONCE.json').exists()
for sibling in root.parent.iterdir():
 if sibling.name.startswith('c1switch2_run_') and sibling.is_dir():
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('status', 'stage', 'preflight', 'probe', 'canary'))
    parser.add_argument('--host', choices=('cpuQ', 'gpuR'), required=True)
    parser.add_argument('--root')
    parser.add_argument('--model', choices=('wm', 'bb'))
    parser.add_argument('--steps', type=int, default=120)
    args = parser.parse_args()
    if args.command == 'status':
        print(json.dumps(status(args.host), indent=2))
    elif args.command == 'stage':
        stage(args.host)
    elif args.command == 'preflight':
        preflight(args.host, args.root)
    elif args.command == 'canary':
        canary(args.host, args.root)
    else:
        probe(args.host, args.root, args.model, args.steps)


if __name__ == '__main__':
    main()
