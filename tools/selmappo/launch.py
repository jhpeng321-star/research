"""Explicit stage/dispatch/status; immutable sources and one exclusive dispatch."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

from .integration import registered
from .ops import REG
from .protocol import CADENCE, LIBRARY, PROTOCOL, digest, safe_path
from .remote import run

WALL = Path('E:/agent-work/c4-wallsw')
EVIDENCE = Path('docs/paper_results/selmappo')


def bundle():
    files = {p.as_posix(): safe_path(p).read_text(encoding='utf-8')
             for p in sorted(Path('tools/selmappo').glob('*.py'))}
    peer = ['tools/c1switch/'+name+'.py' for name in ('__init__', 'core', 'design', 'storage', 'sumo', 'library')]
    peer += ['traffic_shepherd/c4/switchable_wall.py', 'src/wms/control/wall.py']
    for name in peer:
        files[name] = safe_path(WALL/name).read_text(encoding='utf-8')
    files['WALLSW_HANDOFF.md'] = safe_path(WALL/'HANDOFF.md').read_text(encoding='utf-8')
    sources = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}
    binding = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return files, sources, binding


def stage():
    sha = digest('docs/paper_results/SELMAPPO_DESIGN.md')
    registered(REG, sha)
    files, sources, binding = bundle()
    files['REGISTRATION.md'] = safe_path(REG).read_text(encoding='utf-8')
    raw = base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    proposed = dict(schema='selmappo.integration/2', status='PROPOSED', protocol=PROTOCOL,
                    cadence_s=CADENCE, library=list(LIBRARY), design_sha256=sha,
                    wall_module='tools.c1switch.library', files={n:h for n,h in sources.items()
                                                                 if not n.startswith('tools/selmappo/')})
    result = run('cpuQ', f'''
import base64,hashlib,json,os,subprocess,sys,zlib
from pathlib import Path
home=Path.home();parent=home/'selmappo20261031';assert home.is_dir()
parent.mkdir(exist_ok=True)
root=parent/'build_{binding[:16]}'
assert not root.exists(),'existing stage: inspect, never restage/relaunch'
root.mkdir();source=root/'pytools';source.mkdir()
files=json.loads(zlib.decompress(base64.b64decode(sys.stdin.buffer.read())))
for name,text in files.items():
 p=source/name;assert p.is_relative_to(source);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:f.write(text)
for name,expected in {sources!r}.items():
 assert hashlib.sha256((source/name).read_bytes()).hexdigest()==expected
sumo=home/'miniconda3/lib/python3.10/site-packages/sumo'
assert (sumo/'bin/sumo').is_file()
env=dict(os.environ,PYTHONPATH=str(source)+':'+str(home/'e3final/runtime/src'),SELMAPPO_TEST_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME=str(sumo),PATH=str(sumo/'bin')+':'+str(home/'miniconda3/bin')+':/usr/local/bin:/usr/bin:/bin',C1SWITCH_RUNTIME=str(home/'e3final'),C1SWITCH_WRAPPERS=str(home/'wmdata20261031'),C1SWITCH_MAPPO=str(home/'mappospeed/production2/policy_00169.pt'),C1SWITCH_ENGINE_ROOT=str(home/'c1switch20261031/fullrerun_v3'))
os.nice(19)
tests=subprocess.run([sys.executable,'-m','unittest','tools.selmappo.test_cpu','-v'],cwd=source,env=env,capture_output=True,text=True,timeout=90)
print(tests.stdout);print(tests.stderr)
gate=subprocess.run([sys.executable,'-m','tools.selmappo.gate','--root',str(source)],cwd=source,env=env,capture_output=True,text=True,timeout=90)
print(gate.stdout);print(gate.stderr)
record=dict(root=str(root),source=str(source),binding={binding!r},source_sha256={sources!r},design_sha256={sha!r},tests_returncode=tests.returncode,tests_stdout=tests.stdout,tests_stderr=tests.stderr,gate_returncode=gate.returncode,gate_stdout=gate.stdout,gate_stderr=gate.stderr,new_sumo_seconds=0)
with (root/'STAGE.json').open('x') as f:json.dump(record,f,indent=2)
with (root/'ENV.json').open('x') as f:json.dump(env,f,indent=2)
if tests.returncode==0 and gate.returncode==0:
 receipt={proposed!r};receipt['status']='CPU_INTEGRATION_PASS'
 with (root/'INTEGRATION.json').open('x') as f:json.dump(receipt,f,indent=2)
print('STAGE_RECEIPT',json.dumps(record))
sys.exit(0 if tests.returncode==0 and gate.returncode==0 else 1)
''', stdin=raw, timeout=240)
    save_tag(result.stdout, 'STAGE_RECEIPT', f'STAGE_{binding[:16]}.json')


def save_tag(stdout, tag, name):
    rows = [line[len(tag)+1:] for line in stdout.splitlines() if line.startswith(tag+' ')]
    if rows:
        assert EVIDENCE.is_dir()
        with (EVIDENCE/name).open('x', encoding='utf-8') as stream:
            json.dump(json.loads(rows[-1]), stream, indent=2)


def dispatch(build, continuation=False, production_only=False, transport_only=False,
              wait_for_resources=False, history_only=False):
    if wait_for_resources and not (transport_only or history_only):
        raise ValueError('resource-wait supervisor requires an acknowledged production continuation')
    result = run('cpuQ', f'''
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path.home()/'selmappo20261031'/{build!r}
assert root.parent.is_dir() and root.is_dir() and root.name.startswith('build_')
stage=json.loads((root/'STAGE.json').read_text());assert stage['tests_returncode']==0 and stage['gate_returncode']==0
receipt=json.loads((root/'INTEGRATION.json').read_text());env=json.loads((root/'ENV.json').read_text())
if not {wait_for_resources!r}:
 assert os.getloadavg()[0]+1<=.8*os.cpu_count(),'yield to higher-priority runs'
 assert __import__('shutil').disk_usage(root).free>=2*1024**3
for name,expected in receipt['files'].items():
 assert hashlib.sha256((root/'pytools'/name).read_bytes()).hexdigest()==expected
if 'source_sha256' in stage:
 for name,expected in stage['source_sha256'].items():
  assert hashlib.sha256((root/'pytools'/name).read_bytes()).hexdigest()==expected
continuation={continuation!r}
production_only={production_only!r}
transport_only={transport_only!r}
history_only={history_only!r}
assert sum((continuation,production_only,transport_only,history_only))<=1
original=root.parent/'build_bc4c3aae72073204'
if continuation:
 sys.path.insert(0,str(root/'pytools'))
 from tools.selmappo.continuation import validate
 validate(original,root/'pytools/REGISTRATION.md')
 prior=json.loads((root.parent/'FLEET_DISPATCH_ONCE.json').read_text())
 assert prior['build']==str(original)
if production_only:
 sys.path.insert(0,str(root/'pytools'))
 from tools.selmappo.continuation import validate_production,TERMINAL
 from tools.selmappo.protocol import audit_registration
 assert audit_registration(root/'pytools/REGISTRATION.md')['passed']
 original=root.parent/TERMINAL
 validate_production(original,root/'pytools/REGISTRATION.md')
 assert json.loads((root.parent/'FLEET_CONTINUATION_ONCE.json').read_text())['build']==str(original)
 assert json.loads((root.parent/'FLEET_DISPATCH_ONCE.json').read_text())['build']==str(root.parent/'build_bc4c3aae72073204')
 assert stage.get('source_sha256'),'all current source hashes required'
 for other in root.parent.glob('build_*/DISPATCH.json'):
  old=json.loads(other.read_text())
  assert not (Path('/proc')/str(old['pid'])).exists(),'prior fleet trainer PID still exists'
if transport_only:
 sys.path.insert(0,str(root/'pytools'))
 from tools.selmappo.continuation import validate_transport,TRANSPORT
 from tools.selmappo.protocol import audit_registration
 from tools.selmappo.dispatch import admitted
 assert audit_registration(root/'pytools/REGISTRATION.md')['passed']
 if not {wait_for_resources!r}:assert admitted(root,0),'yield to higher-priority runs or low memory/disk'
 original=root.parent/TRANSPORT
 validate_transport(original,root/'pytools/REGISTRATION.md')
 assert json.loads((root.parent/'FLEET_PRODUCTION_ONCE.json').read_text())['build']==str(original)
 assert stage.get('source_sha256'),'all source hashes required'
 for other in root.parent.glob('build_*/DISPATCH.json'):
  old=json.loads(other.read_text())
  assert not (Path('/proc')/str(old['pid'])).exists(),'prior fleet trainer PID still exists'
if history_only:
 sys.path.insert(0,str(root/'pytools'))
 from tools.selmappo.history_continuation import validate,HISTORY
 from tools.selmappo.protocol import audit_registration
 from tools.selmappo.dispatch import admitted
 assert audit_registration(root/'pytools/REGISTRATION.md')['passed']
 if not {wait_for_resources!r}:assert admitted(root,0),'yield to higher-priority runs or low memory/disk'
 original=root.parent/HISTORY
 validate(original,root/'pytools/REGISTRATION.md')
 assert json.loads((root.parent/'FLEET_TRANSPORT_CONTINUATION_ONCE.json').read_text())['build']==str(original)
 assert stage.get('source_sha256'),'all source hashes required'
 for other in root.parent.glob('build_*/DISPATCH.json'):
  old=json.loads(other.read_text())
  assert not (Path('/proc')/str(old['pid'])).exists(),'prior fleet trainer PID still exists'
if transport_only or history_only:
 for proc in Path('/proc').iterdir():
  if proc.name.isdigit():
   try:cmd=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
   except (OSError,ProcessLookupError):continue
   assert str(original) not in cmd or '-c import base64' in cmd,'prior owned descendant still live'
lockname=('FLEET_HISTORY_CONTINUATION_ONCE.json' if history_only else
          'FLEET_TRANSPORT_CONTINUATION_ONCE.json' if transport_only else
          'FLEET_PRODUCTION_ONCE.json' if production_only else
          'FLEET_CONTINUATION_ONCE.json' if continuation else 'FLEET_DISPATCH_ONCE.json')
with (root.parent/lockname).open('x') as lock:
 json.dump(dict(build=str(root),binding=stage['binding']),lock);lock.flush();os.fsync(lock.fileno())
out=root/'training';assert not out.exists()
command=[sys.executable,'-m','tools.selmappo.train','--variant','fleet','--registration',str(root/'pytools/REGISTRATION.md'),'--c1-root',str(root/'pytools'),'--integration',str(root/'INTEGRATION.json'),'--design-sha',receipt['design_sha256'],'--out',str(out),'--workers','12']
if continuation:command+=['--continue-from',str(original)]
if production_only:command+=['--production-from',str(original)]
if transport_only:command+=['--transport-from',str(original)]
if history_only:command+=['--history-from',str(original)]
log=(root/'training.log').open('x')
def lower():os.nice(19)
proc=subprocess.Popen(command,cwd=root/'pytools',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lower)
record=dict(pid=proc.pid,root=str(root),out=str(out),binding=stage['binding'],command=command,
 resource_wait_supervisor={wait_for_resources!r},
 startup_disk_free_bytes=__import__('shutil').disk_usage(root).free,
 startup_load=os.getloadavg(),physics_admission='each episode before precharge; unchanged')
with (root/'DISPATCH.json').open('x') as f:json.dump(record,f,indent=2)
print('DISPATCH_RECEIPT',json.dumps(record))
''')
    save_tag(result.stdout, 'DISPATCH_RECEIPT', f'DISPATCH_{build}.json')


def status(build):
    run('cpuQ', f'''
import json
from pathlib import Path
root=Path.home()/'selmappo20261031'/{build!r};assert root.is_dir()
record=json.loads((root/'DISPATCH.json').read_text());p=Path('/proc')/str(record['pid'])
print('OWN_PID',record['pid'],'exists',p.exists())
if p.exists():print('CMDLINE',(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode())
for name in ('training.log','training/ledger.jsonl','training/training.jsonl','training/status.jsonl','training/failure.jsonl'):
 path=root/name
 if path.exists():print(name,path.read_text()[-14000:])
''')


def inventory():
    run('cpuQ', '''
import json,os
from pathlib import Path
root=Path.home()/'selmappo20261031';assert root.is_dir()
print('RESOURCE',json.dumps(dict(load=os.getloadavg(),cores=os.cpu_count(),max_load_fraction=.8)))
for path in sorted(root.glob('*.json')):print('TASK_LOCK',path.name,path.read_text())
for path in sorted(root.glob('build_*/DISPATCH.json')):
 record=json.loads(path.read_text());proc=Path('/proc')/str(record['pid'])
 print('FLEET_DISPATCH',str(path),json.dumps(record),'pid_exists',proc.exists())
 if proc.exists():print('CMDLINE',(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode())
''')


def reconcile(build):
    result = run('cpuQ', f'''
import hashlib,json
from pathlib import Path
root=Path.home()/'selmappo20261031'/{build!r};assert root.is_dir()
dispatch=json.loads((root/'DISPATCH.json').read_text())
assert not (Path('/proc')/str(dispatch['pid'])).exists(),'own trainer still alive: no reconciliation'
ledger=root/'training/ledger.jsonl';rows=[json.loads(s) for s in ledger.read_text().splitlines()]
assert len(rows)==2 and rows[0]['event']=='START' and rows[1]['event']=='END'
assert rows[0]['id']==rows[1]['id']=='fleet:18007998'
assert rows[0]['reserved_sim_s']==900 and rows[1]['actual_sim_s']==0
record=dict(status='STOPPED_BEFORE_PHYSICS',own_pid=dispatch['pid'],own_pid_exited=True,charged_sim_s=900,actual_sim_s=0,ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),error='frozen env.check_case captured alias rejected fresh TRAIN seed',training_episodes=0,checkpoints=0,retry_authorized=False,root=str(root))
if not (root/'STOP.json').exists():
 with (root/'STOP.json').open('x') as f:json.dump(record,f,indent=2)
else:assert json.loads((root/'STOP.json').read_text())==record
print('STOP_RECEIPT',json.dumps(record))
''')
    save_tag(result.stdout, 'STOP_RECEIPT', f'STOP_{build}.json')


def terminal_stop(build):
    result = run('cpuQ', f'''
import hashlib,json
from pathlib import Path
root=Path.home()/'selmappo20261031'/{build!r};assert root.is_dir()
dispatch=json.loads((root/'DISPATCH.json').read_text())
assert not (Path('/proc')/str(dispatch['pid'])).exists(),'own trainer still alive'
ledger=root/'training/ledger.jsonl';rows=[json.loads(s) for s in ledger.read_text().splitlines()]
assert len(rows)==4
assert [r['event'] for r in rows]==['START','END','START','END']
assert [r['id'] for r in rows]==['fleet:18007998']*2+['fleet:18007999']*2
assert rows[0]['reserved_sim_s']==rows[2]['reserved_sim_s']==900
assert rows[1]['actual_sim_s']==0 and rows[3]['actual_sim_s']==900
assert not list((root/'training').glob('checkpoint_*.pt'))
assert not (root/'training/training.jsonl').exists()
log=(root/'training.log').read_text()
assert "AttributeError: 'NoneType' object has no attribute 'simulation'" in log
record=dict(status='STOPPED_AT_MECHANICS_TERMINAL',own_pid=dispatch['pid'],own_pid_exited=True,
 charged_sim_s=1800,actual_sim_s=900,ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
 log_sha256=hashlib.sha256((root/'training.log').read_bytes()).hexdigest(),
 error='frozen environment closes api before returning terminal step; wrapper queried env.now',
 training_episodes=0,checkpoints=0,retry_authorized=False,root=str(root),
 result_complete=False,mechanics_slots_remaining=0)
with (root/'STOP.json').open('x') as f:json.dump(record,f,indent=2)
print('STOP_RECEIPT',json.dumps(record))
''')
    save_tag(result.stdout, 'STOP_RECEIPT', f'STOP_{build}.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('stage', 'dispatch', 'continue', 'production', 'inventory',
                                          'transport', 'history', 'status', 'reconcile', 'terminal-stop'))
    parser.add_argument('--build')
    parser.add_argument('--wait-for-resources', action='store_true')
    args = parser.parse_args()
    if args.action == 'stage':
        stage()
    elif args.action == 'dispatch':
        dispatch(args.build)
    elif args.action == 'continue':
        dispatch(args.build, continuation=True)
    elif args.action == 'production':
        dispatch(args.build, production_only=True)
    elif args.action == 'transport':
        dispatch(args.build, transport_only=True, wait_for_resources=args.wait_for_resources)
    elif args.action == 'history':
        dispatch(args.build, history_only=True, wait_for_resources=args.wait_for_resources)
    elif args.action == 'inventory':
        inventory()
    elif args.action == 'reconcile':
        reconcile(args.build)
    elif args.action == 'terminal-stop':
        terminal_stop(args.build)
    else:
        status(args.build)
