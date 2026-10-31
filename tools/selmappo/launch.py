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
record=dict(root=str(root),source=str(source),binding={binding!r},design_sha256={sha!r},tests_returncode=tests.returncode,tests_stdout=tests.stdout,tests_stderr=tests.stderr,gate_returncode=gate.returncode,gate_stdout=gate.stdout,gate_stderr=gate.stderr,new_sumo_seconds=0)
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


def dispatch(build, continuation=False):
    result = run('cpuQ', f'''
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path.home()/'selmappo20261031'/{build!r}
assert root.parent.is_dir() and root.is_dir() and root.name.startswith('build_')
stage=json.loads((root/'STAGE.json').read_text());assert stage['tests_returncode']==0 and stage['gate_returncode']==0
receipt=json.loads((root/'INTEGRATION.json').read_text());env=json.loads((root/'ENV.json').read_text())
assert os.getloadavg()[0]<24
assert __import__('shutil').disk_usage(root).free>=2*1024**3
for name,expected in receipt['files'].items():
 assert hashlib.sha256((root/'pytools'/name).read_bytes()).hexdigest()==expected
continuation={continuation!r}
original=root.parent/'build_bc4c3aae72073204'
if continuation:
 sys.path.insert(0,str(root/'pytools'))
 from tools.selmappo.continuation import validate
 validate(original,root/'pytools/REGISTRATION.md')
 prior=json.loads((root.parent/'FLEET_DISPATCH_ONCE.json').read_text())
 assert prior['build']==str(original)
lockname='FLEET_CONTINUATION_ONCE.json' if continuation else 'FLEET_DISPATCH_ONCE.json'
with (root.parent/lockname).open('x') as lock:
 json.dump(dict(build=str(root),binding=stage['binding']),lock);lock.flush();os.fsync(lock.fileno())
out=root/'training';assert not out.exists()
command=[sys.executable,'-m','tools.selmappo.train','--variant','fleet','--registration',str(root/'pytools/REGISTRATION.md'),'--c1-root',str(root/'pytools'),'--integration',str(root/'INTEGRATION.json'),'--design-sha',receipt['design_sha256'],'--out',str(out),'--workers','12']
if continuation:command+=['--continue-from',str(original)]
log=(root/'training.log').open('x')
def lower():os.nice(19)
proc=subprocess.Popen(command,cwd=root/'pytools',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lower)
record=dict(pid=proc.pid,root=str(root),out=str(out),binding=stage['binding'],command=command)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('stage', 'dispatch', 'continue', 'status', 'reconcile'))
    parser.add_argument('--build')
    args = parser.parse_args()
    if args.action == 'stage':
        stage()
    elif args.action == 'dispatch':
        dispatch(args.build)
    elif args.action == 'continue':
        dispatch(args.build, continuation=True)
    elif args.action == 'reconcile':
        reconcile(args.build)
    else:
        status(args.build)
