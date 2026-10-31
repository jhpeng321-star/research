"""Short, immutable full-rerun operations; no legacy redispatch/STOP clearing."""
import argparse
import base64
import io
import json
from pathlib import Path
import tarfile

from .design import sha
from .ops import EVIDENCE, REG, ROOT, ssh
from .storage import write

VERSION = 'fullrerun_v2'
REMOTE = ROOT+'/'+VERSION


def inspect():
    result = ssh('gpuR', f'''import json, os
from pathlib import Path
root=Path({ROOT!r})
out=dict(load=os.getloadavg(), own_pids={{str(p):Path('/proc',str(p)).exists() for p in [523094,524236]}},
 active_locks=[str(p) for p in root.glob('**/DISPATCH.lock')],
 version_exists=(root/{VERSION!r}).exists())
print(json.dumps(out))
''')
    write(EVIDENCE/'FULL_INSPECT.json', result)
    print(json.dumps(result, indent=2))


def stage():
    reg = REG.read_text(encoding='utf-8')
    design_sha = sha('docs/paper_results/C1SWITCH_DESIGN.md')
    if not any('S-C1SWITCH FULL-RERUN v2 REGISTERED BEFORE DISPATCH' in s and design_sha in s
               for s in reg.splitlines()):
        raise ValueError('design SHA not registered')
    files = sorted(Path('tools/c1switch').glob('*.py'))+[
        Path('tests/test_c1switch.py'), Path('docs/paper_results/C1SWITCH_DESIGN.md')]
    sources = {p.as_posix():sha(p) for p in files}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for p in files:
            tar.add(p, arcname=p.as_posix())
    raw = base64.b64encode(buf.getvalue()).decode()
    result = ssh('gpuR', f'''import base64, hashlib, io, json, os, tarfile
from pathlib import Path
root=Path({ROOT!r}); assert root.is_dir() and (root/'STOP').exists()
assert not (root/'DISPATCH.lock').exists()
for pid in [523094,524236]: assert not Path('/proc',str(pid)).exists(), 'old own PID alive/reused; inspect before proceeding'
rows=[json.loads(x) for x in (root/'accounting.jsonl').read_text().splitlines()]
assert sum(r['sim_s'] for r in rows)==930
assert not list(root.glob('{VERSION}/**/DISPATCH.lock'))
version=root/{VERSION!r}; version.mkdir(exist_ok=False)
stage=version/'pytools';stage.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({raw!r})),mode='r:gz') as tar:
 tar.extractall(stage,filter='data')
sources={sources!r}
assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==h for n,h in sources.items())
debit=dict(id='retained_legacy_930',sim_s=930,stage='step0',legacy_ledger_sha256=hashlib.sha256((root/'accounting.jsonl').read_bytes()).hexdigest())
binding=dict(design_sha256={design_sha!r},sources=sources,step0_total_sim_s=47010,new_step0_sim_s=46080,main_authorized=False,retained_debit=debit)
(version/'BINDING.json').write_text(json.dumps(binding,indent=2))
with (version/'accounting.jsonl').open('x') as f:f.write(json.dumps(debit)+'\\n');f.flush();os.fsync(f.fileno())
(version/'EXPOSURE.json').write_bytes((root/'EXPOSURE.json').read_bytes())
print(json.dumps(dict(version=str(version),binding=binding,load=os.getloadavg(),legacy_STOP_retained=True)))
''')
    write(EVIDENCE/'FULL_STAGE.json', result)
    print(json.dumps(result, indent=2))


def cpu():
    result = ssh('gpuR', f'''import contextlib, io, json, os, sys
from pathlib import Path
root=Path({REMOTE!r}); stage=root/'pytools'; assert stage.is_dir()
os.nice(19);os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PATH']='/root/miniconda3/bin:'+os.environ['PATH']
sys.path.insert(0,str(stage))
from tools.c1switch.sumo import setup
setup()
import traci
def forbidden(*a,**kw):raise AssertionError('CPU verification attempted SUMO')
traci.start=forbidden;traci.simulationStep=forbidden
from traffic_shepherd import simulation
simulation.start_sumo=forbidden
from tools.c1switch.design import guard, worlds
for w in worlds('step0')+worlds('main'):guard(w)
import pytest
log=io.StringIO()
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
 code=pytest.main(['-q',str(stage/'tests/test_c1switch.py')])
out=dict(returncode=code,output=log.getvalue(),guarded=36,new_sim_s=0)
(root/'CPU.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
''', timeout=90)
    write(EVIDENCE/'FULL_CPU.json', result)
    print(json.dumps(result, indent=2))


def launch():
    design_sha = sha('docs/paper_results/C1SWITCH_DESIGN.md')
    if not any('S-C1SWITCH FULL-RERUN v2 REGISTERED BEFORE DISPATCH' in s and design_sha in s
               for s in REG.read_text(encoding='utf-8').splitlines()):
        raise ValueError('registration binding missing')
    result = ssh('gpuR', f'''import hashlib, json, os, subprocess, sys, time
from pathlib import Path
legacy=Path({ROOT!r});root=Path({REMOTE!r});stage=root/'pytools'
assert (legacy/'STOP').exists() and not (legacy/'DISPATCH.lock').exists()
assert not (root/'DISPATCH.json').exists() and not (root/'DISPATCH.lock').exists() and not (root/'STOP').exists()
binding=json.loads((root/'BINDING.json').read_text())
assert binding['design_sha256']=={design_sha!r}
assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==h for n,h in binding['sources'].items())
assert json.loads((root/'CPU.json').read_text())['returncode']==0
assert json.loads((root/'EXPOSURE.json').read_text())['pass']
assert os.getloadavg()[0]<=20, 'yield to SUMOCF; no dispatch under priority load'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PATH='/root/miniconda3/bin:'+os.environ['PATH'])
with (root/'worker.log').open('x') as log:
 p=subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch.fullrun'],cwd=stage,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
 out=dict(pid=p.pid,unix=time.time(),version=str(root),new_max_sim_s=46080,total_step0_max_sim_s=47010,workers=1,main_authorized=False)
 (root/'DISPATCH.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
''')
    write(EVIDENCE/'FULL_DISPATCH.json', result)
    print(json.dumps(result, indent=2))


def progress():
    result = ssh('gpuR', f'''import json, os
from pathlib import Path
root=Path({REMOTE!r});out=dict(load=os.getloadavg())
for name in ['DISPATCH.json','STATUS.json','STOP','TRUNK_GATE.json','STEP0_GATE.json']:
 p=root/name
 if p.exists():out[name]=json.loads(p.read_text())
if (root/'DISPATCH.json').exists():
 pid=json.loads((root/'DISPATCH.json').read_text())['pid']
 out['own_pid_exists']=Path('/proc',str(pid)).exists()
if (root/'accounting.jsonl').exists():
 rows=[json.loads(x) for x in (root/'accounting.jsonl').read_text().splitlines()]
 out['charged_sim_s']=sum(r['sim_s'] for r in rows)
out['completed_states']=len(list(root.glob('s*_t*.json')))
closes=[json.loads(p.read_text()) for p in root.glob('jobs/*/CLOSE.json')]
out['closed_runs']=len(closes);out['closed_actual_sim_s_incl_legacy']=930+sum(r['actual_sim_s'] for r in closes)
if (root/'STOP').exists() and (root/'worker.log').exists():out['worker.log']=(root/'worker.log').read_text()
print(json.dumps(out))
''')
    write(EVIDENCE/'FULL_PROGRESS.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('inspect','stage','cpu','launch','progress'))
    globals()[parser.parse_args().mode]()
