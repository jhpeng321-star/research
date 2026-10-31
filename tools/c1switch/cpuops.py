"""Immutable cpuQ continuation, isolated from all old gpuR dispatches."""
import argparse
import base64
import io
import json
from pathlib import Path
import tarfile

from .design import BUDGET, sha
from .ops import EVIDENCE, REG, ssh
from .storage import write
from .storage import canonical

REMOTE = '/home/ubuntu/c1switch20261031/fullrerun_v3'
REG_MARK = 'S-C1SWITCH cpuQ v3 REGISTERED BEFORE DISPATCH'
ENV = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', C1SWITCH_RUNTIME='/home/ubuntu/e3final',
           C1SWITCH_WRAPPERS='/home/ubuntu/wmdata20261031',
           C1SWITCH_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',
           C1SWITCH_ENGINE_ROOT=REMOTE,
           PATH='/home/ubuntu/miniconda3/bin:/usr/local/bin:/usr/bin:/bin')


def registered():
    digest = sha('docs/paper_results/C1SWITCH_DESIGN.md')
    if not any(REG_MARK in line and digest in line for line in REG.read_text(encoding='utf-8').splitlines()):
        raise ValueError('cpuQ design registration missing')
    return digest


def inspect():
    # Own receipt namespaces only; never enumerate or signal peer processes.
    old = ssh('gpuR', '''import json
from pathlib import Path
root=Path('/root/c1switch20261031'); version=root/'fullrerun_v2'
print(json.dumps(dict(legacy_stop=(root/'STOP').exists(),
 dispatch=(version/'DISPATCH.json').exists(),locks=[str(p) for p in root.glob('**/DISPATCH.lock')],
 ledger_sha256=__import__('hashlib').sha256((root/'accounting.jsonl').read_bytes()).hexdigest(),
 charged_sim_s=sum(json.loads(s)['sim_s'] for s in (root/'accounting.jsonl').read_text().splitlines()))))
''')
    current = ssh('cpuQ', f'''import json,os
from pathlib import Path
root=Path({REMOTE!r});out=dict(load=os.getloadavg(),version_exists=root.exists())
for name in ['DISPATCH.json','STATUS.json','STOP','TRUNK_GATE.json','STEP0_GATE.json']:
 p=root/name
 if p.exists():out[name]=json.loads(p.read_text())
if (root/'DISPATCH.json').exists():
 pid=json.loads((root/'DISPATCH.json').read_text())['pid']
 out['own_pid_exists']=Path('/proc',str(pid)).exists()
print(json.dumps(out))
''')
    result = dict(gpuR=old, cpuQ=current)
    write(EVIDENCE/'CPUQ_INSPECT.json', result)
    print(json.dumps(result, indent=2))
    return result


def stage():
    digest = registered()
    inspected = inspect()
    old = inspected['gpuR']
    if old['dispatch'] or old['locks'] or old['charged_sim_s'] != 930 or inspected['cpuQ']['version_exists']:
        raise ValueError('existing dispatch or unresolved accounting; no restage')
    files = sorted(Path('tools/c1switch').glob('*.py')) + [
        Path('tests/test_c1switch.py'), Path('docs/paper_results/C1SWITCH_DESIGN.md')]
    sources = {p.as_posix():sha(p) for p in files}
    engine = Path('traffic_shepherd/c4/branching.py')
    sources['engine/branching.py'] = sha(engine)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for p in files:
            tar.add(p, arcname=p.as_posix())
        tar.add(engine, arcname='engine/branching.py')
    raw = base64.b64encode(buf.getvalue()).decode()
    debit = dict(id='retained_legacy_930', sim_s=930, stage='step0',
                 legacy_ledger_sha256=old['ledger_sha256'])
    binding = dict(design_sha256=digest, sources=sources, per_arm_sim_s=BUDGET,
                   step0_total_sim_s=47010, new_step0_sim_s=46080,
                   main_authorized=False, retained_debit=debit)
    exposure = json.loads((EVIDENCE/'EXPOSURE.json').read_text())
    assert exposure['pass']
    result = ssh('cpuQ', f'''import base64,hashlib,io,json,os,tarfile
from pathlib import Path
root=Path({REMOTE!r});assert root.parent.parent.is_dir()
assert not root.parent.exists(), 'existing task namespace requires reconciliation'
root.parent.mkdir();root.mkdir();stage=root/'pytools';stage.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({raw!r})),mode='r:gz') as tar:tar.extractall(stage,filter='data')
binding={binding!r}
assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==h for n,h in binding['sources'].items())
(root/'BINDING.json').write_text(json.dumps(binding,indent=2))
(root/'STAGED.json').write_text(json.dumps(binding['sources'],indent=2))
(root/'EXPOSURE.json').write_text(json.dumps({exposure!r},indent=2))
with (root/'accounting.jsonl').open('x') as f:f.write(json.dumps(binding['retained_debit'])+'\\n');f.flush();os.fsync(f.fileno())
print(json.dumps(dict(root=str(root),binding=binding,new_sim_s=0)))
''')
    write(EVIDENCE/'CPUQ_STAGE.json', result)
    print(json.dumps(dict(root=REMOTE, files=len(sources), design_sha256=digest)))


def cpu():
    result = ssh('cpuQ', f'''import contextlib,io,json,os,sys
from pathlib import Path
root=Path({REMOTE!r});stage=root/'pytools';assert stage.is_dir()
os.nice(19);os.environ.update({ENV!r});sys.path.insert(0,str(stage))
from tools.c1switch.sumo import setup
setup()
import traci
def forbidden(*a,**kw):raise AssertionError('CPU verification attempted SUMO')
traci.start=forbidden;traci.simulationStep=forbidden
from traffic_shepherd import simulation
simulation.start_sumo=forbidden
from tools.c1switch.design import guard,worlds
panel=worlds('step0')+worlds('main')
for w in panel:guard(w)
import pytest
log=io.StringIO()
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
 code=pytest.main(['-q',str(stage/'tests/test_c1switch.py')])
out=dict(returncode=code,output=log.getvalue(),guarded=len(panel),new_sim_s=0)
(root/'CPU.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
''', timeout=90)
    write(EVIDENCE/'CPUQ_CPU.json', result)
    print(json.dumps(result, indent=2))


def launch():
    digest = registered()
    old = inspect()['gpuR']
    assert not old['dispatch'] and not old['locks'] and old['charged_sim_s'] == 930
    result = ssh('cpuQ', f'''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path({REMOTE!r});stage=root/'pytools'
assert not any((root/n).exists() for n in ['DISPATCH.json','DISPATCH.lock','STOP','worker.log'])
binding=json.loads((root/'BINDING.json').read_text())
assert binding['design_sha256']=={digest!r}
assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==h for n,h in binding['sources'].items())
assert json.loads((root/'CPU.json').read_text())['returncode']==0
assert json.loads((root/'EXPOSURE.json').read_text())['pass']
assert os.getloadavg()[0]<=20
env=dict(os.environ);env.update({ENV!r})
with (root/'worker.log').open('x') as log:
 p=subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch.fullrun'],cwd=stage,env=env,
  stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
 out=dict(pid=p.pid,unix=time.time(),host='cpuQ',version=str(root),workers=1,
          new_max_sim_s=46080,total_step0_max_sim_s=47010,main_authorized=False)
 (root/'DISPATCH.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
''')
    write(EVIDENCE/'CPUQ_DISPATCH.json', result)
    print(json.dumps(result, indent=2))


def progress():
    result = ssh('cpuQ', f'''import json,os
from pathlib import Path
root=Path({REMOTE!r});out=dict(load=os.getloadavg())
for name in ['DISPATCH.json','STATUS.json','STOP','TRUNK_GATE.json','STEP0_GATE.json']:
 p=root/name
 if p.exists():out[name]=json.loads(p.read_text())
if (root/'DISPATCH.json').exists():
 pid=json.loads((root/'DISPATCH.json').read_text())['pid'];out['own_pid_exists']=Path('/proc',str(pid)).exists()
out['charged_sim_s']=sum(json.loads(s)['sim_s'] for s in (root/'accounting.jsonl').read_text().splitlines())
out['completed_states']=len(list(root.glob('s*_t*.json')))
closes=[json.loads(p.read_text()) for p in root.glob('jobs/*/CLOSE.json')]
out['closed_runs']=len(closes);out['closed_actual_sim_s_incl_legacy']=930+sum(r['actual_sim_s'] for r in closes)
if (root/'STOP').exists():out['worker.log']=(root/'worker.log').read_text()
print(json.dumps(out))
''')
    write(EVIDENCE/'CPUQ_PROGRESS.json', result)
    print(json.dumps(result, indent=2))


def collect():
    """Receipt-only export; conditions use the observable native belief only."""
    result = ssh('cpuQ', f'''import hashlib,json,statistics
from pathlib import Path
root=Path({REMOTE!r});out=dict(states=[],sources={{}})
for p in sorted(root.glob('s*_t*.json')):
 raw=p.read_bytes();row=json.loads(raw);out['sources'][p.name]=hashlib.sha256(raw).hexdigest()
 belief=row['branches']['native']['belief'];h=belief['history'][-1]
 seen={{r['vehicle_id']:r for r in h['detections']}}
 seen.update({{r['vehicle_id']:r for r in h['self_reports']}})
 own={{r['vehicle_id'] for r in h['self_reports']}}
 near=[r for r in seen.values() if -100<=r['s_m']<=75.44]
 speeds=[r['v_mps'] for r in near]
 row['observables']=dict(near_count=len(near),near_mean_speed_mps=statistics.mean(speeds) if speeds else None,
  observed_queue_v_lt5=sum(r['v_mps']<5 for r in near),cav_self_count=len(own),
  observed_total_count=len(seen),observed_cav_fraction=len(own)/len(seen) if seen else None,
  oldest_receipt_age_s=max((r['age_s'] for r in seen.values()),default=0),coverage=h['coverage'])
 for b in row['branches'].values():b.pop('belief')
 out['states'].append(row)
for n in ['BINDING.json','STEP0_GATE.json','STATUS.json','STOP','TRUNK_GATE.json','DISPATCH.json']:
 p=root/n
 if p.exists():out[n]=json.loads(p.read_text());out['sources'][n]=hashlib.sha256(p.read_bytes()).hexdigest()
closes=[json.loads(p.read_text()) for p in root.glob('jobs/*/CLOSE.json')]
out['closed_actual_sim_s_incl_legacy']=930+sum(r['actual_sim_s'] for r in closes)
out['charged_sim_s']=sum(json.loads(s)['sim_s'] for s in (root/'accounting.jsonl').read_text().splitlines())
print(json.dumps(out))
''')
    write(EVIDENCE/'CPUQ_COLLECT.json', result)
    print(json.dumps(dict(states=len(result['states']), charged_sim_s=result['charged_sim_s'],
                          gate=result.get('STEP0_GATE.json'), stopped='STOP' in result), indent=2))


def verify_build():
    """CPU-only candidate verification beside, never inside, the running deployment."""
    paths = sorted(Path('tools/c1switch').glob('*.py')) + [
        Path('tests/test_c1switch.py'),Path('tests/test_c1switch_observable.py')]
    sources = {p.as_posix():sha(p) for p in paths}
    version = canonical(sources)[:12]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as tar:
        for p in paths:tar.add(p,arcname=p.as_posix())
    raw = base64.b64encode(buf.getvalue()).decode()
    result = ssh('cpuQ', f'''import base64,contextlib,hashlib,io,json,os,sys,tarfile
from pathlib import Path
root=Path({REMOTE!r}).parent/('cpucheck_'+{version!r});assert root.parent.is_dir()
if root.exists():
 assert (root/'CPU.json').exists(),'incomplete CPU check, reconcile instead of repeating'
 print((root/'CPU.json').read_text());raise SystemExit
root.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({raw!r})),mode='r:gz') as tar:tar.extractall(root,filter='data')
sources={sources!r};assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in sources.items())
os.nice(19);os.environ.update({ENV!r});sys.path.insert(0,str(root))
from tools.c1switch.sumo import setup
setup()
import traci
def forbidden(*a,**kw):raise AssertionError('candidate CPU check attempted SUMO')
traci.start=forbidden;traci.simulationStep=forbidden
from traffic_shepherd import simulation
simulation.start_sumo=forbidden
import pytest
log=io.StringIO()
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
 code=pytest.main(['-q',str(root/'tests/test_c1switch.py'),str(root/'tests/test_c1switch_observable.py')])
out=dict(returncode=code,output=log.getvalue(),sources=sources,new_sim_s=0,device='cpu',root=str(root))
(root/'CPU.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
''', timeout=90)
    write(EVIDENCE/'CPUQ_BUILD_CPU.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('inspect','stage','cpu','launch','progress','collect','verify_build'))
    globals()[parser.parse_args().mode]()
