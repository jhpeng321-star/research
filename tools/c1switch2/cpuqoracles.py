"""Complete the original unstarted oracle sets of all four durable cpuQ WM episodes."""
import argparse
import json
from pathlib import Path

from tools.c1switch.storage import canonical, write
from .design import CONTROLLERS, REGISTRATION, measurement_indices
from .operations import EVIDENCE, env, ssh
from .wmoracles import SUPERVISOR

OLD = '/home/ubuntu/c1switch2_run_292bfffe94b13ed8'
PLAN_PATH = EVIDENCE/'CPUQ_WM_ORACLE_PLAN.json'
SEEDS = (969136, 969137, 969138, 969140)
CAP = 14640
SUPERVISOR_CPUQ = SUPERVISOR.replace('>20:', '>28:').replace('<=6000', '<=14640').replace(
    'branches=12,planned_sim_s=6000', 'branches=24,planned_sim_s=14640')


def validate(value):
    expected = {(seed, controller) for seed in SEEDS for controller in CONTROLLERS}
    if (value['predecessor'] != OLD or value['historical_sim_s'] != 5460 or
            value['new_sim_s'] != CAP or value['new_gpu_s'] != 0 or
            len(value['branches']) != 24 or
            {(b['seed'], b['controller']) for b in value['branches']} != expected):
        raise ValueError('exact original four completed worlds and 24 unstarted alternatives required')
    for branch in value['branches']:
        if branch['sim_s'] != 360+20*measurement_indices(branch['seed'])[0]:
            raise ValueError('original registered prefix cost required')


def probe():
    value = ssh('cpuQ', f'''import hashlib,json
from pathlib import Path
root=Path({OLD!r});d=json.loads((root/'DISPATCH_MODEL.json').read_text());assert d['pid']==649272
p=Path('/proc',str(d['pid']));assert not p.exists() or (p/'stat').read_text().split()[2]=='Z'
raw=(root/'accounting.jsonl').read_bytes();rows=[json.loads(x) for x in raw.splitlines()]
assert sum(r['sim_s'] for r in rows)==5460 and not any('__j' in r['id'] for r in rows)
assert (root/'STOP').is_file();branches=[]
for path in sorted((root/'specs').glob('*.json')):
 spec=json.loads(path.read_text());seed=spec['world']['seed'];assert seed in {SEEDS!r}
 assert (root/'jobs'/f'wm__s{{seed}}__episode/PHYSICS.json').is_file()
 assert not (root/'jobs'/spec['id']).exists()
 branches.append(dict(id=spec['id'],seed=seed,controller=spec['alternative'],sim_s=360+20*len(spec['schedule']),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
print(json.dumps(dict(predecessor=str(root),historical_sim_s=5460,predecessor_ledger_sha256=hashlib.sha256(raw).hexdigest(),branches=branches,new_sim_s=sum(b['sim_s'] for b in branches),new_gpu_s=0,not_before_unix=1793503800)))
''', timeout=30)
    validate(value)
    with PLAN_PATH.open('x') as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps(dict(plan_sha256=canonical(value), sim_s=CAP, branches=24)))


def stage_start():
    value = json.loads(PLAN_PATH.read_text());validate(value)
    text = Path(REGISTRATION).read_text(encoding='utf-8')
    if not any('S-C1SWITCH2 CPUQ WM ORACLES REGISTERED' in line and canonical(value) in line
               for line in text.splitlines()):
        raise ValueError('prospective exact original-oracle authority required')
    root = '/home/ubuntu/c1switch2_run_oracles_'+canonical(value)[:16]
    result = ssh('cpuQ', f'''import hashlib,json,os,shutil,subprocess,sys,sysconfig,time
from pathlib import Path
old=Path({OLD!r});root=Path({root!r});assert root.parent.is_dir() and not root.exists()
assert os.getloadavg()[0]<28 and shutil.disk_usage(root.parent).free>=2*1024**3
assert time.time()>={value['not_before_unix']!r}
assert hashlib.sha256((old/'accounting.jsonl').read_bytes()).hexdigest()=={value['predecessor_ledger_sha256']!r}
assert json.loads((old/'PREFLIGHT.json').read_text())['status']=='PASS'
sources=json.loads((old/'STAGED.json').read_text());root.mkdir()
for name,digest in sources.items():
 raw=(old/'pytools'/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
 target=root/'pytools'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
for name in ('BINDING.json','GATES.json','STAGED.json','PREFLIGHT.json','REGISTRATION.md'):(root/name).write_bytes((old/name).read_bytes())
for name in ('jobs','specs','tasks','logs'):(root/name).mkdir()
for item in {value['branches']!r}:
 raw=(old/'specs'/(item['id']+'.json')).read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256']
 (root/'specs'/(item['id']+'.json')).write_bytes(raw)
(root/'PLAN.json').write_text(json.dumps({value!r},indent=2))
(root/'ORACLE_SUPERVISOR.py').write_text({SUPERVISOR_CPUQ!r})
compile((root/'ORACLE_SUPERVISOR.py').read_text(),'<cpuq-original-oracles>','exec')
os.environ.update({env('cpuQ', root)!r});os.nice(19);sys.path.insert(0,str(root/'pytools'));os.chdir(root/'pytools')
from tools.c1switch2.runner import admitted
binding=admitted(root);assert binding['enabled_arms']==['wm'] and binding['cpu_host']=='cpuQ'
with (root.parent/'C1SWITCH2_CPUQ_WM_ORACLE_CONTINUE_ONCE.json').open('x') as stream:json.dump(dict(root=str(root),plan_sha256={canonical(value)!r},unix=time.time()),stream)
environment=dict(os.environ,PYTHONPATH=str(root/'pytools'))
environment['PATH']=str(Path(sysconfig.get_paths()['purelib'])/'sumo/bin')+':'+environment['PATH']
with (root/'ORACLE.log').open('x') as log:
 p=subprocess.Popen([sys.executable,'-B','-u',str(root/'ORACLE_SUPERVISOR.py'),str(root)],cwd=root/'pytools',env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
out=dict(root=str(root),pid=p.pid,unix=time.time(),new_sim_s_cap=14640,new_gpu_s=0,plan_sha256={canonical(value)!r})
with (root/'DISPATCH_ORACLES.json').open('x') as stream:json.dump(out,stream,indent=2)
print(json.dumps(out))
''', timeout=45)
    write(EVIDENCE/'CPUQ_WM_ORACLE_DISPATCH.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('probe', 'stage-start'))
    if parser.parse_args().command == 'probe':
        probe()
    else:
        stage_start()
