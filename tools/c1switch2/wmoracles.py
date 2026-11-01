"""Finish only twelve unstarted registered WM oracle branches; no picker/GPU restart."""
import argparse
from datetime import datetime
import json
import time

from tools.c1switch.design import checked, sha
from tools.c1switch.storage import canonical, write

from .design import CONTROLLERS, REGISTRATION
from .operations import EVIDENCE, env, ssh

OLD = '/root/c1switch2_run_292bfffe94b13ed8'
PLAN = EVIDENCE/'WM_ORACLE_CONTINUATION_PLAN.json'
NOT_BEFORE = datetime.fromisoformat('2026-11-01T11:30:00+08:00').timestamp()

SUPERVISOR = '''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);plan=json.loads((root/'PLAN.json').read_text())
assert time.time()>=plan['not_before_unix']
assert os.getpriority(os.PRIO_PROCESS,0)==19
assert not (root/'ORACLE_STARTED.json').exists()
with (root/'ORACLE_STARTED.json').open('x') as stream:json.dump(dict(pid=os.getpid(),unix=time.time()),stream)
for item in plan['branches']:
 while os.getloadavg()[0]>20:time.sleep(5)
 assert not (root/'STOP').exists()
 path=root/'specs'/(item['id']+'.json')
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
 rows=[json.loads(x) for x in (root/'accounting.jsonl').read_text().splitlines()] if (root/'accounting.jsonl').exists() else []
 assert sum(r['sim_s'] for r in rows)+item['sim_s']<=6000
 assert not (root/'jobs'/item['id']).exists(),'never replay a branch'
 with (root/'logs'/(item['id']+'.log')).open('x') as log:
  process=subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch2.runner','branch','--root',str(root),'--spec',str(path)],stdout=log,stderr=subprocess.STDOUT)
  code=process.wait()
 if code:
  with (root/'STOP').open('x') as stream:json.dump(dict(id=item['id'],code=code),stream)
  raise RuntimeError('owned oracle child failed; no retry')
 assert (root/'jobs'/item['id']/'RESULT.json').is_file()
 with (root/'PROGRESS.json').open('w') as stream:json.dump(dict(unix=time.time(),last=item['id']),stream)
with (root/'COMPLETE.json').open('x') as stream:json.dump(dict(unix=time.time(),branches=12,planned_sim_s=6000),stream)
'''


def validate_plan(value):
    if (value['predecessor'] != OLD or value['host'] != 'gpuR' or value['new_sim_s'] != 6000 or
            value['new_gpu_s'] != 0 or value['historical_sim_s'] != 2700 or
            value['not_before_unix'] != NOT_BEFORE or len(value['branches']) != 12):
        raise ValueError('exact original oracle-only continuation required')
    expected = {(seed, controller) for seed in (969139, 969143) for controller in CONTROLLERS}
    if {(b['seed'], b['controller']) for b in value['branches']} != expected:
        raise ValueError('complete original six-way sets required')
    if any(b['sim_s'] != (380 if b['seed'] == 969139 else 620) for b in value['branches']):
        raise ValueError('original registered prefix lengths required')
    if len({b['id'] for b in value['branches']}) != 12:
        raise ValueError('duplicate branch')


def probe():
    value = ssh('gpuR', f'''import hashlib,json
from pathlib import Path
root=Path({OLD!r});dispatch=json.loads((root/'DISPATCH_MODEL.json').read_text())
assert dispatch['pid']==763520 and dispatch['arm']=='wm'
proc=Path('/proc',str(dispatch['pid']))
assert not proc.exists() or proc.joinpath('stat').read_text().split()[2]=='Z','old supervisor live'
raw=(root/'accounting.jsonl').read_bytes();rows=[json.loads(x) for x in raw.splitlines()]
assert sum(r['sim_s'] for r in rows)==2700
assert not any('__j' in r['id'] for r in rows),'oracle branches already consumed'
assert json.loads((root/'STOP').read_text())['id']=='wm__s969147'
branches=[]
for path in sorted((root/'specs').glob('*.json')):
 spec=json.loads(path.read_text());seed=spec['world']['seed'];assert seed in (969139,969143)
 assert (root/'jobs'/f'wm__s{{seed}}__episode/PHYSICS.json').is_file()
 assert not (root/'jobs'/spec['id']).exists()
 branches.append(dict(id=spec['id'],seed=seed,controller=spec['alternative'],
  sim_s=300+20*len(spec['schedule'])+60,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
print(json.dumps(dict(predecessor=str(root),host='gpuR',historical_sim_s=2700,
 predecessor_ledger_sha256=hashlib.sha256(raw).hexdigest(),branches=branches,
 new_sim_s=sum(b['sim_s'] for b in branches),new_gpu_s=0,not_before_unix={NOT_BEFORE!r})))
''', timeout=30)
    validate_plan(value)
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != value:
            raise ValueError('preserve original oracle plan')
    else:
        write(PLAN, value)
    print(json.dumps(dict(plan=value, plan_sha256=canonical(value)), indent=2))


def stage():
    value = json.loads(checked(PLAN).read_text())
    validate_plan(value)
    text = checked(REGISTRATION).read_text(encoding='utf-8')
    if not any('S-C1SWITCH2 WM ORACLE CONTINUATION REGISTERED' in line and canonical(value) in line
               for line in text.splitlines()):
        raise ValueError('prospective exact oracle registration required')
    root = '/root/c1switch2_run_oracles_'+canonical(value)[:16]
    result = ssh('gpuR', f'''import hashlib,json
from pathlib import Path
old=Path({OLD!r});root=Path({root!r});assert root.parent.is_dir() and not root.exists()
assert hashlib.sha256((old/'accounting.jsonl').read_bytes()).hexdigest()=={value['predecessor_ledger_sha256']!r}
assert json.loads((old/'PREFLIGHT.json').read_text())['status']=='PASS'
sources=json.loads((old/'STAGED.json').read_text());root.mkdir()
for name,digest in sources.items():
 source=old/'pytools'/name;data=source.read_bytes();assert hashlib.sha256(data).hexdigest()==digest
 target=root/'pytools'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
for name in ('BINDING.json','GATES.json','STAGED.json','PREFLIGHT.json','REGISTRATION.md'):
 (root/name).write_bytes((old/name).read_bytes())
for name in ('jobs','specs','tasks','logs'):(root/name).mkdir()
for item in {value['branches']!r}:
 source=old/'specs'/(item['id']+'.json');data=source.read_bytes()
 assert hashlib.sha256(data).hexdigest()==item['sha256'];(root/'specs'/source.name).write_bytes(data)
(root/'PLAN.json').write_text(json.dumps({value!r},indent=2))
(root/'ORACLE_SUPERVISOR.py').write_text({SUPERVISOR!r})
print(json.dumps(dict(root=str(root),plan_sha256={canonical(value)!r},supervisor_sha256=hashlib.sha256((root/'ORACLE_SUPERVISOR.py').read_bytes()).hexdigest(),new_sim_s=0,new_gpu_s=0)))
''', timeout=45)
    write(EVIDENCE/'WM_ORACLE_CONTINUATION_STAGE.json', result)
    print(json.dumps(result))


def start():
    if time.time() < NOT_BEFORE:
        raise ValueError('priority window; not before11:30')
    stage = json.loads((EVIDENCE/'WM_ORACLE_CONTINUATION_STAGE.json').read_text())
    value = json.loads(PLAN.read_text());validate_plan(value)
    text = checked(REGISTRATION).read_text(encoding='utf-8')
    if not any('S-C1SWITCH2 WM ORACLE CONTINUATION REGISTERED' in line and canonical(value) in line
               for line in text.splitlines()):
        raise ValueError('current exact oracle authority required')
    result = ssh('gpuR', f'''import hashlib,json,os,subprocess,sys,sysconfig,time
from pathlib import Path
root=Path({stage['root']!r});old=Path({OLD!r})
assert os.getloadavg()[0]<20,'yield to priority CPU pressure'
assert time.time()>={NOT_BEFORE!r}
assert hashlib.sha256((old/'accounting.jsonl').read_bytes()).hexdigest()=={value['predecessor_ledger_sha256']!r}
assert hashlib.sha256((root/'ORACLE_SUPERVISOR.py').read_bytes()).hexdigest()=={stage['supervisor_sha256']!r}
assert json.loads((root/'CPU_BINDING_CHECK.json').read_text())['status']=='PASS'
assert not (root/'accounting.jsonl').exists() and not list((root/'jobs').iterdir())
claim=root.parent/'C1SWITCH2_WM_ORACLE_CONTINUE_ONCE.json'
with claim.open('x') as stream:json.dump(dict(root=str(root),plan_sha256={canonical(value)!r},unix=time.time()),stream)
environment=dict(os.environ,**{env('gpuR', stage['root'])!r},PYTHONPATH=str(root/'pytools'))
environment['PATH']=str(Path(sysconfig.get_paths()['purelib'])/'sumo/bin')+':'+environment['PATH']
with (root/'ORACLE.log').open('x') as log:
 process=subprocess.Popen([sys.executable,'-B','-u',str(root/'ORACLE_SUPERVISOR.py'),str(root)],cwd=root/'pytools',env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
out=dict(root=str(root),pid=process.pid,unix=time.time(),new_sim_s_cap=6000,new_gpu_s=0)
(root/'DISPATCH_ORACLES.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
''', timeout=30)
    write(EVIDENCE/'WM_ORACLE_CONTINUATION_DISPATCH.json', result)
    print(json.dumps(result))


def check():
    stage = json.loads((EVIDENCE/'WM_ORACLE_CONTINUATION_STAGE.json').read_text())
    result = ssh('gpuR', f'''import json,os,sys
from pathlib import Path
root=Path({stage['root']!r});os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(root/'pytools'));os.chdir(root/'pytools')
from tools.c1switch2.runner import admitted
binding=admitted(root)
assert binding['enabled_arms']==['wm'] and binding['cpu_host']=='gpuR'
compile((root/'ORACLE_SUPERVISOR.py').read_text(),'<oracle-only-supervisor>','exec')
assert not (root/'accounting.jsonl').exists() and not list((root/'jobs').iterdir())
out=dict(status='PASS',root=str(root),sources=len(binding['sources']),original_binding_and_gates=True,new_sim_s=0,new_gpu_s=0)
with (root/'CPU_BINDING_CHECK.json').open('x') as stream:json.dump(out,stream,indent=2)
print(json.dumps(out))
''', timeout=30)
    write(EVIDENCE/'WM_ORACLE_CONTINUATION_CPU.json', result)
    print(json.dumps(result))


def collect():
    stage = json.loads((EVIDENCE/'WM_ORACLE_CONTINUATION_STAGE.json').read_text())
    result = ssh('gpuR', f'''import json,time
from pathlib import Path
root=Path({stage['root']!r});dispatch=json.loads((root/'DISPATCH_ORACLES.json').read_text())
proc=Path('/proc',str(dispatch['pid']));rows=[json.loads(x) for x in (root/'accounting.jsonl').read_text().splitlines()] if (root/'accounting.jsonl').exists() else []
out=dict(unix=time.time(),root=str(root),pid=dispatch['pid'],pid_state=proc.joinpath('stat').read_text().split()[2] if proc.exists() else 'EXITED',charged_sim_s=sum(r['sim_s'] for r in rows),closed_actual_sim_s=0,branches=[],complete=(root/'COMPLETE.json').exists(),stop=json.loads((root/'STOP').read_text()) if (root/'STOP').exists() else None,new_gpu_s=0)
for job in sorted((root/'jobs').iterdir()):
 if (job/'CLOSE.json').exists():out['closed_actual_sim_s']+=json.loads((job/'CLOSE.json').read_text())['actual_sim_s']
 if (job/'RESULT.json').exists():out['branches'].append(dict(id=job.name,result=json.loads((job/'RESULT.json').read_text())))
if out['stop']:out['log']=(root/'ORACLE.log').read_text()
print(json.dumps(out))
''', timeout=30)
    write(EVIDENCE/('WM_ORACLE_PROGRESS_'+str(int(result['unix']))+'.json'), result)
    print(json.dumps({k: len(v) if k == 'branches' else v for k, v in result.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('probe', 'stage', 'start', 'check', 'collect'))
    command = parser.parse_args().command
    if command == 'probe':
        probe()
    elif command == 'stage':
        stage()
    elif command == 'check':
        check()
    elif command == 'collect':
        collect()
    else:
        start()
