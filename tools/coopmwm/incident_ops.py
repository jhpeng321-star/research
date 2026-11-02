"""Explicit MAIN09:44 incident qualification and read-only recovery census."""
import argparse
import json
from . import ops
from .storage import read,write
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_incidentcpu1'
OVERLAYS=('mppi_runtime','query_errors','plan_service','capacity_broker','incident_ops','test_incident_cpu')


def cpu():
    prior=read(ops.EVIDENCE/'CPU_coopmwm20261102_tailcpu3.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['root']!r});files={{}}
for name,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 files[name.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=40))
    for name in OVERLAYS:
        files['tools/coopmwm/'+name+'.py']=ops.checked(ops.LOCAL/'tools/coopmwm'/(name+'.py')).read_text(encoding='utf-8')
    staged=json.loads(ops.stage('cpuQ',SOURCE,files))
    out=ops.ssh.remote('cpuQ',f'''import sys,os,json,unittest,time
from unittest.mock import patch
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',COOPMWM_COST_SOURCE={SOURCE!r}+'/FROZEN_COSTS.json',COOPMWM_MOTION_SOURCE={SOURCE!r}+'/FROZEN_MOTION_SOURCE.py')
import torch;torch.set_num_threads(1)
names=('test_incident_cpu','test_mppi_runtime_cpu','test_mppi_search_cpu','test_capacity_broker_cpu','test_parallel_groups_cpu','test_tail_runtime_cpu','test_warm_health_cpu','test_warm_spool_cpu','test_exact_grade_cpu','test_awr_cpu')
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in names);start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('no SUMO/inference during CPU qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,root={SOURCE!r},wall_s=time.monotonic()-start)))''',timeout=90)
    print(out);value=json.loads(out.strip().splitlines()[-1]);value['source_hashes']=staged['hashes']
    write(ops.EVIDENCE/'CPU_coopmwm20261102_incidentcpu1.json',value)
    assert value['pass_'] and value['sim_s']==value['skips']==0


def census():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time,os
r=pathlib.Path({ROOT!r});result={{}}
for arm,u in [('B',3),('R',2)]:
 jobs=[]
 for p in sorted((r/'jobs').glob(f'warm_u{{u:03}}_{{arm}}_*')):
  jobs.append(dict(id=p.name,closed=(p/'RESULT.json').exists(),claim=json.loads((p/'CLAIM.json').read_text()) if (p/'CLAIM.json').exists() else None,files=[x.name for x in p.iterdir() if x.is_file() and x.suffix in ('.json','.jsonl','.pt')]))
 result[arm]=jobs
processes=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(bytes([0]))
  if any(b'tools.coopmwm.' in a for a in args):processes.append(dict(pid=int(p.name),args=[a.decode(errors='replace') for a in args if a]))
 except OSError:pass
print(json.dumps(dict(unix=time.time(),jobs=result,processes=processes,new_sumo_sim_s=0)))''',timeout=40))
    write(ops.EVIDENCE/f'INCIDENT0944_CENSUS_{int(value["unix"])}.json',value)
    print(json.dumps(dict(unix=value['unix'],jobs={a:dict(total=len(j),closed=sum(x['closed'] for x in j),interrupted=sum(x['claim'] is not None and not x['closed'] for x in j),unclaimed=sum(x['claim'] is None for x in j)) for a,j in value['jobs'].items()},processes=value['processes'])))
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('cpu','census'));globals()[p.parse_args().mode]()
