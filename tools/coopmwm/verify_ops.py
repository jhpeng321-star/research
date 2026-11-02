"""CPU-only qualification of the landed verifier; no deployment or physics dispatch."""
import argparse
import json
from . import ops
from .storage import read, write

SOURCE = '/home/ubuntu/coopmwm20261102_verifycpu9'
OVERLAYS = ('sumo_verify', 'test_sumo_verify_cpu', 'test_verify_continuation_cpu',
            'exact_grade', 'exact_runtime', 'verification_grades', 'acceleration', 'accel_plant', 'plan_client')


def cpu():
    prior = read(ops.EVIDENCE/'CPU_coopmwm20261102_recoverycpu1.json')
    files = json.loads(ops.ssh.remote('cpuQ', f'''import pathlib,json,hashlib
r=pathlib.Path({prior['root']!r});files={{}}
for n,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 files[n.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''', timeout=40))
    for name in OVERLAYS:
        files['tools/coopmwm/'+name+'.py'] = ops.checked(
            ops.LOCAL/'tools/coopmwm'/(name+'.py')).read_text(encoding='utf-8')
    staged = json.loads(ops.stage('cpuQ', SOURCE, files))
    out = ops.ssh.remote('cpuQ', f'''import os,sys,json,time,unittest,pathlib
from unittest.mock import patch
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',COOPMWM_COST_SOURCE={SOURCE!r}+'/FROZEN_COSTS.json',COOPMWM_MOTION_SOURCE={SOURCE!r}+'/FROZEN_MOTION_SOURCE.py')
r=pathlib.Path({SOURCE!r});attempt=1+len(list(r.glob('CPU_ATTEMPT_*_OWNER.json')))
owner=dict(pid=os.getpid(),start_ticks=pathlib.Path('/proc/self/stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
with (r/f'CPU_ATTEMPT_{{attempt}}_OWNER.json').open('x') as f:json.dump(owner,f)
import torch,pytest;torch.set_num_threads(1)
names=('test_acceleration_cpu','test_exact_grade_cpu','test_awr_cpu','test_incident_recovery_cpu','test_parallel_groups_cpu','test_warm_health_cpu')
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in names)
class Counts:
 def __init__(self):self.tests=0;self.skips=0
 def pytest_runtest_logreport(self,report):
  if report.when=='call':self.tests+=1
  if report.skipped:self.skips+=1
c=Counts();start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('no physics/inference during CPU qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=1).run(suite)
 code=pytest.main([{SOURCE!r}+'/pytools/tools/coopmwm/test_sumo_verify_cpu.py',{SOURCE!r}+'/pytools/tools/coopmwm/test_verify_continuation_cpu.py','-q','-p','no:cacheprovider'],plugins=[c])
value=dict(pass_=code==0 and result.wasSuccessful(),tests=c.tests+result.testsRun,skips=c.skips+len(result.skipped),sim_s=0,root={SOURCE!r},wall_s=time.monotonic()-start)
with (r/f'CPU_ATTEMPT_{{attempt}}_RESULT.json').open('x') as f:json.dump(value,f)
print(json.dumps(value))''', timeout=90)
    print(out)
    value = json.loads(out.strip().splitlines()[-1]);value['source_hashes'] = staged['hashes']
    write(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/', 1)[1]+'.json'), value)
    assert value['pass_'] and value['tests'] >= 15 and value['skips'] == value['sim_s'] == 0


def sessions():
    """Read-only census of transient stdin-Python sessions after transport loss."""
    print(ops.ssh.remote('cpuQ', '''import pathlib,json,os,time
found=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if "compile(sys.stdin.read(),'<coopbandit-remote>'" in cmd:
   st=(p/'stat').read_text().split()
   found.append(dict(pid=int(p.name),state=st[2],start_ticks=st[21],cmd=cmd))
 except (OSError,PermissionError):pass
print(json.dumps(dict(unix=time.time(),sessions=found)))''', timeout=30))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('mode', choices=('cpu','sessions'))
    globals()[p.parse_args().mode]()
