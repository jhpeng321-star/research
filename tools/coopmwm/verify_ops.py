"""CPU-only qualification of the landed verifier; no deployment or physics dispatch."""
import argparse
import json
from . import ops
from .storage import read, write

SOURCE = '/home/ubuntu/coopmwm20261102_verifycpu2'
OVERLAYS = ('sumo_verify', 'test_sumo_verify_cpu', 'acceleration', 'accel_plant', 'plan_client')


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
    out = ops.ssh.remote('cpuQ', f'''import os,sys,json,time,unittest
from unittest.mock import patch
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',COOPMWM_COST_SOURCE={SOURCE!r}+'/FROZEN_COSTS.json',COOPMWM_MOTION_SOURCE={SOURCE!r}+'/FROZEN_MOTION_SOURCE.py')
import torch,pytest;torch.set_num_threads(1)
names=('test_acceleration_cpu','test_exact_grade_cpu','test_awr_cpu','test_incident_recovery_cpu','test_parallel_groups_cpu')
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in names)
class Counts:
 def __init__(self):self.tests=0;self.skips=0
 def pytest_runtest_logreport(self,report):
  if report.when=='call':self.tests+=1
  if report.skipped:self.skips+=1
c=Counts();start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('no physics/inference during CPU qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=1).run(suite)
 code=pytest.main([{SOURCE!r}+'/pytools/tools/coopmwm/test_sumo_verify_cpu.py','-q','-p','no:cacheprovider'],plugins=[c])
print(json.dumps(dict(pass_=code==0 and result.wasSuccessful(),tests=c.tests+result.testsRun,skips=c.skips+len(result.skipped),sim_s=0,root={SOURCE!r},wall_s=time.monotonic()-start)))''', timeout=90)
    print(out)
    value = json.loads(out.strip().splitlines()[-1]);value['source_hashes'] = staged['hashes']
    write(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/', 1)[1]+'.json'), value)
    assert value['pass_'] and value['tests'] >= 15 and value['skips'] == value['sim_s'] == 0


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('mode', choices=('cpu',))
    globals()[p.parse_args().mode]()
