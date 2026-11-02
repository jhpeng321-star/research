"""MAIN18:44 minimal durable u4 recovery; adopt receipts before every dispatch."""
import argparse
import json
import time
from . import ops
from .storage import read,write

ROOT='/home/ubuntu/coopmwm20261102_warm1'
SOURCE='/home/ubuntu/coopmwm20261102_fast1844cpu1'
OVERLAYS=('quota_admission','parallel_groups','capacity_broker','warm_runner',
          'incident_recovery','resource_config','test_fast_resume_cpu','fast_resume_ops')


def admit(host):
    """Use already-qualified secondary spools; failed monitors did no admission."""
    from .protocol import checked,REGISTRY,sha
    original=read(ops.EVIDENCE/'SECOND_CPU_QUALIFICATION.json')
    assert original['pass_'] and original['tests']==26
    expected_hash=original['staged']['hashes']['pytools/tools/coopmwm/hotadd.py']
    target=ops.EVIDENCE/f'FAST1844_ADMISSION_REGISTRATION_{host}.json'
    if not target.exists():
        line=f'\nS-COOPMWM MAIN1844 {host} SECOND-SLOT ADMISSION REPAIR: existing V2 monitor exited BEFORE admission with missing hotadd import. Adopt existing owned READY spool/eight exact gates; invoke original 26-CPU-qualified gate helper by verified file SHA, same restricted endpoint/expected boot-start-PID and search SHA. No worker restart/inference replay/new physics.\n'
        write(target,dict(line=line,helper_sha256=expected_hash,new_sim_s=0))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(target)+'\n')
    reg=read(target)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,time,sys,hashlib,importlib.util
r=pathlib.Path({ROOT!r});sys.path.insert(0,'/home/ubuntu/coopmwm20261102_verifybeta13/pytools')
from tools.coopmwm.warm_broker import Remote
from tools.coopmwm.warm_spool import immutable
p=pathlib.Path('/home/ubuntu/coopmwm20261102_hotadd1/pytools/tools/coopmwm/hotadd.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=={expected_hash!r}
spec=importlib.util.spec_from_file_location('qualified_hotadd',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
d=json.loads((r/('RESOURCE_MONITOR_DISPATCH_'+{host!r}+'_1651_V2.json')).read_text());c=d['config'];tag=c['tag']
proc=pathlib.Path('/proc',str(d['pid']))
assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert not proc.exists() or (proc/'stat').read_text().split()[21]==d['start_ticks'] and (proc/'stat').read_text().split()[2]=='Z'
assert 'ModuleNotFoundError' in (r/('RESOURCE_MONITOR_'+{host!r}+'_1651_V2.log')).read_text()
done=r/('RESOURCE_ADMITTED_'+tag+'.json')
if done.exists():
 print(json.dumps(dict(adopted=True,receipt=json.loads(done.read_text()))))
else:
 s=Remote(r,c['endpoint']).call(dict(action='status'));owner=s['receipts']['SPOOL_OWNER.json']
 for k in ('pid','boot_id','start_ticks'):assert owner[k]==c['expected'][k]
 assert m.qualified(s);service=s['receipts']['SERVICE_OWNER.json'];assert service['pid']==owner['pid']
 assert json.loads((r/'WARM_REMOTE_BINDING.json').read_text())['search_sha256']==c['search_sha256']
 assert c['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
 with (r/'AUTHORITY.md').open('a') as f:f.write({(reg['line']+'amendmentSHA'+sha(target)+chr(10))!r})
 immutable(r/('WARM_REMOTE_ENDPOINT_ADMISSION_'+tag+'_SECOND.json'),dict(amendment_sha256=c['amendment_sha256'],search_sha256=c['search_sha256'],endpoints=[c['endpoint']],owner=owner,gates={{n:s['receipts'][n] for n in m.GATES}},unix=time.time()))
 immutable(r/('GPU_RESOURCE_READY_'+c['endpoint']['slot']+'_1651.json'),dict(slot=c['endpoint']['slot'],owner=owner,service=service,amendment_sha256=c['amendment_sha256']))
 v=dict(unix=time.time(),owner=owner,slot=c['endpoint']['slot'],new_sim_s=0)
 immutable(done,v);print(json.dumps(v))''',timeout=40)
    v=json.loads(out);dest=ops.EVIDENCE/f'FAST1844_ADMITTED_{host}.json'
    if not dest.exists():write(dest,v)
    print(out)


def gpu_status(host):
    roots=dict(gpuA=['/root/coopmwm20261102_incident1','/root/coopmwm20261102_deadlineA2'],
               gpuR=['/root/coopmwm20261102_concurrent3','/root/coopmwm20261102_deadlineR2'])
    out=ops.ssh.remote(host,f'''import pathlib,json,time,subprocess
out=[]
for name in {roots[host]!r}:
 r=pathlib.Path(name);v=dict(root=name)
 for n in ('SPOOL_OWNER.json','SERVICE_OWNER.json','SPOOL_FAILED.json'):
  if (r/n).exists():v[n]=json.loads((r/n).read_text())
 d=v.get('SPOOL_OWNER.json');p=pathlib.Path('/proc',str(d['pid'])) if d else None
 if p and p.exists():
  st=(p/'stat').read_text().split();assert st[21]==d['start_ticks'];v['state']=st[2]
 else:v['state']='exited'
 v['gates']={{p.name:json.loads(p.read_text()).get('pass_') for p in r.glob('STARTUP_*.json')}}
 v['inbox']=len(list((r/'inbox').glob('*.json')));v['forecasts']=len(list((r/'forecasts').glob('*.json')))
 out.append(v)
print(json.dumps(dict(host={host!r},unix=time.time(),workers=out,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True))))''',timeout=40)
    v=json.loads(out);write(ops.EVIDENCE/f'FAST1844_GPU_{host}_{int(v["unix"])}.json',v);print(out)


def cpu():
    prior=read(ops.EVIDENCE/'CPU_coopmwm20261102_verifybeta13.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['root']!r});out={{}}
for n,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 out[n.removeprefix('pytools/')]=p.read_text()
print(json.dumps(out))''',timeout=40))
    for name in OVERLAYS:
        files['tools/coopmwm/'+name+'.py']=ops.checked(ops.LOCAL/'tools/coopmwm'/(name+'.py')).read_text(encoding='utf-8')
    staged=json.loads(ops.stage('cpuQ',SOURCE,files))
    out=ops.ssh.remote('cpuQ',f'''import os,sys,json,time,unittest,pathlib
from unittest.mock import patch
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',COOPMWM_COST_SOURCE={SOURCE!r}+'/FROZEN_COSTS.json',COOPMWM_MOTION_SOURCE={SOURCE!r}+'/FROZEN_MOTION_SOURCE.py')
import torch,pytest;torch.set_num_threads(1)
r=pathlib.Path({SOURCE!r});attempt=1+len(list(r.glob('CPU_ATTEMPT_*_OWNER.json')))
with (r/f'CPU_ATTEMPT_{{attempt}}_OWNER.json').open('x') as f:json.dump(dict(pid=os.getpid(),start_ticks=pathlib.Path('/proc/self/stat').read_text().split()[21],unix=time.time()),f)
names=('test_fast_resume_cpu','test_acceleration_cpu','test_exact_grade_cpu','test_awr_cpu','test_incident_recovery_cpu','test_parallel_groups_cpu','test_warm_health_cpu','test_resource_config_cpu','test_warm_transport_cpu','test_warm_broker_cpu','test_capacity_broker_cpu')
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in names)
class Counts:
 def __init__(self):self.tests=0;self.skips=0
 def pytest_runtest_logreport(self,report):
  if report.when=='call':self.tests+=1
  if report.skipped:self.skips+=1
c=Counts();start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('CPU qualification forbids physics/inference')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=1).run(suite)
 code=pytest.main([str(r/'pytools/tools/coopmwm/test_sumo_verify_cpu.py'),str(r/'pytools/tools/coopmwm/test_verify_continuation_cpu.py'),'-q','-p','no:cacheprovider'],plugins=[c])
v=dict(pass_=code==0 and result.wasSuccessful(),tests=c.tests+result.testsRun,skips=c.skips+len(result.skipped),sim_s=0,root=str(r),wall_s=time.monotonic()-start)
with (r/f'CPU_ATTEMPT_{{attempt}}_RESULT.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=90000//1000)
    print(out);v=json.loads(out.strip().splitlines()[-1]);v['source_hashes']=staged['hashes']
    write(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'),v)
    assert v['pass_'] and v['skips']==v['sim_s']==0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('gpu_status','cpu','admit'))
    p.add_argument('--host',choices=('gpuA','gpuR'));a=p.parse_args()
    if a.action in ('gpu_status','admit'):globals()[a.action](a.host)
    else:globals()[a.action]()
