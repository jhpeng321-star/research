"""Metric-compatible gpuR shard; preserved STOP, cgroup-aware resource evidence."""
import argparse
import hashlib
import json
from .dispatch import EVIDENCE,active_plan,once,fetch,send,remote,read,HOSTS
from .protocol import canonical,digest,sha
from .recovery import wrapper

ROOT='/root/autodl-tmp/coopbandit20261031_labels_r_v5'
GATE='/root/autodl-tmp/coopbandit20261031_metricgate_r_v5'
OLD='/root/autodl-tmp/coopbandit20261031_v3'


def prepare():
    gate=json.loads(fetch('gpuR',GATE,['METRIC_GATE_COMPLETE.json'])['METRIC_GATE_COMPLETE.json'])
    assert gate['bitwise'] and gate['actual_sim_s']==1140
    once(EVIDENCE/'METRIC_GATE_R_V5.json',gate)
    plan=read(EVIDENCE/'RECOVERY_PLAN.json')
    plan['shards']['gpuR']['root']=ROOT
    plan.update(abandoned_charged=6360,abandoned_actual=3846,diagnostic_charged=2280,diagnostic_actual=2280,
                conservative_total=1597200,gpur_recovery_source_sha256=sha(__file__))
    once(EVIDENCE/'RECOVERY_PLAN_V5.json',plan)
    shard=plan['shards']['gpuR'];code=wrapper();code_sha=hashlib.sha256(code.encode()).hexdigest()
    names=[f'jobs/s{w["seed"]}_screen/{n}' for w in shard['worlds'] for n in ('RESULT.json','CLOSE.json')]
    remote('gpuR',f'''import pathlib,shutil,json
r=pathlib.Path({ROOT!r});assert r.parent.is_dir();r.mkdir(exist_ok=True)
old=pathlib.Path({OLD!r});g=pathlib.Path({GATE!r});assert (old/'STOP').exists()
assert not pathlib.Path('/proc/725400').exists(),'old owned supervisor still running'
for name in ['COOPBANDIT_DESIGN.md','APPROVAL.json']+{names!r}:
 p=r/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): assert p.read_bytes()==(old/name).read_bytes()
 else: shutil.copyfile(old/name,p)
binding=json.loads((old/'BINDING.json').read_text());binding['resources']=dict(load_max=112,workers=8,quota_cpus=25,measured_existing_cpu=1.14)
p=r/'BINDING.json'
if p.exists(): assert json.loads(p.read_text())==binding
else: p.write_text(json.dumps(binding))
if not (r/'pytools').exists(): (r/'pytools').symlink_to(old/'pytools',target_is_directory=True)
for name in ('metric_compat.py','METRIC_GATE_COMPLETE.json'):
 p=r/name
 if p.exists(): assert p.read_bytes()==(g/name).read_bytes()
 else: shutil.copyfile(g/name,p)
p=r/'RUN_COMPAT.py'
if p.exists(): assert p.read_text()=={code!r}
else: p.write_text({code!r})
print('gpuR immutable recovery ready; prior STOP and all attempts preserved')
''')
    send('gpuR',ROOT,{'ALLOCATION.json':canonical(dict(host='gpuR',attempts=shard['attempts'],global_plan_sha256=digest(plan)))+'\n',
                      'GLOBAL_ALLOCATION.json':canonical(plan)+'\n','BRANCH_JOBS.json':canonical(shard['jobs'])+'\n',
                      'COMPAT_BINDING.json':canonical(dict(source_sha256=gate['compat_sha256'],wrapper_sha256=code_sha))+'\n'})
    print(json.dumps(dict(root=ROOT,plan_sha256=digest(plan),wrapper_sha256=code_sha,conservative_total=plan['conservative_total'])))


def launch():
    read(EVIDENCE/'RECOVERY_PLAN_V5.json');config=HOSTS['gpuR']
    result=json.loads(remote('gpuR',f'''import pathlib,sys,os,subprocess,json
r=pathlib.Path({ROOT!r});assert r.is_dir();sys.path[:0]={config['deps']!r}
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONPATH=':'.join([str(r),str(r/'pytools')]+{config['deps']!r}),SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
log=(r/'SUPERVISOR_branches.log').open('x')
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u',str(r/'RUN_COMPAT.py')],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps(dict(host='gpuR',root=str(r),pid=p.pid,phase='branches-compatible',workers=8)))
'''))
    once(EVIDENCE/'DISPATCH_BRANCHES_gpuR_V5.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','launch'));a=p.parse_args();globals()[a.mode]()
