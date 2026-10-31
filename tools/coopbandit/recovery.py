"""Explicit diagnosed gpuA recovery after full1140 bitwise scalar-metric gate."""
import argparse
import hashlib
import json
from .dispatch import EVIDENCE,once,fetch,send,remote,read,HOSTS
from .protocol import canonical,digest,sha
from .metric_gate import ROOT as GATE_ROOT

ROOT='/root/autodl-tmp/coopbandit20261031_labels_v4'
OLD='/root/autodl-tmp/coopbandit20261031_v3'


def wrapper():
    return '''import pathlib,os,json,hashlib,traceback
from tools.coopbandit import runner
from tools.coopbandit.storage import read,write,stop
from tools.coopbandit.protocol import sha
import metric_compat
r=pathlib.Path(__file__).parent
base_init=runner.init
def compatible_init(root):
 base_init(root)
 binding=read(pathlib.Path(root)/'COMPAT_BINDING.json')
 assert sha(pathlib.Path(root)/'metric_compat.py')==binding['source_sha256']
 assert sha(pathlib.Path(root)/'RUN_COMPAT.py')==binding['wrapper_sha256']
 assert read(pathlib.Path(root)/'METRIC_GATE_COMPLETE.json')['bitwise']
 metric_compat.install()
runner.init=compatible_init
try:
 (r/'jobs').mkdir(exist_ok=True)
 compatible_init(str(r))
 jobs=read(r/'BRANCH_JOBS.json')
 for t in (300,540):
  job=next(j for j in jobs if j[1]==t and j[2]=='O1' and j[3] is None)
  if runner.completed(r,runner.job_id(job)) is None: runner.branch(job)
 write(r/'RECOVERY_CANARIES_COMPLETE.json',dict(bitwise=True,clocks=[300,540]))
 runner.run(r,phase='branches',max_workers=8)
except BaseException:
 stop(r,traceback.format_exc());raise
'''


def prepare():
    gate=json.loads(fetch('gpuA',GATE_ROOT,['METRIC_GATE_COMPLETE.json'])['METRIC_GATE_COMPLETE.json'])
    if not gate['bitwise'] or gate['actual_sim_s']!=1140:
        raise ValueError('full compatibility gate not passed')
    once(EVIDENCE/'METRIC_GATE_V4.json',gate)
    original=read(EVIDENCE/'SHARDS.json')
    plan=json.loads(canonical(original))
    plan['shards']['gpuA']['root']=ROOT
    plan.update(abandoned_charged=4320,abandoned_actual=1806,diagnostic_charged=1140,diagnostic_actual=1140,
                conservative_total=1594020,compat_sha256=gate['compat_sha256'],recovery_source_sha256=sha(__file__))
    once(EVIDENCE/'RECOVERY_PLAN.json',plan)
    shard=plan['shards']['gpuA']
    code=wrapper();code_sha=hashlib.sha256(code.encode()).hexdigest()
    names=[f'jobs/s{w["seed"]}_screen/{n}' for w in shard['worlds'] for n in ('RESULT.json','CLOSE.json')]
    remote('gpuA',f'''import pathlib,shutil,json
r=pathlib.Path({ROOT!r});assert r.parent.is_dir();r.mkdir(exist_ok=True)
old=pathlib.Path({OLD!r});gate=pathlib.Path({GATE_ROOT!r})
assert (old/'STOP').exists();assert json.loads((gate/'METRIC_GATE_COMPLETE.json').read_text())['bitwise']
for name in ['COOPBANDIT_DESIGN.md','BINDING.json','APPROVAL.json']+{names!r}:
 p=r/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): assert p.read_bytes()==(old/name).read_bytes()
 else: shutil.copyfile(old/name,p)
if not (r/'pytools').exists(): (r/'pytools').symlink_to(old/'pytools',target_is_directory=True)
for name in ('metric_compat.py','METRIC_GATE_COMPLETE.json'):
 p=r/name
 if p.exists(): assert p.read_bytes()==(gate/name).read_bytes()
 else: shutil.copyfile(gate/name,p)
p=r/'RUN_COMPAT.py'
if p.exists(): assert p.read_text()=={code!r}
else: p.write_text({code!r})
print('immutable recovery sources staged; old STOP and tapes preserved')
''')
    files={'ALLOCATION.json':canonical(dict(host='gpuA',attempts=shard['attempts'],global_plan_sha256=digest(plan)))+'\n',
           'GLOBAL_ALLOCATION.json':canonical(plan)+'\n','BRANCH_JOBS.json':canonical(shard['jobs'])+'\n',
           'COMPAT_BINDING.json':canonical(dict(source_sha256=gate['compat_sha256'],wrapper_sha256=code_sha))+'\n'}
    send('gpuA',ROOT,files)
    print(json.dumps(dict(root=ROOT,plan_sha256=digest(plan),wrapper_sha256=code_sha,conservative_total=plan['conservative_total'])))


def launch():
    plan=read(EVIDENCE/'RECOVERY_PLAN.json');config=HOSTS['gpuA']
    result=json.loads(remote('gpuA',f'''import pathlib,sys,os,subprocess,json
r=pathlib.Path({ROOT!r});assert r.is_dir();sys.path[:0]={config['deps']!r}
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONPATH=':'.join([str(r),str(r/'pytools')]+{config['deps']!r}),SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
log=(r/'SUPERVISOR_branches.log').open('x')
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u',str(r/'RUN_COMPAT.py')],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps(dict(host='gpuA',root=str(r),pid=p.pid,phase='branches-compatible',workers=8)))
'''))
    once(EVIDENCE/'DISPATCH_BRANCHES_gpuA.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','launch'));a=p.parse_args();globals()[a.mode]()
