"""Pre-physics launch-adapter correction; existing broker/grader stay alive."""
import argparse
import json
from . import ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_capacitycpu3'


def qualify():
    prior=read(ops.EVIDENCE/'CPU_coopmwm20261102_capacitycpu2.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['root']!r});files={{}}
for name,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;files[name.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=40))
    for name in ('capacity_recovery','test_capacity_recovery_cpu','capacity_launch_ops'):
        files['tools/coopmwm/'+name+'.py']=(ops.LOCAL/'tools/coopmwm'/(name+'.py')).read_text(encoding='utf-8')
    staged=json.loads(ops.stage('cpuQ',SOURCE,files))
    out=ops.ssh.remote('cpuQ',f'''import sys,unittest,os,json,time
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import torch;torch.set_num_threads(1)
from unittest.mock import patch
start=time.monotonic();suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in ('test_capacity_broker_cpu','test_capacity_recovery_cpu','test_warm_health_cpu'))
with patch('subprocess.Popen',side_effect=AssertionError('no SUMO in qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,root={SOURCE!r},wall_s=time.monotonic()-start)))''',timeout=60)
    print(out);v=json.loads(out.strip().splitlines()[-1]);v['source_hashes']=staged['hashes']
    write(ops.EVIDENCE/'CPU_coopmwm20261102_capacitycpu3.json',v);assert v['pass_'] and not v['skips']
    # Real bindings, model/optimizer/RNG and admission preflight, no subprocesses.
    candidate=dict(read(ops.EVIDENCE/'MAIN0508_REPAIR_REGISTRATION.json'),cpu=v,source_root=SOURCE)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from unittest.mock import patch
from tools.coopmwm import capacity_recovery as c
from tools.coopmwm.accel_learner import load
r=pathlib.Path({ROOT!r});candidate=json.loads({json.dumps(candidate)!r})
with patch.object(c,'binding',return_value=candidate),patch('subprocess.Popen',side_effect=AssertionError('preflight no SUMO')):
 wr,b=c.install(r)
 for arm in ('B','R'):
  v=wr.verify(r,arm);policy,opt,rng,payload=load(r/f'{{arm}}_checkpoint_000.pt')
  assert payload['update']==0 and v['resume_update']==0 and v['workers']==(24 if arm=='B' else 4)
  assert not (r/f'WARM_{{arm}}_STARTED_V5.json').exists()
print(json.dumps(dict(pass_=True,real_binding_preflight=True,existing_u0_Adam_RNG=True,new_sumo_sim_s=0)))''',timeout=40)
    preflight=json.loads(out);write(ops.EVIDENCE/'CAPACITY_LAUNCH_PREFLIGHT.json',preflight);print(out)


def register():
    c=read(ops.EVIDENCE/'CPU_coopmwm20261102_capacitycpu3.json');assert c['pass_'] and not c['skips']
    assert read(ops.EVIDENCE/'CAPACITY_LAUNCH_PREFLIGHT.json')['pass_']
    v=dict(cpu=c,preflight=read(ops.EVIDENCE/'CAPACITY_LAUNCH_PREFLIGHT.json'),new_sumo_sim_s=0,
        diagnosis='CPU2 B/R entry failed before claim/checkpoint/physics on duplicate workers keyword in operational binding merge',
        fix='explicit dictionary update plus actual production-binding/u0 Adam-RNG preflight; no budget/science change; original broker3426521/grader3427015 retained')
    path=ops.EVIDENCE/'CAPACITY_LAUNCH_AMENDMENT.json';write(path,v);h=sha(path)
    line='\nS-COOPMWM MAIN0508 LAUNCH-ADAPTER REPAIR REGISTERED amendmentSHA'+h+': '+json.dumps({k:v for k,v in v.items() if k!='cpu'},sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({ROOT!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.warm_spool import immutable
for a in ('B','R'):
 d=json.loads((r/f'WARM_{{a}}_DISPATCH_V5.json').read_text());p=pathlib.Path('/proc',str(d['pid']),'stat')
 assert not p.exists() or (p.read_text().split()[21]==d['start_ticks'] and p.read_text().split()[2]=='Z')
 assert not (r/f'WARM_{{a}}_STARTED_V5.json').exists()
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
b=json.loads((r/'MAIN0508_REPAIR_BINDING.json').read_text());b.update(source_root={SOURCE!r},cpu=json.loads({json.dumps(c)!r}),amendment_sha256={h!r})
immutable(r/'MAIN0508_REPAIR_BINDING_V2.json',b)
print('registered; no physics')''',timeout=40)
    write(ops.EVIDENCE/'CAPACITY_LAUNCH_REGISTRATION.json',dict(v,amendment_sha256=h));print(json.dumps(dict(registered=h)))


def dispatch(arm):
    assert read(ops.EVIDENCE/'CAPACITY_LAUNCH_REGISTRATION.json')['amendment_sha256']
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,subprocess,os,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.capacity_recovery import binding,install
from tools.coopmwm.exact_runtime import assert_live
from tools.coopmwm.warm_spool import immutable
b=binding(r);wr,_=install(r);wr.verify(r,{arm!r});assert_live(r)
assert not (r/{('WARM_'+arm+'_DISPATCH_V6.json')!r}).exists()
d=json.loads((r/'CAPACITY_BROKER_DISPATCH_V2.json').read_text());assert pathlib.Path('/proc',str(d['pid']),'stat').read_text().split()[21]==d['start_ticks']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime');env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
with (r/{('CAPACITY_REPAIR_'+arm+'_V6.log')!r}).open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.capacity_recovery',str(r),'--component',{arm!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()
v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source=str(s),root=str(r),component={arm!r})
immutable(r/{('WARM_'+arm+'_DISPATCH_V6.json')!r},v);print(json.dumps(v))''',timeout=45))
    write(ops.EVIDENCE/f'WARM_{arm}_DISPATCH_V6.json',value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('qualify','register','dispatch'));p.add_argument('--arm',choices=('B','R'));a=p.parse_args()
    dispatch(a.arm) if a.mode=='dispatch' else globals()[a.mode]()
