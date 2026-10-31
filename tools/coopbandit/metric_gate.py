"""Explicit one-shot full1140s gate of scalar-metric compatibility on gpuA."""
import json
from .dispatch import EVIDENCE,SCREEN,fetch,send,once,HOSTS,remote
from .protocol import canonical,sha,worlds

ROOT='/root/autodl-tmp/coopbandit20261031_metricgate_v4'


def stage():
    source=(EVIDENCE.parent/'metric_compat.py').read_text(encoding='utf-8')
    source_sha=sha(EVIDENCE.parent/'metric_compat.py')
    old=HOSTS['gpuA']['root']
    remote('gpuA',f'''import pathlib,shutil,json
r=pathlib.Path({ROOT!r});assert r.parent.is_dir();r.mkdir(exist_ok=True)
old=pathlib.Path({old!r})
for n in ('COOPBANDIT_DESIGN.md','BINDING.json','APPROVAL.json'):
 p=r/n
 if p.exists(): assert p.read_bytes()==(old/n).read_bytes()
 else: shutil.copyfile(old/n,p)
if not (r/'pytools').exists(): (r/'pytools').symlink_to(old/'pytools',target_is_directory=True)
p=r/'metric_compat.py'
if p.exists(): assert p.read_text()=={source!r}
else: p.write_text({source!r})
print('metric compatibility staged without editing frozen runtime or prior roots')
''')
    expected=fetch('cpuQ',SCREEN,['jobs/s981006_screen/RESULT.json'])['jobs/s981006_screen/RESULT.json']
    send('gpuA',ROOT,{'REFERENCE.json':expected,'ALLOCATION.json':canonical(dict(attempts={'s981006_screen':dict(sim_s=1140,phase='data')}))+'\n',
                     'COMPAT_BINDING.json':canonical(dict(source_sha256=source_sha,scope='metric and observable summary only; original policy/safety unchanged'))+'\n'})
    print(json.dumps(dict(root=ROOT,compat_sha256=source_sha,extra_reserved_sim_s=1140)))


def launch():
    config=HOSTS['gpuA'];world=next(w for w in worlds() if w['seed']==981006)
    code=f'''import pathlib,os,json,time,traceback,hashlib
from tools.coopbandit import runner
from tools.coopbandit.storage import read,write,stop
from tools.coopbandit.protocol import canonical,sha
import metric_compat
r=pathlib.Path({ROOT!r});(r/'jobs').mkdir(exist_ok=True)
assert sha(r/'metric_compat.py')==read(r/'COMPAT_BINDING.json')['source_sha256']
write(r/'OWNED_METRIC_GATE.json',dict(pid=os.getpid(),unix=time.time()))
try:
 runner.init(str(r));metric_compat.install()
 got=runner.screen({world!r})
 expected=read(r/'REFERENCE.json')
 differences={{t:[k for k,v in rec.items() if canonical(v)!=canonical(expected['checkpoints'][t][k])] for t,rec in got['checkpoints'].items()}}
 differences={{t:keys for t,keys in differences.items() if keys}}
 write(r/'COMPARISON.json',dict(bitwise=not differences,differences=differences,actual_sim_s=got['actual_sim_s']))
 if differences: raise ValueError('full1140 compatibility gate failed '+str(differences))
 write(r/'METRIC_GATE_COMPLETE.json',dict(bitwise=True,actual_sim_s=1140,compat_sha256=sha(r/'metric_compat.py')))
except BaseException:
 stop(r,traceback.format_exc());raise
'''
    result=json.loads(remote('gpuA',f'''import pathlib,subprocess,sys,os,json
r=pathlib.Path({ROOT!r});sys.path[:0]={config['deps']!r}
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONPATH=':'.join([str(r),str(r/'pytools')]+{config['deps']!r}),SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
log=(r/'SUPERVISOR_metric_gate.log').open('x')
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-c',{code!r}],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps(dict(host='gpuA',root=str(r),pid=p.pid,phase='full1140-metric-gate')))
'''))
    once(EVIDENCE/'DISPATCH_METRIC_GATE_V4.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage','launch'));a=p.parse_args()
    globals()[a.mode]()
