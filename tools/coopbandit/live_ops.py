"""Stage and dispatch the registered fresh-dev panel only after final TRAIN fitting."""
import argparse
import json
from .ops import LOCAL,HOSTS,remote,stage
from .dispatch import SCREEN,EVIDENCE,read,once,status as root_status
from .protocol import sha

ROOT='/home/ubuntu/coopbandit20261031_live_v1'


def prepare():
    final=read(EVIDENCE/'FULL_EXPORT.json')
    if final['partial'] or final['states']!=300:
        raise ValueError('complete 300-state TRAIN panel required')
    train=final['snapshot']
    print(stage(ROOT,'cpuQ'))
    result=json.loads(remote('cpuQ',f'''import os,pathlib,json,shutil,sys,subprocess
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
r=pathlib.Path({ROOT!r});assert r.is_dir();sys.path.insert(0,str(r/'pytools'))
from tools.coopbandit.storage import read,write,admission
from tools.coopbandit.protocol import sha
from tools.coopbandit.backend import setup
from tools.coopbandit.live import allocation
s=pathlib.Path({train!r});assert (s/'DATA_COMPLETE.json').is_file()
summary=read(s/'learned/SUMMARY.json');assert summary['states']==300 and not summary['partial']
old=pathlib.Path({SCREEN!r});binding=read(old/'BINDING.json')
binding['sources']={{str(p.relative_to(r/'pytools')):sha(p) for p in (r/'pytools').rglob('*') if p.is_file() and p.suffix in ('.py','.json')}}
binding['resources']=dict(load_max=24,workers=8)
setup(binding)
policy=s/'learned/policy.pkl';assert sha(policy)==summary['policy_sha256']
live=dict(training_states=300,partial=False,training_snapshot=str(s),policy_sha256=sha(policy),sources=binding['sources'],planned_live_sim_s=99360)
for name,value in [('BINDING.json',binding),('LIVE_BINDING.json',live),('APPROVAL.json',read(old/'APPROVAL.json')),
                   ('ALLOCATION.json',dict(attempts=allocation()))]:
 p=r/name
 if p.exists(): assert read(p)==value
 else: write(p,value)
p=r/'policy.pkl'
if p.exists(): assert sha(p)==sha(policy)
else: shutil.copyfile(policy,p)
import sumo
binary=pathlib.Path(sumo.__file__).parent/'bin/sumo'
assert '1.27.1' in subprocess.check_output([str(binary),'--version'],text=True)
admission(r)
print(json.dumps(dict(host='cpuQ',root=str(r),training_states=300,policy_sha256=sha(policy),reserved_live_sim_s=99360,sim_s=0)))
''',timeout=30))
    once(EVIDENCE/'LIVE_PREPARED.json',result)
    print(json.dumps(result))


def launch():
    read(EVIDENCE/'LIVE_PREPARED.json')
    result=json.loads(remote('cpuQ',f'''import pathlib,os,sys,subprocess,json
r=pathlib.Path({ROOT!r});assert r.is_dir()
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONPATH=str(r/'pytools'),SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
log=(r/'SUPERVISOR_live.log').open('x')
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopbandit.live',str(r),'--workers','8'],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps(dict(host='cpuQ',root=str(r),pid=p.pid,phase='live',workers=8)))
'''))
    once(EVIDENCE/'DISPATCH_LIVE.json',result)
    print(json.dumps(result))


def status():
    current=root_status('cpuQ',ROOT)
    current.pop('result_ids')
    print(json.dumps(current))
    print(remote('cpuQ',f"import pathlib,json; r=pathlib.Path({ROOT!r}); print(json.dumps(dict(complete=(r/'LIVE_COMPLETE.json').exists(),report=(r/'report/SUMMARY.json').exists())))"))


def analyze():
    result=json.loads(remote('cpuQ',f'''import os,sys,json
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy
sys.path[:0]=[{ROOT!r}+'/pytools','/home/ubuntu/coopbandit20261031_cpu1/deps']
from tools.coopbandit.live_report import report
print(json.dumps(report({ROOT!r},'/home/ubuntu/coopbehav20261031_analysis')))
''',timeout=120))
    once(EVIDENCE/'LIVE_ANALYSIS.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','launch','status','analyze'));a=p.parse_args();globals()[a.mode]()
