"""Short, explicit operational sidecar stage/CPU/launch/status commands."""
import argparse
import hashlib
import json
from pathlib import Path
from tools.coopbandit.ops import remote
from tools.coopfree_batch import ROOT

LOCAL=Path(__file__).resolve().parents[1]
REG=Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')


def stage():
    files={name:(LOCAL/'tools'/source).read_text(encoding='utf-8') for name,source in
           [('batch_sidecar.py','coopfree_batch.py'),('test_batch_sidecar.py','test_coopfree_batch.py')]}
    return remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({ROOT!r});assert r.is_dir()
for name,text in {files!r}.items():
 p=r/name
 if p.exists(): assert p.read_text()==text,'immutable operational stage'
 else:
  with p.open('x') as f: f.write(text)
print(json.dumps(dict(sources={{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in {list(files)!r}}},sim_s=0)))
''')


def cpu():
    return remote('cpuQ',f'''import pathlib,sys,os,json,hashlib,importlib.util,unittest,time
from unittest.mock import patch
r=pathlib.Path({ROOT!r});assert r.is_dir();os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(r/'pytools'))
from tools.coopfree.runner import verify
from tools.coopfree.storage import write
verify(r)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
 sys.modules[name]=m;spec.loader.exec_module(m);return m
module('tools.coopfree_batch',r/'batch_sidecar.py')
tests=module('test_batch_sidecar',r/'test_batch_sidecar.py')
with patch('subprocess.Popen',side_effect=AssertionError('CPU only: no simulation/dispatch')):
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
receipt=dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,
 sources={{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ('batch_sidecar.py','test_batch_sidecar.py')}},
 binding_sha256=hashlib.sha256((r/'BINDING.json').read_bytes()).hexdigest(),unix=time.time())
assert receipt['pass_'] and not receipt['skips']
write(r/'BATCH_CPU.json',receipt)
print(json.dumps(receipt))
''')


def launch():
    digest=hashlib.sha256((LOCAL/'tools/coopfree_batch.py').read_bytes()).hexdigest()
    lines=[s for s in REG.read_text(encoding='utf-8').splitlines()
           if 'S-COOPFREE BATCH ACCELERATION REGISTERED' in s and digest in s]
    assert len(lines)==1,'register exact operational sidecar SHA before dispatch'
    return remote('cpuQ',f'''import pathlib,sys,os,json,hashlib,subprocess
r=pathlib.Path({ROOT!r});assert r.is_dir();sys.path.insert(0,str(r/'pytools'))
from tools.coopfree.runner import verify
from tools.coopfree.storage import read,write
verify(r)
cpu=read(r/'BATCH_CPU.json');assert cpu['pass_'] and not cpu['skips'] and cpu['sim_s']==0
for name,expected in cpu['sources'].items(): assert hashlib.sha256((r/name).read_bytes()).hexdigest()==expected
assert cpu['sources']['batch_sidecar.py']=={digest!r}
assert not (r/'BATCH_DISPATCH.json').exists(),'do not duplicate sidecar'
assert not (r/'STOP.json').exists() and not (r/'BATCH_PAUSE.json').exists()
assert pathlib.Path('/proc/339173/stat').read_text().split()[21]=='47835664'
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
write(r/'BATCH_REGISTRATION.json',dict(line={lines[0]!r},source_sha256={digest!r}))
log=(r/'BATCH_SIDECAR.log').open('x')
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u',str(r/'batch_sidecar.py'),str(r),'--workers','8'],
 cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=p.pid,start_ticks=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],
 host='cpuQ',root=str(r),workers=8,source_sha256={digest!r},new_simulation_arms=0,new_charge=0)
write(r/'BATCH_DISPATCH.json',receipt)
print(json.dumps(receipt))
''')


def status():
    return remote('cpuQ',f'''import pathlib,json,os
r=pathlib.Path({ROOT!r})
out={{}}
for name in ('BATCH_DISPATCH.json','BATCH_PAUSE.json','BATCH_RESUMED.json','STOP.json'):
 p=r/name
 if p.exists(): out[name]=json.loads(p.read_text())
for name in ('BATCH_ALLOCATION.json','BATCH_COMPLETE.json'):
 p=r/name
 if p.exists():
  value=json.loads(p.read_text())
  out[name]={{k:v for k,v in value.items() if k not in ('tasks','completed','excluded_existing')}}
  for k in ('tasks','completed','excluded_existing'):
   if k in value: out[name][k+'_count']=len(value[k])
out['supervisor_state']=pathlib.Path('/proc/339173/stat').read_text().split()[2]
out['load']=os.getloadavg()
print(json.dumps(out))
''')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('mode',choices=('stage','cpu','launch','status'))
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    result=globals()[args.mode]()
    print(result)
    if args.receipt:
        from tools.coopfree.storage import write
        write(args.receipt,dict(operation=args.mode,result=json.loads(result)))
