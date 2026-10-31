"""Short explicit SSH operations for the owner-requested HOLD-credit continuation."""
import argparse
import json
from pathlib import Path
from tools.coopbandit import ops as ssh

LOCAL = Path(__file__).resolve().parent
ROOT = '/home/ubuntu/coopfree20261101_build5'


def gate():
    sources = {name: (LOCAL/name).read_text(encoding='utf-8') for name in
               ('coopfree_hold_gate.py', 'coopfree_batch.py')}
    return ssh.remote('cpuQ', f'''import pathlib,json,os,subprocess,sys
r=pathlib.Path({ROOT!r});assert r.is_dir()
assert not (r/'HOLD_GATE_DISPATCH.json').exists(),'already dispatched'
for name,text in json.loads({json.dumps(sources)!r}).items():
 p=r/'pytools/tools'/name;assert p.parent.is_dir()
 if p.exists(): assert p.read_text()==text
 else: p.write_text(text)
sys.path.insert(0,str(r/'pytools'))
from tools.coopfree.storage import write
from tools.coopfree_batch import identity
identity(pathlib.Path('/proc/339173/stat').read_text())
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
log=(r/'HOLD_GATE.log').open('x')
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopfree_hold_gate'],env=env,cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
receipt=dict(pid=p.pid,ticks=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],root=str(r),new_physics=0)
write(r/'HOLD_GATE_DISPATCH.json',receipt)
print(json.dumps(receipt))
''', timeout=20)


def status():
    return ssh.remote('cpuQ', f'''import pathlib,json,os
r=pathlib.Path({ROOT!r});read=lambda p:json.loads(p.read_text())
names=['HOLD_GATE_DISPATCH.json','HOLD_GATE_PARKED.json','BATCH_RESUMED.json','STOP.json','UPDATE_001.json']
out={{n:read(r/n) for n in names if (r/n).exists() and n!='UPDATE_001.json'}}
out['legacy_update_exists']=(r/'UPDATE_001.json').exists()
out['supervisor_stat']=pathlib.Path('/proc/339173/stat').read_text().split()[2:3]
out['load']=os.getloadavg()
out['unfinished']=[p.name for p in (r/'jobs').iterdir() if p.is_dir() and (p/'START.json').exists() and not (p/'CLOSE.json').exists()]
print(json.dumps(out))
''', timeout=20)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('gate','status'));a=p.parse_args()
    print(gate() if a.mode=='gate' else status())
