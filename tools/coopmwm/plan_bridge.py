"""Hot-addable worker broker. Only new worker slots change; search never changes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from . import ops
from .storage import read,write
from .protocol import digest

ROOT='/home/ubuntu/coopmwm20261101_accel1'
GPU_ROOT='/root/coopmwm20261101_accel1'


def pending():
    return json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});assert r.is_dir()
admissions=sorted(r.glob('WORKER_ADMISSION_*.json'))
out=dict(stop=(r/'STOP.json').exists() or (r/'COMPLETE.json').exists(),workers=json.loads((admissions[-1] if admissions else r/'WORKERS.json').read_text())['workers'],records=[])
if not out['stop']:
 for p in (r/'requests').glob('*.json'):
  if not (r/'responses'/p.name).exists():out['records'].append(json.loads(p.read_text()))
if (r/'PRIMARY_TRAINING_COMPLETE.json').exists() and not out['records']:out['stop']=True
if (r/'VIABILITY_PAUSED.json').exists() and not out['records']:out['stop']=True
print(json.dumps(out))
''',timeout=25))


class Service:
    def __init__(self,slot):
        host,index=slot.split('@') if '@' in slot else (slot,None)
        root=GPU_ROOT+('_w'+index if index is not None else '')
        self.host=host
        log=(ops.EVIDENCE/f'ACCEL_GPU_{slot}.log').open('x',encoding='utf-8')
        command=f'nice -n 19 /root/miniconda3/bin/python -B -u -c "import sys;sys.path.insert(0,\'{GPU_ROOT}/pytools\');from pathlib import Path;from tools.coopmwm.plan_service import main;main(Path(\'{root}\'))"'
        self.process=subprocess.Popen(['ssh','-o','BatchMode=yes',host,command],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1)
        write(ops.EVIDENCE/f'ACCEL_GPU_{slot}_SSH.json',dict(pid=self.process.pid,unix=time.time(),host=host,slot=slot))
        write(ops.EVIDENCE/f'ACCEL_GPU_{slot}_READY.json',self.receive('READY'))

    def receive(self,event):
        while True:
            line=self.process.stdout.readline()
            if not line:raise ValueError('owned plan service EOF; no automatic retry')
            if line.startswith('COOPMWM_JSON '):
                v=json.loads(line[len('COOPMWM_JSON '):])
                if v['event']==event:return v

    def call(self,record):
        self.process.stdin.write(json.dumps(record)+'\n');self.process.stdin.flush()
        v=self.receive('RESULT');assert v['key']==record['state_id']
        assert v['result']['record_sha256']==digest(record)
        return v


def run():
    import msvcrt
    claim=(ops.EVIDENCE/'ACCEL_BRIDGE.claim').open('a+b');claim.seek(0)
    if not claim.read(1):claim.write(b'1');claim.flush()
    claim.seek(0);msvcrt.locking(claim.fileno(),msvcrt.LK_NBLCK,1)
    lock=threading.Lock();inflight=set();stop=threading.Event();workers={};errors=[]
    def worker(host):
        service=None
        try:
            service=Service(host)
            while not stop.is_set():
                with lock:
                    state=pending()
                    if state['stop']:stop.set();break
                    from .exploration import regime
                    candidates=sorted(state['records'],key=lambda r:(r.get('mode')=='diagnostic',regime(r['rows'])!='merge-dense',r['time_s'],r['state_id']))
                    r=next((r for r in candidates if r['state_id'] not in inflight),None)
                    if r:inflight.add(r['state_id'])
                if r is None:stop.wait(1);continue
                value=service.call(r)
                ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({ROOT!r});sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.storage import write
write(r/'responses'/{(r['state_id']+'.json')!r},json.loads({json.dumps(value['result'])!r}))
print('persisted')
''',timeout=35)
                write(ops.EVIDENCE/'accel_forecasts'/(r['state_id']+'.json'),dict(host=host,wall_s=value['wall_s'],unix=time.time()))
                with lock:inflight.remove(r['state_id'])
        except BaseException as e:
            errors.append(repr(e));stop.set()
            ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({ROOT!r});sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.storage import write
if not (r/'STOP.json').exists():write(r/'STOP.json',dict(error={repr(str(e))},source='owned plan bridge; no retry'))
print('STOP recorded')
''',timeout=25)
        finally:
            if service:service.process.stdin.close()
    while not stop.is_set():
        state=pending()
        if state['stop']:stop.set();break
        for host in state['workers']:
            if host not in workers:
                # Hot-added host must already have a frozen hash-bound binding and
                # parity qualification; Service verifies before announcing READY.
                t=threading.Thread(target=worker,args=(host,));workers[host]=t;t.start()
        stop.wait(2)
    for t in workers.values():t.join()
    if errors:raise ValueError(errors)


if __name__=='__main__':run()
