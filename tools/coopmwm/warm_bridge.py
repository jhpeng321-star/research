"""Warm-only one-use broker; no local torch dependency, no cold service adoption."""
import json
import subprocess
import threading
import time
from . import ops
from .storage import read,write
from .plan_bridge import Service as OriginalService
from .warm_deploy import ROOT

GPU_ROOT='/root/coopmwm20261102_warmservice1'


class Service(OriginalService):
    def __init__(self,slot):
        host,index=slot.split('@');self.host=host
        root=GPU_ROOT+'_w'+index
        log=(ops.EVIDENCE/f'WARM_GPU_{slot}.log').open('x',encoding='utf-8')
        command=f'nice -n 19 /root/miniconda3/bin/python -B -u -c "import sys;sys.path.insert(0,\'{root}/pytools\');from pathlib import Path;from tools.coopmwm.warm_service import main;main(Path(\'{root}\'))"'
        self.process=subprocess.Popen(['ssh','-o','BatchMode=yes',host,command],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1)
        write(ops.EVIDENCE/f'WARM_GPU_{slot}_SSH.json',dict(pid=self.process.pid,unix=time.time(),host=host,slot=slot))
        write(ops.EVIDENCE/f'WARM_GPU_{slot}_READY.json',self.receive('READY'))


def pending():
    return json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({ROOT!r});assert r.is_dir();sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.storage import read
from tools.coopmwm.exploration import regime
records=[]
for p in (r/'requests').glob('*.json'):
 if not (r/'responses'/p.name).exists():records.append(read(p))
records.sort(key=lambda v:(v.get('mode')=='diagnostic',regime(v['rows'])!='merge-dense',v['time_s'],v['state_id']))
admissions=sorted(r.glob('WARM_WORKER_ADMISSION_*.json'))
out=dict(stop=(r/'STOP.json').exists() or ((r/'WARM_B_PAUSED.json').exists() and (r/'WARM_R_PAUSED.json').exists() and not records),
 workers=read(admissions[-1] if admissions else r/'WORKERS.json')['workers'],records=records)
print(json.dumps(out))''',timeout=30))


def run():
    import msvcrt
    claim=(ops.EVIDENCE/'WARM_BRIDGE.claim').open('a+b');claim.seek(0)
    if not claim.read(1):claim.write(b'1');claim.flush()
    claim.seek(0);msvcrt.locking(claim.fileno(),msvcrt.LK_NBLCK,1)
    lock=threading.Lock();inflight=set();stop=threading.Event();workers={};errors=[]
    def worker(slot):
        service=None
        try:
            service=Service(slot)
            while not stop.is_set():
                with lock:
                    state=pending()
                    if state['stop']:stop.set();break
                    record=next((r for r in state['records'] if r['state_id'] not in inflight),None)
                    if record:inflight.add(record['state_id'])
                if record is None:stop.wait(2);continue
                result=service.call(record)
                ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({ROOT!r});sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.storage import write
write(r/'responses'/{(record['state_id']+'.json')!r},json.loads({json.dumps(result['result'])!r}))
print('persisted')''',timeout=45)
                write(ops.EVIDENCE/'warm_forecasts'/(record['state_id']+'.json'),
                    dict(slot=slot,wall_s=result['wall_s'],unix=time.time(),state_id=record['state_id']))
                with lock:inflight.remove(record['state_id'])
        except BaseException as error:
            errors.append(repr(error));stop.set()
            ops.ssh.remote('cpuQ',f'''import pathlib,sys
r=pathlib.Path({ROOT!r});sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.storage import write
if not (r/'STOP.json').exists():write(r/'STOP.json',dict(error={str(error)!r},source='owned warm bridge; no automatic retry'))
print('STOP recorded')''',timeout=30)
        finally:
            if service:service.process.stdin.close()
    while not stop.is_set():
        state=pending()
        if state['stop']:stop.set();break
        for slot in state['workers']:
            if slot not in workers:
                thread=threading.Thread(target=worker,args=(slot,));workers[slot]=thread;thread.start()
        stop.wait(5)
    for thread in workers.values():thread.join()
    if errors:raise ValueError(errors)


if __name__=='__main__':run()
