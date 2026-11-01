"""Owned batched dual-GPU proposal transport, isolated from historical E2 queue."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import threading
import time
from . import ops,dual,deploy
from .protocol import digest
from .storage import read,write

TAG='1854v1'
GPU_ROOT='/root/coopmwm20261101_volume1'
QUEUE=deploy.ROOT+'/GUIDANCE_1854'


def pending():
    return json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({QUEUE!r});assert r.is_dir()
records=[]
stop=(r.parent/'STOP.json').exists()
if not stop:
 for p in sorted((r/'requests').glob('*.json'),key=lambda p:p.stat().st_mtime):
  if not (r/'responses'/p.name).exists():records.append(json.loads(p.read_text()))
print(json.dumps(dict(stop=stop,records=records)))
''',timeout=30))


def persist(records,value):
    assert value['keys']==[r['state_id'] for r in records]
    assert all(v['record_sha256']==digest(r) for r,v in zip(records,value['results']))
    return ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os
r=pathlib.Path({QUEUE!r});sys.path.insert(0,{GPU_ROOT!r}+'/pytools')
values=json.loads({json.dumps(value['results'])!r});keys=json.loads({json.dumps(value['keys'])!r})
for key,v in zip(keys,values):
 p=r/'responses'/(key+'.json');p.parent.mkdir(exist_ok=True)
 if p.exists():assert json.loads(p.read_text())==v
 else:
  temp=r/'responses'/(key+'.pending')
  with temp.open('x') as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
  os.link(temp,p)
print(json.dumps(dict(persisted=len(keys))))
''',timeout=35)


class Service(dual.Service):
    def __init__(self,host):
        self.host=host
        log=(ops.EVIDENCE/f'GPU_{host}_{TAG}.log').open('x',encoding='utf-8')
        command=('/root/miniconda3/bin/python -B -u -c '
            '"import sys;sys.path.insert(0,\''+GPU_ROOT+'/pytools\');'
            'from pathlib import Path;from tools.coopmwm.volume_service import main;main(Path(\''+GPU_ROOT+'\'))"')
        self.process=subprocess.Popen(['ssh','-o','BatchMode=yes',host,command],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1)
        write(ops.EVIDENCE/f'GPU_{host}_{TAG}_SSH_OWNER.json',dict(pid=self.process.pid,host=host,root=GPU_ROOT,unix=time.time()))
        write(ops.EVIDENCE/f'GPU_{host}_{TAG}_READY.json',self.receive('READY'))

    def call(self,records):
        self.process.stdin.write(json.dumps(dict(batch=records))+'\n');self.process.stdin.flush()
        return self.receive('BATCH_RESULT')


def run():
    import msvcrt
    claim=(ops.EVIDENCE/f'DUAL_{TAG}.claim').open('a+b');claim.seek(0)
    if not claim.read(1):claim.write(b'1');claim.flush()
    claim.seek(0);msvcrt.locking(claim.fileno(),msvcrt.LK_NBLCK,1)
    amendment=read(ops.EVIDENCE/'AMENDMENT_1854.json')
    assert amendment['volume']['release_guided']
    lock=threading.Lock();inflight=set();stop=threading.Event();services=[]
    try:
        with ThreadPoolExecutor(2) as pool:
            services=list(pool.map(Service,('gpuA','gpuR')))
            def worker(service):
                while not stop.is_set():
                    with lock:
                        item=pending()
                        if item['stop']:stop.set();return
                        records=[r for r in item['records'] if r['state_id'] not in inflight][:amendment['volume']['batch_windows']]
                        inflight.update(r['state_id'] for r in records)
                    if not records:stop.wait(1);continue
                    value=service.call(records);persist(records,value)
                    write(ops.EVIDENCE/'volume_forecasts'/(digest(value['keys'])+'.json'),dict(host=service.host,
                        keys=value['keys'],wall_s=value['wall_s'],unix=time.time(),queries=len(records)))
                    with lock:inflight.difference_update(value['keys'])
                    print(json.dumps(dict(host=service.host,queries=len(records),wall_s=value['wall_s'])),flush=True)
            tasks=[pool.submit(worker,s) for s in services]
            for f in tasks:f.add_done_callback(lambda f:stop.set() if f.exception() else None)
            for f in tasks:f.result()
    except BaseException as error:
        dest=ops.EVIDENCE/f'DUAL_{TAG}_ERROR.json'
        if not dest.exists():write(dest,dict(error=repr(error),unix=time.time(),no_redispatch=True))
        raise
    finally:
        for service in services:service.close()
        claim.close()


if __name__=='__main__':run()
