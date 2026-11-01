"""Single owned dual-GPU bridge; immutable requests, no simulation dispatch."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from . import ops
from .deploy import ROOT,EVIDENCE
from .protocol import digest
from .storage import read,write

GPU_ROOT='/root/coopmwm20261101_dual1'
TAG='1755'


def pending():
    return json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});assert r.is_dir()
records=[]
if not (r/'STOP.json').exists():
 for p in (r/'requests').glob('*.json'):
  if not (r/'responses'/p.name).exists():records.append(json.loads(p.read_text()))
print(json.dumps(dict(stop=(r/'STOP.json').exists(),records=records)))
''',timeout=35))


def persist(record,result):
    assert result['record_sha256']==digest(record)
    return ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({ROOT!r});sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.storage import read,write
p=r/'responses'/{(record['state_id']+'.json')!r};value=json.loads({json.dumps(result)!r})
if p.exists():assert read(p)==value
else:write(p,value)
print(json.dumps(dict(persisted=True)))
''',timeout=45)


class Service:
    def __init__(self,host):
        self.host=host
        log=(EVIDENCE/f'GPU_{host}_{TAG}.log').open('x',encoding='utf-8')
        command=('/root/miniconda3/bin/python -B -u -c '
            '"import sys;sys.path.insert(0,\''+GPU_ROOT+'/pytools\');'
            'from pathlib import Path;from tools.coopmwm.service import main;main(Path(\''+GPU_ROOT+'\'))"')
        self.process=subprocess.Popen(['ssh','-o','BatchMode=yes',host,command],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1)
        write(EVIDENCE/f'GPU_{host}_{TAG}_SSH_OWNER.json',dict(pid=self.process.pid,host=host,root=GPU_ROOT,unix=time.time()))
        ready=self.receive('READY');write(EVIDENCE/f'GPU_{host}_{TAG}_READY.json',ready)

    def receive(self,event):
        while True:
            line=self.process.stdout.readline()
            if not line:raise RuntimeError(f'{self.host} service EOF before {event}; no redispatch')
            if line.startswith('COOPMWM_JSON '):
                value=json.loads(line[len('COOPMWM_JSON '):])
                if value['event']==event:return value

    def call(self,record):
        self.process.stdin.write(json.dumps(record)+'\n');self.process.stdin.flush()
        value=self.receive('RESULT');assert value['key']==record['state_id']
        return value

    def close(self):
        self.process.stdin.close()  # EOF only to this owned service.


def credit_difference(a,b):
    differences=[]
    for h in (60,120,300,600):
        x,y=a['credits'][str(h)],b['credits'][str(h)]
        differences.extend(abs(x[k]-y[k]) for k in ('D_G','C_G'))
        differences.extend(abs(i-j) for i,j in zip(x['D_i'],y['D_i']))
    return max(differences)


def run():
    import msvcrt
    claim=(EVIDENCE/f'DUAL_{TAG}.claim').open('a+b')
    claim.seek(0)
    if not claim.read(1):claim.write(b'1');claim.flush()
    claim.seek(0);msvcrt.locking(claim.fileno(),msvcrt.LK_NBLCK,1)
    services=[]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(Service,h) for h in ('gpuA','gpuR')]):
                services.append(future.result())
            services.sort(key=lambda s:s.host)
            first=next(r for r in pending()['records'] if r.get('mode')!='proposal')
            values=list(pool.map(lambda s:s.call(first),services))
            difference=credit_difference(*(v['result'] for v in values))
            # Same frozen draws/model/reducer; gate both hosts before routing proposals.
            if difference>1e-3:raise ValueError('cross-GPU frozen credit parity mismatch '+str(difference))
            write(EVIDENCE/f'DUAL_{TAG}_PARITY.json',dict(pass_=True,max_credit_error=difference,
                hosts=['gpuA','gpuR'],wall_s={s.host:v['wall_s'] for s,v in zip(services,values)},sim_s=0))
            persist(first,values[0]['result'])
            # A no-SUMO proposal timing on the same observable roster, not an outcome selector.
            probe=dict(first,mode='proposal',setting=0)
            probe['state_id']=digest(dict(first=first['state_id'],mode='throughput-probe'))
            timings=list(pool.map(lambda s:s.call(probe),services))
            write(EVIDENCE/f'DUAL_{TAG}_THROUGHPUT.json',dict(unix=time.time(),sim_s=0,
                hosts={s.host:dict(wall_s=v['wall_s'],K=8,horizon_s=300,
                    available=len(v['result'].get('means',{}))-2) for s,v in zip(services,timings)},
                aggregate_queries_per_s=sum(1/v['wall_s'] for v in timings)))
            lock=threading.Lock();inflight=set();stop=threading.Event()
            def worker(service):
                while not stop.is_set():
                    record=None
                    with lock:
                        item=pending()
                        if item['stop']:stop.set();return
                        # Training proposals take priority; diagnostic never gates training.
                        candidates=sorted(item['records'],key=lambda r:(r.get('mode')!='proposal',r['state_id']))
                        record=next((r for r in candidates if r['state_id'] not in inflight),None)
                        if record:inflight.add(record['state_id'])
                    if record is None:stop.wait(2);continue
                    value=service.call(record);persist(record,value['result'])
                    write(EVIDENCE/'dual_forecasts'/(record['state_id']+'.json'),dict(host=service.host,
                        mode=record.get('mode','diagnostic'),wall_s=value['wall_s'],unix=time.time()))
                    with lock:inflight.remove(record['state_id'])
                    print(json.dumps(dict(host=service.host,key=record['state_id'],wall_s=value['wall_s'])),flush=True)
            futures=[pool.submit(worker,s) for s in services]
            for f in futures:f.add_done_callback(lambda f:stop.set() if f.exception() else None)
            for f in futures:f.result()
    except BaseException as error:
        dest=EVIDENCE/f'DUAL_{TAG}_ERROR.json'
        if not dest.exists():write(dest,dict(error=repr(error),unix=time.time()))
        raise
    finally:
        for s in services:s.close()
        claim.close()


def bind(cpu_name):
    cpu=read(EVIDENCE/cpu_name);assert cpu['pass_'] and not cpu['skips'] and not cpu['sim_s']
    for host in ('gpuA','gpuR'):
        staged=json.loads(ops.stage_frozen(host,GPU_ROOT,cpu));assert staged['hashes']==cpu['source_hashes']
        out=ops.ssh.remote(host,f'''import pathlib,json,sys,hashlib
r=pathlib.Path({GPU_ROOT!r});sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.storage import read,write
v=dict(hashes=json.loads({json.dumps(staged['hashes'])!r}),backend='FULL-v4',source='22cc32f6',
 host={host!r},memory_cap_GiB=8,owner_release='2026-11-01 17:55',proposal_cap=10000)
if (r/'GPU_BINDING.json').exists():assert read(r/'GPU_BINDING.json')==v
else:write(r/'GPU_BINDING.json',v)
print(json.dumps(dict(host={host!r},root=str(r),binding_sha256=hashlib.sha256((r/'GPU_BINDING.json').read_bytes()).hexdigest(),sim_s=0)))
''',timeout=45)
        write(EVIDENCE/f'DUAL_{host}_{TAG}_BINDING.json',json.loads(out));print(out)


def start():
    assert not (EVIDENCE/f'DUAL_{TAG}_DISPATCH.json').exists(),'inspect existing dispatch'
    for h in ('gpuA','gpuR'):assert (EVIDENCE/f'DUAL_{h}_{TAG}_BINDING.json').exists()
    log=(EVIDENCE/f'DUAL_{TAG}.log').open('x',encoding='utf-8')
    p=subprocess.Popen([sys.executable,'-B','-u','-m','tools.coopmwm.dual','run'],cwd=ops.LOCAL,
        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    d=dict(pid=p.pid,unix=time.time(),hosts=['gpuA','gpuR'],root=GPU_ROOT)
    write(EVIDENCE/f'DUAL_{TAG}_DISPATCH.json',d);print(json.dumps(d))


def fit_sampling():
    timing=read(EVIDENCE/f'DUAL_{TAG}_THROUGHPUT.json')
    assert all(v['available']>0 for v in timing['hosts'].values()),'timing requires real inherited options'
    metadata=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({ROOT!r});sys.path.insert(0,{('/home/ubuntu/coopmwm20261101_cpu19/pytools')!r})
from tools.coopmwm.exploration import regime
paths=list((r/'jobs').glob('s0_u001_*/RESULT.json'));assert len(paths)==48
starts=[(p.parent/'CLAIM.json').stat().st_mtime for p in paths]
ends=[(p.parent/'CLOSE.json').stat().st_mtime for p in paths]
counts=dict.fromkeys(('merge-dense','ordinary'),0)
for p in paths:
 e=json.loads(p.read_text())
 if e['arm']!='B':continue
 for s in e['steps']:
  if len(s['group'])==2:counts[regime(s['rows'])]+=1
print(json.dumps(dict(initial48episode_wall_s=max(ends)-min(starts),initial_workers=4,new_workers=12,complete_group_counts=counts)))
''',timeout=35))
    counts=metadata['complete_group_counts'];den=counts['merge-dense']+counts['ordinary']/4
    assert den>0
    p=min(.5,timing['aggregate_queries_per_s']*metadata['initial48episode_wall_s']*4/12/den)
    value=dict(metadata=metadata,timing=timing,probabilities={'merge-dense':p,'ordinary':p/4},
        fixed_for='u2+ B/R, both seeds; u1 historical rates',sim_s=0,fit_inputs='infrastructure timing and observable eligibility counts only')
    write(EVIDENCE/'SAMPLING_1755.json',value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('bind','start','run','fit'));p.add_argument('--cpu',default='CPU_coopmwm20261101_cpu19.json');a=p.parse_args()
    if a.mode=='bind':bind(a.cpu)
    elif a.mode=='start':start()
    elif a.mode=='fit':fit_sampling()
    else:run()
