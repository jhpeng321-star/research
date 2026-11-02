"""Detached cpuQ broker: bounded SSH delivery, durable per-request assignment.

GPU model processes are independently detached. Resubmission only delivers an
identical input to the same spool; the spool's durable claim prevents inference
replay. No job/physics dispatch lives here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from .warm_spool import immutable
from .warm_transport import retry,SSHConnectionError
from .warm_health import assert_running,HealthPause
from .protocol import digest
from .warm_rpc import GATES


class Remote:
    def __init__(self,root,endpoint):
        self.root=Path(root);self.endpoint=endpoint;self.owner=None;self.receipts={}

    def raw(self,value):
        e=self.endpoint
        command=['ssh','-T','-o','BatchMode=yes','-o','IdentitiesOnly=yes',
            '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+e['known_hosts'],
            '-o','ConnectTimeout=20','-o','ConnectionAttempts=1',
            '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=8',
            '-i',e['identity_file'],'-p',str(e['port']),e['destination']]
        try:p=subprocess.run(command,input=json.dumps(value),capture_output=True,text=True,timeout=160)
        except subprocess.TimeoutExpired as error:
            raise SSHConnectionError('SSH metadata transport timed out') from error
        if p.returncode:
            error=SSHConnectionError if p.returncode==255 else RuntimeError
            raise error((p.stdout+p.stderr).strip())
        try:return json.loads(p.stdout)
        except json.JSONDecodeError as error:
            # Exit 0 with truncated/empty stdout is a dropped transport stream;
            # every RPC here is idempotent metadata, so it is bounded-retryable.
            raise SSHConnectionError('malformed RPC stdout ('+str(len(p.stdout))+' bytes)') from error

    def verify(self):
        s=self.raw(dict(action='status'))
        if not s['alive'] or s['failed'] or not s['ready']:raise ValueError('GPU worker not alive/qualified')
        owner=s['receipts']['SPOOL_OWNER.json']
        if self.owner is not None and owner!=self.owner:raise ValueError('GPU worker ownership changed')
        for name,previous in self.receipts.items():
            if s['receipts'].get(name)!=previous:raise ValueError('GPU immutable receipt changed '+name)
        self.owner=owner
        self.receipts={n:s['receipts'][n] for n in (*GATES,'SERVICE_OWNER.json')}
        return s

    def event(self,event):
        target=self.root/'broker_events';target.mkdir(exist_ok=True)
        value=dict(event,slot=self.endpoint['slot'],unix=time.time())
        immutable(target/(digest(value)+'.json'),value)

    def call(self,value):
        return retry(lambda:self.raw(value),self.verify,self.event)


def available(root):
    result=[]
    for requests,responses,kind in (('requests','responses','plan'),('centre_requests','centre_responses','centre')):
        if kind=='centre' and not (root/'WARM_B_PAUSED.json').exists():continue
        for p in (root/requests).glob('*.json'):
            if not (root/responses/p.name).exists():result.append((kind,json.loads(p.read_text())))
    from .exploration import regime
    result.sort(key=lambda v:(v[0]=='centre',v[1].get('mode')=='diagnostic',
        regime(v[1]['rows'])!='merge-dense',v[1]['time_s'],v[1]['state_id']))
    return result


def worker_endpoints(root,binding,previous):
    retired=root/'INCIDENT_RETIRED_JOBS.json'
    disabled=set(json.loads(retired.read_text()).get('disabled_slots',[])) if retired.exists() else set()
    admitted=list(binding['endpoints'])
    for p in sorted(root.glob('WARM_REMOTE_ENDPOINT_ADMISSION_*.json')):
        v=json.loads(p.read_text())
        if v['amendment_sha256'] not in (root/'AUTHORITY.md').read_text():raise ValueError('unregistered GPU hot-add')
        if v['search_sha256']!=binding['search_sha256']:raise ValueError('hot-add search differs')
        admitted.extend(v['endpoints'])
    result={k:v for k,v in previous.items() if k not in disabled}
    for e in admitted:
        slot=e['slot']
        if slot in disabled:continue
        if slot in result and result[slot]!=e:raise ValueError('worker endpoint changed')
        result[slot]=e
    return result


def run(root):
    import fcntl
    import signal
    root=Path(root);binding=json.loads((root/'WARM_REMOTE_BINDING.json').read_text())
    lock=(root/'WARM_REMOTE_BROKER.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    owner=dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
    immutable(root/'WARM_REMOTE_BROKER_OWNER.json',owner)
    assignments=root/'broker_assignments';assignments.mkdir(exist_ok=True)
    stop=threading.Event();mutex=threading.Lock();inflight=set();errors=[]
    def worker(endpoint):
        try:
            remote=Remote(root,endpoint);remote.call(dict(action='status'));remote.verify()
            while not stop.is_set():
                assert_running(root,'B')
                if time.time()>binding['deadline_unix']:raise ValueError('registered GPU deadline')
                with mutex:
                    item=None
                    for kind,record in available(root):
                        key=record['state_id'];p=assignments/(key+'.json')
                        assigned=json.loads(p.read_text()) if p.exists() else None
                        if key not in inflight and (assigned is None or assigned['slot']==endpoint['slot']):
                            immutable(p,dict(slot=endpoint['slot'],kind=kind,key=key,record_sha256=digest(record)))
                            inflight.add(key);item=(kind,record);break
                if item is None:stop.wait(2);continue
                kind,record=item;key=record['state_id'];begin=time.monotonic()
                remote.call(dict(action='submit',record=dict(record,mode='centre') if kind=='centre' else record))
                while True:
                    value=remote.call(dict(action='forecast',key=key))
                    if value['forecast'] is not None:break
                    if not value['status']['alive'] or value['status']['failed']:raise ValueError('GPU inference process failed; never replay')
                    assert_running(root,'B');stop.wait(3)
                payload=value['forecast']
                target=root/('centre_responses' if kind=='centre' else 'responses')/(key+'.json')
                immutable(target,payload['result'])
                immutable(root/'broker_completed'/(key+'.json'),dict(slot=endpoint['slot'],kind=kind,key=key,
                    inference_wall_s=payload['wall_s'],delivery_wall_s=time.monotonic()-begin,unix=time.time()))
                with mutex:inflight.remove(key)
        except BaseException as error:
            errors.append(repr(error));stop.set()
            if not isinstance(error,HealthPause):
                immutable(root/'WARM_GUIDED_STOP.json',dict(error=repr(error),source='detached source-bound broker',unix=time.time()))
    (root/'broker_completed').mkdir(exist_ok=True)
    threads={};endpoints={}
    try:
        while not stop.is_set():
            endpoints=worker_endpoints(root,binding,endpoints)
            for slot,e in endpoints.items():
                if slot not in threads:
                    threads[slot]=threading.Thread(target=worker,args=(e,));threads[slot].start()
            stop.wait(3)
    except BaseException as error:
        errors.append(repr(error));stop.set()
        immutable(root/'WARM_GUIDED_STOP.json',dict(error=repr(error),source='detached broker endpoint admission',unix=time.time()))
    finally:
        for t in threads.values():t.join()
    if errors:raise ValueError(errors)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
