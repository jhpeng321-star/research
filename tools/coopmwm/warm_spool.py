"""Durable input adapter for the unchanged, source-bound warm GPU service.

Run this entry point detached on the GPU host. SSH only delivers immutable input
files and reads forecasts; it never owns the model process or its input stream.
An interrupted inference is NOT replayed: its persistent claim requires review.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def immutable(path,value):
    """Same-payload transport replay is allowed; a conflict is fatal."""
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('conflicting immutable spool receipt '+path.name)
        return False
    # Broker/worker ownership serializes each destination. Publish atomically so
    # polling cannot see a partial JSON object after a connection disappears.
    temporary=path.with_name(path.name+'.publishing')
    with temporary.open('xb') as f:
        f.write(canonical(value));f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    return True


def key_of(record):
    key=record['state_id']
    if not isinstance(key,str) or not re.fullmatch(r'[0-9a-f]{64}',key):
        raise ValueError('invalid observable request identity')
    return key


def submit(root,record):
    root=Path(root);key=key_of(record)
    assert root.is_dir() and (root/'GPU_BINDING.json').is_file()
    inbox=root/'inbox';inbox.mkdir(exist_ok=True)
    immutable(inbox/(key+'.json'),record)
    return key


class SpoolInput:
    def __init__(self,root,deadline,*,sleep=time.sleep,clock=time.time):
        self.root=Path(root);self.deadline=deadline;self.sleep=sleep;self.clock=clock
        for name in ('inbox','inference_claims','forecasts'):
            (self.root/name).mkdir(exist_ok=True)

    def __iter__(self):
        while True:
            if (self.root/'SPOOL_STOP.json').exists():return
            if self.clock()>self.deadline:raise ValueError('registered GPU deadline')
            for path in sorted((self.root/'inbox').glob('*.json')):
                record=json.loads(path.read_text());key=key_of(record)
                if path.name!=key+'.json':raise ValueError('spool filename/request mismatch')
                destination=self.root/'forecasts'/path.name
                claim=self.root/'inference_claims'/path.name
                if destination.exists():
                    if not claim.exists():raise ValueError('forecast lacks durable inference claim')
                    previous=json.loads(claim.read_text())
                    if previous['request_sha256']!=hashlib.sha256(canonical(record)).hexdigest():
                        raise ValueError('completed inference input changed')
                    result=json.loads(destination.read_text())
                    if result['key']!=key:raise ValueError('forecast identity mismatch')
                    continue
                if claim.exists():raise ValueError('interrupted inference; inspect claim, never replay')
                immutable(claim,dict(state_id=key,request_sha256=hashlib.sha256(canonical(record)).hexdigest(),
                    pid=os.getpid(),unix=self.clock(),new_sumo_sim_s=0))
                yield canonical(record).decode()+'\n'
                if not destination.exists():raise ValueError('warm service returned without durable forecast')
            self.sleep(2)


def main(root):
    import fcntl
    import signal
    import sys
    root=Path(root);assert root.is_dir()
    # Lock BEFORE bootstrap/startup qualification, not just before serving.
    claim=(root/'SPOOL_WORKER.claim').open('a')
    fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    binding=json.loads((root/'GPU_BINDING.json').read_text())
    assert binding.get('memory_mode')=='warm' and binding.get('startup_qualification')
    assert binding.get('speed2') and binding.get('speed2_transfers')
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    owner=dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),new_sumo_sim_s=0)
    immutable(root/'SPOOL_OWNER.json',owner)
    original=sys.stdin
    try:
        sys.stdin=SpoolInput(root,binding['deadline_unix'])
        from .warm_service import main as service
        service(root)
    except BaseException as error:
        immutable(root/'SPOOL_FAILED.json',dict(owner=owner,error=repr(error),unix=time.time(),
            inference_retry=False,new_sumo_sim_s=0))
        raise
    finally:
        sys.stdin=original;claim.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
