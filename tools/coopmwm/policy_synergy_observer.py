"""Read-only local receipt mirror; detached from the interactive agent lifetime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from . import ops, policy_synergy_ops as ps
from .protocol import sha
from .storage import read, write

RECEIPT=ops.EVIDENCE/'POLICY_SYNERGY_OBSERVER_DISPATCH.json'


def live(d):
    try:
        p=psutil.Process(d['pid'])
        return p.create_time()==d['create_time'] and 'tools.coopmwm.policy_synergy_observer' in p.cmdline()
    except (psutil.NoSuchProcess,psutil.AccessDenied):return False


def run():
    claim=ops.EVIDENCE/'POLICY_SYNERGY_OBSERVER.claim'
    # One-use ownership even across duplicate launches; never controls a remote job.
    with claim.open('x') as f:json.dump(dict(pid=os.getpid(),unix=time.time()),f)
    deadline=read(ops.EVIDENCE/ps.REG)['gpu_deadline']
    while time.time()<deadline:
        try:
            value=ps.status()
            if all(f'POLICY_SYNERGY_u024_{a}.json' in value['reports'] for a in ('B','R')):return
        except Exception as e:
            write(ops.EVIDENCE/f'POLICY_SYNERGY_OBSERVER_ERROR_{time.time_ns()}.json',
                  dict(unix=time.time(),error=repr(e),read_only=True))
        time.sleep(120)


def start():
    if RECEIPT.exists():
        d=read(RECEIPT);print(json.dumps(dict(owner=d,alive=live(d),adopt=True)));return
    assert ops.EVIDENCE.is_dir()
    with (ops.EVIDENCE/'POLICY_SYNERGY_OBSERVER.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','tools.coopmwm.policy_synergy_observer','run'],
            cwd=str(ops.LOCAL),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0,
            start_new_session=os.name!='nt')
    d=dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),unix=time.time(),
           source_sha256=sha(Path(__file__)),read_only=True,interval_s=120,
           purpose='mirror policy synergy receipts; no physics or training dispatch')
    write(RECEIPT,d);print(json.dumps(d))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cmd',choices=('start','run','status'))
    cmd=p.parse_args().cmd
    if cmd=='status':
        d=read(RECEIPT);print(json.dumps(dict(owner=d,alive=live(d))))
    else:globals()[cmd]()
