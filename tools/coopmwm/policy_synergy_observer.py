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
ISOLATED=False


def live(d):
    try:
        p=psutil.Process(d['pid'])
        return p.create_time()==d['create_time'] and 'tools.coopmwm.policy_synergy_observer' in p.cmdline()
    except (psutil.NoSuchProcess,psutil.AccessDenied):return False


def run():
    claim=ops.EVIDENCE/('POLICY_SYNERGY_ISOLATED_OBSERVER.claim' if ISOLATED else 'POLICY_SYNERGY_OBSERVER.claim')
    # One-use ownership even across duplicate launches; never controls a remote job.
    with claim.open('x') as f:json.dump(dict(pid=os.getpid(),unix=time.time()),f)
    deadline=read(ops.EVIDENCE/ps.REG)['gpu_deadline']
    while time.time()<deadline:
        try:
            value=ps.status()
            if ISOLATED:
                from . import interim_r_ops
                interim_r_ops.status()
            suffix='_ISOLATED' if ISOLATED else ''
            if all(f'POLICY_SYNERGY_u024_{a}{suffix}.json' in value['reports'] for a in ('B','R')):return
        except Exception as e:
            prefix='POLICY_SYNERGY_ISOLATED_OBSERVER_ERROR' if ISOLATED else 'POLICY_SYNERGY_OBSERVER_ERROR'
            write(ops.EVIDENCE/f'{prefix}_{time.time_ns()}.json',
                  dict(unix=time.time(),error=repr(e),read_only=True))
        time.sleep(120)


def start():
    if RECEIPT.exists():
        d=read(RECEIPT);print(json.dumps(dict(owner=d,alive=live(d),adopt=True)));return
    assert ops.EVIDENCE.is_dir()
    log_name='POLICY_SYNERGY_ISOLATED_OBSERVER.log' if ISOLATED else 'POLICY_SYNERGY_OBSERVER.log'
    with (ops.EVIDENCE/log_name).open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','tools.coopmwm.policy_synergy_observer','run']+(['--isolated'] if ISOLATED else []),
            cwd=str(ops.LOCAL),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0,
            start_new_session=os.name!='nt')
    d=dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),unix=time.time(),
           source_sha256=sha(Path(__file__)),read_only=True,interval_s=120,
           purpose='mirror policy synergy receipts; no physics or training dispatch')
    write(RECEIPT,d);print(json.dumps(d))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cmd',choices=('start','run','status'))
    p.add_argument('--isolated',action='store_true');a=p.parse_args();cmd=a.cmd
    if a.isolated:
        from . import isolated_synergy_ops as ps
        ISOLATED=True
        RECEIPT=ops.EVIDENCE/'POLICY_SYNERGY_ISOLATED_OBSERVER_DISPATCH.json'
    if cmd=='status':
        d=read(RECEIPT);print(json.dumps(dict(owner=d,alive=live(d))))
    else:globals()[cmd]()
