"""Owned metadata-only waiter for the outstanding cpuP exposure audit.

Never dispatches physics or GPU inference. Successful audit writes immutable
receipts for the next runtime-binding step; no source/authority is auto-amended.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .ops import EVIDENCE,LOCAL,audit_host
from .storage import read,write


def run():
    import msvcrt
    claim=(EVIDENCE/'AUDIT_WATCH.claim').open('a+b')
    claim.seek(0);msvcrt.locking(claim.fileno(),msvcrt.LK_NBLCK,1)
    owner=dict(pid=os.getpid(),started_unix=time.time(),poll_s=300,
               maximum_wall_s=21600,mode='cpuP metadata exposure only',SUMO_sim_s=0)
    write(EVIDENCE/'AUDIT_WATCH_OWNER.json',owner)
    start=time.monotonic();attempt=0
    try:
        while time.monotonic()-start<21600:
            attempt+=1
            try:
                if not (EVIDENCE/'FINAL_EXPOSURE_cpuP.json').exists():audit_host('cpuP')
                result=read(EVIDENCE/'FINAL_EXPOSURE_cpuP.json')
                if not result['pass_']:raise ValueError('cpuP exposure collision; preserve and report')
                write(EVIDENCE/'AUDIT_WATCH_COMPLETE.json',dict(owner=owner,audit=result,
                    status='CPU_P_AUDIT_AVAILABLE_RUNTIME_BINDING_STILL_REQUIRED',SUMO_sim_s=0))
                return
            except Exception as exc:
                receipt=dict(attempt=attempt,unix=time.time(),error=str(exc),SUMO_sim_s=0)
                write(EVIDENCE/f'AUDIT_WAIT_{attempt:03}.json',receipt)
                print(json.dumps(receipt),flush=True)
                if (EVIDENCE/'FINAL_EXPOSURE_cpuP.json').exists():raise
                time.sleep(300)
        write(EVIDENCE/'AUDIT_WATCH_EXPIRED.json',dict(owner=owner,SUMO_sim_s=0))
    finally:claim.close()


def launch():
    assert EVIDENCE.is_dir() and LOCAL.is_dir()
    if (EVIDENCE/'AUDIT_WATCH_DISPATCH.json').exists():raise ValueError('one-use audit watch dispatch already exists')
    with (EVIDENCE/'AUDIT_WATCH.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-B','-m','tools.coopmwm.audit_watch'],
            cwd=LOCAL,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    receipt=dict(pid=child.pid,started_unix=time.time(),argv=['python','-B','-m','tools.coopmwm.audit_watch'],
                 poll_s=300,maximum_wall_s=21600,SUMO_sim_s=0)
    write(EVIDENCE/'AUDIT_WATCH_DISPATCH.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    if args.launch:launch()
    else:run()
