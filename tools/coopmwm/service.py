"""Owned GPU stdio service; low-priority admission between complete requests."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .storage import read,write


def busy():
    """Yield to WMPLAN's queued forecasts; never signal or change peer processes."""
    root=Path('/root/autodl-tmp/wmplan20261101_v2')
    if not root.exists() or (root/'STOP.json').exists() or (root/'COMPLETE.json').exists():return False
    pending=list((root/'requests').glob('*.json'))
    return any(not (root/'responses'/p.name).exists() for p in pending)


def main(root):
    os.nice(19)
    started=time.monotonic()
    from .protocol import sha
    identity=read(root/'GPU_BINDING.json')
    released=identity.get('owner_release')=='2026-11-01 17:55'
    for n,h in identity['hashes'].items():
        if sha(root/n)!=h:raise ValueError('immutable GPU source mismatch '+n)
    os.environ['WMTEACH_BACKEND']='FULL-v4'
    while True:
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free>12*1024 and (released or not busy()):break
        time.sleep(10)
    with contextlib.redirect_stdout(sys.stderr):
        from tools.wmteach.runtime import bootstrap
        rt=bootstrap()
        from .wm import cost_adapter
        qualified=read(root/'FROZEN_COSTS.json')
        rt.frozen.costs=cost_adapter(rt.frozen.costs,qualified['code'])
    owner=dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),backend='FULL-v4')
    write(root/'GPU_READY.json',owner)
    print('COOPMWM_JSON '+json.dumps(dict(event='READY',owner=owner)),flush=True)
    for line in sys.stdin:
        if time.monotonic()-started>24*3600 or time.time()>identity.get('deadline_unix',float('inf')):
            raise ValueError('registered 24h GPU service wall cap reached')
        record=json.loads(line);key=record['state_id'];dest=root/'forecasts'/key;begin=time.monotonic()
        if (dest/'RESULT.json').exists():result=read(dest/'RESULT.json')
        else:
            while not released and busy():time.sleep(5)
            write(dest/'CLAIM.json',dict(owner=owner,state_id=key,mode=record.get('mode','diagnostic')))
            with contextlib.redirect_stdout(sys.stderr):
                from .wm import evaluate,propose
                used=sum(read(p)['mode']=='proposal' for p in root.glob('forecasts/*/CLAIM.json'))
                if record.get('mode')=='proposal' and used>identity.get('proposal_cap',360):
                    result=dict(accepted=False,option=None,targets={},expected_saving=0.,
                        reason='registered GPU forecast cap reached',record_sha256=__import__('tools.coopmwm.protocol',fromlist=['digest']).digest(record))
                else:
                    result=propose(rt,record) if record.get('mode')=='proposal' else evaluate(rt,record)
                if not (root/'GPU_PARITY.json').exists():
                    from .wm import canary
                    check=canary(rt,record,result)
                    if not check.get('deferred'):write(root/'GPU_PARITY.json',check)
            write(dest/'RESULT.json',result)
        print('COOPMWM_JSON '+json.dumps(dict(event='RESULT',key=key,result=result,wall_s=time.monotonic()-begin)),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
