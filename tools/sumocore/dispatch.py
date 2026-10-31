"""Explicit one-shot panel dispatch with own PID/lock receipts and canary gate."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from worker import ROOT,write
from admission import amendments,effective


def main():
    lock=(ROOT/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.getloadavg()[0]<24
    plan=json.loads((ROOT/'plan.json').read_text())
    first=effective(plan['jobs'][0],amendments())
    out=ROOT/'jobs'/first['id']
    canary=json.loads((out/'result.json').read_text())
    assert canary['ok'] and canary['actual_sim_s']==600
    assert canary['initial_audit']['vehicles']==71
    assert canary['initial_audit']['max_position_error']<1e-5
    assert canary['initial_audit']['max_speed_error']<1e-5
    assert not canary['safety'].get('collision_involvements')
    assert not canary['safety'].get('starting_teleports')
    source=json.loads((out/'START.json').read_text())['code']
    for name in ('model_env.py','worker.py','prepare_bundle.py','demand.py','inference.py','supervisor.py','admission.py'):
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==source[name],name
    assert not (ROOT/'DISPATCH_full.json').exists()
    write(ROOT/'DISPATCH_intent.json',dict(unix=time.time(),canary=first['id'],phase='full',
        code=str(Path(__file__).parent),load=os.getloadavg(),threads=1,workers=8,nice=19))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    log=(ROOT/'full.log').open('x')
    process=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python',
        str(Path(__file__).parent/'supervisor.py'),'full'],stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    receipt=dict(pid=process.pid,unix=time.time(),code=str(Path(__file__).parent),canary=first['id'])
    write(ROOT/'DISPATCH_full.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
