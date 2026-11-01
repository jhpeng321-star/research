"""MAIN12:14 supervision: gate original unstarted work once, never retry science."""
import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
import time

from tools.c1switch.storage import write
from . import bbcontinue, selectorops, watch
from . import gpuhold
from .operations import EVIDENCE, ssh

ROOT = watch.ROOT
SCOPE = EVIDENCE/'WATCH_MAIN1214'
DEADLINE = datetime.fromisoformat('2026-11-02T12:14:00+08:00').timestamp()


def admit():
    """Preclaim refusal is observable; any admitted attempt is durably one-use."""
    if gpuhold.active():
        return dict(status='OWNER2014_GPU_HOLD_NO_ADMISSION')
    states = ssh('gpuR', '''import json,os,shutil
print(json.dumps(dict(load=os.getloadavg()[0],free_bytes=shutil.disk_usage('/root').free)))''',timeout=30)
    if states['load'] >= 20 or states['free_bytes'] < 2*1024**3:
        return dict(status='WAIT_CPU_DISK_NO_CLAIM',states=states)
    fleet = EVIDENCE/'SELECTOR_DISPATCH_fleet_ppo_gpuR.json'
    if not fleet.exists() and not (SCOPE/'FLEET_ATTEMPT_ONCE.json').exists():
        with (SCOPE/'FLEET_ATTEMPT_ONCE.json').open('x') as stream:
            json.dump(dict(unix=time.time(),states=states),stream)
        selectorops.dispatch('gpuR',1)
    if bbcontinue.CLAIM.exists() or (SCOPE/'BB_ATTEMPT_ONCE.json').exists():
        return dict(status='ADMITTED_OR_TERMINAL_NO_RETRY',states=states)
    config = json.loads(bbcontinue.BOUND.read_text())
    gpu_states,ready = bbcontinue.resources(config)
    if not ready:
        return dict(status='WAIT_GPU_CPU_DISK_NO_CLAIM',states=gpu_states)
    with (SCOPE/'BB_ATTEMPT_ONCE.json').open('x') as stream:
        json.dump(dict(unix=time.time(),states=gpu_states),stream)
    bbcontinue.start()
    return dict(status='ONE_USE_ADMISSION_COMPLETE',states=gpu_states)


def run():
    watch.WATCH = SCOPE
    with (SCOPE/'RUN_ONCE.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(),unix=time.time(),deadline_unix=DEADLINE),stream)
    heartbeat = 0
    while time.time() < DEADLINE:
        if time.time() >= heartbeat:
            try:
                snapshot = watch.collect()
                snapshot['postnoon_supervision'] = dict(pid=os.getpid(),
                    admission=watch.read_json(SCOPE/'ADMISSION_STATUS.json'),
                    admission_error=watch.read_json(SCOPE/'ADMISSION_ERROR.json'))
                watch.commit_snapshot(snapshot)
            except Exception as exc:
                write(SCOPE/'HEARTBEAT_ERROR.json',dict(unix=time.time(),error=repr(exc)))
            heartbeat = time.time()+1200
        try:
            write(SCOPE/'ADMISSION_STATUS.json',dict(unix=time.time(),**admit()))
        except Exception as exc:
            # Attempt receipts are permanent: no restart, replay or science retry.
            write(SCOPE/'ADMISSION_ERROR.json',dict(unix=time.time(),error=repr(exc)))
        time.sleep(60)
    write(SCOPE/'COMPLETE.json',dict(unix=time.time(),reason='bounded 24h observation window ended'))


def start():
    if SCOPE.exists():
        raise ValueError('supervision already claimed; inspect exact owned process')
    if watch.local_process(25348)['state'] != 'EXITED':
        raise ValueError('original noon observer still live; do not duplicate')
    if not EVIDENCE.is_dir():
        raise ValueError('evidence parent absent')
    SCOPE.mkdir()
    with (SCOPE/'SUPERVISION.log').open('x') as log:
        process = subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch2.supervise','run'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    result = dict(pid=process.pid,unix=time.time(),deadline_unix=DEADLINE,
                  interval_s=1200,resource_observation_s=60,scope=str(SCOPE))
    write(SCOPE/'DISPATCH.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('start','run'))
    {'start':start,'run':run}[parser.parse_args().command]()
