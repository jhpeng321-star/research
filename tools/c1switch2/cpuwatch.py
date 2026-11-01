"""OWNER20:14 CPU-only supervision; one unused CPU partition, no GPU admission."""
import argparse
import json
import os
import subprocess
import sys
import time

from tools.c1switch.storage import write
from . import gpuhold, selectorops, watch
from .operations import EVIDENCE, ssh
from .supervise import DEADLINE

SCOPE = EVIDENCE/'WATCH_OWNER2014'


def admit_cpu():
    # This observer can admit only the original unclaimed per-CAV cpuQ scope.
    # It cannot prepare/copy a stage, restart science, or call any GPU operation.
    if (EVIDENCE/'SELECTOR_DISPATCH_cav_mappo_cpuQ.json').exists() or (SCOPE/'CPU_ATTEMPT_ONCE.json').exists():
        return dict(status='ADMITTED_OR_ATTEMPTED_NO_RETRY')
    prepared = watch.read_json(selectorops.prepared_path('cav_mappo','cpuQ',EVIDENCE))
    if not prepared:
        return dict(status='WAIT_PREPARED_CPU_ONLY')
    states = ssh('cpuQ', f'''import json,os,shutil
from pathlib import Path
root=Path({prepared['root']!r})
print(json.dumps(dict(load=os.getloadavg()[0],free_bytes=shutil.disk_usage(root).free,
 claimed=(root/'SELECTOR_DISPATCH_ONCE.json').exists(),stopped=(root/'STOP').exists(),
 jobs=any((root/'jobs').iterdir()),ledger=(root/'accounting.jsonl').exists())))
''', timeout=30)
    if any(states[k] for k in ('claimed','stopped','jobs','ledger')):
        return dict(status='REMOTE_SCOPE_CLAIMED_OR_TERMINAL_NO_RETRY',states=states)
    if states['load'] >= 28 or states['free_bytes'] < 2*1024**3:
        return dict(status='WAIT_CPU_DISK_NO_CLAIM',states=states)
    with (SCOPE/'CPU_ATTEMPT_ONCE.json').open('x') as stream:
        json.dump(dict(unix=time.time(),states=states,arm='cav_mappo',host='cpuQ',root=prepared['root']),stream)
    selectorops.dispatch('cpuQ',1,'cav_mappo')
    return dict(status='ORIGINAL_CPU_PARTITION_ADMITTED_ONCE',states=states)


def run():
    watch.WATCH = SCOPE
    with (SCOPE/'RUN_ONCE.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(),unix=time.time(),deadline_unix=DEADLINE),stream)
    heartbeat = 0
    while time.time() < DEADLINE:
        if time.time() >= heartbeat:
            try:
                snapshot = watch.collect()
                snapshot['owner2014'] = dict(pid=os.getpid(),gpu_hold=gpuhold.active(),
                    admission=watch.read_json(SCOPE/'ADMISSION_STATUS.json'),
                    admission_error=watch.read_json(SCOPE/'ADMISSION_ERROR.json'))
                watch.commit_snapshot(snapshot)
            except Exception as exc:
                write(SCOPE/'HEARTBEAT_ERROR.json',dict(unix=time.time(),error=repr(exc)))
            heartbeat = time.time()+1200
        try:
            write(SCOPE/'ADMISSION_STATUS.json',dict(unix=time.time(),**admit_cpu()))
        except Exception as exc:
            write(SCOPE/'ADMISSION_ERROR.json',dict(unix=time.time(),error=repr(exc)))
        time.sleep(60)
    write(SCOPE/'COMPLETE.json',dict(unix=time.time(),reason='original Nov2 noon observation deadline'))


def start():
    if not gpuhold.active() or watch.local_process(42548)['state'] != 'EXITED':
        raise ValueError('durable owner GPU hold and exited old admission observer required')
    if SCOPE.exists() or not EVIDENCE.is_dir():
        raise ValueError('observer already claimed or evidence parent absent')
    SCOPE.mkdir()
    with (SCOPE/'WATCH.log').open('x') as log:
        process = subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch2.cpuwatch','run'],
            cwd=watch.ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    result = dict(pid=process.pid,unix=time.time(),deadline_unix=DEADLINE,
        interval_s=1200,resource_observation_s=60,scope=str(SCOPE),gpu_admission=False,
        predecessor_pid=42548,allowed_cpu_arm='cav_mappo',allowed_cpu_host='cpuQ')
    write(SCOPE/'DISPATCH.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('start','run'))
    {'start':start,'run':run}[parser.parse_args().command]()
