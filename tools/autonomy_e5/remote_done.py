"""Detached heartbeat/mirror for exactly one S-E5DONE cpuQ child."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import remote as B

R = Path('/root/autodl-tmp/c4/restart20261026/e5')
CPU = '/home/ubuntu/e3long/operations_e5'


def drive():
    try:
        while True:
            state = B.ssh('cpuQ', f'''import pathlib,json,time
op=pathlib.Path({CPU!r})
(op/'MIRROR_HEARTBEAT.json').write_text(json.dumps(dict(t=time.time())))
failure=op/'DONE_FAILURE.json'; assert not failure.exists(),failure.read_text() if failure.exists() else ''
p=op/'DONE_STATUS.json'
print(json.dumps(json.loads(p.read_text()) if p.exists() else dict(status='STARTING')))
''')
            B.pull('cpuQ')
            B.atomic(R/'DONE_STATUS.json', state)
            assert state['status'] != 'FAILED', state
            if state['status'] == 'COMPLETE':
                B.pull('cpuP')
                return
            time.sleep(15)
    except BaseException as error:
        B.atomic(R/'DONE_STATUS.json', dict(status='WAIT_OWNER', error=repr(error), t=time.time()))
        raise


def launch():
    assert not (R/'DONE_PID.json').exists()
    child = B.ssh('cpuQ', f'''import pathlib,json,subprocess,os
op=pathlib.Path({CPU!r}); assert not (op/'DONE_PID.json').exists()
assert json.loads((op/'DONE_PREFLIGHT.json').read_text())['status']=='PASS'
home='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo'
(op/'MIRROR_HEARTBEAT.json').write_text(json.dumps(dict(t=__import__('time').time())))
with (op/'done_runner.log').open('xb') as log:
 p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u',str(op/'done_runner.py')],
 cwd='/home/ubuntu/e3long/runtime',env={{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','SUMO_HOME':home,'PATH':home+'/bin:/home/ubuntu/miniconda3/bin:/usr/local/bin:/usr/bin:/bin'}},
 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,sid=os.getsid(p.pid),start_ticks=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
assert d['sid']==p.pid
(op/'DONE_PID.json').write_text(json.dumps(d)); print(json.dumps(d))
''')
    B.atomic(R/'DONE_CHILD.json', child)
    with (R/'done_driver.log').open('xb') as log:
        p = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__)), 'drive'],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    receipt = dict(pid=p.pid, sid=os.getsid(p.pid), start_ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    assert receipt['sid'] == p.pid
    B.atomic(R/'DONE_PID.json', receipt)
    print(json.dumps(dict(coordinator=receipt, child=child)))


if __name__ == '__main__':
    assert sys.argv[1] in ('launch', 'drive')
    globals()[sys.argv[1]]()
