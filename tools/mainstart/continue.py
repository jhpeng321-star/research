"""Install a tested driver correction only after its own current policy phase succeeds."""
import json
import os
from pathlib import Path
import subprocess
import time

A = Path('/root/autodl-tmp/c4/v5/mainstart')


def identity(pid):
    try:
        f = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if f[0] == 'Z' else f[19]
    except FileNotFoundError:
        return None


def main():
    request = json.loads((A / 'CONTINUATION_REQUEST.json').read_text())
    (A / 'CONTINUATION_WAITER.json').write_text(json.dumps({'pid': os.getpid(), 'sid': os.getsid(0)}))
    while identity(request['previous_pid']) == request['previous_start_ticks']:
        if (A / 'STOP_CONTINUATION').exists() or Path('/root/main14_loop/STOP_DRIVER').exists():
            print('owner stop: continuation cancelled', flush=True)
            return
        time.sleep(10)
    assert not (A / 'WAIT_OWNER.json').exists(), 'failed phase: continuation refused'
    state = json.loads(Path('/root/main14_loop/LOOP14_STATE.json').read_text())
    assert not state.get('stopped'), 'scientific/budget stop: continuation refused'
    assert state['cycles'][0]['phases']['policy']['status'] == 'done', 'policy did not succeed'
    assert state['cycles'][0]['phases']['collect']['status'] == 'pending' and state['sim_s_charged'] == 0
    assert not (A / 'STOP_CONTINUATION').exists() and not Path('/root/main14_loop/STOP_DRIVER').exists()
    (A / 'STOP_ARM').rename(A / 'STOP_ARM_v1_completed')
    with (A / 'driver_v2.log').open('ab') as log:
        p = subprocess.Popen(['setsid', 'nohup', '/root/miniconda3/bin/python', '-u', str(A / 'mainstart_v2.py'),
                              '--arm', str(A / 'ARMED_V2.json'), '--resume'],
                             cwd='/root/autodl-tmp/c4/v5/repo_main14', stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT,
                             env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    time.sleep(3)
    assert p.poll() is None, 'continuation exited; inspect driver_v2.log'
    (A / 'CONTINUATION_INSTALLED.json').write_text(json.dumps({'pid': p.pid, 'start_ticks': identity(p.pid)}))
    previous_handoff = json.loads((A / 'HANDOFF_INSTALLED.json').read_text())
    assert identity(previous_handoff['pid']) != previous_handoff['start_ticks'], 'old handoff has not exited'
    with (A / 'handoff.log').open('ab') as log:
        helper = subprocess.Popen(['setsid', 'nohup', '/root/miniconda3/bin/python', '-u', str(A / 'handoff.py')],
                                  cwd=A, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                  env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    time.sleep(3)
    assert helper.poll() is None
    (A / 'HANDOFF_CONTINUED.json').write_text(json.dumps({'pid': helper.pid, 'start_ticks': identity(helper.pid)}))
    print(json.dumps({'continued_pid': p.pid, 'start_ticks': identity(p.pid)}), flush=True)


if __name__ == '__main__':
    main()
