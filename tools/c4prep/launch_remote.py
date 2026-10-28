"""Owned C4A operations; no simulator, peer process discovery or signals."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

r = Path('/root/autodl-tmp/c4/restart20261026')
out = r / 'arm/loop/c4prep'
pointer = json.loads((out / 'CURRENT_RUNTIME.json').read_text())
code = Path(pointer['runtime'])
sys.path[:0] = [str(code / 'tools/c4prep'), str(code / 'src'), str(code)]
os.environ.update(PYTHONPATH=f'{code}/src:{code}', C4_ALLOWED_ROOTS='/root/autodl-tmp/c4',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
from common import read, sha, write_new, now

action = sys.argv[1]
fit = out / 'launch_fit_v1'
resolved = out / 'resolved_A_MAIN1500'


def spawn(argv, root, name):
    with (root / (name + '.log')).open('x') as log:
        child = subprocess.Popen(argv, cwd=code, env=os.environ, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    receipt = {'pid': child.pid, 'start_ticks': ticks, 'argv': argv, 't': now()}
    write_new(root / (name + '_PID.json'), receipt)
    return receipt


def owned_alive(receipt):
    path = Path(f"/proc/{receipt['pid']}/stat")
    if not path.exists():
        return False
    values = path.read_text().rsplit(')', 1)[1].split()
    return values[19] == receipt['start_ticks'] and values[0] != 'Z'


if action == 'fit':
    assert read(out / 'MAIN_1500.json')['authority'] == 'MAIN2026-10-28T15:00'
    fit.mkdir(exist_ok=False)
    print(json.dumps(spawn([sys.executable, '-B', str(code / 'tools/c4prep/fit_coordinator.py'),
                            '--output', str(fit)], fit, 'COORDINATOR')))
elif action == 'fit-status':
    receipt = read(fit / 'COORDINATOR_PID.json')
    result = {'alive': owned_alive(receipt), 'pid': receipt['pid']}
    for name in ('FIT_DECISION.json', 'RESULT.json'):
        if (fit / name).exists():
            result[name] = read(fit / name)
    if (fit / 'replay.log').exists():
        result['log_tail'] = (fit / 'replay.log').read_text()[-6000:]
    print(json.dumps(result))
elif action == 'prepare':
    from launch_a import prepare
    prepare()
elif action == 'dry':
    print(json.dumps(spawn([sys.executable, '-B', str(code / 'tools/c4prep/launch.py'),
        'dry-run', '--config', str(resolved / 'CONFIG.json')], resolved, 'DRY_COORDINATOR')))
elif action == 'start':
    from launch import start
    start(resolved / 'CONFIG.json')
elif action in ('status', 'verify'):
    from launch_a import status
    print(json.dumps(status(verify=action == 'verify')))
elif action == 'inspect':
    print(json.dumps({'pointer': pointer, 'authority': read(out / 'MAIN_1500.json'),
        'wm2_gate': read(r / 'arm/loop/longint/gate_wm2/HORIZON_GATE_WM2_REG.json')}))
