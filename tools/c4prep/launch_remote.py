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
resolved = out / ('resolved_A_MAIN1500_' + pointer['commit'][:12])


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
elif action == 'fit-fallback':
    from datetime import datetime, timezone
    import signal
    coordinator = read(fit / 'COORDINATOR_PID.json')
    child = read(fit / 'REPLAY_PID.json')
    assert owned_alive(coordinator) and owned_alive(child)
    proof = read(fit / 'INPUTS.json')
    assert [c['cycle'] for c in proof['cycles']] == [0, 1, 2]
    rows = [read(p) for p in (fit / 'rows').glob('c[012]_*_h*.json')]
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(child['t'])).total_seconds()
    simulated = sum(row['horizon_s'] for row in rows)
    total = sum(c['pairs'] for c in proof['cycles']) * 160
    projected = elapsed * total / simulated
    assert len(rows) >= 3 and projected > 3600
    decision = {'native_anchor': False, 'pessimistic_credit': True, 'quantile': .1, 'k': 8,
        'cycles': [0, 1, 2], 'fitted_scale': None, 'completed_rows': len(rows),
        'elapsed_s': elapsed, 'projected_s': projected, 'budget_s': 3600,
        'reason': 'measured genuine K8 pilot projects beyond authorized 3600s fit budget',
        'owned_replay_pid': child['pid'], 'owned_start_ticks': child['start_ticks']}
    write_new(fit / 'FIT_DECISION.json', decision)
    # Only the live, receipt-bound owned replay child is signalled.
    assert owned_alive(child)
    os.kill(child['pid'], signal.SIGTERM)
    print(json.dumps(decision))
elif action == 'prepare':
    from launch_a import prepare
    prepare()
elif action == 'checks':
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q',
        '--basetemp', str(out / ('launch_checks_' + pointer['commit'][:12])),
        'tests/unit/test_c4launch.py', 'tests/unit/test_c4prep.py', 'tests/unit/test_antiexploit.py'],
        cwd=code, env=env, timeout=35)
    raise SystemExit(result.returncode)
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
