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
        'tests/unit/test_c4launch.py', 'tests/unit/test_c4prep.py', 'tests/unit/test_antiexploit.py',
        'tests/unit/test_longmem.py'], cwd=code, env=env, timeout=35,
        capture_output=True, text=True)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    write_new(out / ('CHECKS_' + pointer['commit'][:12] + '.json'),
        {'source_commit': pointer['commit'], 'exit': result.returncode,
         'stdout': result.stdout, 'stderr': result.stderr, 'cuda_hidden': True})
    if result.returncode == 0:
        gpu = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q',
            'tests/unit/test_longmem.py::test_saved_spill_shares_parameter_views_without_changing_gradients'],
            cwd=code, env=os.environ, timeout=15, capture_output=True, text=True)
        print(gpu.stdout, end='')
        print(gpu.stderr, end='', file=sys.stderr)
        write_new(out / ('GPU_COPY_CHECK_' + pointer['commit'][:12] + '.json'),
            {'source_commit': pointer['commit'], 'exit': gpu.returncode,
             'stdout': gpu.stdout, 'stderr': gpu.stderr})
        if gpu.returncode:
            raise SystemExit(gpu.returncode)
    raise SystemExit(result.returncode)
elif action == 'dry':
    print(json.dumps(spawn([sys.executable, '-B', str(code / 'tools/c4prep/launch.py'),
        'dry-run', '--config', str(resolved / 'CONFIG.json')], resolved, 'DRY_COORDINATOR')))
elif action == 'diagnose':
    result = {'t': now(), 'meminfo': Path('/proc/meminfo').read_text().splitlines()[:5]}
    result['own_cgroup'] = Path('/proc/self/cgroup').read_text()
    for name in ('memory.max', 'memory.current', 'memory.events'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            result[name] = path.read_text()
    for name in ('DRY_COORDINATOR.log', 'DRY_RUN/A.log'):
        path = resolved / name
        if path.exists():
            result[name] = path.read_text()[-7000:]
    root = resolved / 'DRY_RUN/A'
    result['owned_output_files'] = [{'name': p.name, 'bytes': p.stat().st_size}
        for p in root.iterdir() if p.is_file()]
    print(json.dumps(result))
elif action == 'start':
    from launch import start, validate
    config = resolved / 'CONFIG.json'
    doc = validate(config, launching=True)
    decision = {'task': 'S-C4LAUNCH', 'authority': 'MAIN2026-10-28T15:00',
        'candidate': 'A', 'native_anchor': doc['native_anchor'], 'pessimistic_credit': True,
        'quantile': .1, 'k': 8, 'trusted_horizon_s': 90, 'entrant_control': True,
        'config': str(config), 'config_sha256': sha(config),
        'dry_run_receipt_sha256': sha(resolved / 'DRY_RUN_VERIFIED.json'),
        'fit_decision_sha256': doc['fit_decision_sha256'], 'new_sim_s': 0}
    write_new(resolved / 'PRELAUNCH_DECISION.json', decision)
    state_path = r / 'arm/loop/LOOP14_STATE.json'
    state = read(state_path)
    state.setdefault('main_launch_execution_decisions', []).append(decision)
    temp = state_path.with_suffix('.c4launch.tmp')
    with temp.open('x') as stream:
        json.dump(state, stream, indent=1)
    temp.replace(state_path)
    start(resolved / 'CONFIG.json')
elif action in ('status', 'verify'):
    from launch_a import status
    result = status(verify=action == 'verify')
    for mode in ('DRY_RUN', 'POLICY'):
        root = resolved / mode
        memory = root / 'gpu_memory.jsonl'
        item = result[mode]
        if not item.get('error'):
            item.pop('log_tail', None)
        if item.get('alive'):
            stat = Path(f"/proc/{item['pid']}/stat").read_text().rsplit(')', 1)[1].split()
            item['owned_cpu_seconds'] = (int(stat[11]) + int(stat[12])) / os.sysconf('SC_CLK_TCK')
            item['owned_rss_bytes'] = int(stat[21]) * os.sysconf('SC_PAGE_SIZE')
        if memory.exists():
            samples = [json.loads(line) for line in memory.read_text().splitlines()]
            if samples:
                result[mode]['sampled_memory'] = {'samples': len(samples), 'last': samples[-1],
                    'peak_bytes': max(row['bytes'] for row in samples)}
        row = result[mode].get('last')
        if row:
            result[mode]['last'] = {k: row[k] for k in ('t', 'update', 'actor_loss', 'critic_loss',
                'actor_grad_norm', 'critic_grad_norm', 'actor_module_grad_norms', 'physical_residual_mps2',
                'mean_paired_credit', 'pessimistic_paired_credit', 'real_credit_weights',
                'cuda_peak_reserved_bytes', 'imagination_horizon_s') if k in row}
    if action == 'status':
        for name in ('DRY_COORDINATOR_PID.json', 'MAIN_LAUNCH.json'):
            if name in result:
                result[name] = {k: result[name][k] for k in ('pid', 'start_ticks', 't')}
    print(json.dumps(result))
elif action == 'inspect':
    print(json.dumps({'pointer': pointer, 'authority': read(out / 'MAIN_1500.json'),
        'wm2_gate': read(r / 'arm/loop/longint/gate_wm2/HORIZON_GATE_WM2_REG.json')}))
