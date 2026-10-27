"""Detached registered S-AUTONOMY coordinator; all control stays on its GPU host."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

CAP = 280000
R = A = ROOT = loop = None
OFF = False


def status(value, **extra):
    doc = {'t': loop.now(), 'status': value, 'pid': os.getpid(), 'cap': CAP, **extra}
    # Raw state read is intentional while the predecessor owns phase state.
    p = A / 'loop/LOOP14_STATE.json'
    if p.exists():
        state = loop.read_json(p)
        doc.update(cycle=state['cycles'][-1]['cycle'], charged=state['sim_s_charged'],
                   remaining=CAP - state['sim_s_charged'],
                   phases={k: v['status'] for k, v in state['cycles'][-1]['phases'].items()})
    loop.write_atomic(R / 'AUTONOMY_STATUS.json', doc)
    loop.append(R / 'AUTONOMY_EVENTS.jsonl', doc)
    print(json.dumps(doc), flush=True)
    return doc


def bind(cfg, state, amendment):
    reg = loop.check_registration(cfg)
    reg.setdefault('amendments', []).append(amendment)
    loop.write_atomic(A / 'CONFIG.json', cfg)
    reg.setdefault('sources', {})['config_sha256'] = loop.sha256(A / 'CONFIG.json')
    loop.write_atomic(Path(cfg['registration']), reg)
    state['registration_sha256'] = loop.sha256(cfg['registration'])
    loop.save_state(cfg, state)
    loop.append(A / 'loop/cycle_ledger.jsonl', {'event': 'AUTONOMY_registered', **amendment})


def install_budget(markdown_sha):
    cfg = loop.load_config(A / 'CONFIG.json')
    state = loop.load_state(cfg)
    reg = loop.check_registration(cfg)
    before = {'config': cfg, 'registration': reg, 'state': state}
    dest = R / 'AUTONOMY_BEFORE.json'
    assert not dest.exists(), 'budget already installed; inspect existing receipt'
    loop.write_atomic(dest, copy.deepcopy(before))
    reg['budget']['sim_s_total'] = CAP
    # Authority is repeat-until-cap; remove the legacy four-cycle administrative stop.
    reg['budget']['max_cycles'] = max(reg['budget']['max_cycles'],
        len(state['cycles']) + (CAP - state['sim_s_charged']) // loop.collect_sim_s(cfg))
    for k in range(2, reg['budget']['max_cycles']):
        reg['budget'].setdefault('cycle_attempt_allowances', {})[str(k)] = 2 * loop.collect_sim_s(cfg)
        reg.setdefault('collection', {}).setdefault('cycle_long_attempt_allowances', {})[str(k)] = 2 * loop.long_group_sim_s(cfg)
    approval = {'id': 'S-AUTONOMY_OWNER_280000', 't': loop.now(), 'owner_approved': True,
        'authority': 'main 2026-10-27 13:15 S-AUTONOMY', 'sim_s_total': CAP,
        'collection_authorized': not OFF, 'markdown_registration_sha256': markdown_sha,
        'repeat_until_next_collection_exceeds_cap': True, 'operational_retries_per_phase': 1,
        'scientific_failure': 'WAIT_OWNER', 'horizon_s': 60,
        'microbatch_starts': 1, 'effective_batch_starts': 1, 'accumulation_steps': 1,
        'memory_supplement': 'future main only: lossless saved-tensor CPU offload; starts already1 cannot be split',
        'main_existing_policy_untouched': True}
    reg.setdefault('amendments', []).append(approval)
    # Config file is unchanged beneath the active main actor. Only budget/allowance + SHA change.
    loop.write_atomic(Path(cfg['registration']), reg)
    state['registration_sha256'] = loop.sha256(cfg['registration'])
    if OFF:
        pause = {'t': loop.now(), 'status': 'PAUSED_BY_OWNER_PRIORITY_RESUMABLE',
                 'reason': 'paused by owner priority, resumable; gpuR reserved for long-run WM work',
                 'waiter_armed': False, 'policy_launch_authorized': False, 'budget_cap': CAP,
                 'authority': 'main owner-priority13:25 amendment',
                 'retained': ['state', 'checkpoints', 'cycle1 data', 'all charges and failures']}
        state['owner_priority_pause'] = pause
        state['stopped'] = 'PAUSED_BY_OWNER_PRIORITY_RESUMABLE'
        loop.write_atomic(A / 'OWNER_PRIORITY_PAUSE.json', pause)
        loop.write_atomic(R / 'AUTONOMY_STATUS.json', pause)
    loop.save_state(cfg, state)
    loop.write_atomic(R / 'AUTONOMY_APPROVAL.json', approval)
    loop.append(A / 'loop/cycle_ledger.jsonl', {'event': 'AUTONOMY_owner_budget_installed', **approval})
    print(json.dumps({'installed': True, 'cap': CAP, 'charged': state['sim_s_charged'],
                      'registration_sha256': state['registration_sha256']}))


def live(pid):
    return loop.fast_module().process_token(pid['pid']) == str(pid['start_ticks'])


def done_file():
    control = R / 'AUTONOMY_WMDEEP_DONE_PATH.json'
    if control.exists():
        candidate = Path(loop.read_json(control)['path'])
        loop.guard(candidate)
        if candidate.name == 'DONE.json' and candidate.is_file():
            return candidate
    # Discover only specifically named WMDEEP releases and exact completion filenames.
    # Never descend into datasets, score partitions or any prohibited path.
    for base in (R, R.parent):
        for child in base.iterdir():
            if 'wmdeep' not in child.name.lower() or not child.is_dir():
                continue
            loop.guard(child)
            candidate = child / 'DONE.json'
            if candidate.is_file():
                return candidate
    return None


def wait_predecessor():
    if OFF:
        while True:
            done = done_file()
            if done:
                # The marker is written by the WM-test coordinator, never by this waiter.
                status('WMDEEP_DONE_OBSERVED', done=str(done), done_sha256=loop.sha256(done))
                return
            status('WAIT_WMDEEP_DONE', no_gpu_child_started=True)
            time.sleep(30)
    else:
        predecessor = loop.read_json(R / 'C2H60B_LAUNCH.json')
        while live(predecessor):
            status('WAIT_MAIN_C2H60B', predecessor=predecessor)
            time.sleep(30)


def priority():
    if not OFF:
        return
    source = "import json,pathlib; p=pathlib.Path('/root/autodl-tmp/c4/restart20261026/arm/loop/LOOP14_STATE.json');s=json.loads(p.read_text());print(json.dumps(any(p.get('status')=='running' for c in s['cycles'] for n,p in c['phases'].items() if n in ('collect','check'))))"
    while True:
        p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'gpuA',
            '/root/miniconda3/bin/python -B -c ' + shlex.quote(source)], capture_output=True, text=True, timeout=30)
        if p.returncode == 0 and json.loads(p.stdout) is False:
            return
        status('WAIT_MAIN_CPU_PRIORITY')
        time.sleep(30)


def preflight(cfg, state, cases):
    result, boxes = {}, []
    for box in ('cpuP', 'cpuQ'):
        h = loop.host(cfg, box)
        base = str(Path(cfg['collect']['work_root']).parent / 'autonomy')
        try:
            # Explicit files only. run_collect will stage its own full immutable WM bundle.
            for src, name in ((state['policy']['checkpoint'], 'preflight_policy.pt'),
                              (Path(state['policy']['checkpoint']).with_name('native_gate_result.json'),
                               'native_gate_result.json'),
                              (R / 'autonomy/cpu_preflight.py', 'cpu_preflight.py')):
                loop.sync(cfg, 'gpu', str(src), box, base + '/' + name,
                          R / 'autonomy/preflight_transfer.log')
            command = [h['python'], base + '/cpu_preflight.py', h['code_root'],
                       base + '/preflight_policy.pt', state['policy']['sha256'], json.dumps(cases)]
            exports = {**cfg['collect'].get('env', {}), 'PYTHONDONTWRITEBYTECODE': '1'}
            remote = 'env ' + ' '.join(k + '=' + shlex.quote(str(v)) for k, v in exports.items())
            remote += ' ' + shlex.join(command)
            p = subprocess.run([*h['ssh'], remote], capture_output=True, text=True, timeout=180)
            assert p.returncode == 0, p.stderr[-2000:] + p.stdout[-2000:]
            proof = json.loads(p.stdout.splitlines()[-1])
            assert proof['status'] == 'PASS' and proof['charged_sim_s'] == 0
            result[box] = proof
            boxes.append(box)
        except Exception as error:
            result[box] = {'status': 'UNAVAILABLE', 'error': repr(error), 'charged_sim_s': 0}
    loop.write_atomic(R / 'AUTONOMY_PREFLIGHT.json', result)
    status('PREFLIGHT_COMPLETE', available_boxes=boxes, proof=result)
    return boxes


def gpu_guard(cfg, h):
    # Aggregate resources only; never enumerate or inspect other jobs' PIDs.
    total, free = map(float, subprocess.check_output(['nvidia-smi',
        '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(','))
    # Serial phases + offload. Respect the registered 2 GiB headroom and wait on busy GPU.
    need = min(total * float(cfg['policy']['env']['C4_GPU_MEM_FRACTION']), total - 2048)
    if free < need + 1024:
        raise loop.GpuBusy(f'aggregate GPU free {free} < serial reservation {need}+1024 MiB')
    return {'total_mib': total, 'free_mib': free, 'serial': True}


def scientific(text):
    text = str(text).lower()
    return any(x in text for x in ('nonfinite', 'nan', 'native-relative', 'native identity',
        'early signal', 'native smoke', 'harmful', 'safety', 'incident', 'provenance',
        'sha mismatch', 'hash mismatch', 'gate failed', 'reject', 'no trained checkpoint',
        'native acceptance', 'no accepted trained checkpoint', 'not finite', 'did not decrease'))


def run_command(host, argv, *, env, log, timeout_s, cwd=None):
    # All local children inherit no terminal and have their own detached session.
    if host.get('ssh'):
        return ORIGINAL_RUN(host, argv, env=env, log=log, timeout_s=timeout_s, cwd=cwd)
    argv = list(map(str, argv))
    if 'wms.control.policy14_cli' in argv:
        i = argv.index('-m')
        argv[i:i + 2] = [str(R / 'autonomy/memory_entry.py')]
        env = {**env, 'AUTONOMY_RUNTIME': str(ROOT)}
        timeout_s = None  # registered time waiver; 60s never selected by runtime
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('a') as stream:
        proc = subprocess.Popen(argv, cwd=cwd or host['code_root'],
            env={**os.environ, **{k: str(v) for k, v in env.items()}},
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        child = {'pid': proc.pid, 'start_ticks': loop.fast_module().process_token(proc.pid),
                 'sid': os.getsid(proc.pid), 'argv': argv, 'log': str(log)}
        loop.append(R / 'AUTONOMY_CHILDREN.jsonl', child)
        status('CHILD_RUNNING', child=child)
        started = time.monotonic()
        while proc.poll() is None:
            # Heartbeat persists even during hours-long phases. Never kills a child by name.
            time.sleep(20)
            status('CHILD_RUNNING', child=child)
            if timeout_s is not None and time.monotonic() - started > timeout_s:
                assert live(child), 'own child incarnation changed before timeout'
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    assert live(child), 'own child incarnation changed before bounded stop'
                    proc.kill()
                    proc.wait()
                return 124
        return proc.returncode


def configure(cfg, state):
    approval = loop.read_json(R / 'AUTONOMY_APPROVAL.json')
    if not OFF:
        # The existing main actor deliberately stops before simulation. Its gate is reused only
        # after the completed phase and exact writer exit, never restarted/replayed here.
        assert state['cycles'][2]['phases']['policy']['status'] == 'done', 'main C2 policy did not complete'
        from wms.control.nativeinit import verify_collection_gate
        verify_collection_gate(state['policy'])
        assert state.get('stopped') in (None, 'WAIT_OWNER_BUDGET'), state.get('stopped')
        state['stopped'] = None
    config = ROOT / 'configs/c4_policy14_autonomy.toml'
    old_config = Path(cfg['policy']['config'])
    if not old_config.is_absolute():
        old_config = ROOT / old_config
    text = 'extends = ' + json.dumps(str(old_config)) + '\n[run]\n'
    text += 'operational_recovery = ' + json.dumps(str(R / 'AUTONOMY_C2_RECOVERY.json') if OFF else '') + '\n'
    config.write_text(text)
    cfg['policy']['config'] = str(config)
    cfg['resumeprep']['operational_retries_per_error'] = 1
    # No stale overlap is launched: this coordinator executes canonical phases serially.
    cfg['autonomy'] = {'enabled': True, 'cap': CAP, 'horizon_s': 60,
                       'serial_phases': True, 'operational_retries_per_phase': 1}
    if OFF:
        recovery = loop.read_json(R / 'C2H60_RECOVERY.json')
        recovery.update(config=str(config), config_sha256=loop.sha256(config),
                        retry_outputs=[str(A / 'loop/policy/c02.a3'), str(A / 'loop/policy/c02.a4')])
        loop.write_atomic(R / 'AUTONOMY_C2_RECOVERY.json', recovery)
        ph = state['cycles'][2]['phases']['policy']
        assert ph['status'] == 'failed' and ph['attempt'] == 2
        state['cycles'][2].setdefault('autonomy_prior_policy_attempts', []).append(copy.deepcopy(ph))
        ph.update(status='pending', renewed_authority='S-AUTONOMY', previous_error=ph.get('error'))
    amendment = {**approval, 'id': 'S-AUTONOMY_REMOTE_CHAIN', 't': loop.now(),
                 'config': str(config), 'config_sha256': loop.sha256(config)}
    bind(cfg, state, amendment)
    if not OFF:
        # Exact legacy hook is installed only after the main predecessor has exited.
        bind(cfg, state, {**approval, 'id': 'S-C2H60_OWNER_BUDGET_RAISE', 't': loop.now()})
    return cfg


def horizon(cfg, state):
    import main14_autolaunch as waiter
    status('HORIZON_CHECK_RUNNING', wm=state['accepted_wm'])
    target = A / f"horizon_{state['accepted_wm']['sha256']}.json"
    if target.exists() and loop.read_json(target).get('measurement_error'):
        # Preserve failed measurement, then allow the one registered operational retry.
        archive = R / 'autonomy' / (target.stem + '.measurement_error.json')
        if not archive.exists():
            target.rename(archive)
    def run(argv, config, arm_dir, label, timeout=7200, env=None):
        code = run_command(loop.host(config, 'gpu'), argv,
            env=env or waiter.environment(config, arm_dir), log=arm_dir / (label + '.log'),
            timeout_s=timeout, cwd=ROOT)
        if code:
            raise RuntimeError(f'horizon measurement exit{code}')
    waiter.run = run
    waiter.prepare_horizon(cfg, state, A)
    gate = loop.read_json(A / 'horizon_gate_current.json')
    assert gate['wm_sha256'] == state['accepted_wm']['sha256'], 'horizon provenance mismatch'
    if gate.get('measurement_error'):
        raise RuntimeError('horizon measurement error: ' + gate['measurement_error'])
    if gate.get('horizons', {}).get('60', {}).get('verdict') != 'PASS':
        raise loop.Refused('60s scientific horizon gate failed')


def drive():
    if OFF:
        raise RuntimeError('OWNER_PRIORITY_PAUSE: no coop-off waiter or relaunch is authorized')
    status('ARMED', arm='coopoff' if OFF else 'main')
    wait_predecessor()
    # Separate own lock plus both historical arm locks; no overlapping state writers.
    from contextlib import ExitStack
    with ExitStack() as stack:
        for name in ('autonomy.lock', 'driver.lock', 'cycle1.lock'):
            lock = stack.enter_context((A / name).open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = loop.load_config(A / 'CONFIG.json')
        state = loop.load_state(cfg)
        cfg = configure(cfg, state)
        loop.gpu_guard = gpu_guard
        loop.run_cmd = run_command
        from collector import collect
        fast = loop.fast_module()
        fast.collect = lambda l, c, s, p, r: collect(l, c, s, p, r, preflight, status, priority)
        # loop14's factory loads a new module on every call; preserve this adapter
        # for actual run_collect dispatch, conversion, and artifact verification.
        loop.fast_module = lambda: fast
        retries = loop.read_json(R / 'AUTONOMY_RETRIES.json') if (R / 'AUTONOMY_RETRIES.json').exists() else {}
        while True:
            cfg = loop.load_config(A / 'CONFIG.json')
            state = loop.load_state(cfg)
            if (A / 'STOP_ARM').exists() or (A / 'loop/STOP_DRIVER').exists():
                status('WAIT_OWNER', reason='owned arm cooperative STOP marker')
                return
            if state.get('stopped'):
                status('WAIT_OWNER', reason=state['stopped'])
                return
            name = loop._phase(state)['name']
            if name is None:
                status('WAIT_OWNER', reason='completed final budget-funded cycle')
                return
            k = state['cycles'][-1]['cycle']
            if name in ('policy', 'collect') and state['sim_s_charged'] + loop.collect_sim_s(cfg) > CAP:
                # Partial-collection retry is separately budgeted in collector.collect.
                if name == 'policy' or not (A / f'loop/worlds_c{k:02d}.json').exists():
                    state['stopped'] = 'WAIT_OWNER_BUDGET: next collection would exceed280000'
                    loop.save_state(cfg, state)
                    status('WAIT_OWNER', reason=state['stopped'])
                    return
            key = f'{k}:{name}'
            status(name.upper() + '_RUNNING', retry_used=retries.get(key, 0))
            if name in ('collect', 'check'):
                priority()
            try:
                if name == 'policy':
                    horizon(cfg, state)
                code = loop.run_phase(cfg, name)
            except loop.GpuBusy:
                code = 5
            except Exception as error:
                code, detail = 3, repr(error)
            else:
                detail = ''
            if code == 5:
                status('WAIT_GPU_RESOURCE', phase=name)
                time.sleep(30)
                continue
            if code == 0:
                status(name.upper() + '_COMPLETE')
                continue
            state = loop.load_state(cfg)
            ph = state['cycles'][-1]['phases'].get(name, {})
            detail = detail or ph.get('error') or ph.get('refused') or state.get('stopped') or f'exit{code}'
            if code == 4 or scientific(detail) or state.get('stopped'):
                if not state.get('stopped'):
                    state['stopped'] = 'WAIT_OWNER_SCIENTIFIC: ' + ('WM acceptance REJECT' if code == 4 else detail)
                    loop.save_state(cfg, state)
                status('WAIT_OWNER', phase=name, reason=detail, scientific_stop=True)
                return
            if any(t in detail.lower() for t in ('budget', 'allowance', 'cannot fund', 'exhausted')):
                status('WAIT_OWNER', phase=name, reason=detail)
                return
            if retries.get(key, 0) >= 1:
                status('OPERATIONAL_PAUSED', phase=name, reason=detail, retries_exhausted=True)
                return
            retries[key] = 1
            loop.write_atomic(R / 'AUTONOMY_RETRIES.json', retries)
            # The repaired adapters/preflight re-resolve available peers and retain artifacts.
            # Never clear a scientific refusal. Fresh attempt directories preserve failed output.
            if ph.get('status') == 'failed':
                ph.update(status='pending', previous_error=ph.get('error'))
                loop.save_state(cfg, state)
            status('OPERATIONAL_RETRY_ARMED', phase=name, reason=detail, retry=1)
            time.sleep(30)


def main():
    global R, A, ROOT, OFF, loop, ORIGINAL_RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['budget', 'drive'])
    parser.add_argument('--release', required=True)
    parser.add_argument('--markdown-sha')
    args = parser.parse_args()
    R = Path(args.release)
    A = R / 'arm'
    cfg = json.loads((A / 'CONFIG.json').read_text())
    ROOT = Path(cfg['hosts']['gpu']['code_root'])
    OFF = 'coopoff' in R.name
    sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts/c4'), str(R / 'autonomy')]
    os.chdir(ROOT)
    import loop14
    loop = loop14
    ORIGINAL_RUN = loop.run_cmd
    if args.action == 'budget':
        install_budget(args.markdown_sha)
    else:
        try:
            drive()
        except Exception as error:
            status('WAIT_OWNER' if scientific(error) else 'OPERATIONAL_PAUSED', reason=repr(error))
            raise


if __name__ == '__main__':
    main()
