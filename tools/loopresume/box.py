"""S-LOOPRESUME boundary operations, scoped to the existing main arm only."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RELEASE, ARM = ROOT.parent, ROOT.parent/'arm'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/c4')]
import loop14 as loop  # noqa: E402
from wms.control.nativeinit import verify_collection_gate  # noqa: E402


def archive(path):
    if path.exists():
        target = path.with_name(path.name+'.before_loopresume')
        if target.exists():
            raise RuntimeError('boundary backup already exists: '+str(target))
        shutil.copy2(path, target)


def register():
    amendment = loop.read_json(ROOT/'tools/loopresume/AMENDMENT_C1.json')
    cfg = loop.load_config(ARM/'CONFIG.json')
    state = loop.load_state(cfg)
    assert state['sim_s_charged'] == 102607 and state['stopped'] == 'native smoke inconclusive'
    assert state['policy']['sha256'] == amendment['cycle0']['policy_sha256']
    reg = loop.read_json(cfg['registration'])
    target = RELEASE/'AMENDMENT_C1.json'
    if target.exists():
        assert loop.read_json(target) == amendment
    else:
        archive(Path(cfg['registration']))
        archive(loop.state_path(cfg))
        archive(ARM/'CONFIG.json')
        archive(Path(cfg['loop_root'])/'worlds_c00.json')
        loop.write_atomic(target, amendment)
        reg.setdefault('amendments', []).append({**amendment, 'registered_at': loop.now(),
                                               'artifact_sha256': loop.sha256(target)})
        loop.write_atomic(Path(cfg['registration']), reg)
        state['registration_sha256'] = loop.sha256(cfg['registration'])
        loop.save_state(cfg, state)
    print(json.dumps({'registered_before_new_data': True, 'amendment_sha256': loop.sha256(target),
                      'charged': state['sim_s_charged']}))


def tests():
    from tools.restart.box import ROOT as same_root
    assert same_root == ROOT and (RELEASE/'AMENDMENT_C1.json').exists()
    names = ['resumeprep', 'nativeinit', 'longcritic', 'longcritic_followup', 'loop14', 'fastloop',
             'main14', 'coopw', 'policy14_critic', 'sumocredit', 'sumocredit_kcheck', 'c4_dreamer',
             'loop14c', 'policy14_integration', 'policy14e', 'policy14_lanes', 'restart', 'loopresume']
    attempt = 1
    while (RELEASE/f'loopresume_tests_{attempt}.log').exists():
        attempt += 1
    logfile = RELEASE/f'loopresume_tests_{attempt}.log'
    env = {**os.environ, 'PYTHONPATH': str(ROOT/'src')+':'+str(ROOT), 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
           'NATIVEINIT_BASELINE': str(RELEASE/'nativeinit_baseline')}
    env.pop('C4_ALLOWED_ROOTS', None)
    argv = [sys.executable, '-B', '-m', 'pytest', '-q', '-ra', '-p', 'no:cacheprovider',
            '--basetemp', str(RELEASE/f'pytest_loopresume_{attempt}'),
            *['tests/unit/test_'+n+'.py' for n in names]]
    with logfile.open('x') as log:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            pairtemp = RELEASE/f'pytest_loopresume_paircrit_{attempt}'
            pairenv = {**env, 'C4_ALLOWED_ROOTS': str(pairtemp)}
            result = subprocess.run([*argv[:9], str(pairtemp), 'tests/unit/test_paircrit.py'],
                                    cwd=ROOT, env=pairenv, stdout=log, stderr=subprocess.STDOUT)
    print(logfile.read_text())
    proof = {'t': loop.now(), 'exit': result.returncode, 'log': str(logfile),
             'log_sha256': loop.sha256(logfile), 'source_sha256': loop.sha256(RELEASE/'LOOPRESUME_SOURCE.json')}
    loop.write_atomic(RELEASE/f'LOOPRESUME_TESTS_{attempt}.json', proof)
    if result.returncode == 0:
        loop.write_atomic(RELEASE/'LOOPRESUME_TESTS_PASS.json', proof)
    raise SystemExit(result.returncode)


def install():
    proof = loop.read_json(RELEASE/'LOOPRESUME_TESTS_PASS.json')
    assert proof['exit'] == 0 and proof['source_sha256'] == loop.sha256(RELEASE/'LOOPRESUME_SOURCE.json')
    amendment = loop.read_json(RELEASE/'AMENDMENT_C1.json')
    cfg = loop.load_config(ARM/'CONFIG.json')
    state = loop.load_state(cfg)
    assert state['cycles'][-1]['cycle'] == 0 and state['stopped'] == 'native smoke inconclusive'
    assert state['sim_s_charged'] == 102607 and state['policy']['sha256'] == amendment['cycle0']['policy_sha256']
    gate = verify_collection_gate(state['policy'])
    assert gate['passed'] and gate['wm_sha256'] == [state['accepted_wm']['sha256']]
    journal = loop.read_json(Path(cfg['loop_root'])/'worlds_c00.json')
    assert journal['native_smoke']['verdict'] == 'inconclusive' and len(journal['worlds']) == 4
    collected = amendment['cycle0']['collected_worlds']
    assert sorted(map(int, journal['worlds'])) == collected
    worlds = [journal['worlds'][str(s)] for s in collected]
    assert all(loop.fast_module().artifact_valid(w['artifact']) for w in worlds)
    # The original journal also carries prior-generation cycle-0 attempt IDs.
    assert all(journal['attempts'].get(str(s)) == 1 for s in collected)
    assert not set(map(int, journal['attempts'])) & set(amendment['cycle0']['reserved_unconsumed_worlds'])
    for name in ('WAIT_OWNER.json', 'PID.json', 'driver.log'):
        path = ARM/name
        if name == 'PID.json' and path.exists():
            pid = loop.read_json(path)
            assert loop.fast_module().process_token(pid['pid']) != pid['start_ticks']
        if path.exists():
            target = path.with_name(path.name+'.cycle0_scientific_stop')
            assert not target.exists()
            path.rename(target)
    ph = state['cycles'][0]['phases']['collect']
    historical = copy.deepcopy(ph)
    receipt = {'journal': str(Path(cfg['loop_root'])/'worlds_c00.json'), 'worlds': worlds,
               'seeds': collected, 'sim_s_charged': journal['charged'], 'branch_collisions': 0,
               'branch_teleports': 0, 'collection_closed_as_is': True,
               'original_guard': journal['native_smoke'],
               'reserved_unconsumed_worlds': amendment['cycle0']['reserved_unconsumed_worlds']}
    ph.update(status='done', finished=loop.now(), receipt=receipt, closed_by='S-LOOPRESUME',
              original_scientific_stop=historical)
    cfg['policy'].update(config='configs/c4_policy14_loopresume.toml', paircrit=True,
                         longcritic=False, breakdown=False, updates_per_cycle=150)
    cfg['policy']['extra_args'].append('--phase-relative-signal-gate')
    cfg['collect']['native_smoke_harmful_only_from_cycle'] = 1
    cfg['fastloop']['overlap_policy'] = True
    cfg['loopresume'] = {'amendment': str(RELEASE/'AMENDMENT_C1.json'), 'monitored_paircrit': True,
                         'measured_update_selection': True, 'cycle0_closed': True}
    reg = loop.read_json(cfg['registration'])
    reg['budget']['policy_updates_per_cycle'] = 150
    reg['gates']['future_native_smoke'] = amendment['future_guard']
    reg['sources']['loopresume_source_manifest_sha256'] = loop.sha256(RELEASE/'LOOPRESUME_SOURCE.json')
    reg['sources']['fastloop_options']['overlap_policy'] = True
    reg['sources']['config_sha256'] = hashlib.sha256((json.dumps(cfg, indent=1)+'\n').encode()).hexdigest()
    loop.write_atomic(Path(cfg['registration']), reg)
    state.update(stopped=None, registration_sha256=loop.sha256(cfg['registration']))
    loop.write_atomic(ARM/'CONFIG.json', cfg)
    loop.save_state(cfg, state)
    loop.append(Path(cfg['loop_root'])/'cycle_ledger.jsonl', {'event': 'S-LOOPRESUME_boundary', 't': loop.now(),
                'original_stop': historical, 'receipt': receipt, 'charged_unchanged': 102607})
    binding = {'t': loop.now(), 'config_sha256': loop.sha256(ARM/'CONFIG.json'),
               'registration_sha256': state['registration_sha256'],
               'amendment_sha256': loop.sha256(RELEASE/'AMENDMENT_C1.json'),
               'source_sha256': proof['source_sha256'], 'tests': proof,
               'original_mini_pass_sha256': loop.sha256(ARM/'MINI_PASS.json'),
               'native_gate_passed': True, 'charged': state['sim_s_charged'],
               'next_phase': loop._phase(state)['name'], 'overlap_policy': True}
    loop.write_atomic(RELEASE/'LOOPRESUME_BINDINGS.json', binding)
    print(json.dumps(binding))


def launch():
    binding = loop.read_json(RELEASE/'LOOPRESUME_BINDINGS.json')
    assert binding['config_sha256'] == loop.sha256(ARM/'CONFIG.json')
    assert binding['registration_sha256'] == loop.sha256(ARM/'REGISTRATION.json')
    assert binding['source_sha256'] == loop.sha256(RELEASE/'LOOPRESUME_SOURCE.json')
    assert not (ARM/'WAIT_OWNER.json').exists() and not (ARM/'PID.json').exists()
    with (ARM/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (ARM/'driver.log').open('xb') as log:
            proc = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__)), 'drive'],
                                    start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        pid = {'pid': proc.pid, 'start_ticks': loop.fast_module().process_token(proc.pid),
               't': loop.now(), 'task': 'S-LOOPRESUME'}
        loop.write_atomic(ARM/'PID.json', pid)
        loop.write_atomic(RELEASE/'LOOPRESUME_LAUNCH.json', {**pid, 'bindings': binding})
    print(json.dumps(pid))


def drive():
    import main14_autolaunch as waiter
    import loop14_resume
    cfg = loop.load_config(ARM/'CONFIG.json')
    with (ARM/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            waiter.drive(cfg, ARM)
        except Exception as error:
            name = 'WAIT_OWNER.json' if loop14_resume.scientific(error) else 'OPERATIONAL_PAUSED.json'
            if isinstance(error, loop14_resume.OperationalPaused):
                name = 'OPERATIONAL_PAUSED.json'
            loop.write_atomic(ARM/name, {'t': loop.now(), 'error': repr(error), 'task': 'S-LOOPRESUME'})
            raise


def status():
    cfg = loop.load_config(ARM/'CONFIG.json')
    state = loop.load_state(cfg)
    out = {'t': loop.now(), 'charged': state['sim_s_charged'], 'stopped': state['stopped'],
           'accepted_wm': state['accepted_wm'], 'cycles': [{
               'cycle': c['cycle'], 'phases': {k: {'status': v['status'], 'error': v.get('error'),
                    'refused': v.get('refused'), 'receipt': v.get('receipt') if k in ('wm_ft', 'check') else None}
                    for k, v in c['phases'].items()}} for c in state['cycles']]}
    if (ARM/'PID.json').exists():
        out['driver'] = pid = loop.read_json(ARM/'PID.json')
        out['driver_live'] = loop.fast_module().process_token(pid['pid']) == pid['start_ticks']
    for name in ('WAIT_OWNER.json', 'OPERATIONAL_PAUSED.json'):
        if (ARM/name).exists():
            out[name] = loop.read_json(ARM/name)
    for phase in ('policy', 'wm'):
        directory = Path(cfg['loop_root'])/phase/('c01' if phase == 'policy' else 'c00')
        path = directory/'train_log.jsonl'
        if path.exists():
            rows = path.read_text().splitlines()
            out[phase+'_progress'] = {'rows': len(rows), 'last': json.loads(rows[-1]) if rows else None}
        for name in ('gate_panel_result.json', 'native_gate_result.json', 'RUN_SUMMARY.json'):
            if (directory/name).exists():
                value = loop.read_json(directory/name)
                if name == 'RUN_SUMMARY.json':
                    value = {k: value.get(k) for k in ('status', 'error', 'critic_warmup', 'paircrit_monitor',
                             'recommended_updates_per_cycle', 'resumed_from')}
                out[phase+'_'+name] = value
    print(json.dumps(out))
    loop.write_atomic(RELEASE/'LOOPRESUME_STATUS.json', out)
    return out


def wait_gates():
    deadline = time.monotonic()+900
    while time.monotonic() < deadline:
        out = status()
        gate = out.get('policy_gate_panel_result.json')
        if gate or out.get('WAIT_OWNER.json') or out.get('OPERATIONAL_PAUSED.json') or not out.get('driver_live', True):
            return
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'tests', 'install', 'launch', 'drive', 'status', 'wait-gates'])
    args = parser.parse_args()
    globals()[args.action.replace('-', '_')]()


if __name__ == '__main__':
    main()
