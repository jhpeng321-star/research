"""Owner-delegated fallback, separate runtime/state writer beside the owned WM job."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
ROOT = Path(__file__).resolve().parents[2]
RELEASE, ARM = ROOT.parent, ROOT.parent/'arm'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/c4')]
import loop14 as loop  # noqa: E402
import loop14_resume as resume  # noqa: E402
import main14_autolaunch as waiter  # noqa: E402


def live(pid):
    return loop.fast_module().process_token(pid['pid']) == pid['start_ticks']


def event(name, **row):
    loop.append(RELEASE/'C1_EVENTS.jsonl', {'t': loop.now(), 'event': name, **row})


def fallback_inputs(cfg):
    from wms.control.real_returns14 import NPZRealReturnProvider
    from wms.control.real_share import split_provider
    os.environ['C4_ALLOWED_ROOTS'] = ':'.join(cfg['policy']['allowed_roots']+[cfg['loop_root']])
    provider = NPZRealReturnProvider(cfg['policy']['initial_real_returns'])
    fit, dev = split_provider(provider)
    evidence = loop.read_json(ROOT/'tools/longcritic/REAL_SHARE_MEASURED.json')
    assert loop.sha256(provider.path) == evidence['source_sha256']
    assert sorted(set(fit.provenance['seeds'].tolist())) == evidence['fit_worlds']
    assert sorted(set(dev.provenance['seeds'].tolist())) == evidence['development_worlds']
    assert evidence['selected_share'] == 1.0 and evidence['candidates']['1.0']['eligible']
    return {'returns': str(provider.path), 'returns_sha256': loop.sha256(provider.path),
            'native_returns': cfg['policy']['initial_native_returns'],
            'native_returns_sha256': loop.sha256(cfg['policy']['initial_native_returns']),
            'fitting_worlds': evidence['fit_worlds'], 'development_worlds': evidence['development_worlds'],
            'evidence_sha256': loop.sha256(ROOT/'tools/longcritic/REAL_SHARE_MEASURED.json'),
            'fixed_share': 1.0, 'off_policy_bootstrap': True}


def register():
    cfg = loop.load_config(ARM/'CONFIG.json')
    state = loop.load_state(cfg)
    assert state['sim_s_charged'] == 102607 and len(state['cycles']) == 1
    old = RELEASE/'AMENDMENT_C1.json'
    amendment = loop.read_json(ROOT/'tools/loopresume/AMENDMENT_C1.json')
    assert 'owner_delegated_fallback' in amendment
    backup = RELEASE/'AMENDMENT_C1.before_owner_delegated_fallback.json'
    if not backup.exists():
        shutil.copy2(old, backup)
        loop.write_atomic(old, amendment)
        reg = loop.read_json(cfg['registration'])
        reg.setdefault('amendments', []).append({
            'id': 'S-LOOPRESUME-C1-FALLBACK', 'registered_at': loop.now(),
            'authority': amendment['owner_delegated_fallback']['authority'],
            'amendment': amendment['owner_delegated_fallback'], 'artifact_sha256': loop.sha256(old),
            'previous_artifact_sha256': loop.sha256(backup)})
        loop.write_atomic(Path(cfg['registration']), reg)
        state['registration_sha256'] = loop.sha256(cfg['registration'])
        loop.save_state(cfg, state)
    assert loop.read_json(old) == amendment
    print(json.dumps({'registered': True, 'amendment_sha256': loop.sha256(old), 'charged': state['sim_s_charged']}))


def install():
    assert ROOT.name == 'runtime_c1'
    tests = loop.read_json(RELEASE/'LOOPRESUME_C1_TESTS_PASS.json')
    assert tests['exit'] == 0 and tests['source_sha256'] == loop.sha256(RELEASE/'LOOPRESUME_C1_SOURCE.json')
    with (ARM/'cycle1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = loop.load_config(ARM/'CONFIG.json')
        state = loop.load_state(cfg)
        assert len(state['cycles']) == 1 and state['cycles'][0]['phases']['convert']['status'] == 'failed'
        old_pid = loop.read_json(ARM/'PID.json')
        assert not live(old_pid)
        wm_pid = loop.read_json(RELEASE/'LOOPRESUME_WM_PID.json')
        wm = loop.read_json(RELEASE/'LOOPRESUME_WM_ONLY.json')
        assert live(wm_pid) or wm['status'] == 'completed'
        assert wm['status'] in ('wm_ft_running', 'original_acceptance_running', 'completed')
        for name in ('CONFIG.json', 'REGISTRATION.json', 'loop/LOOP14_STATE.json'):
            source = ARM/name
            target = source.with_name(source.name+'.before_owner_delegated_fallback')
            assert not target.exists()
            shutil.copy2(source, target)
        for name in ('PID.json', 'OPERATIONAL_PAUSED.json', 'driver.log'):
            source = ARM/name
            if source.exists():
                target = source.with_name(source.name+'.paired_export_pause')
                assert not target.exists()
                source.rename(target)
        cfg['hosts']['gpu']['code_root'] = str(ROOT)
        cpu_root = '/home/ubuntu/c4/main14/restart20261026/runtime_c1'
        for key in ('cpu', 'cpuP', 'cpuQ'):
            cfg['hosts'][key]['code_root'] = cpu_root
        for value in cfg['mainstart']['collect_hosts'].values():
            value['code_root'] = cpu_root
        p = cfg['policy']
        p.update(config='configs/c4_policy14_c1_fallback.toml', paircrit=False, longcritic=True,
                 breakdown=False, paircrit_deferred_from_cycle=2, updates_per_cycle=150)
        p['allowed_roots'].append(str(ROOT))
        p['extra_args'] = [a.replace(str(RELEASE/'runtime')+'/', str(ROOT)+'/') for a in p['extra_args']]
        cfg['loopresume'].update(owner_delegated_fallback=True, paircrit_min_fitting_worlds=1)
        fallback = fallback_inputs(cfg)
        state['real_returns'] = fallback['returns']
        state['native_real_returns'] = fallback['native_returns']
        converted = copy.deepcopy(wm['converted_outputs'])
        converted['sumocredit'] = wm['confidence_recalibration']
        converted['returns_sha256'] = loop.sha256(converted['returns'])
        converted['native_returns_sha256'] = loop.sha256(converted['native_returns'])
        conv = state['cycles'][0]['phases']['convert']
        history = copy.deepcopy(conv)
        conv.update(status='done', receipt=converted, owner_resolution='paircrit deferred; existing numeric outputs accepted',
                    prior_failure=history, finished=loop.now())
        state['cycles'][0]['phases']['wm_ft'].update(status='running', attempt=1, pid=wm_pid['pid'],
            coordinator_start=wm_pid['start_ticks'], external_receipt=str(RELEASE/'LOOPRESUME_WM_ONLY.json'))
        for key in ('real_credit', 'gate_table', 'gate_mean_table'):
            if converted['sumocredit'].get(key):
                state[key] = converted['sumocredit'][key]
        state['cycles'].append(loop.new_cycle(1))
        state['stopped'] = None
        cfg['resumeprep']['operational_retries_per_error'] = 3
        reg = loop.read_json(cfg['registration'])
        reg['sources']['loopresume_c1_source_sha256'] = tests['source_sha256']
        reg.setdefault('boundary_decisions', []).append({'id': 'S-LOOPRESUME-C1-FALLBACK', 't': loop.now(),
            'cycle': 1, 'paircrit': False, 'fixed_real_share': 1.0, 'fallback_inputs': fallback,
            'wm': state['accepted_wm'], 'wm_config_sha256': loop.sha256(state['accepted_wm']['config']),
            'external_wm': wm_pid, 'overlap': True})
        loop.write_atomic(ARM/'CONFIG.json', cfg)
        reg['sources']['config_sha256'] = loop.sha256(ARM/'CONFIG.json')
        loop.write_atomic(Path(cfg['registration']), reg)
        state['registration_sha256'] = loop.sha256(cfg['registration'])
        loop.save_state(cfg, state)
        binding = {'t': loop.now(), 'config_sha256': loop.sha256(ARM/'CONFIG.json'),
            'registration_sha256': state['registration_sha256'], 'source_sha256': tests['source_sha256'],
            'amendment_sha256': loop.sha256(RELEASE/'AMENDMENT_C1.json'), 'tests': tests,
            'policy': state['policy'], 'training_wm': state['accepted_wm'],
            'training_wm_config_sha256': loop.sha256(state['accepted_wm']['config']),
            'external_wm': wm_pid, 'fallback_inputs': fallback, 'charged': state['sim_s_charged']}
        loop.write_atomic(RELEASE/'C1_BINDINGS.json', binding)
        print(json.dumps(binding))


def launch():
    binding = loop.read_json(RELEASE/'C1_BINDINGS.json')
    assert binding['config_sha256'] == loop.sha256(ARM/'CONFIG.json')
    assert binding['registration_sha256'] == loop.sha256(ARM/'REGISTRATION.json')
    assert binding['source_sha256'] == loop.sha256(RELEASE/'LOOPRESUME_C1_SOURCE.json')
    assert not (ARM/'PID.json').exists()
    with (ARM/'cycle1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (ARM/'driver_c1.log').open('xb') as log:
            proc = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__)), 'drive'],
                start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        pid = {'pid': proc.pid, 'start_ticks': loop.fast_module().process_token(proc.pid), 't': loop.now(),
               'task': 'S-LOOPRESUME-C1-FALLBACK'}
        loop.write_atomic(ARM/'PID.json', pid)
        loop.write_atomic(RELEASE/'C1_LAUNCH.json', {**pid, 'bindings': binding})
    print(json.dumps(pid))


def reconcile_wm(cfg):
    pid = loop.read_json(RELEASE/'LOOPRESUME_WM_PID.json')
    while True:
        doc = loop.read_json(RELEASE/'LOOPRESUME_WM_ONLY.json')
        if doc['status'] == 'completed':
            break
        if doc['status'] == 'OPERATIONAL_PAUSED' or not live(pid):
            raise resume.OperationalPaused('external cycle0 WM/check did not complete: '+str(doc))
        time.sleep(30)
    state = loop.load_state(cfg)
    c0 = state['cycles'][0]
    if c0['phases']['check']['status'] == 'done':
        return
    c0['phases']['wm_ft'].update(status='done', receipt=doc['wm_ft'], finished=doc['finished'])
    c0['phases']['check'].update(status='done', receipt=doc['check'], finished=doc['finished'],
                                external_receipt=str(RELEASE/'LOOPRESUME_WM_ONLY.json'))
    state['accepted_wm'] = doc['accepted_wm']
    state['consecutive_rejects'] = 0 if doc['check']['decision'] == 'ACCEPT' else state['consecutive_rejects']+1
    loop.save_state(cfg, state)
    event('external_wm_reconciled', verdict=doc['check'], policy_wm=state.get('policy_wm'))


def next_boundary(cfg):
    state = loop.load_state(cfg)
    converted = state['cycles'][-1]['phases']['convert']['receipt']
    eligible = converted['paircrit_eligibility']
    enabled = eligible['eligible']
    if enabled:
        assert converted.get('pairing_receipt')
        state.update(real_returns=converted['returns'], native_real_returns=converted['native_returns'],
                     pairing_receipt=converted['pairing_receipt'])
    else:
        state.update(real_returns=cfg['policy']['initial_real_returns'],
                     native_real_returns=cfg['policy']['initial_native_returns'])
    cfg['policy'].update(paircrit=enabled, longcritic=not enabled,
        config='configs/c4_policy14_loopresume.toml' if enabled else 'configs/c4_policy14_c1_fallback.toml')
    reg = loop.read_json(cfg['registration'])
    reg.setdefault('boundary_decisions', []).append({'t': loop.now(), 'cycle': 2,
        'id': 'S-LOOPRESUME-C1-FALLBACK conditional paircrit enablement', 'paircrit': enabled,
        'eligibility': eligible, 'pairing_receipt_sha256': converted.get('pairing_receipt_sha256')})
    loop.write_atomic(ARM/'CONFIG.json', cfg)
    reg['sources']['config_sha256'] = loop.sha256(ARM/'CONFIG.json')
    loop.write_atomic(Path(cfg['registration']), reg)
    state['registration_sha256'] = loop.sha256(cfg['registration'])
    loop.save_state(cfg, state)
    event('cycle2_critic_decision', paircrit=enabled, eligibility=eligible)


def drive():
    cfg = loop.load_config(ARM/'CONFIG.json')
    with (ARM/'cycle1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            for name in ('policy', 'collect', 'convert'):
                state = loop.load_state(cfg)
                if state['cycles'][-1]['phases'][name]['status'] == 'done':
                    continue
                if name == 'policy':
                    waiter.prepare_horizon(cfg, state, ARM)
                    event('policy1_frozen_wm', wm=state['accepted_wm'],
                          wm_config_sha256=loop.sha256(state['accepted_wm']['config']),
                          external_wm=loop.read_json(RELEASE/'LOOPRESUME_WM_PID.json'))
                code = resume.phase_many(loop, cfg, name, ARM)
                while code == 5:
                    time.sleep(30)
                    code = resume.phase_many(loop, cfg, name, ARM)
                if code not in (0, 4):
                    raise resume.OperationalPaused(f'cycle1 {name} exit {code}')
                event('cycle1_phase_done', phase=name)
            reconcile_wm(cfg)
            next_boundary(cfg)
            resume.drive(loop, waiter, cfg, ARM)
        except Exception as error:
            name = 'WAIT_OWNER.json' if resume.scientific(error) else 'OPERATIONAL_PAUSED.json'
            if isinstance(error, resume.OperationalPaused):
                name = 'OPERATIONAL_PAUSED.json'
            loop.write_atomic(ARM/name, {'t': loop.now(), 'error': repr(error), 'task': 'S-LOOPRESUME-C1-FALLBACK'})
            event('stopped', receipt=name, error=repr(error))
            raise


def status():
    cfg = loop.load_config(ARM/'CONFIG.json')
    state = loop.load_state(cfg)
    root = Path(cfg['loop_root'])
    doc = {'t': loop.now(), 'charged': state['sim_s_charged'], 'stopped': state['stopped'],
           'cycle': state['cycles'][-1]['cycle'], 'phases': {k: v['status'] for k, v in state['cycles'][-1]['phases'].items()},
           'policy': state['policy'], 'policy_wm': state.get('policy_wm'), 'accepted_wm': state['accepted_wm'],
           'wm0': loop.read_json(RELEASE/'LOOPRESUME_WM_ONLY.json')['status']}
    if (ARM/'PID.json').exists():
        doc['pid'] = pid = loop.read_json(ARM/'PID.json')
        doc['live'] = live(pid)
    attempt = state['cycles'][-1]['phases']['policy']['attempt']
    out = loop.out_dir(root/'policy', 'c01', attempt)
    doc['policy_attempt'] = attempt
    doc['policy_output'] = str(out)
    updates = out/'updates.jsonl'
    rows = [json.loads(x) for x in updates.read_text().splitlines()] if updates.exists() else []
    doc['updates'] = len(rows)
    doc['updates_this_attempt'] = len(rows)
    doc['last_update'] = rows[-1].get('update') if rows else None
    for name in ('gate_panel_result.json', 'native_gate_result.json', 'RUN_SUMMARY.json'):
        path = out/name
        if path.exists():
            d = loop.read_json(path)
            if name == 'RUN_SUMMARY.json' and d.get('updates'):
                doc['updates'] = len(d['updates'])
            doc[name] = ({k: d.get(k) for k in ('status', 'error', 'real_share', 'recommended_updates_per_cycle',
                         'critic_warmup', 'resumed_from')} if name == 'RUN_SUMMARY.json' else d)
    for name in ('WAIT_OWNER.json', 'OPERATIONAL_PAUSED.json'):
        if (ARM/name).exists():
            doc[name] = loop.read_json(ARM/name)
    journal = root/'worlds_c01.json'
    if journal.exists():
        d = loop.read_json(journal)
        doc['collection'] = {'cases': d['cases'], 'worlds': list(d['worlds']),
                             'charged': d['charged'], 'native_smoke': d.get('native_smoke'),
                             'continuation': d.get('native_smoke_collection')}
    loop.write_atomic(RELEASE/'C1_STATUS.json', doc)
    print(json.dumps(doc))
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'install', 'launch', 'drive', 'status'])
    globals()[parser.parse_args().action]()
