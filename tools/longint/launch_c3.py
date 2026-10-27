"""S-LONGINT cycle-3 POLICY phase on gpuA (registered; zero SUMO; no collection dispatch).

    prepare GATE_JSON   write arm/CONFIG_C3_LONGINT.json (create-only) and validate it
    start               detach `run` (own session) and record its PID/start ticks
    run                 hold the three chain flocks, run the canonical loop14 policy phase,
                        then leave the chain at WAIT_MAIN_C3_COLLECT (STOP_DRIVER stays)
"""
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parents[2]
REL = Path('/root/autodl-tmp/c4/restart20261026')
ARM = REL / 'arm'
CONFIG = ARM / 'CONFIG_C3_LONGINT.json'
OUT = ARM / 'loop/longint'
TASK = 'S-LONGINT'
POLICY_TIMEOUT_H = 24.0  # operational waiver, as S-C2H60 time_limit_waived


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loop14():
    for p in (CODE / 'scripts/c4', CODE / 'src', CODE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import loop14 as module
    return module


def prepare(gate):
    cfg = json.loads((ARM / 'CONFIG.json').read_text())
    gate_doc = json.loads(Path(gate).read_text())
    assert gate_doc['wm_sha256'].startswith('b8797242'), 'gate must be on WM2'
    cfg['hosts']['gpu'] = {**cfg['hosts']['gpu'], 'code_root': str(CODE)}
    p = cfg['policy']
    p['config'] = str(CODE / 'configs/c4_policy14_c3longint.toml')
    args = list(p['extra_args'])
    args[args.index('--horizon-gate') + 1] = str(gate)
    p['extra_args'] = args
    p['updates_per_cycle'] = 150
    cfg['loopresume'] = {**cfg['loopresume'], 'measured_update_selection': False}
    threshold = CODE / 'configs/coopw_horizon_gate_13E_b2.json'
    cfg['looplong'] = {'enabled': True, 'wm_loss': False, 'auto_horizon': True, 'long_credit': True,
                       'speed': True, 'hot_swap': True, 'exploration': True, 'enable_cycle': 3,
                       'pairs300': '/root/autodl-tmp/c4/v5/data/pairs300', 'threshold': str(threshold),
                       'threshold_sha256': sha(threshold), 'groups': 96, 'panel_seed': 20261026,
                       'coop_credit': True, 'imagination_horizon_s': 300}
    cfg['longint_c3'] = {'task': TASK, 'policy_timeout_h_waiver': POLICY_TIMEOUT_H,
                         'base_config_sha256': sha(ARM / 'CONFIG.json'), 'gate': str(gate),
                         'gate_sha256': sha(gate), 'collection_authorized': False,
                         'after_policy': 'WAIT_MAIN_C3_COLLECT'}
    with CONFIG.open('x') as stream:
        json.dump(cfg, stream, indent=1)
    loop = loop14()
    checked = loop.load_config(CONFIG)
    state = loop.load_state(checked)
    nxt = loop._phase(state)
    print(json.dumps({'config': str(CONFIG), 'sha256': sha(CONFIG), 'next': nxt['name'],
                      'cycle': state['cycles'][-1]['cycle'], 'charged': state['sim_s_charged'],
                      'updates': loop.measured_policy_updates(checked, state)}))


def register(markdown_sha):
    """Append one amendment; rebind the loop state to the new registration hash."""
    reg_path, state_path = ARM / 'REGISTRATION.json', ARM / 'loop/LOOP14_STATE.json'
    reg, state = json.loads(reg_path.read_text()), json.loads(state_path.read_text())
    assert state['registration_sha256'] == sha(reg_path)
    assert not any(a.get('id') == 'S-LONGINT-C3' for a in reg['amendments'])
    for path in (reg_path, state_path):
        backup = path.with_name(path.name + '.before_longint_c3')
        with backup.open('x') as stream:
            stream.write(path.read_text())
    reg['amendments'].append({
        'id': 'S-LONGINT-C3', 't': now(), 'authority': 'main 2026-10-27 18:45 act-on-recommendation',
        'markdown_registration_sha256': markdown_sha, 'code_commit': '9fcf73c6', 'cycle': 3,
        'phase': 'policy only', 'wm_sha256': state['accepted_wm']['sha256'],
        'start_policy_sha256': state['policy']['sha256'],
        'native_residual_train_group': True, 'native_residual_train_trunk': True,
        'looplong': {'auto_horizon': True, 'long_credit': True, 'imagination_horizon_s': 300,
                     'critic_bootstrap': False, 'wm_loss': False, 'speed': True, 'hot_swap': True,
                     'exploration': True, 'coop_credit': True},
        'memory': {'imagination_microbatch': 1, 'imagination_checkpoint_segment': 16,
                   'offload_futures': True, 'offload_saved_tensors': True, 'gpu_cap': 0.92},
        'updates': 150, 'measured_update_selection': False, 'policy_timeout_h_waiver': POLICY_TIMEOUT_H,
        'native_relative_gate': 'unchanged', 'new_sim_s': 0, 'collection_authorized': False,
        'after_policy': 'WAIT_MAIN_C3_COLLECT', 'probes': 'not enabled by this task (probe_cqr absent)'})
    tmp = reg_path.with_suffix('.longint.tmp')
    tmp.write_text(json.dumps(reg, indent=1))
    tmp.replace(reg_path)
    old = state['registration_sha256']
    state['registration_sha256'] = sha(reg_path)
    tmp = state_path.with_suffix('.longint.tmp')
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(state_path)
    event = {'t': now(), 'event': 'S-LONGINT_registered', 'registration_sha256_before': old,
             'registration_sha256': state['registration_sha256'], 'markdown_registration_sha256': markdown_sha}
    with (ARM / 'loop/cycle_ledger.jsonl').open('a') as stream:
        stream.write(json.dumps(event) + '\n')
    print(json.dumps(event))


def set_status(status, **extra):
    path = REL / 'AUTONOMY_STATUS.json'
    backup = REL / 'AUTONOMY_STATUS.json.before_longint_c3'
    doc = json.loads(path.read_text())
    if not backup.exists():
        backup.write_text(json.dumps(doc, indent=1))
    doc = copy.deepcopy(doc)
    doc.update(t=now(), status=status, longint_c3={'task': TASK, 'pid': os.getpid(), **extra})
    tmp = path.with_suffix('.longint.tmp')
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(path)
    with (REL / 'AUTONOMY_EVENTS.jsonl').open('a') as stream:
        stream.write(json.dumps({'t': now(), 'event': 'S-LONGINT_' + status, 'pid': os.getpid(), **extra}) + '\n')


def run():
    locks = []
    for name in ('autonomy.lock', 'driver.lock', 'cycle1.lock'):
        handle = (ARM / name).open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    assert (ARM / 'loop/STOP_DRIVER').exists(), 'STOP_DRIVER marker must stay installed'
    loop = loop14()
    original = loop.run_cmd

    def run_cmd(h, argv, *, env, log, timeout_s, cwd=None):
        if 'wms.control.policy14_cli' in map(str, argv):
            timeout_s = 3600 * POLICY_TIMEOUT_H
        return original(h, argv, env=env, log=log, timeout_s=timeout_s, cwd=cwd)

    loop.run_cmd = run_cmd
    set_status('C3_POLICY_RUNNING', config=str(CONFIG), config_sha256=sha(CONFIG))
    try:
        code = loop.run_phase(loop.load_config(CONFIG), 'policy')
    except Exception as error:  # noqa: BLE001 - recorded, chain stays held
        code = 99
        print(f'{type(error).__name__}: {error}', flush=True)
    state = json.loads((ARM / 'loop/LOOP14_STATE.json').read_text())
    ph = state['cycles'][-1]['phases']['policy']
    status = 'WAIT_MAIN_C3_COLLECT' if code == 0 and ph['status'] == 'done' else 'C3_POLICY_STOPPED'
    set_status(status, exit=code, phase=ph.get('status'), error=ph.get('error') or ph.get('refused'),
               policy=state.get('policy'), collection_authorized=False)
    return code


def start():
    log = (OUT / 'c3_policy.driver.log').open('x')
    child = subprocess.Popen([sys.executable, '-B', __file__, 'run'], cwd=CODE, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                             env={**os.environ, 'PYTHONPATH': f'{CODE}/src:{CODE}'})
    ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    doc = {'task': TASK, 'pid': child.pid, 'start_ticks': ticks, 't': now(), 'config': str(CONFIG)}
    with (OUT / 'C3_POLICY_PID.json').open('x') as stream:
        json.dump(doc, stream)
    print(json.dumps(doc))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'register':
        register(sys.argv[2])
    elif mode == 'prepare':
        prepare(sys.argv[2])
    elif mode == 'start':
        start()
    elif mode == 'run':
        sys.exit(run())
