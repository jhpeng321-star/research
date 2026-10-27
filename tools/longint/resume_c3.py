"""MAIN06:20 same-phase C3 recovery; explicit collect/convert boundary only."""
import copy
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REL = Path('/root/autodl-tmp/c4/restart20261026')
ARM = REL / 'arm'
ROOT = REL / 'runtime_longint'
OUT = ARM / 'loop/longint/c3resume'
CONFIG = ARM / 'CONFIG_C3_RESUME.json'
TASK = 'S-C3RESUME'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def setup():
    sys.path[:0] = [str(ROOT / 'scripts/c4'), str(ROOT / 'src'), str(ROOT),
                    str(REL / 'autonomy')]
    import loop14
    return loop14


def event(loop, name, **details):
    row = {'t': loop.now(), 'event': TASK + '_' + name, **details}
    for path in (ARM / 'loop/cycle_ledger.jsonl', REL / 'AUTONOMY_EVENTS.jsonl'):
        loop.append(path, row)


def status(loop, value, **details):
    state = loop.read_json(ARM / 'loop/LOOP14_STATE.json')
    doc = {'t': loop.now(), 'status': value, 'task': TASK, 'pid': os.getpid(),
           'cycle': 3, 'cap': 280000, 'charged': state['sim_s_charged'],
           'remaining': 280000 - state['sim_s_charged'],
           'phases': {k: v['status'] for k, v in state['cycles'][-1]['phases'].items()},
           **details}
    loop.write_atomic(REL / 'AUTONOMY_STATUS.json', doc)
    event(loop, value, **details)
    print(json.dumps(doc), flush=True)


def locks():
    result = []
    for name in ('autonomy.lock', 'driver.lock', 'cycle1.lock'):
        handle = (ARM / name).open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result.append(handle)
    return result


def prepare(markdown_sha):
    loop = setup()
    held = locks()
    assert held and OUT.parent.is_dir() and not OUT.exists()
    cfg = loop.load_config(ARM / 'CONFIG_C3_LONGINT.json')
    state = loop.load_state(cfg)
    ph = state['cycles'][-1]['phases']['policy']
    assert state['cycles'][-1]['cycle'] == 3 and ph['status'] == 'failed' and ph['attempt'] == 4
    assert state['sim_s_charged'] == 213406
    assert state['accepted_wm']['sha256'].startswith('b8797242')
    assert (ARM / 'loop/STOP_DRIVER').exists()
    old = ARM / 'loop/policy/c03.a4'
    summary = loop.read_json(old / 'RUN_SUMMARY.json')
    assert 'OutOfMemoryError: CUDA out of memory' in summary['error']
    import torch
    saved = torch.load(old / 'policy_000514.pt', map_location='cpu', weights_only=True)
    assert saved['update_count'] == 514
    from wms.control.nativeinit import select_update
    best = select_update(summary['native_curve'])
    best_payload = torch.load(old / 'policy_native_best.pt', map_location='cpu', weights_only=True)
    assert best_payload['update_count'] == best
    assert summary['resumed_from']['update_count'] + 150 == 530
    OUT.mkdir()
    loop.write_atomic(OUT / 'BEFORE.json', {'config': cfg, 'state': state,
        'registration': loop.read_json(cfg['registration'])})
    config = ROOT / 'configs/c4_policy14_c3resume.toml'
    receipt = OUT / 'RECOVERY.json'
    assert not config.exists()
    config.write_text('extends = "c4_policy14_c3longint.toml"\n[run]\n'
        + 'operational_recovery = ' + json.dumps(str(receipt)) + '\n'
        + '[trainer]\nimagination_checkpoint_segment = 8\n')
    recovery = {'id': TASK, 'authority': 'main 2026-10-28 06:20',
        'summary': str(old / 'RUN_SUMMARY.json'), 'checkpoint': str(old / 'policy_000514.pt'),
        'config': str(config), 'gate_panel': str(ARM / 'gate_panel.json'),
        'best_checkpoint': str(old / 'policy_native_best.pt'), 'best_update': best,
        'retry_outputs': [str(ARM / 'loop/policy/c03.a5')], 'seed': 965143,
        'phase_updates': 150, 'saved_update': 514, 'checkpoint_every_update': True,
        'real_returns_sha256': sha(state['real_returns'])}
    for key in ('summary', 'checkpoint', 'config', 'gate_panel', 'best_checkpoint'):
        recovery[key + '_sha256'] = sha(recovery[key])
    loop.write_atomic(receipt, recovery)
    cfg['policy']['config'] = str(config)
    cfg['policy']['env'].update(PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                              C4_GPU_MEM_FRACTION='0.92')
    cfg['longint_c3'].update(collection_authorized=True, after_policy='collect then convert only',
                            after_convert='WAIT_MAIN_C3_WMFT')
    loop.write_atomic(CONFIG, cfg)
    reg = loop.read_json(cfg['registration'])
    assert not any(a.get('id') == TASK for a in reg['amendments'])
    amendment = {'id': TASK, 't': loop.now(), 'authority': 'main 2026-10-28 06:20',
        'markdown_registration_sha256': markdown_sha, 'recovery_sha256': sha(receipt),
        'config_sha256': sha(CONFIG), 'script_sha256': sha(__file__),
        'failed_attempt': copy.deepcopy(ph), 'saved_update': 514, 'target_update': 530,
        'best_update_before_retry': best, 'collection_authorized_after_native_gate': True,
        'worlds': 13, 'probes': False, 'cap': 280000, 'after_convert': 'WAIT_MAIN_C3_WMFT',
        'memory': {'microbatch': 1, 'segment': 8, 'offload_futures': True,
                   'offload_saved_tensors': True, 'gpu_cap': 0.92,
                   'allocator': 'expandable_segments:True', 'boundary_gc_empty_cache': True}}
    reg['amendments'].append(amendment)
    loop.write_atomic(Path(cfg['registration']), reg)
    state['registration_sha256'] = sha(cfg['registration'])
    loop.save_state(cfg, state)
    event(loop, 'failed_attempt', attempt=4, saved_update=514, error=ph['error'])
    event(loop, 'retry_registered', **amendment)
    print(json.dumps({'registration_sha256': state['registration_sha256'], 'recovery': recovery}))


def entry(argv):
    """Allocation-lifetime-only cleanup; no trainer mathematics or RNG operations."""
    setup()
    import torch
    from wms.control.dreamer import DreamerTrainer
    original = DreamerTrainer.update

    def update(self, *args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        return original(self, *args, **kwargs)

    DreamerTrainer.update = update
    from wms.control.policy14_cli import main
    return main(argv)


def configure_collect(loop):
    remote = module('c3resume_autonomy', REL / 'autonomy/remote.py')
    remote.R, remote.A, remote.ROOT, remote.loop = REL, ARM, ROOT, loop
    collector = module('c3resume_collector', REL / 'autonomy/collector.py')
    fast = loop.fast_module()
    fast.collect = lambda l, c, s, p, r: collector.collect(
        l, c, s, p, r, remote.preflight,
        lambda value, **kw: status(loop, value, **kw), remote.priority)
    loop.fast_module = lambda: fast


def run_command(loop, original, h, argv, *, env, log, timeout_s, cwd=None):
    argv = list(map(str, argv))
    if 'wms.control.policy14_cli' not in argv:
        return original(h, argv, env=env, log=log, timeout_s=timeout_s, cwd=cwd)
    i = argv.index('-m')
    argv[i:i + 2] = [str(Path(__file__).resolve()), 'entry']
    with Path(log).open('a') as stream:
        child = subprocess.Popen(argv, cwd=cwd or h['code_root'],
            env={**os.environ, **{k: str(v) for k, v in env.items()}},
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
        token = loop.fast_module().process_token(child.pid)
        loop.write_atomic(OUT / 'TRAINER_PID.json', {'pid': child.pid, 'start_ticks': token,
            'sid': os.getsid(child.pid), 'argv': argv})
        while child.poll() is None:
            assert loop.fast_module().process_token(child.pid) in (None, token)
            sample = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            # Persist only our receipt-bound child's memory, never peers' details.
            for line in sample.stdout.splitlines():
                parts = line.split(',')
                if parts[0].strip() == str(child.pid):
                    loop.append(OUT / 'GPU_MEMORY.jsonl', {'t': loop.now(), 'pid': child.pid,
                        'used_mib': int(parts[1].strip())})
            time.sleep(5)
        return child.returncode


def verify_policy(loop, cfg):
    state = loop.load_state(cfg)
    out = Path(state['policy']['checkpoint']).parent
    doc = loop.read_json(out / 'RUN_SUMMARY.json')
    rows = [r for r in doc['updates'] if r['update'] > 514]
    assert [r['update'] for r in rows] == list(range(515, 531))
    import math
    import torch
    for r in rows:
        assert all(math.isfinite(r[k]) for k in (
            'actor_loss', 'critic_loss', 'actor_grad_norm', 'critic_grad_norm'))
    cap = torch.cuda.get_device_properties(0).total_memory * 0.92
    peak_reserved = max(r['cuda_peak_reserved_bytes'] for r in rows)
    peak_allocated = max(r['cuda_peak_allocated_bytes'] for r in rows)
    samples = [json.loads(x) for x in (OUT / 'GPU_MEMORY.jsonl').read_text().splitlines()]
    process_peak = max(r['used_mib'] for r in samples) * 1024**2
    assert max(peak_reserved, peak_allocated, process_peak) < cap
    from wms.control.nativeinit import verify_collection_gate
    gate = verify_collection_gate(state['policy'])
    assert gate['measured_through'] == 530
    proof = {'finite_updates': [r['update'] for r in rows], 'cap_bytes': cap,
        'peak_allocated_bytes': peak_allocated, 'peak_reserved_bytes': peak_reserved,
        'sampled_process_peak_bytes': process_peak, 'sample_interval_s': 5,
        'gate': gate, 'policy': state['policy']}
    loop.write_atomic(OUT / 'POLICY_VERIFIED.json', proof)
    event(loop, 'policy_verified', **proof)
    return proof


def drive():
    loop = setup()
    held = locks()
    assert held and (ARM / 'loop/STOP_DRIVER').exists()
    cfg = loop.load_config(CONFIG)
    original = loop.run_cmd
    loop.run_cmd = lambda h, argv, **kw: run_command(loop, original, h, argv, **kw)
    try:
        state = loop.load_state(cfg)
        assert state['cycles'][-1]['cycle'] == 3 and state['sim_s_charged'] == 213406
        assert state['cycles'][-1]['phases']['policy']['status'] == 'failed'
        status(loop, 'C3_POLICY_RESUMING', saved_update=514, target_update=530)
        code = loop.retry(cfg, 'policy')
        if code:
            status(loop, 'C3_POLICY_STOPPED', exit=code)
            return code
        verify_policy(loop, cfg)
        status(loop, 'C3_NATIVE_GATE_PASS')
        configure_collect(loop)
        for phase in ('collect', 'convert'):
            state = loop.load_state(cfg)
            assert state['cycles'][-1]['cycle'] == 3 and not state.get('stopped')
            assert state['sim_s_charged'] <= 280000
            status(loop, 'C3_' + phase.upper() + '_RUNNING')
            code = loop.run_phase(cfg, phase)
            if code:
                status(loop, 'C3_' + phase.upper() + '_STOPPED', exit=code)
                return code
        state = loop.load_state(cfg)
        assert state['cycles'][-1]['phases']['wm_ft']['status'] == 'pending'
        assert state['sim_s_charged'] <= 280000
        status(loop, 'WAIT_MAIN_C3_WMFT', wm_base_decision='MAIN',
               policy=state['policy'], collect=state['cycles'][-1]['phases']['collect']['receipt'],
               convert=state['cycles'][-1]['phases']['convert']['receipt'])
        return 0
    except Exception as error:
        status(loop, 'OPERATIONAL_PAUSED', reason=repr(error))
        raise


def start():
    loop = setup()
    assert not (OUT / 'PID.json').exists()
    with (OUT / 'driver.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-B', __file__, 'drive'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    proof = {'task': TASK, 'pid': child.pid,
             'start_ticks': loop.fast_module().process_token(child.pid), 'sid': os.getsid(child.pid)}
    loop.write_atomic(OUT / 'PID.json', proof)
    print(json.dumps(proof))


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'prepare':
        prepare(sys.argv[2])
    elif action == 'start':
        start()
    elif action == 'drive':
        sys.exit(drive())
    elif action == 'entry':
        sys.exit(entry(sys.argv[2:]))
