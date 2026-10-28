"""S-C3WMFT bounded gpuA continuation: registered FT/check/gate, no policy dispatch."""
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REL = Path('/root/autodl-tmp/c4/restart20261026')
ARM = REL / 'arm'
ROOT = REL / 'runtime_longint'
OUT = ARM / 'loop/longint/c3wmft'
RETRY = os.environ.get('C3WMFT_RETRY') == '1'
if RETRY:
    OUT = OUT / 'retry1'
CONFIG = ARM / 'CONFIG_C3_WMFT.json'
TASK = 'S-C3WMFT'
WM2 = 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    sys.path[:0] = [str(ROOT / 'scripts/c4'), str(ROOT / 'src'), str(ROOT)]
    import loop14
    return loop14


def locks():
    handles = []
    for name in ('autonomy.lock', 'driver.lock', 'cycle1.lock'):
        handle = (ARM / name).open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(handle)
    return handles


def event(loop, name, **details):
    row = {'t': loop.now(), 'event': TASK + '_' + name, **details}
    for path in (ARM / 'loop/cycle_ledger.jsonl', REL / 'AUTONOMY_EVENTS.jsonl'):
        loop.append(path, row)


def status(loop, value, **details):
    state = loop.read_json(ARM / 'loop/LOOP14_STATE.json')
    doc = {'t': loop.now(), 'status': value, 'task': TASK, 'pid': os.getpid(),
           'cycle': 3, 'charged': state['sim_s_charged'], 'cap': 280000,
           'remaining': 280000 - state['sim_s_charged'],
           'phases': {k: v['status'] for k, v in state['cycles'][-1]['phases'].items()},
           **details}
    loop.write_atomic(REL / 'AUTONOMY_STATUS.json', doc)
    loop.write_atomic(OUT / 'STATUS.json', doc)
    event(loop, value, **details)
    print(json.dumps(doc), flush=True)


def gpu_idle():
    proc = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                           '--format=csv,noheader,nounits'], capture_output=True,
                          text=True, check=True, timeout=15)
    assert not proc.stdout.strip(), 'GPU occupied; no launch'


def prepare(markdown_sha):
    loop = setup()
    held = locks()
    assert held and not OUT.exists()
    cfg = loop.load_config(ARM / 'CONFIG_C3_RESUME_R2.json')
    state = loop.load_state(cfg)
    assert loop.read_json(REL / 'AUTONOMY_STATUS.json')['status'] == 'WAIT_MAIN_C3_WMFT'
    cycle = state['cycles'][-1]
    assert cycle['cycle'] == 3 and state['sim_s_charged'] == 264544
    assert all(cycle['phases'][p]['status'] == 'done' for p in ('policy', 'collect', 'convert'))
    assert all(cycle['phases'][p] == {'status': 'pending', 'attempt': 0} for p in ('wm_ft', 'check'))
    assert state['accepted_wm']['sha256'] == sha(state['accepted_wm']['checkpoint']) == WM2
    assert (ARM / 'loop/STOP_DRIVER').exists() and not state.get('stopped')
    assert cfg['wm']['config'] == 'configs/c4_wm_loop14_r16_ft.toml'
    assert not cfg['looplong']['wm_loss'] and not cfg['fastloop']['cpu_checks']
    assert loop.sha256(cfg['looplong']['threshold']) == cfg['looplong']['threshold_sha256']
    loop.long_module().verify_pairs(loop, state, cycle['phases']['convert']['receipt'])
    gpu_idle()
    OUT.mkdir()
    loop.write_atomic(OUT / 'BEFORE.json', {'config': cfg, 'state': state})
    before_wm = copy.deepcopy(cfg['wm'])
    cfg['wm']['env']['C4_GPU_MEM_FRACTION'] = '0.6'
    loop.write_atomic(CONFIG, cfg)
    reg = loop.read_json(cfg['registration'])
    amendment = {'id': TASK, 't': loop.now(), 'authority': 'MAIN 2026-10-28 09:00',
        'markdown_registration_sha256': markdown_sha, 'base_sha256': WM2,
        'config_sha256': sha(CONFIG), 'recipe_sha256': sha(ROOT / cfg['wm']['config']),
        'script_sha256': sha(__file__), 'wm_before': before_wm, 'wm_after': cfg['wm'],
        'phases': ['wm_ft', 'check'], 'after_check': 'WAIT_MAIN_C4',
        'gate': cfg['looplong'], 'shards': 48, 'operational_ft_retries': 1,
        'new_sim_s': 0, 'charged': 264544, 'gpu_exclusive': True}
    assert not any(a.get('id') == TASK for a in reg['amendments'])
    reg['amendments'].append(amendment)
    loop.write_atomic(Path(cfg['registration']), reg)
    state['registration_sha256'] = sha(cfg['registration'])
    loop.save_state(cfg, state)
    loop.write_atomic(OUT / 'AUTHORITY.json', amendment)
    event(loop, 'MAIN_DECISION_REGISTERED', **amendment)
    print(json.dumps({'authority': amendment, 'registration_sha256': state['registration_sha256']}))


def finite_numbers(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_numbers(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_numbers(v) for v in value)
    return True


def training_finite(row):
    """Use the trainer's optimization contract, not possibly undefined diagnostics."""
    loss = row.get('loss')
    grad = row.get('grad_norm')
    return (isinstance(loss, (int, float)) and math.isfinite(loss)
            and row.get('finite') is not False
            and (grad is None or isinstance(grad, (int, float)) and math.isfinite(grad)))


def repair(markdown_sha):
    loop = setup()
    held = locks()
    assert held and RETRY and not OUT.exists()
    cfg = loop.load_config(CONFIG)
    state = loop.load_state(cfg)
    ph = state['cycles'][-1]['phases']['wm_ft']
    assert ph['status'] == 'failed' and ph['attempt'] == 1
    assert state['sim_s_charged'] == 264544 and state['accepted_wm']['sha256'] == WM2
    for name in ('PID.json', 'CHILDREN.jsonl'):
        for line in (OUT.parent / name).read_text().splitlines() if name.endswith('jsonl') else [(OUT.parent / name).read_text()]:
            old = json.loads(line)
            assert loop.fast_module().process_token(old['pid']) != str(old['start_ticks'])
    rows = training_rows(ARM / 'loop/wm/c03/train_log.jsonl')
    assert len(rows) == 2 and all(training_finite(row) for row in rows)
    assert rows[1]['merge_pivot_on_diagnostic_nll'] == float('inf')
    gpu_idle()
    OUT.mkdir()
    authority = {'id': TASK + '_MONITOR_REPAIR', 't': loop.now(),
        'markdown_registration_sha256': markdown_sha, 'config_sha256': sha(CONFIG),
        'script_sha256': sha(__file__), 'failed_attempt': ph,
        'diagnosis': 'undefined diagnostic NLL falsely classified as nonfinite optimization',
        'first_two_optimization_metrics_finite': True, 'retry': 1,
        'base_sha256': WM2, 'new_sim_s': 0, 'after_check': 'WAIT_MAIN_C4'}
    loop.write_atomic(OUT / 'AUTHORITY.json', authority)
    reg = loop.read_json(cfg['registration'])
    assert not any(a.get('id') == authority['id'] for a in reg['amendments'])
    reg['amendments'].append(authority)
    loop.write_atomic(Path(cfg['registration']), reg)
    state['registration_sha256'] = sha(cfg['registration'])
    loop.save_state(cfg, state)
    event(loop, 'MONITOR_OPERATIONAL_FAILURE_CORRECTED', **authority)
    print(json.dumps(authority))


def training_rows(path):
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows


def install(loop):
    original = loop.run_cmd

    def run_cmd(h, argv, *, env, log, timeout_s, cwd=None):
        argv = list(map(str, argv))
        if not ('wms' in argv and 'train' in argv and '--mode' in argv):
            return original(h, argv, env=env, log=log, timeout_s=timeout_s, cwd=cwd)
        gpu_idle()
        assert env['C4_GPU_MEM_FRACTION'] == '0.6'
        target = Path(argv[argv.index('--out') + 1])
        started = time.monotonic()
        early = False
        with Path(log).open('a') as stream:
            child = subprocess.Popen(argv, cwd=cwd or h['code_root'],
                env={**os.environ, **{k: str(v) for k, v in env.items()}},
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
            token = loop.fast_module().process_token(child.pid)
            record = {'pid': child.pid, 'start_ticks': token, 'sid': os.getsid(child.pid),
                      'argv': argv, 'out': str(target), 'log': str(log)}
            loop.append(OUT / 'CHILDREN.jsonl', record)
            event(loop, 'TRAINER_LAUNCHED', **record)
            while child.poll() is None:
                assert loop.fast_module().process_token(child.pid) in (None, token)
                rows = training_rows(target / 'train_log.jsonl')
                good = all(training_finite(row) for row in rows)
                if rows and not early:
                    receipt = {'t': loop.now(), 'pid': child.pid, 'rows': len(rows),
                               'optimization_metrics_finite': good, 'first_rows': rows[:3]}
                    loop.write_atomic(OUT / (target.name + '_FIRST_FINITE.json'), receipt)
                    event(loop, 'FIRST_STEPS_FINITE', rows=len(rows), finite=good, pid=child.pid)
                    early = True
                if not good or time.monotonic() - started > timeout_s:
                    if loop.fast_module().process_token(child.pid) == token:
                        child.terminate()
                    child.wait(timeout=120)
                    if not good:
                        raise ValueError('nonfinite scientific training metrics; no retry')
                    return 124
                time.sleep(10)
            rows = training_rows(target / 'train_log.jsonl')
            loop.write_atomic(OUT / (target.name + '_TRAIN_VERIFIED.json'), {
                'exit': child.returncode, 'rows': len(rows),
                'optimization_metrics_finite': all(training_finite(row) for row in rows),
                'nonfinite_diagnostics': [{k: str(v) for k, v in row.items()
                    if isinstance(v, float) and not math.isfinite(v)} for row in rows],
                'elapsed_s': time.monotonic() - started})
            if not all(training_finite(row) for row in rows):
                raise ValueError('nonfinite scientific training metrics; no retry')
            return child.returncode

    loop.run_cmd = run_cmd

    def gate(owner, cfg, state, wm, out, *, previous=None):
        section = cfg['looplong']
        assert sha(section['threshold']) == section['threshold_sha256']
        out = Path(out)
        directory = out.parent / 'gate_shards'
        status(loop, 'C3_WM3_HORIZON_RUNNING', wm3_sha256=wm['sha256'])
        env = {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': f'{ROOT}/src:{ROOT}',
               'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'WMS_LOOPLONG_SPEED': '1'}
        code = original(loop.host(cfg, 'gpu'), [sys.executable, '-B',
            str(ROOT / 'tools/longint/gate_shard.py'), 'launch', str(directory),
            wm['checkpoint'], str(ROOT / wm['config']), section['pairs300'], section['threshold'], '48'],
            env=env, log=OUT / 'horizon.log', timeout_s=14400)
        assert code == 0, 'horizon operational failure; retain shards'
        source = directory / 'HORIZON_GATE_WM2.json'
        doc = loop.read_json(source)
        assert doc['wm_sha256'] == wm['sha256'] and doc['schema'] == 'coopw_horizon_gate/1'
        assert doc['threshold'] == loop.read_json(section['threshold'])['threshold']
        assert doc['threshold_source']['sha256'] == section['threshold_sha256']
        assert doc['panel_seed'] == section['panel_seed'] and doc['groups'] <= section['groups']
        with out.open('xb') as stream:
            stream.write(source.read_bytes())
        short, old, target = loop.long_module().horizon_selection(doc,
            previous or state.get('looplong_policy_horizon_s', 60))
        return {'short_nonregression': short, 'previous_horizon_s': old, 'horizon_s': target,
                'gate_file': str(out), 'gate_sha256': sha(out), 'wm': dict(wm)}

    loop.long_module().gate = gate


def run():
    time.sleep(2)  # Let the receipt-writing detached launcher release its locks.
    loop = setup()
    held = locks()
    assert held and (ARM / 'loop/STOP_DRIVER').exists()
    cfg = loop.load_config(CONFIG)
    assert sha(CONFIG) == loop.read_json(OUT / 'AUTHORITY.json')['config_sha256']
    install(loop)
    try:
        status(loop, 'C3_WMFT_RUNNING')
        code = loop.retry(cfg, 'wm_ft') if RETRY else loop.run_phase(cfg, 'wm_ft')
        if code == 3 and not RETRY:
            failed = loop.load_state(cfg)['cycles'][-1]['phases']['wm_ft']
            loop.write_atomic(OUT / 'FAILED_FT_ATTEMPT1.json', failed)
            error = failed.get('error', '')
            if 'nonfinite' in error or 'finite False' in error:
                raise RuntimeError('scientific nonfinite FT failure; no retry')
            event(loop, 'OPERATIONAL_FT_RETRY', failed=failed, retry=1)
            code = loop.retry(cfg, 'wm_ft')
        assert code == 0, f'fine-tune exit {code}'
        status(loop, 'C3_WM3_ACCEPTANCE_RUNNING')
        code = loop.run_phase(cfg, 'check')
        assert code in (0, 4), f'acceptance/gate operational exit {code}'
        state = loop.load_state(cfg)
        assert state['cycles'][-1]['cycle'] == 3 and state['sim_s_charged'] == 264544
        assert (ARM / 'loop/STOP_DRIVER').exists()
        phases = state['cycles'][-1]['phases']
        result = {'task': TASK, 't': loop.now(), 'wm3_sha256': phases['wm_ft']['receipt']['candidate_sha256'],
                  'wm_ft': phases['wm_ft']['receipt'], 'acceptance': phases['check']['receipt'],
                  'accepted_wm': state['accepted_wm'], 'budget_stop': state.get('stopped'),
                  'charged': state['sim_s_charged'], 'status': 'WAIT_MAIN_C4'}
        gate_path = Path(phases['check']['receipt']['out']) / 'HORIZON_GATE.json'
        if gate_path.is_file():
            doc = loop.read_json(gate_path)
            result.update(horizons=doc['horizons'], threshold=doc['threshold'],
                          trusted_horizon_s=loop.long_module().horizon_selection(doc, 90)[2],
                          gate_sha256=sha(gate_path))
        loop.write_atomic(OUT / 'RESULTS.json', result)
        status(loop, 'WAIT_MAIN_C4', results=str(OUT / 'RESULTS.json'),
               wm3_sha256=result['wm3_sha256'], acceptance=result['acceptance']['decision'],
               trusted_horizon_s=result.get('trusted_horizon_s'))
    except Exception:
        error = traceback.format_exc()
        loop.write_atomic(OUT / 'ERROR.json', {'t': loop.now(), 'error': error})
        status(loop, 'C3_WMFT_OPERATIONAL_PAUSED', error=error)
        raise


def launch():
    loop = setup()
    held = locks()
    assert held and not (OUT / 'PID.json').exists()
    gpu_idle()
    event(loop, 'DETACHED_LAUNCH_AUTHORIZED', authority_sha256=sha(OUT / 'AUTHORITY.json'))
    with (OUT / 'driver.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-B', __file__, 'run'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    doc = {'pid': child.pid, 'start_ticks': loop.fast_module().process_token(child.pid),
           'sid': os.getsid(child.pid), 'script_sha256': sha(__file__)}
    loop.write_atomic(OUT / 'PID.json', doc)
    print(json.dumps(doc))


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare(sys.argv[2])
    elif sys.argv[1] == 'repair':
        repair(sys.argv[2])
    elif sys.argv[1] == 'launch':
        launch()
    elif sys.argv[1] == 'run':
        run()
    else:
        raise SystemExit('unknown mode')
