"""Complete only C3 acceptance after the canonical reducer staging omission."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.longint import wmft_c3 as core

PARENT = core.ARM / 'loop/longint/c3wmft'
OUT = PARENT / 'check_repair1'
WM3 = '41e18f09b6060a1f4e4039f02655fcbcd497e32a360cffb71be716fffeaadc84'
MISSING = 'scripts/round13c_waiter.py'


def eligible(state, before):
    """Refuse scientific rejections, another FT, or any changed boundary."""
    assert state['cycles'][-1]['cycle'] == 3 and state['sim_s_charged'] == 264544
    assert state['accepted_wm']['sha256'] == core.WM2
    phases = state['cycles'][-1]['phases']
    ft, check = phases['wm_ft'], phases['check']
    assert ft['status'] == 'done' and ft['attempt'] == 2
    assert ft['receipt']['candidate_sha256'] == WM3 and ft['receipt']['exit'] == 0
    assert check['status'] == 'done' and check['attempt'] == 1
    receipt = check['receipt']
    assert receipt['exit'] == 2 and receipt['decision'] == 'REJECT'
    assert receipt['reasons'] == ['measurement error (fail closed)']
    assert receipt['error'] == f"FileNotFoundError: [Errno 2] No such file or directory: '{core.ROOT / MISSING}'"
    assert state['stopped'] == 'registered SUMO budget cannot fund another collection'
    assert state['consecutive_rejects'] == before['consecutive_rejects'] + 1
    return phases


def prepare(markdown_sha, helper_sha):
    loop = core.setup()
    held = core.locks()
    assert held and not OUT.exists() and (core.ARM / 'loop/STOP_DRIVER').is_file()
    cfg = loop.load_config(core.CONFIG)
    state = loop.load_state(cfg)
    before = loop.read_json(PARENT / 'BEFORE.json')['state']
    eligible(state, before)
    assert loop.read_json(core.REL / 'AUTONOMY_STATUS.json')['status'] == 'WAIT_MAIN_C4'
    for directory in (PARENT, PARENT / 'retry1'):
        records = [loop.read_json(directory / 'PID.json')]
        records += [json.loads(line) for line in (directory / 'CHILDREN.jsonl').read_text().splitlines()]
        for record in records:
            assert loop.fast_module().process_token(record['pid']) != str(record['start_ticks'])
    assert core.sha(core.ROOT / MISSING) == helper_sha
    assert core.sha(state['cycles'][-1]['phases']['wm_ft']['receipt']['candidate']) == WM3
    OUT.mkdir()
    loop.write_atomic(OUT / 'BEFORE.json', state)
    authority = {'id': core.TASK + '_ACCEPTANCE_DEPLOYMENT_REPAIR', 't': loop.now(),
        'markdown_registration_sha256': markdown_sha, 'script_sha256': core.sha(__file__),
        'helper_path': MISSING, 'helper_sha256': helper_sha, 'wm3_sha256': WM3,
        'config_sha256': core.sha(core.CONFIG),
        'accept_source_sha256': core.sha(core.ROOT / 'tools/loop14_checks/accept.py'),
        'prior_consecutive_rejects': before['consecutive_rejects'],
        'phases': ['check'], 'check_attempt': 2, 'new_sim_s': 0, 'charged': 264544,
        'after_check': 'WAIT_MAIN_C4', 'budget_stop_preserved': state['stopped']}
    reg = loop.read_json(cfg['registration'])
    assert not any(a.get('id') == authority['id'] for a in reg['amendments'])
    reg['amendments'].append(authority)
    loop.write_atomic(Path(cfg['registration']), reg)
    state['registration_sha256'] = core.sha(cfg['registration'])
    loop.save_state(cfg, state)
    loop.write_atomic(OUT / 'AUTHORITY.json', authority)
    core.event(loop, 'ACCEPTANCE_DEPLOYMENT_REPAIR_REGISTERED', **authority)
    print(json.dumps(authority))


def complete_check(loop, cfg, state, authority):
    """Call the registered checker directly, preserving the existing budget stop."""
    before = loop.read_json(PARENT / 'BEFORE.json')['state']
    phases = eligible(state, before)
    original = copy.deepcopy(phases['check'])
    state['consecutive_rejects'] = authority['prior_consecutive_rejects']
    ph = {'status': 'running', 'attempt': 2, 'started': loop.now(), 'pid': os.getpid(),
          'coordinator_start': loop.fast_module().process_token(os.getpid()),
          'previous_operational_attempt': original}
    phases['check'] = ph
    loop.save_state(cfg, state)
    core.status(loop, 'C3_WM3_ACCEPTANCE_RUNNING', check_attempt=2, wm3_sha256=WM3)
    receipt = loop.run_check(cfg, state, ph, Path(cfg['loop_root']))
    ph.update(status='done', finished=loop.now(), receipt=receipt)
    assert state['stopped'] == authority['budget_stop_preserved']
    assert state['cycles'][-1]['cycle'] == 3 and state['sim_s_charged'] == 264544
    assert phases['wm_ft']['receipt']['candidate_sha256'] == WM3
    assert (core.ARM / 'loop/STOP_DRIVER').is_file()
    loop.save_state(cfg, state)
    core.event(loop, 'ACCEPTANCE_REPAIR_COMPLETE', receipt=receipt, previous_attempt=original,
               charged=264544, budget_stop=state['stopped'])
    result = {'task': core.TASK, 't': loop.now(), 'wm3_sha256': WM3,
              'wm_ft': phases['wm_ft']['receipt'], 'acceptance': receipt,
              'accepted_wm': state['accepted_wm'], 'budget_stop': state['stopped'],
              'charged': 264544, 'status': 'WAIT_MAIN_C4'}
    gate = Path(receipt['out']) / 'HORIZON_GATE.json'
    if gate.is_file():
        doc = loop.read_json(gate)
        result.update(horizons=doc['horizons'], threshold=doc['threshold'],
                      trusted_horizon_s=loop.long_module().horizon_selection(doc, 90)[2],
                      gate_sha256=core.sha(gate))
    loop.write_atomic(OUT / 'RESULTS.json', result)
    core.status(loop, 'WAIT_MAIN_C4', results=str(OUT / 'RESULTS.json'), wm3_sha256=WM3,
                acceptance=receipt['decision'], trusted_horizon_s=result.get('trusted_horizon_s'))


def run():
    time.sleep(2)
    loop = core.setup()
    held = core.locks()
    assert held
    authority = loop.read_json(OUT / 'AUTHORITY.json')
    assert core.sha(__file__) == authority['script_sha256']
    assert core.sha(core.CONFIG) == authority['config_sha256']
    assert core.sha(core.ROOT / MISSING) == authority['helper_sha256']
    assert core.sha(core.ROOT / 'tools/loop14_checks/accept.py') == authority['accept_source_sha256']
    cfg = loop.load_config(core.CONFIG)
    core.install(loop)  # Same registered gate; this script never calls an FT runner.
    try:
        complete_check(loop, cfg, loop.load_state(cfg), authority)
    except Exception:
        error = traceback.format_exc()
        loop.write_atomic(OUT / 'ERROR.json', {'t': loop.now(), 'error': error})
        core.status(loop, 'C3_CHECK_REPAIR_OPERATIONAL_PAUSED', error=error)
        raise


def launch():
    loop = core.setup()
    held = core.locks()
    assert held and not (OUT / 'PID.json').exists()
    assert loop.read_json(core.REL / 'AUTONOMY_STATUS.json')['status'] == 'WAIT_MAIN_C4'
    core.event(loop, 'ACCEPTANCE_REPAIR_DETACHED_LAUNCH', authority_sha256=core.sha(OUT / 'AUTHORITY.json'))
    with (OUT / 'driver.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-B', __file__, 'run'], cwd=core.ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    doc = {'pid': child.pid, 'start_ticks': loop.fast_module().process_token(child.pid),
           'sid': os.getsid(child.pid), 'script_sha256': core.sha(__file__)}
    loop.write_atomic(OUT / 'PID.json', doc)
    print(json.dumps(doc))


if __name__ == '__main__':
    core.OUT = OUT
    if sys.argv[1] == 'prepare':
        prepare(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'launch':
        launch()
    elif sys.argv[1] == 'run':
        run()
    else:
        raise SystemExit('unknown mode')
