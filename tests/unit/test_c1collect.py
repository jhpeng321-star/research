"""Real dispatch accounting/barriers with mocked scientific work; no SUMO."""
import json
from pathlib import Path

import pytest

from .test_fastloop import fixture, fast, loop


@pytest.mark.parametrize('failed_box', ['cpuP', 'cpuQ', None])
def test_serial_canaries_charge_only_dispatched_worlds(tmp_path, monkeypatch, failed_box):
    from tools.loopresume import c1collect_box as box_module
    from wms.control import nativeinit
    cfg = fixture(tmp_path)
    cfg['policy']['native_init'] = True
    cfg['collect'].update(worlds_per_cycle=6, branch_points=2, native_smoke_harmful_only_from_cycle=1)
    state = loop.load_state(cfg)
    state['cycles'][-1]['cycle'] = 1
    seeds = loop.allocate_seeds(cfg, state, loop.check_registration(cfg), Path(cfg['loop_root']))
    cfg['collect']['startup_canary'] = {'cycle': 1, 'next_attempts': {str(s): 1 for s in seeds}}
    verified, dispatched = [], []

    def world(loop_, cfg_, state_, ph, root_, item, box, attempt):
        if item['index']>=2:
            assert verified == ['cpuP', 'cpuQ']
        ledger = [json.loads(line) for line in (root_/'seed_ledger.jsonl').read_text().splitlines()]
        assert any(r.get('seed')==item['seed'] and r.get('event')=='world_attempt' for r in ledger)
        dispatched.append(box)
        return {'artifact': str(root_/str(item['index'])), 'case': item, 'collect': {}}

    def wait(loop_, cfg_, item, box, attempt, future):
        future.result()
        assert verified == ([] if box=='cpuP' else ['cpuP'])
        if box == failed_box:
            raise RuntimeError('pre-SUMO startup failure')
        verified.append(box)
        return {'box': box, 'verified': True}

    monkeypatch.setattr(fast, '_world', world)
    monkeypatch.setattr(box_module, 'wait_canary', wait)
    monkeypatch.setattr(nativeinit, 'artifact_world_delta', lambda path, n: -1.)
    if failed_box:
        with pytest.raises(RuntimeError, match='pre-SUMO'):
            fast.collect(loop, cfg, state, {'attempt': 1}, Path(cfg['loop_root']))
        assert dispatched == (['cpuP'] if failed_box=='cpuP' else ['cpuP', 'cpuQ'])
    else:
        assert len(fast.collect(loop, cfg, state, {'attempt': 1}, Path(cfg['loop_root']))['worlds']) == 6
    assert state['sim_s_charged'] == len(dispatched)*(loop.collect_sim_s(cfg)//6)


def test_cycle_attempt_allowances_remain_cycle_scoped(tmp_path, monkeypatch):
    cfg = fixture(tmp_path)
    state = loop.load_state(cfg)
    reg = loop.read_json(cfg['registration'])
    per_world = loop.collect_sim_s(cfg)//cfg['collect']['worlds_per_cycle']
    reg['budget']['sim_s_per_cycle'] = per_world
    reg['budget']['cycle_attempt_allowances'] = {'1': loop.collect_sim_s(cfg)}
    loop.write_atomic(Path(cfg['registration']), reg)
    monkeypatch.setattr(fast, '_world', lambda *a: {'artifact': 'mock', 'collect': {}})
    with pytest.raises(loop.Refused, match='per-cycle'):
        fast.collect(loop, cfg, state, {'attempt': 1}, Path(cfg['loop_root']))
    state['cycles'][-1]['cycle'] = 1
    assert len(fast.collect(loop, cfg, state, {'attempt': 1}, Path(cfg['loop_root']))['worlds']) == cfg['collect']['worlds_per_cycle']


def test_probe_uses_registered_ssh_argv_and_never_counts_a_bare_process(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.loopresume import c1collect_box as box_module
    cfg = fixture(tmp_path)
    cfg['hosts']['cpuP']['ssh'] = ['ssh', '-o', 'BatchMode=yes', 'cpuP']
    calls = []

    def run(argv, **kwargs):
        assert all(isinstance(arg, str) for arg in argv)
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=json.dumps({'sumo': [{'pid': 123}], 'outputs': [], 'verified': False}))

    monkeypatch.setattr(box_module.subprocess, 'run', run)
    result = box_module.probe(cfg, {'seed': 968158}, 'cpuP', 4)
    assert not result['verified']
    assert calls[0][0][:-1] == cfg['hosts']['cpuP']['ssh']
    assert '/c01.a4' in calls[0][1]['input'] and '968158' in calls[0][1]['input']


def test_real_wait_propagates_canary_failure_without_further_dispatch(monkeypatch):
    from concurrent.futures import Future
    from tools.loopresume import c1collect_box as box_module
    future = Future()
    future.set_exception(RuntimeError('missing runtime model before first SUMO frame'))
    monkeypatch.setattr(box_module, 'probe', lambda *a: {'verified': False, 'outputs': [], 'sumo': []})
    with pytest.raises(RuntimeError, match='missing runtime model'):
        box_module.wait_canary(loop, {}, {'seed': 968158}, 'cpuP', 4, future)
