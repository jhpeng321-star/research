"""Verify one canary gates all remaining dispatch and retains charge on failure."""
import inspect
import json
from pathlib import Path

import pytest

from tools.loopresume.c1collect3_patch import single_canary_source
from .test_fastloop import fixture, fast, loop


@pytest.mark.parametrize('failure', [False, True])
def test_one_canary_before_every_remaining_world(tmp_path, monkeypatch, failure):
    from tools.loopresume import c1collect_box
    from wms.control import nativeinit
    cfg = fixture(tmp_path)
    cfg['policy']['native_init'] = True
    cfg['collect'].update(worlds_per_cycle=6, branch_points=2, native_smoke_harmful_only_from_cycle=1)
    state = loop.load_state(cfg)
    state['cycles'][-1]['cycle'] = 1
    root = Path(cfg['loop_root'])
    seeds = loop.allocate_seeds(cfg, state, loop.check_registration(cfg), root)
    cfg['collect']['startup_canary'] = {'cycle': 1, 'boxes': ['cpuP'],
                                      'next_attempts': {str(seed): 1 for seed in seeds}}
    verified, dispatched = [], []

    def world(loop_, cfg_, state_, phase, root_, item, box, attempt):
        if item['index'] != 0:
            assert verified == ['cpuP']
        ledger = [json.loads(line) for line in (root_/'seed_ledger.jsonl').read_text().splitlines()]
        assert any(row.get('seed') == item['seed'] and row.get('event') == 'world_attempt' for row in ledger)
        dispatched.append(item['seed'])
        return {'artifact': str(root_/str(item['index'])), 'case': item, 'collect': {}}

    def wait(loop_, cfg_, item, box, attempt, future):
        future.result()
        assert dispatched == [seeds[0]] and not verified
        if failure:
            raise RuntimeError('canary numerical output failed')
        verified.append(box)
        return {'box': box, 'verified': True}

    source = inspect.getsource(fast.collect)
    if "for box_name in ('cpuP', 'cpuQ'):" in source:
        source = single_canary_source(source)
    namespace = {**vars(fast), '_world': world}
    exec(compile(source, '<single-canary-collect>', 'exec'), namespace)
    monkeypatch.setattr(c1collect_box, 'wait_canary', wait)
    monkeypatch.setattr(nativeinit, 'artifact_world_delta', lambda path, n: -1.)
    if failure:
        with pytest.raises(RuntimeError, match='numerical output failed'):
            namespace['collect'](loop, cfg, state, {'attempt': 1}, root)
        assert dispatched == [seeds[0]]
    else:
        assert len(namespace['collect'](loop, cfg, state, {'attempt': 1}, root)['worlds']) == 6
    assert state['sim_s_charged'] == len(dispatched)*(loop.collect_sim_s(cfg)//6)


def test_driver_retries_failed_phase_once_and_pauses_without_remainder(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    from tools.loopresume import c1collect3_box as driver, continue_c1
    state = {'cycles': [{'phases': {'collect': {'status': 'failed'}}}], 'stopped': None}
    calls = []
    fake = SimpleNamespace(load_config=lambda path: {}, load_state=lambda cfg: state,
                           now=lambda: 'test', write_atomic=lambda path, doc: None)

    def retry(cfg, phase):
        calls.append(phase)
        return 3

    fake.retry = retry
    monkeypatch.setitem(sys.modules, 'loop14', fake)
    monkeypatch.setitem(sys.modules, 'loop14_resume', SimpleNamespace(OperationalPaused=RuntimeError))
    monkeypatch.setattr(continue_c1, 'event', lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, 'ARM', tmp_path)
    monkeypatch.setattr(driver, 'RELEASE', tmp_path)
    with pytest.raises(RuntimeError, match='single collection attempt exit 3'):
        driver.drive()
    assert calls == ['collect']
    assert (tmp_path/'C1COLLECT3_ATTEMPT.json').exists()
