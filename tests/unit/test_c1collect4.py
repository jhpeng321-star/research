"""Resume completed bootstrap without recollection; bounded operational continuation."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.loopresume.c1collect4_patch import resume_source
from tools.loopresume.c1collect4_box import phase_once
from .test_fastloop import artifact, fixture, fast, loop


def test_completed_bootstrap_resumes_exactly_nine_without_canary(tmp_path, monkeypatch):
    from wms.control import nativeinit
    from tools.loopresume import c1collect_box
    cfg = fixture(tmp_path)
    state = loop.load_state(cfg)
    reg = loop.read_json(cfg['registration'])
    reg['seed_block'] = [968000, 968099]
    reg['budget'].update(sim_s_per_cycle=50000, sim_s_total=210000)
    loop.write_atomic(Path(cfg['registration']), reg)
    state['registration_sha256'] = loop.sha256(cfg['registration'])
    cfg['policy']['native_init'] = True
    cfg['collect'].update(worlds_per_cycle=13, branch_points=2, native_smoke_harmful_only_from_cycle=1)
    state['cycles'][-1]['cycle'] = 1
    root = Path(cfg['loop_root'])
    seeds = loop.allocate_seeds(cfg, state, loop.check_registration(cfg), root)
    cases = fast.cases(cfg, seeds)
    worlds = {}
    for case in cases[:4]:
        out = root/str(case['seed'])
        artifact(out)
        worlds[str(case['seed'])] = {'artifact': str(out), 'case': case, 'collect': {}}
    doc = {'cases': cases, 'worlds': worlds, 'attempts': {str(s): 1 for s in seeds[:4]},
           'charged': 0, 'policy_sha256': state['policy']['sha256'],
           'wm_sha256': state['accepted_wm']['sha256'],
           'startup_canaries': {'cpuP': {'verified': True}},
           'native_smoke': {'verdict': 'pass', 'passed': True}}
    loop.write_atomic(root/'worlds_c01.json', doc)
    cfg['collect']['startup_canary'] = {'cycle': 1, 'boxes': ['cpuP'],
                                      'next_attempts': {str(s): 1 for s in seeds[4:]}}
    dispatched = []

    def world(loop_, cfg_, state_, ph, root_, case, box, attempt):
        assert case['seed'] not in seeds[:4]
        ledger = [json.loads(s) for s in (root_/'seed_ledger.jsonl').read_text().splitlines()]
        assert any(r.get('seed') == case['seed'] and r.get('sim_s_charged') for r in ledger)
        dispatched.append((case['seed'], box))
        return {'artifact': 'mock-new', 'case': case, 'collect': {}}

    monkeypatch.setattr(c1collect_box, 'wait_canary', lambda *a: pytest.fail('consumed canary repeated'))
    monkeypatch.setattr(nativeinit, 'artifact_world_delta', lambda *a: -1.)
    source = inspect.getsource(fast.collect)
    if 'and batch_index == 0 and ordered:' not in source:
        source = resume_source(source)
    namespace = {**vars(fast), '_world': world}
    exec(compile(source, '<resume-collect>', 'exec'), namespace)
    result = namespace['collect'](loop, cfg, state, {'attempt': 5}, root)
    assert sorted(s for s, _ in dispatched) == seeds[4:]
    assert {b for _, b in dispatched} == {'cpuP', 'cpuQ'}
    assert len(result['worlds']) == 13
    assert state['sim_s_charged'] == 9*(loop.collect_sim_s(cfg)//13)
    assert loop.read_json(root/'worlds_c01.json')['startup_canaries'] == doc['startup_canaries']


@pytest.mark.parametrize('codes,scientific,expected', [([3, 0], False, 0), ([3, 3], False, None),
                                                    ([3], True, None), ([2], False, None)])
def test_one_retry_and_no_scientific_or_refusal_retry(tmp_path, monkeypatch, codes, scientific, expected):
    from tools.loopresume import c1collect4_box as driver
    monkeypatch.setattr(driver, 'RELEASE', tmp_path)
    monkeypatch.setattr(driver, 'ARM', tmp_path)
    remaining = list(codes)
    ph = {'status': 'pending', 'error': 'transport failure', 'refused': 'budget'}
    state = {'cycles': [{'phases': {'convert': ph}}], 'sim_s_charged': 176473}
    calls = []

    def run(cfg, name):
        calls.append(name)
        code = remaining.pop(0)
        ph['status'] = 'failed' if code == 3 else 'pending'
        return code

    fake = SimpleNamespace(load_state=lambda cfg: state, run_phase=run, retry=run,
                           now=lambda: 'test', append=lambda *a: None)
    resume = SimpleNamespace(scientific=lambda *a: scientific, OperationalPaused=RuntimeError)
    if expected is None:
        with pytest.raises(RuntimeError, match='no eligible operational retry'):
            phase_once(fake, resume, {}, 'convert')
    else:
        assert phase_once(fake, resume, {}, 'convert') == expected
    assert len(calls) == len(codes)
