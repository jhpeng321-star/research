"""No-inference checks for the authorized C3 WM-only dispatch boundary."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def harness(tmp_path, monkeypatch):
    if sys.platform == 'win32':
        monkeypatch.setitem(sys.modules, 'fcntl', types.SimpleNamespace())
    path = Path(__file__).parents[2] / 'tools/longint/wmft_c3.py'
    spec = importlib.util.spec_from_file_location('c3wmft_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    arm = tmp_path / 'arm'
    (arm / 'loop').mkdir(parents=True)
    (arm / 'loop/STOP_DRIVER').touch()
    out = tmp_path / 'out'
    out.mkdir()
    calls, writes, statuses, events = [], {}, [], []
    phases = {'wm_ft': {'receipt': {'candidate_sha256': 'wm3'}},
              'check': {'receipt': {'out': str(out), 'decision': 'ACCEPT'}}}
    state = {'cycles': [{'cycle': 3, 'phases': phases}], 'sim_s_charged': 264544,
             'accepted_wm': {'sha256': 'wm3'}, 'stopped': 'registered budget stop'}
    codes = {'wm_ft': 0, 'check': 0}

    def phase(cfg, name):
        calls.append(name)
        return codes[name]

    def retry(cfg, name):
        calls.append('retry:' + name)
        return codes.get('retry', 0)

    loop = types.SimpleNamespace(
        load_config=lambda path: {}, read_json=lambda path: {'config_sha256': 'hash'},
        load_state=lambda cfg: state, run_phase=phase, retry=retry, now=lambda: 'now',
        write_atomic=lambda path, doc: writes.update({Path(path).name: doc}))
    for key, value in {'ARM': arm, 'OUT': out, 'setup': lambda: loop,
                       'locks': lambda: [True], 'sha': lambda path: 'hash',
                       'install': lambda owner: None,
                       'status': lambda owner, value, **kw: statuses.append(value),
                       'event': lambda owner, value, **kw: events.append(value)}.items():
        monkeypatch.setattr(mod, key, value)
    monkeypatch.setattr(mod.time, 'sleep', lambda seconds: None)
    return mod, phases, codes, calls, writes, statuses, events


def test_accept_stops_before_any_cycle4_dispatch(harness):
    mod, _, _, calls, writes, statuses, _ = harness
    mod.run()
    assert calls == ['wm_ft', 'check']
    assert statuses[-1] == 'WAIT_MAIN_C4'
    assert writes['RESULTS.json']['charged'] == 264544
    assert writes['RESULTS.json']['budget_stop'] == 'registered budget stop'


def test_reject_stops_without_retry(harness):
    mod, phases, codes, calls, _, statuses, _ = harness
    codes['check'] = 4
    phases['check']['receipt']['decision'] = 'REJECT'
    mod.run()
    assert calls == ['wm_ft', 'check']
    assert statuses[-1] == 'WAIT_MAIN_C4'


def test_only_one_operational_ft_retry(harness):
    mod, phases, codes, calls, writes, statuses, events = harness
    codes.update(wm_ft=3, retry=3)
    phases['wm_ft']['error'] = 'RuntimeError: CUDA out of memory'
    with pytest.raises(AssertionError, match='fine-tune exit 3'):
        mod.run()
    assert calls == ['wm_ft', 'retry:wm_ft']
    assert 'FAILED_FT_ATTEMPT1.json' in writes
    assert events == ['OPERATIONAL_FT_RETRY']
    assert statuses[-1] == 'C3_WMFT_OPERATIONAL_PAUSED'


def test_nonfinite_is_not_retried(harness):
    mod, phases, codes, calls, _, _, _ = harness
    codes['wm_ft'] = 3
    phases['wm_ft']['error'] = 'ValueError: nonfinite scientific training metrics'
    with pytest.raises(RuntimeError, match='no retry'):
        mod.run()
    assert calls == ['wm_ft']


def test_undefined_diagnostic_does_not_invalidate_finite_optimization(harness):
    mod = harness[0]
    assert mod.training_finite({'loss': 265.5, 'grad_norm': 25.83, 'finite': True,
                               'merge_pivot_on_diagnostic_nll': float('inf')})
    assert not mod.training_finite({'loss': float('nan'), 'grad_norm': 25.83, 'finite': True})
    assert not mod.training_finite({'loss': 265.5, 'grad_norm': float('inf'), 'finite': True})
    assert not mod.training_finite({'loss': 265.5, 'grad_norm': 25.83, 'finite': False})


def test_replacement_cannot_trigger_a_third_attempt(harness, monkeypatch):
    mod, phases, codes, calls, _, statuses, events = harness
    monkeypatch.setattr(mod, 'RETRY', True)
    codes['retry'] = 3
    phases['wm_ft']['error'] = 'RuntimeError: operational failure again'
    with pytest.raises(AssertionError, match='fine-tune exit 3'):
        mod.run()
    assert calls == ['retry:wm_ft']
    assert events == []
    assert statuses[-1] == 'C3_WMFT_OPERATIONAL_PAUSED'
