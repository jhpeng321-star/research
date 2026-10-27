"""No simulation: MAIN's collection authorization must not cross WM-FT boundary."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def coordinator(monkeypatch, tmp_path):
    if __import__('os').name != 'posix':
        pytest.skip('gpuA coordinator uses POSIX chain locks')
    path = Path(__file__).resolve().parents[2] / 'tools/longint/resume_c3.py'
    spec = importlib.util.spec_from_file_location('c3resume_boundary', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, 'ARM', tmp_path)
    (tmp_path / 'loop').mkdir()
    (tmp_path / 'loop/STOP_DRIVER').touch()
    phases = {name: {'status': 'pending', 'receipt': {}}
              for name in ('policy', 'collect', 'convert', 'wm_ft', 'check')}
    phases['policy']['status'] = 'failed'
    state = {'cycles': [{'cycle': 3, 'phases': phases}], 'sim_s_charged': 213406,
             'policy': {'checkpoint': 'selected.pt'}}
    calls, statuses = [], []

    def retry(cfg, phase):
        calls.append('retry:' + phase)
        phases[phase]['status'] = 'done'
        return 0

    def run_phase(cfg, phase):
        calls.append(phase)
        phases[phase]['status'] = 'done'
        if phase == 'collect':
            state['sim_s_charged'] += 36933
        return 0

    loop = SimpleNamespace(load_config=lambda p: {}, load_state=lambda c: state,
                           run_cmd=None, retry=retry, run_phase=run_phase)
    monkeypatch.setattr(mod, 'setup', lambda: loop)
    monkeypatch.setattr(mod, 'locks', lambda: [object()])
    monkeypatch.setattr(mod, 'status', lambda l, s, **kw: statuses.append(s))
    monkeypatch.setattr(mod, 'configure_collect', lambda l: calls.append('install_collector'))
    monkeypatch.setattr(mod, 'verify_policy', lambda l, c: calls.append('verify'))
    return mod, loop, calls, statuses, state


def test_success_stops_before_wm_finetune(coordinator):
    mod, _, calls, statuses, state = coordinator
    assert mod.drive() == 0
    assert calls == ['retry:policy', 'verify', 'install_collector', 'collect', 'convert']
    assert statuses[-1] == 'WAIT_MAIN_C3_WMFT'
    assert state['sim_s_charged'] == 250339
    assert state['cycles'][-1]['phases']['wm_ft']['status'] == 'pending'
    assert (mod.ARM / 'loop/STOP_DRIVER').exists()


def test_failed_policy_never_installs_collector(coordinator):
    mod, loop, calls, statuses, _ = coordinator
    loop.retry = lambda c, p: 3
    assert mod.drive() == 3
    assert calls == []
    assert statuses[-1] == 'C3_POLICY_STOPPED'


def test_failed_verification_cannot_charge_collection(coordinator, monkeypatch):
    mod, _, calls, statuses, state = coordinator

    def fail(loop, cfg):
        raise AssertionError('nonfinite update or memory cap violation')

    monkeypatch.setattr(mod, 'verify_policy', fail)
    with pytest.raises(AssertionError):
        mod.drive()
    assert calls == ['retry:policy']
    assert state['sim_s_charged'] == 213406
    assert statuses[-1] == 'OPERATIONAL_PAUSED'
