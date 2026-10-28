"""No-inference deployment-repair eligibility and hard phase-boundary checks."""
import copy
import sys
import types

import pytest


@pytest.fixture
def repair(monkeypatch, tmp_path):
    if sys.platform == 'win32':
        monkeypatch.setitem(sys.modules, 'fcntl', types.SimpleNamespace())
    from tools.longint import wmft_check_repair as mod
    arm = tmp_path / 'arm'
    (arm / 'loop').mkdir(parents=True)
    (arm / 'loop/STOP_DRIVER').touch()
    monkeypatch.setattr(mod.core, 'ARM', arm)
    monkeypatch.setattr(mod, 'OUT', tmp_path)
    receipt = {'exit': 2, 'decision': 'REJECT', 'reasons': ['measurement error (fail closed)'],
               'error': f"FileNotFoundError: [Errno 2] No such file or directory: '{mod.core.ROOT / mod.MISSING}'"}
    phases = {'wm_ft': {'status': 'done', 'attempt': 2,
                       'receipt': {'candidate_sha256': mod.WM3, 'exit': 0}},
              'check': {'status': 'done', 'attempt': 1, 'receipt': receipt}}
    state = {'cycles': [{'cycle': 3, 'phases': phases}], 'sim_s_charged': 264544,
             'accepted_wm': {'sha256': mod.core.WM2}, 'consecutive_rejects': 1,
             'stopped': 'registered SUMO budget cannot fund another collection'}
    before = {'consecutive_rejects': 0}
    return mod, state, before


@pytest.mark.parametrize('change', ['scientific_reject', 'wrong_error', 'third_check',
                                  'other_cycle', 'changed_budget', 'extra_reject', 'changed_wm'])
def test_refuses_any_unregistered_repair(repair, change):
    mod, state, before = repair
    check = state['cycles'][-1]['phases']['check']
    if change == 'scientific_reject':
        check['receipt'].update(exit=3, reasons=['one-step regression'])
    elif change == 'wrong_error':
        check['receipt']['error'] = 'out of memory'
    elif change == 'third_check':
        check['attempt'] = 2
    elif change == 'other_cycle':
        state['cycles'][-1]['cycle'] = 4
    elif change == 'changed_budget':
        state['sim_s_charged'] += 1
    elif change == 'extra_reject':
        state['consecutive_rejects'] += 1
    else:
        state['accepted_wm']['sha256'] = mod.WM3
    with pytest.raises(AssertionError):
        mod.eligible(state, before)


@pytest.mark.parametrize('decision', ['ACCEPT', 'REJECT'])
def test_check_only_preserves_stops_history_and_exact_ft(repair, monkeypatch, decision, tmp_path):
    mod, state, before = repair
    original_ft = copy.deepcopy(state['cycles'][-1]['phases']['wm_ft'])
    original_check = copy.deepcopy(state['cycles'][-1]['phases']['check'])
    calls, statuses, writes = [], [], {}

    def run_check(cfg, current, ph, root):
        calls.append('check')
        assert ph['attempt'] == 2 and current['consecutive_rejects'] == 0
        assert current['stopped'] == 'registered SUMO budget cannot fund another collection'
        if decision == 'ACCEPT':
            current['accepted_wm'] = {'sha256': mod.WM3}
        else:
            current['consecutive_rejects'] += 1
        return {'decision': decision, 'out': str(tmp_path / 'check2')}

    loop = types.SimpleNamespace(
        read_json=lambda path: {'state': before}, now=lambda: 'now',
        fast_module=lambda: types.SimpleNamespace(process_token=lambda pid: 'token'),
        save_state=lambda cfg, current: None, run_check=run_check,
        write_atomic=lambda path, doc: writes.update({path.name: doc}))
    monkeypatch.setattr(mod.core, 'status', lambda owner, status, **kw: statuses.append(status))
    monkeypatch.setattr(mod.core, 'event', lambda *args, **kw: None)
    authority = {'prior_consecutive_rejects': 0, 'budget_stop_preserved': state['stopped']}
    mod.complete_check(loop, {'loop_root': str(tmp_path)}, state, authority)
    assert calls == ['check']
    assert statuses[-1] == 'WAIT_MAIN_C4'
    assert state['cycles'][-1]['cycle'] == 3 and state['sim_s_charged'] == 264544
    assert state['cycles'][-1]['phases']['wm_ft'] == original_ft
    assert state['cycles'][-1]['phases']['check']['previous_operational_attempt'] == original_check
    assert writes['RESULTS.json']['acceptance']['decision'] == decision
    assert writes['RESULTS.json']['budget_stop'] == authority['budget_stop_preserved']
