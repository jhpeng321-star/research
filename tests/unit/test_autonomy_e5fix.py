"""Retained charges, one repair per failed world, and incomplete-panel fail-closed."""
import json

import pytest

from tools.autonomy_e5 import fix_runner as F
from tools.autonomy_e5 import remote_fix as C


def test_failed_reservations_consume_six_worlds_of_the_fixed_cap():
    original = [dict(id=jid, sim_s=900) for jid in F.FAILED]
    repairs = [dict(id=F.attempt(dict(id=jid)), sim_s=900) for jid in F.FAILED]
    assert F.ledger_total(original+repairs) == 10800
    assert (F.CAP-F.ledger_total(original))//900 == 108
    with pytest.raises(AssertionError, match='duplicate'):
        F.ledger_total(original+repairs+repairs[:1])
    with pytest.raises(AssertionError):
        F.ledger_total([dict(id=str(i), sim_s=900) for i in range(115)])


def test_canary_reused_and_budget_stop_never_analyzes_partial_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'R', tmp_path)
    monkeypatch.setattr(C.B, 'R', tmp_path)
    events = []
    monkeypatch.setattr(C, 'start', lambda mode: events.append(mode) or {'pid': 1})
    monkeypatch.setattr(C, 'cpu', lambda box, source: {'passed': True, 'seed': 973001})
    monkeypatch.setattr(C.B, 'analyze', lambda: pytest.fail('partial-panel analysis'))
    missing = set(range(973216, 973228, 2))
    doc = dict(charged_sim_s=205200, boxes={
        'cpuP': dict(state=dict(status='complete'), seeds=[s for s in range(973000, 973228, 2) if s not in missing]),
        'cpuQ': dict(state=dict(status='WAIT_OWNER_BUDGET'), seeds=list(range(973001, 973228, 2)))})
    calls = []
    def cycle(phase, refresh_p=True):
        calls.append((phase, refresh_p))
        return dict(boxes={'cpuQ': dict(state=dict(status='canary_complete'))}) if phase == 'CPUQ_CANARY' else doc
    monkeypatch.setattr(C, 'cycle', cycle)
    C.drive()
    result = json.loads((tmp_path/'FIX_STATUS.json').read_text())
    assert events == ['canary', 'remainder']
    assert calls == [('CPUQ_CANARY', False), ('RUNNING', True)]
    assert result['status'] == 'WAIT_OWNER_BUDGET'
    assert result['additional_sim_s_required'] == 5400 and set(result['missing_worlds']) == missing
    assert not (tmp_path/'RESULTS.json').exists()


def test_failed_repair_canary_does_not_release_cpuP_or_remainder(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'R', tmp_path)
    monkeypatch.setattr(C.B, 'R', tmp_path)
    starts = []
    monkeypatch.setattr(C, 'start', lambda mode: starts.append(mode) or {})
    def fail(phase, refresh_p=True):
        assert not refresh_p
        raise RuntimeError('canary failed')
    monkeypatch.setattr(C, 'cycle', fail)
    with pytest.raises(RuntimeError, match='canary failed'):
        C.drive()
    assert starts == ['canary']
    assert json.loads((tmp_path/'FIX_STATUS.json').read_text())['status'] == 'WAIT_OWNER'
