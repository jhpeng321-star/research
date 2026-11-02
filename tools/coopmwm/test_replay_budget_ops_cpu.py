import copy
import pytest
from .replay_budget_ops import reconcile, CAP


def fixture():
    b = dict(total=CAP, allocated=CAP, phases=dict(superseded=811640,
        train=2864400, bank_verification_ext=2600000, final_test=691200,
        paired_branch=1297400, interim_u8_test=900000, sumo_verification_k8=1500000,
        other=3835360))
    return b, dict(train=1798800, bank_verification_ext=215280)


def test_protected_balances_never_cover_the_training_shortfall():
    b, charges = fixture(); original = copy.deepcopy(b)
    v = reconcile(b, charges, 811640)
    assert b == original
    assert v['remaining_PPO']['required_sim_s'] == 1209600
    assert v['remaining_PPO']['train_phase_shortfall_sim_s'] == 144000
    assert not v['funding_complete']
    assert v['phases']['sumo_verification_k8']['remaining'] == 1500000
    assert v['new_reserved_sim_s'] == 0
    assert v['total_charged_sim_s'] == 811640+1798800+215280


@pytest.mark.parametrize('fault', ['cap', 'double_count', 'phase_overrun', 'unknown'])
def test_reconciliation_rejects_invalid_accounting(fault):
    b, charges = fixture()
    if fault == 'cap':b['total'] = 15000000
    if fault == 'double_count':charges['superseded'] = 811640
    if fault == 'phase_overrun':charges['train'] = b['phases']['train']+1
    if fault == 'unknown':charges['unregistered'] = 1
    with pytest.raises(AssertionError):reconcile(b, charges, 811640)


def test_authorized_transfer_funds_ppo_without_touching_restricted_phases(monkeypatch,tmp_path):
    import json
    from . import replay_budget_ops as bo
    b,charges=fixture()
    b['phases'].update(checks=432000,sumo_verification=2800000,other=603360)
    charges['checks']=172800
    parent=reconcile(b,charges,811640)
    (tmp_path/bo.NAME).write_text(json.dumps(parent))
    registry=tmp_path/'REGISTRATION.md';registry.write_text('')
    monkeypatch.setattr(bo.ops,'EVIDENCE',tmp_path)
    monkeypatch.setattr(bo,'REGISTRY',registry)
    calls=[]
    def remote(host,code,**kwargs):
        compile(code,'<append-only-transfer>','exec');calls.append(code)
        if len(calls)==1:return json.dumps(dict(charges=charges,superseded=811640))
        assert "B_REPLAY_PHASE_TRANSFER_0706.json" in code
        assert "open('x')" in code
        return '{}'
    monkeypatch.setattr(bo,'remote',remote)
    bo.transfer()
    v=json.loads((tmp_path/'B_REPLAY_PHASE_TRANSFER_0706.json').read_text())
    assert v['transfer']==dict(checks_to_train=144000,sumo_verification_to_train=0,total=144000)
    assert v['funding_complete'] and v['phases']['train']['remaining']==1209600
    assert v['budget']['phases']['checks']==288000 and v['budget']['total']==CAP
    for p in ('bank_verification_ext','final_test','paired_branch','interim_u8_test','sumo_verification_k8'):
        assert v['budget']['phases'][p]==b['phases'][p]
    assert json.loads((tmp_path/bo.NAME).read_text())==parent
