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
