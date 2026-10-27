"""Scientific boundary and budget authorization regressions for S-C2H60."""
import pytest

from tools.loopresume.c2h60_box import budget_authorization, verify_updates


def row(update, horizon=60):
    return dict(update=update, imagination_horizon_s=horizon, imagination_horizon_stage=2,
                actor_loss=1., critic_loss=2., actor_grad_norm=3., critic_grad_norm=4.)


def test_150_new_updates_not_superseded_work():
    verify_updates([row(i, 10 if i % 7 == 0 else 60) for i in range(234, 384)], 233)
    with pytest.raises(ValueError, match='count/sequence'):
        verify_updates([row(i) for i in range(234, 383)], 233)
    with pytest.raises(ValueError, match='count/sequence'):
        verify_updates([row(i) for i in range(233, 383)], 233)


@pytest.mark.parametrize('key', ['actor_loss', 'critic_loss', 'actor_grad_norm', 'critic_grad_norm'])
def test_nonfinite_refuses(key):
    bad = row(234)
    bad[key] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        verify_updates([bad], 233, 1)


def test_30s_fallback_and_unexercised60_refuse():
    with pytest.raises(ValueError, match='not active'):
        verify_updates([row(234, 30)], 233, 1)
    with pytest.raises(ValueError, match='never exercised'):
        verify_updates([row(234, 10)], 233, 1)


def test_only_owner_registered_raise_can_dispatch():
    approval = dict(id='S-C2H60_OWNER_BUDGET_RAISE', owner_approved=True,
                    sim_s_total=213406, markdown_registration_sha256='exact-owner-register-sha',
                    collection_authorized=True)
    reg = dict(budget={'sim_s_total': 210000}, amendments=[])
    assert budget_authorization(reg, 176473, 36933) is None
    reg['budget']['sim_s_total'] = 213406
    assert budget_authorization(reg, 176473, 36933) is None
    reg['amendments'].append(approval)
    assert budget_authorization(reg, 176473, 36933) == approval
    for key, value in [('owner_approved', False), ('markdown_registration_sha256', ''),
                       ('collection_authorized', False), ('sim_s_total', 213405)]:
        reg['amendments'] = [{**approval, key: value}]
        assert budget_authorization(reg, 176473, 36933) is None
