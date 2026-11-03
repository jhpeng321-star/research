"""Regression tests for the six run_separate / bank_statistics review defects (c4/imit-fix2)."""
import copy
import math
import types
import pytest
import torch
from . import bank_imitation as bi
from . import exact_grade
from .bank_statistics import check_keys, install
from .test_bank_imitation_cpu import Policy, pairs


def episode():
    step = dict(sample=dict(context={'members': ['a', 'b']}, value=0., behaviour=dict(executed=True)), reward=0.)
    return dict(steps=[copy.deepcopy(step) for _ in range(3)])


def unexecuted(n):
    sample = dict(context={'members': ['a', 'b']}, action=[[.1], [.1]],
                  behaviour=dict(executed=False, proposal_members=['a', 'b']))
    return [(copy.deepcopy(sample), 1.) for _ in range(n)]


def warm(policy, lr):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss = torch.stack([bi.imitation(policy, s, w) for s, w in pairs(3)]).mean()
    opt.zero_grad();loss.backward();opt.step()
    return opt


def same_adam(a, b):
    assert a['state'].keys() == b['state'].keys() and a['state']
    for k in a['state']:
        for name in ('exp_avg', 'exp_avg_sq', 'step'):
            assert torch.equal(torch.as_tensor(a['state'][k][name]), torch.as_tensor(b['state'][k][name])), (k, name)


def same_params(p, state):
    return all(torch.equal(v, state[k]) for k, v in p.state_dict().items())


# 1 ---------------------------------------------------------------- exact_update call path
@pytest.mark.parametrize('bank', ['full', 'empty'])
def test_exact_update_with_run_separate_has_loss_and_gradient_keys(monkeypatch, bank):
    monkeypatch.setattr(bi, 'run', bi.run_separate)
    p = Policy();opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    bank_pairs = pairs(10) if bank == 'full' else unexecuted(4)
    metrics = exact_grade.exact_update(p, opt, [episode()], torch.Generator().manual_seed(3), bank_pairs,
                                       imitation_epochs=5)
    last = metrics[-1];report = last['bank_imitation']
    assert last['loss'] == report['last_loss'] and last['gradient_norm'] == report['last_gradient_norm']
    assert report['kl_reference'].startswith('frozen pre-imitation')
    if bank == 'full':
        assert report['epochs_completed'] >= 1 and report['last_gradient_norm'] > 0
        assert math.isfinite(report['last_loss'])
    else:
        assert report['pairs_total'] == report['pairs_seen'] == 0
        assert report['last_loss'] == report['last_gradient_norm'] == 0.
        assert report['stop_reason'] == 'empty statistically qualified bank'


# 2 ---------------------------------------------------------------- Adam rollback is a true copy
def test_backtrack_restores_exact_adam_snapshot(monkeypatch):
    p = Policy();opt = warm(p, 1.)
    with torch.no_grad():p.mean.fill_(1.)
    snapshot = copy.deepcopy(opt.state_dict());restored = []
    original = opt.load_state_dict

    def load(sd):
        original(sd);restored.append(copy.deepcopy(opt.state_dict()))
    monkeypatch.setattr(opt, 'load_state_dict', load)
    out = bi.run_separate(p, opt, pairs(257), torch.Generator().manual_seed(7), copy.deepcopy(p))
    pass1_restores = out['epochs'][0]['attempts']-1
    assert pass1_restores >= 2
    for sd in restored[:pass1_restores]:
        same_adam(sd, snapshot)


# 3 ---------------------------------------------------------------- per-epoch progress records and floor
def test_epoch_records_and_pass1_effectiveness():
    old = Policy();p = copy.deepcopy(old)
    with torch.no_grad():p.mean.fill_(1.)
    out = bi.run_separate(p, torch.optim.Adam(p.parameters(), lr=.05), pairs(20),
                          torch.Generator().manual_seed(7), old)
    assert sum(r['accepted'] for r in out['epochs']) == out['epochs_completed'] >= 1
    for row in [r for r in out['epochs'] if r['accepted']]:
        for key in ('step_scale', 'attempts', 'kl', 'awr_loss_before', 'awr_loss_after',
                    'l1_action_error_before', 'l1_action_error_after', 'rejected_optimizer_steps'):
            assert key in row
    first = out['epochs'][0]
    assert out['pass1_step_scale'] == first['step_scale']
    assert out['pass1_loss_drop'] == pytest.approx(first['awr_loss_before']-first['awr_loss_after'])
    assert out['pass1_effective'] is (out['pass1_loss_drop'] > 0)
    assert out['rejected_optimizer_steps'] == sum(r['rejected_optimizer_steps'] for r in out['epochs'])


def test_pass1_below_floor_stops_additional_passes():
    p = Policy();opt = torch.optim.Adam(p.parameters(), lr=50.)
    with torch.no_grad():p.mean.fill_(1.)
    out = bi.run_separate(p, opt, pairs(20), torch.Generator().manual_seed(7), copy.deepcopy(p))
    assert out['pass1_step_scale'] < bi.MIN_ADDITIONAL_PASS_SCALE
    assert out['epochs_completed'] == 1 and 'additional passes stopped' in out['stop_reason']
    assert out['kl'] <= .045


def test_later_pass_never_goes_below_one_64th(monkeypatch):
    calls = []

    def kl(policy, reference, samples):
        calls.append(1);return .01 if len(calls) == 1 else 1.
    monkeypatch.setattr(bi, 'joint_kl', kl)
    p = Policy();opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    out = bi.run_separate(p, opt, pairs(30), torch.Generator().manual_seed(1), copy.deepcopy(p))
    second = out['epochs'][1]
    assert out['epochs_completed'] == 1 and not second['accepted']
    assert second['attempts'] == 7 and second['smallest_step_scale_tried'] == 1/64
    assert 'additional passes stopped' in out['stop_reason']
    assert opt.param_groups[0]['lr'] == 1e-3


# 4 ---------------------------------------------------------------- non-finite attempts are rejected
def test_all_nonfinite_pass1_returns_untouched_policy(monkeypatch):
    p = Policy();opt = warm(p, 1e-2)
    params = copy.deepcopy(p.state_dict());adam = copy.deepcopy(opt.state_dict())
    monkeypatch.setattr(bi, 'imitation', lambda policy, s, w: policy.mean*float('nan'))
    out = bi.run_separate(p, opt, pairs(300), torch.Generator().manual_seed(7), copy.deepcopy(p))
    assert out['imitation_failed'] and out['epochs_completed'] == 0 and 'non-finite' in out['failure_reason']
    assert set(out['epochs'][0]['attempt_status']) == {'nonfinite_loss'}
    assert same_params(p, params);same_adam(opt.state_dict(), adam)
    assert 'last_loss' in out and 'last_gradient_norm' in out


def test_midpass_nonfinite_is_rolled_back_not_raised(monkeypatch):
    real, calls = bi.imitation, []

    def flaky(policy, s, w):   # second minibatch of the first attempt is non-finite
        if torch.is_grad_enabled():calls.append(1)   # count training minibatch terms only
        value = real(policy, s, w)
        return value*float('nan') if 128 < len(calls) <= 256 else value
    monkeypatch.setattr(bi, 'imitation', flaky)
    p = Policy();opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    with torch.no_grad():p.mean.fill_(1.)
    gen = torch.Generator().manual_seed(7)
    out = bi.run_separate(p, opt, pairs(300), gen, copy.deepcopy(p))
    first = out['epochs'][0]
    assert first['attempt_status'][0] == 'nonfinite_loss' and first['accepted']
    assert first['rejected_optimizer_steps'] == 1 and out['backtracks'] >= 1
    assert all(torch.isfinite(v).all() for v in p.state_dict().values())


def test_nonfinite_kl_is_a_rejected_attempt(monkeypatch):
    real, calls = bi.joint_kl, []

    def kl(policy, reference, samples):
        calls.append(1)
        if len(calls) == 1:raise ValueError('nonfinite full-bank KL')
        return real(policy, reference, samples)
    monkeypatch.setattr(bi, 'joint_kl', kl)
    p = Policy();opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    out = bi.run_separate(p, opt, pairs(30), torch.Generator().manual_seed(1), copy.deepcopy(p))
    assert out['epochs'][0]['attempt_status'][:2] == ['nonfinite_kl', 'accepted']
    assert out['epochs_completed'] >= 1 and not out['imitation_failed']


# 5 ---------------------------------------------------------------- executed=False pairs are not trainable
def test_unexecuted_pairs_are_filtered_before_counting():
    old = Policy();p = copy.deepcopy(old)
    out = bi.run_separate(p, torch.optim.Adam(p.parameters(), lr=1e-3), pairs(20)+unexecuted(9),
                          torch.Generator().manual_seed(7), old)
    assert out['pairs_total'] == 20 and out['pairs_filtered_unexecuted'] == 9
    assert out['pairs_seen'] == 20*out['epochs_completed'] and out['member_actions_seen'] == 40*out['epochs_completed']


# 6 ---------------------------------------------------------------- statistics schema guard
def record(**over):
    c = dict(admitted=True, first_seed_advantage_veh_s=10., replication_advantage_veh_s=11.,
             checks=dict(replication=True), extended_advantage_veh_s=5.,
             cooperation=dict(complete=True, cooperation_gain_veh_s=3., replication_cooperation_gain_veh_s=3.1))
    c.update(over);return c


@pytest.mark.parametrize('drop', ['first_seed_advantage_veh_s', 'replication_advantage_veh_s',
                                  'replication_cooperation_gain_veh_s'])
def test_key_missing_from_all_records_raises(drop):
    rows = [record(), record()]
    for c in rows:
        (c['cooperation'] if drop.startswith('replication_coop') else c).pop(drop)
    with pytest.raises(KeyError, match=drop):check_keys(rows)
    rb = types.SimpleNamespace(confirmations=lambda root, arm: dict(enumerate(rows)))
    install(rb,bank_rule='t1727')
    with pytest.raises(KeyError, match=drop):rb.confirmations('r', 'A')


def test_partial_or_inapplicable_absence_is_allowed():
    failed = dict(admitted=False, first_seed_advantage_veh_s=4., checks=dict(confirmation_branches_complete=False))
    single = record(cooperation=dict(complete=True, cooperation_gain_veh_s=None, cooperative=False))
    partial = record();partial['cooperation'].pop('replication_cooperation_gain_veh_s')
    check_keys([failed, single, partial, record()])
    check_keys([failed, single]);check_keys([])
    rb = types.SimpleNamespace(confirmations=lambda root, arm: {'x': record()})
    install(rb,bank_rule='t1727')
    assert rb.confirmations('r', 'A')['x']['cooperation']['cooperative']
