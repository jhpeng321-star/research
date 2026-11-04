"""CPU tests for the imitation-strength feature (c4/imit-strength).

imitation_step_mode: default 'fixed' is byte-identical; 'kl_targeted' doubles the next pass's
step scale (cap 16x) and optimizer steps (cap 4x) only while the accepted pass ends below half
the registered KL limit, keeping the halving backtracking and never exceeding the limit.
v2v_intent_bank_term (default True = current) can drop the bank intent term; the reported bank
intent loss also covers the TRAINED subset (w>0, trainable pairs). Runtime keys are pinned in
BANK_OPTIONS/bind_runtime, one key for B and R (mismatch refuses), like imitation_bank_filter.
"""
import copy
import json

import pytest
import torch

from tools.coopmwm import bank_imitation as bi
from tools.coopmwm import bank_statistics as bs
from tools.coopmwm import exact_grade as xg
from tools.coopmwm import v2v_intent as vi
from tools.coopmwm import v2v_launch3 as launch
from tools.coopmwm import v2v_runner
from tools.coopmwm.test_bank_imitation_cpu import Policy, pairs


@pytest.fixture(autouse=True)
def restore_imitation_runtime():
    old = (bi.RUNTIME_STEP_MODE, bi.RUNTIME_INTENT_BANK_TERM)
    yield
    bi.install(*old)


# ---------------------------------------------------------------- fixed mode is byte-identical

def test_fixed_mode_default_and_explicit_are_identical():
    a, b = Policy(), Policy()
    ra = bi.run_separate(a, torch.optim.Adam(a.parameters(), lr=1e-3), pairs(40),
                         torch.Generator().manual_seed(7), copy.deepcopy(a), passes=5)
    rb = bi.run_separate(b, torch.optim.Adam(b.parameters(), lr=1e-3), pairs(40),
                         torch.Generator().manual_seed(7), copy.deepcopy(b), passes=5,
                         step_mode='fixed')
    assert json.dumps(ra, sort_keys=True) == json.dumps(rb, sort_keys=True)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    # the fixed report carries none of the kl_targeted additions
    assert 'step_mode' not in ra and 'step_scale_cap' not in ra and 'intent_bank_term' not in ra
    assert all('step_scale_mult' not in row and 'optimizer_steps' not in row for row in ra['epochs'])
    assert all(r['step_scale'] == r['smallest_step_scale_tried'] == 1. for r in ra['epochs'])


def test_explicit_fixed_overrides_an_installed_targeted_default():
    bi.install(step_mode='kl_targeted')
    a, b = Policy(), Policy()
    forced = bi.run_separate(a, torch.optim.Adam(a.parameters(), lr=1e-3), pairs(40),
                             torch.Generator().manual_seed(7), copy.deepcopy(a), passes=5,
                             step_mode='fixed')
    installed = bi.run_separate(b, torch.optim.Adam(b.parameters(), lr=1e-3), pairs(40),
                                torch.Generator().manual_seed(7), copy.deepcopy(b), passes=5)
    assert 'step_mode' not in forced and installed['step_mode'] == 'kl_targeted'
    assert installed['epochs'][1]['step_scale_mult'] == 2.   # pass 2 runs at the doubled scale


def test_install_validates_and_restores_module_defaults():
    with pytest.raises(ValueError, match='imitation_step_mode'):
        bi.install(step_mode='bogus')
    with pytest.raises(ValueError, match='v2v_intent_bank_term'):
        bi.install(intent_bank_term=1)
    assert (bi.RUNTIME_STEP_MODE, bi.RUNTIME_INTENT_BANK_TERM) == ('fixed', True)
    bi.install('kl_targeted', False)
    assert (bi.RUNTIME_STEP_MODE, bi.RUNTIME_INTENT_BANK_TERM) == ('kl_targeted', False)


# ------------------------------------------- kl_targeted grows while KL stays under the limit

def test_kl_targeted_grows_when_kl_low_and_never_exceeds_the_limit():
    old = Policy()
    p = copy.deepcopy(old)
    out = bi.run_separate(p, torch.optim.Adam(p.parameters(), lr=1e-4), pairs(300),
                          torch.Generator().manual_seed(7), old, passes=5,
                          step_mode='kl_targeted')
    assert out['epochs_completed'] == 5
    assert out['stop_reason'] == 'all registered imitation epochs complete'
    assert out['step_mode'] == 'kl_targeted' and out['step_scale_cap'] == 16 and out['steps_cap'] == 4
    assert [r['step_scale_mult'] for r in out['epochs']] == [1., 2., 4., 8., 16.]
    assert [r['step_scale'] for r in out['epochs']] == [1., 2., 4., 8., 16.]
    assert [r['steps_mult'] for r in out['epochs']] == [1, 2, 4, 4, 4]
    assert [r['optimizer_steps'] for r in out['epochs']] == [3, 5, 10, 10, 10]
    assert out['optimizer_steps'] == sum(r['optimizer_steps'] for r in out['epochs'])
    assert all(r['accepted'] and r['kl'] <= .045 for r in out['epochs'])
    assert out['kl'] <= .045 and out['action_error_after_mps2'] < out['action_error_before_mps2']
    assert out['backtracks'] == 0


def test_kl_targeted_reduces_action_error_more_than_fixed_on_a_synthetic_bank():
    def bank(n):
        sample = dict(context={'members': ['a', 'b']}, action=[[.9], [.9]],
                      behaviour=dict(executed=True, proposal_members=['a', 'b']))
        return [(copy.deepcopy(sample), 1.) for _ in range(n)]
    fixed, targ = Policy(), Policy()
    rf = bi.run_separate(fixed, torch.optim.Adam(fixed.parameters(), lr=1e-4), bank(300),
                         torch.Generator().manual_seed(7), copy.deepcopy(fixed), passes=5)
    rt = bi.run_separate(targ, torch.optim.Adam(targ.parameters(), lr=1e-4), bank(300),
                         torch.Generator().manual_seed(7), copy.deepcopy(targ), passes=5,
                         step_mode='kl_targeted')
    assert rf['kl'] <= .045 and rt['kl'] <= .045
    assert rf['action_error_after_mps2'] < rf['action_error_before_mps2']
    assert rt['action_error_after_mps2'] < rf['action_error_after_mps2']


# ------------------------------------------------------- backtracking and the KL bound hold

def test_kl_targeted_backtracking_halves_and_growth_resumes_after_a_low_accept(monkeypatch):
    scripted = [.01, 1.0, .03, .01, .01, .01]   # one joint_kl per attempt; later calls repeat .01

    def kl(policy, reference, samples):
        return scripted.pop(0) if scripted else .01
    monkeypatch.setattr(bi, 'joint_kl', kl)
    p = Policy()
    opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    out = bi.run_separate(p, opt, pairs(20), torch.Generator().manual_seed(1), copy.deepcopy(p),
                          passes=5, step_mode='kl_targeted')
    rows = out['epochs']
    assert out['epochs_completed'] == 5 and out['kl'] <= .045
    assert [r['step_scale_mult'] for r in rows] == [1., 2., 2., 4., 8.]
    assert [r['step_scale'] for r in rows] == [1., 1., 2., 4., 8.]
    assert rows[1]['attempt_status'] == ['kl_exceeded', 'accepted']
    assert rows[1]['attempts'] == 2 and out['backtracks'] == 1
    assert all(r['kl'] <= .045 for r in rows)
    assert opt.param_groups[0]['lr'] == 1e-3      # finally-restore untouched by the new mode


def test_kl_targeted_refuses_like_fixed_when_the_limit_never_fits(monkeypatch):
    monkeypatch.setattr(bi, 'joint_kl', lambda *a: 1.0)
    for mode in ('fixed', 'kl_targeted'):
        p = Policy()
        with pytest.raises(RuntimeError, match='cannot fit one complete imitation pass'):
            bi.run_separate(p, torch.optim.Adam(p.parameters(), lr=1e-3), pairs(20),
                            torch.Generator().manual_seed(1), copy.deepcopy(p), passes=5,
                            step_mode=mode)


# ------------------------------------------------- trained-subset intent metric + bank term

def _bank_with_untrained_pairs(tmp_path):
    from tools.coopmwm.test_v2v_intent_cpu import cooperative_bank
    got = cooperative_bank(tmp_path)
    demoted = [(got[0][0], 0.)]                              # weight 0: reported, never trained
    unexecuted = dict(got[1][0])
    unexecuted['behaviour'] = dict(unexecuted['behaviour'], executed=False)
    return demoted + [(unexecuted, 1.)] + got[2:]


def test_bank_intent_loss_is_reported_on_the_trained_subset(tmp_path, monkeypatch):
    from tools.coopmwm.test_v2v_intent_cpu import intent_policy
    pairs_ = _bank_with_untrained_pairs(tmp_path)
    policy, optimizer, generator = intent_policy(tmp_path)
    vi.refresh_bank_intents(policy, pairs_)
    trained = [(s, w) for s, w in pairs_ if bi.trainable(s, w)]
    assert len(trained) == len(pairs_)-2
    expected_before = vi.bank_stats(policy, trained)
    # PPO phase held out: the pre-imitation policy state stays exactly the one expected_before used
    monkeypatch.setattr(xg, '_ppo_exact_update', lambda *a, **k: [])
    monkeypatch.setattr(bi, 'run', bi.run_separate)
    metrics = xg.exact_update(policy, optimizer, [], generator, pairs_, 1.,
                              epochs=1, imitation_epochs=5)
    report = metrics[-1]['bank_imitation']
    assert report['pairs_total'] == 10 and report['pairs_filtered_unexecuted'] == 1
    intent = report['v2v_intent']
    assert intent['before']['pairs'] == 12 and intent['after']['pairs'] == 12
    assert intent['before_trained']['pairs'] == 10 and intent['after_trained']['pairs'] == 10
    assert intent['before_trained'] == expected_before
    assert intent['before_trained']['intent_targets'] < intent['before']['intent_targets']
    assert intent['after_trained']['intent_loss'] < intent['before_trained']['intent_loss']
    json.dumps(metrics)


def test_bank_term_off_removes_the_intent_term(tmp_path, monkeypatch):
    from tools.coopmwm.test_v2v_intent_cpu import cooperative_bank, intent_policy
    pairs_ = cooperative_bank(tmp_path)
    real = vi.bank_term
    calls = []

    def spy(policy, batch):
        calls.append(len(batch))
        return real(policy, batch)
    monkeypatch.setattr(vi, 'bank_term', spy)
    policy, optimizer, _ = intent_policy(tmp_path)
    on = bi.run_separate(policy, optimizer, pairs_, torch.Generator().manual_seed(1),
                         copy.deepcopy(policy), passes=5)
    assert calls and 'intent_bank_term' not in on
    calls.clear()
    (tmp_path/'p2').mkdir()
    policy2, optimizer2, _ = intent_policy(tmp_path/'p2')
    off = bi.run_separate(policy2, optimizer2, copy.deepcopy(pairs_), torch.Generator().manual_seed(1),
                          copy.deepcopy(policy2), passes=5, intent_bank_term=False)
    assert not calls and off['intent_bank_term'] is False
    assert 'intent_bank_term' not in bi.run_separate(policy2, optimizer2, pairs_,
                                                     torch.Generator().manual_seed(1),
                                                     copy.deepcopy(policy2), passes=5,
                                                     intent_bank_term=True)


# ---------------------------------------------------------- launcher / runtime / hooks (B == R)

def _cfg(tmp_path, **kw):
    cfg = v2v_runner.prepare_intent(dict(budget={}), root=tmp_path/'v2v', legacy_root=tmp_path/'legacy',
                                    source_root=tmp_path/'src', common_checkpoint_sha256='a'*64, **kw)
    live = dict(cfg, launch_authorized=True, phase_funding_authorized=True,
                budget=dict(total=launch.CEILING_SIM_S, phases=dict(launch.V2V_PHASES)))
    return cfg, live


def test_runtime_pins_the_prepared_defaults_and_refuses_deviations(tmp_path):
    # owner 2026-11-04 08:52: the registered pin is 'kl_targeted' for B and R together
    assert launch.BANK_OPTIONS['imitation_step_mode'] == 'kl_targeted'
    assert launch.BANK_OPTIONS['v2v_intent_bank_term'] is True
    cfg, live = _cfg(tmp_path)
    assert cfg['imitation_step_mode'] == 'kl_targeted' and cfg['v2v_intent_bank_term'] is True
    assert launch.bind_runtime(dict(config=live), root=tmp_path/'v2v', source_root=tmp_path/'src') is live
    assert (launch.imitation_step_mode_for(live, 'B'), launch.imitation_step_mode_for(live, 'R')) == ('kl_targeted', 'kl_targeted')
    assert (launch.intent_bank_term_for(live, 'B'), launch.intent_bank_term_for(live, 'R')) == (True, True)
    for bad in (dict(imitation_step_mode='fixed'), dict(imitation_step_mode=None),
                dict(imitation_step_mode='bogus'), dict(v2v_intent_bank_term=False),
                dict(v2v_intent_bank_term=None), dict(v2v_intent_bank_term=1),
                dict(imitation_step_mode_by_arm=dict(B='kl_targeted', R='fixed')),
                dict(imitation_step_mode_by_arm=dict(B='fixed', R='fixed')),
                dict(v2v_intent_bank_term_by_arm=dict(B=True, R=False)),
                dict(v2v_intent_bank_term_by_arm=dict(B=False, R=False)),
                dict(v2v_intent_bank_term_by_arm=dict(B=1, R=True))):
        with pytest.raises(launch.Refused):
            launch.bind_runtime(dict(config=dict(live, **bad)), root=tmp_path/'v2v', source_root=tmp_path/'src')
    ok = dict(live, imitation_step_mode_by_arm=dict(B='kl_targeted', R='kl_targeted'),
              v2v_intent_bank_term_by_arm=dict(B=True, R=True))
    assert launch.bind_runtime(dict(config=ok), root=tmp_path/'v2v', source_root=tmp_path/'src') is ok
    for arm in ('A',):
        with pytest.raises(launch.Refused):
            launch.imitation_step_mode_for(live, arm)
        with pytest.raises(launch.Refused):
            launch.intent_bank_term_for(live, arm)
    # helpers accept any registered mode for B/R (the launcher pin binds the runtime value)
    assert launch.imitation_step_mode_for(dict(live, imitation_step_mode='kl_targeted'), 'B') == 'kl_targeted'
    with pytest.raises(launch.Refused, match='imitation_step_mode'):
        launch.imitation_step_mode_for(dict(live, imitation_step_mode='bogus'), 'B')
    with pytest.raises(launch.Refused, match='v2v_intent_bank_term'):
        launch.intent_bank_term_for(dict(live, v2v_intent_bank_term=1), 'B')
    with pytest.raises(ValueError):                       # producer refuses a non-pinned value
        _cfg(tmp_path, imitation_step_mode='fixed')
    with pytest.raises(ValueError):
        _cfg(tmp_path, v2v_intent_bank_term=False)


def test_registered_flip_to_kl_targeted_binds_prepares_and_installs_for_b_and_r(tmp_path, monkeypatch):
    # the registered update boundary flips the launcher pin; B and R move together or not at all
    monkeypatch.setattr(launch, 'BANK_OPTIONS',
                        dict(launch.BANK_OPTIONS, imitation_step_mode='kl_targeted'))
    installed = []
    monkeypatch.setattr(bi, 'install',
                        lambda *, step_mode, intent_bank_term: installed.append((step_mode, intent_bank_term)))
    monkeypatch.setattr(bs, 'install', lambda rb_, bank_rule='boot_v2', sources=None: None)
    cfg = v2v_runner.prepare_intent(dict(budget={}), root=tmp_path/'v2v', legacy_root=tmp_path/'legacy',
                                    source_root=tmp_path/'src', common_checkpoint_sha256='a'*64,
                                    imitation_step_mode='kl_targeted')
    assert cfg['imitation_step_mode'] == 'kl_targeted' and cfg['v2v_intent_bank_term'] is True
    live = dict(cfg, launch_authorized=True, phase_funding_authorized=True,
                budget=dict(total=launch.CEILING_SIM_S, phases=dict(launch.V2V_PHASES)))
    assert launch.bind_runtime(dict(config=live), root=tmp_path/'v2v', source_root=tmp_path/'src') is live
    from tools.coopmwm import v2v_live
    h = v2v_live.hooks(tmp_path, cfg)
    assert installed == [('kl_targeted', True)]
    assert h['bank_pairs'] is not None
    installed.clear()
    with pytest.raises(launch.Refused):
        v2v_live.hooks(tmp_path, dict(cfg, imitation_step_mode_by_arm=dict(B='kl_targeted', R='fixed')))
    assert installed == []
    with pytest.raises(launch.Refused):
        v2v_live.hooks(tmp_path, dict(cfg, v2v_intent_bank_term_by_arm=dict(B=True, R=False)))


def test_hooks_default_install_is_fixed_and_true_for_b_and_r(tmp_path, monkeypatch):
    from tools.coopmwm import v2v_live
    installed = []
    monkeypatch.setattr(bi, 'install',
                        lambda *, step_mode, intent_bank_term: installed.append((step_mode, intent_bank_term)))
    monkeypatch.setattr(bs, 'install', lambda rb_, bank_rule='boot_v2', sources=None: None)
    v2v_live.hooks(tmp_path, dict(bank_statistics='per_seed_v1'))
    assert installed == [('fixed', True)]
