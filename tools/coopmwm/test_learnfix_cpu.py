"""c4/learnfix CPU contracts (LEARNDIAG fix): EV-BC labels/weights, fairness on acting rows, choice-row PPO."""
# ruff: noqa: F811 (imported pytest fixtures are test parameters)
from copy import deepcopy

import pytest
import torch

from . import fullpol_learn as fl, learnfix as lf, track2_backend as tb
from .test_track2_backend_cpu import cfg, cpu_threads, fake_load, reward_result, sample  # noqa: F401


def test_settings_registration():
    assert lf.settings({}) is None
    v = lf.value()
    assert lf.settings(dict(learnfix=v)) == v and v['epochs'] == 4 and v['minibatch_ticks'] == 32
    for bad in (dict(v, rule='x'), dict(v, epochs=0), dict(v, extra=1), dict(v, row_weight_cap=2.)):
        with pytest.raises(ValueError, match='learnfix'):
            lf.settings(dict(learnfix=bad))


def test_balance_equal_totals_and_cap():
    w = lf.balance(['act']*2+['none']*1042)
    assert w['act'] == 1. and w['none'] == pytest.approx(2/1042)
    assert 2*w['act'] == pytest.approx(1042*w['none'])                      # equal total weight
    assert lf.balance(['act']*10+['none']*4) == dict(act=.4, none=1.)        # row weight <= 1
    assert lf.balance(['act']*3) == dict(act=1., none=0.)
    assert lf.balance(['none']*50) == dict(act=0., none=0.)                  # lone none labels: never pushed


def test_net_outcome_credit_signed_on_acting_rows_with_running_baseline():
    """Acting rows get net = A_J - B_J (delay + fairness excess) minus the running per-cell baseline; non-acting rows
    keep their local reward. Example: e1 (acted, 300 veh-s worse than its paired A episode, e.g. harmful fairness
    excess) is penalised on its two acting rows; e2 (acted, 100 better) is credited; e3 did not act."""
    targets = {('e1', 300): [-1., -1.], ('e1', 301): [-1., -1.], ('e2', 300): [-1.], ('e3', 300): [-1.]}
    acting = {('e1', 300): [True, False], ('e1', 301): [False, True], ('e2', 300): [True], ('e3', 300): [False]}
    cells = dict(e1='D2800_p20', e2='D2800_p20', e3='D2800_p20')
    out, r = lf.apply_net_acting(targets, {'e1': -300., 'e2': 100., 'e3': -100.}, acting, cells)
    assert r['running_baseline'] == {'D2800_p20': -100.} and r['previous_baseline'] == {}
    assert out[('e1', 300)] == [-1.-100., -1.] and out[('e1', 301)] == [-1., -1.-100.]   # (-300 - -100) / 2
    assert out[('e2', 300)] == [-1.+200.] and out[('e3', 300)] == [-1.]                    # credit; non-acting
    out, r = lf.apply_net_acting(targets, {'e2': 100.}, acting, cells, previous={'D2800_p20': -100., 'D3200_p20': 5.})
    assert r['running_baseline'] == {'D2800_p20': 0., 'D3200_p20': 5.}                     # .5 * -100 + .5 * 100
    assert out[('e2', 300)] == [-1.+100.]


class _B:
    def __init__(self, mask):
        self.mask = torch.tensor(mask)


def test_advantages_cell_baseline_and_choice_normalisation():
    ticks = [dict(batch=_B([[True, True], [True, False]]), adv=[5., 100.], cell='A'),
             dict(batch=_B([[True, True]]), adv=[3.], cell='A'),
             dict(batch=_B([[True, True]]), adv=[-10.], cell='B')]
    r = lf.advantages(ticks)
    assert r['cell_baseline'] == dict(A=4., B=-10.) and r['choice_rows'] == 3
    assert float(ticks[0]['norm_adv'][1]) == 0.                              # no choice: no advantage
    vals = torch.cat([t['norm_adv'][t['batch'].mask[:, 1:].any(1)] for t in ticks])
    assert float(vals.mean()) == pytest.approx(0., abs=1e-6)


def _trig(vid, table, mask):
    return dict(context=dict(members=[vid]), tables=[table], mask=[mask])


def test_ev_labels_act_explicit_none_unlabelled():
    """act = a qualifying family offered (teacher admits / no teacher); none ONLY on an explicit teacher reject;
    otherwise unlabelled (no BC loss)."""
    entry = lambda fam: dict(family='free_program_initiate', template=dict(source=dict(variant=fam)))  # noqa: E731
    rows = {'m': dict(family='m', lb95_veh_s=5., mean_veh_s=6.)}
    offered = _trig('v', [None, entry('m')], [True, True])
    other = _trig('v', [None, entry('x')], [True, True])
    assert fl._learnfix_targets(None, offered, rows, None)[0][2] == 'act'
    assert fl._learnfix_targets(None, offered, rows, lambda r, i: 'admit')[0][2] == 'act'
    assert fl._learnfix_targets(None, offered, rows, lambda r, i: 'reject')[0][:1] == (0,)
    assert fl._learnfix_targets(None, offered, rows, lambda r, i: 'reject')[0][2] == 'none'
    assert fl._learnfix_targets(None, offered, rows, lambda r, i: 'unknown')[0][2] == 'unlabelled'
    assert fl._learnfix_targets(None, other, rows, None)[0][2] == 'unlabelled'          # no target offered
    assert fl._learnfix_targets(None, _trig('v', [None, None], [True, False]), rows, None) == [None]
    # inherited path unchanged: no target offered -> 'none' label (the LEARNDIAG collapse)
    assert fl.ev_targets(None, other, rows)[0][0] == 0


def _episodes(cfg, n):
    import json
    from pathlib import Path
    _, s, _ = sample(cfg)
    episodes, rows = [], []
    for k in range(n):
        episodes.append(dict(id=f'ep{k}', arm='B', world=dict(cell_key='D2800_p20', seed=k),
                             totals=dict(team_delay_vs=1000.+10*k), steps=[dict(time_s=300, sample=deepcopy(s))]))
        a = Path(cfg['run_root'])/'jobs'/f'clean1_warm_u001_A_{k}'
        a.mkdir(parents=True, exist_ok=True)
        (a/'RESULT.json').write_text(json.dumps(dict(totals=dict(team_delay_vs=1050.))))
        for r in reward_result(cfg, s)['rows']:
            rows.append(dict(r, episode_id=f'ep{k}'))
    return episodes, dict(reward_result(cfg, s), rows=rows)


@pytest.mark.parametrize('fix', [False, True])
def test_train_learnfix_receipt_kl_cap_entropy_and_rng(cfg, monkeypatch, fix):
    fake_load(monkeypatch)
    run = dict(cfg, **({'learnfix': lf.value(minibatch_ticks=4)} if fix else {}))
    runtime = tb.build_arm_runtime(run, 'B')['runtime']
    runtime.safety_checked = True
    monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {}))
    episodes, envelope = _episodes(run, 12)
    gen = runtime.generator.get_state().clone()
    rng = torch.get_rng_state().clone()
    report = runtime.train(episodes, rewards=envelope)
    assert torch.equal(runtime.generator.get_state(), gen)                  # shared episode RNG untouched
    if not fix:
        assert 'learnfix' not in report and report['trigger']
        return
    assert torch.equal(torch.get_rng_state(), rng)                         # minibatch order: own generator
    r = report['learnfix']
    assert report['trigger'] == [] and r['ticks'] == 12 and r['rule'] == lf.RULE
    assert r['final_choice_kl'] <= r['kl_budget'] and all(s['choice_kl'] <= r['kl_budget'] for s in r['steps']
                                                         if s['accepted'])
    assert r['entropy_weight'] <= .1*r['adv_rms']/max(r['mean_entropy'], 1e-6)+1e-12
    assert r['entropy_weight'] <= cfg['trigger_ppo']['entropy_weight']
    epochs = {s['epoch'] for s in r['steps']}
    assert epochs <= set(range(12)) and all(s['ticks'] <= 4 for s in r['steps'])
    assert max(epochs) < 4 or all(s['choice_kl'] < .01 for s in r['steps'] if s['epoch'] < max(epochs))  # floor rule
    net = report['learnfix_net']
    assert [e['net_veh_s'] for e in net['episodes']] == [50.-10*k for k in (0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9)]


def test_learnfix2_settings_none_bias_and_credit_cap(cfg, monkeypatch):
    assert lf.settings2({}) is None
    v = lf.value2()
    assert lf.settings2(dict(learnfix2=v)) == v and v['none_bias'] == lf.NONE_BIAS and v['credit_cap'] == 2. and v['update_kl_cap'] == .075
    for bad in (dict(v, rule='x'), dict(v, none_bias=4), dict(v, extra=1), dict(v, credit_cap=0.)):
        with pytest.raises(ValueError, match='learnfix2'):
            lf.settings2(dict(learnfix2=bad))
    # credit cap: |net - baseline| <= 2 x median |net|
    targets = {(e, 300): [0.] for e in 'abcd'}
    acting = {k: [True] for k in targets}
    out, r = lf.apply_net_acting(targets, dict(a=10., b=-10., c=20., d=1000.), acting, cap_mult=2.)
    assert r['credit_cap']['median_abs_net'] == 15. and r['credit_cap']['cap'] == 30.
    assert all(abs(v[0]) <= 30. for v in out.values()) and out[('d', 300)] == [30.]
    out_nocap, _ = lf.apply_net_acting(targets, dict(a=10., b=-10., c=20., d=1000.), acting)
    assert out_nocap[('d', 300)][0] > 30.                                        # default: no cap


def test_none_bias_shifts_only_the_none_logit(cfg, monkeypatch):
    fake_load(monkeypatch)
    plain = tb.build_arm_runtime(cfg, 'B')['runtime']
    biased = tb.build_arm_runtime(dict(cfg, learnfix2=lf.value2()), 'B')['runtime']
    assert plain.policy.trigger.none_bias is None and 'none_bias' not in plain.policy.trigger.state_dict()
    b = biased.policy.trigger
    assert float(b.none_bias) == pytest.approx(lf.NONE_BIAS) and any(p is b.none_bias for p in b.learning_parameters())
    assert any(p is b.none_bias for g in biased.policy.trigger_optimizer.param_groups for p in g['params'])
    _, s, _ = sample(cfg)
    batch = tb._batch(s['track2_trigger'])
    b.load_state_dict(dict(plain.policy.trigger.state_dict(), none_bias=b.none_bias.detach()))
    l0 = plain.policy.trigger.distribution(batch.context, batch.descriptors, batch.mask)[0]
    l1 = b.distribution(batch.context, batch.descriptors, batch.mask)[0]
    assert torch.allclose(l1[:, 0], l0[:, 0]+lf.NONE_BIAS) and torch.equal(l1[:, 1:], l0[:, 1:])


def test_bc_guard_reverts_steps_over_the_cumulative_update_cap(cfg, monkeypatch):
    """learnfix2 guard: an EV-BC step is kept only while the cumulative all-choice-row KL vs the update-start
    policy stays <= cap; over it the step is reverted and BC stops (unlabelled rows cannot drift unbounded)."""
    from .trigger_policy import TriggerBatch
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    _, s, _ = sample(cfg)
    b = tb._batch(s['track2_trigger'])
    choice = b.mask[:, 1:].any(1)
    assert bool(choice.any())
    i = int(choice.nonzero()[0])
    slot = int(b.mask[i, 1:].nonzero()[0])+1
    actions = torch.zeros_like(b.actions)
    actions[i] = slot
    same = torch.nn.functional.one_hot(actions, b.mask.shape[1]).bool()
    w = torch.zeros(len(actions))
    w[i] = 1.
    record = dict(batch=TriggerBatch(b.context, b.tables, b.descriptors, b.mask, actions, b.logp, b.values),
                  weight=1., equivalent=same, row_weight=w, meta=dict(kind='ev_bc'))
    trig = runtime.policy.trigger
    for g in runtime.policy.trigger_optimizer.param_groups:
        g['lr'] = .05
    ticks = [dict(batch=b)]
    start = [trig.distribution(b.context, b.descriptors, b.mask)[0].detach()]
    before = deepcopy(trig.state_dict())
    tight = fl.bc_steps(runtime.policy, [record], epochs=5, kl_budget=10., max_grad_norm=.5, guard=(ticks, start, 1e-9))
    assert tight['passes'] == [dict(accepted=False, kl=tight['passes'][0]['kl'], choice_kl=tight['passes'][0]['choice_kl'])]
    assert all(torch.equal(v, trig.state_dict()[k]) for k, v in before.items())       # reverted
    loose = fl.bc_steps(runtime.policy, [record], epochs=3, kl_budget=10., max_grad_norm=.5, guard=(ticks, start, 10.))
    assert all(p['accepted'] for p in loose['passes']) and len(loose['passes']) == 3
    assert lf.choice_kl(trig, ticks, start) == pytest.approx(loose['passes'][-1]['choice_kl'])


def test_supervise_requires_the_update_start_reference_under_learnfix2(cfg, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(fl, 'learn_settings', lambda c: dict(ev_bc=True, report_kl_unlabelled=False,
                                                             report_valsplit_auc=False))
    monkeypatch.setattr(fl, 'bank_records', lambda policy, c, arm: ([], {}))
    monkeypatch.setattr(fl, 'ev_bc_records', lambda c, arm, episodes: ([], None))
    monkeypatch.setattr(fl, 'active_slot', lambda c: ({}, None))
    policy = SimpleNamespace(layer=SimpleNamespace(config=SimpleNamespace(kl_budget=.045)))
    runtime = SimpleNamespace(cfg=dict(cfg, learnfix2=lf.value2(), imitation_epochs=5), arm='B', policy=policy)
    with pytest.raises(ValueError, match='update-start'):
        fl.supervise(runtime, [])
