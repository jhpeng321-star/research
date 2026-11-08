"""LEARNCHECK offline check: pure metric and verdict helpers (no recorded data, no SUMO)."""
import importlib.util
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    'learnfix_offline_check', Path(__file__).resolve().parents[1]/'ops'/'learnfix_offline_check.py')
lc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lc)


def _lp(logits, mask):
    return torch.log_softmax(torch.tensor(logits).masked_fill(~mask, -torch.inf), -1)


MASK = torch.tensor([[True, True, True], [True, False, False]])     # row 0 choice, row 1 none-only


def test_choice_kl_ignores_none_only_rows_and_is_zero_for_identity():
    a = _lp([[0., 0., 0.], [5., 0., 0.]], MASK)
    b = _lp([[1., 0., 0.], [5., 0., 0.]], MASK)
    assert lc.choice_kl([a], [a], [MASK]) == 0.
    p, q = a[0].exp(), b[0].exp()
    expect = float((p*(p.log()-q.log())).sum())
    assert lc.choice_kl([a], [b], [MASK]) == pytest.approx(expect, rel=1e-5)


def test_p_none_choice_only_counts_choice_rows():
    a = _lp([[0., 0., 0.], [0., 0., 0.]], MASK)
    out = lc.p_none_choice([a], [MASK])
    assert out['rows'] == 1 and out['mean'] == pytest.approx(1/3, rel=1e-5)


def test_p_target_uses_equivalent_set_on_act_rows_only():
    class B:
        actions = torch.tensor([2, 0])
    rec = dict(batch=B(), equivalent=torch.tensor([[False, True, True], [True, False, False]]),
               row_weight=torch.tensor([1., 1.]))
    a = _lp([[0., 0., 0.], [0., 0., 0.]], MASK)
    out = lc.p_target([rec], [a])
    assert out['rows'] == 1 and out['mean'] == pytest.approx(2/3, rel=1e-5)


def _res(code, **kw):
    base = dict(code=code, choice_kl_total=0.02, choice_kl_ppo=0.004,
                p_target_act=dict(before=dict(mean=0.2, rows=5), after=dict(mean=0.3, rows=5)),
                p_none_choice=dict(before=dict(mean=0.2), after=dict(mean=0.22)), p_target_demo=None,
                fairness=dict(status='probed', all_raw_lower=True, mean_norm_lower=True))
    base.update(kw)
    return base


def test_verdict_new_pass_and_each_failure():
    assert all(v in (True, None) for v in lc.verdict(_res('new')).values())
    assert lc.verdict(_res('new', choice_kl_total=0.005))['c1_kl_in_band'] is False
    assert lc.verdict(_res('new', choice_kl_total=0.05))['c1_kl_in_band'] is False
    v = lc.verdict(_res('new', p_target_act=dict(before=dict(mean=0.3), after=dict(mean=0.29))))
    assert v['c2_p_target_rises'] is False
    assert lc.verdict(_res('new', p_none_choice=dict(before=dict(mean=.2), after=dict(mean=.26))))[
        'c3_p_none_rise_ok'] is False
    assert lc.verdict(_res('new', fairness=dict(status='probed', all_raw_lower=False,
                                                mean_norm_lower=True)))['c4_fairness'] is False
    assert lc.verdict(_res('new', fairness=dict(status='fair reward off')))['c4_fairness'] is None


def test_verdict_old_control():
    assert lc.verdict(_res('old'))['c5_old_ppo_kl_near_zero'] is True
    assert lc.verdict(_res('old', choice_kl_ppo=0.02))['c5_old_ppo_kl_near_zero'] is False


def test_guard_refuses_forbidden_paths():
    for p in ('/x/HOLDOUT/a', '/x/sealed/b', '/x/cal_split', '/y/j4b_q', '/y/chunk5/z', '/inc1/r', '/heldout/'):
        with pytest.raises(SystemExit):
            lc.guard(p)
    assert lc.guard('/home/ubuntu/coopmwm20261108_clean5c4') == '/home/ubuntu/coopmwm20261108_clean5c4'
