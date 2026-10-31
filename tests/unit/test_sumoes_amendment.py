"""MAIN1440 scientific-law and resume guards; CPU only, no SUMO."""
import pytest
import torch

from tools.sumoes.caprel import executed, scalar_executed
from tools.sumoes.es import normalized_step
from tools.sumoes.remote.wm_screen import decide
from tools.sumoes.remote.run import stop_reason


def test_sign_weighted_unit_step_is_scale_invariant_and_minimizes_cost():
    theta = torch.zeros(2, dtype=torch.float64)
    dirs = torch.tensor([[3., 0.], [0., 40.]], dtype=torch.float64)
    delta = torch.tensor([1000., -0.01], dtype=torch.float64)
    candidate = normalized_step(theta, dirs, delta, .2)
    assert torch.equal(candidate, torch.tensor([-.1, .1], dtype=torch.float64))
    assert torch.equal(candidate, normalized_step(theta, dirs * 77, delta * 22, .2))
    assert torch.equal(theta, normalized_step(theta, dirs, delta * 0, .2))
    assert float((candidate - theta).norm()) <= .2


@pytest.mark.parametrize('base,cap,v', [(1.1, -.7, 10.), (-4., -5., 10.),
                                      (-1., .8, 10.), (-4., -8., 1.)])
def test_caprel_braking_has_authority_below_native_base_and_scalar_matches(base, cap, v):
    args = [torch.tensor(x, dtype=torch.float64) for x in (base, cap, v)]
    for r in (-1.2, -.1, 0., .1, 1.2):
        observed = float(executed(*args, torch.tensor(r, dtype=torch.float64)))
        assert observed == scalar_executed(base, cap, v, r)
    ctrl0 = max(-v / .5, min(base, cap))
    assert scalar_executed(base, cap, v, 0.) == ctrl0
    if base > cap and ctrl0 > -v / .5 and ctrl0 > -3.:
        assert scalar_executed(base, cap, v, -.1) == pytest.approx(ctrl0 - .1)
        assert scalar_executed(base, cap, v, 1.) == ctrl0


def test_screen_rejects_confident_harm_and_uncertainty():
    kept, rows = decide([[-5.] * 8, [1.] * 8, [5.] * 8, [3.] * 8,
                         [2.] * 8, [-1., -1., 10., 10., 10., 10., 10., 10.]])
    assert kept == [2, 3, 4]
    assert not rows[0]['confident'] and not rows[5]['confident']
    with pytest.raises(ValueError):
        decide([[float('nan')] * 8])


def test_four_reject_stop_includes_screen_no_steps_and_old_update_zero():
    assert stop_reason([dict(update=0, accepted=False)] * 3) is None
    rows = [dict(update=i, accepted=False) for i in range(4)]
    rows[-1].update(accepted=None, no_step=True)
    assert stop_reason(rows)
    rows[-2]['accepted'] = True
    assert stop_reason(rows) is None
