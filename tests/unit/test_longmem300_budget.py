"""S-LONGMEM300: the aggregate allocator budget ceiling is configurable, default unchanged."""
import pytest

from wms.control import parallel_futures as pf


def test_default_budget_cap_is_the_registered_60_percent(monkeypatch):
    monkeypatch.delenv('WMS_SPEED_FUTURE_BUDGET_CAP', raising=False)
    assert pf.budget_cap() == .60


@pytest.mark.parametrize('raw,expected', [('.85', .85), ('0.95', .95), ('.6', .6)])
def test_budget_cap_override(monkeypatch, raw, expected):
    monkeypatch.setenv('WMS_SPEED_FUTURE_BUDGET_CAP', raw)
    assert pf.budget_cap() == expected


@pytest.mark.parametrize('raw', ['0', '-.1', '.96', '1.2', 'abc'])
def test_budget_cap_rejects_invalid(monkeypatch, raw):
    monkeypatch.setenv('WMS_SPEED_FUTURE_BUDGET_CAP', raw)
    with pytest.raises(ValueError):
        pf.budget_cap()


@pytest.mark.parametrize('mode', ['1', '2'])
def test_reward_recompute_flag_preserves_update_exactly(monkeypatch, mode):
    """WMS_LM300_RECOMPUTE_REWARD=1/2 (reward recomputed in LONGMEM) is exact and default OFF."""
    import torch
    from tests.unit.test_longmem import trainer, assert_tree_close

    def run(flag):
        if flag:
            monkeypatch.setenv('WMS_LM300_RECOMPUTE_REWARD', mode)
        else:
            monkeypatch.delenv('WMS_LM300_RECOMPUTE_REWARD', raising=False)
        t = trainer(1, 3, True, counterfactual=True, entropy_target=.5)
        out = []
        for _ in range(2):
            torch.manual_seed(91 + t.update_count)
            out.append(t.update())
        return t, out

    base, a = run(False)
    flagged, b = run(True)
    for x, y in zip(a, b):
        for key in ('actor_loss', 'returns', 'return_scale', 'actor_grad_norm'):
            assert_tree_close(x[key], y[key], atol=0, rtol=0)
    for x, y in zip(base.actor_parameters, flagged.actor_parameters):
        assert torch.equal(x, y)
        assert torch.equal(x.grad, y.grad)


def test_forward_pool_size_is_separately_configurable(monkeypatch):
    monkeypatch.setenv('WMS_SPEED_FUTURE_WORKERS', '2')
    monkeypatch.delenv('WMS_SPEED_FORWARD_WORKERS', raising=False)
    assert pf.worker_count() == pf.worker_count('forward') == pf.worker_count('backward') == 2
    monkeypatch.setenv('WMS_SPEED_FORWARD_WORKERS', '3')
    assert pf.worker_count('forward') == 3 and pf.worker_count() == 2
    monkeypatch.setenv('WMS_SPEED_FORWARD_WORKERS', '9')
    with pytest.raises(ValueError):
        pf.worker_count('forward')


def test_pair_bias_checkpoint_is_exact_and_default_off(monkeypatch):
    import torch
    from wms.model import residual_net as rn

    torch.manual_seed(5)
    cfg = rn.HdvConfig() if hasattr(rn, 'HdvConfig') else None
    attn = rn._KinAttention(cfg)
    b, k = 2, 6
    x = torch.randn(b, k, cfg.d_model, requires_grad=True)
    mask = torch.ones(b, k, dtype=torch.bool)
    geo = dict(s=torch.rand(b, k) * 300, lane=torch.randint(0, 2, (b, k)),
               v=torch.rand(b, k) * 30)

    def run(flag):
        if flag:
            monkeypatch.setenv('WMS_LM300_PAIRBIAS_CKPT', '1')
        else:
            monkeypatch.delenv('WMS_LM300_PAIRBIAS_CKPT', raising=False)
        attn.zero_grad()
        if x.grad is not None:
            x.grad = None
        out, a = attn(x, mask, geo)
        (out.square().sum() + a.sum()).backward()
        return out.detach().clone(), x.grad.clone(), [p.grad.clone() for p in attn.parameters()
                                                      if p.grad is not None]

    base, flagged = run(False), run(True)
    assert torch.equal(base[0], flagged[0]) and torch.equal(base[1], flagged[1])
    for u, w in zip(base[2], flagged[2]):
        assert torch.equal(u, w)
