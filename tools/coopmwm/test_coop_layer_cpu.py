"""T2D contracts: real MAPPO/V2V networks, synthetic observable CPU inputs."""
import copy
import itertools

import pytest
import torch
from torch import nn

from .acceleration import PHI, RHO, Policy
from .coop_layer import CooperationLayer, LayerConfig
from .v2v_intent import IntentPolicy


def registration(**changes):
    # Test registration only; the implementation supplies no live-run defaults.
    values = dict(feature_dim=29, hidden_dim=8, learning_rate=.001,
                  kl_budget=.045, clip_ratio=.2, max_grad_norm=.5, enabled=True)
    return LayerConfig(**dict(values, **changes))


def context(n=2, extra=0, *, separate=False):
    x = [[(i+j+1)/100 for j in range(29)] for i in range(n)]
    blocks = {f'cav{i}': [(i+j)/10 for j in range(extra)] for i in range(n)}
    if not separate:
        x = [row+blocks[f'cav{i}'] for i, row in enumerate(x)]
    result = dict(x=x, rsu=[.1]*12, members=[f'cav{i}' for i in range(n)])
    if separate:
        result['v2v'] = blocks
    return result


class TriggerStub(nn.Module):
    """T2A adapter contract only; no production substitute for the other task."""
    def __init__(self, width=29):
        super().__init__()
        self.logits = nn.Linear(width, 3)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, features):
        return self.logits(features)


def build(*, trigger=True, **changes):
    torch.manual_seed(21)
    base = Policy()
    # A trained, nonzero actor ensures identity is not a zero-policy accident.
    with torch.no_grad():
        base.actor[-1].weight.fill_(.02)
        base.actor[-1].bias.fill_(.03)
    cfg = registration(**changes)
    layer = CooperationLayer(base, cfg, trigger_head=TriggerStub() if trigger else None)
    return base, layer


def sample(layer, seed=9, *, deterministic=False):
    return layer.sample(context(), torch.Generator().manual_seed(seed), deterministic,
                        layer_generator=torch.Generator().manual_seed(seed+100),
                        move_offsets=[[0., -.2, .1], [0., -.1, .3]] if layer.trigger_head else None)[0]


def records(layer):
    return [dict(sample=sample(layer, seed), group_advantage=1. if seed % 2 else -.4)
            for seed in range(5, 13)]


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('trigger', [False, True])
@pytest.mark.parametrize('deterministic', [False, True])
def test_zero_init_preserves_action_bytes_rng_and_ar_memory(trigger, deterministic):
    base, layer = build(trigger=trigger)
    base_rng, layer_base_rng = [torch.Generator().manual_seed(97) for _ in range(2)]
    trigger_rng = torch.Generator().manual_seed(91)
    a_memory = b_memory = {'cav0': .2, 'cav1': -.1}
    for _ in range(4):
        a, a_memory = base.sample(context(), base_rng, deterministic, a_memory)
        b, b_memory = layer.sample(context(), layer_base_rng, deterministic, b_memory,
            layer_generator=trigger_rng,
            move_offsets=[[0., -.2, .1], [0., -.1, .3]] if trigger else None)
        assert torch.tensor(a['action']).numpy().tobytes() == torch.tensor(b['action']).numpy().tobytes()
        assert a['own_action'] == b['own_action']
        assert a['previous_noise'] == b['previous_noise']
        assert a['value'] == b['value']
        assert a_memory == b_memory
        assert torch.equal(base_rng.get_state(), layer_base_rng.get_state())
        assert float(layer.joint_kl(b).detach()) == pytest.approx(0., abs=1e-7)
        assert float(layer.score(b)[0].detach()) == b['logp']
    assert_tree_equal(base(context()), layer(context()))


def test_off_is_complete_base_sample_identity_even_with_nonzero_heads():
    base, layer = build()
    layer.enabled = False
    with torch.no_grad():
        layer.residual[-1].bias.fill_(7.)
        layer.trigger_gain.fill_(2.)
    a = base.sample(context(), torch.Generator().manual_seed(2), memory={'cav0': .4})
    b = layer.sample(context(), torch.Generator().manual_seed(2), memory={'cav0': .4})
    assert a == b
    assert_tree_equal(base.score(a[0]), layer.score(b[0]))
    default = CooperationLayer(base)
    assert default.optimizer is None
    with pytest.raises(ValueError, match='off'):
        default.update([], epochs=1)
    assert default.sample(context(), torch.Generator().manual_seed(2)) == base.sample(
        context(), torch.Generator().manual_seed(2))


@pytest.mark.parametrize('separate', [False, True])
def test_v2v_intent_base_is_identical_and_completely_frozen_after_updates(separate):
    torch.manual_seed(40)
    base = IntentPolicy(horizon=5)
    with torch.no_grad():
        base.actor[-1].bias.fill_(.1)
        base.intent_head.bias.fill_(.2)
    cfg = registration(feature_dim=29+48)
    layer = CooperationLayer(base, cfg, trigger_head=TriggerStub(29+48))
    ctx = context(extra=48, separate=separate)
    snapshot = copy.deepcopy(base.state_dict())
    a = base.sample(ctx, torch.Generator().manual_seed(7))
    b = layer.sample(ctx, torch.Generator().manual_seed(7),
                     layer_generator=torch.Generator().manual_seed(8),
                     move_offsets=[[0., -.2, .1], [0., -.1, .3]])
    assert a[0]['action'] == b[0]['action'] and a[1] == b[1]
    before = [p.clone() for p in layer.layer_parameters()]
    layer.train()
    assert not layer.base.training and not layer.trigger_head.training
    result = layer.update([dict(sample=b[0], group_advantage=1.)], epochs=3)
    assert any(r['accepted'] for r in result)
    assert any(not torch.equal(a, b) for a, b in zip(before, layer.layer_parameters()))
    assert_tree_equal(snapshot, base.state_dict())
    assert_tree_equal(snapshot, layer.base.state_dict())  # includes intent_bounds buffer
    assert all(not p.requires_grad and p.grad is None for p in layer.base.parameters())
    assert_tree_equal(base(ctx), layer.base(ctx))
    assert base.publish(ctx) == layer.base.publish(ctx)


def test_group_advantage_updates_only_layer_and_base_copy_is_isolated():
    base, layer = build()
    owned = copy.deepcopy(layer.base.state_dict())
    before = [p.clone() for p in layer.layer_parameters()]
    trigger_before = copy.deepcopy(layer.trigger_head.state_dict())
    rollout = records(layer)
    metrics = layer.update(rollout, epochs=3)
    assert len(metrics) == 3 and all(r['accepted'] for r in metrics)
    assert all(0 <= r['kl'] <= layer.config.kl_budget for r in metrics)
    assert any(not torch.equal(a, b) for a, b in zip(before, layer.layer_parameters()))
    assert any(not torch.equal(trigger_before[k], v)
               for k, v in layer.trigger_head.state_dict().items())
    assert float(layer.trigger_gain.detach()) != 0.
    assert_tree_equal(owned, layer.base.state_dict())
    assert_tree_equal(owned, base.state_dict())
    actual = {id(p) for g in layer.optimizer.param_groups for p in g['params']}
    assert actual == set(map(id, layer.layer_parameters()))
    assert actual.isdisjoint(map(id, layer.base.parameters()))
    assert all(p.grad is None for p in layer.base.parameters())
    # A caller retaining an old base optimizer cannot mutate the owned base.
    with torch.no_grad():
        base.actor[-1].bias.add_(10.)
    assert_tree_equal(owned, layer.base.state_dict())


def test_joint_kl_matches_exhaustive_categorical_and_multivariate_gaussian_reference():
    _, layer = build()
    rollout = sample(layer)
    with torch.no_grad():
        layer.trigger_gain.fill_(.3)
        layer.residual[-1].bias.fill_(.07)
        layer.trigger_head.logits.bias.copy_(torch.tensor([.2, -.3, .1]))
    old = rollout['coop_layer']
    offsets = torch.tensor(old['offsets'], dtype=torch.float64)
    old_logits = torch.tensor(old['old_logits'], dtype=torch.float64)
    new_logits = layer.trigger_head(torch.tensor(old['features'])).detach().double()
    p, q = old_logits.softmax(-1), new_logits.softmax(-1)
    sigma2 = float(layer.base.log_std.detach().exp().square()*(1-PHI**2))
    covariance = sigma2*((1-RHO)*torch.eye(2, dtype=torch.float64)+RHO*torch.ones(2, 2, dtype=torch.float64))
    categorical = (p*(p.log()-q.log())).sum()
    gaussian = torch.zeros((), dtype=torch.float64)
    for choices in itertools.product(range(3), repeat=2):
        choice = torch.tensor(choices)
        old_mean = torch.tensor(old['old_residual'], dtype=torch.float64)
        old_mean = old_mean+old['old_gain']*offsets[torch.arange(2), choice]
        new_mean = .07+float(layer.trigger_gain.detach().tanh())*offsets[torch.arange(2), choice]
        old_dist = torch.distributions.MultivariateNormal(old_mean, covariance_matrix=covariance)
        new_dist = torch.distributions.MultivariateNormal(new_mean, covariance_matrix=covariance)
        probability = p[torch.arange(2), choice].prod()
        gaussian += probability*torch.distributions.kl_divergence(old_dist, new_dist)
    assert float(layer.joint_kl(rollout).detach()) == pytest.approx(float(categorical+gaussian), abs=2e-6)


def test_nonzero_layer_adds_residual_and_selected_move_without_changing_base_noise():
    base, layer = build()
    with torch.no_grad():
        layer.residual[-1].bias.fill_(.04)
        layer.trigger_gain.fill_(.3)
    raw, raw_memory = base.sample(context(), torch.Generator().manual_seed(9))
    actual, actual_memory = layer.sample(context(), torch.Generator().manual_seed(9),
        layer_generator=torch.Generator().manual_seed(109),
        move_offsets=[[0., -.2, .1], [0., -.1, .3]])
    record = actual['coop_layer']
    choices = torch.tensor(record['choices'])
    offsets = torch.tensor(record['offsets'])[torch.arange(2), choices]
    correction = .04+float(layer.trigger_gain.detach().tanh())*offsets
    assert torch.equal(torch.tensor(actual['action']), torch.tensor(raw['action'])+correction[:, None])
    assert actual_memory == raw_memory
    assert actual['previous_noise'] == raw['previous_noise']
    # Independently score the conditional, shared-noise action distribution.
    with torch.no_grad():
        mean = base(context())[0].squeeze(-1)
        conditional = mean+correction+PHI*torch.tensor(raw['previous_noise']).squeeze(-1)
        sigma2 = float(base.log_std.exp().square()*(1-PHI**2))
    covariance = sigma2*((1-RHO)*torch.eye(2)+RHO*torch.ones(2, 2))
    distribution = torch.distributions.MultivariateNormal(conditional, covariance_matrix=covariance)
    action = torch.tensor(actual['action']).squeeze(-1)
    logits = torch.tensor(record['old_logits'])
    categorical = logits.log_softmax(-1)[torch.arange(2), choices].sum()
    assert actual['logp'] == pytest.approx(float(distribution.log_prob(action)+categorical), abs=1e-6)


def test_kl_rejection_restores_parameters_and_populated_adam_state():
    _, layer = build()
    assert layer.update(records(layer), epochs=1)[0]['accepted']
    rollout = records(layer)
    # Force a proposed overshoot after Adam moments already exist.
    layer.optimizer.param_groups[0]['lr'] = 100.
    weights = copy.deepcopy(layer.state_dict())
    optimizer = copy.deepcopy(layer.optimizer.state_dict())
    metrics = layer.update(rollout, epochs=4)
    assert len(metrics) == 1 and not metrics[0]['accepted']
    assert metrics[0]['kl'] > layer.config.kl_budget
    assert_tree_equal(weights, layer.state_dict())
    assert_tree_equal(optimizer, layer.optimizer.state_dict())
    assert all(p.grad is None for p in layer.parameters())


def test_nonfinite_proposal_rolls_back_both_weights_and_optimizer(monkeypatch):
    _, layer = build(trigger=False)
    rollout = records(layer)
    before = copy.deepcopy(layer.state_dict())
    optimizer_before = copy.deepcopy(layer.optimizer.state_dict())
    step = layer.optimizer.step

    def poisoned_step(*args, **kwargs):
        step(*args, **kwargs)
        with torch.no_grad():
            layer.residual[-1].bias.fill_(float('nan'))

    monkeypatch.setattr(layer.optimizer, 'step', poisoned_step)
    metrics = layer.update(rollout, epochs=1)
    assert not metrics[0]['accepted']
    assert_tree_equal(before, layer.state_dict())
    assert_tree_equal(optimizer_before, layer.optimizer.state_dict())


def test_checkpoint_state_dict_and_optimizer_resume_are_exact():
    _, layer = build()
    layer.update(records(layer), epochs=2)
    _, restored = build()
    restored.load_state_dict(copy.deepcopy(layer.state_dict()))
    restored.optimizer.load_state_dict(copy.deepcopy(layer.optimizer.state_dict()))
    assert sample(layer) == sample(restored)
    rollout = records(layer)
    assert layer.update(rollout, epochs=1) == restored.update(rollout, epochs=1)
    assert_tree_equal(layer.state_dict(), restored.state_dict())
    assert_tree_equal(layer.optimizer.state_dict(), restored.optimizer.state_dict())


def test_rng_separation_and_candidate_validation_fail_before_base_rng_consumption():
    _, layer = build()
    rng = torch.Generator().manual_seed(3)
    state = rng.get_state().clone()
    for other in (None, rng):
        with pytest.raises(ValueError, match='separate'):
            layer.sample(context(), rng, layer_generator=other)
        assert torch.equal(state, rng.get_state())
    for offsets, message in ((None, 'requires observable'),
                             ([[1., 0., 0.]]*2, 'none'),
                             ([[0., float('nan'), 0.]]*2, 'finite'),
                             ([[0., 0.]]*2, 'match')):
        with pytest.raises(ValueError, match=message):
            layer.sample(context(), rng, layer_generator=torch.Generator(), move_offsets=offsets)
        assert torch.equal(state, rng.get_state())


@pytest.mark.parametrize('change', [dict(feature_dim=0), dict(hidden_dim=True),
    dict(learning_rate=0), dict(kl_budget=float('inf')), dict(clip_ratio=1),
    dict(max_grad_norm=-1), dict(enabled=1)])
def test_invalid_registration_refused(change):
    with pytest.raises(ValueError):
        registration(**change)


def test_no_deterministic_executed_or_nonfinite_group_training():
    _, layer = build()
    for record in (dict(sample=sample(layer, deterministic=True), group_advantage=1.),
                   dict(sample=sample(layer), group_advantage=float('nan'))):
        with pytest.raises(ValueError):
            layer.update([record], epochs=1)
    record = dict(sample=sample(layer), group_advantage=1.)
    record['sample']['behaviour'] = dict(executed=True)
    with pytest.raises(ValueError, match='own-action'):
        layer.update([record], epochs=1)


def test_optimizer_contamination_unfreezing_and_stale_rollout_refused():
    _, layer = build()
    rollout = records(layer)
    p = next(layer.base.parameters())
    layer.optimizer.add_param_group(dict(params=[p]))
    with pytest.raises(ValueError, match='exactly the layer'):
        layer.update(rollout, epochs=1)
    _, layer = build()
    next(layer.base.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match='frozen'):
        sample(layer)
    _, layer = build()
    rollout = records(layer)
    with torch.no_grad():
        layer.residual[-1].bias.fill_(1.)
    with pytest.raises(ValueError, match='already exceeds'):
        layer.update(rollout, epochs=1)


def test_empty_members_remain_supported_and_bad_features_refuse():
    base, layer = build(trigger=False)
    ctx = context(n=0)
    a = base.sample(ctx, torch.Generator().manual_seed(2))
    b = layer.sample(ctx, torch.Generator().manual_seed(2))
    assert a == b
    assert_tree_equal(base.score(a[0]), layer.score(b[0]))
    assert_tree_equal(base(ctx), layer(ctx))
    with pytest.raises(ValueError, match='width'):
        layer.sample(context(extra=1), torch.Generator())
    bad = context()
    bad['x'][0][0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        layer.sample(bad, torch.Generator())


def test_cpu_only_refuses_non_cpu_module_without_allocating_gpu():
    with pytest.raises(ValueError, match='CPU only'):
        CooperationLayer(nn.Linear(2, 1, device='meta'))
