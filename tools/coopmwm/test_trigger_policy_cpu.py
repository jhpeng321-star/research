"""T2A contracts: local moves API stub only; no simulator, dispatch or truth."""
import copy
from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from .trigger_policy import TriggerPolicy, MoveExecutor, ppo_update
from .v2v_intent import IntentPolicy
from .v2v_runtime import Policy as V2VPolicy


@dataclass(frozen=True)
class Move:
    family: str = 'ease_off'
    members: tuple = ('ego', 'peer')
    onset_s: float = 0.
    hold_s: float = 20.
    depth: float = .5


class MovesStub:
    FAMILIES = ('none', 'ease_off', 'wall', 'gap_open', 'ramp_assist')

    def __init__(self):
        self.calls, self.offset_calls = [], []

    def candidates(self, obs_state, rng, n):
        self.calls.append((copy.deepcopy(obs_state), n))
        # Deliberately stochastic to detect accidental PPO re-sampling.
        rng.uniform()
        return list(obs_state['choices'])

    def offsets(self, move, t):
        self.offset_calls.append((move, t))
        if t < move.onset_s or t >= move.onset_s+move.hold_s:
            return {}
        return {vid: -move.depth for vid in move.members}


def setup_policy(*, freeze_base=True, dim=48, n=3):
    torch.manual_seed(191)
    base = IntentPolicy() if dim == 48 else V2VPolicy()
    # Exercise a real nontrivial loaded MAPPO actor, not only its zero head.
    with torch.no_grad():
        base.actor[-1].weight.normal_(std=.05)
        base.broadcast[-1].weight.normal_(std=.05)
    stub = MovesStub()
    policy = TriggerPolicy(base, v2v_dim=dim, n_candidates=n, partner_slots=5,
                           freeze_base=freeze_base, moves_api=stub)
    context = dict(members=['ego', 'peer'], x=[[.1]*29, [.3]*29], rsu=[.2]*12,
                   v2v={'ego': [.5]*dim, 'peer': [.8]*dim})
    states = {'ego': dict(choices=[Move(), Move('gap_open', depth=.8)]),
              'peer': dict(choices=[Move('wall')])}
    visible = {vid: ('ego', 'peer') for vid in context['members']}
    return policy, context, states, visible, stub


def decide(policy, context, states, visible, *, deterministic=False, seed=71):
    return policy.decide(context, obs_states=states, observable_cavs=visible,
                         rng=np.random.default_rng(27),
                         generator=torch.Generator().manual_seed(seed), deterministic=deterministic)


def update(policy, batch, advantages=None, returns=None, **overrides):
    kwargs = dict(clip_ratio=.2, value_weight=.5, entropy_weight=.01, max_grad_norm=1.)
    kwargs.update(overrides)
    return ppo_update(policy, policy.optimizer(lr=.001), batch,
                      advantages if advantages is not None else [1., -.5],
                      returns if returns is not None else [2., -.8], **kwargs)


@pytest.mark.parametrize('dim', [24, 48])
def test_base_actions_noise_scores_and_state_are_byte_identical(dim):
    policy, ctx, states, visible, _ = setup_policy(dim=dim)
    saved = copy.deepcopy(policy.base_policy.state_dict())
    a, amem = policy.base_policy.sample(ctx, torch.Generator().manual_seed(9))
    b, bmem = policy.sample(ctx, torch.Generator().manual_seed(9))
    assert a == b and amem == bmem
    assert torch.equal(policy.score(a)[0], policy.base_policy.score(a)[0])
    batch = decide(policy, ctx, states, visible)
    assert torch.equal(policy(ctx)[0], policy.base_policy(ctx)[0])
    update(policy, batch)
    assert all(torch.equal(v, saved[k]) for k, v in policy.base_policy.state_dict().items())
    assert all(p.grad is None for p in policy.base_policy.parameters())
    after, memory = policy.sample(ctx, torch.Generator().manual_seed(9))
    assert after == a and memory == amem


def test_attachment_preserves_global_rng_and_validates_width():
    base = IntentPolicy()
    rng = torch.get_rng_state().clone()
    TriggerPolicy(base, v2v_dim=48, n_candidates=3, partner_slots=5, moves_api=MovesStub())
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match='v2v_dim'):
        TriggerPolicy(base, v2v_dim=24, n_candidates=3, partner_slots=5, moves_api=MovesStub())


def test_per_cav_discrete_move_table_masks_snapshot_and_score_replay():
    policy, ctx, states, visible, stub = setup_policy()
    batch = decide(policy, ctx, states, visible)
    assert [n for _, n in stub.calls] == [3, 3]
    assert batch.mask.tolist() == [[True, True, True, False], [True, True, False, False]]
    assert batch.tables[0][0] is None and batch.tables[0][1] == Move()
    logp, values, entropy = policy.evaluate(batch)
    assert torch.equal(logp, batch.logp) and torch.equal(values, batch.values)
    assert logp.shape == values.shape == entropy.shape == (2,)
    assert not batch.logp.requires_grad and not batch.values.requires_grad
    logits, _ = policy.distribution(batch.context, batch.descriptors, batch.mask)
    assert logits.softmax(-1)[0, 3] == 0 and logits.softmax(-1)[1, 2:].sum() == 0
    selected = batch.selected()
    assert set(selected) == {'ego', 'peer'}
    for i, vid in enumerate(ctx['members']):
        assert selected[vid] == batch.tables[i][batch.actions[i]]
    ctx['x'][0][0] = 999.
    states['ego']['choices'].clear()
    assert batch.context['x'][0][0] == .1 and len(batch.tables[0]) == 4
    policy.evaluate(batch)
    assert len(stub.calls) == 2  # no new candidates during PPO scoring


def test_deterministic_and_stochastic_trigger_reproducibility():
    policy, ctx, states, visible, _ = setup_policy()
    a = decide(policy, ctx, states, visible, seed=55)
    b = decide(policy, ctx, states, visible, seed=55)
    assert torch.equal(a.actions, b.actions)
    c = decide(policy, ctx, states, visible, deterministic=True)
    logits, _ = policy.distribution(c.context, c.descriptors, c.mask)
    assert torch.equal(c.actions, logits.argmax(-1))


def test_none_only_and_empty_road():
    policy, ctx, states, visible, _ = setup_policy()
    states = {vid: dict(choices=[Move('none')]) for vid in ctx['members']}
    batch = decide(policy, ctx, states, visible)
    assert batch.actions.tolist() == [0, 0] and batch.logp.tolist() == [0., 0.]
    assert batch.selected() == {'ego': None, 'peer': None}
    assert policy.evaluate(batch)[2].tolist() == [0., 0.]
    empty = decide(policy, dict(members=[], x=[], rsu=[0.]*12), {}, {})
    assert empty.actions.shape == (0,) and empty.selected() == {}
    assert all(v.numel() == 0 for v in policy.evaluate(empty))
    with pytest.raises(ValueError, match='empty'):
        update(policy, empty, [], [])


@pytest.mark.parametrize('bad', [Move(members=('peer',)), Move(members=('ego', 'hdv')),
                                Move(members=('ego', 'ego')), Move(members=('ego', 'far')),
                                Move(onset_s=31.), Move(hold_s=19.), Move(depth=float('nan')),
                                Move(family='new_unregistered'),
                                Move(members=('ego', 'peer', 'p2', 'p3', 'p4'))])
def test_candidates_refuse_nonlocal_members_and_unregistered_parameters(bad):
    policy, ctx, states, visible, _ = setup_policy()
    states['ego']['choices'] = [bad]
    with pytest.raises(ValueError, match='move|registered'):
        decide(policy, ctx, states, visible)


def test_missing_v2v_and_candidate_overflow_refused():
    policy, ctx, states, visible, _ = setup_policy()
    del ctx['v2v']
    with pytest.raises(ValueError, match='partner observations'):
        decide(policy, ctx, states, visible)
    policy, ctx, states, visible, _ = setup_policy(n=1)
    with pytest.raises(ValueError, match='budget'):
        decide(policy, ctx, states, visible)
    states['ego']['choices'] = []
    visible['ego'] = ('peer',)
    with pytest.raises(ValueError, match='observable CAV slots'):
        decide(policy, ctx, states, visible)


def test_candidate_order_equivariance_and_family_partners_parameters_are_encoded():
    policy, ctx, states, visible, _ = setup_policy()
    a = decide(policy, ctx, states, visible)
    states['ego']['choices'].reverse()
    b = decide(policy, ctx, states, visible)
    alogits, _ = policy.distribution(a.context, a.descriptors, a.mask)
    blogits, _ = policy.distribution(b.context, b.descriptors, b.mask)
    assert torch.equal(alogits[0, [0, 2, 1, 3]], blogits[0])
    f = len(policy.families)
    assert a.descriptors[0, 1, policy.families.index('ease_off')] == 1
    assert a.descriptors[0, 1, f:f+5].tolist() == [1., 1., 0., 0., 0.]
    assert a.descriptors[0, 1, -3:-1].tolist() == [0., 0.]
    assert a.descriptors[0, 1, -1] == pytest.approx((.5-.3)/(.7))


def test_ppo_increases_selected_probability_and_learns_value_with_frozen_base():
    policy, ctx, states, visible, _ = setup_policy()
    batch = decide(policy, ctx, states, visible)
    batch.actions = torch.tensor([1, 1])
    logp, values, _ = policy.evaluate(batch)
    batch.logp = logp.detach().clone()
    metrics = update(policy, batch, [1., 1.], (values.detach()+1).tolist(), entropy_weight=0.)
    new_logp, new_values, _ = policy.evaluate(batch)
    assert bool((new_logp > logp.detach()).all())
    assert bool((new_values > values.detach()).all())
    assert metrics['approx_kl'] == pytest.approx(0., abs=1e-7)
    assert metrics['clip_fraction'] == 0 and metrics['cav_count'] == 2
    assert metrics['grad_norm'] > 0


def test_ppo_clips_both_advantage_signs_against_recorded_logp():
    policy, ctx, states, visible, _ = setup_policy()
    batch = decide(policy, ctx, states, visible)
    current = policy.evaluate(batch)[0].detach()
    batch.logp = current-torch.log(torch.tensor([2., .5]))
    metrics = update(policy, batch, [1., -1.], value_weight=0., entropy_weight=0.)
    assert metrics['policy_loss'] == pytest.approx(-(.2), abs=1e-6)
    assert metrics['clip_fraction'] == 1.
    assert metrics['approx_kl'] > 0


def test_explicit_finetuning_reaches_actor_and_v2v_columns_but_not_critic():
    policy, ctx, states, visible, _ = setup_policy(freeze_base=False)
    before = copy.deepcopy(policy.base_policy.state_dict())
    batch = decide(policy, ctx, states, visible)
    update(policy, batch)
    base = policy.base_policy
    assert base.actor[0].weight.grad[:, 29:29+48].abs().sum() > 0
    assert not torch.equal(before['actor.0.weight'], base.actor[0].weight)
    assert not torch.equal(before['broadcast.2.weight'], base.broadcast[-1].weight)
    assert all(torch.equal(v, before[k]) for k, v in base.state_dict().items()
               if k.startswith(('critic.', 'intent_head.')) or k == 'log_std')


def test_imitation_targets_are_indices_into_same_masked_table_and_train_head():
    policy, ctx, states, visible, _ = setup_policy()
    batch = decide(policy, ctx, states, visible)
    loss = policy.imitation_loss(batch, [1, 1])
    loss.backward()
    assert sum(p.grad.abs().sum() for p in policy.trigger_head.parameters()) > 0
    assert all(p.grad is None for p in policy.base_policy.parameters())
    with pytest.raises(ValueError, match='masked imitation'):
        policy.imitation_loss(batch, [3, 2])
    batch.actions = torch.tensor([3, 2])
    with pytest.raises(ValueError, match='masked trigger'):
        policy.evaluate(batch)


def test_ppo_refuses_foreign_optimizer_and_bad_targets():
    policy, ctx, states, visible, _ = setup_policy()
    batch = decide(policy, ctx, states, visible)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    with pytest.raises(ValueError, match='exactly'):
        ppo_update(policy, optimizer, batch, [1., 1.], [1., 1.], clip_ratio=.2,
                   value_weight=.5, entropy_weight=0., max_grad_norm=1.)
    for advantages in ([float('nan'), 0.], [1.]):
        with pytest.raises(ValueError, match='finite and match'):
            update(policy, batch, advantages=advantages)
    with pytest.raises(ValueError, match='PPO settings'):
        update(policy, batch, clip_ratio=1.)


def test_cpu_checkpoint_restores_logits_optimizer_and_generator(tmp_path):
    policy, ctx, states, visible, _ = setup_policy()
    batch = decide(policy, ctx, states, visible)
    opt = policy.optimizer(lr=.001)
    kwargs = dict(clip_ratio=.2, value_weight=.5, entropy_weight=.01, max_grad_norm=1.)
    ppo_update(policy, opt, batch, [1., -.5], [2., -.8], **kwargs)
    rng = torch.Generator().manual_seed(22)
    path = tmp_path/'trigger.pt'
    torch.save(dict(policy=policy.state_dict(), optimizer=opt.state_dict(), rng=rng.get_state()), path)
    restored, _, _, _, _ = setup_policy()
    saved = torch.load(path, map_location='cpu', weights_only=False)
    restored.load_state_dict(saved['policy'], strict=True)
    other_opt = restored.optimizer(lr=.001)
    other_opt.load_state_dict(saved['optimizer'])
    assert all(torch.equal(a, b) for a, b in zip(policy.evaluate(batch), restored.evaluate(batch)))
    ppo_update(policy, opt, batch, [1., -.5], [2., -.8], **kwargs)
    ppo_update(restored, other_opt, batch, [1., -.5], [2., -.8], **kwargs)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in policy.state_dict().items())
    resumed_rng = torch.Generator().set_state(saved['rng'])
    assert torch.equal(torch.rand(4, generator=rng), torch.rand(4, generator=resumed_rng))


def start(executor, move=Move(), *, now=100., until=150., ego='ego', visible=('ego', 'peer')):
    return executor.start(move, ego=ego, observable_cav_ids=visible, now=now, until=until)


def requests(executor, *, now=100., base=None, eligible=('ego', 'peer'), bounds=None):
    return executor.requests(base or {'ego': 1., 'peer': -.2, 'hdv': 8.},
                             {'ego': .2, 'peer': .1, 'hdv': 8.}, now=now,
                             eligible_cav_ids=eligible,
                             feasible_bounds=bounds or {'ego': (-3., 2.6), 'peer': (-3., 2.6)})


def test_execution_uses_offsets_on_base_not_native_then_returns_ownership_requests():
    stub = MovesStub()
    executor = MoveExecutor(moves_api=stub)
    assert start(executor)
    result = requests(executor)
    assert result['ego'] == dict(requested_raw=.5, acceleration_residual=.3)
    assert result['peer']['requested_raw'] == pytest.approx(-.7)
    assert result['peer']['acceleration_residual'] == pytest.approx(-.8)
    assert 'hdv' not in result and stub.offset_calls[-1][1] == 0.
    # It is an overlay; the caller's MAPPO/native maps are not mutated.
    base = {'ego': 1., 'peer': -.2, 'hdv': 8.}
    original = dict(base)
    requests(executor, now=101., base=base)
    assert base == original and stub.offset_calls[-1][1] == 1.


def test_execution_feasibility_cap_missing_departed_and_none_identity():
    executor = MoveExecutor(moves_api=MovesStub())
    assert requests(executor) == {}  # exactly no layer request when off
    assert not start(executor, None) and not start(executor, Move('none'))
    assert requests(executor) == {}
    assert start(executor)
    out = requests(executor, base={'ego': 4., 'peer': -4.})
    assert out['ego']['requested_raw'] == 1.2 and out['peer']['requested_raw'] == -3.
    assert requests(executor, now=101., eligible=('ego',)) .keys() == {'ego'}
    assert requests(executor, now=102., base={'ego': 1.}).keys() == {'ego'}
    out = requests(executor, now=103., bounds={'ego': (-3., .3), 'peer': (-.4, 1.2)})
    assert out['ego']['requested_raw'] == .3 and out['peer']['requested_raw'] == -.4
    assert requests(executor, now=120.) == {}  # API owns the release profile
    assert requests(executor, now=150.) == {} and not executor.active


def test_execution_onset_conflict_rejection_disjoint_moves_and_expiry():
    executor = MoveExecutor(moves_api=MovesStub())
    assert start(executor, replace(Move(), onset_s=5.), until=125.)
    assert requests(executor) == {}
    assert not start(executor, ego='peer')
    assert len(executor.active) == 1
    assert start(executor, Move(members=('other',)), ego='other', visible=('other',))
    assert len(executor.active) == 2
    assert requests(executor, now=105.)['ego']['requested_raw'] == .5
    assert start(executor, now=125., until=155.)
    assert len(executor.active) == 2
    with pytest.raises(ValueError, match='monotonic'):
        requests(executor, now=124.)


def test_execution_rejects_nonobservable_members_bad_expiry_and_api_corruption():
    stub = MovesStub()
    executor = MoveExecutor(moves_api=stub)
    with pytest.raises(ValueError, match='observable CAV'):
        start(executor, Move(members=('ego', 'hdv')))
    with pytest.raises(ValueError, match='expiry'):
        start(executor, until=100.)
    assert start(executor)
    stub.offsets = lambda move, t: {'hdv': -.2}
    with pytest.raises(ValueError, match='undeclared'):
        requests(executor)
    stub.offsets = lambda move, t: {'ego': float('nan')}
    with pytest.raises(ValueError, match='nonfinite'):
        requests(executor)
    stub.offsets = lambda move, t: {'ego': -.5}
    with pytest.raises(ValueError, match='feasibility'):
        requests(executor, bounds={'ego': (2., 3.), 'peer': (-3., 1.2)})
