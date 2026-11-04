"""T2A contracts: local moves API stub only; no simulator, dispatch or truth."""
import copy
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .trigger_policy import TriggerPolicy, MoveExecutor, ppo_update
from .acceleration import Controller, encode
from .v2v_intent import IntentPolicy, intent_model
from .v2v_integrate import widen_context
from .v2v_runtime import Policy as V2VPolicy, controller_class


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
    if dim == 48:
        context['trigger_v2v'] = copy.deepcopy(context['v2v'])
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


def sensor_rows():
    return [dict(vid='ego', s_m=-400., v_mps=10., lane=0, is_cav=True, length_m=5.),
            dict(vid='peer', s_m=-380., v_mps=12., lane=0, is_cav=True, length_m=5.),
            dict(vid='distant_hdv', s_m=0., v_mps=5., lane=0, is_cav=False, length_m=5.)]


@pytest.mark.parametrize('freeze_base', [True, False])
def test_real_encoder_global_rsu_cannot_change_trigger_but_local_inputs_can(freeze_base):
    policy, _, states, visible, _ = setup_policy(freeze_base=freeze_base)
    # A trained V2V checkpoint can have nonzero partner columns as well as the
    # nonzero loaded broadcast from setup_policy. Exercise both input channels.
    with torch.no_grad():
        policy.base_policy.actor[0].weight[:, 29:77].normal_(std=.05)
    rows = sensor_rows()

    def context(sensor):
        ctx = encode(sensor)
        return dict(ctx, v2v={vid: [0.]*48 for vid in ctx['members']},
                    trigger_v2v={vid: [0.]*48 for vid in ctx['members']})

    ctx = context(rows)
    distant = copy.deepcopy(rows)
    distant[-1]['v_mps'] = 25.
    remote_ctx = context(distant)
    assert ctx['x'] == remote_ctx['x'] and ctx['v2v'] == remote_ctx['v2v']
    assert ctx['rsu'] != remote_ctx['rsu']
    assert not torch.equal(policy.base_policy(ctx)[0], policy.base_policy(remote_ctx)[0])
    first = decide(policy, ctx, states, visible)
    remote = decide(policy, remote_ctx, states, visible)
    assert torch.equal(first.descriptors, remote.descriptors)
    assert torch.equal(first.mask, remote.mask)
    logits, values = policy.distribution(ctx, first.descriptors, first.mask)
    remote_logits, remote_values = policy.distribution(remote_ctx, remote.descriptors, remote.mask)
    assert torch.equal(logits.softmax(-1), remote_logits.softmax(-1))
    assert torch.equal(values, remote_values)
    assert torch.equal(first.actions, remote.actions) and torch.equal(first.logp, remote.logp)

    # Trigger evaluation must not require the global block at all.
    local_only = {k: v for k, v in ctx.items() if k != 'rsu'}
    assert all(torch.equal(a, b) for a, b in zip(
        (logits, values), policy.distribution(local_only, first.descriptors, first.mask)))
    local = copy.deepcopy(rows)
    local[1]['v_mps'] = 22.
    local_logits, _ = policy.distribution(context(local), first.descriptors, first.mask)
    assert not torch.equal(logits.softmax(-1)[0], local_logits.softmax(-1)[0])
    reports = copy.deepcopy(local_only)
    reports['trigger_v2v']['ego'][6:12] = [.5]*6  # legitimate recorded local partner report
    report_logits, _ = policy.distribution(reports, first.descriptors, first.mask)
    assert not torch.equal(logits.softmax(-1)[0], report_logits.softmax(-1)[0])
    assert torch.equal(logits[1], report_logits[1])  # no cross-ego pooling


def test_intent_discovery_requires_explicit_module_wrapper_adapter():
    policy, _, _, _, _ = setup_policy()
    assert intent_model(policy) is policy.base_policy
    assert intent_model(SimpleNamespace(policy=policy)) is policy.base_policy
    arbitrary = torch.nn.Module()
    arbitrary.base_policy = policy.base_policy
    assert intent_model(arbitrary) is None  # keep the existing arbitrary-module guard
    plain, _, _, _, _ = setup_policy(dim=24)
    assert intent_model(plain) is None


@pytest.mark.parametrize('deterministic', [True, False])
def test_actual_intent_controller_wrapper_two_ticks_preserves_actions_noise_and_latency(deterministic):
    policy, _, _, _, _ = setup_policy()
    with torch.no_grad():
        policy.base_policy.intent_head.weight.normal_(std=.05)
    control = controller_class(Controller)
    baseline = control(policy.base_policy, torch.Generator().manual_seed(4), deterministic)
    wrapped = control(policy, torch.Generator().manual_seed(4), deterministic)
    previous_published = None
    for now in (301, 302, 304):
        rows = sensor_rows()
        rows[0]['v_mps'] += now-301
        commands = {'ego': .2, 'peer': .2}
        assert wrapped.apply(rows, commands, now) == baseline.apply(rows, commands, now)
        assert wrapped.policy is policy
        sample = wrapped.events[-1]['sample']
        low_level = copy.deepcopy(sample)
        del low_level['context']['trigger_v2v']
        del low_level['published_trigger_intent']
        assert low_level == baseline.events[-1]['sample']
        assert wrapped.memory == baseline.memory
        assert torch.equal(wrapped.generator.get_state(), baseline.generator.get_state())
        assert wrapped.intent_board == baseline.intent_board
        assert sample['published_intent'] == policy.base_policy.publish(sample['context'])
        assert sample['published_trigger_intent'] == policy.publish_trigger_intent(sample['context'])
        assert set(sample['published_intent']) == {'ego', 'peer'}
        for ego, partner in (('ego', 'peer'), ('peer', 'ego')):
            block = sample['context']['v2v'][ego]
            assert len(block) == 48 and block[5] == 1.
            assert block[18:24] == [0.]*6  # no second partner
            if now == 302:
                assert block[6:11] == pytest.approx(previous_published[partner], abs=1e-7)
                assert block[11] == 1.
            else:
                assert block[6:12] == [0.]*6  # first tick or stale after skipped tick
                assert sample['context']['trigger_v2v'][ego][6:12] == [0.]*6
        if now == 302:
            assert sample['published_intent']['peer'] != previous_published['peer']
        previous_published = copy.deepcopy(sample['published_intent'])
        issued = dict(time_s=now, vehicles=[dict(r, issued_acceleration_mps2=.3) for r in rows])
        wrapped.observe_issued(issued)
        baseline.observe_issued(issued)


@pytest.mark.parametrize('freeze_base', [True, False])
def test_actual_controller_remote_state_cannot_leak_through_learned_partner_intents(freeze_base):
    policy, _, states, visible, _ = setup_policy(freeze_base=freeze_base)
    with torch.no_grad():
        policy.base_policy.actor[0].weight[:, 29:77].normal_(std=.2)
        policy.base_policy.intent_head.weight.normal_(std=.2)
    control = controller_class(Controller)
    worlds = [control(policy, torch.Generator().manual_seed(4), True) for _ in range(2)]
    previous = None
    for now in (301, 302, 303, 305):
        contexts, batches = [], []
        for world, remote_speed in zip(worlds, (5., 25.)):
            rows = sensor_rows()
            rows[-1]['v_mps'] = remote_speed
            world.apply(rows, {'ego': .2, 'peer': .2}, now)
            ctx = world.events[-1]['sample']['context']
            contexts.append(ctx)
            batches.append(decide(policy, ctx, states, visible))
        a, b = batches
        assert contexts[0]['x'] == contexts[1]['x']
        assert contexts[0]['rsu'] != contexts[1]['rsu']
        assert a.tables == b.tables and torch.equal(a.descriptors, b.descriptors)
        assert torch.equal(a.mask, b.mask)
        outputs = [policy.distribution(c, a.descriptors, a.mask) for c in contexts]
        assert torch.equal(outputs[0][0], outputs[1][0])
        assert torch.equal(outputs[0][1], outputs[1][1])
        assert torch.equal(a.actions, b.actions) and torch.equal(a.logp, b.logp)
        assert contexts[0]['trigger_v2v'] == contexts[1]['trigger_v2v']
        if now in (302, 303):
            assert contexts[0]['v2v'] != contexts[1]['v2v']  # reproduce the indirect channel
        for ego, partner in (('ego', 'peer'), ('peer', 'ego')):
            block = contexts[0]['trigger_v2v'][ego]
            if now in (302, 303):
                assert block[6:11] == pytest.approx(previous[partner], abs=1e-7)
                assert block[11] == 1.
            else:
                assert block[6:12] == [0.]*6
        previous = copy.deepcopy(worlds[0].events[-1]['sample']['published_trigger_intent'])
        assert previous == worlds[1].events[-1]['sample']['published_trigger_intent']
        # Collection and PPO replay use the same local-only recorded payloads,
        # including after a fine-tuned actor update; no board is queried at replay.
        if now == 303:
            left, right = copy.deepcopy(policy), copy.deepcopy(policy)
            assert update(left, a) == update(right, b)
            assert all(torch.equal(v, right.state_dict()[k]) for k, v in left.state_dict().items())
            assert all(torch.equal(x, y) for x, y in zip(left.evaluate(a), right.evaluate(b)))


@pytest.mark.parametrize('freeze_base', [True, False])
def test_actual_controller_local_partner_intent_remains_usable(freeze_base):
    policy, _, states, visible, _ = setup_policy(freeze_base=freeze_base)
    with torch.no_grad():
        policy.base_policy.actor[0].weight[:, 29:77].normal_(std=.2)
        policy.base_policy.intent_head.weight.normal_(std=.2)
    control = controller_class(Controller)
    worlds = [control(policy, torch.Generator().manual_seed(4), True) for _ in range(2)]
    for now in (301, 302):
        contexts = []
        for world, speed in zip(worlds, (5., 25.)):
            rows = sensor_rows()+[dict(vid='peer_local_hdv', s_m=-290., v_mps=speed,
                                      lane=0, is_cav=False, length_m=5.)]
            world.apply(rows, {'ego': .2, 'peer': .2}, now)
            contexts.append(world.events[-1]['sample']['context'])
        assert contexts[0]['x'][0] == contexts[1]['x'][0]  # outside ego sensing
        assert contexts[0]['x'][1] != contexts[1]['x'][1]  # inside reporting peer sensing
    assert contexts[0]['trigger_v2v']['ego'][:6] == contexts[1]['trigger_v2v']['ego'][:6]
    assert contexts[0]['trigger_v2v']['ego'][6:11] != contexts[1]['trigger_v2v']['ego'][6:11]
    batch = decide(policy, contexts[0], states, visible)
    logits, values = policy.distribution(contexts[0], batch.descriptors, batch.mask)
    other_logits, other_values = policy.distribution(contexts[1], batch.descriptors, batch.mask)
    assert not torch.equal(logits.softmax(-1)[0], other_logits.softmax(-1)[0])
    assert not torch.equal(values[0], other_values[0])
    # Isolate the legitimate local partner report as the only changed ego input.
    replaced = copy.deepcopy(contexts[1])
    replaced['trigger_v2v']['ego'] = contexts[0]['trigger_v2v']['ego']
    replaced_logits, replaced_values = policy.distribution(replaced, batch.descriptors, batch.mask)
    assert torch.equal(logits[0], replaced_logits[0]) and torch.equal(values[0], replaced_values[0])


@pytest.mark.parametrize('freeze_base', [True, False])
def test_trigger_replay_ignores_base_intents_even_in_widened_rows_and_requires_local_payload(freeze_base):
    policy, ctx, states, visible, _ = setup_policy(freeze_base=freeze_base)
    with torch.no_grad():
        policy.base_policy.actor[0].weight[:, 29:77].normal_(std=.2)
    batch = decide(policy, ctx, states, visible)
    expected = policy.evaluate(batch)
    # A collector may store widened low-level rows. Neither these columns nor
    # the separate base board may bypass the recorded local-only trigger board.
    widened = widen_context(ctx, ctx['v2v'])
    for row in widened['x']:
        row[29:] = [17.]*48
    widened['v2v'] = {vid: [-31.]*48 for vid in ctx['members']}
    del widened['rsu']
    replay = replace(batch, context=copy.deepcopy(widened))
    assert all(torch.equal(x, y) for x, y in zip(expected, policy.evaluate(replay)))
    assert torch.equal(policy.imitation_loss(batch, [1, 1]), policy.imitation_loss(replay, [1, 1]))
    # Snapshot survives later board mutation; PPO never recomputes old intents
    # with a new head or falls back to globally conditioned base payloads.
    ctx['trigger_v2v']['ego'][6:12] = [999.]*6
    assert all(torch.equal(x, y) for x, y in zip(expected, policy.evaluate(batch)))
    for missing in (None, {}, {'ego': [0.]*48}):
        invalid = copy.deepcopy(widened)
        invalid['trigger_v2v'] = missing
        with pytest.raises(ValueError, match='local-only'):
            policy.evaluate(replace(batch, context=invalid))
        with pytest.raises(ValueError, match='local-only'):
            policy.imitation_loss(replace(batch, context=invalid), [1, 1])


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
    del ctx['trigger_v2v']
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


def test_explicit_finetuning_reaches_actor_and_v2v_columns_but_not_global_broadcast():
    policy, ctx, states, visible, _ = setup_policy(freeze_base=False)
    before = copy.deepcopy(policy.base_policy.state_dict())
    batch = decide(policy, ctx, states, visible)
    update(policy, batch)
    base = policy.base_policy
    assert base.actor[0].weight.grad[:, 29:29+48].abs().sum() > 0
    assert not torch.equal(before['actor.0.weight'], base.actor[0].weight)
    assert all(p.grad is None for p in base.broadcast.parameters())
    assert not {id(p) for p in base.broadcast.parameters()} & {
        id(p) for p in policy.learning_parameters()}
    assert all(torch.equal(v, before[k]) for k, v in base.state_dict().items()
               if k.startswith(('broadcast.', 'critic.', 'intent_head.')) or k == 'log_std')


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
