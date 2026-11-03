"""PART 4 (INTEGRATOR, c4/intent-int) end-to-end CPU gate for the intent-sharing build.

Chain under test, all CPU, no SUMO/GPU:
  1. a MAPPO-u0-shaped checkpoint (accel_learner.save schema) loads through
     v2v_model.load_checkpoint with the intent binding keys
     (``{'v2v': True, 'v2v_intent': True, 'v2v_intent_horizon': 5, 'accel_bounds': ...}``)
     -> zero-initialised v2v/intent input extension PLUS zero-initialised intent
     head; executed actions at load are byte-identical to the non-v2v load;
  2. 3 synthetic CAVs over several control steps -> each CAV's planned commands
     are published at step t (V2VIntentBuffer) and observed by its partners at
     t+1 (one-step V2V latency: fresh at t+1, stale/absent at t+2 with a zero
     intent + intent_valid=0 mask); with the option off the block is EXACTLY the
     plain v2v block (byte-identical);
  3. ONE full-pass imitation update on a synthetic cooperative bank
     (bank_imitation.run_separate with intent=) lowers BOTH the action
     imitation error and the masked plan-future intent error;
  4. all options off = byte-identical behaviour: accel_learner.update without
     the intent kwarg (the shipped path) produces metrics and post-update
     weights identical to the explicit intent=None call, with no intent keys.
"""
import copy

import numpy as np
import torch

from tools.coopmwm.accel_learner import load as ckpt_load, save as ckpt_save, update
from tools.coopmwm.acceleration import Policy, encode
from tools.coopmwm.bank_imitation import action_error, intent_error, run_separate
from tools.coopmwm.v2v_integrate import policy_call, widen_context
from tools.coopmwm.v2v_model import (INTENT_BOUNDS, V2V_INTENT_DIM, load_checkpoint,
                                    planned_commands)
from tools.coopmwm.v2v_obs import (V2V_DIM, V2VConfig, V2VIntentBuffer, intent_block_width,
                                   publish_intents, v2v_features, v2v_intent_features)


class WidenedModule(Policy):
    """The runtime-shaped extended policy: forward handles widened contexts
    exactly like tools/coopmwm/v2v_runtime.Policy (policy_call routing)."""

    def forward(self, context):
        return policy_call(self, context)

INTENT_CFG = dict(v2v=True, v2v_intent=True, v2v_intent_horizon=5,
                  v2v_dim=V2V_INTENT_DIM, accel_bounds=INTENT_BOUNDS)
K = 4                      # v2v_obs.V2V_K
H = 5                      # v2v_intent_horizon
STRIDE = 6 + H + 1         # per-partner intent slot stride
BASE_ACTOR_IN = 29 + 4     # sensor features + broadcast latent
BASE_CRITIC_IN = 29 + 12


def rows(step):
    """3 synthetic CAVs in the whitelisted sensor schema, positions advancing."""
    base = [(-30., 8., 0), (-12., 9.5, 0), (25., 10., 1)]
    return [dict(vid=f'cav{i}', s_m=s + step * v, v_mps=v, lane=lane, is_cav=True, length_m=5.)
            for i, (s, v, lane) in enumerate(base)]


def fixture_checkpoint(path):
    """A real-shape checkpoint: warmed Adam moments, nonzero update id, charged bookkeeping."""
    torch.set_num_threads(1)
    torch.manual_seed(47)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    mean, value = policy(encode(rows(0)))
    (mean.sum() + value.sum()).backward()
    optimizer.step()
    ckpt_save(path, policy, optimizer, torch.Generator().manual_seed(9), 3, {'charged': 1})
    return policy


def intent_blocks(state_rows, members, buffer, step, cfg):
    """{member vid: float intent block} (plain floats, the bank/runtime payload)."""
    table = {r['vid']: r for r in state_rows}
    return {vid: [float(t) for t in v2v_intent_features(table, vid, {}, buffer, step, cfg)]
            for vid in members}


def slot_view(block, member_index=0):
    """(K, STRIDE) view of one member's intent block (partner-major)."""
    arr = np.asarray(block, dtype=np.float32).reshape(K, STRIDE)
    return arr


def valid_bits(block, member_index=0):
    return slot_view(block)[..., STRIDE - 1]


def test_intent_load_extension_zero_init_and_actions_identical_at_load(tmp_path):
    path = tmp_path/'A_checkpoint_000.pt'
    fixture_checkpoint(path)
    # options off (default): the registered loader path, byte-identical state
    plain, _, _, _ = load_checkpoint(path)
    registered, _, _, _ = ckpt_load(path)
    for a, b in zip(plain.state_dict().values(), registered.state_dict().values()):
        assert torch.equal(a, b)
    assert not hasattr(plain, 'intent')
    # the intent binding keys: widened inputs AND the zero-initialised intent head
    ext, ext_opt, _, _ = load_checkpoint(path, config=INTENT_CFG)
    assert tuple(ext.actor[0].weight.shape) == (64, BASE_ACTOR_IN+V2V_INTENT_DIM)
    assert tuple(ext.critic[0].weight.shape) == (128, BASE_CRITIC_IN+V2V_INTENT_DIM)
    assert torch.equal(ext.actor[0].weight[:, 29:29+V2V_INTENT_DIM], torch.zeros(64, V2V_INTENT_DIM))
    assert torch.equal(ext.critic[0].weight[:, 29:29+V2V_INTENT_DIM], torch.zeros(128, V2V_INTENT_DIM))
    assert tuple(ext.intent.weight.shape) == (H, 64)
    assert torch.equal(ext.intent.weight, torch.zeros(H, 64))
    assert torch.equal(ext.intent.bias, torch.zeros(H))
    # 3 synthetic CAVs -> zero blocks at load -> executed actions identical to non-v2v
    context = encode(rows(0))
    assert context['members'] == ['cav0', 'cav1', 'cav2']
    zero_blocks = {vid: [0.]*V2V_INTENT_DIM for vid in context['members']}
    wide = widen_context(context, zero_blocks)
    assert len(wide['x'][0]) == 29+V2V_INTENT_DIM
    base, _, _, _ = ckpt_load(path)
    x = torch.as_tensor(wide['x'], dtype=torch.float32)
    rsu = torch.as_tensor(context['rsu'], dtype=torch.float32)
    from tools.coopmwm.v2v_model import forward_with_intent
    mean_ext, value_ext, planned = forward_with_intent(ext, x, rsu)
    mean_base, value_base = base(context)
    assert float((mean_ext-mean_base).detach().abs().max()) <= 1e-7
    assert float((value_ext-value_base).detach().abs().max()) <= 1e-7
    # the zero-initialised intent head plans the neutral zero command at load
    assert float(planned.detach().abs().max()) == 0.


def test_intents_published_at_t_observed_by_partners_at_t_plus_1(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path/'A_checkpoint_000.pt'
    fixture_checkpoint(path)
    ext, _, _, _ = load_checkpoint(path, config=INTENT_CFG)
    buffer = V2VIntentBuffer(horizon=H)
    cfg = V2VConfig(v2v=True, v2v_intent=True, v2v_intent_horizon=H)
    from tools.coopmwm.v2v_model import forward_with_intent
    for step in range(4):
        state = rows(step)
        context = encode(state)
        blocks = intent_blocks(state, context['members'], buffer, step, cfg)
        wide = widen_context(context, blocks)
        _, _, planned = forward_with_intent(ext, torch.as_tensor(wide['x'], dtype=torch.float32),
                                            torch.as_tensor(context['rsu'], dtype=torch.float32))
        # t=0: nothing published yet -> every partner slot masked (valid 0, intents 0)
        if step == 0:
            for vid in context['members']:
                assert float(valid_bits(blocks[vid]).max()) == 0.
        else:
            for vid in context['members']:
                bits = valid_bits(blocks[vid])
                # each ego's real partners (2 of the K=4 slots within 100 m) saw the t-1 broadcast
                assert float(bits[:2].min()) == 1. and float(bits[2:].max()) == 0.
        # policy publishes its planned commands at t (runner hook, post-step)
        published = publish_intents(buffer, step, {vid: [float(v) for v in row.detach()]
                                                   for vid, row in zip(context['members'], planned)})
        assert published == 3
    # one-step latency: the latest broadcast (t=3) is fresh at t=4, stale at t=5
    fresh = buffer.fresh('cav0', 4)
    assert fresh is not None and fresh.shape == (H,)
    assert buffer.fresh('cav0', 5) is None
    state4 = rows(4)
    mid = intent_blocks(state4, ['cav0', 'cav1', 'cav2'], buffer, 4, cfg)
    for vid in ('cav0', 'cav1', 'cav2'):
        bits = valid_bits(mid[vid])
        assert float(bits[:2].min()) == 1. and float(bits[2:].max()) == 0.
    late = intent_blocks(state4, ['cav0', 'cav1', 'cav2'], buffer, 5, cfg)
    for vid in ('cav0', 'cav1', 'cav2'):
        assert float(valid_bits(late[vid]).max()) == 0.
    # the intent payload of a FRESH broadcast is byte-exact the published row
    table = {r['vid']: r for r in state4}
    block = v2v_intent_features(table, 'cav1', {}, buffer, 4, cfg)
    view = slot_view(block)
    ego_row = np.asarray(buffer.fresh('cav0', 4), dtype=np.float32)
    assert np.array_equal(view[0, 6:6+H], ego_row)
    # option OFF: EXACTLY the plain v2v block (byte-identical)
    plain_cfg = V2VConfig(v2v=True)
    state = rows(2)
    table = {r['vid']: r for r in state}
    for vid in ('cav0', 'cav1', 'cav2'):
        assert np.array_equal(v2v_intent_features(table, vid, {}, buffer, 2, plain_cfg),
                              v2v_features(table, vid, {}, plain_cfg))
    assert intent_block_width(V2VConfig()) == V2V_DIM
    assert intent_block_width(cfg) == V2V_INTENT_DIM == K*STRIDE


def _bank_pairs(contexts, blocks, intent_targets, action):
    pairs = []
    for context, block, target in zip(contexts, blocks, intent_targets):
        wide = widen_context(context, block)
        sample = dict(context=dict(wide, v2v=block, intent_target=target),
                      behaviour=dict(executed=True, proposal_members=list(context['members'])),
                      action=[[action] for _ in context['members']])
        pairs.append((sample, 1.))
    return pairs


def test_one_full_pass_imitation_update_lowers_action_and_intent_error(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(101)
    path = tmp_path/'A_checkpoint_000.pt'
    fixture_checkpoint(path)
    ext, ext_opt, _, _ = load_checkpoint(path, config=INTENT_CFG)
    ext.__class__ = WidenedModule   # runtime-shaped forward (widened context routing)
    for group in ext_opt.param_groups:
        group['lr'] = 0.05   # test-only scale so ONE update is decisive on CPU
    cfg = V2VConfig(v2v=True, v2v_intent=True, v2v_intent_horizon=H)
    contexts, blocks, targets = [], [], []
    for step in range(6):
        state = rows(step)
        context = encode(state)
        block = intent_blocks(state, context['members'], None, step, cfg)
        target = {vid: dict(target=[0.5]*H, mask=[1.]*H) for vid in context['members']}
        contexts.append(context); blocks.append(block); targets.append(target)
    pairs = _bank_pairs(contexts, blocks, targets, 0.4)
    reference = copy.deepcopy(ext).eval()
    action_before = action_error(ext, pairs)
    intent_before = intent_error(ext, pairs)
    assert intent_before is not None and intent_before > 0.25
    v2v_before = ext.actor[0].weight[:, 29:29+V2V_INTENT_DIM].clone()
    head_before = ext.intent.weight.clone()
    report = run_separate(ext, ext_opt, pairs, torch.Generator().manual_seed(5),
                          reference, coefficient=1., passes=5,
                          intent=dict(weight=.1, horizon=H, bounds=INTENT_BOUNDS))
    assert report['epochs_completed'] >= 5
    assert report['action_error_after_mps2'] < action_before
    assert report['intent']['error_after_mps2'] < intent_before
    # the gradient really flowed through the v2v channels and the intent head
    assert float((ext.actor[0].weight[:, 29:29+V2V_INTENT_DIM]-v2v_before).abs().sum()) > 0.
    assert float((ext.intent.weight-head_before).abs().sum()) > 0.


def _episode(world_seed):
    context = encode(rows(0))
    sample, _ = Policy().sample(context, torch.Generator().manual_seed(7), deterministic=True)
    sample['behaviour'] = dict(regime='ordinary')
    return dict(world=dict(seed=world_seed, cell='c'), steps=[dict(time_s=0, reward=-1., sample=sample)])


def test_options_off_byte_identical_update_behaviour(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path/'A_checkpoint_000.pt'
    fixture_checkpoint(path)
    episodes = [_episode(1), _episode(2)]
    a, opt_a, _, _ = ckpt_load(path)
    b, opt_b, _, _ = ckpt_load(path)
    ga, gb = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    metrics_a = update(a, opt_a, episodes, ga)
    metrics_b = update(b, opt_b, episodes, gb, intent=None)   # the OFF path == shipped path
    assert metrics_a == metrics_b
    assert not any('intent' in m or 'intent_error' in m for m in metrics_a)
    for pa, pb in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(pa, pb)
