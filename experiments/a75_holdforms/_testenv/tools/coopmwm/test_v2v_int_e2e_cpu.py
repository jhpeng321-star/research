"""PART 4 (INTEGRATOR) end-to-end CPU gate for the V2V intent-sharing build.

Chain under test, all CPU, no SUMO/GPU:
  1. save a real checkpoint shape (accel_learner.save, schema coopnative/1, warmed
     optimizer) -> v2v_model.load_checkpoint with the live-runner config keys
     (``config={'v2v': True, 'v2v_dim': V2V_DIM}``) -> zero-init widened extension;
  2. build the v2v observation block for a synthetic multi-CAV state (v2v_obs
     v2v_features: gap, lane offset, speed, accel, previous-second command, presence)
     -> widen the policy context (v2v_integrate.widen_context) -> run the policy ->
     actions byte-identical to the non-v2v load (zero-init identity);
  3. ONE imitation step on a synthetic bank pair with partner features
     (context['v2v'], accel_learner.imitation) -> the widened v2v input weights change.
"""
import torch

from tools.coopmwm.accel_learner import imitation, load as ckpt_load, save as ckpt_save
from tools.coopmwm.acceleration import Policy, encode
from tools.coopmwm.v2v_integrate import WidenedPolicy, policy_call, widen_context
from tools.coopmwm.v2v_model import V2V_DIM, load_checkpoint
from tools.coopmwm.v2v_obs import V2VConfig, v2v_features

BASE_INPUT = 29 + 4          # actor input: 29 sensor features + 4 broadcast latent


def synthetic_state():
    """Multi-CAV synthetic state in the whitelisted sensor schema (2 near CAVs,
    1 CAV on another lane, 1 HDV - the HDV is never a v2v partner)."""
    return [
        dict(vid='cav0', s_m=-30., v_mps=8., lane=0, is_cav=True, length_m=5.),
        dict(vid='cav1', s_m=-12., v_mps=9.5, lane=0, is_cav=True, length_m=5.),
        dict(vid='cav2', s_m=25., v_mps=10., lane=1, is_cav=True, length_m=5.),
        dict(vid='hdv9', s_m=-50., v_mps=7., lane=2, is_cav=False, length_m=5.),
    ]


LAST_COMMANDS = {'cav0': .4, 'cav1': -1.2, 'cav2': 0.}   # previous-second commands


def v2v_blocks(state, members):
    cfg = V2VConfig(v2v=True)
    table = {row['vid']: row for row in state}
    return {vid: [float(t) for t in v2v_features(table, vid, LAST_COMMANDS, cfg)]
            for vid in members}


def fixture_checkpoint(path):
    """A real-shape checkpoint: warmed Adam moments, nonzero update id, charged bookkeeping."""
    torch.set_num_threads(1)
    torch.manual_seed(47)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    mean, value = policy(encode(synthetic_state()))
    (mean.sum() + value.sum()).backward()
    optimizer.step()
    ckpt_save(path, policy, optimizer, torch.Generator().manual_seed(9), 3, {'charged': 1})
    return policy


def test_v2v_load_extension_and_actions_identical_to_non_v2v_at_load(tmp_path):
    path = tmp_path/'A_checkpoint_000.pt'
    fixture_checkpoint(path)
    # flag off: the registered loader path, byte-identical state
    base, _, _, _ = load_checkpoint(path)
    registered, _, _, _ = ckpt_load(path)
    for a, b in zip(base.state_dict().values(), registered.state_dict().values()):
        assert torch.equal(a, b)
    # the live-runner keys: config={'v2v': True, 'v2v_dim': 24}
    ext, ext_opt, _, _ = load_checkpoint(path, config={'v2v': True, 'v2v_dim': V2V_DIM})
    assert tuple(ext.actor[0].weight.shape) == (64, BASE_INPUT+V2V_DIM)
    assert tuple(ext.critic[0].weight.shape) == (128, 29+12+V2V_DIM)
    # the extension is zero-initialised exactly on the new v2v channels
    assert torch.equal(ext.actor[0].weight[:, 29:29+V2V_DIM], torch.zeros(64, V2V_DIM))
    assert torch.equal(ext.critic[0].weight[:, 29:29+V2V_DIM], torch.zeros(128, V2V_DIM))
    # synthetic multi-CAV state -> v2v observation block -> widened policy context
    state = synthetic_state()
    context = encode(state)
    assert context['members'] == ['cav0', 'cav1', 'cav2']
    wide = widen_context(context, v2v_blocks(state, context['members']))
    assert len(wide['x'][0]) == 29+V2V_DIM
    # actions byte-identical to the non-v2v load (zero init kills the new channels)
    mean_ext, value_ext = policy_call(ext, wide)
    mean_base, value_base = base(context)
    assert float((mean_ext-mean_base).detach().abs().max()) <= 1e-7
    assert float((value_ext-value_base).detach().abs().max()) <= 1e-7


def test_one_imitation_step_on_synthetic_bank_pair_changes_v2v_weights(tmp_path):
    path = tmp_path/'A_checkpoint_000.pt'
    fixture_checkpoint(path)
    ext, ext_opt, _, _ = load_checkpoint(path, config={'v2v': True, 'v2v_dim': V2V_DIM})
    wrapper = WidenedPolicy(ext)
    state = synthetic_state()
    context = encode(state)
    wide = widen_context(context, v2v_blocks(state, context['members']))
    members = context['members']
    # one bank pair with partner features (the BANK payload: context['v2v'], weight > 0)
    sample = dict(context=dict(wide, v2v=v2v_blocks(state, members)),
                  behaviour=dict(executed=True, proposal_members=list(members)),
                  action=[[.3], [-.7], [.1]])
    new_before = ext.actor[0].weight[:, 29:29+V2V_DIM].clone()
    old_before = ext.actor[0].weight[:, :29].clone()
    loss = imitation(wrapper, sample, 1.)
    ext_opt.zero_grad()
    loss.backward()
    ext_opt.step()          # exactly ONE imitation step
    new_after = ext.actor[0].weight[:, 29:29+V2V_DIM]
    assert float((new_after-new_before).abs().sum()) > 0.
    # the imitation gradient really flowed through the widened context
    assert float((ext.actor[0].weight[:, :29]-old_before).abs().sum()) > 0.
