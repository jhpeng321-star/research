"""V2V intent-sharing warm start: widened actor/critic inputs, zero-initialised.

Per-CAV policy and critic live in tools/coopmwm/acceleration.py (Policy,
subclassing tools/coopnative/policy.py:45 with actor :49 and critic :50) and
checkpoints load through tools/coopmwm/accel_learner.py (the Policy rebinding
of tools/coopnative/learner.py:68 load). This module adds:

  extend_for_v2v(model, extra_dim)   widen the first actor and critic input
                                     layers by extra_dim with ZERO-initialised
                                     new input weights, so outputs are
                                     IDENTICAL to the original at load;
   load_checkpoint(..., v2v_dim=N)    registered loader path that extends the
                                      policy after loading and remaps the Adam
                                      state with zero-gapped moments;
  forward_with_v2v(model, x, rsu)    base actor/critic forward over widened
                                     per-CAV features for the observation side.

Everything sits behind the `v2v` config flag (default off): with the flag off
load_checkpoint is byte-identical to the registered accel_learner.load.

INTENT HEAD (part B, branch c4/intent-head): behind the `v2v_intent` option
(default off) the model also carries a small zero-initialised linear INTENT
HEAD off the shared actor trunk. It outputs the agent's PLANNED absolute
acceleration commands for the next H=5 control steps (same units and clip as
the executed action), trained by a supervised loss to the agent's own REALISED
next-H commands from its rollout (masked at episode end / despawn). The head
never feeds the executed action, so a loaded MAPPO u0 acts identically.
"""
import random

import numpy as np
import torch
from torch import nn
from .accel_learner import load as base_load
from .acceleration import Policy

try:
    from .v2v_obs import V2V_DIM  # observation part, branch c4/v2v-intent
except ImportError:
    V2V_DIM = 4  # local stub until tools/coopmwm/v2v_obs.py lands
try:
    from .v2v_obs import V2V_INTENT_DIM as OBS_V2V_INTENT_DIM  # c4/intent-obs
    from .v2v_obs import V2V_INTENT_H as OBS_V2V_INTENT_H
except ImportError:
    OBS_V2V_INTENT_DIM = None  # local constant below until that branch lands
    OBS_V2V_INTENT_H = None

FEATURES = 29  # tools/coopnative/policy.py:9 sensor feature width, unchanged
ACTOR_LATENT = 4  # broadcast latent concatenated per member, policy.py:49
CRITIC_RSU = 12  # rsu block concatenated to pooled features, policy.py:50
INSERT_AT = FEATURES  # v2v channels join after the per-CAV x/pooled block,
# before the actor latent and critic rsu tails, which stay tail-aligned.

# --- intent option (part B); block contract shared with branch c4/intent-obs ---
V2V_INTENT_K = 4  # nearest CAV partners, same slots/ordering as v2v_obs.V2V_K
V2V_INTENT_H = OBS_V2V_INTENT_H if OBS_V2V_INTENT_H is not None else 5  # planned commands, 1 s each
#: Per-partner base block width (gap, lane offset, speed, accel, last_cmd, present).
V2V_INTENT_BASE = 6
#: Registered default clip for planned commands = the action clip (binding accel_bounds).
INTENT_BOUNDS = (-3., 2.6)
#: Registered supervised weight added to the PPO objective (config v2v_intent_weight).
DEFAULT_INTENT_WEIGHT = .1


def intent_v2v_dim(horizon=V2V_INTENT_H):
    """Width of the intent v2v block: per partner the existing 6 features + H
    intent values + one intent_valid bit (absent/stale mask), for K partners."""
    if int(horizon) == V2V_INTENT_H and OBS_V2V_INTENT_DIM is not None:
        return int(OBS_V2V_INTENT_DIM)  # part A's landed constant, byte-exact
    return int(V2V_INTENT_K)*(V2V_INTENT_BASE+int(horizon)+1)


V2V_INTENT_DIM = intent_v2v_dim()  # 48 at the registered horizon 5


def intent_options(config=None):
    """(horizon, weight) for the intent option; (0, 0.) when off (default)."""
    if not config or not config.get('v2v_intent'):
        return 0, 0.
    horizon = int(config.get('v2v_intent_horizon') or V2V_INTENT_H)
    weight = config.get('v2v_intent_weight')
    return horizon, float(weight) if weight is not None else DEFAULT_INTENT_WEIGHT


def intent_config(binding=None):
    """Registered intent-loss configuration for a run binding; None when off.

    The supervised clip is the binding's own action clip (accel_bounds), never a
    hand-set behavioural threshold; weight defaults to DEFAULT_INTENT_WEIGHT.
    """
    horizon, weight = intent_options(binding)
    if horizon <= 0:
        return None
    bounds = tuple(binding.get('accel_bounds') or INTENT_BOUNDS) if binding else INTENT_BOUNDS
    return dict(horizon=horizon, weight=weight, bounds=bounds)


def v2v_extra(config=None, v2v_dim=None):
    """Resolve the widened-input width; 0 when both flags are off (default).

    The intent block contains the plain 6-feature partner block, so a truthy
    `v2v_intent` supersedes `v2v` and widens by the K*(6+H+1) intent width.
    """
    if v2v_dim is not None:
        return int(v2v_dim)
    if not config:
        return 0
    horizon, _ = intent_options(config)
    if horizon > 0:
        return intent_v2v_dim(horizon)
    if not config.get('v2v'):
        return 0
    dim = config.get('v2v_dim')
    return int(dim) if dim is not None else int(V2V_DIM)


def _first_linear(sequence):
    for module in sequence:
        if isinstance(module, nn.Linear):
            return module
    raise ValueError('no input layer available to extend for v2v')


def _insert_columns(tensor, extra_dim):
    """Zero gap of extra_dim columns at INSERT_AT; the tail block shifts right.

    Used for input weights and for optimizer moments alike: the actor input is
    (x | latent) and the critic input is (pooled | rsu), so the new v2v
    channels must sit between the x/pooled block and that tail for the copied
    weights to keep multiplying the same features.
    """
    rows = tensor.shape[:-1]
    out = torch.zeros(*rows, tensor.shape[-1]+extra_dim, dtype=tensor.dtype)
    out[..., :INSERT_AT] = tensor[..., :INSERT_AT]
    out[..., INSERT_AT+extra_dim:] = tensor[..., INSERT_AT:]
    return out


def _widen_input_layer(layer, extra_dim):
    """Copy the input layer into a wider one; new input weights are ZERO."""
    original = int(layer.weight.shape[1])
    layer.weight = nn.Parameter(_insert_columns(layer.weight.detach(), extra_dim))
    return original


def extend_for_v2v(model, extra_dim, intent_horizon=0, intent_bounds=INTENT_BOUNDS):
    """Zero-extend the first actor and critic input layers by extra_dim.

    Existing weights and biases are copied unchanged and every new input
    weight is zero-initialised, so actor and critic outputs on the original
    feature block are IDENTICAL to the loaded model for any finite extra
    input. extra_dim<=0 is a no-op. With intent_horizon>0 a zero-initialised
    intent head is also attached (new module, appended last: parameter order of
    the loaded policy is unchanged, so the executed action path is untouched).
    Returns the extension record.
    """
    extra_dim = int(extra_dim)
    if extra_dim <= 0 and int(intent_horizon) <= 0:
        return []
    record = []
    if extra_dim > 0:
        record = [dict(head=name, input_features=_widen_input_layer(_first_linear(getattr(model, name)), extra_dim),
            extra_dim=extra_dim) for name in ('actor', 'critic')]
    if int(intent_horizon) > 0:
        add_intent_head(model, int(intent_horizon), intent_bounds)
        record.append(dict(head='intent', horizon=int(intent_horizon)))
    return record


def add_intent_head(model, horizon, bounds=INTENT_BOUNDS):
    """Small zero-initialised linear intent head off the shared actor trunk.

    The trunk is the actor stack without its output layer; the head maps the
    same 64-wide trunk features to H planned absolute acceleration commands,
    clipped to `bounds` (the action clip). Zero init keeps the initial plan at
    the neutral zero command and contributes no gradient to the trunk at load,
    so the executed action is IDENTICAL to the plain loaded policy.
    """
    if hasattr(model, 'intent'):
        raise ValueError('policy already carries an intent head')
    width = int(model.actor[-1].weight.shape[1])
    head = nn.Linear(width, int(horizon))
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    model.intent = head  # appended last: existing parameter indices unchanged
    model.register_buffer('intent_low', torch.tensor(float(bounds[0])), persistent=False)
    model.register_buffer('intent_high', torch.tensor(float(bounds[1])), persistent=False)
    return head


def forward_with_intent(model, x, rsu):
    """(mean, value, planned) over widened per-CAV features x (n, F+extra).

    mean/value mirror forward_with_v2v exactly; `planned` (n, H) is the intent
    head read off the shared actor trunk, clipped to the action bounds. Rows
    narrower than the widened input are zero-padded (absent v2v channels read
    as zeros, matching the zero-initialised input columns).
    """
    if not hasattr(model, 'intent'):
        raise ValueError('policy has no intent head')
    x = torch.as_tensor(x, dtype=torch.float32)
    need = int(model.actor[0].weight.shape[1])-ACTOR_LATENT
    if x.dim() != 2 or x.shape[1] > need:
        raise ValueError('intent input width mismatch')
    if x.shape[1] < need:
        x = torch.cat((x, torch.zeros(len(x), need-x.shape[1])), -1)
    latent = model.broadcast(torch.as_tensor(rsu, dtype=torch.float32)).expand(len(x), -1)
    hidden = model.actor[:-1](torch.cat((x, latent), -1))
    mean = model.actor[-1](hidden)
    pooled = x.mean(0) if len(x) else torch.zeros(x.shape[1], dtype=x.dtype)
    value = model.critic(torch.cat((pooled, torch.as_tensor(rsu, dtype=torch.float32)))).squeeze(-1)
    planned = model.intent(hidden).clamp(model.intent_low, model.intent_high)
    return mean, value, planned


def planned_commands(model, context):
    """Intent plan (n, H) for a rollout/bank context (widened or plain)."""
    x = context.get('x') or ()
    width = int(model.actor[0].weight.shape[1])-ACTOR_LATENT
    rows = torch.as_tensor(x, dtype=torch.float32) if len(x) else torch.zeros((0, width))
    return forward_with_intent(model, rows, context['rsu'])[2]


def intent_examples(episodes, horizon, bounds=INTENT_BOUNDS):
    """Realised next-H command targets aligned with the update records.

    For the record of episode step i, member v: target[j] is the absolute
    acceleration command realised at step i+1+j (requests' requested_raw, else
    the baseline command - the accel_runner absolute_accels convention),
    clipped to `bounds`; mask[j] is 0 beyond the episode end or once v is no
    longer commanded (despawn). Returns (targets, masks), flattened exactly
    like accel_learner.update flattens its records.
    """
    horizon = int(horizon)
    low, high = float(bounds[0]), float(bounds[1])
    targets, masks = [], []
    for ep in episodes:
        steps = ep['steps']
        for i, s in enumerate(steps):
            members = s['sample']['context']['members']
            target = torch.zeros((len(members), horizon))
            mask = torch.zeros((len(members), horizon))
            for m, vid in enumerate(members):
                for j in range(1, horizon+1):
                    if i+j >= len(steps):
                        break
                    nxt = steps[i+j]
                    if vid not in (nxt.get('commands') or {}):
                        continue
                    row = (nxt.get('requests') or {}).get(vid)
                    value = float(row['requested_raw']) if row else float(nxt['commands'][vid])
                    target[m, j-1] = min(max(value, low), high)
                    mask[m, j-1] = 1.
            targets.append(target)
            masks.append(mask)
    return targets, masks


def _intent_forward(model, samples, targets, masks):
    for sample, target, mask in zip(samples, targets, masks):
        planned = planned_commands(model, sample['context'])
        yield planned, torch.as_tensor(target, dtype=torch.float32), torch.as_tensor(mask, dtype=torch.float32)


def intent_loss(model, samples, targets, masks):
    """Masked mean squared command error (m/s^2)^2 between plan and realisation."""
    terms = [(planned-target).square()*mask for planned, target, mask in _intent_forward(model, samples, targets, masks)]
    if not terms:
        return torch.zeros(())
    total = sum(float(m.sum()) for m in masks)
    return sum(t.sum() for t in terms)/max(total, 1.)


@torch.no_grad()
def intent_error(model, samples, targets, masks):
    """Masked mean absolute command error in m/s^2; None where nothing is masked in."""
    terms = [(planned-target).abs()*mask for planned, target, mask in _intent_forward(model, samples, targets, masks)]
    total = sum(float(m.sum()) for m in masks)
    if not total:
        return None
    return sum(float(t.sum()) for t in terms)/total


def forward_with_v2v(model, x, rsu):
    """Base actor/critic forward over widened per-CAV features x (n, F+extra).

    Mirrors tools/coopnative/policy.py:56 forward with the widened sensor
    block; the broadcast latent stays per-member and the critic pools all
    member features against the same rsu block.
    """
    latent = model.broadcast(rsu).expand(len(x), -1)
    mean = model.actor(torch.cat((x, latent), -1))
    pooled = x.mean(0) if len(x) else torch.zeros(x.shape[1], dtype=x.dtype)
    value = model.critic(torch.cat((pooled, rsu))).squeeze(-1)
    return mean, value


def _rebuilt_optimizer(policy, previous):
    """Fresh Adam over widened parameters with remapped moments.

    extend_for_v2v only widens the two input-layer weight tensors in place, so
    module registration order - and therefore optimizer state indices - are
    unchanged; widened moment tensors are zero-gapped at INSERT_AT exactly like
    the weights and step counters are preserved. Newly attached heads (the
    intent head) have no saved moments: they start fresh, and their indices are
    appended to the saved parameter group so the sizes match again.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    state = previous.state_dict()
    parameters = list(policy.parameters())
    transferred = {}
    for index, entry in state['state'].items():
        shape = parameters[index].shape
        transferred[index] = {key: value if not torch.is_tensor(value) or value.dim() == 0
            or value.shape[-1] == shape[-1] or value.shape[:-1] != shape[:-1]
            else _insert_columns(value, shape[-1]-value.shape[-1])
            for key, value in entry.items()}
    listed = {index for group in state['param_groups'] for index in group['params']}
    missing = [index for index in range(len(parameters)) if index not in listed]
    if missing:
        anchor = max(listed, default=-1)
        group = next((g for g in state['param_groups'] if anchor in g['params']), state['param_groups'][0])
        group['params'] = list(group['params'])+missing
    optimizer.load_state_dict(dict(state=transferred, param_groups=state['param_groups']))
    return optimizer


BASE_INPUT = dict(actor=FEATURES+ACTOR_LATENT, critic=FEATURES+CRITIC_RSU)


INTENT_STATE_KEYS = {'intent.weight', 'intent.bias'}  # absent from pre-intent checkpoints


def _load_extended(path, payload, extra_dim, intent_horizon=0, intent_bounds=INTENT_BOUNDS):
    """learner.load against an already-widened checkpoint (learner.py:68 path)."""
    policy = Policy()
    extend_for_v2v(policy, extra_dim, intent_horizon, intent_bounds)
    result = policy.load_state_dict(payload['policy'], strict=False)
    if result.unexpected_keys:
        raise ValueError('unexpected checkpoint keys '+repr(sorted(result.unexpected_keys)))
    if any(key not in INTENT_STATE_KEYS for key in result.missing_keys):
        raise ValueError('checkpoint missing non-intent keys '+repr(sorted(result.missing_keys)))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    optimizer.load_state_dict(payload['optimizer'])
    generator = torch.Generator()
    generator.set_state(payload['generator'])
    torch.set_rng_state(payload['torch_rng'])
    np.random.set_state(payload['numpy_rng'])
    random.setstate(payload['python_rng'])
    return policy, optimizer, generator, payload


def load_checkpoint(path, config=None, v2v_dim=None):
    """Registered loader, extended after loading when the v2v/v2v_intent flag is on.

    Flag off (default) this is byte-identical to accel_learner.load. With
    v2v_dim=N (or config={'v2v': True}) the loaded policy is widened by N
    zero-initialised inputs and the optimizer moments are remapped. With
    config={'v2v_intent': True, 'v2v_intent_horizon': H} the input is widened by
    the K*(6+H+1) intent width and the zero-initialised intent head is attached;
    both leave the executed action identical at load. Loading an
    already-extended checkpoint of the same width is accepted unchanged (intent
    head weights are taken from the checkpoint when present, else zero-init).
    """
    extra_dim = v2v_extra(config, v2v_dim)
    horizon, _ = intent_options(config)
    if extra_dim <= 0:
        return base_load(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    widths = {head: int(saved['policy'][f'{head}.0.weight'].shape[1])
              for head in ('actor', 'critic')}
    if widths == {head: base+extra_dim for head, base in BASE_INPUT.items()}:
        return _load_extended(path, saved, extra_dim, horizon)
    if widths != BASE_INPUT:
        raise ValueError('v2v checkpoint width mismatch '
            f'{widths} against base {BASE_INPUT} and extra_dim {extra_dim}')
    policy, optimizer, generator, payload = base_load(path)
    extend_for_v2v(policy, extra_dim, horizon)
    optimizer = _rebuilt_optimizer(policy, optimizer)
    return policy, optimizer, generator, payload
