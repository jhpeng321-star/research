"""V2V intent-sharing model hooks - REHEARSAL STUB (part 5, branch c4/v2v-rehearsal).

LOCAL STUB of the part-2 model contract (branch c4/v2v-model,
tools/coopmwm/v2v_model.py): same interface (``extend_for_v2v(model,
extra_dim)``, ``load_checkpoint(path, config, v2v_dim)``, ``v2v_extra``,
``forward_with_v2v``) and the same zero-initialised widening semantics, so the
integrated module can replace this file unchanged. Used only while c4/v2v-int
has not published V2V_INT_READY.json.

Semantics (identical to the part-2 contract):

- ``extend_for_v2v(model, extra_dim)`` widens the FIRST actor and critic input
  layers by ``extra_dim`` ZERO-initialised input weights, inserted at
  ``INSERT_AT`` (= 29 sensor features): the actor input becomes
  ``[x(29) | v2v(extra) | latent(4)]`` and the critic input
  ``[pooled(29) | v2v(extra) | rsu(12)]``. Outputs on the original feature
  block are IDENTICAL to the loaded model.
- ``bind_forward(model, extra_dim)`` (rehearsal integration glue, the part-4
  surface) binds a v2v-aware forward that consumes ``context['v2v']``
  (``{member vid: features}``) and concatenates it at INSERT_AT; contexts
  without the key (plain PPO episodes, replayed prefixes) get zeros, so
  behaviour is unchanged while the widened inputs are actually exercised.
- ``load_for(path, config)`` shape-driven loader: extends/binds when the
  checkpoint is already widened, or when ``config`` enables v2v on a
  base-width checkpoint; byte-identical to ``accel_learner.load`` otherwise.
"""
import random

import numpy as np
import torch
from torch import nn

from .accel_learner import load as base_load
from .acceleration import Policy
from .v2v_obs import V2V_DIM  # observation part (rehearsal stub, same contract)

FEATURES = 29  # tools/coopnative/policy.py:9 sensor feature width, unchanged
ACTOR_LATENT = 4  # broadcast latent concatenated per member, policy.py:49
CRITIC_RSU = 12  # rsu block concatenated to pooled features, policy.py:50
INSERT_AT = FEATURES  # v2v channels join after the per-CAV x/pooled block,
# before the actor latent and critic rsu tails, which stay tail-aligned.


def v2v_extra(config=None, v2v_dim=None):
    """Resolve the widened-input width; 0 when the v2v flag is off (default)."""
    if v2v_dim is not None:
        return int(v2v_dim)
    if not config or not config.get('v2v'):
        return 0
    dim = config.get('v2v_dim')
    return int(dim) if dim is not None else int(V2V_DIM)


def _first_linear(sequence):
    for module in sequence:
        if isinstance(module, nn.Linear):
            return module
    raise ValueError('no input layer available to extend for v2v')


def _insert_columns(tensor, extra_dim):
    """Zero gap of extra_dim columns at INSERT_AT; the tail block shifts right."""
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


def extend_for_v2v(model, extra_dim):
    """Zero-extend the first actor and critic input layers by extra_dim.

    Existing weights and biases are copied unchanged and every new input
    weight is zero-initialised, so actor and critic outputs on the original
    feature block are IDENTICAL to the loaded model for any finite extra
    input. extra_dim<=0 is a no-op. Returns the extension record.
    """
    extra_dim = int(extra_dim)
    if extra_dim <= 0:
        return []
    return [dict(head=name, input_features=_widen_input_layer(_first_linear(getattr(model, name)), extra_dim),
                 extra_dim=extra_dim) for name in ('actor', 'critic')]


def forward_with_v2v(model, x, rsu, v2v=None):
    """Base actor/critic forward over widened per-CAV features x (n, F+extra).

    Mirrors tools/coopnative/policy.py:56 forward with the widened sensor
    block; ``v2v`` is the per-member extra block ``(n, extra_dim)`` (zeros when
    None). The broadcast latent stays per-member and the critic pools all
    member features against the same rsu block.
    """
    latent = model.broadcast(rsu).expand(len(x), -1)
    mean = model.actor(torch.cat((x, latent), -1))
    pooled = x.mean(0) if len(x) else torch.zeros(x.shape[1], dtype=x.dtype)
    value = model.critic(torch.cat((pooled, rsu))).squeeze(-1)
    return mean, value


def bind_forward(model, extra_dim):
    """Bind a v2v-aware forward consuming ``context['v2v']`` (integration glue).

    ``context['v2v']`` maps member vid -> features (list/tensor of extra_dim);
    members without an entry - or contexts without the key at all - get zeros,
    matching the zero-initialised extension columns. Only bound when extra_dim
    is positive; the plain forward is untouched otherwise.
    """
    extra_dim = int(extra_dim)
    if extra_dim <= 0:
        return None

    def forward(context, _model=model, _extra=extra_dim):
        x = torch.tensor(context['x'], dtype=torch.float32).reshape(-1, FEATURES)
        rsu = torch.tensor(context['rsu'], dtype=torch.float32)
        members = list(context.get('members') or [])
        block = context.get('v2v')
        if block is None:
            v = torch.zeros(len(x), _extra)
        else:
            rows = []
            for vid in members:
                row = block.get(vid)
                row = [0.0]*_extra if row is None else [float(a) for a in row]
                if len(row) != _extra:
                    raise ValueError(f'v2v block width {len(row)} != {_extra} for '+vid)
                rows.append(row)
            v = torch.tensor(rows, dtype=torch.float32).reshape(len(x), _extra)
        return forward_with_v2v(_model, torch.cat((x, v), -1), rsu)

    model.forward = forward
    model.v2v_extra_dim = extra_dim
    return forward


def _rebuilt_optimizer(policy, previous):
    """Fresh Adam over widened parameters with remapped moments."""
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
    optimizer.load_state_dict(dict(state=transferred, param_groups=state['param_groups']))
    return optimizer


def _checkpoint_widths(policy_state):
    heads = {}
    for head in ('actor', 'critic'):
        key = f'{head}.0.weight'
        if key not in policy_state:
            raise ValueError('v2v checkpoint lacks input layer '+key)
        heads[head] = int(policy_state[key].shape[1])
    return heads


BASE_INPUT = dict(actor=FEATURES+ACTOR_LATENT, critic=FEATURES+CRITIC_RSU)


def _load_extended_payload(payload, extra_dim):
    """Build a widened, bound policy from an already-widened checkpoint payload."""
    policy = Policy()
    extend_for_v2v(policy, extra_dim)
    bind_forward(policy, extra_dim)
    policy.load_state_dict(payload['policy'])
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    optimizer.load_state_dict(payload['optimizer'])
    generator = torch.Generator()
    generator.set_state(payload['generator'])
    torch.set_rng_state(payload['torch_rng'])
    np.random.set_state(payload['numpy_rng'])
    random.setstate(payload['python_rng'])
    return policy, optimizer, generator, payload


def load_for(path, config=None):
    """Shape-driven ``accel_learner.load``: v2v-aware, byte-identical by default.

    - Base-width checkpoint + v2v off (default): exactly ``accel_learner.load``.
    - Base-width checkpoint + ``config`` v2v on: load, widen by ``v2v_extra``,
      bind the v2v forward and remap the Adam moments (zero-gapped).
    - Widened checkpoint: load into a widened, bound policy of the same extra
      width regardless of ``config`` (collector reloads of newer checkpoints).
    """
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('schema') != 'coopnative/1':
        raise ValueError('native-only checkpoint required')
    widths = _checkpoint_widths(saved['policy'])
    if widths == BASE_INPUT:
        extra = v2v_extra(config)
        if extra <= 0:
            return base_load(path)
        policy, optimizer, generator, payload = base_load(path)
        extend_for_v2v(policy, extra)
        bind_forward(policy, extra)
        optimizer = _rebuilt_optimizer(policy, optimizer)
        return policy, optimizer, generator, payload
    extra = widths['actor']-BASE_INPUT['actor']
    if extra <= 0 or widths != {head: base+extra for head, base in BASE_INPUT.items()}:
        raise ValueError(f'v2v checkpoint width mismatch {widths} against base {BASE_INPUT}')
    return _load_extended_payload(saved, extra)


def load_checkpoint(path, config=None, v2v_dim=None):
    """Part-2 contract loader (config-driven); delegates to :func:`load_for`."""
    del v2v_dim  # resolved inside load_for through v2v_extra(config)
    return load_for(path, config)


def fit_policy(policy, weights):
    """Widen+bind ``policy`` in place to match ``weights``; no-op for base widths.

    Used wherever a fresh ``Policy()`` is about to ``load_state_dict`` weights
    that may be v2v-widened (accel_plant episodes, branch continuations).
    """
    extra = int(weights['actor.0.weight'].shape[1])-BASE_INPUT['actor']
    if extra <= 0:
        return policy
    if int(weights['critic.0.weight'].shape[1])-BASE_INPUT['critic'] != extra:
        raise ValueError('inconsistent v2v widths between actor and critic weights')
    extend_for_v2v(policy, extra)
    bind_forward(policy, extra)
    return policy
