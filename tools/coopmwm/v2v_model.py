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
    the weights and step counters are preserved.
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
    optimizer.load_state_dict(dict(state=transferred, param_groups=state['param_groups']))
    return optimizer


BASE_INPUT = dict(actor=FEATURES+ACTOR_LATENT, critic=FEATURES+CRITIC_RSU)


def _load_extended(path, payload, extra_dim):
    """learner.load against an already-widened checkpoint (learner.py:68 path)."""
    policy = Policy()
    extend_for_v2v(policy, extra_dim)
    policy.load_state_dict(payload['policy'])
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    optimizer.load_state_dict(payload['optimizer'])
    generator = torch.Generator()
    generator.set_state(payload['generator'])
    torch.set_rng_state(payload['torch_rng'])
    np.random.set_state(payload['numpy_rng'])
    random.setstate(payload['python_rng'])
    return policy, optimizer, generator, payload


def load_checkpoint(path, config=None, v2v_dim=None):
    """Registered loader, extended after loading when the v2v flag is on.

    Flag off (default) this is byte-identical to accel_learner.load. With
    v2v_dim=N (or config={'v2v': True}) the loaded policy is widened by N
    zero-initialised inputs and the optimizer moments are remapped. Loading an
    already-extended checkpoint of the same width is accepted unchanged.
    """
    extra_dim = v2v_extra(config, v2v_dim)
    if extra_dim <= 0:
        return base_load(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    widths = {head: int(saved['policy'][f'{head}.0.weight'].shape[1])
              for head in ('actor', 'critic')}
    if widths == {head: base+extra_dim for head, base in BASE_INPUT.items()}:
        return _load_extended(path, saved, extra_dim)
    if widths != BASE_INPUT:
        raise ValueError('v2v checkpoint width mismatch '
            f'{widths} against base {BASE_INPUT} and extra_dim {extra_dim}')
    policy, optimizer, generator, payload = base_load(path)
    extend_for_v2v(policy, extra_dim)
    optimizer = _rebuilt_optimizer(policy, optimizer)
    return policy, optimizer, generator, payload
