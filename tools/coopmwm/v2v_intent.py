"""Planned-command intent sharing for the V2V policy (option ``v2v_intent``, DEFAULT OFF).

MAPPO-PIS-like (arXiv 2408.06656) but learned, identical for A/B/R-v2v:

- POLICY: :class:`IntentPolicy` is the v2v-widened per-CAV policy plus an INTENT
  HEAD, one zero-initialised ``Linear(64, H)`` off the shared actor trunk (the
  64-wide hidden activation that feeds the unchanged action head).  It predicts
  the next ``H`` planned commands (absolute acceleration, m/s^2, seconds
  t+1..t+H, the same unit as the executed command; the published value is
  clipped to the registered ``accel_bounds``).  The action path is untouched:
  at load from MAPPO u0 the zero-initialised input columns and the separate head
  keep the executed actions identical to the current v2v policy.
- BROADCAST: after each policy step every CAV publishes its intent
  (``sample['published_intent']``); at step t an ego observes, per partner slot,
  the intent published at t-1 plus ``intent_valid`` (tools/coopmwm/v2v_obs.py
  ``v2v_intent_features``).  Per-partner features 6 + H + 1.
- TRAINING: masked MSE between the intent head at step t and the agent's own
  REALISED commands at t+1..t+H (``step_commands``: the issued pre-safety
  absolute command = native/MAPPO command + residual, or the plan's absolute
  command; clipped to the action bounds).  Positions past the episode end, after
  a despawn or across a missing second are masked.  Added to the PPO loss with
  ``v2v_intent_loss_weight`` (default 1.0).  For B/R bank pairs the plan's own
  future commands (inside the plan window) are intent targets in the imitation
  phase.  Bank partner intents are NEVER recorded future commands: each update
  (exact_grade.exact_update) recomputes them with a no-grad pass of the CURRENT
  intent head on each partner's own t-1 bank context (:func:`refresh_bank_intents`),
  exactly as a deployed partner publishes at t-1; a partner whose t-1 context is
  not in the bank entry reads intent_valid=0 with zero intents.
- TRUNK: with ``v2v_intent_detach_trunk`` (DEFAULT True) the intent loss trains
  the intent head only (trunk features detached), so it cannot move the action
  trunk, the PPO KL or the imitation KL; the intent gradient share is logged.

COOPERATION SIGNAL (option ``v2v_coop_signal``, DEFAULT OFF, requires
``v2v_intent``): the policy additionally gets a zero-initialised coop head
(``Linear(64, 1+K+1)``) off the same trunk: one logit for ``p_coop`` (sigmoid;
"I am in a cooperative manoeuvre") and ``K+1`` yield-slot logits (softmax over
the K partner slots + none; "I am giving way to this partner").  It is
broadcast with the SAME t-1 latency as the intents (``sample['published_coop']``);
partners observe two extra features per slot (``partner_p_coop`` and whether
they are the yield target; tools/coopmwm/v2v_obs.py, block width
``v2v_coop_dim(K, H)`` = 56 for K=4, H=5) through ZERO-INITIALISED input
columns, so the loaded policy acts identically.  TRAINING TARGETS: bank pairs
of entries whose plan is labelled cooperative within its seed only -- members
of that plan get ``p_coop=1`` and yield target = the plan member whose gap
their deceleration opens (the nearest other plan member in the SAME lane
BEHIND them at the plan-start trace frame, when the member's plan window
contains a deceleration; sign rule, no behavioural threshold; else none), the
slot resolved through the pair's recorded partner ranking (not among the
partner slots -> none).  All other pairs and PPO rollouts get NO coop
supervision (masked; nothing pushes p_coop down outside cooperative plans).
The coop head shares the intent trunk-detach setting and is added to the
imitation loss with ``v2v_coop_signal_loss_weight`` (default 1.0).  Arms
without a bank (A) never see a coop target: their coop head stays
unsupervised (documented fairness consequence; see COOPSIG_READY.json).

Config keys (binding): ``v2v_intent`` (bool, default False), ``v2v_intent_horizon``
(int, default 5), ``v2v_intent_loss_weight`` (float, default 1.0 when on),
``v2v_intent_detach_trunk`` (bool, default True),
``v2v_coop_signal`` (bool, default False; requires ``v2v_intent``),
``v2v_coop_signal_loss_weight`` (float, default 1.0 when on),
``v2v_dim`` must equal the block width (``v2v_intent_dim(K, H)`` = 48 for K=4, H=5;
``v2v_coop_dim(K, H)`` = 56 with the coop signal).
"""
import math

import numpy as np
import torch
from torch import nn

from .acceleration import Policy as BasePolicy
from .v2v_integrate import widen_context
from .v2v_model import FEATURES, extend_for_v2v
from .v2v_obs import (V2V_COOP_SIGNAL_FEATURES, V2V_INTENT_HORIZON, V2V_K, V2V_PARTNER_FEATURES,
                      coop_columns, intent_columns, v2v_coop_dim, v2v_intent_dim)

DEFAULT_LOSS_WEIGHT = 1.
#: Intent loss trains the intent head only (trunk features detached) unless the binding says False.
DEFAULT_DETACH_TRUNK = True
#: Coop-signal loss weight when ``v2v_coop_signal`` is on.
DEFAULT_COOP_LOSS_WEIGHT = 1.
#: Registered acceleration bounds (accel_deploy / binding ``accel_bounds``); the binding wins.
ACCEL_BOUNDS = (-3., 2.6)
TARGET_RULE = ('own issued pre-safety absolute command at t+1..t+H (native+residual or plan '
               'absolute), clipped to accel_bounds; masked past episode end/despawn/missing second')
COOP_TARGET_RULE = ('bank pairs of plans labelled cooperative within their seed only: plan members '
                    'get p_coop=1 and yield target = nearest other plan member in the SAME lane '
                    'BEHIND them at the plan-start trace frame when their plan window contains a '
                    'deceleration (min command < 0, sign rule; slot resolved through the recorded '
                    'partner ranking, not among the slots -> none); all other pairs and PPO rollouts '
                    'are unsupervised (masked)')


def settings(cfg):
    """Resolved intent settings from a binding, or None when ``v2v_intent`` is off (default)."""
    if not cfg or not cfg.get('v2v_intent'):
        if cfg and cfg.get('v2v_coop_signal'):
            raise ValueError('v2v_coop_signal requires v2v_intent=True')
        return None
    horizon = int(cfg.get('v2v_intent_horizon', V2V_INTENT_HORIZON))
    weight = float(cfg.get('v2v_intent_loss_weight', DEFAULT_LOSS_WEIGHT))
    bounds = tuple(float(b) for b in (cfg.get('accel_bounds') or ACCEL_BOUNDS))
    if horizon < 1:
        raise ValueError('v2v_intent_horizon must be >= 1')
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('v2v_intent_loss_weight must be finite and >= 0')
    if len(bounds) != 2 or not bounds[0] < bounds[1]:
        raise ValueError('accel_bounds must be (low, high)')
    detach = cfg.get('v2v_intent_detach_trunk', DEFAULT_DETACH_TRUNK)
    if not isinstance(detach, bool):
        raise ValueError('v2v_intent_detach_trunk must be a bool')
    coop = bool(cfg.get('v2v_coop_signal'))
    coop_weight = 0.
    if coop:
        coop_weight = float(cfg.get('v2v_coop_signal_loss_weight', DEFAULT_COOP_LOSS_WEIGHT))
        if not math.isfinite(coop_weight) or coop_weight < 0:
            raise ValueError('v2v_coop_signal_loss_weight must be finite and >= 0')
    dim = v2v_coop_dim(V2V_K, horizon) if coop else v2v_intent_dim(V2V_K, horizon)
    if cfg.get('v2v_dim') is not None and int(cfg['v2v_dim']) != dim:
        label = 'coop signal block width' if coop else 'intent block width'
        raise ValueError(f'v2v_dim {cfg["v2v_dim"]} != {label} {dim}')
    return dict(horizon=horizon, loss_weight=weight, bounds=bounds, k=V2V_K, dim=dim,
                detach_trunk=detach, coop=coop, coop_loss_weight=coop_weight)


def config_keys(horizon=V2V_INTENT_HORIZON, loss_weight=DEFAULT_LOSS_WEIGHT, detach_trunk=None,
                coop=False, coop_loss_weight=DEFAULT_COOP_LOSS_WEIGHT):
    """Exact binding keys for an intent-ON arm (identical for A, B and R)."""
    dim = v2v_coop_dim(V2V_K, horizon) if coop else v2v_intent_dim(V2V_K, horizon)
    keys = dict(v2v=True, v2v_dim=dim, v2v_intent=True,
                v2v_intent_horizon=int(horizon), v2v_intent_loss_weight=float(loss_weight),
                v2v_intent_detach_trunk=DEFAULT_DETACH_TRUNK if detach_trunk is None else bool(detach_trunk))
    if coop:
        keys.update(v2v_coop_signal=True, v2v_coop_signal_loss_weight=float(coop_loss_weight))
    return keys


# ---------------------------------------------------------------- policy

class IntentPolicy(BasePolicy):
    """v2v-widened per-CAV policy (block width K*(6+H+1), +2 per slot with the coop signal)
    with a planned-command intent head and, when ``coop`` is on, a cooperation-signal head."""

    def __init__(self, horizon=V2V_INTENT_HORIZON, loss_weight=DEFAULT_LOSS_WEIGHT, bounds=ACCEL_BOUNDS,
                 detach_trunk=DEFAULT_DETACH_TRUNK, coop=False, coop_loss_weight=DEFAULT_COOP_LOSS_WEIGHT):
        super().__init__()
        extend_for_v2v(self, v2v_coop_dim(V2V_K, horizon) if coop else v2v_intent_dim(V2V_K, horizon))
        self._attach_intent(horizon, loss_weight, bounds, detach_trunk, coop, coop_loss_weight)

    def _attach_intent(self, horizon, loss_weight, bounds, detach_trunk=DEFAULT_DETACH_TRUNK,
                       coop=False, coop_loss_weight=DEFAULT_COOP_LOSS_WEIGHT):
        self.intent_horizon, self.intent_k = int(horizon), V2V_K
        self.intent_detach_trunk = bool(detach_trunk)
        self.v2v_width = (v2v_coop_dim(V2V_K, horizon) if coop else v2v_intent_dim(V2V_K, horizon))
        self.intent_loss_weight = float(loss_weight)
        self.coop_enabled = bool(coop)
        self.coop_loss_weight = float(coop_loss_weight)
        hidden = int(self.actor[-1].in_features)
        with torch.random.fork_rng(devices=[]):  # no global RNG side effect
            head = nn.Linear(hidden, self.intent_horizon)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.intent_head = head
        if self.coop_enabled:
            # [p_coop logit, K partner-slot yield logits + none]; zero init keeps the
            # loaded policy's actions identical (and p_coop=0.5, uniform yield).
            with torch.random.fork_rng(devices=[]):
                coop_head = nn.Linear(hidden, 1+self.intent_k+1)
            nn.init.zeros_(coop_head.weight)
            nn.init.zeros_(coop_head.bias)
            self.coop_head = coop_head
        self.register_buffer('intent_bounds', torch.tensor([float(b) for b in bounds], dtype=torch.float32))

    def _inputs(self, context):
        x = context['x']
        if x and len(x[0]) == FEATURES:
            blocks = context.get('v2v')
            if not blocks:
                raise ValueError('V2V policy requires recorded partner observations')
            context = widen_context(context, blocks)
            x = context['x']
        if not x:
            x = torch.empty((0, FEATURES+self.v2v_width), dtype=torch.float32)
        else:
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.shape[1] != FEATURES+self.v2v_width:
            raise ValueError('V2V intent feature width mismatch')
        return x, torch.as_tensor(context['rsu'], dtype=torch.float32)

    def heads(self, context):
        """(mean, value, trunk hidden); same ops as v2v_model.forward_with_v2v."""
        x, rsu = self._inputs(context)
        latent = self.broadcast(rsu).expand(len(x), -1)
        hidden = self.actor[:-1](torch.cat((x, latent), -1))
        mean = self.actor[-1](hidden)
        pooled = x.mean(0) if len(x) else torch.zeros(x.shape[1], dtype=x.dtype)
        value = self.critic(torch.cat((pooled, rsu))).squeeze(-1)
        return mean, value, hidden

    def forward(self, context):
        mean, value, _ = self.heads(context)
        return mean, value

    def intent(self, context):
        """Raw intent head (n_members, H); unclipped (training uses this).

        With ``intent_detach_trunk`` the trunk features are detached: the intent loss
        then reaches only the intent head, never the shared action/value trunk."""
        hidden = self.heads(context)[2]
        return self.intent_head(hidden.detach() if self.intent_detach_trunk else hidden)

    @torch.no_grad()
    def publish(self, context):
        """{member vid: H planned commands clipped to the action bounds} for broadcast."""
        lo, hi = (float(b) for b in self.intent_bounds)
        values = self.intent(context).clamp(lo, hi)
        return {vid: [float(v) for v in row] for vid, row in zip(context['members'], values)}

    def coop(self, context):
        """Raw coop head (n_members, 1+K+1): [p_coop logit, K yield-slot logits + none].

        Same trunk-detach discipline as :meth:`intent`."""
        hidden = self.heads(context)[2]
        return self.coop_head(hidden.detach() if self.intent_detach_trunk else hidden)

    @torch.no_grad()
    def publish_coop(self, context):
        """{member vid: [p_coop, K+1 yield probabilities]} for broadcast (unresolved slots)."""
        out = self.coop(context)
        p = torch.sigmoid(out[:, 0])
        probs = torch.softmax(out[:, 1:], -1)
        return {vid: [float(p[i]), [float(v) for v in probs[i]]]
                for i, vid in enumerate(context['members'])}


def intent_model(policy, depth=4):
    """The IntentPolicy behind runtime wrappers (Proxy, _OneShot, MixedPolicy), or None."""
    for _ in range(depth):
        if policy is None:
            return None
        if isinstance(policy, IntentPolicy):
            return policy
        nxt = None
        for name in ('real', 'policy', 'base_policy', 'new', 'model'):
            value = getattr(policy, name, None) if not isinstance(policy, nn.Module) else None
            if value is not None and value is not policy:
                nxt = value
                break
        policy = nxt
    return None


def convert(policy, horizon, loss_weight=DEFAULT_LOSS_WEIGHT, bounds=ACCEL_BOUNDS,
            detach_trunk=DEFAULT_DETACH_TRUNK, coop=False, coop_loss_weight=DEFAULT_COOP_LOSS_WEIGHT):
    """Turn an already v2v-widened base policy into an IntentPolicy in place."""
    expected = v2v_coop_dim(V2V_K, horizon) if coop else v2v_intent_dim(V2V_K, horizon)
    width = int(policy.actor[0].weight.shape[1])-FEATURES-4
    if width != expected:
        raise ValueError(f'widened width {width} != block width {expected}')
    policy.__class__ = IntentPolicy
    policy._attach_intent(horizon, loss_weight, bounds, detach_trunk, coop, coop_loss_weight)
    return policy


def with_head(policy, optimizer):
    """Fresh Adam over all parameters; existing moments/steps kept, the new head starts empty.

    The intent head registers after every existing parameter, so existing indices are unchanged.
    """
    state = optimizer.state_dict()
    n = len(list(policy.parameters()))
    groups = [dict(state['param_groups'][0], params=list(range(n)))]
    rebuilt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rebuilt.load_state_dict(dict(state=state['state'], param_groups=groups))
    return rebuilt


def _regrid(tensor, horizon, coop=False):
    """v2v-only (6 per partner) input columns -> intent/coop layout (zero gaps)."""
    width = len(V2V_PARTNER_FEATURES)
    old, new = V2V_K*width, (v2v_coop_dim(V2V_K, horizon) if coop else v2v_intent_dim(V2V_K, horizon))
    stride = width+horizon+1+(len(V2V_COOP_SIGNAL_FEATURES) if coop else 0)
    out = torch.zeros(*tensor.shape[:-1], tensor.shape[-1]-old+new, dtype=tensor.dtype)
    out[..., :FEATURES] = tensor[..., :FEATURES]
    for slot in range(V2V_K):
        dst = FEATURES+slot*stride
        out[..., dst:dst+width] = tensor[..., FEATURES+slot*width:FEATURES+(slot+1)*width]
    out[..., FEATURES+new:] = tensor[..., FEATURES+old:]
    return out


def _add_coop(tensor, horizon):
    """intent layout (6+H+1 per partner) -> coop layout (6+H+1+2), ZERO coop columns."""
    width = len(V2V_PARTNER_FEATURES)
    old_stride = width+horizon+1
    new_stride = old_stride+len(V2V_COOP_SIGNAL_FEATURES)
    old, new = V2V_K*old_stride, v2v_coop_dim(V2V_K, horizon)
    out = torch.zeros(*tensor.shape[:-1], tensor.shape[-1]-old+new, dtype=tensor.dtype)
    out[..., :FEATURES] = tensor[..., :FEATURES]
    for slot in range(V2V_K):
        dst = FEATURES+slot*new_stride
        out[..., dst:dst+old_stride] = tensor[..., FEATURES+slot*old_stride:FEATURES+(slot+1)*old_stride]
    out[..., FEATURES+new:] = tensor[..., FEATURES+old:]
    return out


def _regrid_input_layers(policy, optimizer, fn):
    """Re-grid both input layers and their 2D Adam moments with ``fn`` (indices unchanged)."""
    index = {id(p): i for i, p in enumerate(policy.parameters())}
    state = optimizer.state_dict()
    for head in ('actor', 'critic'):
        layer = getattr(policy, head)[0]
        i = index[id(layer.weight)]
        layer.weight = nn.Parameter(fn(layer.weight.detach()))
        for key, value in state['state'].get(i, {}).items():
            if torch.is_tensor(value) and value.dim() == 2:
                state['state'][i][key] = fn(value)
    rebuilt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rebuilt.load_state_dict(state)
    return rebuilt


def _from_v2v_only(policy, optimizer, horizon, coop=False):
    """Re-grid a v2v-only (block 24) policy and its Adam moments to the intent/coop block width."""
    return _regrid_input_layers(policy, optimizer, lambda t: _regrid(t, horizon, coop))


def _intent_sd_into_coop(policy, saved_policy, horizon):
    """Merge an intent-only state dict into a coop IntentPolicy: input layers re-gridded with
    ZERO coop columns, the zero-initialised coop head kept. Returns the Adam-compatible check."""
    sd = dict(saved_policy)
    for head in ('actor', 'critic'):
        key = f'{head}.0.weight'
        sd[key] = _add_coop(sd[key], horizon)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if unexpected or set(missing) != {'coop_head.weight', 'coop_head.bias'}:
        raise ValueError('intent checkpoint does not match the coop policy layout')


def _intent_optimizer_into_coop(policy, saved_optimizer, horizon):
    """Adam over the coop policy with the saved intent moments, input-layer moments re-gridded.

    The coop head registers after every existing parameter, so the saved indices name the
    same parameters once ``coop_head.*`` is removed from the new ordering."""
    old_names = [n for n, _p in policy.named_parameters() if not n.startswith('coop_head')]
    name_by_old_index = {i: n for i, n in enumerate(old_names)}
    new_index = {n: i for i, (n, _p) in enumerate(policy.named_parameters())}
    regrid = {'actor.0.weight', 'critic.0.weight'}
    transferred = {}
    for i, entry in saved_optimizer['state'].items():
        name = name_by_old_index[int(i)]
        transferred[new_index[name]] = {
            key: (_add_coop(value, horizon) if name in regrid and torch.is_tensor(value)
                  and value.dim() == 2 else value) for key, value in entry.items()}
    rebuilt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    # the coop head registers after every existing parameter: same groups, extended index list
    groups = [dict(saved_optimizer['param_groups'][0], params=list(range(len(list(policy.parameters())))))]
    rebuilt.load_state_dict(dict(state=transferred, param_groups=groups))
    return rebuilt


def load_checkpoint(path, config):
    """Intent-ON loader for MAPPO u0 (base width), v2v-only (block 24), intent and coop checkpoints.

    u0, v2v-only and intent checkpoints are widened with ZERO new input columns and zero heads,
    so the executed actions equal the current (coop) policy; Adam moments are kept/re-gridded.
    A coop checkpoint requires ``v2v_coop_signal`` on.
    """
    import random
    s = settings(config)
    if s is None:
        raise ValueError('v2v_intent loader requires v2v_intent=True')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    has_coop = 'coop_head.weight' in saved['policy']
    if has_coop and not s['coop']:
        raise ValueError('coop checkpoint requires v2v_coop_signal=True')
    if 'intent_head.weight' in saved['policy']:
        if int(saved['policy']['intent_head.weight'].shape[0]) != s['horizon']:
            raise ValueError('intent checkpoint horizon differs from v2v_intent_horizon')
        policy = IntentPolicy(s['horizon'], s['loss_weight'], s['bounds'], s['detach_trunk'],
                              s['coop'], s['coop_loss_weight'])
        if has_coop:
            policy.load_state_dict(saved['policy'])
            optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
            optimizer.load_state_dict(saved['optimizer'])
        else:
            _intent_sd_into_coop(policy, saved['policy'], s['horizon'])
            optimizer = _intent_optimizer_into_coop(policy, saved['optimizer'], s['horizon'])
        generator = torch.Generator()
        generator.set_state(saved['generator'])
        torch.set_rng_state(saved['torch_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        return policy, optimizer, generator, saved
    from .v2v_model import BASE_INPUT, load_checkpoint as widened
    width = int(saved['policy']['actor.0.weight'].shape[1])-BASE_INPUT['actor']
    if width == V2V_K*len(V2V_PARTNER_FEATURES):
        policy, optimizer, generator, payload = widened(path, v2v_dim=width)
        optimizer = _from_v2v_only(policy, optimizer, s['horizon'], s['coop'])
    else:
        policy, optimizer, generator, payload = widened(path, v2v_dim=s['dim'])
    convert(policy, s['horizon'], s['loss_weight'], s['bounds'], s['detach_trunk'],
            s['coop'], s['coop_loss_weight'])
    return policy, with_head(policy, optimizer), generator, payload


def from_weights(weights, *, enabled=False):
    """Spawned-worker reconstruction from a state dict carrying ``intent_head.*``."""
    if not enabled:
        raise ValueError('widened policy without explicit V2V job/branch flag')
    horizon = int(weights['intent_head.weight'].shape[0])
    bounds = [float(b) for b in weights.get('intent_bounds', torch.tensor(ACCEL_BOUNDS))]
    model = IntentPolicy(horizon, bounds=bounds, coop='coop_head.weight' in weights)
    model.load_state_dict(weights)
    return model


# ---------------------------------------------------------------- targets

def step_commands(step, bounds=ACCEL_BOUNDS):
    """{vid: issued pre-safety absolute command of this step}, clipped to the action bounds.

    Live/branch tape steps carry ``commands`` (native/MAPPO base) and ``requests``
    (``requested_raw`` = base + residual, or the plan's absolute command); plan tapes
    without them fall back to ``sample['absolute_accels']``.  Non-finite values are dropped.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    raw = {}
    commands = step.get('commands')
    if commands is not None:
        requests = step.get('requests') or {}
        for vid, base in commands.items():
            raw[vid] = requests[vid]['requested_raw'] if vid in requests else base
    else:
        raw.update((step.get('sample') or {}).get('absolute_accels') or {})
    out = {}
    for vid, value in raw.items():
        if value is not None and math.isfinite(float(value)):
            out[vid] = min(hi, max(lo, float(value)))
    return out


def commands_by_time(steps, bounds=ACCEL_BOUNDS):
    return {int(s['time_s']): step_commands(s, bounds) for s in steps}


def own_targets(members, t, by_time, horizon, allowed=None, end_s=None):
    """(targets, mask) lists [n][H]: own command at t+k, k=1..H; masked when unavailable."""
    targets = [[0.]*horizon for _ in members]
    mask = [[0.]*horizon for _ in members]
    for k in range(1, horizon+1):
        if end_s is not None and t+k >= end_s:
            break
        later = by_time.get(int(t)+k)
        if later is None:
            continue
        for j, vid in enumerate(members):
            if (allowed is None or vid in allowed) and vid in later:
                targets[j][k-1] = later[vid]
                mask[j][k-1] = 1.
    return targets, mask


def episode_targets(steps, horizon, bounds=ACCEL_BOUNDS):
    """Per step: (targets, mask) aligned with ``sample['context']['members']``."""
    by_time = commands_by_time(steps, bounds)
    return [own_targets(s['sample']['context']['members'], int(s['time_s']), by_time, horizon)
            for s in steps]


def refresh_bank_intents(policy, pairs):
    """Recompute every bank pair's partner-intent (and coop-signal) inputs with the CURRENT
    head (no grad), in place.

    Deployment-consistent: a partner's intent visible at second t is what the partner itself
    publishes from its own t-1 context (``IntentPolicy.publish``/``publish_coop``, clipped to
    the action bounds / sigmoid+softmax).  Pairs carry ``intent_link`` = dict(entry, t,
    slots={ego: partner vids in slot order}) from bank_v2v.attach_pairs; the partner's t-1
    context is the same entry's kept pair at t-1, already refreshed (ascending t), so intents
    chain exactly like the live board.  A partner that is not a member of that t-1 context,
    or whose t-1 pair is absent (plan start, excluded second), gets zero intents and
    intent_valid=0 (per partner); coop features read zeros likewise (``p_coop`` 0, no yield).
    With ``coop_enabled`` the coop features use the t-1 partner ranking of the SAME entry to
    resolve the published yield slot into a target vid.  Recorded commands are NEVER used as
    inputs (they are only the ego's own intent TARGETS).  Shared (oversampled) pair objects
    are refreshed once.  Returns a JSON-safe report.
    """
    groups, seen = {}, set()
    for sample, _w in pairs:
        link = sample.get('intent_link')
        if link is None or id(sample) in seen:
            continue
        seen.add(id(sample))
        groups.setdefault(link['entry'], {}).setdefault(int(link['t']), []).append(sample)
    width, h = len(V2V_PARTNER_FEATURES), policy.intent_horizon
    coop = getattr(policy, 'coop_enabled', False)
    coop_w = len(V2V_COOP_SIGNAL_FEATURES) if coop else 0
    stride = width+h+1+coop_w
    report = dict(pairs_refreshed=0, partner_slots=0, partner_slots_valid=0,
                  partner_slots_no_prev_context=0,
                  rule='current intent head (no grad) on the partner own t-1 bank context; '
                       'missing t-1 context or non-member partner -> zeros, intent_valid 0')
    if coop:
        report.update(coop_slots=0, coop_slots_valid=0)
    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            for by_t in groups.values():
                for t in sorted(by_t):
                    prev = by_t.get(t-1)
                    published = policy.publish(prev[0]['context']) if prev else {}
                    coop_board = None
                    if coop:
                        prev_slots = (prev[0].get('intent_link') or {}).get('slots') or {} if prev else {}
                        coop_board = (resolve_coop(policy.publish_coop(prev[0]['context']), prev_slots)
                                      if prev else {})
                    for sample in by_t[t]:
                        blocks = sample['context'].get('v2v') or {}
                        for ego, slots in sample['intent_link']['slots'].items():
                            block = blocks.get(ego)
                            if block is None:
                                continue
                            for slot, vid in enumerate(slots[:policy.intent_k]):
                                start = slot*stride+width
                                values = published.get(vid)
                                block[start:start+h] = [float(v) for v in values] if values else [0.]*h
                                block[start+h] = 1. if values else 0.
                                report['partner_slots'] += 1
                                report['partner_slots_valid'] += bool(values)
                                report['partner_slots_no_prev_context'] += not prev
                                if coop:
                                    signal = coop_board.get(vid)
                                    block[start+h+1:start+h+1+coop_w] = (
                                        [float(signal[0]), 1. if signal[1] == ego else 0.] if signal
                                        else [0.]*coop_w)
                                    report['coop_slots'] += 1
                                    report['coop_slots_valid'] += bool(signal)
                        report['pairs_refreshed'] += 1
    finally:
        policy.train(was_training)
    report['partner_intent_coverage'] = (report['partner_slots_valid']/report['partner_slots']
                                         if report['partner_slots'] else None)
    return report


def attach_bank_targets(pairs, tape, entry, horizon, bounds=ACCEL_BOUNDS):
    """Plan members' own future plan commands as intent targets (pair i = second t0+i).

    Targets stop at the plan end (``t0 + duration_s``); non-plan members are masked.
    Mutates the (already deep-copied) pair samples and returns ``pairs``.
    """
    by_time = commands_by_time(tape['steps'], bounds)
    t0, end = int(entry['start_s']), int(entry['start_s'])+int(entry['duration_s'])
    for i, (sample, _w) in enumerate(pairs):
        members = list(sample['context'].get('members') or sample['behaviour']['proposal_members'])
        allowed = set(sample.get('behaviour', {}).get('proposal_members') or ())
        sample['intent_target'], sample['intent_mask'] = own_targets(
            members, t0+i, by_time, horizon, allowed=allowed, end_s=end)
    return pairs


def yield_target_vid(commands_by_t, entry, start_rows, vid, allowed):
    """The plan member whose gap ``vid``'s deceleration opens, or None.

    Sign rule on the member's own plan-window commands (at least one command < 0, no
    behavioural threshold); the target is the NEAREST other plan member in the SAME
    lane BEHIND ``vid`` at the plan-start trace frame (max s_m; vid order tie-break).
    """
    seq = [c for t in range(int(entry['start_s']), int(entry['start_s'])+int(entry['duration_s']))
           for c in [commands_by_t.get(int(t), {}).get(vid)] if c is not None]
    if not seq or min(seq) >= 0. or vid not in start_rows:
        return None
    row = start_rows[vid]
    behind = [(float(start_rows[y]['s_m']), str(y)) for y in allowed
              if y != vid and y in start_rows and start_rows[y].get('lane') == row.get('lane')
              and float(start_rows[y]['s_m']) < float(row['s_m'])]
    return max(behind)[1] if behind else None


def attach_coop_targets(pairs, tape, entry, trace_path, cooperative, bounds=ACCEL_BOUNDS):
    """Coop-signal supervision for one bank entry's pairs (``v2v_coop_signal`` ON only).

    Only entries whose plan is labelled cooperative within its seed (the confirmation
    record's cooperation label) get targets, and only plan members are supervised:
    ``sample['coop_targets'][vid] = [p_coop=1, yield_slot or None]`` with the slot index
    resolved through the pair's recorded partner ranking (``intent_link.slots``); a
    determinable yield target that is not among the ego's current partner slots reads
    as none.  Non-cooperative entries (and PPO rollouts) get NO coop targets.  Mutates
    the (already deep-copied) pair samples and returns ``pairs``.
    """
    if not cooperative or not pairs:
        return pairs
    by_time = commands_by_time(tape['steps'], bounds)
    t0 = int(entry['start_s'])
    from . import bank_v2v
    try:
        frame = bank_v2v.trace_frames(trace_path).get(t0) or {}
    except (FileNotFoundError, OSError):
        frame = {}
    start_rows = {v['vid']: v for v in frame.get('vehicles', [])}
    for sample, _w in pairs:
        allowed = list(sample.get('behaviour', {}).get('proposal_members') or ())
        slots = (sample.get('intent_link') or {}).get('slots') or {}
        targets = {}
        for vid in allowed:
            target_vid = yield_target_vid(by_time, entry, start_rows, vid, allowed)
            ranking = list(slots.get(vid) or ())
            slot = ranking.index(target_vid) if target_vid is not None and target_vid in ranking else None
            targets[vid] = [1., slot]
        if targets:
            sample['coop_targets'] = targets
    return pairs


# ---------------------------------------------------------------- losses and reports

def resolve_yield(probs, slots):
    """Yield vid for yield probabilities over K partner slots + none (deterministic argmax;
    lowest index on ties; the none index or any slot beyond ``slots`` resolves to None)."""
    slot = int(np.argmax(np.asarray(probs, dtype=np.float64).reshape(-1)))
    slots = list(slots or ())
    return slots[slot] if 0 <= slot < len(slots) else None


def resolve_coop(published, slots):
    """{vid: [p_coop, yield_vid or None]} from ``publish_coop`` output and per-ego rankings."""
    return {vid: [float(p), resolve_yield(probs, slots.get(vid))]
            for vid, (p, probs) in (published or {}).items()}


def coop_terms(policy, contexts, targets):
    """(sum of supervised BCE(p_coop)+CE(yield) terms, supervised member count), with grad.

    ``targets`` maps member vid -> [p_coop target, yield slot or None]; a None slot labels
    the none class.  Members absent from ``targets`` are masked (no supervision)."""
    total = sum(p.sum() for p in policy.coop_head.parameters())*0
    count = 0
    for context, target in zip(contexts, targets):
        members = context.get('members') or ()
        rows = [j for j, vid in enumerate(members) if target.get(vid) is not None]
        if not rows:
            continue
        out = policy.coop(context)[rows]
        p_t = torch.tensor([float(target[v][0]) for v in members if target.get(v) is not None])
        slots = [target[v][1] for v in members if target.get(v) is not None]
        y_t = torch.tensor([int(s) if s is not None and 0 <= int(s) < policy.intent_k
                            else policy.intent_k for s in slots])
        total = total+(torch.nn.functional.binary_cross_entropy_with_logits(out[:, 0], p_t)
                       + torch.nn.functional.cross_entropy(out[:, 1:], y_t))
        count += len(rows)
    return total, count


def masked_terms(policy, contexts, targets, masks):
    """(sum of masked squared errors, masked count) over aligned contexts (with grad)."""
    total = sum(p.sum() for p in policy.intent_head.parameters())*0
    count = 0.
    for context, target, mask in zip(contexts, targets, masks):
        if not context['members']:
            continue
        m = torch.as_tensor(mask, dtype=torch.float32).reshape(-1, policy.intent_horizon)
        if not bool(m.any()):
            continue
        y = torch.as_tensor(target, dtype=torch.float32).reshape(-1, policy.intent_horizon)
        total = total+((policy.intent(context)-y).square()*m).sum()
        count += float(m.sum())
    return total, count


def mean_loss(policy, contexts, targets, masks):
    total, count = masked_terms(policy, contexts, targets, masks)
    return total/max(count, 1.), count


def coop_bank_term(policy, batch):
    """Weighted coop loss over bank pairs carrying ``coop_targets`` (v2v_coop_signal ON only)."""
    rows = [s for s, _w in batch if s.get('coop_targets')]
    total, count = coop_terms(policy, [s['context'] for s in rows], [s['coop_targets'] for s in rows])
    return total/max(count, 1.)*policy.coop_loss_weight


def bank_term(policy, batch):
    """Weighted intent loss over bank pairs carrying ``intent_target`` (imitation-phase hook);
    with the coop signal on, the coop term over pairs carrying ``coop_targets`` is added."""
    rows = [s for s, _w in batch if 'intent_target' in s]
    loss, _ = mean_loss(policy, [s['context'] for s in rows], [s['intent_target'] for s in rows],
                        [s['intent_mask'] for s in rows])
    loss = loss*policy.intent_loss_weight
    if getattr(policy, 'coop_enabled', False):
        loss = loss+coop_bank_term(policy, batch)
    return loss


@torch.no_grad()
def coop_bank_stats(policy, pairs):
    """Supervised coop loss/count over bank pairs carrying ``coop_targets``."""
    rows = [s for s, _w in pairs if s.get('coop_targets')]
    total, count = coop_terms(policy, [s['context'] for s in rows], [s['coop_targets'] for s in rows])
    return dict(coop_loss=float(total)/count if count else None, coop_members_supervised=int(count),
                coop_pairs=len(rows))


@torch.no_grad()
def bank_stats(policy, pairs):
    rows = [s for s, _w in pairs if 'intent_target' in s]
    loss, count = mean_loss(policy, [s['context'] for s in rows], [s['intent_target'] for s in rows],
                            [s['intent_mask'] for s in rows])
    out = dict(intent_loss=float(loss) if count else None, intent_targets=int(count), pairs=len(rows))
    if getattr(policy, 'coop_enabled', False):
        out.update(coop_bank_stats(policy, pairs))
    return out


def weight_norms(policy):
    """Frobenius norms of the partner-intent input columns (H values + valid bit per slot);
    with the coop signal on, the coop input columns (2 per slot) additionally."""
    cols = [FEATURES+c for c in intent_columns(policy.intent_k, policy.intent_horizon)]
    out = dict(actor=float(policy.actor[0].weight[:, cols].detach().norm()),
               critic=float(policy.critic[0].weight[:, cols].detach().norm()))
    if getattr(policy, 'coop_enabled', False):
        ccols = [FEATURES+c for c in coop_columns(policy.intent_k, policy.intent_horizon)]
        out.update(coop_actor=float(policy.actor[0].weight[:, ccols].detach().norm()),
                   coop_critic=float(policy.critic[0].weight[:, ccols].detach().norm()))
    return out


def coverage(contexts, horizon, k=V2V_K, coop=False):
    """Share of present partner slots whose intent was valid, over member blocks."""
    width = len(V2V_PARTNER_FEATURES)
    stride = width+horizon+1+(len(V2V_COOP_SIGNAL_FEATURES) if coop else 0)
    present = valid = 0
    for context in contexts:
        for block in (context.get('v2v') or {}).values():
            for slot in range(k):
                if len(block) >= (slot+1)*stride and block[slot*stride+width-1] > 0:
                    present += 1
                    valid += block[slot*stride+width+horizon] > 0
    return dict(partner_intent_coverage=valid/present if present else None,
                partner_slots_present=int(present), partner_slots_intent_valid=int(valid))


def grad_share(policy):
    """||intent-head gradient|| / ||all gradients|| from the current ``.grad`` (scale-invariant,
    so identical before or after clip_grad_norm_).

    Exact share of the intent loss when ``intent_detach_trunk`` (only the intent loss reaches
    the head and the head feeds nothing else); a head-only lower bound otherwise."""
    head_ids = {id(p) for p in policy.intent_head.parameters()}
    head = total = 0.
    for p in policy.parameters():
        if p.grad is None:
            continue
        sq = float(p.grad.detach().double().square().sum())
        total += sq
        if id(p) in head_ids:
            head += sq
    return math.sqrt(head/total) if total > 0 and math.isfinite(total) else 0.


def share_summary(shares):
    return dict(mean=float(np.mean(shares)) if shares else None, max=float(max(shares)) if shares else None,
                steps=len(shares))


class PPOIntent:
    """Per-update PPO-phase intent loss over rollout records; None-target records are skipped."""

    def __init__(self, policy, episodes, extra_records=0):
        self.policy, h = policy, policy.intent_horizon
        self.contexts, self.targets, self.masks = [], [], []
        for ep in episodes:
            for step, (target, mask) in zip(ep['steps'], episode_targets(ep['steps'], h, self.bounds())):
                self.contexts.append(step['sample']['context'])
                self.targets.append(target)
                self.masks.append(mask)
        self.extra = int(extra_records)
        self.shares = []
        self.report = dict(horizon=h, loss_weight=policy.intent_loss_weight, target_rule=TARGET_RULE,
                           detach_trunk=policy.intent_detach_trunk,
                           **coverage(self.contexts, h, policy.intent_k,
                                      getattr(policy, 'coop_enabled', False)))
        before = self.stats()
        self.report.update(intent_loss_before=before['intent_loss'], intent_targets=before['intent_targets'],
                           partner_intent_weight_norm_before=weight_norms(policy))

    def bounds(self):
        return tuple(float(b) for b in self.policy.intent_bounds)

    def term(self, indices):
        """Weighted masked MSE for record indices (ints); bank/extra records carry no PPO target."""
        rows = [int(i) for i in indices if int(i) < len(self.contexts)]
        loss, _ = mean_loss(self.policy, [self.contexts[i] for i in rows], [self.targets[i] for i in rows],
                            [self.masks[i] for i in rows])
        return loss*self.policy.intent_loss_weight

    @torch.no_grad()
    def stats(self):
        loss, count = mean_loss(self.policy, self.contexts, self.targets, self.masks)
        return dict(intent_loss=float(loss) if count else None, intent_targets=int(count))

    def log_grad(self):
        self.shares.append(grad_share(self.policy))

    def close(self, metrics):
        self.report.update(intent_grad_share=share_summary(self.shares),
                           intent_loss_after=self.stats()['intent_loss'],
                           partner_intent_weight_norm_after=weight_norms(self.policy))
        if metrics:
            metrics[-1]['v2v_intent'] = self.report
        else:
            metrics.append(dict(v2v_intent=self.report))
        return metrics


def ppo_setup(policy, episodes, extra_records=0):
    """None for every non-intent policy (default path byte-identical)."""
    if not getattr(policy, 'intent_horizon', 0):
        return None
    return PPOIntent(policy, episodes, extra_records)
