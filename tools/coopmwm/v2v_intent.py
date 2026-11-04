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

Config keys (binding): ``v2v_intent`` (bool, default False), ``v2v_intent_horizon``
(int, default 5), ``v2v_intent_loss_weight`` (float, default 1.0 when on),
``v2v_intent_detach_trunk`` (bool, default True),
``v2v_dim`` must equal ``v2v_intent_dim(K, H)`` (48 for K=4, H=5).
"""
import math

import numpy as np
import torch
from torch import nn

from .acceleration import Policy as BasePolicy
from .v2v_integrate import widen_context
from .v2v_model import FEATURES, extend_for_v2v
from .v2v_obs import (LANE_SIGNED, V2V_INTENT_HORIZON, V2V_K, V2V_PARTNER_FEATURES,
                      intent_columns, lane_intent_columns, lane_intent_dim,
                      v2v_intent_dim)

DEFAULT_LOSS_WEIGHT = 1.
#: Intent loss trains the intent head only (trunk features detached) unless the binding says False.
DEFAULT_DETACH_TRUNK = True
#: Registered acceleration bounds (accel_deploy / binding ``accel_bounds``); the binding wins.
ACCEL_BOUNDS = (-3., 2.6)
TARGET_RULE = ('own issued pre-safety absolute command at t+1..t+H (native+residual or plan '
               'absolute), clipped to accel_bounds; masked past episode end/despawn/missing second')
LANE_TARGET_RULE = ('own realised lane choice at t+1..t+H as signed WM lane delta '
                    '(keep 0 / left -1 / right +1); masked past episode end/despawn/missing second')


def settings(cfg):
    """Resolved intent settings from a binding, or None when ``v2v_intent`` is off (default)."""
    if not cfg or not cfg.get('v2v_intent'):
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
    # ``lane_actions`` (DEFAULT OFF) widens the per-partner block by H lane intents + a valid bit.
    dim = (lane_intent_dim(V2V_K, horizon) if cfg.get('lane_actions')
           else v2v_intent_dim(V2V_K, horizon))
    if cfg.get('v2v_dim') is not None and int(cfg['v2v_dim']) != dim:
        raise ValueError(f'v2v_dim {cfg["v2v_dim"]} != intent block width {dim}')
    return dict(horizon=horizon, loss_weight=weight, bounds=bounds, k=V2V_K, dim=dim, detach_trunk=detach)


def config_keys(horizon=V2V_INTENT_HORIZON, loss_weight=DEFAULT_LOSS_WEIGHT, detach_trunk=None):
    """Exact binding keys for an intent-ON arm (identical for A, B and R)."""
    return dict(v2v=True, v2v_dim=v2v_intent_dim(V2V_K, horizon), v2v_intent=True,
                v2v_intent_horizon=int(horizon), v2v_intent_loss_weight=float(loss_weight),
                v2v_intent_detach_trunk=DEFAULT_DETACH_TRUNK if detach_trunk is None else bool(detach_trunk))


# ---------------------------------------------------------------- policy

class IntentPolicy(BasePolicy):
    """v2v-widened per-CAV policy (block width K*(6+H+1)) with a planned-command intent head."""

    def __init__(self, horizon=V2V_INTENT_HORIZON, loss_weight=DEFAULT_LOSS_WEIGHT, bounds=ACCEL_BOUNDS,
                 detach_trunk=DEFAULT_DETACH_TRUNK):
        super().__init__()
        extend_for_v2v(self, v2v_intent_dim(V2V_K, horizon))
        self._attach_intent(horizon, loss_weight, bounds, detach_trunk)

    def _attach_intent(self, horizon, loss_weight, bounds, detach_trunk=DEFAULT_DETACH_TRUNK):
        self.intent_horizon, self.intent_k = int(horizon), V2V_K
        self.intent_detach_trunk = bool(detach_trunk)
        self.v2v_width = v2v_intent_dim(V2V_K, horizon)
        self.intent_loss_weight = float(loss_weight)
        hidden = int(self.actor[-1].in_features)
        with torch.random.fork_rng(devices=[]):  # no global RNG side effect
            head = nn.Linear(hidden, self.intent_horizon)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.intent_head = head
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
            detach_trunk=DEFAULT_DETACH_TRUNK):
    """Turn an already v2v-widened (block width K*(6+H+1)) base policy into an IntentPolicy in place."""
    width = int(policy.actor[0].weight.shape[1])-FEATURES-4
    if width != v2v_intent_dim(V2V_K, horizon):
        raise ValueError(f'widened width {width} != intent block width {v2v_intent_dim(V2V_K, horizon)}')
    policy.__class__ = IntentPolicy
    policy._attach_intent(horizon, loss_weight, bounds, detach_trunk)
    return policy


def with_head(policy, optimizer):
    """Fresh Adam over all parameters; existing moments/steps kept, the new head starts empty.

    The intent head registers after every existing parameter, so existing indices are unchanged.
    """
    state = optimizer.state_dict()
    n = len(list(policy.parameters()))
    groups = [dict(state['param_groups'][0], params=list(range(n)))]
    from .v2v_model import install_fresh_columns
    rebuilt = install_fresh_columns(torch.optim.Adam(policy.parameters(), lr=3e-4))
    rebuilt.load_state_dict(dict(state=state['state'], param_groups=groups))
    return rebuilt


def _regrid(tensor, horizon):
    """v2v-only (6 per partner) input columns -> intent layout (6 + H + 1 per partner), zero gaps."""
    width = len(V2V_PARTNER_FEATURES)
    old, new = V2V_K*width, v2v_intent_dim(V2V_K, horizon)
    out = torch.zeros(*tensor.shape[:-1], tensor.shape[-1]-old+new, dtype=tensor.dtype)
    out[..., :FEATURES] = tensor[..., :FEATURES]
    for slot in range(V2V_K):
        dst = FEATURES+slot*(width+horizon+1)
        out[..., dst:dst+width] = tensor[..., FEATURES+slot*width:FEATURES+(slot+1)*width]
    out[..., FEATURES+new:] = tensor[..., FEATURES+old:]
    return out


def _from_v2v_only(policy, optimizer, horizon):
    """Re-grid a v2v-only (block 24) policy and its Adam moments to the intent block width."""
    index = {id(p): i for i, p in enumerate(policy.parameters())}
    state = optimizer.state_dict()
    for head in ('actor', 'critic'):
        layer = getattr(policy, head)[0]
        i = index[id(layer.weight)]
        layer.weight = nn.Parameter(_regrid(layer.weight.detach(), horizon))
        for key, value in state['state'].get(i, {}).items():
            if torch.is_tensor(value) and value.dim() == 2:
                state['state'][i][key] = _regrid(value, horizon)
            elif key == 'v2v_column_offset' and torch.is_tensor(value) and value.dim() == 1:
                state['state'][i][key] = _regrid(value, horizon)   # fresh-column bookkeeping (I1)
    from .v2v_model import install_fresh_columns, mark_fresh_columns
    rebuilt = install_fresh_columns(torch.optim.Adam(policy.parameters(), lr=3e-4))
    rebuilt.load_state_dict(state)
    # MAIN decision I1: the new partner-intent columns start from a fresh Adam state (step 0).
    cols = [FEATURES+c for c in intent_columns(V2V_K, horizon)]
    return mark_fresh_columns(rebuilt, policy, dict(actor=cols, critic=cols))


def load_checkpoint(path, config):
    """Intent-ON loader for MAPPO u0 (base width), v2v-only (block 24) and intent checkpoints.

    u0 and v2v-only checkpoints are widened with ZERO intent input columns and a zero head,
    so the executed actions equal the current v2v policy; Adam moments are kept/re-gridded.
    """
    import random
    s = settings(config)
    if s is None:
        raise ValueError('v2v_intent loader requires v2v_intent=True')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if 'intent_head.weight' in saved['policy']:
        if int(saved['policy']['intent_head.weight'].shape[0]) != s['horizon']:
            raise ValueError('intent checkpoint horizon differs from v2v_intent_horizon')
        policy = IntentPolicy(s['horizon'], s['loss_weight'], s['bounds'], s['detach_trunk'])
        policy.load_state_dict(saved['policy'])
        from .v2v_model import install_fresh_columns
        optimizer = install_fresh_columns(torch.optim.Adam(policy.parameters(), lr=3e-4))
        optimizer.load_state_dict(saved['optimizer'])
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
        optimizer = _from_v2v_only(policy, optimizer, s['horizon'])
    else:
        policy, optimizer, generator, payload = widened(path, v2v_dim=s['dim'])
    convert(policy, s['horizon'], s['loss_weight'], s['bounds'], s['detach_trunk'])
    return policy, with_head(policy, optimizer), generator, payload


def from_weights(weights, *, enabled=False):
    """Spawned-worker reconstruction from a state dict carrying ``intent_head.*``."""
    if not enabled:
        raise ValueError('widened policy without explicit V2V job/branch flag')
    horizon = int(weights['intent_head.weight'].shape[0])
    bounds = [float(b) for b in weights.get('intent_bounds', torch.tensor(ACCEL_BOUNDS))]
    model = IntentPolicy(horizon, bounds=bounds)
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


def lane_choices_by_time(steps):
    """{second: {vid: signed WM lane delta}} of the recorded lane choices (lane_actions ON)."""
    out = {}
    for s in steps:
        sample = s.get('sample') or {}
        choice = sample.get('lane_choice')
        members = (sample.get('context') or {}).get('members') or []
        if choice is not None and len(choice) == len(members):
            out[int(s['time_s'])] = {v: LANE_SIGNED[int(c)] for v, c in zip(members, choice)}
    return out


def lane_own_targets(members, t, by_time, horizon, end_s=None):
    """(targets, mask) of the realised signed lane choices at t+1..t+H, aligned with members."""
    targets = [[0.]*horizon for _ in members]
    mask = [[0.]*horizon for _ in members]
    for k in range(1, horizon+1):
        if end_s is not None and t+k >= end_s:
            break
        later = by_time.get(int(t)+k)
        if later is None:
            continue
        for j, vid in enumerate(members):
            if vid in later:
                targets[j][k-1] = later[vid]
                mask[j][k-1] = 1.
    return targets, mask


def lane_episode_targets(steps, horizon):
    """Per step: (lane targets, mask) aligned with ``sample['context']['members']``."""
    by_time = lane_choices_by_time(steps)
    return [lane_own_targets(s['sample']['context']['members'], int(s['time_s']), by_time, horizon)
            for s in steps]


def refresh_bank_intents(policy, pairs):
    """Recompute every bank pair's partner-intent inputs with the CURRENT head (no grad), in place.

    Deployment-consistent: a partner's intent visible at second t is what the partner itself
    publishes from its own t-1 context (``IntentPolicy.publish``, clipped to the action bounds).
    Pairs carry ``intent_link`` = dict(entry, t, slots={ego: partner vids in slot order}) from
    bank_v2v.attach_pairs; the partner's t-1 context is the same entry's kept pair at t-1, already
    refreshed (ascending t), so intents chain exactly like the live board.  A partner that is not a
    member of that t-1 context, or whose t-1 pair is absent (plan start, excluded second), gets
    zero intents and intent_valid=0 (per partner).  Recorded commands are NEVER used as inputs
    (they are only the ego's own intent TARGETS).  Shared (oversampled) pair objects are refreshed
    once.  Returns a JSON-safe report.
    """
    groups, seen = {}, set()
    for sample, _w in pairs:
        link = sample.get('intent_link')
        if link is None or id(sample) in seen:
            continue
        seen.add(id(sample))
        groups.setdefault(link['entry'], {}).setdefault(int(link['t']), []).append(sample)
    width, h = len(V2V_PARTNER_FEATURES), policy.intent_horizon
    lane = getattr(policy, 'lane_intent_head', None) is not None   # lane_actions ON layout
    stride = width+(2*h+2 if lane else h+1)
    report = dict(pairs_refreshed=0, partner_slots=0, partner_slots_valid=0,
                  partner_slots_no_prev_context=0,
                  rule='current intent head (no grad) on the partner own t-1 bank context; '
                       'missing t-1 context or non-member partner -> zeros, intent_valid 0')
    if lane:
        report['lane_rule'] = ('current lane intent head (no grad) on the partner own t-1 bank '
                               'context; missing t-1 context or non-member partner -> zeros, '
                               'lane_valid 0')
        report['partner_lane_slots'] = 0
        report['partner_lane_slots_valid'] = 0
    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            for by_t in groups.values():
                for t in sorted(by_t):
                    prev = by_t.get(t-1)
                    published = policy.publish(prev[0]['context']) if prev else {}
                    published_lane = policy.publish_lane(prev[0]['context']) if lane and prev else {}
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
                                if lane:
                                    lane_start = start+h+1
                                    lane_values = published_lane.get(vid)
                                    block[lane_start:lane_start+h] = ([float(v) for v in lane_values]
                                                                      if lane_values else [0.]*h)
                                    block[lane_start+h] = 1. if lane_values else 0.
                                    report['partner_lane_slots'] += 1
                                    report['partner_lane_slots_valid'] += bool(lane_values)
                        report['pairs_refreshed'] += 1
    finally:
        policy.train(was_training)
    report['partner_intent_coverage'] = (report['partner_slots_valid']/report['partner_slots']
                                         if report['partner_slots'] else None)
    if lane:
        report['partner_lane_intent_coverage'] = (report['partner_lane_slots_valid'] /
                                                  report['partner_lane_slots']
                                                  if report['partner_lane_slots'] else None)
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


# ---------------------------------------------------------------- losses and reports

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


def lane_masked_terms(policy, contexts, targets, masks):
    """(sum of masked squared errors, masked count) of the LANE intent head (lane_actions ON)."""
    total = sum(p.sum() for p in policy.lane_intent_head.parameters())*0
    count = 0.
    for context, target, mask in zip(contexts, targets, masks):
        if not context['members']:
            continue
        m = torch.as_tensor(mask, dtype=torch.float32).reshape(-1, policy.intent_horizon)
        if not bool(m.any()):
            continue
        y = torch.as_tensor(target, dtype=torch.float32).reshape(-1, policy.intent_horizon)
        total = total+((policy.lane_intent(context)-y).square()*m).sum()
        count += float(m.sum())
    return total, count


def lane_mean_loss(policy, contexts, targets, masks):
    total, count = lane_masked_terms(policy, contexts, targets, masks)
    return total/max(count, 1.), count


def bank_term(policy, batch):
    """Weighted intent loss over bank pairs carrying ``intent_target`` (imitation-phase hook)."""
    rows = [s for s, _w in batch if 'intent_target' in s]
    loss, _ = mean_loss(policy, [s['context'] for s in rows], [s['intent_target'] for s in rows],
                        [s['intent_mask'] for s in rows])
    return loss*policy.intent_loss_weight


@torch.no_grad()
def bank_stats(policy, pairs):
    rows = [s for s, _w in pairs if 'intent_target' in s]
    loss, count = mean_loss(policy, [s['context'] for s in rows], [s['intent_target'] for s in rows],
                            [s['intent_mask'] for s in rows])
    return dict(intent_loss=float(loss) if count else None, intent_targets=int(count), pairs=len(rows))


def weight_norms(policy):
    """Frobenius norms of the partner-intent input columns (H values + valid bit per slot).

    With ``lane_actions`` ON the lane-intent columns are reported beside them."""
    cols = [FEATURES+c for c in intent_columns(policy.intent_k, policy.intent_horizon)]
    out = dict(actor=float(policy.actor[0].weight[:, cols].detach().norm()),
               critic=float(policy.critic[0].weight[:, cols].detach().norm()))
    if getattr(policy, 'lane_intent_head', None) is not None:
        lane = [FEATURES+c for c in lane_intent_columns(policy.intent_k, policy.intent_horizon)]
        out['lane_actor'] = float(policy.actor[0].weight[:, lane].detach().norm())
        out['lane_critic'] = float(policy.critic[0].weight[:, lane].detach().norm())
    return out


def coverage(contexts, horizon, k=V2V_K, lane=False):
    """Share of present partner slots whose intent was valid, over member blocks.

    ``lane=True`` (``lane_actions`` ON layout) also reports the lane-intent coverage."""
    width = len(V2V_PARTNER_FEATURES)
    stride = width+(2 if lane else 1)*int(horizon)+(2 if lane else 1)
    present = valid = lane_valid = 0
    for context in contexts:
        for block in (context.get('v2v') or {}).values():
            for slot in range(k):
                if len(block) >= (slot+1)*stride and block[slot*stride+width-1] > 0:
                    present += 1
                    valid += block[slot*stride+width+horizon] > 0
                    if lane:
                        lane_valid += block[slot*stride+width+2*horizon+1] > 0
    out = dict(partner_intent_coverage=valid/present if present else None,
               partner_slots_present=int(present), partner_slots_intent_valid=int(valid))
    if lane:
        out['partner_lane_intent_coverage'] = lane_valid/present if present else None
        out['partner_slots_lane_intent_valid'] = int(lane_valid)
    return out


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
    """Per-update PPO-phase intent loss over rollout records; None-target records are skipped.

    With ``lane_actions`` ON (policy carries ``lane_intent_head``) the LANE intent head trains
    beside the command intent head: masked MSE against the realised signed lane choices at
    t+1..t+H (:data:`tools/coopmwm/v2v_obs.LANE_SIGNED`), weighted by the policy's
    ``lane_intent_loss_weight``."""

    def __init__(self, policy, episodes, extra_records=0):
        self.policy, h = policy, policy.intent_horizon
        self.lane = getattr(policy, 'lane_intent_head', None) is not None
        self.contexts, self.targets, self.masks = [], [], []
        self.lane_targets, self.lane_masks = [], []
        for ep in episodes:
            steps = ep['steps']
            lane_pairs = lane_episode_targets(steps, h) if self.lane else None
            for step, (target, mask) in zip(steps, episode_targets(steps, h, self.bounds())):
                self.contexts.append(step['sample']['context'])
                self.targets.append(target)
                self.masks.append(mask)
                if self.lane:
                    lane_target, lane_mask = lane_pairs[len(self.contexts)-1]
                    self.lane_targets.append(lane_target)
                    self.lane_masks.append(lane_mask)
        self.extra = int(extra_records)
        self.shares = []
        self.report = dict(horizon=h, loss_weight=policy.intent_loss_weight, target_rule=TARGET_RULE,
                           detach_trunk=policy.intent_detach_trunk,
                           **coverage(self.contexts, h, policy.intent_k, lane=self.lane))
        if self.lane:
            self.report['lane_target_rule'] = LANE_TARGET_RULE
            self.report['lane_detach_trunk'] = policy.lane_intent_detach_trunk
        before = self.stats()
        self.report.update(intent_loss_before=before['intent_loss'], intent_targets=before['intent_targets'],
                           partner_intent_weight_norm_before=weight_norms(policy))
        if self.lane:
            self.report['lane_intent_loss_before'] = before['lane_intent_loss']

    def bounds(self):
        return tuple(float(b) for b in self.policy.intent_bounds)

    def term(self, indices):
        """Weighted masked MSE for record indices (ints); bank/extra records carry no PPO target."""
        rows = [int(i) for i in indices if int(i) < len(self.contexts)]
        loss, _ = mean_loss(self.policy, [self.contexts[i] for i in rows], [self.targets[i] for i in rows],
                            [self.masks[i] for i in rows])
        if self.lane:
            lane, _ = lane_mean_loss(self.policy, [self.contexts[i] for i in rows],
                                     [self.lane_targets[i] for i in rows],
                                     [self.lane_masks[i] for i in rows])
            loss = loss+self.policy.lane_intent_loss_weight*lane
        return loss*self.policy.intent_loss_weight

    @torch.no_grad()
    def stats(self):
        loss, count = mean_loss(self.policy, self.contexts, self.targets, self.masks)
        out = dict(intent_loss=float(loss) if count else None, intent_targets=int(count))
        if self.lane:
            lane, lane_count = lane_mean_loss(self.policy, self.contexts, self.lane_targets,
                                              self.lane_masks)
            out['lane_intent_loss'] = float(lane) if lane_count else None
        return out

    def log_grad(self):
        self.shares.append(grad_share(self.policy))

    def close(self, metrics):
        after = self.stats()
        self.report.update(intent_grad_share=share_summary(self.shares),
                           intent_loss_after=after['intent_loss'],
                           partner_intent_weight_norm_after=weight_norms(self.policy))
        if self.lane:
            self.report['lane_intent_loss_after'] = after['lane_intent_loss']
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
