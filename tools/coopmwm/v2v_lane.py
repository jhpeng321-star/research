"""Lane-change actions for the V2V arms (option ``lane_actions``, DEFAULT OFF).

Owner 2026-11-04: "build the case where the mappo can set the lane change so it can
get more cooperations. just build it for now".  Identical for A/B/R-v2v:

- POLICY: :class:`LanePolicy` is the per-CAV acceleration policy plus a DISCRETE LANE
  HEAD ``Linear(64, 3)`` (keep / left / right) off the shared actor trunk, next to the
  acceleration head.  Neutral initialisation: zero weight and bias
  ``[0, -LANE_NEUTRAL_LOGIT, -LANE_NEUTRAL_LOGIT]`` with ``LANE_NEUTRAL_LOGIT = 104``
  (``exp(-104)`` underflows float32), so the loaded distribution is EXACTLY
  ``p(keep) = 1``: every sampled lane action is ``keep`` = defer to native SUMO lane
  changing, and the joint log-prob equals the acceleration log-prob at load.  The
  lane choice is sampled from a DEDICATED lane generator (never the rollout
  generator), so the acceleration noise stream is untouched and loaded checkpoints
  behave exactly as now step-for-step; the generator state persists through
  ``learner.save``/``load`` as the ``lane_generator_state`` buffer.
- EXECUTION: a non-keep choice becomes a WM target lane (current lane + signed delta,
  clipped to 0..2) issued through the REAL plant actuator
  (``tools/coopfree/backend.tick`` -> ``CavSumoEnv.step(accel, lanes=...)``), which
  uses SUMO's own lane-change safety model (``laneChangeMode`` = LC_MODE_POLICY 513,
  safety-gapped ``changeLane``, junction/edge-end/ramp-lane guards); it is NEVER
  bypassed.  Ticks where every member chose keep pass ``lanes=None`` and stay fully
  native.  Per step the controller records ``lane_outcome`` =
  :func:`lane_outcome` (requested / issued / blocked / executed + the env's own
  counter breakdown) on its last event.
- PPO: the joint log-prob of (accel, lane) per member is ``log p(accel) +
  log p(lane)`` (both :meth:`LanePolicy.sample` and :meth:`LanePolicy.score`), and
  ``accel_learner.update`` adds the lane-head entropy term
  ``-lane_actions_entropy_weight * mean H(lane)`` for every arm.
- INTENT SHARING: with ``v2v_intent`` also on, a zero-initialised LANE INTENT HEAD
  publishes the next ``H`` lane actions as SIGNED values in {-1, 0, +1} (WM lane
  delta; :data:`tools/coopmwm/v2v_obs.LANE_SIGNED`); partners observe them at
  ``t-1`` latency with a ``lane_valid`` bit (``v2v_obs.v2v_lane_intent_features``).
  The per-partner block becomes ``6 + 2H + 2`` (72 wide for K=4, H=5); the extra
  input columns are zero-initialised, so inputs still don't move a loaded policy.
  The lane intent head trains like the command intent head (masked MSE against the
  realised signed lane choices), and ``v2v_intent.refresh_bank_intents`` refreshes
  partner lane intents from the current head.

Config keys (binding, identical for A/B/R): ``lane_actions`` (bool, default False),
``lane_actions_entropy_weight`` (float, default 0.01), ``lane_intent_loss_weight``
(float, default 1.0), ``lane_intent_detach_trunk`` (bool, default True).  With
``lane_actions`` on, ``v2v_dim`` must be the lane block width (``v2v_obs.
lane_intent_dim`` when ``v2v_intent`` is on, else ``V2V_DIM``).

NOTE for a future training run: with the exact-neutral init the left/right
probabilities are exactly zero, so the lane head receives zero policy-gradient and
entropy gradient until a registered schedule moves the bias (e.g. re-initialising
``lane_head.bias[1:]`` at a registered update boundary); a plain entropy bonus alone
cannot lift an exactly-degenerate categorical.
"""
import math

import numpy as np
import torch
from torch import nn

from .acceleration import Policy as BasePolicy
from .v2v_intent import IntentPolicy
from .v2v_model import FEATURES, extend_for_v2v, install_fresh_columns
from .v2v_obs import LANE_SIGNED, V2V_DIM, V2V_INTENT_HORIZON, V2V_K, lane_intent_dim

#: Softmax gap that makes the neutral init exact: exp(-104) == 0 in float32.
LANE_NEUTRAL_LOGIT = 104.
#: Keep / left / right (index of the discrete lane choice).
LANE_KEEP, LANE_LEFT, LANE_RIGHT = 0, 1, 2
#: Registered seed of the dedicated lane generator (never the rollout generator).
LANE_SEED = 800014
DEFAULT_ENTROPY_WEIGHT = .01
DEFAULT_LANE_INTENT_LOSS_WEIGHT = 1.
DEFAULT_LANE_INTENT_DETACH_TRUNK = True
LANE_OUTCOME_KEYS = ('issued', 'noop', 'refused_junction', 'refused_not_allowed',
                     'refused_no_lane', 'deferred_edge_end')
TARGET_RULE = ('own realised lane choice at t+1..t+H as signed WM lane delta '
               '(keep 0 / left -1 / right +1); masked past episode end/despawn/missing second')


def settings(cfg):
    """Resolved lane settings from a binding, or None when ``lane_actions`` is off (default)."""
    if not cfg or not cfg.get('lane_actions'):
        return None
    entropy = float(cfg.get('lane_actions_entropy_weight', DEFAULT_ENTROPY_WEIGHT))
    weight = float(cfg.get('lane_intent_loss_weight', DEFAULT_LANE_INTENT_LOSS_WEIGHT))
    for name, value in (('lane_actions_entropy_weight', entropy),
                        ('lane_intent_loss_weight', weight)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and >= 0')
    detach = cfg.get('lane_intent_detach_trunk', DEFAULT_LANE_INTENT_DETACH_TRUNK)
    if not isinstance(detach, bool):
        raise ValueError('lane_intent_detach_trunk must be a bool')
    intent = None
    if cfg.get('v2v_intent'):
        from .v2v_intent import settings as intent_settings
        intent = intent_settings(cfg)          # lane-aware: validates the lane block width
    return dict(entropy_weight=entropy, lane_intent_loss_weight=weight,
                lane_intent_detach_trunk=detach, intent=intent,
                dim=int(cfg['v2v_dim']) if cfg.get('v2v_dim') is not None else None)


def config_keys(horizon=V2V_INTENT_HORIZON, entropy_weight=DEFAULT_ENTROPY_WEIGHT,
                lane_intent_loss_weight=DEFAULT_LANE_INTENT_LOSS_WEIGHT,
                lane_intent_detach_trunk=DEFAULT_LANE_INTENT_DETACH_TRUNK,
                v2v_intent=True):
    """Exact binding keys for a lane_actions-ON arm (identical for A, B and R)."""
    from .v2v_intent import config_keys as intent_keys
    out = dict(lane_actions=True, lane_actions_entropy_weight=float(entropy_weight),
               lane_intent_loss_weight=float(lane_intent_loss_weight),
               lane_intent_detach_trunk=bool(lane_intent_detach_trunk))
    if v2v_intent:
        out.update(intent_keys(horizon))
        out['v2v_dim'] = lane_intent_dim(V2V_K, horizon)
    return out


# ---------------------------------------------------------------- policy

class _LaneHead:
    """Shared lane-head attachment and sampling (mixin over the acceleration policies)."""

    def _attach_lane(self):
        hidden = int(self.actor[-1].in_features)
        with torch.random.fork_rng(devices=[]):   # no global RNG side effect
            head = nn.Linear(hidden, 3)
        nn.init.zeros_(head.weight)
        with torch.no_grad():
            head.bias.copy_(torch.tensor([0., -LANE_NEUTRAL_LOGIT, -LANE_NEUTRAL_LOGIT]))
        self.lane_head = head
        self.register_buffer('lane_generator_state',
                             torch.Generator().manual_seed(LANE_SEED).get_state())
        self._lane_generator = torch.Generator()

    def hidden(self, context):
        """Shared actor-trunk hidden activation (the input of the acceleration head)."""
        raise NotImplementedError

    def lane_logits(self, context):
        """Raw lane head (n_members, 3); unmasked (every member can keep)."""
        return self.lane_head(self.hidden(context))

    def _lane_roll(self, logits, deterministic):
        """(choice, log_prob, entropy) per member from the DEDICATED lane generator state."""
        self._lane_generator.set_state(self.lane_generator_state)
        log_p = torch.log_softmax(logits.float(), -1)
        if deterministic:
            choice = logits.argmax(-1)
        else:
            choice = torch.multinomial(log_p.exp(), 1, generator=self._lane_generator)
            choice = choice.reshape(-1)
        self.lane_generator_state.copy_(self._lane_generator.get_state())
        picked = log_p.gather(-1, choice.reshape(-1, 1)).squeeze(-1)
        p = log_p.exp()
        entropy = -torch.where(p > 0, p*log_p, torch.zeros_like(p)).sum(-1)
        return choice, picked, entropy

    @torch.no_grad()
    def sample(self, context, generator, deterministic=False, memory=None):
        sample, noise_memory = BasePolicy.sample(self, context, generator, deterministic, memory)
        if len(sample['action']):
            choice, picked, entropy = self._lane_roll(self.lane_logits(context), deterministic)
            sample['lane_choice'] = [int(c) for c in choice]
            sample['lane_logp'] = float(picked.sum())
            sample['lane_entropy'] = float(entropy.mean())
            sample['logp'] = sample['logp']+sample['lane_logp']
        else:
            sample['lane_choice'] = []
            sample['lane_logp'] = 0.
            sample['lane_entropy'] = 0.
        return sample, noise_memory

    def score(self, sample):
        """Joint log-prob of the stored (accel, lane) actions and the critic value."""
        logp, value = BasePolicy.score(self, sample)
        if not sample.get('lane_choice'):
            return logp, value
        logits = self.lane_logits(sample['context'])
        choice = torch.as_tensor(sample['lane_choice'], dtype=torch.long).reshape(-1)
        if int(choice.shape[0]) != int(logits.shape[0]):
            raise ValueError('lane choice count differs from the context members')
        log_p = torch.log_softmax(logits.float(), -1)
        picked = log_p.gather(-1, choice.reshape(-1, 1)).squeeze(-1)
        return logp+picked.sum(), value


class LanePolicy(_LaneHead, BasePolicy):
    """Acceleration policy + discrete lane head at the plain 33-wide input width."""

    def __init__(self):
        BasePolicy.__init__(self)
        self._attach_lane()

    def hidden(self, context):
        x = torch.as_tensor(context['x'], dtype=torch.float32).reshape(-1, FEATURES)
        rsu = torch.as_tensor(context['rsu'], dtype=torch.float32)
        latent = self.broadcast(rsu).expand(len(x), -1)
        return self.actor[:-1](torch.cat((x, latent), -1))


class LaneV2VPolicy(_LaneHead, BasePolicy):
    """Acceleration policy + lane head over the widened v2v partner inputs (no intent)."""

    def __init__(self):
        BasePolicy.__init__(self)
        extend_for_v2v(self, V2V_DIM)
        self._attach_lane()

    def hidden(self, context):
        x = context['x']
        if x and len(x[0]) == FEATURES:
            blocks = context.get('v2v')
            if not blocks:
                raise ValueError('V2V policy requires recorded partner observations')
            from .v2v_integrate import widen_context
            context = widen_context(context, blocks)
            x = context['x']
        if not x:
            x = torch.empty((0, FEATURES+V2V_DIM), dtype=torch.float32)
        else:
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.shape[1] != FEATURES+V2V_DIM:
            raise ValueError('V2V feature width mismatch')
        from .v2v_model import forward_with_v2v
        rsu = torch.as_tensor(context['rsu'], dtype=torch.float32)
        latent = self.broadcast(rsu).expand(len(x), -1)
        return self.actor[:-1](torch.cat((x, latent), -1))


class LaneIntentPolicy(_LaneHead, IntentPolicy):
    """v2v intent policy (command intents) + lane head + LANE INTENT head (width K*(6+2H+2))."""

    def __init__(self, horizon=V2V_INTENT_HORIZON, loss_weight=1., bounds=(-3., 2.6),
                 detach_trunk=True, lane_entropy_weight=DEFAULT_ENTROPY_WEIGHT,
                 lane_intent_loss_weight=DEFAULT_LANE_INTENT_LOSS_WEIGHT,
                 lane_intent_detach_trunk=DEFAULT_LANE_INTENT_DETACH_TRUNK):
        BasePolicy.__init__(self)                 # skip IntentPolicy.__init__ (48-wide extension)
        extend_for_v2v(self, lane_intent_dim(V2V_K, horizon))
        IntentPolicy._attach_intent(self, horizon, loss_weight, bounds, detach_trunk)
        self.v2v_width = lane_intent_dim(V2V_K, horizon)
        self.lane_entropy_weight = float(lane_entropy_weight)
        self.lane_intent_loss_weight = float(lane_intent_loss_weight)
        self.lane_intent_detach_trunk = bool(lane_intent_detach_trunk)
        self._attach_lane()
        hidden = int(self.actor[-1].in_features)
        with torch.random.fork_rng(devices=[]):
            head = nn.Linear(hidden, horizon)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.lane_intent_head = head

    def hidden(self, context):
        return self.heads(context)[2]

    def lane_intent(self, context):
        """Raw lane intent head (n_members, H); signed WM lane deltas, unclipped."""
        hidden = self.heads(context)[2]
        return self.lane_intent_head(hidden.detach() if self.lane_intent_detach_trunk else hidden)

    @torch.no_grad()
    def publish_lane(self, context):
        """{member vid: H signed lane deltas clipped to [-1, 1]} for broadcast."""
        values = self.lane_intent(context).clamp(-1., 1.)
        return {vid: [float(v) for v in row] for vid, row in zip(context['members'], values)}


def _lane_intent_kwargs(lane):
    """LaneIntentPolicy kwargs from resolved lane settings (:func:`settings` output)."""
    return dict(horizon=lane['intent']['horizon'], loss_weight=float(lane['intent']['loss_weight']),
                bounds=tuple(float(b) for b in lane['intent']['bounds']),
                detach_trunk=bool(lane['intent']['detach_trunk']),
                lane_entropy_weight=float(lane['entropy_weight']),
                lane_intent_loss_weight=float(lane['lane_intent_loss_weight']),
                lane_intent_detach_trunk=bool(lane['lane_intent_detach_trunk']))


def lane_intent_model(policy, depth=4):
    """The LaneIntentPolicy behind runtime wrappers (Proxy, ...), or None."""
    for _ in range(depth):
        if policy is None:
            return None
        if isinstance(policy, LaneIntentPolicy):
            return policy
        nxt = None
        for name in ('real', 'policy', 'base_policy', 'new', 'model'):
            value = getattr(policy, name, None) if not isinstance(policy, nn.Module) else None
            if value is not None and value is not policy:
                nxt = value
                break
        policy = nxt
    return None


def convert(policy, lane):
    """Attach the lane heads to an already-loaded/widened policy IN PLACE (neutral init stands).

    ``lane`` is :func:`settings` output; ``lane['intent']`` selects the LaneIntentPolicy
    layout (command + lane intents) over the plain LaneV2VPolicy."""
    if lane is None:
        raise ValueError('lane convert requires lane_actions=True')
    if lane['intent'] is not None:
        if not hasattr(policy, 'intent_horizon'):
            raise ValueError('lane intent convert requires the command intent head')
        policy.__class__ = LaneIntentPolicy
        policy.v2v_width = lane_intent_dim(policy.intent_k, policy.intent_horizon)
        policy.lane_entropy_weight = lane['entropy_weight']
        policy.lane_intent_loss_weight = lane['lane_intent_loss_weight']
        policy.lane_intent_detach_trunk = lane['lane_intent_detach_trunk']
        policy._attach_lane()
        hidden = int(policy.actor[-1].in_features)
        with torch.random.fork_rng(devices=[]):
            head = nn.Linear(hidden, policy.intent_horizon)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        policy.lane_intent_head = head
        return policy
    policy.__class__ = LaneV2VPolicy
    policy._attach_lane()
    return policy


def with_head(policy, optimizer):
    """Fresh Adam over all parameters; existing moments/steps kept, the new heads start empty."""
    state = optimizer.state_dict()
    n = len(list(policy.parameters()))
    groups = [dict(state['param_groups'][0], params=list(range(n)))]
    rebuilt = install_fresh_columns(torch.optim.Adam(policy.parameters(), lr=3e-4))
    rebuilt.load_state_dict(dict(state=state['state'], param_groups=groups))
    return rebuilt


def _rng_payload(saved):
    import random
    generator = torch.Generator()
    generator.set_state(saved['generator'])
    torch.set_rng_state(saved['torch_rng'])
    np.random.set_state(saved['numpy_rng'])
    random.setstate(saved['python_rng'])
    return generator


def load_checkpoint(path, config):
    """Lane-ON loader for MAPPO u0, v2v, intent and already-lane checkpoints."""
    lane = settings(config)
    if lane is None:
        raise ValueError('lane loader requires lane_actions=True')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if 'lane_head.weight' in saved['policy']:          # already-lane checkpoint: reconstruct
        if lane['intent'] is not None:
            if 'lane_intent_head.weight' not in saved['policy']:
                raise ValueError('lane intent checkpoint lacks the lane intent head')
            policy = LaneIntentPolicy(**_lane_intent_kwargs(lane))
        else:
            policy = LaneV2VPolicy()
        policy.load_state_dict(saved['policy'])
        optimizer = install_fresh_columns(torch.optim.Adam(policy.parameters(), lr=3e-4))
        optimizer.load_state_dict(saved['optimizer'])
        return policy, optimizer, _rng_payload(saved), saved
    if lane['intent'] is not None:
        # The lane-aware intent loader lands u0/v2v-only/intent checkpoints on the lane block
        # (command intent head attached, Adam kept/re-gridded/fresh per MAIN decision I1).
        from .v2v_intent import load_checkpoint as intent_load
        policy, optimizer, generator, payload = intent_load(path, config)
        if not isinstance(policy, LaneIntentPolicy):
            convert(policy, lane)
            optimizer = with_head(policy, optimizer)
        return policy, optimizer, generator, payload
    from .v2v_model import load_checkpoint as widened
    policy, optimizer, generator, payload = widened(path, v2v_dim=V2V_DIM)
    convert(policy, lane)
    return policy, with_head(policy, optimizer), generator, payload


def from_weights(weights, *, enabled=False):
    """Spawned-worker reconstruction from a state dict carrying ``lane_head.*``."""
    if not enabled:
        raise ValueError('widened policy without explicit V2V job/branch flag')
    if 'lane_intent_head.weight' in weights:
        horizon = int(weights['intent_head.weight'].shape[0])
        bounds = [float(b) for b in weights.get('intent_bounds', torch.tensor((-3., 2.6)))]
        model = LaneIntentPolicy(horizon, bounds=bounds)
        model.load_state_dict(weights)
        return model
    model = LaneV2VPolicy()
    model.load_state_dict(weights)
    return model


# ---------------------------------------------------------------- execution (the REAL plant actuator)

def lane_targets(sample, rows):
    """{vid: target WM lane} for non-keep choices; the SUMO machinery decides feasibility."""
    choice = sample.get('lane_choice')
    if choice is None:
        return {}
    if len(choice) != len(sample['context']['members']):
        raise ValueError('lane choice count differs from the context members')
    lanes = {}
    for r in rows:
        if r['vid'] not in lanes:
            lanes[r['vid']] = int(r['lane'])
    out = {}
    for vid, action in zip(sample['context']['members'], choice):
        delta = int(LANE_SIGNED[int(action)])
        if delta == 0:
            continue
        current = lanes.get(vid)
        if current is None:
            raise ValueError('lane choice for an unobserved member: '+str(vid))
        out[vid] = min(2, max(0, current+delta))
    return out


def plan_lane_targets(requests, rows):
    """{vid: absolute target WM lane} of committed-plan lane requests ({vid: 0/1/2 action}).

    Same current-lane + signed-delta clipping as :func:`lane_targets` (c4/lane-plans, pinned at
    the merge: a plan member's lane action supersedes the policy lane choice).  Requests for
    members absent from ``rows`` (despawned mid-plan) are skipped; the actuator cannot place
    them and the tick's SUMO lane-change safety model stays the sole execution judge."""
    lanes = {}
    for r in rows:
        if r['vid'] not in lanes:
            lanes[r['vid']] = int(r['lane'])
    out = {}
    for vid, action in (requests or {}).items():
        delta = int(LANE_SIGNED[int(action)])
        if delta == 0:
            continue
        current = lanes.get(vid)
        if current is not None:
            out[vid] = min(2, max(0, current+delta))
    return out


def lanes_tensor(targets, env):
    """``[1, K]`` WM target lanes for ``CavSumoEnv.step(accel, lanes=...)``; None when all keep.

    ``None`` keeps the tick fully native (SUMO's own discretionary lane changing for
    every CAV); a nonempty target set puts the tick's commanded CAVs under the env's
    policy lane control (LC_MODE_POLICY) for that tick - SUMO's own safety model."""
    if not targets:
        return None
    ids = env.slot_vehicle_ids()
    tensor = torch.full((1, len(ids)), -1., dtype=torch.float32)
    placed = False
    for slot, vid in enumerate(ids):
        if vid in targets:
            tensor[0, slot] = float(targets[vid])
            placed = True
    return tensor if placed else None


def lane_outcome(counts):
    """Per-step lane record from ``info['lane_commands']``: requested/issued/blocked/executed."""
    counts = dict(counts or {})
    requested = sum(int(counts.get(key, 0)) for key in LANE_OUTCOME_KEYS)
    issued = int(counts.get('issued', 0))
    return dict(requested=requested, issued=issued, blocked=requested-issued,
                executed=int(counts.get('completed', 0)), counts=counts)


# ---------------------------------------------------------------- PPO (joint loss pieces)

def lane_entropy(policy, contexts):
    """Mean lane-head entropy over aligned contexts (with grad)."""
    total = None
    count = 0
    for context in contexts:
        logits = policy.lane_logits(context)
        if not len(logits):
            continue
        log_p = torch.log_softmax(logits.float(), -1)
        p = log_p.exp()
        entropy = -torch.where(p > 0, p*log_p, torch.zeros_like(p)).sum(-1)
        total = entropy.sum() if total is None else total+entropy.sum()
        count += int(len(logits))
    if total is None:
        total = policy.lane_head.weight.sum()*0
    return total/max(count, 1)


class LanePPO:
    """Per-minibatch lane entropy term for the joint PPO loss (every arm; None-target safe)."""

    def __init__(self, policy, episodes):
        self.policy = policy
        self.contexts = [step['sample']['context'] for ep in episodes for step in ep['steps']]
        self.last_entropy = None

    def entropy_term(self, indices):
        """``-weight * mean entropy`` over the requested record indices (ints)."""
        rows = [int(i) for i in indices if int(i) < len(self.contexts)]
        if not rows:
            self.last_entropy = None
            return self.policy.lane_head.weight.sum()*0
        entropy = lane_entropy(self.policy, [self.contexts[i] for i in rows])
        self.last_entropy = float(entropy.detach())
        return -self.policy.lane_entropy_weight*entropy

    def stats(self):
        with torch.no_grad():
            entropy = float(lane_entropy(self.policy, self.contexts))
        neutral = bool(torch.equal(self.policy.lane_head.bias[1:],
                                   torch.full_like(self.policy.lane_head.bias[1:], -LANE_NEUTRAL_LOGIT)))
        return dict(lane_entropy=entropy, lane_neutral_init=neutral)


def ppo_setup(policy, episodes):
    """None for every non-lane policy (default path byte-identical)."""
    if getattr(policy, 'lane_head', None) is None:
        return None
    return LanePPO(policy, episodes)
