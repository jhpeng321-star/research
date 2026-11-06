"""Opt-in per-CAV move trigger on the existing V2V MAPPO actor (CPU only).

Action 0 is always ``none``; actions 1..N select a recorded move candidate.
Each candidate is a joint family/observable-partners/parameter bucket from
``moves.candidates(obs_state, rng, n)``. No behavioural ranges are invented:
onset/hold/depth are normalised using variety.VARIETY_REGISTERED. Callers
provide *local observable* states and CAV identities, never a global frame.

The base policy's forward/sample/score paths are delegated unchanged. Trigger
sampling is separate (decision cadence belongs to the caller). PPO consumes
the recorded table, mask and per-CAV old log probabilities, not fresh draws.
Trigger features use only per-ego sensor/V2V inputs with a constant zero latent.
Intent-enabled triggers require a separate recorded ``trigger_v2v`` block whose
messages come from the local-only publisher, never the base intent board or RSU.
Frozen mode detaches the actor trunk
and excludes ALL base parameters from the trigger optimizer; explicit fine-tuning
includes the actor only. The low-level MAPPO action path stays separate.

MoveExecutor returns raw acceleration requests to the existing single-writer
controller, never TraCI commands. The downstream speedMode-27 safety path
remains authoritative. The moves dependency is imported lazily; tests inject
only a local stub until Track T1A lands.
"""
import copy
import math
from dataclasses import dataclass
from collections.abc import Mapping

import torch
from torch import nn
from torch.distributions import Categorical

from .v2v_integrate import widen_context
from .v2v_model import FEATURES, ACTOR_LATENT
from .variety import VARIETY_REGISTERED


def _moves(api):
    if api is None:
        from . import moves
        return moves
    return api


def _field(move, key):
    return move[key] if isinstance(move, Mapping) else getattr(move, key)


def _members(move):
    return tuple(_field(move, 'members'))


def _validate_move(move, ego, observable_cav_ids, families):
    family = _field(move, 'family')
    if family not in families:
        raise ValueError('unregistered move family')
    if family == 'none':
        return
    members = _members(move)
    if (not 1 <= len(members) <= 4 or len(set(members)) != len(members)
            or ego not in members or not set(members) <= set(observable_cav_ids)):
        raise ValueError('move members must be 1..4 observable CAVs including ego')
    for name, key in (('onset_s', 'variety_onset_window_s'),
                      ('hold_s', 'variety_hold_window_s'),
                      ('depth', 'variety_depth_window')):
        value = float(_field(move, name))
        lo, hi = VARIETY_REGISTERED[key]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'move {name} outside registered variety range')


@dataclass
class TriggerBatch:
    """Detached rollout snapshot; tables contain None at action 0 / padded slots.

    ``actions/logp/values`` are vectors in context['members'] order. Keep this
    batch alongside caller-computed per-CAV advantages and returns. Candidate
    descriptors and masks are fixed for the entire PPO update.
    """
    context: dict
    tables: tuple
    descriptors: torch.Tensor
    mask: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    values: torch.Tensor

    def selected(self):
        return {vid: copy.deepcopy(table[int(action)]) for vid, table, action in
                zip(self.context['members'], self.tables, self.actions)}


class TriggerPolicy(nn.Module):
    """Wrap a loaded V2V/intent actor without widening or altering its weights.

    ``n_candidates`` is an explicit registered action-budget choice. ``v2v_dim``
    MUST match the loaded actor (24 for V2V, 48 for registered intent). The
    caller's ordered observable CAV list defines partner slots in descriptors;
    its width must fit the supplied ``partner_slots`` (normally V2V K + ego).
    Candidate scoring shares weights across slots, so reordering the move table
    only reorders its categorical probabilities.

    ``extension`` (W1 v2v_wall / coop_awareness; DEFAULT None = byte-identical) adds
    protocol option families, per-candidate descriptor columns and per-ego context
    columns read from the recorded context. Every extra first-layer input column is
    zero-initialised on top of the unchanged default initialisation.
    """
    def __init__(self, base_policy, *, v2v_dim, n_candidates, partner_slots,
                 freeze_base=True, moves_api=None, extension=None):
        super().__init__()
        if (type(v2v_dim) is not int or v2v_dim < 1 or type(n_candidates) is not int
                or n_candidates < 1 or type(partner_slots) is not int or partner_slots < 1):
            raise ValueError('positive integer dimensions required')
        if not isinstance(freeze_base, bool):
            raise ValueError('freeze_base must be bool')
        if base_policy.actor[0].weight.shape[1] != FEATURES+v2v_dim+ACTOR_LATENT:
            raise ValueError('trigger v2v_dim does not match base actor')
        if any(p.device.type != 'cpu' for p in base_policy.parameters()):
            raise ValueError('trigger policy is CPU only')
        self.base_policy = base_policy
        self.v2v_dim, self.n_candidates = v2v_dim, n_candidates
        self.partner_slots, self.freeze_base = partner_slots, freeze_base
        self.moves_api = _moves(moves_api)
        self.families = tuple(self.moves_api.FAMILIES)
        if 'none' not in self.families or len(set(self.families)) != len(self.families):
            raise ValueError('moves.FAMILIES must be unique and include none')
        width = len(self.families)+partner_slots+3
        hidden = base_policy.actor[-1].in_features
        # Attachment does not consume the base policy's exploration RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.trigger_head = nn.Sequential(nn.Linear(hidden+width, hidden), nn.Tanh(),
                                              nn.Linear(hidden, 1))
            self.trigger_value = nn.Linear(hidden, 1)
        self.extension = extension
        self.option_families = self.families
        self.descriptor_width = width
        if extension is not None:
            if set(extension.families) & set(self.families):
                raise ValueError('extension families must be new option families')
            from .v2v_wall import widen_columns
            self.option_families = self.families+tuple(extension.families)
            self.descriptor_width = width+extension.descriptor_width
            self.trigger_head[0] = widen_columns(self.trigger_head[0], hidden+width,
                                                 extension.descriptor_width+extension.context_width)

    def forward(self, context):
        return self.base_policy(context)

    def sample(self, *args, **kwargs):
        return self.base_policy.sample(*args, **kwargs)

    def score(self, sample):
        return self.base_policy.score(sample)

    def intent_base_policy(self):
        """Explicit opt-in discovery adapter for the existing V2V controller."""
        return self.base_policy

    @torch.no_grad()
    def publish_trigger_intent(self, context):
        """Local-only messages; the runtime keeps a separate t-1 trigger board.

        Reuse the loaded intent head and per-CAV trunk, but with zero broadcast
        and *only* locally generated received intents. Never call base.publish:
        that publication and its low-level action path remain independent.
        """
        lo, hi = (float(b) for b in self.base_policy.intent_bounds)
        values = self.base_policy.intent_head(self._hidden(context)).clamp(lo, hi)
        return {vid: row.tolist() for vid, row in zip(context['members'], values)}

    def _hidden(self, context):
        x = context['x']
        if len(x) != len(context['members']) or len(set(context['members'])) != len(x):
            raise ValueError('context member/feature mismatch')
        # Even pre-widened rows can contain global-conditioned base intents.
        # Always replace that block with the separately recorded local payload;
        # missing provenance must fail closed, including during PPO/CE replay.
        if getattr(self.base_policy, 'intent_horizon', 0):
            blocks = context.get('trigger_v2v')
            if len(x) and (not blocks or any(v not in blocks for v in context['members'])):
                raise ValueError('trigger requires recorded local-only trigger_v2v partner observations')
            if any(len(row) not in (FEATURES, FEATURES+self.v2v_dim) for row in x):
                raise ValueError('trigger sensor feature width mismatch')
            if any(len(blocks[v]) != self.v2v_dim for v in context['members']):
                raise ValueError('trigger local-only V2V feature width mismatch')
            x = [list(row[:FEATURES])+list(blocks[vid])
                 for vid, row in zip(context['members'], x)]
        if len(x) and len(x[0]) == FEATURES:
            blocks = context.get('v2v')
            if not blocks:
                raise ValueError('trigger requires recorded V2V partner observations')
            x = widen_context(context, blocks)['x']
        x = torch.as_tensor(x, dtype=torch.float32).reshape(len(x), FEATURES+self.v2v_dim)
        if len(x) != len(context['members']) or len(set(context['members'])) != len(x):
            raise ValueError('context member/feature mismatch')
        # Reuse the loaded per-CAV trunk, but never its shared global broadcast.
        # Zero padding preserves the actor's input layout without a data channel
        # from context['rsu']; even a trained nonzero broadcast cannot leak in.
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_base):
            latent = x.new_zeros((len(x), ACTOR_LATENT))
            hidden = self.base_policy.actor[:-1](torch.cat((x, latent), -1))
        return hidden.detach() if self.freeze_base else hidden

    def _descriptor(self, move, visible):
        out = self._base_descriptor(move, visible)
        if self.extension is not None:
            out += self.extension.descriptor(move)
        return out

    def _base_descriptor(self, move, visible):
        family = 'none' if move is None else _field(move, 'family')
        out = [float(f == family) for f in self.families]
        members = () if move is None else _members(move)
        out += [float(v in members) for v in visible]+[0.]*(self.partner_slots-len(visible))
        if move is None:
            return out+[0., 0., 0.]
        for name, key in (('onset_s', 'variety_onset_window_s'),
                          ('hold_s', 'variety_hold_window_s'),
                          ('depth', 'variety_depth_window')):
            lo, hi = VARIETY_REGISTERED[key]
            out.append((float(_field(move, name))-lo)/(hi-lo) if hi > lo else 0.)
        return out

    def distribution(self, context, descriptors, mask):
        hidden = self._hidden(context)
        count = self.n_candidates+1
        width = self.descriptor_width
        if descriptors.shape != (len(hidden), count, width) or mask.shape != (len(hidden), count):
            raise ValueError('trigger table shape mismatch')
        if mask.dtype != torch.bool or (len(hidden) and not bool(mask[:, 0].all())):
            raise ValueError('none must always be available')
        inputs = torch.cat((hidden[:, None, :].expand(-1, count, -1), descriptors), -1)
        if self.extension is not None and self.extension.context_width:
            extra = self.extension.context_features(context)
            inputs = torch.cat((inputs, extra[:, None, :].expand(-1, count, -1)), -1)
        logits = self.trigger_head(inputs).squeeze(-1).masked_fill(~mask, -torch.inf)
        return logits, self.trigger_value(hidden).squeeze(-1)

    @torch.no_grad()
    def decide(self, context, *, obs_states, observable_cavs, rng, generator, deterministic=False):
        """Draw local candidates once, then sample one joint move bucket per CAV.

        ``obs_states[ego]`` is passed unchanged to moves.candidates. The separate
        ``observable_cavs[ego]`` must come from that ego's sensing/V2V reports
        (CAVs only, including ego), not context['members'] as a global fallback.
        Intent-enabled contexts must record ``trigger_v2v`` from the separate
        local-only runtime board; legacy base intent blocks are not a fallback.
        """
        context = copy.deepcopy(context)
        tables, descriptors, masks = [], [], []
        for ego in context['members']:
            visible = tuple(observable_cavs[ego])
            if (ego not in visible or len(set(visible)) != len(visible)
                    or len(visible) > self.partner_slots):
                raise ValueError('observable CAV slots must be unique, bounded and include ego')
            candidates = list(self.moves_api.candidates(obs_states[ego], rng, self.n_candidates))
            if len(candidates) > self.n_candidates:
                raise ValueError('moves.candidates exceeded requested action budget')
            table = [None]
            for move in candidates:
                _validate_move(move, ego, visible, self.option_families)
                if _field(move, 'family') != 'none':
                    table.append(copy.deepcopy(move))
            mask = [True]*len(table)+[False]*(self.n_candidates+1-len(table))
            table += [None]*(self.n_candidates+1-len(table))
            tables.append(tuple(table))
            descriptors.append([self._descriptor(move, visible) for move in table])
            masks.append(mask)
        shape = (len(tables), self.n_candidates+1)
        desc = torch.tensor(descriptors, dtype=torch.float32).reshape(*shape, self.descriptor_width)
        mask = torch.tensor(masks, dtype=torch.bool).reshape(shape)
        logits, values = self.distribution(context, desc, mask)
        if not len(tables):
            actions, logp = torch.empty(0, dtype=torch.long), torch.empty(0)
        else:
            actions = (logits.argmax(-1) if deterministic else
                       torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1))
            logp = Categorical(logits=logits).log_prob(actions)
        return TriggerBatch(context, tuple(tables), desc, mask, actions, logp, values)

    def evaluate(self, batch):
        logits, values = self.distribution(batch.context, batch.descriptors, batch.mask)
        if not len(values):
            return values, values, values
        if (batch.actions.shape != values.shape or batch.actions.dtype != torch.long
                or bool(((batch.actions < 0) | (batch.actions > self.n_candidates)).any())
                or not bool(batch.mask.gather(1, batch.actions[:, None]).all())):
            raise ValueError('invalid or masked trigger action')
        dist = Categorical(logits=logits)
        return dist.log_prob(batch.actions), values, dist.entropy()

    def learning_parameters(self):
        parameters = list(self.trigger_head.parameters())+list(self.trigger_value.parameters())
        if not self.freeze_base:
            parameters += list(self.base_policy.actor.parameters())
        return parameters

    def optimizer(self, *, lr):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError('learning rate must be positive and finite')
        return torch.optim.Adam(self.learning_parameters(), lr=lr)

    def imitation_loss(self, batch, targets):
        """Cross entropy for T2B/T2D; targets are indices into THIS recorded table."""
        logits, _ = self.distribution(batch.context, batch.descriptors, batch.mask)
        targets = torch.as_tensor(targets, dtype=torch.long)
        if (targets.shape != (len(logits),) or bool(((targets < 0) | (targets > self.n_candidates)).any())
                or not bool(batch.mask.gather(1, targets[:, None]).all())):
            raise ValueError('invalid or masked imitation target')
        return nn.functional.cross_entropy(logits, targets) if len(logits) else logits.sum()


def ppo_update(policy, optimizer, batch, advantages, returns, *, clip_ratio,
               value_weight, entropy_weight, max_grad_norm):
    """One per-CAV clipped PPO minibatch step; caller owns GAE/epochs/KL budget.

    Hyperparameters are explicit registration inputs. Advantages are not
    silently normalised (a singleton CAV minibatch remains useful). Detached
    old log probabilities/targets prevent backpropagation through the rollout.
    Returns scalar diagnostics, including the usual nonnegative KL estimator.
    """
    if not (math.isfinite(clip_ratio) and 0 < clip_ratio < 1
            and math.isfinite(value_weight) and value_weight >= 0
            and math.isfinite(entropy_weight) and entropy_weight >= 0
            and math.isfinite(max_grad_norm) and max_grad_norm > 0):
        raise ValueError('invalid PPO settings')
    parameters = policy.learning_parameters()
    actual = [p for group in optimizer.param_groups for p in group['params']]
    if len(actual) != len(parameters) or set(map(id, actual)) != set(map(id, parameters)):
        raise ValueError('optimizer must own exactly the trigger learning parameters')
    logp, values, entropy = policy.evaluate(batch)
    if not len(values):
        raise ValueError('empty trigger minibatch')
    advantages = torch.as_tensor(advantages, dtype=values.dtype).detach()
    returns = torch.as_tensor(returns, dtype=values.dtype).detach()
    old = batch.logp.detach()
    if any(t.shape != values.shape or not bool(torch.isfinite(t).all())
           for t in (advantages, returns, old, logp, values, entropy)):
        raise ValueError('PPO vectors must be finite and match CAV count')
    log_ratio = logp-old
    ratio = log_ratio.exp()
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError('nonfinite PPO ratio')
    objective = torch.minimum(ratio*advantages, ratio.clamp(1-clip_ratio, 1+clip_ratio)*advantages)
    actor_loss = -objective.mean()
    value_loss = .5*(returns-values).square().mean()
    loss = actor_loss+value_weight*value_loss-entropy_weight*entropy.mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()), policy_loss=float(actor_loss.detach()),
                value_loss=float(value_loss.detach()), entropy=float(entropy.mean().detach()),
                approx_kl=float(((ratio-1)-log_ratio).mean().detach()),
                clip_fraction=float(((ratio-1).abs() > clip_ratio).float().mean()),
                grad_norm=float(norm), cav_count=len(values))


class MoveExecutor:
    """Stateful, nonoverlapping moves; requests flow into existing ownership.

    Caller supplies the registered commitment expiry (``until``) explicitly;
    no duration/release behaviour beyond moves.offsets is assumed. Conflicting
    moves are rejected atomically; callers can schedule in stable ego order.
    Departed/non-CAV members are never requested. Feasibility bounds are
    per-tick observable/controller bounds, with the shared 1.2 m/s^2 CAV cap.
    Unaffected MAPPO commands are never clamped or rewritten by this layer.
    """
    def __init__(self, *, moves_api=None):
        self.moves_api = _moves(moves_api)
        self.active = []
        self.last_time = -math.inf

    def _advance(self, now):
        if not math.isfinite(now) or now < self.last_time:
            raise ValueError('move clock must be finite and monotonic')
        self.last_time = now
        self.active = [entry for entry in self.active if now < entry['until']]

    def start(self, move, *, ego, observable_cav_ids, now, until):
        self._advance(now)
        if move is None:
            return False
        _validate_move(move, ego, observable_cav_ids,
                       getattr(self.moves_api, 'EXECUTABLE_FAMILIES', self.moves_api.FAMILIES))
        if _field(move, 'family') == 'none':
            return False
        if not math.isfinite(until) or until <= now:
            raise ValueError('commitment expiry must be finite and after start')
        occupied = {vid for entry in self.active for vid in _members(entry['move'])}
        if occupied.intersection(_members(move)):
            return False
        self.active.append(dict(move=copy.deepcopy(move), start=now, until=until))
        return True

    def requests(self, base_actions, native_commands, *, now, eligible_cav_ids, feasible_bounds):
        """Compose offsets on absolute MAPPO actions and return acceleration.requests ABI.

        ``base_actions`` are native command + MAPPO residual, not residuals.
        Merge returned entries into the base request dict before the existing
        feasibility/CommandOwnership stage. This method never writes a vehicle.
        """
        self._advance(now)
        eligible = set(eligible_cav_ids)
        out = {}
        for entry in self.active:
            move = entry['move']
            offsets = self.moves_api.offsets(move, now-entry['start'])
            if not set(offsets) <= set(_members(move)):
                raise ValueError('moves.offsets returned an undeclared member')
            for vid, offset in offsets.items():
                offset = float(offset)
                if not math.isfinite(offset):
                    raise ValueError('nonfinite move offset')
                if vid not in eligible or vid not in base_actions or vid not in native_commands or offset == 0.:
                    continue
                lo, hi = map(float, feasible_bounds[vid])
                hi = min(hi, 1.2)
                base, native = float(base_actions[vid]), float(native_commands[vid])
                if not all(math.isfinite(v) for v in (lo, hi, base, native)) or lo > hi:
                    raise ValueError('invalid command or feasibility bounds')
                raw = min(hi, max(lo, base+offset))
                out[vid] = dict(requested_raw=raw, acceleration_residual=raw-native)
        return out
