"""T2D: CPU-only cooperation on top of an owned, frozen MAPPO copy.

``CooperationLayer(base)`` is OFF and delegates sampling byte-for-byte. To turn
it on, supply a registered ``LayerConfig`` and optionally the T2A trigger head
(or an nn.Module adapter): ``head(per_cav_features) -> [CAV, candidate] logits``.
Candidate zero is ``none``. The caller supplies observable, feasible T1A
``moves.offsets`` for each candidate's acting CAV as ``move_offsets[CAV, K]``;
partner selection/commitments remain with T2A and the single-writer controller.
This module never issues commands or changes the existing safety bounds.

The residual output AND the trigger amplitude start at zero. Even a stochastic
trigger therefore leaves the sampled MAPPO actions exactly unchanged initially.
Trigger draws use a separate, caller-owned CPU generator; MAPPO AR(1) noise and
its memory are untouched. The trigger amplitude is tanh-bounded in [-1, 1].

``update`` consumes a sequence of {sample, group_advantage} records from T2E.
It has no critic, per-CAV reward, or base-policy optimizer. Each proposed Adam
step is checked against the rollout distribution using exact joint KL
(categorical triggers plus the correlated MAPPO Gaussian, averaged over groups).
An excessive or nonfinite step restores BOTH layer weights and optimizer state.
All training settings must be explicitly supplied by the run registration.
"""
from dataclasses import dataclass
import copy
import math

import torch
from torch import nn

from tools.coopfree.policy import gaussian_log_prob
from .acceleration import PHI, RHO


@dataclass(frozen=True)
class LayerConfig:
    feature_dim: int
    hidden_dim: int
    learning_rate: float
    kl_budget: float
    clip_ratio: float
    max_grad_norm: float
    enabled: bool = False

    def __post_init__(self):
        for key in ('feature_dim', 'hidden_dim'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{key} must be a positive registered integer')
        for key in ('learning_rate', 'kl_budget', 'max_grad_norm'):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not math.isfinite(self.clip_ratio) or not 0 < self.clip_ratio < 1:
            raise ValueError('clip_ratio must be in (0, 1)')
        if not isinstance(self.enabled, bool):
            raise ValueError('enabled must be boolean')


def _cpu(module):
    if any(t.device.type != 'cpu' for t in (*module.parameters(), *module.buffers())):
        raise ValueError('cooperation layer is CPU only')


class CooperationLayer(nn.Module):
    def __init__(self, base, config=None, *, trigger_head=None):
        super().__init__()
        _cpu(base)
        if trigger_head is not None:
            _cpu(trigger_head)
        # Own the frozen copy: an optimizer retained by a caller cannot edit it.
        self.base = copy.deepcopy(base).eval().requires_grad_(False)
        for p in self.base.parameters():
            p.grad = None
        self.config = config
        self.enabled = bool(config and config.enabled)
        self.trigger_head = trigger_head
        if config is not None:
            self.residual = nn.Sequential(nn.Linear(config.feature_dim, config.hidden_dim),
                                          nn.Tanh(), nn.Linear(config.hidden_dim, 1))
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)
        else:
            self.residual = None
        self.trigger_gain = nn.Parameter(torch.zeros(()), requires_grad=trigger_head is not None)
        # Trigger adapters are evaluated deterministically, including during PPO.
        # Gradients still flow; dropout/BatchNorm running-state drift is avoided.
        if trigger_head is not None:
            trigger_head.eval()
        self.optimizer = (torch.optim.Adam(self.layer_parameters(), lr=config.learning_rate)
                          if self.enabled else None)

    def layer_parameters(self):
        """Only the small layer, never MAPPO actor, critic, intent or log_std."""
        return [p for name, p in self.named_parameters()
                if not name.startswith('base.') and p.requires_grad]

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        if self.trigger_head is not None:
            self.trigger_head.eval()
        return self

    def _check_frozen(self):
        _cpu(self)
        if self.base.training or any(p.requires_grad or p.grad is not None
                                     for p in self.base.parameters()):
            raise ValueError('MAPPO base must stay frozen and in eval mode')

    def _features(self, context):
        x = context['x']
        if x and len(x[0]) != self.config.feature_dim and context.get('v2v'):
            from .v2v_integrate import widen_context
            x = widen_context(context, context['v2v'])['x']
        features = torch.as_tensor(x, dtype=self.residual[0].weight.dtype).detach()
        if not len(context['members']):
            features = features.reshape(0, self.config.feature_dim)
        if features.shape != (len(context['members']), self.config.feature_dim):
            raise ValueError('cooperation feature width/member mismatch')
        if not bool(torch.isfinite(features).all()):
            raise ValueError('nonfinite cooperation features')
        return features

    def _distribution(self, features, move_offsets=None):
        residual = self.residual(features).squeeze(-1)
        logits = (self.trigger_head(features) if self.trigger_head is not None
                  else residual.new_zeros((len(features), 1)))
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
                or logits.shape[0] != len(features) or logits.shape[1] < 1
                or not bool(torch.isfinite(logits).all())):
            raise ValueError('trigger adapter must return finite [CAV, candidate] logits')
        if move_offsets is None:
            if self.trigger_head is not None:
                raise ValueError('trigger head requires observable move_offsets, none at index 0')
            offsets = logits.new_zeros(logits.shape)
        else:
            offsets = torch.as_tensor(move_offsets, dtype=residual.dtype).detach()
            if offsets.shape != logits.shape or not bool(torch.isfinite(offsets).all()):
                raise ValueError('move_offsets must be finite and match trigger logits')
            if bool((offsets[:, 0] != 0).any()):
                raise ValueError('trigger candidate 0 must be none (zero offset)')
        return residual, logits, self.trigger_gain.tanh(), offsets

    @staticmethod
    def _correction(residual, gain, offsets, choices):
        return (residual + gain*offsets.gather(1, choices[:, None]).squeeze(1))[:, None]

    def forward(self, context):
        """Base mean/value plus residual; discrete moves are sampled via sample()."""
        self._check_frozen()
        with torch.no_grad():
            mean, value = self.base(context)
        if not self.enabled:
            return mean, value
        return mean + self.residual(self._features(context)), value

    @torch.no_grad()
    def sample(self, context, generator, deterministic=False, memory=None, *,
               layer_generator=None, move_offsets=None):
        self._check_frozen()
        if not self.enabled or not context['members']:
            return self.base.sample(context, generator, deterministic, memory)
        if self.trigger_head is not None and not deterministic:
            if layer_generator is None or layer_generator is generator:
                raise ValueError('stochastic triggers require a separate layer_generator')
            if layer_generator.device.type != 'cpu':
                raise ValueError('layer_generator must be CPU')
        features = self._features(context)
        residual, logits, gain, offsets = self._distribution(features, move_offsets)
        choices = (logits.argmax(-1) if deterministic or self.trigger_head is None or not len(features)
                   else torch.multinomial(logits.softmax(-1), 1,
                                          generator=layer_generator).squeeze(-1))
        sample, next_memory = self.base.sample(context, generator, deterministic, memory)
        correction = self._correction(residual, gain, offsets, choices)
        base_action = torch.as_tensor(sample['action'], dtype=residual.dtype).reshape(-1, 1)
        # Preserve signed zeros as well as all other action bytes at identity.
        action = torch.where(correction == 0, base_action, base_action + correction)
        sample['action'] = action.tolist()
        sample['own_action'] = action.tolist()
        sample['coop_layer'] = dict(features=features.tolist(), offsets=offsets.tolist(),
            choices=choices.tolist(), old_residual=residual.tolist(), old_logits=logits.tolist(),
            old_gain=float(gain), base_logp=sample['logp'])
        # This is the joint trigger/acceleration score, not base MAPPO's score.
        sample['logp'] = float(self.score(sample)[0])
        return sample, next_memory

    def score(self, sample):
        self._check_frozen()
        if not self.enabled or not sample['context']['members']:
            with torch.no_grad():
                return self.base.score(sample)
        record = sample['coop_layer']
        features = torch.as_tensor(record['features'], dtype=self.residual[0].weight.dtype)
        features = features.reshape(-1, self.config.feature_dim)
        residual, logits, gain, offsets = self._distribution(features, record['offsets'])
        choices = torch.as_tensor(record['choices'], dtype=torch.long)
        correction = self._correction(residual, gain, offsets, choices)
        with torch.no_grad():
            mean, value = self.base(sample['context'])
        if not len(mean):
            return residual.sum()*0, value
        conditional = mean + correction + PHI*torch.as_tensor(sample['previous_noise']).reshape(-1, 1)
        sigma = self.base.log_std.detach().exp()*math.sqrt(1-PHI**2)
        action = torch.as_tensor(sample['own_action'], dtype=mean.dtype).reshape(-1, 1)
        logp = gaussian_log_prob(action, conditional, RHO, sigma)
        logp = logp + logits.log_softmax(-1).gather(1, choices[:, None]).sum()
        return logp, value

    def joint_kl(self, sample):
        """Exact KL(old rollout || current), integrating ALL trigger choices.

        CAV trigger draws are independent conditional on the recorded features.
        Gaussian innovations retain the base policy's shared covariance RHO.
        """
        record = sample['coop_layer']
        features = torch.as_tensor(record['features'], dtype=self.residual[0].weight.dtype)
        features = features.reshape(-1, self.config.feature_dim)
        residual, logits, gain, offsets = self._distribution(features, record['offsets'])
        old_residual = residual.new_tensor(record['old_residual'])
        old_logits = logits.new_tensor(record['old_logits']).reshape_as(logits)
        old_logp = old_logits.log_softmax(-1)
        old_prob = old_logp.exp()
        categorical = (old_prob*(old_logp-logits.log_softmax(-1))).sum()
        delta = residual[:, None]-old_residual[:, None]+(gain-record['old_gain'])*offsets
        expected = (old_prob*delta).sum(-1)
        variance = (old_prob*(delta-expected[:, None]).square()).sum(-1)
        n = len(features)
        a, b = 1-RHO, 1-RHO+n*RHO
        quadratic = (expected.square()+variance).sum()/a
        quadratic = quadratic-RHO*(expected.sum().square()+variance.sum())/(a*b)
        sigma2 = self.base.log_std.detach().exp().square().squeeze()*(1-PHI**2)
        return categorical + .5*quadratic/sigma2

    def _check_optimizer(self):
        if self.optimizer is None:
            raise ValueError('cooperation layer is off; no layer optimizer')
        actual = [p for group in self.optimizer.param_groups for p in group['params']]
        expected = self.layer_parameters()
        if len(actual) != len(expected) or set(map(id, actual)) != set(map(id, expected)):
            raise ValueError('optimizer must own exactly the layer parameters')

    def update(self, records, *, epochs):
        """Layer-only PPO with externally supplied counterfactual group advantage.

        No normalization or reward substitution is performed. Deterministic and
        executed-plan samples cannot be used as stochastic layer PPO evidence.
        A group sample is the atomic record; the KL budget is per group.
        """
        self._check_frozen()
        self._check_optimizer()
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError('epochs must be a positive registered integer')
        records = list(records)
        if not records:
            raise ValueError('no cooperation training records')
        for record in records:
            sample = record['sample']
            if (sample.get('deterministic') or not sample['context']['members']
                    or sample.get('behaviour', {}).get('executed') or 'coop_layer' not in sample):
                raise ValueError('layer PPO requires stochastic own-action group samples')
            if not math.isfinite(record['group_advantage']) or not math.isfinite(sample['logp']):
                raise ValueError('nonfinite group advantage or rollout score')
        parameters = self.layer_parameters()

        def kl():
            return torch.stack([self.joint_kl(r['sample']) for r in records]).mean()

        with torch.no_grad():
            initial = float(kl())
        if not math.isfinite(initial) or initial > self.config.kl_budget:
            raise ValueError('rollout already exceeds registered layer KL budget')
        metrics = []
        for _ in range(epochs):
            old_weights = [p.detach().clone() for p in parameters]
            old_optimizer = copy.deepcopy(self.optimizer.state_dict())
            scores = torch.stack([self.score(r['sample'])[0] for r in records])
            old_scores = scores.new_tensor([r['sample']['logp'] for r in records])
            advantage = scores.new_tensor([r['group_advantage'] for r in records])
            ratios = (scores-old_scores).exp()
            clipped = ratios.clamp(1-self.config.clip_ratio, 1+self.config.clip_ratio)
            loss = -torch.minimum(ratios*advantage, clipped*advantage).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite cooperation PPO loss')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm,
                                                 error_if_nonfinite=True)
            try:
                self.optimizer.step()
                with torch.no_grad():
                    current_kl = float(kl())
                accepted = math.isfinite(current_kl) and current_kl <= self.config.kl_budget
            except Exception:
                with torch.no_grad():
                    for p, old in zip(parameters, old_weights):
                        p.copy_(old)
                self.optimizer.load_state_dict(old_optimizer)
                self.optimizer.zero_grad(set_to_none=True)
                raise
            if not accepted:
                with torch.no_grad():
                    for p, old in zip(parameters, old_weights):
                        p.copy_(old)
                self.optimizer.load_state_dict(old_optimizer)
            self.optimizer.zero_grad(set_to_none=True)
            metrics.append(dict(loss=float(loss.detach()), kl=current_kl,
                                accepted=accepted, gradient_norm=float(norm)))
            if not accepted:
                break
        self._check_frozen()
        return metrics
