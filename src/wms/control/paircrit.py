"""S-PAIRCRIT: directly regress signed CRN cost differences, never WM levels."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch
from torch import nn

from .breakdown import BreakdownHead, HORIZONS, event_labels, merge_speed, rule_sha256
from .longcritic import LongCritic, LongCriticConfig
from .real_returns14 import NPZRealReturnProvider
from .start_states import checked_path


@dataclass
class PairCriticConfig:
    enabled: bool = False
    policy_returns: str | None = None
    native_returns: str | None = None
    pairing_receipt: str | None = None
    policy_sha256: str | None = None
    reliability_file: str | None = None
    warmup_updates: int = 100

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.warmup_updates) is not int or self.warmup_updates < 0:
            raise ValueError('invalid paircrit enable/warmup settings')


@dataclass
class BreakdownConfig:
    enabled: bool = False
    checkpoint: str | None = None
    gate_checkpoint: str | None = None

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('breakdown.enabled must be boolean')


class PairedProvider:
    """Exact row alignment; receipt binds common saved-start IDs and both files."""
    def __init__(self, policy, native, receipt, expected_policy_sha256):
        self.policy, self.native = policy, native
        path = checked_path(receipt)
        proof = json.loads(path.read_text(encoding='utf-8'))
        for key, provider in [('policy_sha256', policy), ('native_sha256', native)]:
            if proof.get(key) != hashlib.sha256(provider.path.read_bytes()).hexdigest():
                raise ValueError('paired source hash mismatch')
        if proof.get('schema') != 'paircrit_pairs/1' or proof.get('split') != 'TRAIN':
            raise ValueError('paired receipt must declare TRAIN paircrit_pairs/1')
        if (not expected_policy_sha256 or proof.get('policy_checkpoint_sha256') != expected_policy_sha256
                or policy.provenance.get('policy_checkpoint') != expected_policy_sha256
                or policy.provenance.get('source_kind') != 'current_policy'
                or native.provenance.get('source_kind') != 'native'):
            raise ValueError('paired targets require a bound current-policy/native cohort')
        for key in ('seeds', 'requested_population', 'horizon_s'):
            if key not in policy.provenance or not np.array_equal(policy.provenance[key], native.provenance.get(key)):
                raise ValueError(f'paired metadata mismatch: {key}')
        if any(not torch.equal(value, native._states.get(key)) for key, value in policy._states.items()):
            raise ValueError('paired observable start states differ')
        if not (policy.provenance['horizon_s'] == 300).all():
            raise ValueError('real paired head currently requires matched 300s continuations')
        rows = proof.get('pairs', [])
        if len(rows) != policy._returns.numel() or len({r['pair_key'] for r in rows}) != len(rows):
            raise ValueError('missing/duplicate common saved-start identity')
        for i, row in enumerate(rows):
            if (row.get('seed') != int(policy.provenance['seeds'][i])
                    or row.get('horizon_s') != float(policy.provenance['horizon_s'][i])
                    or not row.get('observable_start_sha256') or not row.get('same_saved_start')):
                raise ValueError('missing common seed/horizon/start proof')
        self.proof = proof
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.fit_indices = torch.tensor([i for i, s in enumerate(policy.provenance['seeds'])
            if int(hashlib.sha256(f'965140:{int(s)}'.encode()).hexdigest(), 16) % 5 != 0])
        self.development_seeds = {int(s) for s in policy.provenance['seeds']
            if int(hashlib.sha256(f'965140:{int(s)}'.encode()).hexdigest(), 16) % 5 == 0}
        if not self.fit_indices.numel():
            raise ValueError('no paired fitting worlds')

    def sample(self, batch_size, device, update):
        indices = self.fit_indices[torch.from_numpy(np.random.default_rng(20261026 + update).integers(
            self.fit_indices.numel(), size=batch_size))]
        states = {k: v[indices].to(device) for k, v in self.policy._states.items()}
        # Input returns are -J; output is COST advantage J_policy - J_native.
        target = (self.native._returns[indices] - self.policy._returns[indices]).to(device)
        return states, target


class RiskGroupSignal(nn.Module):
    requires_state = True

    def __init__(self, base, head):
        super().__init__()
        self.base, self.risk = base, head
        self.projection = nn.Linear(6, 8, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, obs, state, rsu_field=None):
        if not isinstance(state, dict):
            raise ValueError('risk group feature requires observable state')
        from .rsu_field import build_rsu_field
        base = self.base(obs, state['s_m'], build_rsu_field(state))
        risk = self.risk(state).flatten(1)
        return base + self.projection(risk)[:, None, :] * obs.cav_mask[..., None]


class TailValue(nn.Module):
    """forward is reward bootstrap (-cost advantage); cost_advantage exposes A."""
    def __init__(self, base, *, paired=False, risk=None):
        super().__init__()
        self.base, self.paired, self.risk = base, paired, risk
        if paired:
            last = self.base.head[-1]
            self.base.head[-1] = nn.Linear(last.in_features, 3)
        if risk is not None:
            self.projection = nn.Linear(6, 3 if paired else 1, bias=False)
            nn.init.zeros_(self.projection.weight)
        if paired:
            # Do not initialize an advantage with an inherited absolute value.
            nn.init.zeros_(self.base.head[-1].weight)
            nn.init.zeros_(self.base.head[-1].bias)

    def cost_advantage(self, state, horizon_s=300):
        out = self.base(state)
        if self.risk is not None:
            term = self.projection(self.risk(state).flatten(1))
            term = term * getattr(self.base, 'value_scale', 1.)
            out = out + (term if self.paired else term.squeeze(-1))
        return out[:, HORIZONS.index(horizon_s)] if self.paired else out

    def forward(self, state):
        value = self.cost_advantage(state)
        return -value if self.paired else value


class PairCritic:
    def __init__(self, trainer, config, provider, wm_sha256s):
        if trainer.config.gamma != 1 or not trainer.counterfactual.enabled:
            raise ValueError('paired critic requires gamma=1 and native counterfactuals')
        self.config, self.provider = config, provider
        # Reuse the production no-grad CRN engine, with NO absolute native critic.
        self.engine = object.__new__(LongCritic)
        from .longcritic import reliability_weights
        self.engine.config = LongCriticConfig(enabled=True, horizons_s=[60, 120, 300])
        self.engine.warm = True
        self.engine.weights = {}
        if config.reliability_file:
            doc = json.loads(checked_path(config.reliability_file).read_text(encoding='utf-8'))
            self.engine.weights = reliability_weights(doc, wm_sha256s)
        self.receipt = {'schema': 'paircrit/1', 'target': 'J_policy-J_native',
            'bootstrap': '-A; native rows zero', 'absolute_baseline': False,
            'pairing_sha256': provider.sha256, 'wm_weights': self.engine.weights,
            'policy_checkpoint_sha256': config.policy_sha256,
            'off_policy_after_actor_update': True,
            'tail_assumption': 'residual policy-native cost after 300s is zero'}

    def loss(self, trainer, starts, scale):
        state, target = self.provider.sample(trainer.config.batch_size, trainer.device, trainer.update_count + 1)
        divisor = max(scale, trainer.config.real_return_scale)
        loss = ((trainer.critic.cost_advantage(state) - target) / divisor).square().mean()
        metrics = {'paircrit_real_mse': float(loss.detach()) * divisor ** 2}
        wm, extra = self.engine.targets(trainer, starts)
        if wm is not None:
            states, reward_delta, weight = wm
            # Longcritic target was shrunk; unshrink mean for unbiased weighted
            # direct difference regression (zero reliability still has zero mass).
            mean_cost = -reward_delta / weight.clamp_min(torch.finfo(weight.dtype).tiny)
            aux = (weight * ((trainer.critic.cost_advantage(states, extra['longcritic_horizon_s'])
                              - mean_cost) / divisor).square()).mean()
            loss = loss + aux
            metrics['paircrit_wm_loss'] = float(aux.detach())
        metrics.update(extra)
        return loss, metrics

    def warmup(self, trainer, updates):
        for i in range(updates):
            state, target = self.provider.sample(trainer.config.batch_size, trainer.device, i + 1)
            loss = ((trainer.critic.cost_advantage(state) - target) / trainer.config.real_return_scale).square().mean()
            trainer.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainer.critic.parameters(), trainer.config.grad_clip, error_if_nonfinite=True)
            trainer.critic_optimizer.step()
        trainer.target_critic.load_state_dict(trainer.critic.state_dict())
        return {'updates': updates, **self.receipt}

    def state_dict(self):
        return self.receipt

    def load_state_dict(self, doc):
        keys = ('pairing_sha256', 'policy_checkpoint_sha256', 'wm_weights')
        if any(doc.get(k) != self.receipt.get(k) for k in keys):
            if self.config.warmup_updates == 0:
                raise ValueError('paired provenance changed; positive boundary warmup required')
            self.receipt['rebound_from_pairing_sha256'] = doc.get('pairing_sha256')
            self.receipt['boundary_warmup_required'] = True


def load_risk(path, policy_sha256=None):
    doc = torch.load(checked_path(path), map_location='cpu', weights_only=True)
    if (doc.get('schema') != 'paircrit_risk/1' or doc.get('split') != 'TRAIN'
            or doc.get('rule_sha256') != rule_sha256(doc['rule'])
            or not doc.get('calibration_worlds') or not doc.get('fit_worlds')
            or set(doc['calibration_worlds']) & set(doc['fit_worlds'])
            or (policy_sha256 and doc.get('policy_checkpoint_sha256') != policy_sha256)):
        raise ValueError('risk checkpoint lacks native rule/disjoint calibration/provenance')
    head = BreakdownHead()
    head.load_state_dict(doc['head'])
    if not bool(torch.isfinite(head.temperature).all() and (head.temperature > 0).all()):
        raise ValueError('invalid risk calibration temperatures')
    if head.boundary_mps.item() != float(np.float32(doc['rule']['boundary_mps'])):
        raise ValueError('risk model/native-rule boundary mismatch')
    if doc.get('calibration_qualified') and not doc.get('qualification', {}).get('qualified'):
        raise ValueError('risk qualification claim lacks independent-world evidence')
    return head.eval().requires_grad_(False), doc


def install(trainer, pair_config, risk_config, wm_sha256s):
    """Called only for enabled flags; legacy optimizer/model/RNG path untouched."""
    if not pair_config.enabled and not risk_config.enabled:
        return
    risk = None
    if risk_config.enabled:
        risk, trainer.breakdown_receipt = load_risk(risk_config.checkpoint, pair_config.policy_sha256)
        risk = risk.to(trainer.device)
        trainer.group_encoder = RiskGroupSignal(trainer.group_encoder.encoder, risk).to(trainer.device)
        trainer.actor_parameters = list(trainer.policy.parameters()) + [
            p for p in trainer.group_encoder.parameters() if p.requires_grad]
        trainer.actor_optimizer = torch.optim.Adam(trainer.actor_parameters, lr=trainer.config.actor_lr)
    trainer.critic = TailValue(trainer.critic, paired=pair_config.enabled, risk=risk).to(trainer.device)
    trainer.target_critic = copy.deepcopy(trainer.critic).eval().requires_grad_(False)
    trainer.critic_optimizer = torch.optim.Adam(
        [p for p in trainer.critic.parameters() if p.requires_grad], lr=trainer.config.critic_lr)
    if pair_config.enabled:
        policy = NPZRealReturnProvider(pair_config.policy_returns)
        native = NPZRealReturnProvider(pair_config.native_returns)
        provider = PairedProvider(policy, native, pair_config.pairing_receipt, pair_config.policy_sha256)
        trainer.paircrit = PairCritic(trainer, pair_config, provider, wm_sha256s)


def wm_event_frequencies(imagined, rule):
    """Production CRN K rollouts -> [start,arm,T] frequencies + valid sample counts.

    Input has paired ALL/B rows from the same LongCritic engine, futures and
    driver/arrival/lane/action noise. No new or independent draws are generated.
    """
    if len(imagined['states']) < 2:
        raise ValueError('WM risk requires K>=2 paired futures')
    labels, masks = [], []
    for states in imagined['states']:
        v = torch.stack([merge_speed(s) for s in states]).detach().cpu().numpy()
        times = np.linspace(0, imagined['imagination_horizon_s'], len(states))
        y, m, _ = event_labels(v, times, rule)
        labels.append(y)
        masks.append(m)
    y, m = np.asarray(labels), np.asarray(masks)
    count = m.sum(0)
    freq = (y * m).sum(0) / count.clip(1)
    # Engine ordering: policy row then native row. Public contract: native,policy.
    return freq[[1, 0]][None], count[[1, 0]][None]


def breakdown_gate(head, states):
    """Additional deployment veto: no predicted risk increase at ANY horizon/start.

    Zero non-inferiority margin, inherited native-relative gate still required.
    This is predictive evidence only; a calibrated fit is not efficacy proof.
    """
    with torch.no_grad():
        p = head(states)
    delta = p[:, 1] - p[:, 0]
    return {'horizons_s': list(HORIZONS), 'policy_minus_native': delta.cpu().tolist(),
            'passed': bool(torch.isfinite(delta).all() and (delta <= 0).all()),
            'margin': 0., 'predictive_only': True}
