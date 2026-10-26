"""Registered TRAIN-world development selection of real/difference critic loss mass."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib

import numpy as np
import torch

from .dreamer import _rng_state, _restore_rng
from .longcritic import difference_loss
from .real_returns14 import RealReturnBatch

CANDIDATES = (0.3, 0.5, 0.7, 0.9, 1.0)
SEED = 965140


@dataclass
class RealShareConfig:
    enabled: bool = False
    updates: int = 20
    bootstrap_receipt: str | None = None

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or type(self.updates) is not int or self.updates != 20:
            raise ValueError('real_share requires boolean enabled and registered 20 updates')


def development_world(seed):
    return int(hashlib.sha256(f'{SEED}:{int(seed)}'.encode()).hexdigest(), 16) % 5 == 0


class SubsetProvider:
    def __init__(self, provider, indices):
        self.provider = provider
        self.indices = np.asarray(indices, np.int64)
        if not len(self.indices):
            raise ValueError('real-share partition has no examples')
        self.provenance = dict(provider.provenance)
        n = provider._returns.numel()
        for name, value in self.provenance.items():
            if isinstance(value, np.ndarray) and value.shape == (n,):
                self.provenance[name] = value[self.indices].copy()
        self.path = provider.path

    def __call__(self, *, batch_size, device, update):
        rng = np.random.default_rng(self.provider.seed + int(update))
        index = torch.as_tensor(self.indices[rng.integers(len(self.indices), size=batch_size)])
        return RealReturnBatch({k: v[index].to(device) for k, v in self.provider._states.items()},
                               self.provider._returns[index].to(device), 1.)


def split_provider(provider):
    n = provider._returns.numel()
    seeds = np.asarray(provider.provenance.get('seeds', provider.provenance.get('seed', [])))
    if seeds.ndim == 0:
        seeds = np.full(n, int(seeds))
    if seeds.shape != (n,):
        raise ValueError('real-share selection requires traffic-world seed provenance')
    dev = np.asarray([development_world(s) for s in seeds])
    return SubsetProvider(provider, np.flatnonzero(~dev)), SubsetProvider(provider, np.flatnonzero(dev))


def exclude_development_starts(sampler, seeds):
    """Metadata-only exclusion before any start loading, retaining original indices."""
    import re
    out = copy.copy(sampler)
    keep = set()
    for index, meta in sampler.metadata.items():
        world = str(meta.get('world', meta.get('world_id', '')))
        match = re.search(r'w(\d+)m', world)
        if match and int(match[1]) in seeds:
            continue
        if meta.get('seed') is not None and int(meta['seed']) in seeds:
            continue
        keep.add(index)
    out.groups = {p: [i for i in rows if i in keep] for p, rows in sampler.groups.items()}
    out.groups = {p: rows for p, rows in out.groups.items() if rows}
    out.penetrations = sorted(out.groups)
    if not out.penetrations:
        raise ValueError('no fitting starts remain after development-world exclusion')
    out.metadata = {i: m for i, m in sampler.metadata.items() if i in keep}
    if sampler.source_groups is not None:
        out.source_groups = {p: [(name, [i for i in rows if i in keep])
                                 for name, rows in sampler.source_groups[p]
                                 if any(i in keep for i in rows)] for p in out.penetrations}
    return out


def tensor_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def select_largest(errors, seeds):
    """Errors [candidate, example]; paired world bootstrap, fixed 95% upper bound."""
    seeds = np.asarray(seeds)
    worlds = np.unique(seeds)
    if len(worlds) < 2:
        return .3, {'reason': 'fewer than two independent development worlds'}
    grouped = np.stack([np.asarray(errors)[:, seeds == s].mean(1) for s in worlds], 1)
    rng = np.random.default_rng(SEED)
    pick = rng.integers(len(worlds), size=(2000, len(worlds)))
    rows, selected = {}, .3
    for i, share in enumerate(CANDIDATES):
        delta = grouped[i] - grouped[0]
        upper = float(np.quantile(delta[pick].mean(1), .95))
        finite = bool(np.isfinite(grouped[i]).all())
        eligible = finite and upper <= 0
        if eligible:
            selected = share
        rows[str(share)] = {'mse_raw': float(grouped[i].mean()), 'delta_mean': float(delta.mean()),
                            'delta_upper95': upper, 'eligible': eligible}
    return selected, {'worlds': worlds.tolist(), 'candidates': rows,
                      'rule': 'largest share with paired world-bootstrap upper95(MSE_share-MSE_0.3)<=0'}


def calibrate_share(trainer, fit, dev, *, policy_sha256, config=None):
    cfg = config or RealShareConfig(enabled=True)
    source = fit.provider
    receipt = {'selected_share': .3, 'source_sha256': hashlib.sha256(source.path.read_bytes()).hexdigest(),
               'initial_critic_sha256': tensor_digest(trainer.critic), 'policy_sha256': policy_sha256,
               'wm_ids': trainer.world_model_ids, 'updates_per_candidate': cfg.updates,
               'fit_worlds': sorted(set(fit.provenance['seeds'].tolist())),
                'development_worlds': sorted(set(dev.provenance['seeds'].tolist()))}
    if cfg.bootstrap_receipt and policy_sha256 is None:
        # Owner explicitly approved the measured1.0 share for the NEW scratch
        # residual actor. Reuse only its pinned real-target evidence, never its
        # rejected actor/critic tensors. This is an off-policy critic bootstrap,
        # not a new claim that these are the fresh actor's real outcomes.
        import json
        from .start_states import checked_path
        path = checked_path(cfg.bootstrap_receipt)
        evidence = json.loads(path.read_text())
        row = evidence.get('candidates', {}).get('1.0', {})
        if (evidence.get('selected_share') != 1.0 or not row.get('eligible')
                or not np.isfinite(row.get('delta_upper95', np.nan)) or row['delta_upper95'] > 0
                or evidence.get('source_sha256') != receipt['source_sha256']
                or evidence.get('policy_sha256') != source.provenance.get('policy_checkpoint')
                or evidence.get('fit_worlds') != receipt['fit_worlds']
                or evidence.get('development_worlds') != receipt['development_worlds']):
            raise ValueError('registered real-share bootstrap evidence is stale/mismatched')
        trainer.config = replace(trainer.config, real_return_mix=1.0)
        return {**receipt, 'selected_share': 1.0, 'offpolicy_bootstrap': True,
                'selection_evidence_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'selection_source_policy': evidence['policy_sha256'],
                'actor_adopted': False, 'critic_adopted': False}
    if (source.provenance['source_kind'] != 'current_policy'
            or source.provenance.get('policy_checkpoint') != policy_sha256):
        receipt['refused'] = 'real targets are not of the pinned current policy; keep 0.3'
        trainer.config = replace(trainer.config, real_return_mix=.3)
        return receipt
    if trainer.longcritic is None:
        raise ValueError('real-share selector requires difference-only longcritic assistance')
    rng = _rng_state(trainer.device)
    sampler_state = copy.deepcopy(trainer.start_states.state_dict())
    tilt_state = copy.deepcopy(trainer._tilt_sampler.state_dict()) if trainer._tilt_sampler else None
    count = trainer.update_count
    cache = []
    try:
        for step in range(cfg.updates):
            trainer.update_count = count + step
            with torch.no_grad():
                real = fit(batch_size=trainer.config.batch_size, device=trainer.device, update=step + 1)
                start = trainer._sample_starts(trainer.start_states)
                target, _ = trainer.longcritic.targets(trainer, start)
                cache.append((real, target))
    finally:
        trainer.update_count = count
        trainer.start_states.load_state_dict(sampler_state)
        if trainer._tilt_sampler:
            trainer._tilt_sampler.load_state_dict(tilt_state)
        _restore_rng(rng, trainer.device)
    index = torch.as_tensor(dev.indices)
    dev_states = {k: v[index].to(trainer.device) for k, v in source._states.items()}
    truth = source._returns[index].to(trainer.device)
    scale = trainer.config.real_return_scale
    candidates, errors = [], []
    for share in CANDIDATES:
        _restore_rng(rng, trainer.device)
        critic = copy.deepcopy(trainer.critic)
        critic.train()
        optimizer = torch.optim.Adam(critic.parameters(), lr=trainer.config.critic_lr)
        optimizer.load_state_dict(copy.deepcopy(trainer.critic_optimizer.state_dict()))
        for real, target in cache:
            loss = share * ((critic(real.states) - real.returns) / scale).square().mean()
            if target is not None:
                states, delta, weight = target
                loss = loss + (1 - share) * difference_loss(critic, trainer.longcritic.native_critic,
                                                           states, delta, weight, scale)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), trainer.config.grad_clip,
                                           error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            critic.eval()
            errors.append((critic(dev_states) - truth).square().cpu().numpy())
        candidates.append((copy.deepcopy(critic.state_dict()), copy.deepcopy(optimizer.state_dict())))
        del critic, optimizer
    _restore_rng(rng, trainer.device)
    selected, report = select_largest(errors, dev.provenance['seeds'])
    critic_state, optimizer_state = candidates[CANDIDATES.index(selected)]
    trainer.critic.load_state_dict(critic_state)
    trainer.critic_optimizer.load_state_dict(optimizer_state)
    trainer.target_critic.load_state_dict(critic_state)
    trainer.config = replace(trainer.config, real_return_mix=selected)
    receipt.update(report, selected_share=selected, selected_critic_sha256=tensor_digest(trainer.critic))
    return receipt
