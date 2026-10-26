"""S-LONGCRITIC: real-anchored, difference-only long value regression (no simulator)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field, replace

import torch

from .counterfactual import KIND_ALL, KIND_BASE, Variants, acting_cavs
from .dreamer import _restore_rng, _rng_state
from .real_returns14 import warm_start_real_critic
from .start_states import checked_path


@dataclass
class LongCriticConfig:
    enabled: bool = False
    reliability_file: str | None = None
    native_returns: str | None = None
    horizons_s: list = field(default_factory=lambda: [60, 120, 180, 240, 300])
    every: int = 1
    native_warmup_updates: int = 100

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("longcritic.enabled must be boolean")
        if (not self.horizons_s or any(type(h) is not int or not 60 <= h <= 300
                                       for h in self.horizons_s)
                or self.horizons_s != sorted(set(self.horizons_s))):
            raise ValueError("longcritic horizons must be unique increasing integers in 60..300")
        for key in ("every", "native_warmup_updates"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"longcritic.{key} must be a positive integer")


def reliability_weights(doc, wm_sha256s):
    """95% lower bounds, not tuned behavior cutoffs; absent/stale evidence = zero.

    Chance-corrected sign skill times squared positive correlation, using the
    existing GROUP-bootstrap intervals (preserves dependence between arms).
    No horizon interpolation/extrapolation and no transfer between checkpoints.
    """
    if (doc.get("schema") != "coopw_horizon_gate/1"
            or doc.get("measurement_error")
            or set(wm_sha256s) != {doc.get("wm_sha256")}):
        return {}
    out = {}
    for h, row in doc.get("horizons", {}).items():
        try:
            sign = float(row["sign_ci95"][0])
            corr = float(row["pearson_ci95"][0])
            groups = int(row["n_groups"])
            horizon = int(h)
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if groups < 3 or not (math.isfinite(sign) and math.isfinite(corr)
                              and 0 <= sign <= 1 and -1 <= corr <= 1):
            continue
        out[horizon] = max(0., 2 * sign - 1) * max(0., corr) ** 2
    return out


def difference_target(policy_rewards, native_rewards, reliability):
    """[future,control-step,start] -> shrunk raw reward difference, never levels.

    Empirical signal/(signal + variance of the mean) adds per-start uncertainty
    shrinkage to the independently measured, horizon-specific reliability.
    """
    delta = (policy_rewards - native_rewards).sum(1).detach()
    if delta.shape[0] < 2:
        # One deterministic future cannot measure predictive uncertainty.
        return delta.mean(0) * 0, torch.zeros_like(delta.mean(0))
    mean = delta.mean(0)
    variance = delta.var(0, unbiased=True) / delta.shape[0]
    shrink = mean.square() / (mean.square() + variance).clamp_min(torch.finfo(mean.dtype).tiny)
    weight = shrink * float(reliability)
    return mean * weight, weight


def difference_loss(critic, native_critic, states, delta, weight, scale):
    """Real-only native value supplies the anchor. Both model outputs enter ONLY as Δ.

    The trained policy critic is the SAME critic used by the actor's short-horizon
    terminal bootstrap. Its state derivatives, rather than an additive constant in
    the actor loss, carry the delayed payoff into the policy gradient.
    """
    with torch.no_grad():
        anchor = native_critic(states)
    prediction = critic(states)
    return (weight.detach() * ((prediction - anchor - delta.detach()) / scale).square()).mean()


class LongCritic:
    def __init__(self, trainer, config, native_provider, wm_sha256s):
        self.config = config
        if trainer.config.real_return_mix <= 0 or trainer.config.gamma != 1:
            raise ValueError("longcritic requires real_return_mix>0 and gamma=1 (no WM level targets)")
        if native_provider.provenance["source_kind"] != "native":
            raise ValueError("longcritic anchor must contain real native returns")
        if not bool((native_provider.provenance["horizon_s"] >= max(config.horizons_s)).all()):
            raise ValueError("longcritic native anchor must cover every configured difference horizon")
        real_meta = getattr(trainer.real_return_provider, 'provenance', {})
        if 'horizon_s' in real_meta and set(real_meta['horizon_s'].tolist()) != set(native_provider.provenance['horizon_s'].tolist()):
            raise ValueError('real policy and native anchor continuation horizons must match')
        self.provider = native_provider
        self.native_critic = copy.deepcopy(trainer.critic)
        self.optimizer = torch.optim.Adam(self.native_critic.parameters(), lr=trainer.config.critic_lr)
        self.warm = False
        self.weights = {}
        self.receipt = {"weights": {}, "rule": "max(0,2*sign_lo95-1)*max(0,corr_lo95)^2",
                        "target": "V_policy(s)-stop(V_native_real(s)) = shrunk WM reward difference",
                        "native_horizons_s": sorted(set(native_provider.provenance["horizon_s"].tolist())),
                        "tail_assumption": "after selected horizon, residual policy-native advantage is zero"}
        self.receipt['native_source_sha256'] = (hashlib.sha256(native_provider.path.read_bytes()).hexdigest()
                                               if hasattr(native_provider, 'path') else None)
        seed_meta = native_provider.provenance.get('seeds')
        self.receipt['native_fit_seeds'] = sorted(set(seed_meta.tolist())) if seed_meta is not None else None
        if config.reliability_file:
            try:
                path = checked_path(config.reliability_file)
                doc = json.loads(path.read_text(encoding="utf-8"))
                self.weights = reliability_weights(doc, wm_sha256s)
                self.receipt.update(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    wm_sha256=doc.get("wm_sha256"), metric_rule=doc.get("rule"))
            except (OSError, ValueError) as error:
                self.receipt["refused"] = str(error)
        self.receipt["weights"] = self.weights

    def warmup(self, trainer):
        # Reuse the authoritative-real-only fitter without modifying the policy
        # trainer's critic, optimiser, dual, update count or RNG stream.
        from types import SimpleNamespace
        proxy = SimpleNamespace(config=trainer.config, critic=self.native_critic,
                                target_critic=copy.deepcopy(self.native_critic),
                                critic_optimizer=self.optimizer, real_return_provider=self.provider,
                                device=trainer.device)
        receipt = warm_start_real_critic(proxy, self.config.native_warmup_updates)
        self.native_critic.eval().requires_grad_(False)
        self.warm = True
        return receipt

    def targets(self, trainer, batch):
        active = [h for h in self.config.horizons_s if self.weights.get(h, 0) > 0]
        metrics = {"longcritic_weights": self.weights, "longcritic_active": False}
        if not active or trainer.update_count % self.config.every:
            return None, metrics
        if not self.warm:
            raise ValueError("longcritic native anchor has not been warmed on real returns")
        h = active[(trainer.update_count // self.config.every) % len(active)]
        # Budgeted one start per update, deterministic rotation; preserve original
        # history, demand, entry queue and CRNs, never reconstruct driver labels.
        from .start_states import StartBatch
        i = trainer.update_count % batch.state["s_m"].shape[0]
        n = batch.state["s_m"].shape[0]
        def take(value):
            if torch.is_tensor(value) and value.ndim and value.shape[0] == n:
                return value[i:i + 1]
            if isinstance(value, dict):
                return {k: take(v) for k, v in value.items()}
            return value
        start = StartBatch(take(batch.state), rollout_kwargs=take(batch.rollout_kwargs),
                           metadata=batch.metadata[i:i + 1], entry_feat=take(batch.entry_feat),
                           entry_queue=take(batch.entry_queue))
        old = trainer.config
        training, curriculum = trainer._training_imagination, trainer.horizon_curriculum
        rng = _rng_state(trainer.device)
        try:
            trainer.config = replace(old, horizon=h, low_penetration_horizon_cap=False,
                                     future_samples=max(2, old.future_samples),
                                     entry_pad_max=max(old.entry_pad_max,
                                                       math.ceil(old.entry_pad_max * h / old.horizon)))
            trainer._training_imagination, trainer.horizon_curriculum = True, None
            with torch.no_grad():
                start = trainer._entry_slots(start, torch.Generator().manual_seed(7_000_003 + trainer.update_count))
                cav = acting_cavs(start.state)
                variants = Variants(torch.tensor([0, 0], device=trainer.device),
                                    torch.cat((cav, torch.zeros_like(cav))),
                                    torch.tensor([KIND_ALL, KIND_BASE], device=trainer.device),
                                    1, {0: 1}, {}, [])
                imagined = trainer.imagine(start, variants=variants, difference_only=True)
                delta, weight = difference_target(imagined["rewards"][..., :1],
                                                   imagined["rewards"][..., 1:], self.weights[h])
                states = {k: v.detach() for k, v in start.state.items()}
        finally:
            trainer.config = old
            trainer._training_imagination, trainer.horizon_curriculum = training, curriculum
            _restore_rng(rng, trainer.device)
        metrics.update(longcritic_active=True, longcritic_horizon_s=h,
                       longcritic_delta=float(delta.mean()), longcritic_weight=float(weight.mean()))
        return (states, delta, weight), metrics

    def loss(self, trainer, batch, scale):
        target, metrics = self.targets(trainer, batch)
        if target is None:
            return None, metrics
        states, delta, weight = target
        loss = difference_loss(trainer.critic, self.native_critic, states, delta, weight,
                               max(scale, trainer.config.real_return_scale))
        metrics['longcritic_loss'] = float(loss.detach())
        return loss, metrics

    def state_dict(self):
        return {"native_critic": self.native_critic.state_dict(), "optimizer": self.optimizer.state_dict(),
                "warm": self.warm, "receipt": self.receipt}

    def load_state_dict(self, doc):
        prior = doc.get('receipt', {})
        if (prior.get('native_source_sha256') != self.receipt.get('native_source_sha256')
                or prior.get('native_fit_seeds') != self.receipt.get('native_fit_seeds')
                or prior.get('native_horizons_s') != self.receipt.get('native_horizons_s')):
            self.receipt['anchor_reset'] = 'native returns or continuation horizon changed; rewarm only fitting worlds'
            self.warm = False
            self.native_critic.requires_grad_(True)
            return
        self.native_critic.load_state_dict(doc["native_critic"])
        self.optimizer.load_state_dict(doc["optimizer"])
        self.warm = bool(doc["warm"])
        if self.warm:
            self.native_critic.eval().requires_grad_(False)
