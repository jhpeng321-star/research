"""Neural networks for all three baselines.

- `FeatureEncoder`  — shared observation encoder:
                      ego(4)->32 | neighbors(20x5)->32 (DeepSets: one shared
                      Linear(4->32)+ReLU per neighbor, trust-weighted mean
                      pooling) | zone(7)->64, concatenated to a 128-dim
                      embedding.
- `SelfishPPOModel` — local PPO/IPPO model for the Selfish baseline
                      (actor and critic both see only the local 128-dim encoding).
- `MAPPOModel`      — MAPPO model for Flat-MARL / Shepherd: decentralized actor
                      on the local encoding, centralized critic that pools all
                      active peer CAVs per component (ego/zone masked mean,
                      neighbors trust-weighted sum).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

import env.features as features
from env.features import (
    MAX_NEIGHBORS_DEFAULT,
    NEIGHBOR_SLOT,
    ZONE_FEATURE_DIM,
    raw_obs_dim,
)
from env.mappo import mappo_obs_dim

SELFISH_MODEL_ID = "selfish_ppo_model"
MAPPO_MODEL_ID = "mappo_model"


class FeatureEncoder(nn.Module):
    """Shared observation encoder for Selfish PPO / Flat-MARL / Shepherd."""

    def __init__(
        self,
        max_neighbors: int = MAX_NEIGHBORS_DEFAULT,
        ego_embed_dim: int = 32,
        nbr_embed_dim: int = 32,
        zone_embed_dim: int = 64,
        context_dim: int = 0,
        context_embed_dim: int = 32,
    ):
        super().__init__()
        self.max_neighbors = max_neighbors
        self.ego_embed_dim = ego_embed_dim
        self.nbr_embed_dim = nbr_embed_dim
        self.zone_embed_dim = zone_embed_dim
        self.context_dim = max(0, int(context_dim))
        self.context_embed_dim = int(context_embed_dim) if self.context_dim else 0
        self.out_dim = ego_embed_dim + nbr_embed_dim + zone_embed_dim + self.context_embed_dim

        # Read EGO_DIM at construct time so legacy checkpoints (ego=4) can
        # monkeypatch env.features.EGO_DIM before Policy.from_checkpoint.
        self.ego_dim = int(features.EGO_DIM)
        self.ego_embed = nn.Sequential(
            nn.Linear(self.ego_dim, ego_embed_dim),
            nn.ReLU(),
        )
        # DeepSets neighbor branch: trust is a pooling weight, not an input.
        self.nbr_embed = nn.Sequential(
            nn.Linear(NEIGHBOR_SLOT - 1, nbr_embed_dim),
            nn.ReLU(),
        )
        self.zone_embed = nn.Sequential(
            nn.Linear(ZONE_FEATURE_DIM, zone_embed_dim),
            nn.ReLU(),
        )
        self.context_embed = (
            nn.Sequential(
                nn.Linear(self.context_dim, self.context_embed_dim),
                nn.ReLU(),
            )
            if self.context_dim
            else None
        )

    def forward(self, raw_obs: torch.Tensor) -> torch.Tensor:
        """Encode a batch, including every declared physical context field."""
        return torch.cat(self.forward_all_components(raw_obs), dim=-1)

    def forward_all_components(self, raw_obs: torch.Tensor) -> tuple:
        """Return ego, neighbor, zone, and optional-context embeddings."""
        h_ego, h_nbr, h_zone, context = self._encode_fixed(raw_obs)
        if self.context_embed is None:
            h_context = raw_obs.new_zeros((h_ego.shape[0], 0))
        else:
            h_context = self.context_embed(context)
        return h_ego, h_nbr, h_zone, h_context

    def forward_components(self, raw_obs: torch.Tensor) -> tuple:
        """Legacy three-component view retained for context-free checkpoints."""
        h_ego, h_nbr, h_zone, _context = self._encode_fixed(raw_obs)
        return h_ego, h_nbr, h_zone

    def _encode_fixed(self, raw_obs: torch.Tensor) -> tuple:
        if raw_obs.dim() == 1:
            raw_obs = raw_obs.unsqueeze(0)
        raw_obs = torch.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0)

        ego_dim = self.ego_dim
        ego = raw_obs[:, :ego_dim]
        zone = raw_obs[:, ego_dim : ego_dim + ZONE_FEATURE_DIM]
        nbr_start = ego_dim + ZONE_FEATURE_DIM
        nbr_end = nbr_start + self.max_neighbors * NEIGHBOR_SLOT
        nbr_flat = raw_obs[:, nbr_start:nbr_end]
        nbr = nbr_flat.view(-1, self.max_neighbors, NEIGHBOR_SLOT)
        context = raw_obs[:, nbr_end : nbr_end + self.context_dim]
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                f"observation has {raw_obs.shape[-1]} physical fields but encoder "
                f"requires {nbr_end + self.context_dim} (context_dim={self.context_dim})"
            )

        h_ego = self.ego_embed(ego)
        h_zone = self.zone_embed(zone)
        feats = nbr[:, :, : NEIGHBOR_SLOT - 1]
        w = nbr[:, :, NEIGHBOR_SLOT - 1 :]
        e = self.nbr_embed(feats)
        denom = w.sum(dim=1).clamp(min=1.0)
        h_nbr = (e * w).sum(dim=1) / denom
        return h_ego, h_nbr, h_zone, context


def _init_log_std_bias(
    policy_head: nn.Sequential,
    log_std_init: float = -2.0,
    *,
    zero_mean: bool = False,
) -> None:
    """Init the final layer's log_std bias so a continuous policy starts near
    coast (small std) instead of RLlib's default std=1.0.

    Three 600k campaigns on the dense corridor crashed at iteration ~3
    regardless of reward shaping: with std=1.0 m/s^2 the untrained Gaussian
    jerks full-range every step, gridlocking traffic from step 0 — the critic
    sees no contrast and PPO flails. The discrete action space ships the same
    property for free (20% coast at random init + bounded modes). This gives
    the continuous head the equivalent start: mean ~0, std ~ e^-2 = 0.14.
    Output layout is [mean..., log_std...]: bias the second half.
    """
    final = None
    for m in reversed(policy_head):
        if isinstance(m, nn.Linear):
            final = m
            break
    if final is None or final.bias is None:
        return
    n = final.bias.shape[0] // 2
    if n == 0:
        return
    with torch.no_grad():
        if zero_mean:
            # IDM residual mode: exact zero nominal residual for every state.
            final.weight[:n].zero_()
            final.bias[:n].zero_()
        # State-independent narrow initial exploration distribution.
        final.weight[n:].zero_()
        final.bias[n:] = float(log_std_init)


def _init_hybrid_dial_bias(
    policy_head: nn.Sequential,
    n_modes: int,
    dial_init: float = -0.7,
    log_std_init: float = -2.0,
) -> None:
    """Hybrid composite head init (7-output layout [logits(n) | mean | log_std]):
    dial mean bias -> gentle default (raw -0.7 -> mag ~0.15), dial log_std
    bias -> tight exploration (std 0.14). v1/v2 (2026-08-05) intervened at
    half strength everywhere at init and harmed every seed; gentle init lets
    mode selection learn before strength matters.
    """
    if n_modes <= 0:
        return
    final = None
    for m in reversed(policy_head):
        if isinstance(m, nn.Linear):
            final = m
            break
    if final is None or final.bias is None:
        return
    if final.bias.shape[0] != n_modes + 2:
        return
    with torch.no_grad():
        final.bias[n_modes] = float(dial_init)  # dial mean
        final.bias[n_modes + 1] = float(log_std_init)  # dial log_std


class SelfishPPOModel(TorchModelV2, nn.Module):
    """Selfish (local PPO/IPPO) model: FeatureEncoder + local policy/value heads."""

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom = dict(model_config.get("custom_model_config", {}) or {})
        custom.update(kwargs)
        max_neighbors = int(custom.get("max_neighbors", 20))
        hidden = int(custom.get("hidden", 128))
        self.local_dim = int(custom.get("local_obs_dim", obs_space.shape[0]))
        self.action_mask_dim = int(custom.get("action_mask_dim", 0))
        self.raw_local_dim = int(
            custom.get(
                "raw_local_dim",
                max(0, self.local_dim - self.action_mask_dim),
            )
        )
        self.force_action_zero_legal = bool(
            custom.get("force_action_zero_legal", True)
        )
        self.encoder = FeatureEncoder(
            max_neighbors=max_neighbors,
            context_dim=int(custom.get("context_dim", 0)),
            context_embed_dim=int(custom.get("context_embed_dim", 32)),
        )
        d = self.encoder.out_dim
        self.policy_head = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_outputs),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._value = None
        self._continuous = isinstance(action_space, gym.spaces.Box)
        if self._continuous:
            _init_log_std_bias(
                self.policy_head,
                custom.get("log_std_init", -2.0),
                zero_mean=bool(custom.get("zero_mean_init", False)),
            )
            _init_hybrid_dial_bias(
                self.policy_head, num_outputs, custom.get("hybrid_dial_init", -0.7)
            )

    @override(ModelV2)
    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        obs = input_dict["obs"].float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        raw = obs[:, : self.raw_local_dim] if self.action_mask_dim > 0 else obs
        z = self.encoder(raw)
        self._value = self.value_head(z).squeeze(-1)
        logits = self.policy_head(z)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        # DiagGaussian: last dim is log_std — keep Normal(loc, scale) finite.
        if self._continuous and logits.shape[-1] >= 2:
            # RLlib DiagGaussian layout: [mean..., log_std...] in halves.
            # Half-split is back-compatible with Box(1) (n=1).
            n = logits.shape[-1] // 2
            mean = torch.clamp(logits[..., :n], -10.0, 10.0)
            log_std = torch.clamp(logits[..., n:], -5.0, 2.0)
            logits = torch.cat([mean, log_std], dim=-1)
        elif self.action_mask_dim > 0:
            amask = obs[:, -self.action_mask_dim :].clone()
            if self.force_action_zero_legal:
                amask[:, 0] = 1.0
            all_zero = amask.sum(dim=-1) <= 0.0
            amask[all_zero, 0] = 1.0
            logits = logits + torch.log(amask.clamp(min=1e-10))
        return logits, state

    @override(ModelV2)
    def value_function(self) -> torch.Tensor:
        assert self._value is not None
        return self._value


class MAPPOModel(TorchModelV2, nn.Module):
    """Flat-MARL / Shepherd MAPPO model: local actor + per-component pooled critic.

    Packed obs layout: local | peers(max_cavs * local) | trust_weights(max_cavs).
    Critic pools:
      - ego: masked mean across CAVs -> 32
      - nbr: trust-weighted sum across CAVs -> 32
      - zone: masked mean across CAVs -> 64
      Total: 128-dim -> value_head.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom = dict(model_config.get("custom_model_config", {}) or {})
        custom.update(kwargs)
        max_neighbors = int(custom.get("max_neighbors", 20))
        hidden = int(custom.get("hidden", 128))
        self.max_cavs = int(custom.get("max_cavs_critic", 32))
        self.local_dim = int(custom.get("local_obs_dim", raw_obs_dim(max_neighbors)))
        self.action_mask_dim = int(custom.get("action_mask_dim", 0))
        self.raw_local_dim = int(
            custom.get(
                "raw_local_dim",
                max(0, self.local_dim - self.action_mask_dim),
            )
        )
        self.force_action_zero_legal = bool(
            custom.get("force_action_zero_legal", True)
        )

        self.encoder = FeatureEncoder(
            max_neighbors=max_neighbors,
            context_dim=int(custom.get("context_dim", 0)),
            context_embed_dim=int(custom.get("context_embed_dim", 32)),
        )
        d = self.encoder.out_dim  # 128

        self.policy_head = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_outputs),
        )
        # Critic input is 128-dim (same as actor), not 256.
        self.value_head = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._value = None
        self._continuous = isinstance(action_space, gym.spaces.Box)
        # Hybrid (mode, magnitude): Tuple(Discrete(n_modes), Box(1)) — a
        # Categorical mode + Gaussian dial composite (RLlib auto-builds
        # TorchMultiActionDistribution for the Tuple). num_outputs = n+2.
        self._hybrid = isinstance(action_space, gym.spaces.Tuple)
        self._n_modes = int(action_space.spaces[0].n) if self._hybrid else 0
        _ = mappo_obs_dim(self.local_dim, self.max_cavs)  # documented layout
        if self._continuous:
            _init_log_std_bias(
                self.policy_head,
                custom.get("log_std_init", -2.0),
                zero_mean=bool(custom.get("zero_mean_init", False)),
            )
        if self._hybrid:
            _init_hybrid_dial_bias(
                self.policy_head,
                self._n_modes,
                custom.get("hybrid_dial_init", -0.7),
                custom.get("log_std_init", -2.0),
            )
        if (
            not self._continuous
            and not self._hybrid
            and bool(custom.get("zero_mean_init", False))
        ):
            # A fresh binary activation gate must start at exactly 50/50 on
            # genuine two-action decisions.  Masked samples remain deterministic.
            final = self.policy_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def _split_obs(self, obs: torch.Tensor):
        """Split packed obs into local, peers, trust_weights."""
        local = obs[:, : self.local_dim]
        peer_flat = obs[:, self.local_dim : self.local_dim + self.max_cavs * self.local_dim]
        peers = peer_flat.view(-1, self.max_cavs, self.local_dim)
        trust = obs[:, -self.max_cavs :]
        return local, peers, trust

    @override(ModelV2)
    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        obs = input_dict["obs"].float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        local, peers, trust = self._split_obs(obs)

        # Actor: encode local obs features only (mask bits are not embedded).
        if self.action_mask_dim > 0:
            z_i = self.encoder(local[:, : self.raw_local_dim])
        else:
            z_i = self.encoder(local)
        logits = self.policy_head(z_i)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        if self._hybrid and logits.shape[-1] == self._n_modes + 2:
            # Composite output: [mode logits | dial mean | dial log_std].
            # The eligibility mask rides the obs and applies to the mode
            # logits MODEL-SIDE (identical semantics to the discrete
            # champion) — no env-side argmax, no gradient mismatch.
            mode_logits = logits[..., : self._n_modes]
            dial_mean = torch.clamp(logits[..., self._n_modes : self._n_modes + 1], -1.0, 1.0)
            dial_log_std = torch.clamp(logits[..., -1:], -5.0, 2.0)
            if self.action_mask_dim > 0:
                amask = local[:, -self.action_mask_dim :]
                amask = amask.clone()
                if self.force_action_zero_legal:
                    amask[:, 0] = 1.0
                all_zero = amask.sum(dim=-1) <= 0.0
                amask[all_zero, 0] = 1.0
                mode_logits = mode_logits + torch.log(amask.clamp(min=1e-10))
            logits = torch.cat([mode_logits, dial_mean, dial_log_std], dim=-1)
        elif self._continuous and logits.shape[-1] >= 2:
            # RLlib DiagGaussian layout: [mean..., log_std...] in halves.
            # Half-split is back-compatible with Box(1) (n=1).
            n = logits.shape[-1] // 2
            mean = torch.clamp(logits[..., :n], -10.0, 10.0)
            log_std = torch.clamp(logits[..., n:], -5.0, 2.0)
            logits = torch.cat([mean, log_std], dim=-1)
        elif self.action_mask_dim > 0 and not self._continuous:
            # Illegal modes → ~0 probability (coast always unmasked).
            amask = local[:, -self.action_mask_dim :]
            amask = amask.clone()
            if self.force_action_zero_legal:
                amask[:, 0] = 1.0
            all_zero = amask.sum(dim=-1) <= 0.0
            amask[all_zero, 0] = 1.0
            logits = logits + torch.log(amask.clamp(min=1e-10))

        # Critic: encode all CAVs, pool per-component.
        b = peers.shape[0]
        peers_flat = peers.reshape(b * self.max_cavs, self.local_dim)
        if self.action_mask_dim > 0:
            peers_raw = peers_flat[:, : self.raw_local_dim]
        else:
            peers_raw = peers_flat
        h_ego_all, h_nbr_all, h_zone_all, h_context_all = self.encoder.forward_all_components(
            peers_raw
        )

        h_ego_all = h_ego_all.view(b, self.max_cavs, -1)  # (B, max_cavs, 32)
        h_nbr_all = h_nbr_all.view(b, self.max_cavs, -1)  # (B, max_cavs, 32)
        h_zone_all = h_zone_all.view(b, self.max_cavs, -1)  # (B, max_cavs, 64)
        h_context_all = h_context_all.view(b, self.max_cavs, -1)

        # Mask: active = trust > 0
        mask = (trust > 0).float()
        mask_f = mask.unsqueeze(-1)  # (B, max_cavs, 1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)

        # Ego: masked mean
        pooled_ego = (h_ego_all * mask_f).sum(dim=1) / denom  # (B, 32)

        # Neighbors: trust-weighted sum
        trust_f = trust.unsqueeze(-1)  # (B, max_cavs, 1)
        pooled_nbr = (h_nbr_all * trust_f).sum(dim=1)  # (B, 32)

        # Zone and optional physical context: masked means over active CAVs.
        pooled_zone = (h_zone_all * mask_f).sum(dim=1) / denom  # (B, 64)
        pooled_context = (h_context_all * mask_f).sum(dim=1) / denom

        z_critic = torch.cat([pooled_ego, pooled_nbr, pooled_zone, pooled_context], dim=-1)
        self._value = self.value_head(z_critic).squeeze(-1)
        self._value = torch.nan_to_num(self._value, nan=0.0, posinf=0.0, neginf=0.0)

        return logits, state

    @override(ModelV2)
    def value_function(self) -> torch.Tensor:
        assert self._value is not None
        return self._value


def register_selfish_ppo_model() -> None:
    """Register the Selfish local PPO model with RLlib's ModelCatalog."""
    from ray.rllib.models import ModelCatalog

    ModelCatalog.register_custom_model(SELFISH_MODEL_ID, SelfishPPOModel)


def register_mappo_model() -> None:
    """Register the MAPPO model with RLlib's ModelCatalog."""
    from ray.rllib.models import ModelCatalog

    ModelCatalog.register_custom_model(MAPPO_MODEL_ID, MAPPOModel)
