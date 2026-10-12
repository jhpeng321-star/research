"""Small shared CAV actor, zone-conditioned RSU encoder and privileged critic."""
from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal

from .contract import (ACCEL_MAX_MPS2, ACCEL_MIN_MPS2, F_OWN, F_REPORT, GROUP_SIGNAL_DIM,
                       MERGE_S0_M, MERGE_S1_M, OWN_FEATURES, REPORT_FEATURES, Observation)
from .obs import FEATURE_NORMALIZATION, normalize_features


def _masked_attention(attention: nn.MultiheadAttention, query: Tensor, keys: Tensor,
                      mask: Tensor) -> Tensor:
    """Avoid all-masked softmax rows with a zero dummy key for empty sets."""
    empty = ~mask.any(-1, keepdim=True)
    keys = torch.cat((keys, keys.new_zeros(keys.shape[0], 1, keys.shape[-1])), 1)
    mask = torch.cat((mask, empty), 1)
    out, _ = attention(query, keys, keys, key_padding_mask=~mask, need_weights=False)
    return torch.where(empty[..., None], torch.zeros_like(out), out)


class CavPolicy(nn.Module):
    """Shared actor: (Observation, group [B,K,8]) -> (accel, log_prob) [B,K]."""

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        self.own_encoder = nn.Sequential(nn.Linear(F_OWN, hidden_dim), nn.SiLU(),
                                         nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.report_encoder = nn.Sequential(nn.Linear(F_REPORT, hidden_dim), nn.SiLU())
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim + GROUP_SIGNAL_DIM, hidden_dim),
                                  nn.SiLU(), nn.Linear(hidden_dim, 2))

    def forward(self, obs: Observation, group_signal: Tensor, *, deterministic: bool = False
                ) -> tuple[Tensor, Tensor]:
        b, k, _ = obs.own.shape
        own = self.own_encoder(normalize_features(obs.own, OWN_FEATURES))
        reports = torch.where(obs.report_mask[..., None], obs.reports,
                              torch.zeros_like(obs.reports))
        reports = self.report_encoder(normalize_features(reports, REPORT_FEATURES))
        context = _masked_attention(self.attention, own.reshape(b * k, 1, -1),
                                    reports.reshape(b * k, reports.shape[2], -1),
                                    obs.report_mask.reshape(b * k, -1)).reshape(b, k, -1)
        signal = torch.where(obs.cav_mask[..., None], group_signal, torch.zeros_like(group_signal))
        mean, log_std = self.head(torch.cat((own, context, signal), -1)).unbind(-1)
        distribution = Normal(mean, log_std.clamp(-5.0, 1.0).exp())
        latent = mean if deterministic else distribution.rsample()
        scale = (ACCEL_MAX_MPS2 - ACCEL_MIN_MPS2) / 2.0
        midpoint = (ACCEL_MAX_MPS2 + ACCEL_MIN_MPS2) / 2.0
        accel = midpoint + scale * torch.tanh(latent)
        # Stable log(1-tanh(z)^2), including saturated latent values.
        log_jacobian = math.log(scale) + 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))
        log_prob = distribution.log_prob(latent) - log_jacobian
        return (torch.where(obs.cav_mask, accel, torch.zeros_like(accel)),
                torch.where(obs.cav_mask, log_prob, torch.zeros_like(log_prob)))


class _SetBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.SiLU(),
                                 nn.Linear(2 * hidden_dim, hidden_dim))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.norm1(x + _masked_attention(self.attention, x, x, mask))
        x = self.norm2(x + self.ffn(x))
        return torch.where(mask[..., None], x, torch.zeros_like(x))


class RsuGroupEncoder(nn.Module):
    """CAV-only set encoder: (obs, positions [B,K]) -> per-CAV signals [B,K,8].

    Smooth upstream/merge/downstream zone memberships condition a shared pooled
    set context. There is no vehicle-count, density or penetration input.
    """

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        self.input = nn.Linear(F_OWN + 1, hidden_dim)
        self.set_block = _SetBlock(hidden_dim, n_heads)
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim + 4, hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, GROUP_SIGNAL_DIM), nn.Tanh())

    def forward(self, obs: Observation, positions: Tensor) -> Tensor:
        mask = obs.cav_mask
        s = torch.where(mask, positions, torch.zeros_like(positions))
        position = normalize_features(s[..., None], ("s_m",))
        x = self.input(torch.cat((normalize_features(obs.own, OWN_FEATURES), position), -1))
        x = self.set_block(x, mask)
        pooled = x.sum(1, keepdim=True) / mask.sum(1).clamp_min(1)[:, None, None]
        upstream = torch.sigmoid((MERGE_S0_M - s) / 10.0)
        downstream = torch.sigmoid((s - MERGE_S1_M) / 10.0)
        zones = torch.stack((upstream, 1.0 - upstream - downstream, downstream), -1)
        signal = self.head(torch.cat((x, pooled.expand_as(x), position, zones), -1))
        return torch.where(mask[..., None], signal, torch.zeros_like(signal))


class TeamCritic(nn.Module):
    """Privileged all-present-vehicle set transformer: state -> value [B]."""

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.input = nn.Linear(8, hidden_dim)
        self.blocks = nn.ModuleList([_SetBlock(hidden_dim, n_heads) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, 1))

    def forward(self, state: Mapping[str, Tensor]) -> Tensor:
        mask = state["present"].bool()
        if all(name in state for name in ("s_m", "v_mps", "a_mps2", "lane")):
            # Scalar fields are authoritative after WM-S rollout steps. Keep
            # deployment networks independent of the world-model package.
            lane = state["lane"].long()
            onehot = F.one_hot(lane.clamp(0, 2), 3).to(state["s_m"].dtype)
            onehot = torch.where((lane >= 0)[..., None], onehot, torch.zeros_like(onehot))
            length = state.get("length_m", torch.full_like(
                state["s_m"], FEATURE_NORMALIZATION["length_m"]))
            kin = torch.cat((state["s_m"][..., None], onehot, state["v_mps"][..., None],
                             state["a_mps2"][..., None], length[..., None]), -1)
        else:
            kin = state["veh_kin"]
        kin = torch.where(mask[..., None], kin, torch.zeros_like(kin))
        features = ("s_m", "lane_0", "lane_1", "lane_2", "v_mps", "a_mps2", "length_m")
        cav = torch.where(mask, state["is_cav"].bool(), torch.zeros_like(mask)).to(kin.dtype)
        x = self.input(torch.cat((normalize_features(kin, features), cav[..., None]), -1))
        for block in self.blocks:
            x = block(x, mask)
        pooled = x.sum(1) / mask.sum(1).clamp_min(1)[..., None]
        value = self.head(pooled).squeeze(-1)
        return torch.where(mask.any(1), value, torch.zeros_like(value))
