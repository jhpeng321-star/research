"""Tests for FeatureEncoder.forward_components and per-component pooling."""

import torch

from env.features import (
    raw_obs_dim,
)
from models import FeatureEncoder


def test_forward_components_shapes():
    """forward_components should return (h_ego, h_nbr, h_zone) with correct dims."""
    encoder = FeatureEncoder(max_neighbors=20)
    dim = raw_obs_dim(20)
    # Create a dummy observation
    obs = torch.randn(4, dim)
    h_ego, h_nbr, h_zone = encoder.forward_components(obs)
    assert h_ego.shape == (4, 32)
    assert h_nbr.shape == (4, 32)
    assert h_zone.shape == (4, 64)


def test_forward_equals_concat_components():
    """forward() should equal cat(forward_components())."""
    encoder = FeatureEncoder(max_neighbors=20)
    dim = raw_obs_dim(20)
    obs = torch.randn(2, dim)
    h_ego, h_nbr, h_zone = encoder.forward_components(obs)
    z = encoder.forward(obs)
    z_cat = torch.cat([h_ego, h_nbr, h_zone], dim=-1)
    assert torch.allclose(z, z_cat, atol=1e-6)


def test_critic_per_component_pooling():
    """Simulate the MAPPO critic's per-component pooling logic."""
    encoder = FeatureEncoder(max_neighbors=20)
    dim = raw_obs_dim(20)
    max_cavs = 4

    # Simulate 2 agents, each with 4 peers (2 active, 2 padded)
    b = 2
    peers = torch.randn(b, max_cavs, dim)
    trust = torch.zeros(b, max_cavs)
    trust[:, 0] = 0.9  # peer 0 active
    trust[:, 1] = 0.6  # peer 1 active
    # peers 2, 3 are padded (trust = 0)

    peers_flat = peers.reshape(b * max_cavs, dim)
    h_ego_all, h_nbr_all, h_zone_all = encoder.forward_components(peers_flat)

    h_ego_all = h_ego_all.view(b, max_cavs, -1)
    h_nbr_all = h_nbr_all.view(b, max_cavs, -1)
    h_zone_all = h_zone_all.view(b, max_cavs, -1)

    mask = (trust > 0).float()
    mask_f = mask.unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp(min=1.0)

    # Ego: masked mean
    pooled_ego = (h_ego_all * mask_f).sum(dim=1) / denom
    assert pooled_ego.shape == (b, 32)

    # Neighbors: trust-weighted sum
    trust_f = trust.unsqueeze(-1)
    pooled_nbr = (h_nbr_all * trust_f).sum(dim=1)
    assert pooled_nbr.shape == (b, 32)

    # Zone: masked mean
    pooled_zone = (h_zone_all * mask_f).sum(dim=1) / denom
    assert pooled_zone.shape == (b, 64)

    # Concatenate
    z_critic = torch.cat([pooled_ego, pooled_nbr, pooled_zone], dim=-1)
    assert z_critic.shape == (b, 128)


def test_padded_peers_zero_trust():
    """Padded peers (trust=0) should contribute nothing to neighbor pooling."""
    encoder = FeatureEncoder(max_neighbors=20)
    dim = raw_obs_dim(20)

    # One agent, 4 peer slots, only peer 0 active with trust 1.0
    peers = torch.randn(1, 4, dim)
    trust = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    peers_flat = peers.reshape(4, dim)
    _, h_nbr_all, _ = encoder.forward_components(peers_flat)
    h_nbr_all = h_nbr_all.view(1, 4, -1)

    trust_f = trust.unsqueeze(-1)
    pooled_nbr = (h_nbr_all * trust_f).sum(dim=1)

    # Should equal just peer 0's embedding
    assert torch.allclose(pooled_nbr[0], h_nbr_all[0, 0], atol=1e-6)


if __name__ == "__main__":
    test_forward_components_shapes()
    test_forward_equals_concat_components()
    test_critic_per_component_pooling()
    test_padded_peers_zero_trust()
    print("ok")
