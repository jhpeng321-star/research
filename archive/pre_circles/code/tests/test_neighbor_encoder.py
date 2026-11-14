"""Tests for the DeepSets neighbor encoder branch of FeatureEncoder.

Design: one shared Linear(4 -> 32) + ReLU encodes each neighbor independently
from (rel_dist, rel_speed, rel_lane, turn_signal); trust is a pooling weight
only (never encoded); pooling is the trust-weighted *mean*
``sum(e * w) / max(sum(w), 1)`` so the embedding scale does not grow with the
number of neighbors.
"""

import numpy as np
import torch
import torch.nn as nn

from env.features import EGO_DIM, NEIGHBOR_SLOT, ZONE_FEATURE_DIM, raw_obs_dim
from models import FeatureEncoder


def _make_obs(neighbor_slots, max_neighbors: int = 20) -> torch.Tensor:
    """Build a raw obs tensor from [(features[4], trust_weight), ...] slots."""
    dim = raw_obs_dim(max_neighbors)
    obs = np.zeros(dim, dtype=np.float32)
    nbr_start = EGO_DIM + ZONE_FEATURE_DIM
    for i, (features, weight) in enumerate(neighbor_slots):
        base = nbr_start + i * NEIGHBOR_SLOT
        obs[base : base + 4] = features
        obs[base + 4] = weight
    return torch.from_numpy(obs).unsqueeze(0)


def test_shared_neighbor_embedding_structure():
    """One shared Linear(4->32)+ReLU; no per-feature Linear(1->32) layers."""
    enc = FeatureEncoder(max_neighbors=20)
    assert hasattr(enc, "nbr_embed"), "expected a shared nbr_embed module"
    assert isinstance(enc.nbr_embed, nn.Sequential)
    linear, activation = enc.nbr_embed[0], enc.nbr_embed[1]
    assert isinstance(linear, nn.Linear)
    assert linear.in_features == 4  # dist/speed/lane/signal; trust excluded
    assert linear.out_features == 32
    assert isinstance(activation, nn.ReLU)
    for legacy in ("embed_dist", "embed_speed", "embed_lane", "embed_signal"):
        assert not hasattr(enc, legacy), f"legacy per-feature layer {legacy}"


def test_output_dims_unchanged():
    enc = FeatureEncoder(max_neighbors=20)
    obs = torch.randn(3, raw_obs_dim(20))
    h_ego, h_nbr, h_zone = enc.forward_components(obs)
    assert h_ego.shape == (3, 32)
    assert h_nbr.shape == (3, 32)
    assert h_zone.shape == (3, 64)
    assert enc(obs).shape == (3, 128)


def test_mean_pooling_scale_invariant_to_neighbor_count():
    """3 vs 20 identical neighbors must pool to the same embedding (mean, not sum)."""
    torch.manual_seed(0)
    enc = FeatureEncoder(max_neighbors=20)
    features = [10.0, -2.0, 1.0, -1.0]
    obs_few = _make_obs([(features, 1.0)] * 3)
    obs_many = _make_obs([(features, 1.0)] * 20)
    _, h_few, _ = enc.forward_components(obs_few)
    _, h_many, _ = enc.forward_components(obs_many)
    assert torch.allclose(h_few, h_many, atol=1e-5), (
        "trust-weighted mean must be count-invariant; "
        f"|diff| max {(h_few - h_many).abs().max().item():.4f}"
    )


def test_zero_weight_padding_slots_do_not_affect_pooling():
    """Garbage in padding slots (weight 0) must not change the pooled embedding."""
    torch.manual_seed(0)
    enc = FeatureEncoder(max_neighbors=20)
    features = [5.0, 1.0, 0.0, -1.0]
    real = [(features, 0.8), (features, 0.6), (features, 0.4)]
    garbage = [([999.0, -999.0, 7.0, 1.0], 0.0)] * 17
    _, h_clean, _ = enc.forward_components(_make_obs(real))
    _, h_padded, _ = enc.forward_components(_make_obs(real + garbage))
    assert torch.allclose(h_clean, h_padded, atol=1e-5)


def test_trust_weights_act_as_mean_pooling_weights():
    """Weighted mean: pooled value lies at the trust-weighted average of members."""
    torch.manual_seed(0)
    enc = FeatureEncoder(max_neighbors=20)
    obs = _make_obs([([1.0, 0.0, 0.0, 0.0], 1.0), ([-1.0, 0.0, 0.0, 0.0], 3.0)])
    _, h_nbr, _ = enc.forward_components(obs)
    # Encode each neighbor alone (weight 1 → mean = its own embedding).
    _, h_a, _ = enc.forward_components(_make_obs([([1.0, 0.0, 0.0, 0.0], 1.0)]))
    _, h_b, _ = enc.forward_components(_make_obs([([-1.0, 0.0, 0.0, 0.0], 1.0)]))
    expected = (1.0 * h_a + 3.0 * h_b) / 4.0
    assert torch.allclose(h_nbr, expected, atol=1e-5)


if __name__ == "__main__":
    test_shared_neighbor_embedding_structure()
    test_output_dims_unchanged()
    test_mean_pooling_scale_invariant_to_neighbor_count()
    test_zero_weight_padding_slots_do_not_affect_pooling()
    test_trust_weights_act_as_mean_pooling_weights()
    print("ok")
