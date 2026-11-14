"""P4 interface: head shapes, counter causality and tensor contract."""
from __future__ import annotations

import pytest
import torch

from arrival import ArrivalConfig, IdeaAArrivalHead, past_bin_mask


def test_arrival_head_shapes_and_ranges():
    head = IdeaAArrivalHead(ArrivalConfig())
    head.eval()
    counter = torch.rand(3, 2, 30)
    out = head(counter)
    for key in ("count_mu", "count_log_alpha", "speed_mu", "speed_log_sigma"):
        assert out[key].shape == (3, 2), key
        assert torch.isfinite(out[key]).all(), key
    assert (out["count_mu"] >= 0).all()
    assert (out["speed_mu"] > 1.0).all()


def test_config_rejects_other_entrance_counts():
    with pytest.raises(ValueError):
        ArrivalConfig(n_entrances=3)


def test_counter_causality_only_observed_bins_matter():
    head = IdeaAArrivalHead(ArrivalConfig())
    head.eval()
    g = torch.Generator().manual_seed(0)
    c1 = torch.rand(2, 2, 30, generator=g)
    c2 = c1.clone()
    c2[..., 5:] = torch.rand(2, 2, 25, generator=g) * 100.0

    observed = past_bin_mask(5, batch=2)
    o1 = head(c1, valid_bins=observed)
    o2 = head(c2, valid_bins=observed)
    for key in ("count_mu", "count_log_alpha", "speed_mu", "speed_log_sigma"):
        assert torch.equal(o1[key], o2[key]), key

    # the observed (past) bins DO drive the output.
    c3 = c1.clone()
    c3[..., :5] += 10.0
    o3 = head(c3, valid_bins=observed)
    assert not torch.equal(o1["count_mu"], o3["count_mu"])


def test_forward_rejects_wrong_counter_shape():
    head = IdeaAArrivalHead(ArrivalConfig())
    with pytest.raises(ValueError):
        head(torch.zeros(2, 2, 10))


def test_parameter_count_reported(capsys):
    full = IdeaAArrivalHead(ArrivalConfig())
    small = IdeaAArrivalHead(ArrivalConfig(d_model=64))
    n_full = int(sum(p.numel() for p in full.parameters()))
    n_small = int(sum(p.numel() for p in small.parameters()))
    with capsys.disabled():
        print("\n[arrival] parameter_count head=%d small=%d"
              % (n_full, n_small))
    assert 0 < n_small < n_full
