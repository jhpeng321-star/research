"""P4 sampler: deterministic draws + insertion into the P3 world state."""
from __future__ import annotations

import numpy as np
import torch

from arrival import ArrivalConfig, ArrivalSampler
from synth_arrival import make_world


def _head_out(batch=3):
    return {
        "count_mu": torch.full((batch, 2), 1.5),
        "count_log_alpha": torch.full((batch, 2), -1.0),
        "speed_mu": torch.full((batch, 2), 12.0),
        "speed_log_sigma": torch.full((batch, 2), -0.5),
    }


def test_sample_counts_deterministic_clipped_int64():
    s = ArrivalSampler(ArrivalConfig())
    a = s.sample_counts(_head_out(), rng=np.random.default_rng(7))
    b = s.sample_counts(_head_out(), rng=np.random.default_rng(7))
    assert a.shape == (3, 2) and a.dtype == torch.int64
    assert torch.equal(a, b)
    assert int(a.min()) >= 0
    assert int(a.max()) <= ArrivalConfig().max_insert_per_entrance


def test_sample_speeds_nonnegative_deterministic():
    s = ArrivalSampler(ArrivalConfig())
    a = s.sample_speeds(_head_out(), rng=np.random.default_rng(11))
    b = s.sample_speeds(_head_out(), rng=np.random.default_rng(11))
    assert a.shape == (3, 2)
    assert torch.allclose(a, b)
    assert float(a.min()) >= 0.0


def test_insert_places_generic_hdv_behind_lane_occupancy():
    world = make_world(batch=2, slots=6, seed=0, free_slots=3)
    before = make_world(batch=2, slots=6, seed=0, free_slots=3)
    counts = torch.tensor([[1, 1], [2, 2]], dtype=torch.int64)
    speeds = torch.full((2, 2), 9.0)
    sampler = ArrivalSampler(ArrivalConfig())
    new, inserted = sampler.insert(world, counts, speeds)

    assert inserted.shape == (2, 2) and inserted.dtype == torch.int64
    assert (inserted <= counts).all()
    assert int(inserted.sum()) <= 3 * 2        # free slots available
    # input world was not mutated.
    assert torch.equal(world.present, before.present)
    assert torch.equal(world.s_m, before.s_m)

    for b in range(2):
        # entrance 0 -> lane 1, entrance 1 -> lane 2; every inserted vehicle
        # enters behind the original same-lane occupancy.
        for slot in range(6):
            if new.present[b, slot] and not world.present[b, slot]:
                assert new.is_cav[b, slot].item() is False
                assert int(new.wall_group[b, slot]) == -1
                assert float(new.length_m[b, slot]) == 5.0
                entry_lane = int(new.lane[b, slot])
                assert entry_lane in (1, 2)
                occ = world.present[b] & (world.lane[b] == entry_lane)
                assert float(new.s_m[b, slot]) < float(world.s_m[b][occ].min())


def test_insert_is_capped_per_entrance():
    world = make_world(batch=1, slots=6, seed=1, free_slots=6)
    cfg = ArrivalConfig(max_insert_per_entrance=1)
    sampler = ArrivalSampler(cfg)
    new, inserted = sampler.insert(
        world, torch.tensor([[5, 5]]), torch.full((1, 2), 8.0))
    assert int(inserted.sum()) == 2
