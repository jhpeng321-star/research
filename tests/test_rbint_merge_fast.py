"""Dense reference vs ego-query parity, including hard geometric degeneracies."""
from dataclasses import replace

import pytest
import torch

from wms.physics.config import PhysicsConfig
from wms.physics.junction_merge import apply_merge, merge_geometry
from wms.physics.junction_query import merge_geometry_at


def scene(dtype, device='cpu'):
    rng = torch.Generator().manual_seed(4015)
    k = torch.zeros(3, 31, 7, dtype=dtype)
    k[..., 0] = torch.rand(3, 31, generator=rng, dtype=dtype)*120-20
    lane = torch.randint(3, (3, 31), generator=rng)
    k[..., 1:4] = torch.nn.functional.one_hot(lane, 3)
    k[..., 4] = torch.rand(3, 31, generator=rng, dtype=dtype)*35
    k[..., 6] = torch.rand(3, 31, generator=rng, dtype=dtype)*5+3
    mask = torch.rand(3, 31, generator=rng) > .25
    p = {name: torch.rand(3, 31, generator=rng, dtype=dtype)*scale+base
         for name, base, scale in (('v0', 17., 25.), ('accel', .5, 3.),
                                  ('decel', 1., 6.), ('time_headway', .4, 2.),
                                  ('min_gap', 1., 3.))}
    # Equal-front target cars, body overlap, absent rows and an empty target lane.
    k[:, :3, 0] = 20
    k[:, :3, 1:4] = torch.tensor([[0., 0., 1.], [0., 1., 0.], [0., 1., 0.]])
    mask[:, :3] = True
    k[2, :, 1:4] = torch.tensor([0., 0., 1.])
    return k.to(device), mask.to(device), {n:v.to(device) for n,v in p.items()}


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_query_preserves_every_dense_geometry_field_and_tie(dtype):
    k, mask, p = scene(dtype)
    physics = PhysicsConfig(merge_model='sumo')
    dense = merge_geometry(k, mask, physics, p)
    for indices in (torch.tensor([[0], [7], [12]]), torch.arange(31)[None].expand(3, -1)):
        fast = merge_geometry_at(k, mask, indices, physics, p)
        assert fast.keys() == dense.keys()
        for key in dense:
            assert torch.equal(fast[key], dense[key].gather(-1, indices)), key


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_sequential_reservations_and_persistent_lookahead_are_bit_exact(dtype):
    k, mask, p = scene(dtype)
    physics = PhysicsConfig(merge_model='sumo')
    lane = k[..., 1:4].argmax(-1)
    deciders = mask.clone()
    deciders[1, 2:4] = False
    proposals = torch.where(lane == 0, torch.ones_like(lane), lane)
    look = k[..., 4] + 5
    dense = apply_merge(k, mask, proposals, deciders, physics, p, lookahead_speed=look)
    fast = apply_merge(k, mask, proposals, deciders,
                       replace(physics, sumo_merge_query_fast=True), p, lookahead_speed=look)
    assert torch.equal(dense, fast)


def test_multiple_accepted_merges_preserve_sequential_occupancy_reservations():
    kin = torch.zeros(2, 4, 7)
    kin[..., 6] = 5
    kin[:, :3, 3] = 1
    kin[:, 3, 1] = 1
    kin[..., 0] = torch.tensor([[74., 73., 50., -500.], [74., 60., 50., -500.]])
    mask = torch.ones(2, 4, dtype=torch.bool)
    lane = kin[..., 1:4].argmax(-1)
    physics = PhysicsConfig(merge_model='sumo')
    look = torch.full((2, 4), 10.)
    dense = apply_merge(kin, mask, lane, mask, physics, lookahead_speed=look)
    fast = apply_merge(kin, mask, lane, mask, replace(physics, sumo_merge_query_fast=True),
                       lookahead_speed=look)
    assert torch.equal(dense, fast)
    assert (dense == 1).sum(-1).tolist() == [2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA parity requires a free GPU')
def test_cuda_query_and_reservations_match_dense():
    k, mask, p = scene(torch.float32, 'cuda')
    physics = PhysicsConfig(merge_model='sumo')
    indices = torch.tensor([[0], [7], [12]], device='cuda')
    dense = merge_geometry(k, mask, physics, p)
    fast = merge_geometry_at(k, mask, indices, physics, p)
    assert all(torch.equal(fast[n], dense[n].gather(-1, indices)) for n in dense)
    lane = k[..., 1:4].argmax(-1)
    a = apply_merge(k, mask, lane, mask, physics, p)
    b = apply_merge(k, mask, lane, mask, replace(physics, sumo_merge_query_fast=True), p)
    assert torch.equal(a, b)
