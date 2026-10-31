from dataclasses import replace

import pytest
import torch

from wms.model.lc2013_adapter import resolve_shifted_lanes
from wms.physics.config import PhysicsConfig
from wms.physics.lane_conflict import resolve_lane_conflicts
from wms.physics.lane_conflict_fast import resolve_lane_conflicts_fast


def scenes(device='cpu'):
    rng = torch.Generator().manual_seed(95)
    s = torch.randint(-30, 31, (7, 31), generator=rng).float()
    length = torch.randint(3, 8, s.shape, generator=rng).float()
    lane = torch.randint(0, 3, s.shape, generator=rng)
    proposed = torch.randint(0, 3, s.shape, generator=rng)
    present = torch.rand(s.shape, generator=rng) > .3
    movable = torch.rand(s.shape, generator=rng) > .4
    present[0] = False
    movable[1] = False
    proposed[2] = lane[2]
    return [value.to(device) for value in (s, length, lane, proposed, present, movable)]


@pytest.mark.parametrize('margin', [0., .2])
def test_exact_arbitration_with_ties_absence_fixed_cavs_and_conflicting_proposals(margin):
    values = scenes()
    assert torch.equal(resolve_lane_conflicts(*values, margin),
                       resolve_lane_conflicts_fast(*values, margin))


def test_shifted_coordinate_fixed_point_matches_reference_complete_assignment():
    s, length, lane, proposed, present, movable = scenes()
    s -= 1000
    physics = PhysicsConfig(merge_model='sumo')
    dense = resolve_shifted_lanes(s, length, lane, proposed, present, movable, physics)
    fast = resolve_shifted_lanes(s, length, lane, proposed, present, movable,
                                 replace(physics, sumo_lane_conflict_fast=True))
    assert all(torch.equal(a, b) for a,b in zip(dense, fast))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA parity requires GPU capacity')
def test_cuda_candidate_filter_has_identical_stable_reservations():
    values = scenes('cuda')
    assert torch.equal(resolve_lane_conflicts(*values, 0.),
                       resolve_lane_conflicts_fast(*values, 0.))
