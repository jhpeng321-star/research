import torch
import pytest

from wms.physics.config import PhysicsConfig
from wms.physics.lc2013 import KEEP_RIGHT, lane_change
from wms.physics.lc2013_geometry import corridor_context


def scene(rows):
    k = torch.zeros(1, len(rows), 7, dtype=torch.float64)
    for i, (s, lane, speed) in enumerate(rows):
        k[0, i, 0], k[0, i, 1 + lane] = s, 1
        k[0, i, 4], k[0, i, 6] = speed, 5
    return k, torch.ones(1, len(rows), dtype=torch.bool)


def test_corridor_lengths_are_edge_local_not_remaining_or_infinite():
    s = torch.tensor([[-100., -100., 30., 30., 30., 100., -100.]], dtype=torch.float64)
    lane = torch.tensor([[0, 1, 0, 1, 2, 0, 2]])
    context = corridor_context(s, lane, PhysicsConfig())
    torch.testing.assert_close(context['position'],
        torch.tensor([[2545.67, 2544.22, 30., 30., 30., 24.56, 278.66]], dtype=s.dtype))
    torch.testing.assert_close(context['current_length'],
        torch.tensor([[3197.55, 3197.55, 553.33, 553.33, 75.44, 477.89, 454.1]], dtype=s.dtype))


def test_finite_continuation_slows_keep_right_accumulation():
    # Helper's hand-computable vector3: ballistic dt1, vMax30, v20,
    # continuation1000, b4.5, tau1, acceptance140, brakeGap130:
    # delta=floor(2e7*(29/140)/5)=828571; first request at step25.
    p = PhysicsConfig(dt_s=1, speed_limit_mps=30)
    k, mask = scene([(0, 0, 20)])
    params = {'decel': torch.tensor([[4.5]]), 'time_headway': torch.tensor([[1.]])}
    mem = None
    for step in range(1, 26):
        mem = lane_change(k, mask, p, memory=mem, driver_params=params,
            remaining_distance=torch.tensor([[1000.]]),
            target_remaining_distance=torch.tensor([[1000.]]),
            current_lane_position=torch.tensor([[0.]]), ballistic=True)
        assert mem['lc2013_keep_right'].item() == -step * 828571
        assert mem['next_lane'].item() == (0 if step < 25 else 1)
    assert mem['lc2013_reason'].item() == KEEP_RIGHT


def test_equally_good_through_lanes_do_not_invent_strategic_stay_or_urgency():
    k, mask = scene([(500, 0, 25)])
    p = PhysicsConfig()
    mem = lane_change(k, mask, p)
    mem['lc2013_keep_right'][:] = -40_000_000
    out = lane_change(k, mask, p, memory=mem)
    assert out['lc2013_reason'].item() == KEEP_RIGHT
    assert out['next_lane'].item() == 1


def test_length_and_remaining_distance_have_separate_source_uses():
    # Near the exit, finite LENGTH still permits speed gain's 20s test when
    # v10 and length477.89, though remaining distance is only53.33.
    k, mask = scene([(500, 1, 10), (530, 1, 5)])
    out = lane_change(k, mask, PhysicsConfig())
    assert out['lc2013_target_length'][0, 0].item() == pytest.approx(477.89)
    assert out['lc2013_position'][0, 0].item() == pytest.approx(424.56)
    assert out['next_lane'][0, 0] == 0
