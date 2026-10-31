"""CPU numeric/source-contract tests; no SUMO or truth files."""
import torch

from wms.physics.config import PhysicsConfig
from wms.physics.lc2013 import (COOPERATIVE, KEEP_RIGHT, SPEED_GAIN, STRATEGIC,
                                brake_gap, lane_change, maximum_safe_stop, secure_gap)


def scene(rows):
    k = torch.zeros(1, len(rows), 7, dtype=torch.float64)
    for i, (s, lane, v) in enumerate(rows):
        k[0, i, 0], k[0, i, 1 + lane] = s, 1
        k[0, i, 4], k[0, i, 6] = v, 5
    return k, torch.ones(1, len(rows), dtype=torch.bool)


def test_idm_secure_gap_numeric_and_opening():
    a = torch.tensor(2.6)
    b = torch.tensor(4.5)
    tau = torch.tensor(1.)
    assert secure_gap(torch.tensor(20.), torch.tensor(20.), a, b, tau) == 20
    expected = 20 + 200 / (2 * (2.6 * 4.5)**0.5)
    assert abs(secure_gap(torch.tensor(20.), torch.tensor(10.), a, b, tau) - expected) < 1e-5
    assert secure_gap(torch.tensor(5.), torch.tensor(30.), a, b, tau) == 0


def test_euler_stop_inverts_discrete_braking():
    v = torch.tensor([0., 1., 10., 25.], dtype=torch.float64)
    b, tau = torch.full_like(v, 4.5), torch.full_like(v, 1.)
    gap = brake_gap(v, b, .5, headway=tau)
    torch.testing.assert_close(maximum_safe_stop(gap + 1e-6, b, tau, .5), v)


def test_required_ramp_strategic_and_geometry():
    p = PhysicsConfig()
    k, mask = scene([(p.merge_zone_s1_m - 30, 2, 12)])
    out = lane_change(k, mask, p)
    assert out['next_lane'].item() == 1
    assert out['lc2013_reason'].item() == STRATEGIC
    k[..., 0] = p.merge_zone_s0_m - 1
    assert lane_change(k, mask, p)['next_lane'].item() == 2


def test_both_gap_checks_and_body_overlap():
    p = PhysicsConfig()
    for rows in ([(40, 2, 10), (49, 1, 10)],
                 [(40, 2, 10), (31, 1, 10)],
                 [(40, 2, 0), (40, 1, 0)]):
        k, mask = scene(rows)
        out = lane_change(k, mask, p)
        assert out['lc2013_wish'][0, 0]
        assert not out['lc2013_safe'][0, 0]
        assert out['next_lane'][0, 0] == 2


def test_assertive_scales_secure_gap_but_not_min_gap():
    k, mask = scene([(40, 2, 10), (55, 1, 10)])
    p = PhysicsConfig()
    assert not lane_change(k, mask, p)['lc2013_safe'][0, 0]
    assert lane_change(k, mask, p, driver_params={
        'lcAssertive': torch.full((1, 2), 2.)})['lc2013_safe'][0, 0]


def test_speed_gain_accumulates_then_changes_left():
    p = PhysicsConfig()
    k, mask = scene([(0, 1, 15), (30, 1, 5)])
    mem = None
    out = lane_change(k, mask, p, memory=mem)
    assert out['lc2013_left'][0, 0] > 0
    for _ in range(10):
        out = lane_change(k, mask, p, memory=out)
        if out['next_lane'][0, 0] == 0:
            break
    assert out['next_lane'][0, 0] == 0
    assert out['lc2013_reason'][0, 0] == SPEED_GAIN


def test_keep_right_source_five_second_accumulator_and_reset():
    k, mask = scene([(0, 0, 25)])
    p = PhysicsConfig()
    mem = None
    for _ in range(10):
        mem = lane_change(k, mask, p, memory=mem)
        assert mem['next_lane'].item() == 0
    mem = lane_change(k, mask, p, memory=mem)
    assert mem['next_lane'].item() == 1
    assert mem['lc2013_reason'].item() == KEEP_RIGHT
    k[..., 1], k[..., 2] = 0, 1
    mem = lane_change(k, mask, p, memory=mem)
    assert mem['lc2013_keep_right'].item() == 0


def test_blocked_wishes_persist_and_identity_or_absence_reset():
    k, mask = scene([(0, 1, 15), (30, 1, 5), (0, 0, 15)])
    ids = torch.tensor([[1, 2, 3]])
    p = PhysicsConfig()
    out = lane_change(k, mask, p, identity=ids)
    out2 = lane_change(k, mask, p, memory=out, identity=ids)
    assert out2['lc2013_left'][0, 0] > out['lc2013_left'][0, 0]
    ids[0, 0] = 4
    reset = lane_change(k, mask, p, memory=out2, identity=ids)
    assert reset['lc2013_left'][0, 0] == out['lc2013_left'][0, 0]
    mask[0, 0] = False
    assert lane_change(k, mask, p, memory=out2)['lc2013_left'][0, 0] == 0


def test_cooperative_move_clears_urgent_ramp_follower():
    k, mask = scene([(40, 2, 10), (31, 1, 10)])
    out = lane_change(k, mask, PhysicsConfig())
    assert out['next_lane'][0, 1] == 0
    assert out['lc2013_reason'][0, 1] == COOPERATIVE


def test_no_spurious_left_wish_and_disabled_behaviours():
    k, mask = scene([(0, 1, 25)])
    out = lane_change(k, mask, PhysicsConfig())
    assert out['next_lane'].item() == 1
    k, mask = scene([(0, 0, 25)])
    params = {'lcSpeedGain': torch.zeros((1, 1))}
    for _ in range(20):
        out = lane_change(k, mask, PhysicsConfig(), driver_params=params, memory=out)
    assert out['next_lane'].item() == 0


def test_invalid_follower_parameters_fail_closed():
    k, mask = scene([(40, 2, 10), (-40, 1, 10)])
    out = lane_change(k, mask, PhysicsConfig(), driver_params={
        'accel': torch.tensor([[2.6, float('nan')]])})
    assert not out['lc2013_safe'][0, 0]


def test_connector_is_not_a_lane_zero_obstacle_or_decider():
    k, mask = scene([(40, 2, 0), (40, 0, 0)])
    k[0, 1, 1:4] = 0
    out = lane_change(k, mask, PhysicsConfig())
    assert out['next_lane'][0, 0] == 1
    assert out['next_lane'][0, 1] == -1
    assert not out['lc2013_present'][0, 1]
