"""CPU numeric/source-contract tests; no SUMO or truth files."""
import torch

from wms.physics.config import PhysicsConfig
from wms.physics.lc2013 import (COOPERATIVE, KEEP_RIGHT, SPEED_GAIN, STRATEGIC,
                                NUMERICAL_EPS, brake_gap, lane_change, maximum_safe_stop, secure_gap)


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
    torch.testing.assert_close(maximum_safe_stop(gap + NUMERICAL_EPS, b, tau, .5), v)


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
        mem = lane_change(k, mask, p, memory=mem,
                          target_remaining_distance=torch.full((1, 1), 1e6))
        assert mem['next_lane'].item() == 0
    mem = lane_change(k, mask, p, memory=mem,
                      target_remaining_distance=torch.full((1, 1), 1e6))
    assert mem['next_lane'].item() == 1
    assert mem['lc2013_reason'].item() == KEEP_RIGHT
    k[..., 1], k[..., 2] = 0, 1
    mem = lane_change(k, mask, p, memory=mem)
    assert mem['lc2013_keep_right'].item() == 0


def test_mature_keep_right_waits_when_target_lane_becomes_slower():
    k, mask = scene([(-200, 0, 5), (-100, 1, 5)])
    p = PhysicsConfig()
    mem = lane_change(k, mask, p)
    mem['lc2013_keep_right'][0, 0] = -40_000_000
    out = lane_change(k, mask, p, memory=mem)
    assert out['lc2013_safe'][0, 0]  # safety alone must not permit the wish
    assert out['lc2013_this_speed'][0, 0] - 5 / 3.6 > out['lc2013_target_speed'][0, 0]
    assert out['lc2013_keep_right'][0, 0] == -40_000_000  # retained, not erased
    assert not out['lc2013_wish'][0, 0]
    assert out['next_lane'][0, 0] == 0
    mask[0, 1] = False
    resumed = lane_change(k, mask, p, memory=out)
    assert resumed['lc2013_reason'][0, 0] == KEEP_RIGHT
    assert resumed['next_lane'][0, 0] == 1


def test_mature_speed_gain_requires_sumo_numerical_epsilon_not_machine_epsilon():
    # The target's30m/s vs current anticipated29.99m/s yields gain~0.00033:
    # positive but smaller than SUMO's documented NUMERICAL_EPS0.001.
    k, mask = scene([(0, 1, 5), (37.426, 1, 30)])
    p = PhysicsConfig(speed_limit_mps=30)
    params = {'time_headway': torch.ones((1, 2))}
    mem = lane_change(k, mask, p, driver_params=params)
    mem['lc2013_left'][0, 0] = 3_000_000
    out = lane_change(k, mask, p, memory=mem, driver_params=params)
    gain = (out['lc2013_target_speed'] - out['lc2013_this_speed']) / out['lc2013_target_speed']
    assert 1e-6 < gain[0, 0] < NUMERICAL_EPS
    assert out['lc2013_safe'][0, 0]
    assert out['lc2013_left'][0, 0] > 2_000_000
    assert out['next_lane'][0, 0] == 1


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


def test_uncongested_right_overtake_prevention_emits_advice_and_left_wish():
    k, mask = scene([(0, 1, 30), (100, 0, 20)])
    p = PhysicsConfig()
    out = lane_change(k, mask, p)
    assert out['lc2013_own_has_advice'][0, 0]
    assert out['lc2013_own_advice_accel'][0, 0] < 0
    assert out['lc2013_left'][0, 0] > 0
    allowed = lane_change(k, mask, p, allow_overtaking_right=True)
    assert not allowed['lc2013_own_has_advice'][0, 0]
    assert allowed['lc2013_left'][0, 0] == 0
    k[0, 0, 4] = 60 / 3.6 - .01
    assert not lane_change(k, mask, p)['lc2013_own_has_advice'][0, 0]


def test_signed_slow_current_leader_cap_suppresses_a_mature_keep_right_wish():
    # Valid observer-initialized state: inferred desired speeds29/28 while
    # observed speeds30/31. Source1752 gives(95-2.5)/(29-31)=-46.25s.
    k, mask = scene([(-500, 0, 30), (-400, 0, 31)])
    p = PhysicsConfig()
    params = {'v0': torch.tensor([[29., 28.]])}
    mem = lane_change(k, mask, p, driver_params=params)
    mem['lc2013_keep_right'][0, 0] = -20_000_001
    out = lane_change(k, mask, p, memory=mem, driver_params=params)
    assert out['lc2013_keep_signed_cap'][0, 0]
    assert out['lc2013_keep_full_seconds'][0, 0] == -46.25
    assert out['lc2013_keep_right'][0, 0] > -20_000_001
    assert out['next_lane'][0, 0] == 0


def test_faster_target_leader_does_not_overwrite_the_later_current_leader_gap_cap():
    k, mask = scene([(-500, 0, 25), (-400, 0, 20), (-470, 1, 35)])
    p = PhysicsConfig()
    out = lane_change(k, mask, p, driver_params={'v0': torch.tensor([[30., 25., 35.]])})
    # Target leader35>vMax30: its nearby gap is ignored by1742, then the own
    # slow leader caps full-speed time to(95-2.5)/(30-20)=9.25s.
    assert out['lc2013_keep_full_seconds'][0, 0] == 9.25


def test_zero_closing_speed_in_current_leader_cap_keeps_wishes_finite():
    k, mask = scene([(-500, 0, 25), (-400, 0, 30)])
    out = lane_change(k, mask, PhysicsConfig(), driver_params={'v0': torch.tensor([[30., 25.]])})
    assert torch.isfinite(out['lc2013_keep_full_seconds']).all()
    assert torch.isfinite(out['lc2013_keep_right']).all()


def test_float32_ramp_end_representation_does_not_disable_a_strategic_merge():
    p = PhysicsConfig()
    kin, mask = scene([(p.merge_zone_s1_m, 2, 0)])
    reference = lane_change(kin, mask, p)
    assert reference['next_lane'].item() == 1
    for observed_kin in (kin.float(), kin.float().double()):
        observed = lane_change(observed_kin, mask, p)
        assert observed['next_lane'].item() == 1
        assert observed['lc2013_reason'].item() == STRATEGIC
        assert observed['lc2013_position'].item() == p.merge_zone_s1_m


def test_float32_one_ulp_beyond_ramp_end_is_not_a_new_merge_window():
    p = PhysicsConfig()
    kin, mask = scene([(p.merge_zone_s1_m, 2, 0)])
    kin = kin.float()
    kin[..., 0] = torch.nextafter(kin[..., 0], torch.full_like(kin[..., 0], float('inf')))
    out = lane_change(kin, mask, p)
    assert out['next_lane'].item() == 2
    assert not out['lc2013_wish'].item()
    exact, mask = scene([(p.merge_zone_s1_m, 2, 0)])
    exact[..., 0] = torch.nextafter(exact[..., 0], torch.full_like(exact[..., 0], float('inf')))
    assert lane_change(exact, mask, p)['next_lane'].item() == 2


def test_cooperative_early_return_does_not_also_accumulate_a_speed_gain_wish():
    # Ramp request makes its target follower clear left. The follower also has
    # a slow current leader, but source1669 returns before1820's wish update.
    kin,mask = scene([(40,2,10),(31,1,10),(60,1,5)])
    out = lane_change(kin,mask,PhysicsConfig())
    assert out['lc2013_reason'][0,1] == COOPERATIVE
    assert out['lc2013_cooperative_return'][0,1]
    assert out['lc2013_target_speed'][0,1] > out['lc2013_this_speed'][0,1]
    assert out['lc2013_left'][0,1] == 0


def test_source_zero_cost_cooperation_at_zero_parameter_is_not_an_extra_disable_gate():
    kin,mask = scene([(40,2,10),(31,1,10),(60,1,5)])
    out = lane_change(kin,mask,PhysicsConfig(),driver_params={'lcCooperative':torch.zeros(1,3)})
    assert out['lc2013_reason'][0,1] == COOPERATIVE
    assert out['next_lane'][0,1] == 0
    assert out['lc2013_left'][0,1] == 0


def test_mature_speed_gain_urgency_requests_source_follower_deceleration():
    kin,mask = scene([(0,1,10),(30,1,5),(-10,0,15)])
    p = PhysicsConfig()
    memory = lane_change(kin,mask,p)
    memory['lc2013_left'][0,0] = 51 * 10_000_000
    out = lane_change(kin,mask,p,memory=memory)
    assert out['lc2013_reason'][0,0] == SPEED_GAIN
    assert out['lc2013_blocked_follower'][0,0]
    assert out['lc2013_speed_gain_urgent'][0,0]
    assert out['lc2013_urgent'][0,0]
    assert out['lc2013_has_advice'][0,2]
    assert out['lc2013_advice_accel'][0,2] == -p.idm()[1]


def test_speed_gain_below_source_urgency_threshold_does_not_request_help():
    kin,mask = scene([(0,1,10),(30,1,5),(-10,0,15)])
    p = PhysicsConfig()
    memory = lane_change(kin,mask,p)
    memory['lc2013_left'][0,0] = 49 * 10_000_000
    out = lane_change(kin,mask,p,memory=memory)
    assert out['lc2013_reason'][0,0] == SPEED_GAIN
    assert not out['lc2013_speed_gain_urgent'][0,0]
    assert not out['lc2013_has_advice'][0,2]
