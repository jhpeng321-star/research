"""W2 merge-pressure trigger inputs: observable-only packets, timestamps and missingness (CPU)."""
import math

import numpy as np
import pytest

from . import wall_pressure as wp
from .merge_features import ObservedVehicle, SensorReport

F = {name: i for i, name in enumerate(wp.PRESSURE_FEATURES)}


def veh(vid, s, lane, v=10., length=5.):
    return ObservedVehicle(vid, float(s), int(lane), float(v), float(length))


def report(t, source, rows=(), lanes=(0, 1, 2)):
    return SensorReport(float(t), source, tuple(lanes), tuple(rows))


def test_feature_names_are_observable_quantities_only():
    banned = ('truth', 'driver', 'vtype', 'param', 'tau', 'accel_max', 'decel', 'sigma', 'global',
              'rsu', 'penetration', 'demand', 'hidden')
    assert not [n for n in wp.PRESSURE_FEATURES if any(b in n for b in banned)]
    assert wp.PRESSURE_DIM == len(wp.SCALE)
    for name in ('ramp', 'main'):
        assert f'{name}_missing' in F and f'{name}_report_age_s' in F
    assert {'position_missing', 'last_wall_missing', 'main_speed_missing'} <= set(F)


def test_no_own_packet_is_all_missing_without_partner_fallback():
    partner = report(99, veh('p', -40, 1), [veh('q', -30, 2, v=1.)])
    out = wp.pressure_features(None, 'ego', {'p': partner}, now_s=100, partner_ids=['p'])
    assert np.array_equal(out, wp.missing_block())
    # A packet that is not the ego's own (wrong id) is missing too.
    out = wp.pressure_features(report(100, veh('x', -50, 1)), 'ego', now_s=100)
    assert np.array_equal(out, wp.missing_block())


def test_far_upstream_ego_sees_position_but_no_merge_region():
    ego = veh('ego', -420, 1)
    out = wp.pressure_features(report(100, ego, [veh('h', -400, 2)]), 'ego', now_s=100)
    assert out[F['position_missing']] == 0
    assert out[F['dist_to_merge_zone_m']] == pytest.approx(wp.ZONE_S0_M+420)
    assert out[F['dist_to_merge_m']] == pytest.approx(420)
    assert out[F['in_merge_zone']] == 0
    # Its sensing [-520, -320] never reaches the zone [-100, 75.44]: missing, never zero-as-empty.
    assert out[F['ramp_missing']] == 1 and out[F['main_missing']] == 1
    assert out[F['ramp_queue_veh']] == 0 and out[F['main_occupancy']] == 0


def test_own_sensing_counts_queue_arrivals_occupancy_and_speed():
    ego = veh('ego', -60, 1, v=8.)
    rows = [veh('r1', -20, 2, v=0.), veh('r2', 10, 2, v=2.), veh('r3', -80, 2, v=12.),
            veh('m1', -50, 0, v=6., length=10.), veh('m2', -90, 1, v=4.),
            veh('far', -300, 0, v=30.)]                       # outside own sensing: ignored
    out = wp.pressure_features(report(100, ego, rows), 'ego', now_s=100)
    assert out[F['ramp_missing']] == 0 and out[F['ramp_report_age_s']] == 0
    assert out[F['ramp_queue_veh']] == 2                     # r1, r2 (v < 5 m/s, s <= merge end)
    assert out[F['ramp_arrival_count']] == 2                 # r1, r3 upstream of merge start
    sensed = 2*(40.-wp.ZONE_S0_M)                            # [-100, 40] on lanes 0 and 1
    assert out[F['main_occupancy']] == pytest.approx((10.+5.+5.)/sensed)   # m1, m2, ego
    assert out[F['main_mean_speed_mps']] == pytest.approx((6.+4.+8.)/3)
    assert out[F['main_coverage_frac']] == pytest.approx(sensed/(2*(wp.MERGE_S1_M-wp.ZONE_S0_M)))
    assert out[F['in_merge_zone']] == 1
    # Uncovered lane is missing, not empty: drop lane 2 from the packet's declared coverage.
    out = wp.pressure_features(report(100, ego, [r for r in rows if r.lane != 2], lanes=(0, 1)),
                               'ego', now_s=100)
    assert out[F['ramp_missing']] == 1 and out[F['ramp_queue_veh']] == 0


def test_partner_packets_t_minus_one_selected_in_range_only_with_age():
    ego = veh('ego', -150, 1)
    partner = report(99, veh('p', -60, 0), [veh('r1', 0, 2, v=0.)])
    out = wp.pressure_features(report(100, ego), 'ego', {'p': partner}, now_s=100, partner_ids=['p'])
    assert out[F['ramp_missing']] == 0 and out[F['ramp_queue_veh']] == 1
    assert out[F['ramp_report_age_s']] == 1                  # only the t-1 partner covers the ramp
    base = wp.pressure_features(report(100, ego), 'ego', now_s=100)
    for reports, ids in (({'p': partner}, []),                                   # not selected
                         ({'p': report(98, veh('p', -60, 0), [veh('r1', 0, 2, v=0.)])}, ['p']),  # stale
                         ({'p': report(100, veh('p', -60, 0), [veh('r1', 0, 2, v=0.)])}, ['p']),  # same step
                         ({'p': report(99, veh('p', 60, 0), [veh('r1', 0, 2, v=0.)])}, ['p'])):   # out of radio
        assert np.array_equal(wp.pressure_features(report(100, ego), 'ego', reports, now_s=100,
                                                   partner_ids=ids), base)


def test_wall_memory_is_own_commitments_with_missingness():
    memory = {}
    now = wp.update_memory(memory, [(('a', 'b'), 100, 140)], 120)
    assert now['a'] == wp.WallMemory(None, True) and 'c' not in now
    later = wp.update_memory(memory, [], 150)
    out = wp.pressure_features(None, 'a', now_s=150, memory=later['a'])
    assert out[F['time_since_last_wall_s']] == 10 and out[F['last_wall_missing']] == 0
    assert out[F['wall_active']] == 0
    active = wp.pressure_features(None, 'a', now_s=120, memory=now['a'])
    assert active[F['wall_active']] == 1 and active[F['last_wall_missing']] == 0
    never = wp.pressure_features(None, 'c', now_s=150, memory=None)
    assert never[F['last_wall_missing']] == 1 and never[F['time_since_last_wall_s']] == 0
    # A future-starting commitment is not yet memory.
    assert wp.update_memory({}, [(('d',), 200, 230)], 150) == {}


def test_scaled_block_is_finite_and_bounded_for_the_head():
    block = wp.missing_block()
    assert np.isfinite(wp.scaled(block)).all()
    with pytest.raises(ValueError):
        wp.scaled(np.full(wp.PRESSURE_DIM, math.nan))
    with pytest.raises(ValueError):
        wp.scaled(np.zeros(3))


def test_live_policy_blocks_ignore_unsensed_vehicles_and_unwhitelisted_fields():
    from .test_wall_w2_cpu import base_cfg, run_policy, scene
    from . import track2_backend as tb
    from .v2v_intent import IntentPolicy
    import torch
    cfg = dict(base_cfg(), wall_pressure_features=True)
    torch.manual_seed(7)
    p = tb.EpisodePolicy(IntentPolicy(), cfg, 'B')
    rows = scene()
    p.observe(rows, 300)
    first = {k: list(v) for k, v in p.pressure_blocks.items()}
    torch.manual_seed(7)
    q = tb.EpisodePolicy(IntentPolicy(), cfg, 'B')
    noisy = [dict(r, driver_tau=9.9, vtype='secret') for r in scene()]     # hidden fields: never read
    noisy.append(dict(vid='far', s_m=900., v_mps=3., lane=2, is_cav=False, length_m=5.))
    q.observe(noisy, 300)
    assert {k: list(v) for k, v in q.pressure_blocks.items()} == first
    assert set(first) == {'c1', 'c2', 'c3'}
    _, samples = run_policy(cfg, ticks=3)
    assert all(s['track2']['merge_pressure'] == s['context']['merge_pressure'] for s in samples)
