from dataclasses import asdict, replace

import pytest

from traffic_shepherd.gentle_wall import GentleProfile, GentleWallPolicy, resolve_command, ramped_braking_distance, gentle_envelope
from traffic_shepherd.formation import make_proposal, assess_formation, FormationProfile
from traffic_shepherd.paired_experiments import PolicyInput
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import SensorFrame, NeighborMeasurement


def packets(now=0., speed=24., acceleration=0., **kwargs):
    return tuple(CAVReport(SensorFrame(v, now, speed, acceleration, 30., i,
        distance_to_merge_m=2000., front=None, rear=None, front_coverage_m=100.,
        rear_coverage_m=100., lane_id=f'L{i}', position_m=1000., **kwargs), now, 'damping')
        for i, v in enumerate(('a', 'b')))


def policy():
    return GentleWallPolicy(dict(arm='wall', decision_time_s=0, policy_spec=dict(
        member_ids=['a', 'b'], required_lanes=['L0', 'L1'], profile=asdict(GentleProfile()))),
        {'registration_sha256': 'synthetic_registered_treatment'})


def ctx(now=0., rows=None, numeric=None):
    return PolicyInput(now, packets(now) if rows is None else rows, (), numeric or {},
        frozenset() if numeric else frozenset(r.frame.vehicle_id for r in (packets(now) if rows is None else rows)))


def test_absolute_bound_and_slew_after_servo_not_a_residual():
    p, f = GentleProfile(), packets()[0].frame
    first, proof = resolve_command(f, -30., p)
    assert first == -.15 and proof['requested_acceleration_mps2'] == -30
    second, _ = resolve_command(replace(f, acceleration_mps2=first), -30., p, previous=first)
    assert second == -.3
    reverse, _ = resolve_command(replace(f, acceleration_mps2=second), 30., p, previous=second)
    assert reverse == -.15
    assert FormationProfile().acceleration_min_mps2 == -3.


def test_incompatible_measured_braking_or_speed_limit_returns_no_command():
    f = packets(acceleration=-2.)[0].frame
    assert resolve_command(f, -5., GentleProfile())[0] is None
    assert resolve_command(replace(f, acceleration_mps2=0., allowed_speed_mps=20.), 0., GentleProfile())[0] is None


def test_feasibility_accounts_for_gentle_deceleration_and_ramp():
    rows = packets(speed=28.)
    p = make_proposal(rows, now_s=0., required_lanes=('L0', 'L1'), member_ids=('a', 'b'), profile=GentleProfile())
    old = replace(p, profile=FormationProfile())
    assert assess_formation(p, rows, now_s=0.).estimated_assembly_s > assess_formation(old, rows, now_s=0.).estimated_assembly_s
    distance, seconds = ramped_braking_distance(28., 24., 0., p.profile)
    assert seconds > 4/.3 and distance > (28**2-24**2)/(.6)


def test_missing_reports_and_changed_lane_abort_terminally():
    for mode in ('missing', 'lane'):
        obj = policy()
        assert obj(ctx()).wall_commands
        rows = packets(1.)
        rows = rows[:1] if mode == 'missing' else (replace(rows[0], frame=replace(rows[0].frame, lane_id='other')), rows[1])
        out = obj(ctx(1., rows))
        assert out.phase == 'abort' and not out.wall_commands
        assert not obj(ctx(2.)).wall_commands


def test_actual_projected_gentle_command_refuses_protective_geometry():
    obj = policy()
    rows = tuple(replace(r, frame=replace(r.frame, front=NeighborMeasurement(16., -5.))) for r in packets())
    out = obj(ctx(rows=rows))
    assert out.phase == 'refused' and not out.wall_commands
    assert 'clearance' in out.reason


def test_front_closure_accounts_for_braking_after_the_next_tick():
    obj = policy()
    rows = tuple(replace(r, frame=replace(r.frame, front=NeighborMeasurement(50., -6.))) for r in packets())
    # Current TTC is >8 s; gentle braking distance still does not fit.
    out = obj(ctx(rows=rows))
    assert out.phase == 'refused' and not out.wall_commands
    assert 'ramped_front_braking_infeasible' in out.reason


def test_saturation_persists_through_phases_and_native_transition_is_explicit():
    obj = policy()
    speed, accel = 24., 0.
    seen = set()
    for now in range(90):
        rows = packets(float(now), speed=speed, acceleration=accel)
        out = obj(ctx(float(now), rows))
        seen.add(out.phase)
        if out.wall_commands:
            new = out.wall_commands['a']
            assert -.3 <= new <= 1.5 and abs(new-accel) <= .15+1e-9
            speed += new
            accel = new
        else:
            assert out.phase == 'release'
            assert not obj(ctx(float(now+1), packets(float(now+1), speed=speed))).wall_commands
            break
    assert {'assembly', 'hold', 'release'} <= seen
    assert out.reason == 'gentle_transition_aligned_return_to_identical_damping'


def test_native_release_with_nonzero_wall_acceleration_requires_coast_transition():
    obj = policy()
    obj(ctx())
    obj.ready_at, obj.taper_at = 0., 1.
    obj.previous_commands = {'a': .15, 'b': .15}
    out = obj(ctx(1., packets(1., speed=22., acceleration=.15)))
    assert out.phase == 'taper' and dict(out.wall_commands) == {'a': 0., 'b': 0.}
    assert out.details['dissolution']['mode'] == 'wall_coast_transition_before_native_handover'
    out = obj(ctx(2., packets(2., speed=22., acceleration=0.)))
    assert out.phase == 'release' and not out.wall_commands


def test_report_release_uses_ramp_feasibility_and_keeps_original_envelope_immutable():
    rows = packets()
    proposal = make_proposal(rows, now_s=0., required_lanes=('L0','L1'), member_ids=('a','b'), profile=GentleProfile())
    near = tuple(replace(r, frame=replace(r.frame, vehicle_id='down'+r.frame.vehicle_id,
        position_m=1177., speed_mps=3., distance_to_merge_m=1823.)) for r in rows)
    evidence = gentle_envelope(proposal, rows+near, now_s=0., previous_commands={})
    assert evidence['available'] and not evidence['gentle_stopping_feasible']
    assert evidence['comfortable_deceleration_mps2'] == .3
    assert 'target_speed_mps' not in evidence


def test_profile_cannot_tune_bound_or_ramp():
    with pytest.raises(ValueError):
        GentleProfile(acceleration_change_limit_mps3=.3)


def test_gentle_report_bridge_safety_refusal_and_lost_evidence():
    for downstream_speed in (0., 21.):
        obj = policy()
        obj(ctx())
        obj.ready_at, obj.taper_at = 0., 1.
        obj.previous_commands = {'a': 0., 'b': 0.}
        rows = packets(1., speed=22.)
        ahead = tuple(replace(r, frame=replace(r.frame, vehicle_id='down'+r.frame.vehicle_id,
            position_m=1177., speed_mps=downstream_speed, distance_to_merge_m=1823.)) for r in rows)
        out = obj(ctx(1., rows+ahead))
        if downstream_speed == 0.:
            assert out.phase == 'abort' and not out.wall_commands
            assert 'protective_braking_infeasible' in out.reason
        else:
            assert out.phase == 'taper' and set(out.wall_commands.values()) == {-.15}
            out = obj(ctx(2., packets(2., speed=21.85, acceleration=-.15)))
            assert out.phase == 'abort' and 'evidence_lost' in out.reason


def test_numeric_protective_handback_is_never_clipped_into_gentle_command():
    obj = policy()
    obj(ctx())
    obj.ready_at, obj.taper_at = 0., 1.
    out = obj(ctx(1., packets(1., speed=22.), numeric={'a': -2., 'b': -1.}))
    assert out.phase == 'release' and not out.wall_commands
    assert 'protective_numeric_handover' in out.reason
