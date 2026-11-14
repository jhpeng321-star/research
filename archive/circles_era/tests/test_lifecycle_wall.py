from dataclasses import asdict, replace

import pytest

from traffic_shepherd.formation import make_proposal
from traffic_shepherd.lifecycle_wall import (
    LifecycleProfile, LifecycleWallPolicy, remaining_plan, VERSION, coast_terminal,
)
from traffic_shepherd.paired_experiments import PolicyInput
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import SensorFrame, NeighborMeasurement


def packets(now=0., speed=24., acceleration=0., distance=2000., position=1000.):
    return tuple(CAVReport(SensorFrame(v, now, speed, acceleration, 30., i,
        distance_to_merge_m=distance, front=None, rear=None, front_coverage_m=100.,
        rear_coverage_m=100., lane_id=f'L{i}', position_m=position), now, 'damping')
        for i, v in enumerate(('a', 'b')))


def policy(start=0):
    return LifecycleWallPolicy(dict(arm='wall', decision_time_s=start, policy_spec=dict(
        member_ids=['a', 'b'], required_lanes=['L0', 'L1'], profile=asdict(LifecycleProfile()))),
        {'registration_sha256': 'synthetic_registered_treatment'})


def context(now=0., rows=None, numeric=None):
    rows = packets(now) if rows is None else rows
    return PolicyInput(now, rows, (), {} if numeric is None else numeric,
        frozenset(r.frame.vehicle_id for r in rows if r.frame.timestamp_s == now)
        if numeric is None else frozenset())


def proposal(rows):
    return make_proposal(rows, now_s=0., required_lanes=('L0', 'L1'),
        member_ids=('a', 'b'), profile=LifecycleProfile())


def advance(rows, out):
    answer = []
    for r in rows:
        f = r.frame
        a = out.wall_commands[f.vehicle_id]
        travel = f.speed_mps+.5*a
        answer.append(replace(r, sent_at_s=f.timestamp_s+1, frame=replace(f,
            timestamp_s=f.timestamp_s+1, speed_mps=f.speed_mps+a, acceleration_mps2=a,
            position_m=f.position_m+travel, distance_to_merge_m=f.distance_to_merge_m-travel)))
    return tuple(answer)


@pytest.mark.parametrize('field,value', [('hold_s',40.), ('acceleration_min_mps2',-3.),
    ('acceleration_change_limit_mps3',.3), ('taper_s',20.), ('readiness_samples',True)])
def test_frozen_profile(field, value):
    with pytest.raises(ValueError):
        LifecycleProfile(**{field:value})


def test_positive_acceleration_is_integrated_before_braking_and_first_step_matches():
    rows = packets(acceleration=.9)
    p = proposal(rows)
    plan = remaining_plan(p, rows, now_s=0., previous_commands={})
    neutral = remaining_plan(p, packets(), now_s=0., previous_commands={})
    assert plan['feasible']
    assert plan['remaining_assembly_s'] > neutral['remaining_assembly_s']
    assert plan['planned_samples'][0]['commands_mps2']['a'] == pytest.approx(.75)
    obj = policy()
    out = obj(context(rows=rows))
    assert dict(out.wall_commands) == plan['planned_samples'][0]['commands_mps2']
    assert out.details['lifecycle_version'] == VERSION


def test_nonvacuous_complete_maneuver_and_measured_coast_handback():
    obj, rows, phases = policy(), packets(acceleration=.9), []
    for now in range(90):
        out = obj(context(now, rows))
        phases.append(out.phase)
        if not out.wall_commands:
            assert out.phase == 'release'
            assert coast_terminal([r.frame for r in rows], obj.previous_commands, obj.profile)
            assert out.details['actual_hold_elapsed_s'] == 20
            break
        plan = out.details['lifecycle_plan']['commitment']
        assert plan['planned_samples'][0]['commands_mps2'] == dict(out.wall_commands)
        for r in rows:
            a = out.wall_commands[r.frame.vehicle_id]
            assert -.3 <= a <= 1.5
            assert abs(a-r.frame.acceleration_mps2) <= .15+1e-9
        rows = advance(rows, out)
    else:
        pytest.fail('nonvacuous feasible maneuver must terminate')
    # An already coast-aligned end of HOLD needs no invented transition command.
    assert {'assembly','hold','release'} <= set(phases)
    later = tuple(replace(r, sent_at_s=now+1, frame=replace(r.frame,timestamp_s=now+1)) for r in rows)
    assert not obj(context(now+1,later)).wall_commands


def test_remaining_hold_does_not_recharge_completed_time():
    rows = packets(speed=22.)
    p = replace(proposal(rows), target_speed_mps=22.)
    early = remaining_plan(p, rows, now_s=0., previous_commands={'a':0.,'b':0.},
        readiness_count=3, ready_at_s=0.)
    late_rows = packets(19.,speed=22.)
    late = remaining_plan(p, late_rows, now_s=19., previous_commands={'a':0.,'b':0.},
        readiness_count=3, ready_at_s=0.)
    assert early['remaining_hold_s'] == 20
    assert late['remaining_hold_s'] == 1
    assert late['remaining_assembly_s'] == 0
    assert early['remaining_total_s']-late['remaining_total_s'] == 19


def downstream(rows, separation, speed):
    return tuple(replace(r, frame=replace(r.frame, vehicle_id='down'+r.frame.vehicle_id,
        position_m=r.frame.position_m+separation, speed_mps=speed,
        distance_to_merge_m=r.frame.distance_to_merge_m-separation)) for r in rows)


def test_map_and_reported_traffic_constraints_are_separate():
    rows = packets(distance=150.)
    plan = remaining_plan(proposal(rows), rows, now_s=0., previous_commands={})
    assert not plan['map_boundary_feasible']
    assert plan['reported_traffic_feasible']
    rows = packets()
    plan = remaining_plan(proposal(rows), rows+downstream(rows,150.,0.),
        now_s=0., previous_commands={})
    assert plan['map_boundary_feasible']
    assert not plan['reported_traffic_feasible']
    assert plan['failure_kind'] == 'traffic'


def test_early_exit_when_full_hold_fails_and_coast_exit_fits():
    obj, rows = policy(), packets()
    for now in range(50):
        # Introduce an acquired distant constraint only once measured HOLD starts.
        if obj.ready_at is not None:
            rows = tuple(replace(r, frame=replace(r.frame, acceleration_mps2=-.15)) for r in rows)
            obj.previous_commands = {'a':-.15,'b':-.15}
            out = obj(context(now, rows+downstream(rows,200.,0.)))
            assert out.phase == 'taper'
            assert not out.details['lifecycle_plan']['commitment']['feasible']
            assert out.details['lifecycle_plan']['exit_now']['feasible']
            assert set(out.wall_commands.values()) == {0.}
            assert out.details['early_exit_reason'] == 'reported_traffic_intersects_remaining_maneuver'
            return
        out = obj(context(now,rows))
        assert out.wall_commands
        rows = advance(rows,out)
    pytest.fail('HOLD never reached')


@pytest.mark.parametrize('mode',['missing','lane','stale','coverage','position'])
def test_information_or_member_loss_aborts_without_future_commands(mode):
    obj = policy()
    assert obj(context()).wall_commands
    rows = packets(1.)
    if mode == 'missing':
        rows = rows[:1]
    else:
        changes = {'lane':dict(lane_id='elsewhere'), 'stale':dict(timestamp_s=-1.),
            'coverage':dict(rear_coverage_m=0.), 'position':dict(position_m=None)}[mode]
        rows = (replace(rows[0], frame=replace(rows[0].frame,**changes)),rows[1])
    out = obj(context(1.,rows))
    assert out.phase == 'abort' and not out.wall_commands
    assert not obj(context(2.)).wall_commands


def test_exit_deadline_is_not_falsely_successful_handback():
    obj = policy()
    assert obj(context()).wall_commands
    obj.taper_at = -9.
    out = obj(context(1.,packets(1.,acceleration=.3)))
    assert out.phase == 'abort'
    assert 'without_measured_coast_terminal' in out.reason


def test_front_prefix_closure_and_current_rear_next_tick_are_protected():
    rows = packets()
    rows = tuple(replace(r,frame=replace(r.frame,
        front=NeighborMeasurement(50.,-6.))) for r in rows)
    out = policy()(context(rows=rows))
    assert out.phase == 'refused'
    assert not out.details['lifecycle_plan']['commitment']['reported_traffic_feasible']
    rear_rows = tuple(replace(r,frame=replace(r.frame,
        rear=NeighborMeasurement(15.1,5.))) for r in packets())
    out = policy()(context(rows=rear_rows))
    assert out.phase == 'refused' and 'rear' in out.reason


def test_empty_remote_coverage_is_declared_unknown_not_certified():
    rows=packets()
    plan=remaining_plan(proposal(rows),rows,now_s=0.,previous_commands={})
    assert plan['feasible']
    assert not plan['downstream_reports_available']
    assert not plan['unobserved_traffic_certified']
    assert not plan['safety_guarantee']
    assert plan['rear_protection_horizon_s'] == 1.
    assert not plan['rear_future_response_certified']


def test_prefix_closure_can_fail_even_when_terminal_closure_recovers():
    rows = packets()
    # Front is initially only 1 m/s slower, so initial TTC=6 s passes.
    rows = tuple(replace(r,frame=replace(r.frame,
        front=NeighborMeasurement(6.,-1.))) for r in rows)
    plan = remaining_plan(proposal(rows),rows,now_s=0.,previous_commands={})
    last = plan['planned_samples'][-1]
    final_margin = 1.-last['distance_travelled_m']['a']+23.*last['elapsed_s']
    assert final_margin > 0
    assert plan['minimum_traffic_margin_m'] < 0
    assert not plan['feasible']


def test_planned_hold_cannot_predict_violating_its_own_compactness_contract():
    rows = packets()
    rows = (rows[0],replace(rows[1],frame=replace(rows[1].frame,
        distance_to_merge_m=1950.,position_m=1050.)))
    p = replace(proposal(rows),target_speed_mps=24.)
    plan = remaining_plan(p,rows,now_s=0.,previous_commands={'a':0.,'b':0.},
        readiness_count=3,ready_at_s=0.)
    assert not plan['feasible']
    assert plan['failure_kind'] == 'formation'
    assert plan['reason'] == 'predicted_loss_of_compact_hold'
    assert plan['remaining_total_s'] < 3


def test_stale_remote_report_is_excluded_and_missingness_reported():
    rows = packets(2.)
    old = downstream(packets(0.),200.,0.)
    plan = remaining_plan(proposal(rows),rows+old,now_s=2.,previous_commands={})
    assert plan['stale_report_count'] == 2
    assert not plan['downstream_reports_available']


def test_response_incompatibility_does_not_issue_stronger_braking():
    obj=policy()
    obj(context())
    out=obj(context(1.,packets(1.,acceleration=-.8)))
    assert out.phase=='abort' and not out.wall_commands


def test_clock_no_reengagement_and_no_invented_native_acceleration():
    obj=policy()
    obj(context())
    with pytest.raises(ValueError):
        obj(context(2.))
    assert coast_terminal([r.frame for r in packets()],{'a':0.,'b':0.},LifecycleProfile())
    assert not coast_terminal([r.frame for r in packets()],{},LifecycleProfile())


def exact_failed_v1_entry(distance=2399.7839344848917):
    # Acquired values from frozen lifecycle_mechanics_v1 accelerating_complete,
    # t=5; no outcomes or future trajectory enter the policy. Generic test IDs.
    rows = packets(5.,speed=23.345754604179337,acceleration=.31855186227641497,
                   distance=distance,position=1100.2160655151083)
    rows = tuple(replace(r,frame=replace(r.frame,allowed_speed_mps=24.)) for r in rows)
    return (rows[0],replace(rows[1],frame=replace(rows[1].frame,
        distance_to_merge_m=distance+10.,position_m=rows[1].frame.position_m-10.)))


def test_exact_saved_entry_delays_hold_without_changing_commands_or_readiness():
    obj,rows = policy(start=5),exact_failed_v1_entry()
    out = obj(context(5.,rows))
    plan = out.details['lifecycle_plan']['commitment']
    assert out.phase == 'assembly' and plan['feasible']
    assert plan['predicted_hold_start_s'] == 20.
    assert plan['remaining_assembly_s'] == 15.
    assert plan['remaining_hold_s'] == 20.
    assert plan['remaining_exit_s'] == 1.
    assert [r['candidate_hold_start_s'] for r in plan['readiness_candidates']] == [14.,15.,16.,20.]
    assert not out.details['instantaneous_readiness_satisfied']
    assert out.wall_commands['a'] == pytest.approx(.16855186227641497)
    predicted_initial = plan['planned_samples']
    for now in range(6,42):
        rows = advance(rows,out)
        out = obj(context(now,rows))
        if now <= 19:
            assert out.phase == 'assembly'
            assert out.details['assembly_completed_at_s'] is None
            assert not out.details['hold_commitment_started']
        if now == 14:
            assert out.details['instantaneous_readiness_satisfied']
            assert out.details['readiness_count'] == 3
        if now == 17:
            assert not out.details['instantaneous_readiness_satisfied']
            assert out.details['readiness_count'] == 0
        if now == 20:
            assert out.phase == 'hold'
            assert out.details['assembly_completed_at_s'] == 20.
            assert out.details['readiness_count'] == 3
            assert out.details['lifecycle_plan']['commitment']['remaining_assembly_s'] == 0
        if out.wall_commands:
            # Prefixes include all delayed assembly; no suffix reset or altered
            # physical law is hidden behind the postponed phase commitment.
            assert dict(out.wall_commands) == predicted_initial[now-5]['commands_mps2']
        else:
            assert out.phase == 'release' and now == 41
            assert out.details['actual_hold_elapsed_s'] == 20
            break


def test_candidate_replanning_preserves_assembly_road_cost_and_input_state():
    rows = exact_failed_v1_entry(distance=150.)
    p = make_proposal(rows,now_s=5.,required_lanes=('L0','L1'),member_ids=('a','b'),profile=LifecycleProfile())
    previous = {'a':.31855186227641497,'b':.31855186227641497}
    before = dict(previous)
    plan = remaining_plan(p,rows,now_s=5.,previous_commands=previous)
    assert previous == before
    assert rows[0].frame.distance_to_merge_m == 150.
    assert plan['readiness_candidates'][0]['reason'] == 'predicted_loss_of_compact_hold'
    assert not plan['feasible'] and not plan['map_boundary_feasible']
    assert plan['remaining_assembly_s'] == 15.
    assert plan['minimum_map_margin_m'] < -600


def test_delayed_readiness_cannot_cross_original_assembly_deadline():
    rows = exact_failed_v1_entry()
    p = make_proposal(rows,now_s=5.,required_lanes=('L0','L1'),member_ids=('a','b'),profile=LifecycleProfile())
    # Same transient appears with only one second left in the original window.
    late = tuple(replace(r,sent_at_s=54.,frame=replace(r.frame,timestamp_s=54.)) for r in rows)
    plan = remaining_plan(p,late,now_s=54.,previous_commands={})
    assert not plan['feasible']
    assert plan['failure_kind'] == 'assembly'
    assert plan['remaining_total_s'] <= 1
    assert not plan['maintainable_hold_ready_now']


def test_committed_hold_cannot_be_relabelled_assembly_to_hide_failure():
    rows = packets(speed=24.)
    rows = (rows[0],replace(rows[1],frame=replace(rows[1].frame,
        distance_to_merge_m=1950.,position_m=1050.)))
    p = replace(proposal(rows),target_speed_mps=24.)
    plan = remaining_plan(p,rows,now_s=0.,previous_commands={'a':0.,'b':0.},
        readiness_count=3,ready_at_s=0.)
    assert not plan['feasible']
    assert plan['candidate_count'] == 1
    assert plan['failure_kind'] == 'formation'
    assert plan['remaining_assembly_s'] == 0
