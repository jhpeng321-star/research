from dataclasses import asdict, replace

import pytest

from output.h0_exit_probe_v1 import policy as h0_exit_probe
from traffic_shepherd.cohort_study import SELECTOR, ReportSelectedWall
from output.h0_exit_probe_v1.policy import VERSION, H0ExitProbePolicy
from traffic_shepherd.lifecycle_wall import LifecycleProfile, LifecycleWallPolicy
from traffic_shepherd.paired_experiments import PolicyInput
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import SensorFrame


def packets(now=0., speed=24., acceleration=0., distance=2000., position=1000.):
    return tuple(CAVReport(SensorFrame(v, now, speed, acceleration, 30., i,
        distance_to_merge_m=distance, front=None, rear=None, front_coverage_m=100.,
        rear_coverage_m=100., lane_id=f'L{i}', position_m=position), now, 'damping')
        for i, v in enumerate(('a', 'b')))


def job():
    return dict(arm='wall', decision_time_s=0, policy_spec=dict(
        member_ids=['a', 'b'], required_lanes=['L0', 'L1'], profile=asdict(LifecycleProfile())))


REGISTRATION = {'registration_sha256': 'synthetic_registered_treatment'}


def context(now=0., rows=None):
    rows = packets(now) if rows is None else rows
    return PolicyInput(now, rows, (), {},
        frozenset(r.frame.vehicle_id for r in rows if r.frame.timestamp_s == now))


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


TERMINALS = frozenset({'release', 'abort', 'refused'})


def _run_pair(limit=90):
    """Forward the wrapper world and the plain lifecycle world on identical
    prefixes; return both policies, outputs, per-tick delegate state and the
    branch tick."""
    wrapped = H0ExitProbePolicy(job(), REGISTRATION, first_factory=LifecycleWallPolicy)
    plain = LifecycleWallPolicy(job(), REGISTRATION)
    rows_w, rows_p, outputs, states = packets(acceleration=.9), packets(acceleration=.9), [], []
    branch_tick = None
    for now in range(limit):
        out_w = wrapped(context(float(now), rows_w))
        out_p = plain(context(float(now), rows_p))
        outputs.append((out_w, out_p))
        states.append((wrapped.first.ready_at, wrapped.first.taper_at))
        if branch_tick is None and wrapped.receipt['branch'] is not None:
            branch_tick = now
        if out_w.phase in TERMINALS and out_p.phase in TERMINALS:
            break
        rows_w = advance(rows_w, out_w) if out_w.wall_commands else rows_w
        rows_p = advance(rows_p, out_p) if out_p.wall_commands else rows_p
    return wrapped, plain, outputs, states, branch_tick


def test_branch_occurs_and_prefix_is_identical_to_executed_lifecycle():
    wrapped, _, outputs, states, branch_tick = _run_pair()
    assert branch_tick is not None and branch_tick >= 1
    for now, (out_w, out_p) in enumerate(outputs[:branch_tick]):
        assert out_w.phase == out_p.phase
        assert out_w.reason == out_p.reason
        assert dict(out_w.wall_commands) == dict(out_p.wall_commands)
        assert out_w.proposal_sha256 == out_p.proposal_sha256
        plain_details = {k: v for k, v in out_p.details.items()}
        wrapped_details = {k: v for k, v in out_w.details.items() if k != 'h0_exit_probe'}
        assert wrapped_details == plain_details
    assert outputs[branch_tick][1].phase == 'hold'
    assert outputs[branch_tick][0].phase != 'hold'


def test_branch_delivers_zero_hold_through_existing_exit_and_handback():
    wrapped, plain, outputs, states, branch_tick = _run_pair()
    assert wrapped.receipt['version'] == VERSION
    branch = wrapped.receipt['branch']
    assert branch['h0_delivered'] is True
    assert branch['exit_admitted'] is True
    assert branch['branch_request_time_s'] == float(branch_tick)
    assert branch['exit_start_s'] == float(branch_tick)
    assert branch['first_allowed_command_difference_s'] == float(branch_tick)
    assert branch['requested_profile_hold_s'] == 20. and branch['delivered_hold_target_s'] == 0.
    assert branch['committed_hold_readiness_admitted'] is True
    assert branch['readiness_count_at_branch'] >= wrapped.first.profile.readiness_samples
    assert branch['hold_plan']['maintainable_hold_ready_now'] is True
    assert branch['proposed_exit_plan']['feasible'] is True
    assert states[branch_tick] == (float(branch_tick), float(branch_tick))
    out = outputs[branch_tick][0]
    assert out.details['actual_hold_elapsed_s'] == 0.
    assert out.details['exit_started_at_s'] == float(branch_tick)
    phases = [o[0].phase for o in outputs[branch_tick:]]
    assert phases[0] in {'taper', 'release'}
    assert 'release' in phases
    released = next(o[0] for o in outputs[branch_tick:] if o[0].phase == 'release')
    assert not released.wall_commands
    assert released.details['actual_hold_elapsed_s'] == 0.
    plain_release = next(o[1] for o in outputs if o[1].phase == 'release')
    assert plain_release.details['actual_hold_elapsed_s'] == 20.


def test_taper_commands_respect_wall_bounds_and_zero_request():
    wrapped, _, outputs, states, branch_tick = _run_pair()
    for out_w, _ in outputs[branch_tick:]:
        for value in out_w.wall_commands.values():
            assert -.3-1e-9 <= value <= 1.5+1e-9
    taper_rows = [o[0] for o in outputs[branch_tick:] if o[0].phase == 'taper']
    assert taper_rows, 'expected an explicit taper phase'
    for row in taper_rows:
        for vid, evidence in row.details['command_resolution'].items():
            assert evidence['requested_acceleration_mps2'] == 0.


def test_inadmissible_exit_falls_back_to_retained_original_policy(monkeypatch):
    real = h0_exit_probe.remaining_plan

    def fake(proposal, reports, *, now_s, previous_commands, readiness_count=0,
             ready_at_s=None, exit_started_at_s=None, exit_now=False):
        if exit_now:
            return dict(feasible=False, failure_kind='response',
                        reason='synthetic_exit_inadmissible', remaining_total_s=None)
        return real(proposal, reports, now_s=now_s, previous_commands=previous_commands,
                    readiness_count=readiness_count, ready_at_s=ready_at_s,
                    exit_started_at_s=exit_started_at_s, exit_now=exit_now)

    monkeypatch.setattr(h0_exit_probe, 'remaining_plan', fake)
    wrapped, plain, outputs, states, branch_tick = _run_pair()
    branch = wrapped.receipt['branch']
    assert branch is not None and branch['exit_admitted'] is False
    assert branch['fallback'] == 'retained_original_H20_lifecycle'
    assert branch['h0_delivered'] is False
    assert branch['exit_start_s'] is None
    assert outputs[branch_tick][0].phase == 'hold'
    assert dict(outputs[branch_tick][0].wall_commands) == dict(outputs[branch_tick][1].wall_commands)
    assert states[branch_tick] == (float(branch_tick), None)
    plain_release = next(o[1] for o in outputs if o[1].phase == 'release')
    fallback_release = next(o[0] for o in outputs if o[0].phase == 'release')
    assert fallback_release.details['actual_hold_elapsed_s'] == 20. == plain_release.details['actual_hold_elapsed_s']
    assert fallback_release.reason == plain_release.reason


def test_wrapper_cannot_arm_without_measured_readiness(monkeypatch):
    monkeypatch.setattr(h0_exit_probe, '_settled', lambda rows, proposal: False)
    wrapped, plain, outputs, states, branch_tick = _run_pair()
    assert branch_tick is None and wrapped.receipt['branch'] is None
    # The delegate keeps its own measured readiness and completes the ordinary
    # H20 policy; the wrapper neither arms nor alters any command.
    for out_w, out_p in outputs:
        assert out_w.phase == out_p.phase
        assert dict(out_w.wall_commands) == dict(out_p.wall_commands)


def test_report_selected_wall_delegate_is_wrapped_not_replaced():
    bound = dict(selector=SELECTOR, evidence_sha256='e', selection_sha256='s',
                 proposal_sha256='p')
    spec = job()['policy_spec']
    spec['cohort_selection'] = bound
    wrapped = H0ExitProbePolicy(dict(arm='wall', decision_time_s=600, policy_spec=spec), REGISTRATION)
    assert isinstance(wrapped.first, ReportSelectedWall)
    assert wrapped._inner() is None
    out = wrapped(PolicyInput(5., (), (), {}, frozenset()))
    assert out.phase == 'damping'
    assert out.details['h0_exit_probe']['version'] == VERSION


def test_receipt_is_snapshotted_per_output_and_rejects_foreign_input():
    wrapped = H0ExitProbePolicy(job(), REGISTRATION, first_factory=LifecycleWallPolicy)
    with pytest.raises(TypeError):
        wrapped(object())
    out = wrapped(context(0.))
    wrapped.receipt['version'] = 'mutated'
    assert out.details['h0_exit_probe']['version'] == VERSION


def test_no_arming_when_formation_screen_fails(monkeypatch):
    real_assess = h0_exit_probe.assess_formation

    def failed(proposal, reports, *, now_s):
        result = real_assess(proposal, reports, now_s=now_s)
        return replace(result, status='synthetic_screen_failure')

    monkeypatch.setattr(h0_exit_probe, 'assess_formation', failed)
    wrapped, plain, outputs, states, branch_tick = _run_pair(limit=6)
    assert branch_tick is None and wrapped.receipt['branch'] is None
    out_w, out_p = outputs[0]
    assert out_w.phase == out_p.phase  # identical refusal/abort path


def test_delegated_lifecycle_still_enforces_its_one_second_clock():
    wrapped = H0ExitProbePolicy(job(), REGISTRATION, first_factory=LifecycleWallPolicy)
    wrapped(context(0.))
    with pytest.raises(ValueError):
        wrapped(context(2.))
