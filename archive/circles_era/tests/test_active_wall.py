"""Synthetic pure regressions; no wall/predictor performance claims or SUMO."""
from dataclasses import FrozenInstanceError, replace
import pytest

from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import SensorFrame, NeighborMeasurement
from traffic_shepherd.wall import (WallMember, WallParameters, WallProposal,
    assess_feasibility, diagnostic_benefit_approval, authorize_wall, WallLifecycle)


def proposal(**changes):
    # Deliberate synthetic fixture choices, not scientific defaults.
    parameters = WallParameters(control_dt_s=1., assembly_s=2., hold_s=2., taper_s=2.,
        target_speed_mps=10., speed_gain_per_s=.5, acceleration_min_mps2=-1.,
        acceleration_max_mps2=.5, assembly_speed_tolerance_mps=.5,
        maximum_member_span_m=20., max_report_age_s=1., minimum_front_gap_m=5.,
        minimum_rear_gap_m=5., minimum_front_ttc_s=3., minimum_rear_ttc_s=3.)
    values = dict(start_s=0., required_lanes=("L0","L1"),
                  members=(WallMember("a","L0"),WallMember("b","L1")), parameters=parameters)
    values.update(changes)
    return WallProposal(**values)


def reports(now=0., speed=10.):
    return [CAVReport(SensorFrame(vehicle_id=v,timestamp_s=now,speed_mps=speed,
        acceleration_mps2=0.,allowed_speed_mps=30.,lane_index=i,distance_to_merge_m=100.+i*5,
        front=NeighborMeasurement(30.,0.),rear=NeighborMeasurement(30.,0.),
        front_coverage_m=100.,rear_coverage_m=100.,lane_id=f"L{i}",position_m=200.),
        sent_at_s=now,owner="damping") for i,v in enumerate(("a","b"))]


def authorized(p=None, initial_reports=None):
    p = p or proposal()
    feasible = assess_feasibility(p,initial_reports if initial_reports is not None else reports(),now_s=0.)
    benefit = diagnostic_benefit_approval(p,feasible,report_only_score=.2,
        explicit_threshold=.1,diagnostic_source="synthetic report-only test judgment; no learned predictor")
    return p, feasible, benefit, authorize_wall(p,feasible,benefit)


def test_immutable_fingerprint_binds_all_treatment_parameters():
    p = proposal()
    with pytest.raises(FrozenInstanceError):
        p.start_s = 1.
    assert p.fingerprint == proposal().fingerprint
    assert replace(p,parameters=replace(p.parameters,hold_s=3.)).fingerprint != p.fingerprint
    assert replace(p,start_s=1.).fingerprint != p.fingerprint
    with pytest.raises(ValueError,match="distinct CAV"):
        proposal(members=(WallMember("a","L0"),WallMember("a","L1")))


def test_missing_members_reports_and_unsafe_geometry_are_different():
    p = proposal()
    missing = replace(p,members=(p.members[0],))
    assert assess_feasibility(missing,reports(),now_s=0.).status == "insufficient_members"
    incomplete = assess_feasibility(p,reports()[:1],now_s=0.)
    assert incomplete.status == "insufficient_information"
    unsafe = reports()
    unsafe[1] = replace(unsafe[1],frame=replace(unsafe[1].frame,distance_to_merge_m=150.))
    assert assess_feasibility(p,unsafe,now_s=0.).status == "unsafe_geometry"


def test_expected_lanes_are_not_inferred_from_reporters():
    p = proposal(required_lanes=("L0","L1","L2"))
    assert assess_feasibility(p,reports(),now_s=0.).status == "insufficient_members"
    unknown = reports()
    unknown[0] = replace(unknown[0],frame=replace(unknown[0].frame,lane_id=None))
    assert assess_feasibility(proposal(),unknown,now_s=0.).status == "insufficient_information"


@pytest.mark.parametrize("kind", ["front","rear"])
def test_reported_front_and_rear_ttc_guard(kind):
    rows = reports()
    neighbor = NeighborMeasurement(10.,-5. if kind == "front" else 5.)
    rows[0] = replace(rows[0],frame=replace(rows[0].frame,**{kind:neighbor}))
    gate = assess_feasibility(proposal(),rows,now_s=0.)
    assert gate.status == "unsafe_geometry" and kind in gate.reason


def test_missing_coverage_is_not_safe_empty_space():
    rows = reports()
    rows[0] = replace(rows[0],frame=replace(rows[0].frame,front=None,front_coverage_m=2.))
    assert assess_feasibility(proposal(),rows,now_s=0.).status == "insufficient_information"


def test_only_typed_cav_packets_and_fresh_evidence():
    with pytest.raises(TypeError,match="CAVReport"):
        assess_feasibility(proposal(),[{"global_density":100}],now_s=0.)
    assert assess_feasibility(proposal(),reports(),now_s=2.).status == "insufficient_information"
    with pytest.raises(ValueError,match="one latest"):
        assess_feasibility(proposal(),reports()+reports()[:1],now_s=0.)


def test_two_gates_bind_same_proposal_and_same_snapshot():
    p, feasibility, benefit, _ = authorized()
    wrong = replace(p,parameters=replace(p.parameters,target_speed_mps=9.))
    with pytest.raises(ValueError,match="identical"):
        authorize_wall(wrong,feasibility,benefit)
    other_snapshot = assess_feasibility(p,reports(speed=11.),now_s=0.)
    with pytest.raises(ValueError,match="snapshot"):
        authorize_wall(p,other_snapshot,benefit)
    rejected = diagnostic_benefit_approval(p,feasibility,report_only_score=-.2,
        explicit_threshold=.1,diagnostic_source="synthetic report-only negative judgment")
    with pytest.raises(ValueError,match="both diagnostic"):
        authorize_wall(p,feasibility,rejected)
    assert not benefit.scientifically_validated


def test_safety_gate_cannot_be_bypassed_by_positive_benefit():
    p = proposal()
    feasible = assess_feasibility(p,reports()[:1],now_s=0.)
    benefit = diagnostic_benefit_approval(p,feasible,report_only_score=100.,
        explicit_threshold=0.,diagnostic_source="synthetic judgment")
    with pytest.raises(ValueError,match="both diagnostic"):
        authorize_wall(p,feasible,benefit)


def test_assembly_hold_taper_release_and_return_same_damping_mask():
    p,_,_,authorization = authorized()
    lifecycle = WallLifecycle(p,authorization)
    records = [lifecycle.step(reports(float(t)),now_s=float(t),damping_commands={"a":-.8,"b":-.4}) for t in range(8)]
    assert [r.phase for r in records] == ["assembly","assembly","hold","hold","taper","taper","release","release"]
    assert records[5].commands == {"a":-.4,"b":-.2}
    assert records[6].commands == records[7].commands == {}
    # A returned dictionary cannot mutate a previously emitted immutable step.
    altered = records[5].commands
    altered["a"] = 100.
    assert records[5].commands["a"] == -.4


def test_assembly_deadline_abort_is_terminal():
    p,_,_,authorization = authorized(initial_reports=reports(0.,12.))
    lifecycle = WallLifecycle(p,authorization)
    lifecycle.step(reports(0.,12.),now_s=0.,damping_commands={})
    lifecycle.step(reports(1.,12.),now_s=1.,damping_commands={})
    aborted = lifecycle.step(reports(2.,12.),now_s=2.,damping_commands={})
    assert aborted.phase == "abort" and "deadline" in aborted.reason and aborted.commands == {}
    assert lifecycle.step(reports(3.,10.),now_s=3.,damping_commands={}).phase == "abort"


def test_taper_missing_damping_aborts_instead_of_inventing_zero():
    p,_,_,authorization = authorized()
    lifecycle = WallLifecycle(p,authorization)
    for t in range(4):
        lifecycle.step(reports(float(t)),now_s=float(t),damping_commands={})
    result = lifecycle.step(reports(4.),now_s=4.,damping_commands={"a":-.2})
    assert result.phase == "abort" and result.commands == {}
    assert "damping reference" in result.reason


def test_member_loss_aborts_whole_wall_without_partial_command_map():
    p,_,_,authorization = authorized()
    lifecycle = WallLifecycle(p,authorization)
    lifecycle.step(reports(),now_s=0.,damping_commands={})
    result = lifecycle.step(reports(1.)[:1],now_s=1.,damping_commands={})
    assert result.phase == "abort" and result.commands == {}
    assert "insufficient_information" in result.reason


def test_control_time_cannot_repeat_or_skip_assembly():
    p,_,_,authorization = authorized()
    with pytest.raises(ValueError,match="proposed start"):
        WallLifecycle(p,authorization).step(reports(2.),now_s=2.,damping_commands={})
    lifecycle = WallLifecycle(p,authorization)
    lifecycle.step(reports(),now_s=0.,damping_commands={})
    with pytest.raises(ValueError,match="exactly one"):
        lifecycle.step(reports(),now_s=0.,damping_commands={})


def test_activation_snapshot_change_requires_new_both_gate_authorization():
    p,_,_,authorization = authorized()
    lifecycle = WallLifecycle(p,authorization)
    result = lifecycle.step(reports(0.,11.),now_s=0.,damping_commands={})
    assert result.phase == "abort" and result.commands == {}
    assert "reauthorization" in result.reason
    assert lifecycle.step(reports(1.),now_s=1.,damping_commands={}).phase == "abort"
    # The new observation can only activate after both gates bind its digest.
    _,_,_,updated = authorized(p,initial_reports=reports(0.,11.))
    assert WallLifecycle(p,updated).step(reports(0.,11.),now_s=0.,damping_commands={}).phase == "assembly"


def test_taper_retains_wall_envelope_but_release_does_not_constrain_damping():
    p,_,_,authorization = authorized()
    lifecycle = WallLifecycle(p,authorization)
    records = [lifecycle.step(reports(float(t)),now_s=float(t),
        damping_commands={"a":-3.,"b":1.5}) for t in range(7)]
    assert records[5].commands == {"a":-1.,"b":.5}
    assert records[6].phase == "release" and records[6].commands == {}


def test_prestart_diagnostic_authorization_is_explicitly_unsupported():
    p = proposal(start_s=1.)
    feasibility = assess_feasibility(p,reports(0.),now_s=0.)
    benefit = diagnostic_benefit_approval(p,feasibility,report_only_score=.2,
        explicit_threshold=.1,diagnostic_source="synthetic report-only judgment")
    with pytest.raises(ValueError,match="exactly at the proposed start"):
        authorize_wall(p,feasibility,benefit)
    current = assess_feasibility(p,reports(1.),now_s=1.)
    current_benefit = diagnostic_benefit_approval(p,current,report_only_score=.2,
        explicit_threshold=.1,diagnostic_source="synthetic report-only judgment")
    authorization = authorize_wall(p,current,current_benefit)
    assert WallLifecycle(p,authorization).step(reports(1.),now_s=1.,damping_commands={}).phase == "assembly"
