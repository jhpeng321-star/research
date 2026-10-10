"""Synthetic contracts only; no test here is physical formation evidence."""
from dataclasses import asdict, replace
import json

import pytest

from traffic_shepherd.formation import (
    FormationController, FormationProfile, FormationProposal, assess_formation,
    authorize_experiment, make_proposal,
)
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame
from traffic_shepherd.wall import WallMember


def reports(now=0., *, distances=(3000., 3150.), speeds=(22., 22.), front_gap=80., rear_gap=80.):
    return [CAVReport(SensorFrame(vehicle_id=v, timestamp_s=now, speed_mps=speeds[i],
        acceleration_mps2=0., allowed_speed_mps=30., lane_index=i,
        distance_to_merge_m=distances[i], lane_id=f"L{i}", position_m=4000.-distances[i],
        front=None if front_gap is None else NeighborMeasurement(front_gap, 0.),
        rear=None if rear_gap is None else NeighborMeasurement(rear_gap, 0.),
        front_coverage_m=100., rear_coverage_m=100.), now, "damping")
        for i, v in enumerate(("a", "b"))]


def setup(rows=None, profile=None, arm="wall"):
    rows = rows or reports()
    proposal = make_proposal(rows, now_s=0., required_lanes=("L0", "L1"),
                             member_ids=("a", "b"), profile=profile)
    assessment = assess_formation(proposal, rows, now_s=0.)
    authorization = authorize_experiment(proposal, assessment,
                                        registration_id="synthetic_fixture_only", arm=arm)
    return proposal, FormationController(proposal, authorization)


def test_profile_units_and_exact_adopted_actuator():
    p = FormationProfile()
    assert (p.control_dt_s, p.acceleration_min_mps2, p.acceleration_max_mps2) == (1., -3., 1.5)
    assert p.position_gain_per_s == .12 and p.speed_gain_per_s == .6
    assert (p.assembly_deadline_s, p.hold_s, p.taper_s) == (50., 60., 10.)
    with pytest.raises(ValueError, match="adopted"):
        replace(p, acceleration_min_mps2=-1.)
    with pytest.raises(ValueError, match="consecutive"):
        replace(p, readiness_samples=1)


@pytest.mark.parametrize("span", [10., 25., 49., 50., 51., 75., 100., 150.])
def test_initial_separation_is_not_formed_wall_admission_cutoff(span):
    rows = reports(distances=(3000., 3000.+span))
    proposal, _ = setup(rows)
    assessment = assess_formation(proposal, rows, now_s=0.)
    assert assessment.approved and not assessment.physically_validated
    assert assessment.initial_or_current_span_m == span
    assert assessment.estimated_assembly_s > 0
    assert assessment.estimated_delivery_runway_m > 0
    assert proposal.profile.formed_span_m == 50.


def test_upstream_member_closes_and_downstream_member_yields_longitudinally():
    proposal, controller = setup()
    result = controller.step(reports(), now_s=0., damping_commands={"a":0., "b":0.})
    assert result.phase == "assembly"
    assert result.commands == {"a": -3., "b": pytest.approx(.6)}
    assert dict(result.desired_speed_items) == {"a": 17., "b": 23.}
    assert result.span_m == 150. and not result.compact
    assert proposal.target_speed_mps == 20.


def test_cached_member_report_cannot_authorize_command_without_current_local_reference():
    _, controller = setup()
    initial = reports()
    assert controller.step(initial, now_s=0., damping_commands={"a": 0., "b": 0.}).phase == "assembly"
    # The RSU may legitimately retain this one-second-old report. Missing
    # current CAV execution evidence must nevertheless abort the entire cohort.
    step = controller.step(initial, now_s=1., damping_commands={"a": 0.})
    assert step.phase == "abort" and not step.commands
    assert step.reason == "missing_current_local_damping_reference"


@pytest.mark.parametrize("speed, expected", [(0., 0.), (4., 2.), (22., 20.), (30., 24.)])
def test_target_is_acquired_speed_rule_at_all_states(speed, expected):
    proposal, _ = setup(reports(speeds=(speed, speed)))
    assert proposal.target_speed_mps == expected
    # No congestion, density, true penetration or hidden type inputs are exposed.
    assert set(asdict(proposal)) == {"start_s", "required_lanes", "members", "target_speed_mps", "profile"}


def test_missing_members_information_unsafe_lane_and_runway_refusals_are_distinct():
    proposal, _ = setup()
    fewer = replace(proposal, members=(proposal.members[0],))
    assert assess_formation(fewer, reports(), now_s=0.).status == "insufficient_members"
    assert assess_formation(proposal, reports()[:1], now_s=0.).reason == "missing_member_report"
    rows = reports()
    rows[1] = replace(rows[1], frame=replace(rows[1].frame, lane_id="OTHER"))
    assert assess_formation(proposal, rows, now_s=0.).reason == "member_lane_loss"
    assert assess_formation(proposal, reports(distances=(65., 100.)), now_s=0.).status == "insufficient_runway"


def test_current_runway_and_candidate_delivery_estimate_are_distinct():
    rows = reports(distances=(400., 550.))
    proposal = make_proposal(rows, now_s=0., required_lanes=("L0", "L1"), member_ids=("a", "b"))
    a = assess_formation(proposal, rows, now_s=0.)
    assert a.approved and a.estimated_runway_slack_m < 0
    assert "not_yet_measured" in a.reason
    assert a.estimated_delivery_runway_m < a.worst_duration_runway_m


@pytest.mark.parametrize("label, relative", [("front", -5.), ("rear", 5.)])
def test_front_rear_local_safety_refuses_current_ttc(label, relative):
    proposal, _ = setup()
    rows = reports()
    rows[0] = replace(rows[0], frame=replace(rows[0].frame,
        **{label: NeighborMeasurement(10., relative)}))
    assert assess_formation(proposal, rows, now_s=0.).reason == f"{label}_clearance_or_ttc"


def test_command_would_compress_rear_then_whole_wall_aborts_without_partial_actions():
    rows = reports(rear_gap=8.)
    _, controller = setup(rows)
    step = controller.step(rows, now_s=0., damping_commands={"a":0., "b":0.})
    assert step.phase == "abort" and step.commands == {}
    assert "rear_next_interval" in step.reason


def test_absence_of_detection_requires_observed_coverage():
    rows = reports(front_gap=None)
    proposal, _ = setup(rows)
    assert assess_formation(proposal, rows, now_s=0.).approved
    rows[0] = replace(rows[0], frame=replace(rows[0].frame, front_coverage_m=0.))
    assert assess_formation(proposal, rows, now_s=0.).reason == "front_coverage_unknown"


def test_typed_unique_received_packets_and_clock_validation():
    proposal, _ = setup()
    with pytest.raises(TypeError, match="CAVReport"):
        assess_formation(proposal, [{"global_density": 90}], now_s=0.)
    with pytest.raises(ValueError, match="latest received"):
        assess_formation(proposal, reports()+reports()[:1], now_s=0.)
    assert assess_formation(proposal, reports(), now_s=2.).reason == "stale_or_future_member_report"
    assert assess_formation(proposal, reports(now=1.), now_s=0.).reason == "stale_or_future_member_report"


def test_authorization_is_explicit_assignment_bound_to_proposal_and_evidence():
    rows = reports()
    p, controller = setup(rows)
    a = assess_formation(p, rows, now_s=0.)
    with pytest.raises(ValueError, match="experimental"):
        authorize_experiment(p, a, registration_id="", arm="wall")
    with pytest.raises(ValueError, match="this proposal"):
        authorize_experiment(replace(p, target_speed_mps=19.), a, registration_id="test")
    step = controller.step(reports(speeds=(21., 22.)), now_s=0., damping_commands={"a":0., "b":0.})
    assert step.phase == "abort" and "evidence_changed" in step.reason
    auth = authorize_experiment(p, a, registration_id="fixture")
    assert auth.source == "registered_experimental_assignment"
    assert auth.qualification_sha256 is None and auth.benefit_artifact_sha256 is None
    assert "benefit_score" not in json.dumps(asdict(auth))


def test_consecutive_readiness_hold_taper_and_exact_release():
    initial = reports(distances=(3000., 3000.), speeds=(10., 10.))
    profile = replace(FormationProfile(), hold_s=3., taper_s=2.)
    p, controller = setup(initial, profile)
    outputs = []
    # Initial target=8; readiness begins at t=1 after a synthetic measured settle.
    for t in range(9):
        row = initial if t == 0 else reports(float(t), distances=(3000., 3000.), speeds=(8., 8.))
        outputs.append(controller.step(row, now_s=float(t), damping_commands={"a": -.4, "b": .6}))
    assert [o.phase for o in outputs] == ["assembly", "assembly", "assembly", "hold", "hold", "hold", "taper", "taper", "release"]
    assert outputs[3].readiness_count == 3 and outputs[3].assembly_completed_at_s == 3.
    assert outputs[6].commands == {"a": -.2, "b": .3}
    assert outputs[7].commands == {"a": -.4, "b": .6}
    assert outputs[8].commands == {}
    assert p.profile.formed_span_m == 50.


def test_nonconsecutive_compact_samples_do_not_qualify_hold():
    initial = reports(distances=(3000., 3000.), speeds=(10., 10.))
    _, controller = setup(initial)
    controller.step(initial, now_s=0., damping_commands={"a":0., "b":0.})
    for t, speeds in [(1., (8., 8.)), (2., (8., 8.)), (3., (10., 8.)), (4., (8., 8.))]:
        step = controller.step(reports(t, distances=(3000., 3000.), speeds=speeds), now_s=t, damping_commands={"a":0., "b":0.})
    assert step.phase == "assembly" and step.readiness_count == 1


def test_deadline_failure_not_synthetic_span_pass_and_abort_is_terminal():
    initial = reports()
    _, controller = setup(initial, replace(FormationProfile(), assembly_deadline_s=2.))
    steps = [controller.step(reports(float(t)), now_s=float(t), damping_commands={"a":0., "b":0.}) for t in range(4)]
    assert [s.phase for s in steps] == ["assembly", "assembly", "abort", "abort"]
    assert "deadline" in steps[2].reason and steps[2].commands == {}


@pytest.mark.parametrize("problem", ["report_loss", "lane_loss", "cut_in", "runway"])
def test_dynamic_failure_aborts_entire_cohort_and_returns_damping_mask(problem):
    _, controller = setup()
    controller.step(reports(), now_s=0., damping_commands={"a":0., "b":0.})
    rows = reports(1.)
    if problem == "report_loss":
        rows = rows[:1]
    elif problem == "lane_loss":
        rows[1] = replace(rows[1], frame=replace(rows[1].frame, lane_id="other"))
    elif problem == "cut_in":
        rows[1] = replace(rows[1], frame=replace(rows[1].frame, front=NeighborMeasurement(4., -2.)))
    else:
        rows[1] = replace(rows[1], frame=replace(rows[1].frame, distance_to_merge_m=60.))
    step = controller.step(rows, now_s=1., damping_commands={"a":0., "b":0.})
    assert step.phase == "abort" and step.commands == {}


def test_assembly_only_tapers_on_readiness_without_hold():
    initial = reports(distances=(3000., 3000.), speeds=(10., 10.))
    _, controller = setup(initial, arm="assembly_only")
    outputs = [controller.step(initial if t == 0 else reports(float(t), distances=(3000., 3000.), speeds=(8., 8.)),
                               now_s=float(t), damping_commands={"a":0., "b":0.}) for t in range(4)]
    assert outputs[-1].phase == "taper"
    assert all(row.phase != "hold" for row in outputs)


def test_lost_compact_hold_and_missing_taper_reference_abort():
    initial = reports(distances=(3000., 3000.), speeds=(10., 10.))
    _, controller = setup(initial, replace(FormationProfile(), hold_s=1., taper_s=2.))
    for t in range(4):
        controller.step(initial if t == 0 else reports(float(t), distances=(3000., 3000.), speeds=(8.,8.)),
                        now_s=float(t), damping_commands={"a":0., "b":0.})
    step = controller.step(reports(4., distances=(3000.,3000.), speeds=(8.,8.)), now_s=4., damping_commands={"a":0.})
    assert step.phase == "abort" and "damping_reference" in step.reason
    _, controller = setup(initial)
    for t in range(4):
        controller.step(initial if t == 0 else reports(float(t), distances=(3000.,3000.), speeds=(8.,8.)), now_s=float(t), damping_commands={"a":0., "b":0.})
    step = controller.step(reports(4., distances=(3000.,3051.), speeds=(8.,8.)), now_s=4., damping_commands={"a":0., "b":0.})
    assert step.phase == "abort" and "compact_hold" in step.reason


@pytest.mark.parametrize("native, numeric", [(("a","b"),{}), (("a",),{"b":.2})])
def test_valid_native_damping_reference_releases_entire_mixed_cohort(native, numeric):
    initial = reports(distances=(3000.,3000.), speeds=(10.,10.))
    _, controller = setup(initial, replace(FormationProfile(),hold_s=1.,taper_s=2.))
    for t in range(4):
        controller.step(initial if t == 0 else reports(float(t),distances=(3000.,3000.),speeds=(8.,8.)),
                        now_s=float(t),damping_commands={"a":0., "b":0.})
    step = controller.step(reports(4.,distances=(3000.,3000.),speeds=(8.,8.)),
                           now_s=4.,damping_commands=numeric,damping_native_ids=native)
    assert step.phase == "release" and step.commands == {}
    assert step.reason == "identical_damping_requests_native_explicit_handback"


def test_native_reference_is_not_a_missing_numeric_reference_or_foreign_mask():
    _, controller = setup()
    with pytest.raises(ValueError,match="disjoint"):
        controller.step(reports(),now_s=0.,damping_commands={"a":0.},damping_native_ids=("a",))
    _, controller = setup()
    with pytest.raises(ValueError,match="members"):
        controller.step(reports(),now_s=0.,damping_commands={"a":0., "b":0.},damping_native_ids=("hdv",))


def test_clock_cannot_skip_repeat_or_start_late_and_membership_is_immutable():
    proposal, controller = setup()
    with pytest.raises(ValueError, match="registered start"):
        controller.step(reports(1.), now_s=1., damping_commands={"a":0., "b":0.})
    controller.step(reports(), now_s=0., damping_commands={"a":0., "b":0.})
    with pytest.raises(ValueError, match="exactly one"):
        controller.step(reports(), now_s=0., damping_commands={"a":0., "b":0.})
    with pytest.raises(ValueError, match="distinct CAV"):
        FormationProposal(0., ("L0","L1"), (WallMember("a","L0"), WallMember("a","L1")), 20., FormationProfile())
    assert replace(proposal, target_speed_mps=19.).fingerprint != proposal.fingerprint


def test_no_trafficsimulator_api_or_hidden_field_dependency_in_pure_controller():
    import inspect
    import traffic_shepherd.formation as module
    source = inspect.getsource(module)
    for forbidden in ("import traci", ".getIDList(", ".getLaneID(", ".slowDown(", ".setSpeed("):
        assert forbidden not in source


@pytest.mark.parametrize("span, speed_difference", [(10.,0.), (49.,0.), (51.,0.), (150.,0.), (150.,4.), (150.,-4.)])
def test_feedback_closes_distance_in_ideal_point_mass_algebra_not_physical_sumo(span, speed_difference):
    distances = [5000., 5000.+span]
    speeds = [22.+speed_difference/2, 22.-speed_difference/2]
    initial = reports(distances=distances, speeds=speeds, front_gap=None, rear_gap=None)
    proposal, controller = setup(initial)
    ready_at = None
    for t in range(140):
        rows = reports(float(t), distances=distances, speeds=speeds, front_gap=None, rear_gap=None)
        step = controller.step(rows, now_s=float(t), damping_commands={"a":0., "b":0.})
        if step.phase == "hold" and ready_at is None:
            ready_at = t
        assert step.phase != "abort"
        for i, name in enumerate(("a", "b")):
            acceleration = step.commands.get(name, 0.)
            distances[i] -= speeds[i] + .5*acceleration
            speeds[i] = max(0., min(30., speeds[i]+acceleration))
    assert ready_at is not None and ready_at <= proposal.profile.assembly_deadline_s
    assert step.phase == "release"
    assert step.span_m < span

