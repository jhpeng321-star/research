"""Report-acquisition and explicit ownership tests; no simulator execution."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from traffic_shepherd.formation import FormationProfile, make_proposal
from traffic_shepherd.paired_experiments import PolicyInput, PolicyOutput, fixed_formation_policy
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import SensorFrame, NeighborMeasurement
from traffic_shepherd.report_dissolution import VERSION, downstream_envelope, ReportDissolutionPolicy


def packets(now=0., gap=177., downstream_speed=3., rear_gap=100.):
    rows = []
    for i, name in enumerate(("a", "b")):
        frame = SensorFrame(name, now, 24., 0., 30., i, distance_to_merge_m=1000.,
            front=None, rear=NeighborMeasurement(rear_gap, 0.) if rear_gap < 100 else None,
            front_coverage_m=100., rear_coverage_m=100., lane_id=f"L{i}", position_m=1000.)
        rows.append(CAVReport(frame, now, "wall"))
        ahead = replace(frame, vehicle_id=f"down{i}", speed_mps=downstream_speed,
            position_m=1000.+gap, distance_to_merge_m=1000.-gap, rear=None)
        rows.append(CAVReport(ahead, now, "native"))
    return rows


def proposal():
    return make_proposal(packets(), now_s=0., required_lanes=("L0", "L1"), member_ids=("a", "b"),
        profile=FormationProfile(version=VERSION, target_speed_reduction_mps=0., hold_s=20.))


def controlled_wrapper():
    p = proposal()
    obj = ReportDissolutionPolicy.__new__(ReportDissolutionPolicy)
    obj.job = dict(arm="wall", decision_time_s=0)
    obj.proposal, obj.started_at, obj.terminal = p, None, None
    obj.base = lambda _: PolicyOutput({}, "release", "identical_damping_requests_native_explicit_handback",
        p.member_ids, p.fingerprint, "original", {})
    return obj


def context(now=1., rows=None, numeric=False):
    return PolicyInput(now, tuple(rows or packets(now)), (), {"a": -.5, "b": -.5} if numeric else {},
                       frozenset() if numeric else frozenset({"a", "b"}))


def test_envelope_uses_acquired_geometry_and_fixed_physical_units():
    value = downstream_envelope(proposal(), packets(), now_s=0.)
    assert value["target_speed_mps"] == pytest.approx((3**2+2*1.5*(177-5-5-24))**.5)
    assert value["target_speed_mps"] < 24.
    assert not value["safety_guarantee"]
    assert downstream_envelope(proposal(), packets(gap=208, downstream_speed=27), now_s=0.)["target_speed_mps"] > 24.


def test_no_future_stale_cross_lane_or_missing_information_as_clear_road():
    with pytest.raises(ValueError, match="future"):
        downstream_envelope(proposal(), packets(1.), now_s=0.)
    assert not downstream_envelope(proposal(), packets(), now_s=2.)["available"]
    rows = [r for r in packets() if r.frame.vehicle_id in {"a", "b"}]
    assert not downstream_envelope(proposal(), rows, now_s=0.)["available"]
    rows += [replace(r, frame=replace(r.frame, lane_id="Other")) for r in packets() if r.frame.vehicle_id.startswith("down")]
    assert not downstream_envelope(proposal(), rows, now_s=0.)["available"]


def test_no_report_constraint_preserves_actual_native_handback():
    wrapper = controlled_wrapper()
    result = wrapper(context(rows=packets(1., gap=208., downstream_speed=27.)))
    assert result.phase == "release" and not result.wall_commands
    assert result.reason == "identical_damping_requests_native_explicit_handback"


def test_bridge_is_wall_owned_and_releases_as_numeric_local_feedback_returns():
    wrapper = controlled_wrapper()
    result = wrapper(context())
    assert result.phase == "taper" and set(result.wall_commands) == {"a", "b"}
    assert all(-3 <= a < 0 for a in result.wall_commands.values())
    assert result.details["dissolution"]["continued_wall_ownership"]
    assert not result.details["dissolution"]["override_released"]
    released = wrapper(context(2., numeric=True))
    assert released.phase == "release" and not released.wall_commands
    assert released.reason == "report_bridge_local_numeric_feedback_available"
    assert not wrapper(context(3.)).wall_commands  # No re-engagement after release.


def test_timeout_evidence_loss_and_rear_guard_release_whole_cohort():
    wrapper = controlled_wrapper()
    assert wrapper(context()).wall_commands
    assert not wrapper(context(11.)).wall_commands
    wrapper = controlled_wrapper()
    wrapper(context())
    rows = [r for r in packets(2.) if r.frame.vehicle_id in {"a", "b"}]
    assert wrapper(context(2., rows=rows)).reason == "report_bridge_evidence_unavailable"
    wrapper = controlled_wrapper()
    result = wrapper(context(rows=packets(1., rear_gap=5.)))
    assert result.phase == "abort" and not result.wall_commands


def test_current_local_reference_loss_releases_without_using_cached_report():
    wrapper = controlled_wrapper()
    wrapper(context())
    missing = SimpleNamespace(now_s=2., reports=packets(2.), damping_commands={}, damping_native_ids={"a"})
    result = wrapper(missing)
    assert result.phase == "abort" and not result.wall_commands


def test_actual_formation_assembly_and_hold_are_unchanged_until_native_release():
    spec = dict(profile=asdict(proposal().profile), member_ids=["a", "b"], required_lanes=["L0", "L1"])
    job = dict(arm="wall", decision_time_s=0, policy_spec=spec)
    registration = {"registration_sha256": "synthetic_explicit_assignment"}
    original = fixed_formation_policy(job, registration)
    revised = ReportDissolutionPolicy(job, registration)
    for now in range(22):
        observed = context(float(now))
        assert revised(observed) == original(observed)
    observed = context(22.)
    assert original(observed).phase == "release"
    assert revised(observed).phase == "taper"
    assert revised(context(23., numeric=True)).phase == "release"
