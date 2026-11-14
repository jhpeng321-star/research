"""Factorial policy tests use acquired synthetic packets, never SUMO."""
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from traffic_shepherd.brake_hold_policy import (
    BrakeHoldPolicy, BrakeHoldProfile, VERSION, policy_factory,
)
from traffic_shepherd.formation import make_proposal
from traffic_shepherd.gentle_wall import (
    GentleProfile, GentleWallPolicy, gentle_envelope, ramped_braking_distance, resolve_command,
)
from traffic_shepherd.paired_experiments import PolicyInput
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import SensorFrame


def packets(now=0., *, speed=24., acceleration=0.):
    return tuple(CAVReport(SensorFrame(v, now, speed, acceleration, 40., i,
        distance_to_merge_m=3000., front=None, rear=None, front_coverage_m=100.,
        rear_coverage_m=100., lane_id=f"L{i}", position_m=1000.), now, "damping")
        for i, v in enumerate(("a", "b")))


def job(profile=None, *, decision=0):
    return dict(arm="wall", decision_time_s=decision, policy_spec=dict(
        member_ids=["a", "b"], required_lanes=["L0", "L1"],
        profile=asdict(profile or BrakeHoldProfile())))


def context(now, *, speed=24., acceleration=0., rows=None, numeric=None):
    rows = packets(now, speed=speed, acceleration=acceleration) if rows is None else rows
    native = frozenset(r.frame.vehicle_id for r in rows) if numeric is None else frozenset()
    return PolicyInput(float(now), rows, (), numeric or {}, native)


def physical_payload(output):
    value = deepcopy(output.payload())
    value.pop("proposal_sha256")
    value.pop("brake_hold_study_version", None)
    value.pop("brake_hold_factors", None)
    if "assessment" in value:
        value["assessment"].pop("proposal_fingerprint")
    return value


@pytest.mark.parametrize("minimum", [-.3, -3.])
@pytest.mark.parametrize("hold", [0., 20., 40.])
def test_only_registered_factor_combinations_keep_canonical_fields(minimum, hold):
    profile = BrakeHoldProfile(acceleration_min_mps2=minimum, hold_s=hold)
    actual, expected = asdict(profile), asdict(GentleProfile())
    for field in ("version", "acceleration_min_mps2", "hold_s"):
        actual.pop(field)
        expected.pop(field)
    assert actual == expected
    assert profile.version == VERSION


@pytest.mark.parametrize("changes", [
    {"acceleration_min_mps2": -1.}, {"hold_s": 10.}, {"hold_s": False},
    {"acceleration_change_limit_mps3": .3}, {"control_dt_s": .5},
    {"position_gain_per_s": .2}, {"assembly_deadline_s": 60.},
    {"taper_s": 0.}, {"minimum_front_ttc_s": 2.}, {"readiness_samples": 2},
    {"readiness_samples": 3.0},
    {"version": "gentle_report_wall/1"}, {"acceleration_max_mps2": 3.},
])
def test_rejects_extra_factors(changes):
    with pytest.raises(ValueError):
        BrakeHoldProfile(**changes)


def test_same_gentle_commands_and_physical_details_through_native_release():
    old = GentleWallPolicy(job(GentleProfile()), {})
    new = policy_factory(job(), {})
    speed, acceleration = 24., 0.
    phases = set()
    for now in range(80):
        acquired = context(now, speed=speed, acceleration=acceleration)
        reference, actual = old(acquired), new(acquired)
        assert physical_payload(actual) == physical_payload(reference)
        phases.add(actual.phase)
        if actual.wall_commands:
            acceleration = actual.wall_commands["a"]
            speed += acceleration
        elif actual.phase == "release":
            assert not new(context(now+1, speed=speed)).wall_commands
            break
    assert {"assembly", "hold", "release"} <= phases
    assert new.proposal.fingerprint != old.proposal.fingerprint


@pytest.mark.parametrize("minimum", [-.3, -3.])
@pytest.mark.parametrize("hold", [0., 20., 40.])
def test_hold_clock_starts_only_after_three_settled_samples(minimum, hold):
    obj = BrakeHoldPolicy(job(BrakeHoldProfile(acceleration_min_mps2=minimum, hold_s=hold)), {})
    assert obj(context(0)).phase == "assembly"  # Target freezes at 22 m/s.
    outputs = {}
    for now in range(1, int(hold)+5):
        outputs[now] = obj(context(now, speed=22.))
    assert obj.ready_at == 3. and obj.taper_at == 3.+hold
    assert sum(o.phase == "hold" for o in outputs.values()) == hold
    assert outputs[3+int(hold)].phase == "release"
    assert all(o.phase == "assembly" for t, o in outputs.items() if t < 3)


@pytest.mark.parametrize("minimum", [-.3, -3.])
def test_shared_slew_and_absolute_authority_are_applied_after_servo(minimum):
    p, prior = BrakeHoldProfile(acceleration_min_mps2=minimum), None
    for _ in range(22):
        frame = packets(speed=30., acceleration=prior or 0.)[0].frame
        command, evidence = resolve_command(frame, -30., p, previous=prior)
        assert minimum <= command <= 1.5
        assert abs(command-(prior or 0.)) <= .15+1e-10
        assert evidence["requested_acceleration_mps2"] == -30.
        prior = command
    assert prior == minimum
    reversed_command, _ = resolve_command(replace(frame, acceleration_mps2=prior), 30., p, previous=prior)
    assert reversed_command == pytest.approx(prior+.15)


def test_selected_braking_authority_enters_same_ramped_envelope():
    rows = packets()
    downstream = tuple(replace(r, frame=replace(r.frame, vehicle_id="down"+r.frame.vehicle_id,
        speed_mps=3., position_m=1400., distance_to_merge_m=2600.)) for r in rows)
    evidence = {}
    for minimum in (-.3, -3.):
        p = BrakeHoldProfile(acceleration_min_mps2=minimum)
        proposal = make_proposal(rows, now_s=0., required_lanes=("L0", "L1"),
                                 member_ids=("a", "b"), profile=p)
        evidence[minimum] = gentle_envelope(proposal, rows+downstream, now_s=0., previous_commands={})
        assert evidence[minimum]["comfortable_deceleration_mps2"] == -minimum
        assert evidence[minimum]["acceleration_change_limit_mps3"] == .15
    assert not evidence[-.3]["gentle_stopping_feasible"]
    assert evidence[-3.]["gentle_stopping_feasible"]
    assert ramped_braking_distance(24., 3., 0., BrakeHoldProfile())[0] > ramped_braking_distance(
        24., 3., 0., BrakeHoldProfile(acceleration_min_mps2=-3.))[0]


@pytest.mark.parametrize("minimum", [-.3, -3.])
def test_protective_handover_and_member_loss_are_terminal(minimum):
    for loss in (False, True):
        obj = BrakeHoldPolicy(job(BrakeHoldProfile(acceleration_min_mps2=minimum)), {})
        assert obj(context(0)).wall_commands
        obj.ready_at, obj.taper_at = 0., 1.
        acquired = (context(1, rows=packets(1)[:1]) if loss else
                    context(1, speed=22., numeric={"a": minimum-.1, "b": 0.}))
        output = obj(acquired)
        assert output.phase == ("abort" if loss else "release")
        assert not output.wall_commands
        assert not obj(context(2)).wall_commands


def test_incompatible_measured_response_aborts_without_fabricated_reference():
    obj = BrakeHoldPolicy(job(), {})
    assert obj(context(0)).wall_commands
    result = obj(context(1, acceleration=-.80218868849))
    assert result.phase == "abort" and not result.wall_commands
    assert "response_incompatible" in result.reason


def test_namespace_serializes_with_evaluator_and_preserves_clock_refusal():
    obj = BrakeHoldPolicy(job(decision=1), {})
    out = obj(context(0))
    row = dict(time_s=0., assignment="registered_experiment", **out.payload())
    assert row["brake_hold_study_version"] == VERSION
    assert row["brake_hold_factors"]["hold_s"] == 20.
    assert row["phase"] == "damping" and not row["wall_commands"]
    with pytest.raises(ValueError, match="late"):
        obj(context(2))
    refused = BrakeHoldPolicy(job(), {})
    assert refused(context(0, rows=packets()[:1])).phase == "refused"
    assert refused(context(1)).phase == "refused"
    with pytest.raises(ValueError, match="one-second"):
        refused(context(3))


def test_caller_profile_mutation_cannot_add_an_unregistered_factor():
    original = job()
    obj = BrakeHoldPolicy(original, {})
    original["policy_spec"]["profile"]["hold_s"] = 10.
    assert obj(context(0)).details["brake_hold_factors"]["hold_s"] == 20.
