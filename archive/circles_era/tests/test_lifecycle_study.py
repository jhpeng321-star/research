"""Independent physical scoring and immutable study contract regressions."""
from dataclasses import asdict
import json

import pytest

from traffic_shepherd import lifecycle_study as study
from traffic_shepherd.escalation_evidence import BenefitCriteria
from traffic_shepherd.lifecycle_wall import LifecycleProfile
from tests.test_brake_hold_study import lifecycle_fixture, write_rows


def finite_fixture(path, actual_hold=5):
    job, formation, commands = lifecycle_fixture(path, 20)
    terminal = 3 + actual_hold
    formation = [row for row in formation if row["time_s"] <= terminal]
    commands = [row for row in commands if row["timestamp_s"] <= terminal]
    formation[-1].update(phase="release", owner_by_member={v: "native" for v in ("a", "b")}, wall_command_ids=[])
    for row in commands[-2:]:
        row.update(owner="native", released_to_native=True, wall_request_mps2=None,
                   issued_acceleration_mps2=None, issued_target_speed_mps=None)
    write_rows(path / "telemetry.jsonl", commands)
    job["policy_spec"]["profile"] = asdict(LifecycleProfile())
    for row in formation:
        for member in row["members"]:
            member["acceleration_mps2"] = 0.
    write_rows(path / "formation.jsonl", formation)
    policy = [row for row in study.rows(path / "policy.jsonl") if row["time_s"] <= terminal]
    policy[-1]["phase"] = "release"
    policy[3].update(assembly_completed_at_s=3.,
        lifecycle_plan=dict(commitment=dict(feasible=True, remaining_assembly_s=0,
            maintainable_hold_ready_now=True, predicted_hold_start_s=3., remaining_hold_s=20)))
    if actual_hold < 20:
        policy[-1].update(early_exit_reason="reported_traffic_intersects_remaining_maneuver",
            lifecycle_plan=dict(commitment=dict(feasible=False), exit_now=dict(feasible=True)))
    write_rows(path / "policy.jsonl", policy)
    return job, formation, commands, policy


def test_valid_early_exit_is_not_a_completed_twenty_second_hold(tmp_path):
    job, _, _, _ = finite_fixture(tmp_path / "run")
    result = study.score_lifecycle(tmp_path / "run", job)
    assert result["registered_maneuver_delivered"]
    assert result["finite_policy_early_exit"]
    assert result["measured_coast_terminal"]
    assert not result["full_nominal_hold_delivered"]
    assert not result["fixed_maximum_hold_maneuver_delivered"]
    assert result["hold_seconds"] == 5


def test_full_hold_still_requires_real_zero_command_terminal(tmp_path):
    path = tmp_path / "run"
    job, _, commands, _ = finite_fixture(path, actual_hold=20)
    assert study.score_lifecycle(path, job)["registered_maneuver_delivered"]
    commands[-3]["issued_acceleration_mps2"] = -.15
    write_rows(path / "telemetry.jsonl", commands)
    result = study.score_lifecycle(path, job)
    assert result["full_nominal_hold_delivered"]
    assert not result["measured_coast_terminal"]
    assert not result["registered_maneuver_delivered"]


def test_measured_acceleration_must_satisfy_terminal_condition(tmp_path):
    path = tmp_path / "run"
    job, formation, _, _ = finite_fixture(path)
    formation[-1]["members"][0]["acceleration_mps2"] = -.3
    write_rows(path / "formation.jsonl", formation)
    assert not study.score_lifecycle(path, job)["registered_maneuver_delivered"]


def test_early_exit_requires_bound_continuation_and_exit_evidence(tmp_path):
    path = tmp_path / "run"
    job, _, _, policy = finite_fixture(path)
    policy[-1]["lifecycle_plan"]["exit_now"]["feasible"] = False
    write_rows(path / "policy.jsonl", policy)
    result = study.score_lifecycle(path, job)
    assert result["controlled_exit_verified"]
    assert not result["early_exit_evidence_verified"]
    assert not result["registered_maneuver_delivered"]


def test_last_half_second_of_early_hold_must_remain_compact(tmp_path):
    path = tmp_path / "run"
    job, formation, _, _ = finite_fixture(path)
    formation[-1]["members"][1]["distance_m"] += 100
    write_rows(path / "formation.jsonl", formation)
    assert not study.score_lifecycle(path, job)["registered_maneuver_delivered"]


def test_hold_may_follow_later_maintainable_readiness(tmp_path):
    path = tmp_path / "run"
    job, formation, _, policy = finite_fixture(path, actual_hold=20)
    for row in formation:
        if 3 <= row["time_s"] < 6:
            row["phase"] = "assembly"
    for row in policy:
        if 3 <= row["time_s"] < 6:
            row["phase"] = "assembly"
    policy[6].update(assembly_completed_at_s=6.,
        lifecycle_plan=dict(commitment=dict(feasible=True, remaining_assembly_s=0,
            maintainable_hold_ready_now=True, predicted_hold_start_s=6., remaining_hold_s=20)))
    policy[-1].update(early_exit_reason="reported_traffic_intersects_remaining_maneuver",
        lifecycle_plan=dict(commitment=dict(feasible=False), exit_now=dict(feasible=True)))
    write_rows(path / "formation.jsonl", formation)
    write_rows(path / "policy.jsonl", policy)
    result = study.score_lifecycle(path, job)
    assert result["measured_ready_at_s"] == 3
    assert result["committed_hold_at_s"] == 6
    assert result["registered_maneuver_delivered"]
    formation[10]["members"][0]["speed_mps"] = 25.
    write_rows(path / "formation.jsonl", formation)
    assert not study.score_lifecycle(path, job)["registered_maneuver_delivered"]


def test_measured_readiness_alone_is_not_maintainable_commitment(tmp_path):
    path = tmp_path / "run"
    job, _, _, policy = finite_fixture(path)
    policy[3]["lifecycle_plan"]["commitment"]["feasible"] = False
    write_rows(path / "policy.jsonl", policy)
    assert not study.score_lifecycle(path, job)["registered_maneuver_delivered"]


def effect(gain=0., service=0., collisions=0):
    return dict(control="damping", treatment="wall", excess_collisions=collisions,
        excess_teleports=0, excess_unfinished=0, control_unfinished_requested=0, treatment_unfinished_requested=0,
        improvements=dict(braking_seconds_per_request=gain, effort_per_request=gain,
        system_seconds_per_request=service, ttc_seconds_per_request=gain, source_wait_seconds_per_request=gain),
        complete_requested_service_control=True, complete_requested_service_treatment=True,
        treatment_collision_involvements=collisions, treatment_starting_teleports=0,
        control_collision_involvements=0, control_starting_teleports=0)


def test_joint_success_requires_materiality_and_zero_safety_events():
    criteria = BenefitCriteria(
        primary_minimum_improvement={"system_seconds_per_request": 1., "braking_seconds_per_request": .02},
        maximum_cost_increase={"system_seconds_per_request": .5, "source_wait_seconds_per_request": .1,
            "braking_seconds_per_request": .005, "effort_per_request": .5, "ttc_seconds_per_request": .02})
    assert not study.joint_non_deterioration(effect(), criteria)["joint_material_success"]
    assert study.joint_non_deterioration(effect(.1, 2), criteria)["joint_material_success"]
    assert not study.joint_non_deterioration(effect(.1, 2, 1), criteria)["joint_material_success"]
    result = study.joint_non_deterioration(effect(-.001, 2), criteria)
    assert result["material_primary_improvement"]
    assert not result["all_continuous_costs_nonworsening"]
    assert result["original_label_unchanged"]


def test_part_a_must_select_current_network(tmp_path, monkeypatch):
    path = tmp_path / "part_a.json"
    monkeypatch.setattr(study, "PART_A", path)
    payload = dict(decision="retain_original_network_and_current_control_region_for_part_B",
        extension_required_for_part_B=False, simulations_started=0,
        source_sha256={"D:/research/highway_merge_shepherd_long/osm.net.xml.gz": "abc"})
    path.write_text(json.dumps(payload))
    study.verify_part_a(dict(network_sha256="abc"))
    with pytest.raises(ValueError, match="unchanged-network"):
        study.verify_part_a(dict(network_sha256="other"))


def test_jobs_are_exact_two_existing_development_cases():
    jobs = study.jobs()
    assert {j["seed"] for j in jobs} == {4107, 4108}
    assert len(jobs) == 2
    assert all(j["duration_s"] == 2400 and j["decision_time_s"] == 600 for j in jobs)
    assert all(j["policy_spec"]["profile"] == asdict(LifecycleProfile()) for j in jobs)
