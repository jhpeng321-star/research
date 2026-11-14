"""Synthetic evidence regressions, not physical efficacy evidence."""
from copy import deepcopy
import gzip
import importlib
import json
from pathlib import Path

import pytest

from traffic_shepherd.paired_experiments import digest
from tests.test_lifecycle_study import finite_fixture


def load(name):
    return importlib.import_module("traffic_shepherd."+name)


evidence, duration = load("duration_evidence"), load("duration_study")


def write(path, values):
    path.write_text("".join(json.dumps(row)+"\n" for row in values))


@pytest.fixture
def branch_fixture(tmp_path):
    path = tmp_path / "fixture"
    job, formation, commands, policies = finite_fixture(path, actual_hold=5)
    prior = {"a":0., "b":0.}
    spec = dict(version=duration.VERSION, hold_offset_s=5., expected_hold_start_s=3., branch_time_s=8.,
        branch_evidence_sha256="b"*64, previous_wall_commands_sha256=digest(prior),
        proposal_sha256="a"*64, source_manifest_sha256="c"*64)
    job["policy_spec"]["duration_branch"] = spec
    row = policies[-1]
    row.update(proposal_sha256=spec["proposal_sha256"], early_exit_reason=None)
    row["duration_branch"] = dict(version=duration.VERSION, hold_offset_s=5., registered_hold_start_s=3.,
        registered_branch_time_s=8., original_profile_hold_s=20., deliberate_release=True,
        feasibility_triggered=False, requested=True, verified=True, request_time_s=8., actual_hold_start_s=3.,
        exit_started_at_s=8., continuation_feasible=True, exit_now_feasible=True,
        proposal_sha256=spec["proposal_sha256"], branch_evidence_sha256=spec["branch_evidence_sha256"],
        source_manifest_sha256=spec["source_manifest_sha256"], previous_wall_commands=prior,
        previous_wall_commands_sha256=spec["previous_wall_commands_sha256"],
        continuation_plan=dict(feasible=True, remaining_hold_s=15, remaining_assembly_s=0),
        exit_now_plan=dict(feasible=True, remaining_hold_s=0, remaining_assembly_s=0))
    write(path / "policy.jsonl", policies)
    return path, job, formation, commands, policies


def test_deliberate_release_passes_without_fabricating_dynamic_constraint(branch_fixture):
    path, job, *_ = branch_fixture
    result = evidence.score_release(path, job)
    assert result["duration_deliberate_release_verified"]
    assert result["registered_maneuver_delivered"]
    assert result["hold_seconds"] == 5
    assert not result["finite_policy_early_exit"]
    assert not result["early_exit_evidence_verified"]
    assert not result["full_nominal_hold_delivered"]


@pytest.mark.parametrize("failure", ["fake_constraint", "source", "prior", "not_requested", "infeasible", "time"])
def test_deliberate_branch_requires_every_bound_current_proof(branch_fixture, failure):
    path, job, _, _, policies = branch_fixture
    at = policies[-1]
    if failure == "fake_constraint":
        at["early_exit_reason"] = "invented_dynamic_constraint"
    elif failure == "source":
        at["duration_branch"]["source_manifest_sha256"] = "d"*64
    elif failure == "prior":
        at["duration_branch"]["previous_wall_commands"]["a"] = .1
    elif failure == "not_requested":
        at["duration_branch"]["requested"] = False
    elif failure == "infeasible":
        at["duration_branch"]["exit_now_plan"]["feasible"] = False
    else:
        at["duration_branch"]["request_time_s"] = 9.
    write(path / "policy.jsonl", policies)
    assert not evidence.score_release(path, job)["duration_deliberate_release_verified"]


def test_valid_assignment_does_not_hide_failed_physical_coast(branch_fixture):
    path, job, formation, _, _ = branch_fixture
    formation[-1]["members"][0]["acceleration_mps2"] = -.3
    write(path / "formation.jsonl", formation)
    result = evidence.score_release(path, job)
    assert result["duration_deliberate_release_verified"]
    assert not result["registered_maneuver_delivered"]


def test_valid_active_hold_but_failed_source_exit_is_zero_allocation(monkeypatch):
    monkeypatch.setattr(evidence, "score_lifecycle", lambda *args: dict(committed_hold_at_s=612.,
        assembly_complete=True, registered_maneuver_delivered=False))
    result = evidence.opportunity("not_read", {}, {})
    assert not result["eligible"]
    assert result["reason"] == "source_maneuver_not_delivered"


@pytest.fixture
def prefix_files(tmp_path):
    header = dict(kind="header", sample_timing="initial_then_post_physics_before_next_control_or_shutdown")
    samples = [dict(kind="sample", time_s=n*.5, vehicles=[dict(vehicle_id="x", speed_mps=20.+n)]) for n in range(7)]
    with gzip.open(tmp_path / "interactions.jsonl.gz", "wt") as stream:
        for row in [header, *samples]:
            stream.write(json.dumps(row)+"\n")
    # No packets are required to hash an empty acquired interval; command masks are separate.
    commands = [dict(kind="command", timestamp_s=float(t), vehicle_id="x", value=t) for t in range(4)]
    write(tmp_path / "telemetry.jsonl", commands)
    write(tmp_path / "intervals.jsonl", [dict(start_s=float(t), end_s=float(t+1), measured=t) for t in range(3)])
    return tmp_path, header, samples, commands


def test_branch_prefix_includes_halfsecond_kinematics_and_excludes_branch_command(prefix_files):
    path, _, _, commands = prefix_files
    first = evidence.branch_prefix(path, 2.)
    assert first["diagnostic_sample_count"] == 5
    assert first["issued_commands_strictly_before_branch"]["count"] == 2
    assert first["acquired_reports_inclusive_branch"]["count"] == 3
    assert not first["private_simulator_state_reconstructed"]
    commands[2]["value"] = "intended branch divergence"
    write(path / "telemetry.jsonl", commands)
    assert evidence.branch_prefix(path, 2.) == first
    commands[1]["value"] = "invalid prior divergence"
    write(path / "telemetry.jsonl", commands)
    assert evidence.branch_prefix(path, 2.) != first


def test_branch_kinematic_difference_cannot_hide_in_same_cav_commands(prefix_files):
    path, header, samples, _ = prefix_files
    first = evidence.branch_prefix(path, 2.)
    samples[4]["vehicles"][0]["speed_mps"] = 999.
    with gzip.open(path / "interactions.jsonl.gz", "wt") as stream:
        for row in [header, *samples]:
            stream.write(json.dumps(row)+"\n")
    second = evidence.branch_prefix(path, 2.)
    assert first["offline_measured_kinematics_branch_sha256"] != second["offline_measured_kinematics_branch_sha256"]


def test_missing_halfsecond_measured_sample_rejected(prefix_files):
    path, header, samples, _ = prefix_files
    with gzip.open(path / "interactions.jsonl.gz", "wt") as stream:
        for row in [header, *samples[:2], *samples[3:]]:
            stream.write(json.dumps(row)+"\n")
    with pytest.raises(ValueError, match="noncontiguous"):
        evidence.branch_prefix(path, 2.)


def test_physics_after_branch_does_not_enter_branch_proof(prefix_files):
    path, header, samples, _ = prefix_files
    first = evidence.branch_prefix(path, 2.)
    samples[5]["vehicles"] = [{"unseen_future":"not a branch feature"}]
    with gzip.open(path / "interactions.jsonl.gz", "wt") as stream:
        for row in [header, *samples]:
            stream.write(json.dumps(row)+"\n")
    assert first == evidence.branch_prefix(path, 2.)


def test_duration_difference_is_direct_wall_wall_not_damping_alias():
    from tests.test_escalation_evidence import run
    wm, continuation = run("wall", system=20.)
    rm, release = run("wall", system=21.)
    before = deepcopy((wm, rm))
    left, right = evidence.outcome_costs(release), evidence.outcome_costs(continuation)
    vector = {key:left[key]-right[key] for key in left}
    assert vector["system_seconds_per_request"] == 1.
    assert (wm, rm) == before
    assert wm["controller"] == rm["controller"] == "wall"


def test_full_comparator_uses_true_damping_and_keeps_prebranch_absolute_events(tmp_path, monkeypatch):
    from dataclasses import asdict
    from tests.test_escalation_evidence import run, criteria
    dm, ds = run(system=22.)
    sm, ss = run("wall", system=20.)
    tm, ts = run("wall", system=21.)
    ss["totals"]["collision_involvements"] = 4
    ts["totals"]["collision_involvements"] = 4
    source_path, reference_path = tmp_path / "continue", tmp_path / "damping"
    release_path = tmp_path / "jobs" / "release"
    source_job = dict(id="continue")
    job = dict(id="release", seed=4107, decision_time_s=600, policy_spec=dict(
        duration_branch=dict(branch_time_s=617.), profile=asdict(evidence.LifecycleProfile())))
    reg = dict(criteria=criteria().as_dict(), communication={})
    data = dict(reg, identity=dict(active_source_sha256={"source.py":"fixed"}),
        continuations={"release":dict(path=str(source_path), registration_path="source_reg", job=source_job,
            manifest_sha256="hash", source_bridge={}, opportunity=dict(branch_prefix={}, continued_to_nominal_end=True))},
        references={"release":dict(path=str(reference_path), manifest_sha256="hash")})
    monkeypatch.setattr(evidence.cohort, "_registered_arm", lambda path, *args:
                        (tm, ts) if Path(path) == release_path else (sm, ss))
    monkeypatch.setattr(evidence.cohort, "_registration", lambda *args: reg)
    monkeypatch.setattr(evidence, "sha", lambda *args: "hash")
    monkeypatch.setattr(evidence, "source_bridge", lambda *args: {})
    monkeypatch.setattr(evidence, "branch_prefix", lambda *args: {})
    monkeypatch.setattr(evidence.cohort, "_snapshot", lambda *args: {})
    monkeypatch.setattr(evidence.cohort, "_prefix", lambda *args: None)
    monkeypatch.setattr(evidence, "score_release", lambda *args:
                        dict(duration_deliberate_release_verified=True, registered_maneuver_delivered=True))
    monkeypatch.setattr(evidence, "score_lifecycle", lambda *args: dict(registered_maneuver_delivered=True))
    monkeypatch.setattr(evidence, "command_contract", lambda *args: dict(passed=True))
    monkeypatch.setattr(evidence, "reference_proof", lambda *args: (dm, ds, dict(manifest_sha256="hash")))
    original = evidence.paired_effect
    calls = []
    def observe(control_manifest, control, treatment_manifest, treatment):
        calls.append((control_manifest["controller"], treatment_manifest["controller"]))
        return original(control_manifest, control, treatment_manifest, treatment)
    monkeypatch.setattr(evidence, "paired_effect", observe)
    result = evidence.compare_case(tmp_path, data, job)
    assert calls == [("damping", "wall"), ("damping", "wall")]
    assert result["duration_contrast"]["cost_release_minus_cost_continue"]["system_seconds_per_request"] == 1.
    assert result["benefit"]["label"] == "harmful"  # Four absolute events cannot be cancelled away.
    assert result["duration_contrast"]["absolute_outcomes"]["release"]["collision_involvements"] == 4
    assert sm["controller"] == tm["controller"] == "wall"
