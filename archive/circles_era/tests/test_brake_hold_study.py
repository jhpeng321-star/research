"""Saved-evidence and lifecycle regressions; all files and actors are synthetic."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from traffic_shepherd import brake_hold_study as study
from traffic_shepherd.brake_hold_policy import BrakeHoldProfile
from traffic_shepherd.paired_experiments import PolicyOutput
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import SensorFrame


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")


def lifecycle_fixture(path, hold=0):
    path.mkdir()
    terminal = 3 + hold
    policy, formation, commands = [], [], []
    for time in range(terminal+1):
        phase = "assembly" if time < 3 else "hold" if time < terminal else "release"
        policy.append(dict(time_s=float(time), phase=phase, reason="synthetic"))
        for vid in ("a", "b"):
            commands.append(dict(kind="command", timestamp_s=float(time), vehicle_id=vid,
                owner="native" if time == terminal else "wall", released_to_native=time == terminal,
                wall_request_mps2=None if time == terminal else 0.,
                issued_acceleration_mps2=None if time == terminal else 0.,
                issued_target_speed_mps=None if time == terminal else 22.))
    for half in range(2*terminal+1):
        time = half/2
        phase = "assembly" if time < 3 else "hold" if time < terminal else "release"
        formation.append(dict(time_s=time, phase=phase, target_speed_mps=22.,
            owner_by_member={v: "native" if time == terminal else "wall" for v in ("a", "b")},
            wall_command_ids=[] if time == terminal else ["a", "b"],
            members=[dict(id=v, present=True, lane=f"L{i}", distance_m=2000.,
                speed_mps=24. if time == 0. else 22.) for i, v in enumerate(("a", "b"))]))
    write_rows(path/"policy.jsonl", policy)
    write_rows(path/"formation.jsonl", formation)
    write_rows(path/"telemetry.jsonl", commands)
    job = dict(decision_time_s=0, policy_spec=dict(member_ids=["a", "b"],
        required_lanes=["L0", "L1"], profile=asdict(BrakeHoldProfile(hold_s=hold))))
    return job, formation, commands


@pytest.mark.parametrize("hold", [0, 20, 40])
def test_readiness_and_full_hold_include_zero_duration(tmp_path, hold):
    path = tmp_path/"case"
    job, _, _ = lifecycle_fixture(path, hold)
    result = study.lifecycle(path, job)
    assert result["measured_ready_at_s"] == 3.
    assert result["hold_seconds"] == hold
    assert result["registered_maneuver_delivered"]
    assert result["handback"]["verified"]


def test_last_half_second_hold_endpoint_must_remain_compact(tmp_path):
    path = tmp_path/"case"
    job, formation, _ = lifecycle_fixture(path, 20)
    formation[-1]["members"][1]["distance_m"] += 80.
    write_rows(path/"formation.jsonl", formation)
    result = study.lifecycle(path, job)
    assert not result["full_nominal_hold_delivered"]
    assert not result["registered_maneuver_delivered"]


def test_registered_lanes_cannot_be_redefined_by_measured_first_row(tmp_path):
    path = tmp_path/"case"
    job, formation, _ = lifecycle_fixture(path)
    for row in formation:
        row["members"][0]["lane"] = "unregistered_lane"
    write_rows(path/"formation.jsonl", formation)
    result = study.lifecycle(path, job)
    assert not result["assembly_complete"]
    assert not result["registered_maneuver_delivered"]


def test_missing_initial_physics_cannot_manufacture_valid_lifecycle(tmp_path):
    path = tmp_path/"case"
    job, formation, _ = lifecycle_fixture(path)
    write_rows(path/"formation.jsonl", formation[2:])
    with pytest.raises(ValueError, match="(clock|start|decision)"):
        study.lifecycle(path, job)


def test_zero_hold_cannot_enter_taper_before_current_three_sample_readiness(tmp_path):
    path = tmp_path/"case"
    job, formation, _ = lifecycle_fixture(path)
    for row in formation:
        if 1. <= row["time_s"] < 3.:
            row["phase"] = "taper"
    write_rows(path/"formation.jsonl", formation)
    policy = list(study.rows(path/"policy.jsonl"))
    for row in policy:
        if 1. <= row["time_s"] < 3.:
            row["phase"] = "taper"
    write_rows(path/"policy.jsonl", policy)
    result = study.lifecycle(path, job)
    assert not result["registered_maneuver_delivered"]


def test_absent_at_entry_refusal_is_a_retained_failure_not_an_exception(tmp_path):
    path = tmp_path/"case"
    job, _, _ = lifecycle_fixture(path)
    write_rows(path/"policy.jsonl", [dict(time_s=0., phase="refused", reason="missing_member")])
    write_rows(path/"formation.jsonl", [dict(time_s=0., phase="refused", target_speed_mps=None,
        owner_by_member={"a": None, "b": None}, wall_command_ids=[],
        members=[dict(id="a", present=False), dict(id="b", present=False)])])
    write_rows(path/"telemetry.jsonl", [])
    result = study.lifecycle(path, job)
    assert not result["assembly_complete"]
    assert not result["registered_maneuver_delivered"]
    assert result["terminal_phase"] == "refused"


def test_duplicate_actual_command_is_not_silently_deduplicated(tmp_path):
    request = dict(time_s=0., wall_commands={"a": -.15}, command_resolution={"a": dict(
        measured_entry_or_current_acceleration_mps2=0., resolved_wall_acceleration_mps2=-.15)})
    issued = dict(kind="command", timestamp_s=0., vehicle_id="a", owner="wall",
                  issued_acceleration_mps2=-.15)
    write_rows(tmp_path/"policy.jsonl", [request])
    write_rows(tmp_path/"telemetry.jsonl", [issued, issued])
    result = study.command_contract(tmp_path, BrakeHoldProfile())
    assert not result["passed"]


def test_replay_accepts_only_received_packet_schema_and_replays_all_ticks(tmp_path, monkeypatch):
    old = tmp_path/"old"
    path = old/"jobs/s4108_m2700_p20_t600_gentle_wall"
    path.mkdir(parents=True)
    (path/"manifest.json").write_text("{}")
    (old/"registration.json").write_text(json.dumps(dict(communication=dict(max_report_age_s=1., history_s=20.))))
    packet = CAVReport(SensorFrame("a", 0., 24., 0., 30., 0,
        front_coverage_m=100., rear_coverage_m=100.), 0., "native")
    write_rows(path/"telemetry.jsonl", [dict(kind="rsu_received_report", received_at_s=0., **packet.payload())])
    write_rows(path/"policy.jsonl", [dict(time_s=float(t), phase="release", reason="synthetic",
        member_ids=[], wall_commands={}) for t in range(2400)])
    monkeypatch.setattr(study, "OLD", old)
    monkeypatch.setattr(study.paired, "verified_arm", lambda _: (
        dict(telemetry_sha256="synthetic", policy_sha256="synthetic"), {}))
    monkeypatch.setattr(study, "source_bridge", lambda *args: {"synthetic": True})
    calls = []
    class SyntheticController:
        def decide(self, frame, *, now_s):
            calls.append((frame.vehicle_id, now_s))
            return SimpleNamespace(fallback_reason="synthetic_native", command_acceleration_mps2=None)
    monkeypatch.setattr(study, "CirclesController", SyntheticController)
    monkeypatch.setattr(study, "policy_factory", lambda *args: lambda ctx:
        PolicyOutput(phase="release", reason="synthetic"))
    result = study.replay_saved(4108, [{"id": "synthetic_replay"}], {})
    assert result["decisions_replayed"] == 2400
    assert result["all_physical_outputs_exact"]
    assert calls == [("a", 0.)]


def registration_fixture(tmp_path, monkeypatch):
    root = tmp_path/"registration"
    (root/"source_snapshot").mkdir(parents=True)
    source = root/"source_snapshot/synthetic_source.py"
    source.write_text("# Synthetic integrity fixture, no simulator.\n")
    preflight = root/"preflight.json"
    preflight.write_text("{}")
    jobs = [dict(id=study.cell_id(seed, cap, hold), seed=seed,
        policy_spec=dict(profile=asdict(BrakeHoldProfile(acceleration_min_mps2=cap, hold_s=hold))))
        for seed in (4107, 4108) for cap in (-.3, -3.) for hold in (0, 20, 40)]
    monkeypatch.setattr(study, "jobs", lambda: jobs)
    bindings = {}
    for job in jobs:
        binding = dict(mode=study.binding_mode(job))
        if binding["mode"] != "simulate":
            binding["source_path"] = str(study.OLD/"jobs"/f"s{job['seed']}_m2700_p20_t600_gentle_wall")
        bindings[job["id"]] = binding
    data = dict(schema=study.SCHEMA, jobs=jobs, bindings=bindings,
        maximum_attempts=8, maximum_simulated_seconds=19200, automatic_retries=0, automatic_successors=False,
        identity=dict(active_source_sha256={source.name: study.sha(source)}),
        preflight_path=str(preflight), preflight_sha256=study.sha(preflight))
    return root, source, data


def write_registration(root, data):
    data["registration_sha256"] = study.paired.digest({k: v for k, v in data.items() if k != "registration_sha256"})
    path = root/"registration.json"
    path.write_text(json.dumps(data))
    return path


def test_reused_binding_cannot_spend_ninth_allocation_even_with_recomputed_digest(tmp_path, monkeypatch):
    root, _, data = registration_fixture(tmp_path, monkeypatch)
    path = write_registration(root, data)
    study.read_registration(path, require_current=False)
    data["bindings"][study.cell_id(4108, -.3, 0)]["mode"] = "simulate"
    write_registration(root, data)
    with pytest.raises(ValueError, match="allocation mode"):
        study.read_registration(path, require_current=False)


def test_registered_source_snapshot_is_verified_during_offline_analysis(tmp_path, monkeypatch):
    root, source, data = registration_fixture(tmp_path, monkeypatch)
    path = write_registration(root, data)
    source.write_text("# Changed after registration.\n")
    with pytest.raises(ValueError, match="source snapshot"):
        study.read_registration(path, require_current=False)
