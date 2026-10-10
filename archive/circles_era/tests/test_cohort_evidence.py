"""Synthetic integrity regressions only; these do not establish physical efficacy."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from traffic_shepherd.damping import CirclesController
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.paired_experiments import PolicyOutput
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame

from traffic_shepherd import cohort_evidence as evidence
COMM = dict(delay_s=0., max_report_age_s=1., history_s=20.)


def dump(path, values):
    path.write_text("".join(json.dumps(row)+"\n" for row in values), encoding="utf-8")


def policy(context):
    now = context.now_s
    if now < 2:
        return PolicyOutput({"a": .1 if now == 0 else 0.}, "assembly" if now == 0 else "hold",
                            "test", ("a",), "proposal", "evidence", {"test_context_time": now})
    return PolicyOutput({}, "release", "test", ("a",), "proposal", "evidence", {"test_context_time": now})


def selected(context):
    result = policy(context)
    return PolicyOutput(result.wall_commands, result.phase, result.reason, result.member_ids,
        result.proposal_sha256, result.evidence_sha256,
        dict(result.details, cohort_selector="test", cohort_selection_verified=True))


@pytest.fixture
def saved(tmp_path):
    controller, api = CirclesController(), evidence._ReplayAPI()
    owner = CommandOwnership(api, {"a", "b"}, controller)
    telemetry, policies, previous = [], [], {}
    for tick in range(5):
        now = float(tick)
        frames = {}
        for vid in ("a", "b"):
            if vid == "b" and tick == 2:
                continue  # Live CAV with no current acquired observation.
            front = NeighborMeasurement(30., 0.) if tick in (0, 1, 3) else None
            frame = SensorFrame(vid, now, 20., 0., 30., 0,
                distance_to_merge_m=2000., front=front, front_coverage_m=100., rear_coverage_m=100.,
                lane_id="road_0", position_m=500.)
            frames[vid] = frame
            prior = previous.get(vid)
            packet = CAVReport(frame, now, "native" if prior is None else prior.owner,
                None if prior is None else prior.network_output_mps2,
                None if prior is None else prior.issued_acceleration_mps2,
                None if prior is None else prior.fallback_reason)
            telemetry.append(dict(kind="rsu_received_report", received_at_s=now, **packet.payload()))
        result = policy(SimpleNamespace(now_s=now))
        policies.append(dict(time_s=now, assignment="synthetic", **result.payload()))
        commands = owner.step(frames, now_s=now, live_ids={"a", "b"}, wall_commands=result.wall_commands)
        for record in commands:
            telemetry.append(dict(kind="command", **record.as_dict()))
        previous = {record.vehicle_id: record for record in commands}
        telemetry.append(dict(kind="response", timestamp_s=now+.5, hidden_future_noise=12345))
    dump(tmp_path / "telemetry.jsonl", telemetry)
    dump(tmp_path / "policy.jsonl", policies)
    return tmp_path


def test_exact_real_circles_owner_replay_covers_native_release_numeric_and_missing(saved):
    result = evidence._replay_stream(saved, policy, selected, COMM, duration=5)
    assert result["decision_count"] == 5
    assert (result["first_decision_s"], result["last_decision_s"]) == (0., 4.)
    assert result["actual_native_release_writes"] >= 3
    assert result["damping_records"] >= 1 and result["wall_records"] == 2
    assert result["missing_observation_live_decisions"] == 1
    assert result["command_records"] == 10


@pytest.mark.parametrize("key,value", [("released_to_native", False), ("issued_target_speed_mps", 1.),
    ("actor_input_si", [0., 0., 0.]), ("fallback_reason", "wrong")])
def test_every_actual_command_field_is_compared(saved, key, value):
    values = list(evidence.rows(saved / "telemetry.jsonl"))
    target = next(row for row in values if row["kind"] == "command"
                  and row["timestamp_s"] == 2. and row["vehicle_id"] == "a")
    target[key] = value
    dump(saved / "telemetry.jsonl", values)
    with pytest.raises(ValueError, match="command record differs"):
        evidence._replay_stream(saved, policy, selected, COMM, duration=5)


def test_core_policy_metadata_cannot_silently_disagree(saved):
    def bad(context):
        output = selected(context)
        return PolicyOutput(output.wall_commands, output.phase, output.reason, output.member_ids,
            output.proposal_sha256, output.evidence_sha256, dict(output.details, unknown_diagnostic=1))
    with pytest.raises(ValueError, match="core differs"):
        evidence._replay_stream(saved, policy, bad, COMM, duration=5)


@pytest.mark.parametrize("mutation", ["duplicate", "future", "prior_owner", "late"])
def test_report_packet_integrity_and_causality(saved, mutation):
    values = list(evidence.rows(saved / "telemetry.jsonl"))
    index = next(i for i, row in enumerate(values) if row["kind"] == "rsu_received_report"
                 and row["received_at_s"] == 1. and row["frame"]["vehicle_id"] == "a")
    if mutation == "duplicate":
        values.insert(index, deepcopy(values[index]))
    elif mutation == "future":
        values[index]["frame"]["timestamp_s"] = 2.
    elif mutation == "late":
        values[index]["frame"]["timestamp_s"] = 0.
    else:
        values[index]["owner"] = "native"
    dump(saved / "telemetry.jsonl", values)
    with pytest.raises(ValueError):
        evidence._replay_stream(saved, policy, selected, COMM, duration=5)


def test_truncated_policy_cannot_be_complete(saved):
    values = list(evidence.rows(saved / "policy.jsonl"))
    dump(saved / "policy.jsonl", values[:-1])
    with pytest.raises(ValueError, match="policy clock"):
        evidence._replay_stream(saved, policy, selected, COMM, duration=5)


def test_future_physical_noise_never_enters_policy(saved):
    def inspect(context):
        assert set(vars(context)) == {"now_s", "reports", "history", "damping_commands", "damping_native_ids"}
        assert all(packet.frame.timestamp_s <= context.now_s for packet in context.reports)
        assert all(receipt <= context.now_s for receipt, _ in context.history)
        return selected(context)
    assert evidence._replay_stream(saved, policy, inspect, COMM, duration=5)["decision_count"] == 5


def test_source_registration_snapshot_must_match_immutable_digest(tmp_path):
    source = tmp_path / "source_snapshot"
    source.mkdir()
    (source / "policy.py").write_text("original")
    data = dict(identity=dict(active_source_sha256={"policy.py": evidence.sha(source / "policy.py")}))
    data["registration_sha256"] = evidence.paired.digest(data)
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(data))
    assert evidence._registration(path) == data
    (source / "policy.py").write_text("changed")
    with pytest.raises(ValueError, match="snapshot changed"):
        evidence._registration(path)


def test_registered_arm_checks_exact_job_and_communication(monkeypatch):
    job = {field: field for field in evidence.JOB_FIELDS.values()}
    job["duration_s"] = 2400
    identity = dict(config_sha256="frozen")
    reg = dict(jobs=[job], identity=identity, registration_sha256="r", communication=COMM)
    manifest = {key: job[field] for key, field in evidence.JOB_FIELDS.items()}
    manifest.update(identity, registration_sha256="r", mechanics_only=False, communication=COMM)
    monkeypatch.setattr(evidence.paired, "verified_arm", lambda _: (manifest, {}))
    evidence._registered_arm("unused", job, reg)
    manifest["policy_spec"] = "different"
    with pytest.raises(ValueError, match="registered job"):
        evidence._registered_arm("unused", job, reg)


def test_prefix_includes_report_and_command_histories():
    left = {key: key for key in evidence.PREFIX_KEYS}
    evidence._prefix(left, left)
    for key in evidence.PREFIX_KEYS:
        right = dict(left, **{key: "changed"})
        with pytest.raises(ValueError, match="prefix changed"):
            evidence._prefix(left, right)


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    from dataclasses import asdict
    from tests.test_escalation_evidence import run, criteria
    dm, ds = run()
    bm, bs = run("wall", system=21.)
    tm, ts = run("wall", system=22.)
    dp, bp, cp = tmp_path / "d", tmp_path / "b", tmp_path / "jobs" / "c"
    for path in (dp, bp, cp):
        path.mkdir(parents=True)
    snapshot = dict(decision_time_s=600., predecision_state_sha256="same-prefix",
        predecision_report_evidence_sha256="reports", predecision_report_prefix_sha256="report-history",
        predecision_command_prefix_sha256="command-history", proposal_sha256="same-proposed-wall")
    selection = dict(selection_sha256="selection", selected=dict(proposal_sha256="same-proposed-wall"))
    bjob = dict(id="b", seed=4107, decision_time_s=600,
                policy_spec=dict(profile=asdict(evidence.LifecycleProfile())))
    cjob = dict(bjob, id="c")
    bproof = dict(job=bjob, source_bridge={}, path=str(bp), manifest_sha256="hash")
    br = dict(criteria=criteria().as_dict(), communication=COMM)
    data = dict(jobs=[cjob], identity=dict(active_source_sha256={"source.py": "fixed"}),
        criteria=br["criteria"], communication=COMM, references={"c":dict(path=str(dp), manifest_sha256="hash")},
        bound_snapshots={"c":snapshot}, selections={"c":selection},
        bindings={"c":dict(mode="simulate", path=str(cp), source_job_id="b", source_manifest_sha256="hash")})
    monkeypatch.setattr(evidence, "_preceding", lambda *args: (bp, bm, bs, br, bproof))
    monkeypatch.setattr(evidence, "reference_proof", lambda *args: (dm, ds, dict(manifest_sha256="hash")))
    snapshots = {dp:deepcopy(snapshot), bp:deepcopy(snapshot), cp:deepcopy(snapshot)}
    monkeypatch.setattr(evidence, "_snapshot", lambda p: snapshots[Path(p)])
    monkeypatch.setattr(evidence, "_cohort", lambda: SimpleNamespace(snapshot_selection=lambda s: selection,
        bind_selected_job=lambda *args: (cjob, selection)))
    monkeypatch.setattr(evidence, "_registered_arm", lambda *args: (tm, ts))
    monkeypatch.setattr(evidence, "sha", lambda *args: "hash")
    monkeypatch.setattr(evidence, "command_contract", lambda *args: dict(passed=True))
    monkeypatch.setattr(evidence, "score_lifecycle", lambda *args: dict(registered_maneuver_delivered=False))
    rows = [dict(time_s=600., phase="assembly", reason="test", lifecycle_plan={},
        cohort_selection_verified=True, cohort_selection_sha256="selection", cohort_selection=selection),
        dict(time_s=602., phase="release", reason="test", lifecycle_plan={})]
    monkeypatch.setattr(evidence, "rows", lambda *args: iter(rows))
    return SimpleNamespace(root=tmp_path, data=data, job=cjob, bm=bm, tm=tm, snapshots=snapshots,
                           dp=dp, bp=bp, cp=cp, policy_rows=rows)


def test_comparison_retains_harm_failed_delivery_and_c_minus_b(comparison):
    f = comparison
    result = evidence.compare_case(f.root, f.data, f.job)
    assert result["benefit"]["label"] == "harmful"
    assert not result["lifecycle"]["registered_maneuver_delivered"]
    assert result["preceding_b"]["cohort_minus_preceding_cost_per_request"]["system_seconds_per_request"] == 1.
    assert result["new_simulated_seconds"] == 2400
    assert not result["changed_existing_manifests"]


@pytest.mark.parametrize("failure", ["prefix", "future", "assignment", "criteria"])
def test_comparison_rejects_unmatched_inputs_before_any_effect(comparison, monkeypatch, failure):
    f = comparison
    if failure == "prefix":
        f.snapshots[f.cp]["predecision_command_prefix_sha256"] = "changed"
    elif failure == "future":
        f.tm["exogenous_future_sha256"] = "different"
    elif failure == "assignment":
        f.policy_rows[0]["cohort_selection_verified"] = False
    else:
        f.data["criteria"] = {}
    monkeypatch.setattr(evidence, "paired_effect", lambda *args: pytest.fail("derived effect before integrity check"))
    with pytest.raises(ValueError):
        evidence.compare_case(f.root, f.data, f.job)


def test_verified_reuse_has_zero_new_seconds_and_zero_c_minus_b(comparison, monkeypatch):
    f = comparison
    proof = dict(actual_proof="synthetic_test_boundary")
    f.data["bindings"]["c"].update(mode="verified_reuse", path=str(f.bp), reuse_proof=proof)
    monkeypatch.setattr(evidence, "replay_reuse", lambda *args: proof)
    result = evidence.compare_case(f.root, f.data, f.job)
    assert result["new_simulated_seconds"] == 0
    assert set(result["preceding_b"]["cohort_minus_preceding_cost_per_request"].values()) == {0.}
    assert result["reused_trace_is_not_independent"]


def test_reuse_does_not_accept_a_changed_complete_proof(comparison, monkeypatch):
    f = comparison
    f.data["bindings"]["c"].update(mode="verified_reuse", path=str(f.bp), reuse_proof={"complete":True})
    monkeypatch.setattr(evidence, "replay_reuse", lambda *args: {"complete":False})
    with pytest.raises(ValueError, match="replay proof changed"):
        evidence.compare_case(f.root, f.data, f.job)
