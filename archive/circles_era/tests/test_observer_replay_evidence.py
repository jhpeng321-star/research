"""Offline replay rejection and causal acquisition tests; SUMO is mocked."""
import json
from dataclasses import replace
import gzip
import hashlib
from types import SimpleNamespace

import pytest

from traffic_shepherd import observer_replay_evidence as evidence
from traffic_shepherd import paired_experiments as paired
from traffic_shepherd.workflow import sha
from traffic_shepherd.interaction_diagnostics import OfflineInteractionObserver
from traffic_shepherd.metrics import RequestLedger, service_summary
from traffic_shepherd.scenarios import TripRequest
from tests.test_paired_experiments import constant_policy, job, registration, report
from tests.test_paired_experiments import runtime as runtime


@pytest.fixture
def replay_pair(tmp_path, runtime):
    paths = [tmp_path / "original", tmp_path / "replay"]
    for path in paths:
        runtime["time"] = 0.
        paired.evaluate_arm(job=job(), registration=registration(), output=path,
                            policy=constant_policy("damping"))
    return paths


def change_json(path, name, update, *, jsonl=False):
    file = path / f"{name}.{'jsonl' if jsonl else 'json'}"
    content = [json.loads(line) for line in file.read_text().splitlines()] if jsonl else json.loads(file.read_text())
    update(content)
    file.write_text("".join(json.dumps(r) + "\n" for r in content) if jsonl else json.dumps(content))
    if name != "manifest":
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest[f"{name}_sha256"] = sha(file)
        manifest_path.write_text(json.dumps(manifest))


def test_full_equality_includes_packets_states_and_outcomes(replay_pair):
    result = evidence.compare_replay(*replay_pair)
    assert result["exact_replay"] and result["numerical_tolerance"] == 0.
    assert set(result["comparisons"]) == set(evidence.ORIGINAL_STREAMS)
    kinds = result["comparisons"]["telemetry"]["rows_by_kind"]
    assert set(kinds) == {"rsu_received_report", "cav_report", "command", "response"}
    assert result["decision_snapshots"]["count"] == 1
    assert "source_wait_vehicle_seconds" in result["summary"]["compared_fields"]
    assert not result["complete_2400_s_replay"]
    assert not result["diagnostic_channel_validated"]


def test_provenance_and_wall_time_are_not_physical_differences(replay_pair):
    _, replay = replay_pair
    change_json(replay, "summary", lambda r: r.update(elapsed_wall_seconds=999., assignment="observer audit"))
    change_json(replay, "policy", lambda rows: [r.update(assignment="observer audit") for r in rows], jsonl=True)
    change_json(replay, "manifest", lambda r: r.update(job_id="replay-original", registration_sha256="observer",
                                                      active_source_sha256={"new_observer.py": "new"}))
    assert evidence.compare_replay(*replay_pair)["exact_replay"]


@pytest.mark.parametrize("kind,field", [("response", "measured_acceleration_mps2"),
                                      ("command", "issued_acceleration_mps2"),
                                      ("rsu_received_report", "frame"), ("cav_report", "frame")])
def test_any_late_cav_divergence_fails_even_with_rehashed_artifact(replay_pair, kind, field):
    def modify(rows):
        row = next(r for r in reversed(rows) if r["kind"] == kind)
        if field == "frame":
            row[field]["acceleration_mps2"] += .001
        else:
            row[field] += .001
    change_json(replay_pair[1], "telemetry", modify, jsonl=True)
    with pytest.raises(ValueError, match="replay telemetry differs"):
        evidence.compare_replay(*replay_pair)


@pytest.mark.parametrize("key,value", [("source_wait_vehicle_seconds", 2.),
                                       ("unfinished_requested", 1), ("minimum_closing_ttc_s", .1)])
def test_service_and_safety_summary_fields_cannot_be_ignored(replay_pair, key, value):
    change_json(replay_pair[1], "summary", lambda r: r.update({key: value}))
    with pytest.raises(ValueError, match="replay summary differs"):
        evidence.compare_replay(*replay_pair)


def test_policy_state_change_fails_even_when_commands_remain_equal(replay_pair):
    change_json(replay_pair[1], "policy", lambda rows: rows[-1].update(reason="new reason"), jsonl=True)
    with pytest.raises(ValueError, match="replay policy differs"):
        evidence.compare_replay(*replay_pair)


def test_truncated_stream_and_unbound_original_rejected(replay_pair):
    change_json(replay_pair[1], "telemetry", lambda rows: rows.pop(), jsonl=True)
    with pytest.raises(ValueError, match="row count differs"):
        evidence.compare_replay(*replay_pair)
    (replay_pair[0] / "traffic.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="changed arm artifact"):
        evidence.compare_replay(*replay_pair)


def packet(acquired=1., received=None, *, kind="rsu_received_report"):
    row = dict(kind=kind, **report(acquired).payload())
    if kind == "rsu_received_report":
        row["received_at_s"] = acquired if received is None else received
    return row


def available(rows, now=1., max_age=1.):
    return evidence.available_report_evidence(rows, cutoff_s=now, history_s=20., max_report_age_s=max_age)


def crosswalk(acquired=1.):
    return dict(reporter_id="a", timestamp_s=acquired,
                front=dict(vehicle_id="human", gap_m=40., relative_speed_mps=0.), rear=None)


EVENT = dict(event_id="offline-event-1", start_s=30., vehicle_ids=["a", "human"])


def test_only_received_before_cutoff_counts_and_same_tick_postaction_report_is_excluded():
    rows = [packet(1., 3.), packet(2., kind="cav_report")]
    assert available(rows, 2.)["history_packet_count"] == 0
    result = available(rows, 3., max_age=2.)
    assert result["history_packet_count"] == 1
    assert result["fresh_reporter_ids"] == ["a"]


def test_stale_history_is_not_current_warning_and_future_event_is_not_predicted():
    result = evidence.audit_event_acquisition(EVENT, available([packet()], now=10.), [crosswalk()])
    assert result["future_event_at_cutoff"] and result["lead_to_event_s"] == 20.
    assert result["identity_crosswalk_complete"]
    person = result["participants"][1]
    assert person["state_observed_before_cutoff"]
    hit = person["observed_samples"][0]
    assert hit["channel"] == "anonymous_front_detection"
    assert not hit["identity_available_online"] and not hit["fresh_current_report"]
    assert not result["event_prediction_demonstrated"] and not result["causal_mediation_demonstrated"]


def test_missing_identity_mapping_is_unknown_not_demonstrated_absence():
    result = evidence.audit_event_acquisition(EVENT, available([packet()]))
    assert not result["identity_crosswalk_complete"]
    assert result["missing_crosswalks"] == [dict(reporter_id="a", timestamp_s=1.)]
    assert result["participants"][0]["state_observed_before_cutoff"]
    assert not result["participants"][1]["state_observed_before_cutoff"]


@pytest.mark.parametrize("change", [lambda link: link["front"].update(gap_m=40.1),
                                   lambda link: link["front"].update(relative_speed_mps=.1),
                                   lambda link: link.update(front=None)])
def test_privileged_neighbor_mapping_must_match_actual_anonymous_measurement(change):
    link = crosswalk()
    change(link)
    with pytest.raises(ValueError, match="crosswalk disagrees"):
        evidence.audit_event_acquisition(EVENT, available([packet()]), [link])


def test_conflicting_or_impossible_receipts_and_changed_bindings_fail():
    wrong = packet(1., .5)
    with pytest.raises(ValueError, match="chronology"):
        available([wrong])
    wrong = packet()
    wrong["frame"]["speed_mps"] += 1.
    with pytest.raises(ValueError, match="conflicting"):
        available([packet(), wrong])
    result = available([packet()])
    result["evidence"]["now_s"] = 0.
    with pytest.raises(ValueError, match="binding"):
        evidence.audit_event_acquisition(EVENT, result)


def test_same_tick_receipt_is_available_before_action_but_postevent_is_hindsight():
    result = evidence.audit_event_acquisition(dict(EVENT, start_s=.5), available([packet()]), [crosswalk()])
    assert result["participants"][0]["observed_samples"][0]["fresh_current_report"]
    assert not result["future_event_at_cutoff"]
    assert result["lead_to_event_s"] == -.5


@pytest.fixture
def diagnostic(tmp_path, monkeypatch):
    requests = [TripRequest("a", 0., "route", "normal_car", True),
                TripRequest("human", 0., "route", "normal_car", False)]
    clock = dict(now=0.)
    api = SimpleNamespace(vehicle=SimpleNamespace(
        getLaneID=lambda _: "L0", getRoadID=lambda _: "edge",
        getLanePosition=lambda v: (10. if v == "a" else 55.)+20.*clock["now"],
        getLength=lambda _: 5., getSpeed=lambda _: 20.,
        getAcceleration=lambda v: -.25 if v == "a" else 0.,
        getLeader=lambda v, _: ("human", 37.5) if v == "a" else None,
        getMinGap=lambda _: 2.5), lane=SimpleNamespace(getLength=lambda _: 1000.))
    observer = OfflineInteractionObserver(tmp_path / "interactions.jsonl.gz", requests=requests, horizon_s=2)
    ledger = RequestLedger(requests)
    for now in (0., .5, 1., 1.5, 2.):
        clock["now"] = now
        departed = ["a", "human"] if now == .5 else []
        observer.sample(api, now_s=now, live_ids=["a", "human"] if now else [], departed_ids=departed)
        if now:
            ledger.advance(now, departed, [])
    meta = observer.close()
    received = report(1.)
    received = replace(received, frame=replace(received.frame, acceleration_mps2=-.25, position_m=30., rear_coverage_m=25.))
    (tmp_path / "telemetry.jsonl").write_text(json.dumps(dict(kind="rsu_received_report", received_at_s=1., **received.payload()))+"\n")
    totals = dict(braking_vehicle_seconds=0., acceleration_energy=.125, ttc_below_3_vehicle_seconds=0.,
                  collision_involvements=0, starting_teleports=0, requested_not_arrived_vehicle_seconds=4.)
    intervals = [dict(start_s=float(t-1), end_s=float(t), **{k: v/2 for k, v in totals.items()}) for t in (1, 2)]
    (tmp_path / "intervals.jsonl").write_text("".join(json.dumps(r)+"\n" for r in intervals))
    manifest = dict(seed=4107, mainline_vph=2700, penetration=.2, role="exposed", duration_s=2,
                    communication=dict(delay_s=0.), interaction_diagnostics=meta)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    summary = dict(totals=totals, minimum_closing_ttc_s=None, **service_summary(ledger, 2.))
    monkeypatch.setattr(evidence, "request_schedule", lambda *args, **kwargs: requests)
    monkeypatch.setattr(evidence, "verified_arm", lambda _: (manifest, summary))
    return tmp_path, manifest, summary


def rewrite_diagnostic(fixture, mutate):
    path, manifest, _ = fixture
    file = path / "interactions.jsonl.gz"
    with gzip.open(file, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    mutate(rows)
    raw = "".join(json.dumps(r, separators=(",", ":"))+"\n" for r in rows).encode()
    file.write_bytes(gzip.compress(raw, mtime=0))
    manifest["interaction_diagnostics"].update(sha256=sha(file), uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                                               uncompressed_bytes=len(raw))


def test_diagnostic_validator_reconstructs_physics_service_and_anonymous_geometry(diagnostic):
    result = evidence.validate_diagnostics(diagnostic[0])
    assert result["diagnostic_channel_validated"]
    assert result["samples"] == 5 and result["vehicle_rows"] == 8
    assert result["acquired_cav_crosswalks"] == 1
    assert result["source_wait_reconstructed"] and result["every_physical_interval_reconstructed"]


@pytest.mark.parametrize("mutate,message", [
    (lambda rows: rows[2].update(time_s=.75), "clock"),
    (lambda rows: rows[3]["vehicles"][0].update(sensor_front_candidate_id="other"), "identity"),
    (lambda rows: rows[2]["vehicles"][0].update(acceleration_effort=0.), "effort"),
    (lambda rows: rows[2].update(departed_ids=[]), "origin"),
    (lambda rows: rows[-1].update(complete=False), "footer"),
    (lambda rows: rows[2].update(new_collision_involvement_ids=["a"]), "collision"),
])
def test_diagnostic_corruption_fails_even_if_hashes_are_recomputed(diagnostic, mutate, message):
    rewrite_diagnostic(diagnostic, mutate)
    with pytest.raises(ValueError, match=message):
        evidence.validate_diagnostics(diagnostic[0])
