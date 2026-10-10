"""Complete2400 s PI savedrun oracle tests; no simulator."""

import pytest
from tests.cv06_saved_fixture import saved as fs_saved
from tests.cv06_saved_fixture import lines, write
from output.controller_verification_v12 import independent_review as review


def refresh(root):
    manifest = review.read(root / "manifest.json")
    for name in ("summary", "telemetry", "intervals"):
        manifest[name + "_sha256"] = review.sha(
            root / (name + (".json" if name == "summary" else ".jsonl"))
        )
    write(root / "manifest.json", manifest)


@pytest.fixture
def saved(tmp_path):
    root, e = fs_saved.__wrapped__(tmp_path)
    e["pi_transitive_source_sha256"] = {"v7.py": "frozen"}
    manifest = review.read(root / "manifest.json")
    manifest["controller"] = "pi-external-u"
    manifest["pi_transitive_source_sha256"] = e["pi_transitive_source_sha256"]
    manifest["controller_identity"] = dict(
        session_version="pi_session/1",
        adapter_version="pi_owner_adapter/2",
        u_mps=31.29,
        u_provenance="fixed_registered_setpoint",
        u_evidence_id="cv11-registered-u31.29",
        cav_ids=["cav"],
    )
    write(root / "manifest.json", manifest)
    original = list(review.rows(root / "telemetry.jsonl"))
    data = []
    prior = 31.29
    for tick in (1.0, 2.0):
        command = next(x for x in original if x["kind"] == "command" and x["timestamp_s"] == tick)
        report = next(
            x for x in original if x["kind"] == "cav_report" and x["frame"]["timestamp_s"] == tick
        )
        raw = 0.5 * 32.29 + 0.5 * prior
        evidence = dict(
            kind="pi_source_evidence",
            vehicle_id="cav",
            timestamp_s=tick,
            frame=report["frame"],
            generation=1,
            previous_raw_command_mps=prior,
            raw_proposal_mps=raw,
            requested_acceleration_mps2=raw - 31.29,
            issued_acceleration_mps2=0.0,
            issued_target_speed_mps=31.29,
            owner_record={k: v for k, v in command.items() if k != "kind"},
            u_report=dict(
                value_mps=31.29,
                provenance="fixed_registered_setpoint",
                evidence_id="cv11-registered-u31.29",
                report_timestamp_s=None,
                receipt_timestamp_s=None,
                max_age_s=None,
            ),
        )
        data.append(evidence)
        for status in ("written_owner_receipt", "committed_raw_proposal"):
            data.append(
                dict(
                    kind="pi_owner_receipt",
                    vehicle_id="cav",
                    timestamp_s=tick,
                    status=status,
                    reason=None,
                    raw_command_mps=raw,
                    issued_target_speed_mps=31.29,
                )
            )
        data.extend([command, report])
        data.extend(
            x for x in original if x["kind"] == "response" and x["command_timestamp_s"] == tick
        )
        prior = raw
    lines(root / "telemetry.jsonl", data)
    summary = review.read(root / "summary.json")
    summary["controller"] = "pi-external-u"
    write(root / "summary.json", summary)
    refresh(root)
    return root, e


def test_complete_saved_run_passes(saved):
    result = review.verify_run(*saved)
    assert result["integrity_pass"], result
    assert result["safety_pass"]
    assert result["physical_steps"] == 4800 and result["intervals"] == 2400
    assert result["tracking"]["counts"] == {"delivered": 1, "departure_censored": 1}
    assert result["physical_vector"]["service"] == 2.75


@pytest.mark.parametrize("kind", ["pi_source_evidence", "pi_owner_receipt", "command", "response"])
def test_missing_stream_row_rejected_with_updated_hashes(saved, kind):
    root, e = saved
    data = list(review.rows(root / "telemetry.jsonl"))
    data.pop(next(i for i, x in enumerate(data) if x["kind"] == kind))
    lines(root / "telemetry.jsonl", data)
    refresh(root)
    assert not review.verify_run(root, e)["integrity_pass"]


@pytest.mark.parametrize("field", ["previous_raw_command_mps", "raw_proposal_mps", "generation"])
def test_independent_memory_or_raw_mismatch_fails(saved, field):
    root, e = saved
    data = list(review.rows(root / "telemetry.jsonl"))
    next(x for x in data if x["kind"] == "pi_source_evidence" and x["timestamp_s"] == 2.0)[
        field
    ] += 1
    lines(root / "telemetry.jsonl", data)
    refresh(root)
    assert not review.verify_run(root, e)["integrity_pass"]


@pytest.mark.parametrize("fault", ["U", "dependency", "service", "lifecycle", "mode"])
def test_identity_or_physical_accounting_fails(saved, fault):
    root, e = saved
    if fault in ("U", "dependency"):
        m = review.read(root / "manifest.json")
        if fault == "U":
            m["controller_identity"]["u_mps"] = 30.0
        else:
            m["pi_transitive_source_sha256"] = {}
        write(root / "manifest.json", m)
    elif fault == "service":
        m = review.read(root / "summary.json")
        m["system_seconds_per_requested_vehicle"] += 1
        write(root / "summary.json", m)
    else:
        data = list(review.rows(root / "physics_steps.jsonl"))
        if fault == "lifecycle":
            data[1]["departed_ids"] = []
        else:
            data[1]["speed_modes"]["cav"] = 0
        lines(root / "physics_steps.jsonl", data)
    refresh(root)
    assert not review.verify_run(root, e)["integrity_pass"]


def test_collision_stops_safety_but_retains_integrity(saved):
    root, e = saved
    data = list(review.rows(root / "intervals.jsonl"))
    data[1]["collision_involvements"] = 1
    lines(root / "intervals.jsonl", data)
    m = review.read(root / "summary.json")
    m["totals"]["collision_involvements"] = 1
    write(root / "summary.json", m)
    refresh(root)
    result = review.verify_run(root, e)
    assert result["integrity_pass"] and not result["safety_pass"]


@pytest.mark.parametrize("missing", [True, False])
def test_valid_initial_refusal_then_reset_matches_generation(saved, missing):
    root, e = saved
    data = list(review.rows(root / "telemetry.jsonl"))
    output = []
    reason = "missing_observation" if missing else "missing_or_uncovered_leader"
    for row in data:
        t = row.get(
            "command_timestamp_s", row.get("timestamp_s", row.get("frame", {}).get("timestamp_s"))
        )
        if t == 1.0:
            if row["kind"] in ("pi_source_evidence", "pi_owner_receipt"):
                continue
            if row["kind"] == "command":
                row.update(
                    owner="native",
                    issued_acceleration_mps2=None,
                    issued_target_speed_mps=None,
                    fallback_reason=reason,
                )
                output.append(
                    dict(
                        kind="pi_owner_receipt",
                        vehicle_id="cav",
                        timestamp_s=1.0,
                        status="owner_row_without_pi",
                        reason=None,
                        raw_command_mps=None,
                        issued_target_speed_mps=None,
                    )
                )
            elif row["kind"] == "cav_report":
                if missing:
                    output.append(
                        dict(
                            kind="observation_unavailable",
                            vehicle_id="cav",
                            timestamp_s=1.0,
                            reason="missing",
                        )
                    )
                    continue
                row["frame"]["front"] = None
                row.update(owner="native", issued_acceleration_mps2=None, fallback_reason=reason)
            elif row["kind"] == "response":
                row["owner"] = "native"
        elif t == 2.0:
            if row["kind"] == "pi_source_evidence":
                row.update(
                    generation=2,
                    previous_raw_command_mps=31.29,
                    raw_proposal_mps=31.79,
                    requested_acceleration_mps2=0.5,
                )
            elif row["kind"] == "pi_owner_receipt":
                row["raw_command_mps"] = 31.79
        output.append(row)
    lines(root / "telemetry.jsonl", output)
    summary = review.read(root / "summary.json")
    summary["command_counts"] = {"native": 1, "damping": 1}
    summary["fallback_counts"] = {reason: 1}
    write(root / "summary.json", summary)
    refresh(root)
    verdict = review.verify_run(root, e)
    assert verdict["integrity_pass"], verdict
