"""Offline-only integrity and acquisition checks for registered observer replays.

This module never launches SUMO or supplies a policy input. It compares the full
original experiment, not merely a favorable prefix. Event participants and their
identities are retrospective audit data, never report-derived online features.
"""
from collections import Counter
import hashlib
import gzip
from itertools import zip_longest
import json
from math import isclose, isfinite
from pathlib import Path

from .escalation_observations import report_from_payload
from .paired_experiments import digest, verified_arm
from .reporting import RSUReportBuffer
from .metrics import RequestLedger, braking_effort, closing_ttc, service_summary
from .scenarios import request_schedule
from .workflow import sha


ORIGINAL_JOB_IDS = (
    "s4107_m2700_p20_damping",
    "s4107_m2700_p20_t600_wall",
    "s4107_m2700_p20_t600_assembly_only",
    "s4108_m2700_p20_damping",
    "s4108_m2700_p20_t600_wall",
)
ORIGINAL_STREAMS = ("telemetry", "intervals", "traffic", "policy", "formation", "offline_state")
IDENTITY_FIELDS = (
    "seed", "mainline_vph", "penetration", "role", "controller", "duration_s",
    "mechanics_only", "decision_time_s", "snapshot_times_s", "policy_spec", "communication",
    "scenario_identity", "exogenous_future_sha256", "config_sha256", "model_sha256",
    "network_sha256", "sumo_binary_sha256", "scenario_template_sha256",
    "formation_samples_are_lifecycle_scoped",
)
# These are provenance/timing, not simulator outcomes. No recursive key removal:
# an identically named physical field nested elsewhere remains part of equality.
SUMMARY_PROVENANCE_FIELDS = frozenset({"elapsed_wall_seconds", "assignment"})


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank row in {Path(path).name}:{number}")
            yield json.loads(line)


def _difference(left, right, prefix=""):
    """Bounded first differing field; do not print complete vehicle histories."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            location = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                return location
            if _canonical(left[key]) != _canonical(right[key]):
                return _difference(left[key], right[key], location)
    return prefix or "row"


def compare_replay(original, replay):
    """Raise on any physical/controller difference; accept provenance changes only.

Both arms must pass independently reconstructed snapshot hashes and full-horizon
interval/summary checks. All original streams must match, including packets
received before every action, issued commands, measured responses, policy state,
formation geometry, and privileged decision snapshots. New diagnostic streams
are deliberately not mistaken for original evidence. Their own validity is a
separate check. This returns ``exact_replay=True`` only after all checks pass.
"""
    original, replay = Path(original), Path(replay)
    before, old_summary = verified_arm(original)
    after, new_summary = verified_arm(replay)
    for field in IDENTITY_FIELDS:
        if field not in before or field not in after:
            raise ValueError(f"replay identity missing: {field}")
        if _canonical(before[field]) != _canonical(after[field]):
            raise ValueError(f"replay identity differs: {field}")
    if before["role"] not in {"exposed", "development"}:
        raise ValueError("observer audit is restricted to exposed/development evidence")
    comparisons = {}
    for name in ORIGINAL_STREAMS:
        left_rows, right_rows = _rows(original / f"{name}.jsonl"), _rows(replay / f"{name}.jsonl")
        fingerprint, counts, number = hashlib.sha256(), Counter(), 0
        sentinel = object()
        for number, (left, right) in enumerate(zip_longest(left_rows, right_rows, fillvalue=sentinel), 1):
            if left is sentinel or right is sentinel:
                raise ValueError(f"replay {name} row count differs at row {number}")
            if name == "policy":
                left = {k: v for k, v in left.items() if k != "assignment"}
                right = {k: v for k, v in right.items() if k != "assignment"}
            canonical = _canonical(left)
            if canonical != _canonical(right):
                raise ValueError(f"replay {name} differs at row {number}: {_difference(left, right)}")
            fingerprint.update(canonical + b"\n")
            if name == "telemetry":
                counts[left["kind"]] += 1
        comparisons[name] = dict(exact=True, rows=number, canonical_sha256=fingerprint.hexdigest())
        if counts:
            comparisons[name]["rows_by_kind"] = dict(counts)
    old_snapshots = json.loads((original / "decision_snapshots.json").read_text(encoding="utf-8"))
    new_snapshots = json.loads((replay / "decision_snapshots.json").read_text(encoding="utf-8"))
    if _canonical(old_snapshots) != _canonical(new_snapshots):
        raise ValueError(f"replay decision snapshots differ: {_difference(old_snapshots, new_snapshots)}")
    old_physical = {k: v for k, v in old_summary.items() if k not in SUMMARY_PROVENANCE_FIELDS}
    new_physical = {k: v for k, v in new_summary.items() if k not in SUMMARY_PROVENANCE_FIELDS}
    if _canonical(old_physical) != _canonical(new_physical):
        raise ValueError(f"replay summary differs: {_difference(old_physical, new_physical)}")
    return dict(
        schema="observer_replay_integrity/1", exact_replay=True, numerical_tolerance=0.,
        original_job_id=before["job_id"], replay_job_id=after["job_id"], duration_s=before["duration_s"],
        complete_2400_s_replay=before["duration_s"] == 2400,
        original_manifest_file_sha256=sha(original / "manifest.json"),
        replay_manifest_file_sha256=sha(replay / "manifest.json"),
        original_registration_sha256=before["registration_sha256"],
        replay_registration_sha256=after["registration_sha256"], comparisons=comparisons,
        decision_snapshots=dict(exact=True, count=len(old_snapshots), canonical_sha256=digest(old_snapshots)),
        summary=dict(exact=True, canonical_sha256=digest(old_physical), compared_fields=sorted(old_physical)),
        ignored_provenance=dict(summary=sorted(SUMMARY_PROVENANCE_FIELDS), policy=["assignment"],
                                manifest="Compared physical identity fields explicitly; source/allocation provenance may differ."),
        source_wait_final_exact=True, source_wait_independent_interval_series=False,
        diagnostic_channel_validated=False,
        scope="Exact original evidence only; diagnostic validity, mechanism and CAV predictability remain separate.",
    )


def _equal_measurement(actual, expected, context):
    if expected is None:
        okay = actual is None
    else:
        okay = isinstance(actual, (int, float)) and isfinite(actual) and isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-8)
    if not okay:
        raise ValueError(f"diagnostic measurement mismatch: {context}")


def _validate_acquisition(frame, states, now):
    """Independent sensor-emulator geometry, using raw measured positions/lengths."""
    state = states.get(frame["vehicle_id"])
    if state is None or frame["timestamp_s"] != now:
        raise ValueError("diagnostic acquisition state absent or off-clock")
    for report_key, state_key in (("speed_mps", "speed_mps"), ("acceleration_mps2", "acceleration_mps2"),
                                  ("lane_id", "lane_id"), ("position_m", "lane_position_m")):
        if _canonical(frame[report_key]) != _canonical(state[state_key]):
            raise ValueError(f"diagnostic acquisition state mismatch: {report_key}")
    coverage = dict(front=min(100., max(0., state["lane_length_m"]-state["lane_position_m"])),
                    rear=min(100., max(0., state["lane_position_m"]-state["length_m"])))
    candidates = {"front": [], "rear": []}
    for other in states.values():
        if other["vehicle_id"] == state["vehicle_id"] or other["lane_id"] != state["lane_id"]:
            continue
        side = "front" if other["lane_position_m"] >= state["lane_position_m"] else "rear"
        gap = (other["lane_position_m"]-other["length_m"]-state["lane_position_m"] if side == "front"
               else state["lane_position_m"]-state["length_m"]-other["lane_position_m"])
        if gap <= coverage[side]:
            candidates[side].append((gap, other["vehicle_id"]))
    for side in ("front", "rear"):
        measured = frame[side]
        chosen = min(candidates[side]) if candidates[side] else None
        expected = None if chosen is None else dict(gap_m=chosen[0],
            relative_speed_mps=states[chosen[1]]["speed_mps"]-state["speed_mps"])
        if _canonical(measured) != _canonical(expected) or frame[f"{side}_coverage_m"] != coverage[side]:
            raise ValueError("diagnostic crosswalk disagrees with actual CAV sensing")
        if state[f"sensor_{side}_candidate_id"] != (chosen[1] if chosen else None):
            raise ValueError("diagnostic sensor candidate identity disagrees with measured geometry")
        if chosen and any(_canonical(state[f"same_lane_{side}_{key}"]) != _canonical(expected[key])
                          for key in ("gap_m", "relative_speed_mps")):
            raise ValueError("diagnostic sensor candidate scalar disagrees with measured geometry")


def validate_diagnostics(replay):
    """Verify new offline evidence before another registered replay may launch.

Reconstruct the complete half-second physical and requested-service ledger;
match all actual CAV receipts to independent acquisition-time sensor geometry.
Only recomputation roundoff uses 1e-8 absolute / 1e-12 relative tolerance. The
separate original/replay comparison continues to require exact equality.
"""
    replay = Path(replay)
    manifest, summary = verified_arm(replay)
    meta = manifest.get("interaction_diagnostics")
    if not meta or not meta.get("complete") or meta.get("online_input") is not False:
        raise ValueError("complete offline diagnostic metadata required")
    if manifest["communication"].get("delay_s", 0.) != 0.:
        raise ValueError("this replay validator requires the registered zero-delay channel")
    requests = request_schedule(manifest["seed"], manifest["mainline_vph"],
                                penetration=manifest["penetration"], role=manifest["role"])
    expected_requests = [dict(vehicle_id=r.vehicle_id, requested_departure_s=r.depart_s,
        origin_route=r.route, is_cav=r.is_cav) for r in sorted(requests, key=lambda r: r.vehicle_id)]
    ledger = RequestLedger(requests)
    origin_ledgers = {origin: RequestLedger([r for r in requests if r.route == origin])
                      for origin in sorted({r.route for r in requests})}
    cav_ids = {r.vehicle_id for r in requests if r.is_cav}
    known_ids = {r.vehicle_id for r in requests}
    packets = (row for row in _rows(replay / "telemetry.jsonl") if row["kind"] in
               {"rsu_received_report", "observation_unavailable"})
    next_packet = next(packets, None)
    intervals = iter(_rows(replay / "intervals.jsonl"))
    fingerprint, byte_count, sample_count, vehicle_rows, crosswalk_count = hashlib.sha256(), 0, 0, 0, 0
    previous_ids, previous_colliding, totals, pending_interval = set(), set(), Counter(), Counter()
    minimum, header, footer = None, None, None
    with gzip.open(replay / meta["file"], "rb") as stream:
        for raw in stream:
            fingerprint.update(raw)
            byte_count += len(raw)
            row = json.loads(raw)
            if header is None:
                header = row
                if (header.get("kind") != "header" or header.get("schema") != meta["schema"]
                        or header.get("channel") != "offline_evaluation_only"
                        or header.get("physics_dt_s") != .5 or header.get("horizon_s") != manifest["duration_s"]
                        or header.get("sensor_range_m") != 100. or header.get("ttc_leader_range_m") != 200.
                        or header.get("requests") != expected_requests):
                    raise ValueError("diagnostic header/exogenous requests mismatch")
                continue
            if footer is not None:
                raise ValueError("diagnostic data after completion footer")
            if row.get("kind") == "footer":
                footer = row
                continue
            now = sample_count*.5
            if (row.get("kind") != "sample" or row.get("time_s") != now
                    or row.get("interval_start_s") != max(0., now-.5)
                    or row.get("interval_dt_s") != (.5 if now else 0.) or now > manifest["duration_s"]):
                raise ValueError("diagnostic half-second clock mismatch")
            ids = [r["vehicle_id"] for r in row["vehicles"]]
            if ids != sorted(set(ids)) or not set(ids) <= known_ids:
                raise ValueError("diagnostic population unknown, duplicated or unsorted")
            states = {r["vehicle_id"]: r for r in row["vehicles"]}
            if (row["first_observed_ids"] != sorted(set(ids)-previous_ids)
                    or row["disappeared_ids"] != sorted(previous_ids-set(ids))):
                raise ValueError("diagnostic live-population transitions mismatch")
            acquired = set()
            while next_packet is not None:
                receipt = next_packet.get("received_at_s", next_packet.get("timestamp_s"))
                if receipt > now:
                    break
                if receipt != now or now % 1.:
                    raise ValueError("diagnostic CAV receipt/acquisition clock mismatch")
                if next_packet["kind"] == "rsu_received_report":
                    frame = next_packet["frame"]
                    reporter = frame["vehicle_id"]
                    _validate_acquisition(frame, states, now)
                    crosswalk_count += 1
                else:
                    reporter = next_packet["vehicle_id"]
                if reporter in acquired:
                    raise ValueError("duplicate CAV acquisition or unavailable marker")
                acquired.add(reporter)
                next_packet = next(packets, None)
            if now % 1. == 0. and now < manifest["duration_s"] and acquired != set(ids) & cav_ids:
                raise ValueError("diagnostic CAV report population mismatch")
            measured = braking_effort([s["acceleration_mps2"] for s in row["vehicles"]], .5)
            measured.update(ttc_below_3_vehicle_seconds=0., collision_involvements=0, starting_teleports=0)
            for state in row["vehicles"]:
                leader = states.get(state["ttc_leader_id"])
                ttc = None if leader is None else closing_ttc(state["ttc_gap_m"], state["speed_mps"], leader["speed_mps"])
                if bool(leader) != state["ttc_leader_state_available"]:
                    raise ValueError("diagnostic TTC leader availability mismatch")
                _equal_measurement(state["closing_ttc_s"], ttc, "closing TTC")
                if leader is not None:
                    _equal_measurement(state["ttc_closing_speed_mps"], state["speed_mps"]-leader["speed_mps"], "closing speed")
                if ttc is not None and now:
                    minimum = ttc if minimum is None else min(minimum, ttc)
                low = .5 if now and ttc is not None and ttc < 3. else 0.
                _equal_measurement(state["low_ttc_vehicle_seconds"], low, "vehicle TTC contribution")
                _equal_measurement(state["braking_vehicle_seconds"], .5 if now and state["acceleration_mps2"] < -3. else 0., "vehicle braking")
                _equal_measurement(state["acceleration_effort"], .5*state["acceleration_mps2"]**2 if now else 0., "vehicle effort")
                measured["ttc_below_3_vehicle_seconds"] += low
            colliding = set(row["colliding_ids"])
            if row["new_collision_involvement_ids"] != sorted(colliding-previous_colliding):
                raise ValueError("diagnostic collision involvement identities mismatch")
            measured["collision_involvements"] = len(colliding-previous_colliding)
            measured["starting_teleports"] = row["starting_teleports"]
            if row.get("starting_teleport_ids_available"):
                teleport_ids = row["starting_teleport_ids"]
                if len(set(teleport_ids)) != len(teleport_ids) or len(teleport_ids) != row["starting_teleports"]:
                    raise ValueError("diagnostic teleport event identities mismatch")
            if now:
                measured["requested_not_arrived_vehicle_seconds"] = ledger.advance(now, row["departed_ids"], row["arrived_ids"])
            else:
                measured = Counter({k: 0. for k in measured})
                measured["requested_not_arrived_vehicle_seconds"] = 0.
                if row["departed_ids"] or row["arrived_ids"]:
                    raise ValueError("diagnostic initial lifecycle events invalid")
            if set(row["service_by_origin"]) != set(origin_ledgers):
                raise ValueError("diagnostic request-origin groups mismatch")
            for origin, subledger in origin_ledgers.items():
                prior_system, prior_source = subledger.system_vehicle_seconds, subledger.source_vehicle_seconds
                if now:
                    subledger.advance(now, [v for v in row["departed_ids"] if v in subledger.requests],
                                      [v for v in row["arrived_ids"] if v in subledger.requests])
                expected = dict(requested_population=len(subledger.requests), requested_due=subledger.due(now),
                    inserted=len(subledger.inserted), arrived=len(subledger.arrived),
                    source_waiting=subledger.due(now)-len(subledger.inserted),
                    requested_not_arrived=subledger.due(now)-len(subledger.arrived),
                    system_vehicle_seconds=subledger.system_vehicle_seconds,
                    source_wait_vehicle_seconds=subledger.source_vehicle_seconds,
                    interval_system_vehicle_seconds=subledger.system_vehicle_seconds-prior_system,
                    interval_source_wait_vehicle_seconds=subledger.source_vehicle_seconds-prior_source)
                for key, expected_value in expected.items():
                    _equal_measurement(row["service_by_origin"][origin][key], expected_value, f"origin {origin}/{key}")
            totals.update(measured)
            pending_interval.update(measured)
            if now and now % 1. == 0.:
                saved = next(intervals, None)
                if saved is None or saved["end_s"] != now:
                    raise ValueError("diagnostic/saved interval clock mismatch")
                for key, expected_value in pending_interval.items():
                    _equal_measurement(saved.get(key, 0.), expected_value, f"physical interval {now}/{key}")
                pending_interval.clear()
            previous_ids, previous_colliding = set(ids), colliding
            sample_count += 1
            vehicle_rows += len(ids)
    if (footer is None or footer.get("complete") is not True
            or footer.get("final_time_s") != manifest["duration_s"]
            or footer.get("sample_count") != sample_count or footer.get("vehicle_rows") != vehicle_rows
            or sample_count != 2*manifest["duration_s"]+1 or sample_count != meta["sample_count"]
            or vehicle_rows != meta["vehicle_rows"] or next_packet is not None or next(intervals, None) is not None):
        raise ValueError("diagnostic footer or complete horizon mismatch")
    if fingerprint.hexdigest() != meta["uncompressed_sha256"] or byte_count != meta["uncompressed_bytes"]:
        raise ValueError("diagnostic uncompressed hash/size mismatch")
    for key, expected_value in totals.items():
        _equal_measurement(summary["totals"].get(key, 0.), expected_value, f"episode total {key}")
    for key, expected_value in service_summary(ledger, manifest["duration_s"]).items():
        _equal_measurement(summary[key], expected_value, f"episode service {key}")
    _equal_measurement(summary["minimum_closing_ttc_s"], minimum, "episode minimum TTC")
    return dict(schema="observer_diagnostic_validation/1", diagnostic_channel_validated=True,
        replay_manifest_file_sha256=sha(replay / "manifest.json"), compressed_sha256=meta["sha256"],
        uncompressed_sha256=fingerprint.hexdigest(), samples=sample_count, vehicle_rows=vehicle_rows,
        acquired_cav_crosswalks=crosswalk_count, all_requested_service_reconstructed=True,
        source_wait_reconstructed=True, every_physical_interval_reconstructed=True,
        reconstruction_absolute_tolerance=1e-8, reconstruction_relative_tolerance=1e-12,
        original_replay_equality_tolerance=0., online_inputs_added=False)


def available_report_evidence(telemetry_rows, *, cutoff_s, history_s, max_report_age_s):
    """Rebuild actual pre-action packet availability, excluding post-action reports.

The retained history matches PolicyInput: receipt age defines history retention;
the fresh current packet additionally uses acquisition age. Same-tick received
packets are admissible because the paired runner logs receipt before policy.
Future receipts are never pulled backward to their acquisition timestamp.
"""
    if not isfinite(cutoff_s) or cutoff_s < 0:
        raise ValueError("invalid decision cutoff")
    buffer = RSUReportBuffer(max_age_s=max_report_age_s, history_s=history_s)
    retained, seen, receipt_previous = [], {}, -1.
    for row in telemetry_rows:
        if row.get("kind") != "rsu_received_report":
            continue
        payload = {k: v for k, v in row.items() if k not in {"kind", "received_at_s"}}
        report = report_from_payload(payload)
        receipt = row["received_at_s"]
        if not isfinite(receipt) or receipt < report.sent_at_s:
            raise ValueError("invalid report receipt chronology")
        if receipt < receipt_previous:
            raise ValueError("received report stream is not chronological")
        receipt_previous = receipt
        if receipt > cutoff_s:
            continue
        key = (report.frame.vehicle_id, report.frame.timestamp_s)
        if key in seen:
            if seen[key] != report:
                raise ValueError("conflicting acquired report identity")
            continue
        seen[key] = report
        buffer.receive(report, received_at_s=receipt)
        if receipt >= cutoff_s-history_s:
            retained.append(dict(received_at_s=receipt, **payload))
    fresh = [r.payload() for r in buffer.reports(now_s=cutoff_s)]
    evidence = dict(now_s=cutoff_s, reports=fresh, history=retained)
    return dict(evidence=evidence, evidence_sha256=digest(evidence),
                fresh_reporter_ids=[r["frame"]["vehicle_id"] for r in fresh],
                history_packet_count=len(retained), source_kind="rsu_received_report",
                neighbor_identity_available_online=False, total_population_known_online=False)


def audit_event_acquisition(event, report_evidence, observation_links=()):
    """Link retrospective event participants to actually received local samples.

``observation_links`` is an OFFLINE acquisition-time crosswalk, one item per CAV
frame: ``reporter_id``, ``timestamp_s``, and front/rear mappings containing
``vehicle_id``, ``gap_m`` and ``relative_speed_mps`` (or None). A link is credited
only after matching both saved anonymous measurements exactly. Its privileged
identity is never exposed as a report feature. Missing crosswalks mean unknown
attribution, not an observed absence of a vehicle.
"""
    if not event.get("event_id") or not isfinite(event["start_s"]):
        raise ValueError("event requires an identity and finite onset")
    targets = event["vehicle_ids"]
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("event participants must be distinct and nonempty")
    evidence = report_evidence["evidence"]
    if digest(evidence) != report_evidence["evidence_sha256"]:
        raise ValueError("report evidence binding changed")
    now = evidence["now_s"]
    links = {}
    for link in observation_links:
        key = (link["reporter_id"], link["timestamp_s"])
        if key in links:
            raise ValueError("duplicate offline acquisition crosswalk")
        links[key] = link
    hits, missing, matched = {v: [] for v in targets}, [], 0
    fresh_keys = {(r["frame"]["vehicle_id"], r["frame"]["timestamp_s"]) for r in evidence["reports"]}
    for row in evidence["history"]:
        frame = row["frame"]
        key = (frame["vehicle_id"], frame["timestamp_s"])
        if row["received_at_s"] > now or frame["timestamp_s"] > now:
            raise ValueError("future packet in acquisition evidence")
        common = dict(reporter_id=key[0], acquired_at_s=key[1], received_at_s=row["received_at_s"],
                      acquisition_age_s=now-key[1], fresh_current_report=key in fresh_keys)
        if key[0] in hits:
            hits[key[0]].append(dict(common, channel="cav_self_state", identity_available_online=True))
        link = links.get(key)
        if link is None:
            if frame.get("front") is not None or frame.get("rear") is not None:
                missing.append(dict(reporter_id=key[0], timestamp_s=key[1]))
            continue
        for direction in ("front", "rear"):
            detected, mapped = frame.get(direction), link.get(direction)
            if (detected is None) != (mapped is None):
                raise ValueError("crosswalk disagrees with acquired detection missingness")
            if detected is None:
                continue
            if any(_canonical(detected[k]) != _canonical(mapped[k]) for k in ("gap_m", "relative_speed_mps")):
                raise ValueError("crosswalk disagrees with acquired anonymous measurement")
            matched += 1
            if mapped["vehicle_id"] in hits:
                hits[mapped["vehicle_id"]].append(dict(common, channel=f"anonymous_{direction}_detection",
                    identity_available_online=False, gap_m=detected["gap_m"],
                    relative_speed_mps=detected["relative_speed_mps"]))
    return dict(schema="event_acquisition_audit/1", offline_only=True, event_id=event["event_id"],
        event_start_s=event["start_s"], cutoff_s=now, lead_to_event_s=event["start_s"]-now,
        future_event_at_cutoff=event["start_s"] > now, evidence_sha256=report_evidence["evidence_sha256"],
        participants=[dict(vehicle_id=v, observed_samples=hits[v], state_observed_before_cutoff=bool(hits[v]))
                      for v in targets],
        identity_crosswalk_complete=not missing, missing_crosswalks=missing, matched_neighbor_samples=matched,
        event_prediction_demonstrated=False, causal_mediation_demonstrated=False,
        note="Participant state availability is retrospective and does not establish advance event warning or benefit predictability.")
