"""Offline measured single-wall delivery, separate from policy phase assertions.

Readiness uses the 1 s decision clock and target-speed tolerance. HOLD uses every
0.5 s measured boundary. Issued-command records verify handback separately;
neither compactness nor telemetry alone establishes safety or traffic benefit.
"""
from dataclasses import asdict, is_dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _grid(time_s, start_s, dt_s):
    return abs((time_s-start_s)/dt_s-round((time_s-start_s)/dt_s)) <= 1e-7


def _configuration(proposal):
    p = asdict(proposal) if is_dataclass(proposal) else dict(proposal)
    profile = p["profile"]
    lanes, members = tuple(p["required_lanes"]), tuple(p["members"])
    assigned = {m["vehicle_id"]: m["lane_id"] for m in members}
    if (not lanes or len(set(lanes)) != len(lanes) or len(assigned) != len(members)
            or len(assigned) != len(lanes) or set(assigned.values()) != set(lanes)):
        raise ValueError("one distinct named member per explicit map lane required")
    numeric = ("assembly_deadline_s", "hold_s", "taper_s", "formed_span_m",
               "formed_speed_spread_mps", "target_speed_tolerance_mps", "control_dt_s")
    if any(not _number(profile[k]) or profile[k] < 0 for k in numeric):
        raise ValueError("finite nonnegative profile values required")
    if profile["control_dt_s"] != 1. or min(profile[k] for k in
            ("assembly_deadline_s", "hold_s", "taper_s", "formed_span_m")) <= 0:
        raise ValueError("positive phase durations and the adopted 1 s control clock required")
    if type(profile["readiness_samples"]) is not int or profile["readiness_samples"] < 2:
        raise ValueError("at least two consecutive control readiness samples required")
    if (not _number(p["start_s"]) or p["start_s"] < 0 or not _grid(p["start_s"], 0., 1.)
            or not _number(p["target_speed_mps"]) or p["target_speed_mps"] < 0):
        raise ValueError("valid proposal start and target required")
    return p, profile, assigned


def join_physics_and_phases(physics_samples, control_samples, *, start_s, member_ids):
    """Join mechanics' separate offline 0.5 s census and 1 s phase records.

    The physics member list must be a complete OFFLINE census of the registered
    cohort. A missing member is then explicitly absent, not missing CAV coverage.
    This helper must never impute presence/absence in online report packets.
    """
    controls = {}
    for row in control_samples:
        now = row["time_s"]
        if not _number(now) or not _grid(now, start_s, 1.) or now in controls:
            raise ValueError("distinct control-grid phase records required")
        controls[now] = row
    joined = []
    for raw in physics_samples:
        now = raw["time_s"]
        if not _number(now):
            raise ValueError("finite physics clock required")
        if now < start_s:
            continue
        if not _grid(now, start_s, .5):
            raise ValueError("adopted 0.5 s physics samples required")
        command_time = start_s + int((now-start_s)//1.)
        if command_time not in controls:
            raise ValueError("missing controlling phase for a physical sample")
        phase = controls[command_time]
        observed = {m["id"]: dict(m, present=m.get("present", True)) for m in raw["members"]}
        if len(observed) != len(raw["members"]):
            raise ValueError("duplicate offline cohort member")
        joined.append(dict(time_s=now, phase=phase["phase"], reason=phase.get("reason"),
            command_time_s=command_time,
            members=[observed.get(v, dict(id=v, present=False)) for v in member_ids]))
    return joined


def _geometry(row, assigned, profile, target):
    members = row["members"]
    if not isinstance(members, list):
        raise ValueError("measured members must be a list")
    by_id = {m["id"]: m for m in members}
    if len(by_id) != len(members):
        raise ValueError("duplicate measured member identity")
    absent, unknown, valid, wrong_lane = [], [], [], []
    for member, lane in assigned.items():
        m = by_id.get(member)
        if m is None:
            unknown.append(member)
            continue
        if m.get("present") is False:
            absent.append(member)
            continue
        if not all(_number(m.get(k)) for k in ("distance_m", "speed_mps")):
            unknown.append(member)
            continue
        if m["speed_mps"] < 0:
            raise ValueError("negative measured speed")
        if m.get("lane") != lane:
            wrong_lane.append(member)
        valid.append(m)
    complete = len(valid) == len(assigned) and not wrong_lane and not (set(by_id)-set(assigned))
    span = max(m["distance_m"] for m in valid)-min(m["distance_m"] for m in valid) if valid else None
    spread = max(m["speed_mps"] for m in valid)-min(m["speed_mps"] for m in valid) if valid else None
    compact = complete and span <= profile["formed_span_m"] and spread <= profile["formed_speed_spread_mps"]
    settled = compact and all(abs(m["speed_mps"]-target) <= profile["target_speed_tolerance_mps"] for m in valid)
    return dict(span_m=span, speed_spread_mps=spread, compact=bool(compact), settled=bool(settled),
                absent=absent, measurement_unknown=unknown, wrong_lane=wrong_lane)


def _commands(telemetry, assigned):
    records = {}
    for row in telemetry:
        if row.get("kind") not in {None, "command", "issued_command"} or row.get("vehicle_id") not in assigned:
            continue
        now, vehicle = row.get("timestamp_s"), row["vehicle_id"]
        if not _number(now) or row.get("owner") not in {"native", "damping", "wall"}:
            raise ValueError("invalid issued-command clock or owner")
        key = (now, vehicle)
        if key in records:
            raise ValueError("duplicate issued command for the same member and clock")
        records[key] = row
    return records


def _handback(now, geometry, commands, assigned):
    failures, native_members, live_returned = [], [], []
    for member in assigned:
        row = commands.get((now, member))
        if member in geometry["absent"]:
            if row is not None:
                failures.append("command_to_absent_member")
            continue
        if row is None:
            failures.append("missing_terminal_owner_record")
            continue
        if row.get("wall_request_mps2") is not None or row["owner"] == "wall":
            failures.append("wall_command_retained_at_terminal")
            continue
        if "wall_request_mps2" not in row:
            failures.append("terminal_wall_mask_unknown")
            continue
        if row["owner"] == "native":
            native_members.append(member)
            prior = [r for (t, v), r in commands.items() if v == member and t < now]
            prior = max(prior, key=lambda r: r["timestamp_s"]) if prior else None
            if prior and prior["owner"] == "wall" and row.get("released_to_native") is not True:
                failures.append("native_override_not_released")
        elif not all(_number(row.get(k)) for k in ("issued_acceleration_mps2", "issued_target_speed_mps")):
            failures.append("damping_reengagement_command_missing")
        live_returned.append(member)
    return dict(verified=not failures, failures=sorted(set(failures)),
                native_members=native_members, returned_live_members=live_returned,
                absent_members=geometry["absent"])


def score_formation(samples, telemetry, *, proposal, arm="wall"):
    """Score a complete recorded attempt, keeping genuine aborts as outcomes.

    Trailing observations after the first release/abort/refusal do not require
    departed-member geometry. Command records are still checked for illicit
    wall re-engagement after that terminal event in this single-attempt run.
    A malformed/missing physics clock raises; observed missing members produce
    explicit delivery failure, not a fabricated measurement or dropped case.
    """
    if arm not in {"wall", "assembly_only"}:
        raise ValueError("score an attempted wall or declared assembly-only diagnostic")
    p, profile, assigned = _configuration(proposal)
    start, target = p["start_s"], p["target_speed_mps"]
    source = []
    for row in samples:
        if not _number(row.get("time_s")):
            raise ValueError("invalid formation observation clock")
        if row["time_s"] >= start:
            source.append(row)
    if not source or source[0]["time_s"] != start:
        raise ValueError("formation observations must begin at the exact decision")
    terminal_index = next((i for i, row in enumerate(source) if row["phase"] in {"release", "abort", "refused"}), None)
    rows = source if terminal_index is None else source[:terminal_index+1]
    for index, row in enumerate(rows):
        if abs(row["time_s"]-start-index*.5) > 1e-7:
            raise ValueError("missing, duplicate or irregular 0.5 s formation sample")
        if row["phase"] not in {"assembly", "hold", "taper", "release", "abort", "refused", "damping"}:
            raise ValueError("unknown formation lifecycle phase")
    geometry = [_geometry(row, assigned, profile, target) for row in rows]
    issued = _commands(telemetry, assigned)
    failures, readiness, entry_valid = set(), 0, False
    readiness_trace, entry_time, first_ready, hold_started = [], None, None, None
    previous = None
    transitions = dict(assembly={"assembly", "hold", "taper", "release", "abort"},
        hold={"hold", "taper", "release", "abort"}, taper={"taper", "release", "abort"},
        damping={"damping"}, refused={"refused"}, release={"release"}, abort={"abort"})
    for row, observed in zip(rows, geometry):
        now, phase = row["time_s"], row["phase"]
        on_control = _grid(now, start, 1.)
        if previous is None and phase not in {"assembly", "abort", "refused", "damping"}:
            failures.add("lifecycle_did_not_start_in_assembly")
        if previous is not None and phase not in transitions[previous]:
            failures.add("invalid_lifecycle_transition")
        if phase != previous and previous is not None and not on_control:
            failures.add("phase_changed_off_control_grid")
        if on_control and entry_time is None:
            readiness = readiness+1 if observed["settled"] else 0
            readiness_trace.append(dict(time_s=now, consecutive_control_samples=readiness,
                                        target_settled=observed["settled"]))
            if first_ready is None and readiness >= profile["readiness_samples"]:
                first_ready = now
        entering = (phase == "hold" and previous != "hold") or (
            arm == "assembly_only" and phase in {"taper", "release"} and previous == "assembly")
        if entering:
            entry_time = now
            entry_valid = (on_control and readiness >= profile["readiness_samples"]
                and now-start <= profile["assembly_deadline_s"])
            if not entry_valid:
                failures.add("entry_without_current_target_settled_readiness")
            if phase == "hold":
                hold_started = now
        if phase == "assembly" and now-start >= profile["assembly_deadline_s"]:
            failures.add("assembly_deadline_missed")
        if phase in {"assembly", "hold", "taper"}:
            if observed["absent"]:
                failures.add("member_absent_during_active_treatment")
            if observed["measurement_unknown"]:
                failures.add("member_measurement_missing_during_active_treatment")
            if observed["wrong_lane"]:
                failures.add("member_lane_loss_during_active_treatment")
        previous = phase

    hold_seconds, continuous, longest, taper_seconds = 0., 0., 0., 0.
    hold_invalid = False
    for index, row in enumerate(rows[:-1]):
        if row["phase"] == "hold":
            if geometry[index]["compact"] and geometry[index+1]["compact"]:
                hold_seconds += .5
                continuous += .5
                longest = max(longest, continuous)
            else:
                continuous, hold_invalid = 0., True
                failures.add("noncompact_or_missing_hold_boundary")
        else:
            continuous = 0.
        if row["phase"] == "taper":
            taper_seconds += .5
    hold_delivered = bool(entry_valid and arm == "wall" and not hold_invalid and
                          longest >= profile["hold_s"])
    if arm == "wall" and not hold_delivered:
        failures.add("required_continuous_hold_not_delivered")
    terminal = rows[-1] if terminal_index is not None else None
    handback = dict(verified=False, failures=["terminal_not_observed"], native_members=[],
                    returned_live_members=[], absent_members=[])
    if terminal:
        handback = _handback(terminal["time_s"], geometry[-1], issued, assigned)
        failures.update(handback["failures"])
        if not _grid(terminal["time_s"], start, 1.):
            failures.add("terminal_off_control_grid")
        if any(t > terminal["time_s"] and (r.get("owner") == "wall" or r.get("wall_request_mps2") is not None)
               for (t, _), r in issued.items()):
            failures.add("wall_command_after_terminal")
    else:
        failures.add("terminal_not_observed")
    terminal_phase = terminal["phase"] if terminal else None
    terminal_reason = terminal.get("reason") if terminal else None
    native_handoff = bool(terminal_phase == "release" and handback["verified"] and
        handback["native_members"] and terminal_reason == "identical_damping_requests_native_explicit_handback")
    taper_complete = taper_seconds >= profile["taper_s"]
    if terminal_phase == "release" and not (taper_complete or native_handoff):
        failures.add("taper_incomplete_without_explicit_native_handback")
    if terminal_phase == "abort":
        failures.add("aborted")
    if terminal_phase == "refused":
        failures.add("refused_without_attempt")
    delivered = bool(arm == "wall" and hold_delivered and terminal_phase == "release" and
                     handback["verified"] and not failures)
    diagnostic_delivered = bool(arm == "assembly_only" and entry_valid and terminal_phase == "release"
                               and handback["verified"] and not failures)
    return dict(schema="measured_formation_delivery/1", arm=arm,
        delivered=delivered, assembly_only_delivered=diagnostic_delivered,
        assembly_complete=entry_valid, first_measured_readiness_s=first_ready,
        assembly_complete_at_s=entry_time if entry_valid else None,
        readiness_clock_s=1., readiness_samples_required=profile["readiness_samples"],
        readiness_trace=readiness_trace, physics_clock_s=.5,
        initial_span_m=geometry[0]["span_m"],
        maximum_observed_span_m=max((g["span_m"] for g in geometry if g["span_m"] is not None), default=None),
        compact_hold_delivered=hold_delivered, compact_hold_seconds=hold_seconds,
        longest_contiguous_compact_hold_seconds=longest, hold_started_at_s=hold_started,
        taper_seconds=taper_seconds, full_taper_completed=taper_complete,
        explicit_native_handback=native_handoff, terminal_phase=terminal_phase,
        terminal_time_s=terminal["time_s"] if terminal else None, terminal_reason=terminal_reason,
        terminal_ownership=handback, release_ownership_verified=bool(terminal_phase == "release" and handback["verified"]),
        abort_ownership_verified=bool(terminal_phase == "abort" and handback["verified"]),
        complete_observation_horizon=terminal is not None,
        scored_physics_samples=len(rows), ignored_postterminal_geometry_samples=len(source)-len(rows),
        failure_reasons=sorted(failures), safety_established=False,
        interpretation="Offline measured formation and command handback; no efficacy or safety guarantee.")


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def score_formation_files(formation_path, telemetry_path, *, proposal, arm="wall"):
    """Stream member command rows from full-run telemetry and bind input files."""
    result = score_formation(_read_jsonl(formation_path), _read_jsonl(telemetry_path), proposal=proposal, arm=arm)
    result["source_sha256"] = dict(formation=_file_hash(formation_path), telemetry=_file_hash(telemetry_path))
    return result
