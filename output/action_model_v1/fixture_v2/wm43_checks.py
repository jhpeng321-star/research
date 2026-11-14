"""WM-43 machine-evaluated per-run acceptance checks (pure functions).

Every check parses only run-dir evidence: telemetry.jsonl, intervals.jsonl
and scenario_manifest.json (plus the receipt passed by the child).  A check
returns ``ok`` with a detail string; no check reads memory state, so the
receipts are independently re-evaluable after the run.
"""
from __future__ import annotations

import json
from pathlib import Path

from wm43_plan import (A3_ABORT_LATEST_S, ARM_SPECS, ASSEMBLY_DEADLINE_S,
                       ASSEMBLY_SPAN_TOL_M, ASSEMBLY_TIMEOUT_S,
                       ASSEMBLY_TIMEOUT_TOL_S, CHECK_LISTS, CHECKS_SCHEMA,
                       CLOSE_GAIN_PER_S, CLOSE_MAX_DELTA_MPS, HORIZON_TICKS,
                       HOLD_SPEED_TOL_MPS, MIN_HOLD_SAMPLES,
                       SMOKE_CHECKS_SCHEMA, WALL_LAW_VERSION)

WALL_EPS = 1e-3
PACING_TOL = 1e-3
RELEASE_WINDOW_S = 1.0 + 1e-6


def load_rows(run_dir: Path) -> tuple[list[dict], list[dict], dict]:
    telemetry, intervals = [], []
    with (Path(run_dir) / "telemetry.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                telemetry.append(json.loads(line))
    with (Path(run_dir) / "intervals.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                intervals.append(json.loads(line))
    manifest = json.loads((Path(run_dir) / "scenario" / "scenario_manifest.json")
                          .read_text(encoding="utf-8"))
    return telemetry, intervals, manifest


def by_kind(telemetry: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in telemetry:
        out.setdefault(row["kind"], []).append(row)
    return out


def _check(cid: str, ok: bool, detail: str) -> dict:
    return {"id": cid, "ok": bool(ok), "detail": detail}


def _single_writer_scan(commands: list[dict], mixed: list[dict],
                        members: tuple) -> tuple[list, list]:
    """Per-tick single-writer scan shared by the full and smoke check sets.

    Returns (duplicate_writes, path_mismatches): every control tick must
    carry exactly one ownership record per live roster vehicle; wall-owner
    rows must be exactly the registered member set.
    """
    commands_by_tick: dict[float, list[dict]] = {}
    for row in commands:
        commands_by_tick.setdefault(row["timestamp_s"], []).append(row)
    mixed_by_tick: dict[float, list[dict]] = {}
    for row in mixed:
        mixed_by_tick.setdefault(row["timestamp_s"], []).append(row)
    duplicates, mismatches = [], []
    for tick in sorted(set(commands_by_tick) | set(mixed_by_tick)):
        crows = commands_by_tick.get(tick, [])
        ids = [r["vehicle_id"] for r in crows]
        if len(ids) != len(set(ids)):
            duplicates.append(tick)
        if sorted(ids) != sorted(r["vehicle_id"] for r in mixed_by_tick.get(tick, [])):
            mismatches.append(tick)
        wall_ids = sorted(r["vehicle_id"] for r in crows if r["owner"] == "wall")
        if wall_ids and wall_ids != sorted(members):
            mismatches.append(f"{tick}:wall_set")
    return duplicates, mismatches


def evaluate(world: dict, run_dir: Path, receipt: dict | None = None) -> dict:
    """Evaluate the arm's registered check list for one run directory."""
    arm = world["arm"]
    spec = ARM_SPECS[arm]
    members = tuple(spec["members"])
    extras = tuple(spec["extra_cavs"])
    roster = members + extras
    wall_speed = float(world["wall_speed_mps"])
    telemetry, intervals, manifest = load_rows(run_dir)
    kinds = by_kind(telemetry)
    commands = kinds.get("command", [])
    responses = kinds.get("response", [])
    mixed = kinds.get("mixed_assignment", [])
    events = kinds.get("wall_event", [])
    proximity = kinds.get("wall_proximity", [])
    injections = kinds.get("scripted_actuator", [])
    frame_table = {row["timestamp_s"]: row["members"]
                   for row in kinds.get("wall_frames", [])}

    commands_by_tick: dict[float, list[dict]] = {}
    for row in commands:
        commands_by_tick.setdefault(row["timestamp_s"], []).append(row)
    mixed_by_tick: dict[float, list[dict]] = {}
    for row in mixed:
        mixed_by_tick.setdefault(row["timestamp_s"], []).append(row)
    phase_of_tick: dict[float, str | None] = {}
    for tick, rows in sorted(mixed_by_tick.items()):
        phases = {r.get("wall_phase") for r in rows if r["vehicle_id"] in members}
        phase_of_tick[tick] = next(iter(phases)) if len(phases) == 1 else None
    # frame speed at tick t per vehicle = last response measured at t (the
    # command that produced it was issued at t-1; the frame at t reads the
    # same simulator state that the t-1 substep left behind)
    frame_speed: dict[str, dict[float, float]] = {}
    for row in responses:  # telemetry is chronological: the last row per
        frame_speed.setdefault(row["vehicle_id"], {})[row["timestamp_s"]] =             row["measured_speed_mps"]  # (vehicle, time) is the latest substep
    depart_speed = {v["vehicle_id"]: float(v["depart_speed_mps"])
                    for v in manifest["roster"]}

    assembly = [e for e in events if e["event"] == "assembly_start"]
    assembled = [e for e in events if e["event"] == "assembled"]
    released = [e for e in events if e["event"] == "released"]
    aborts = [e for e in events if e["event"] == "safety_abort"]
    abort_ts = aborts[0]["timestamp_s"] if aborts else None
    release_ts = released[0]["timestamp_s"] if released else None

    results: list[dict] = []

    # ---- common -------------------------------------------------------
    interval_rows_ok = len(intervals) == HORIZON_TICKS
    complete = bool(receipt.get("exit_status") == "complete") if receipt else True
    results.append(_check(
        "C01_run_complete",
        complete and interval_rows_ok
        and (not intervals or abs(intervals[-1]["end_s"] - HORIZON_TICKS) <= 1e-6),
        f"receipt={receipt.get('exit_status') if receipt else 'n/a'}, "
        f"interval_rows={len(intervals)}/{HORIZON_TICKS}, "
        f"ticks_executed={receipt.get('summary', {}).get('ticks_executed') if receipt else 'n/a'}"))
    collisions = sum(int(r["collision_involvements"]) for r in intervals)
    results.append(_check("C02_zero_collisions", collisions == 0,
                          f"collision_involvements={collisions}"))
    teleports = sum(int(r["starting_teleports"]) for r in intervals)
    results.append(_check("C03_zero_teleports", teleports == 0,
                          f"starting_teleports={teleports}"))
    dup, mismatch = _single_writer_scan(commands, mixed, members)
    wall_active_ticks = {e["timestamp_s"] for e in assembly} | {
        r["timestamp_s"] for r in commands if r["owner"] == "wall"}
    results.append(_check("C04_single_writer_path", not dup and not mismatch,
                          f"duplicate_writes_at={dup[:5]}, path_mismatches={mismatch[:5]} "
                          f"(ticks_checked={len(set(commands_by_tick) | set(mixed_by_tick))})"))
    speed_rows = [r for r in mixed if r.get("wall_speed_mps") is not None]
    speed_ok = (manifest["wall_speed_mps"] == wall_speed == 8.0
                and all(abs(r["wall_speed_mps"] - wall_speed) <= WALL_EPS for r in speed_rows)
                and all(abs(e["wall_speed_mps"] - wall_speed) <= WALL_EPS
                        for e in events if "wall_speed_mps" in e))
    results.append(_check(
        "C05_speed_mode_31_contract", speed_ok,
        "wall speed strictly above the 2.0 safety floor and identical across "
        f"plan/events/telemetry ({wall_speed} m/s); ownership validated speedMode31 "
        "every tick (runtime contract; the run completed)"))
    cmd_ids = {r["vehicle_id"] for r in commands}
    roster_ok = (set(manifest_roster_ids(manifest)) == cmd_ids == set(roster)
                 and all(e.get("wall_id") == spec["wall_id"] for e in events)
                 and all(sorted(e["members"]) == sorted(members)
                         for e in assembled))
    results.append(_check("C06_roster_and_wall_identity", roster_ok,
                          f"telemetry_ids={sorted(cmd_ids)} vs roster={sorted(roster)}"))

    law_events = [e for e in events if e["event"] in ("assembly_start", "assembled")]
    results.append(_check(
        "C07_law_version_identity",
        bool(assembly) and all(e.get("law_version") == WALL_LAW_VERSION
                               for e in law_events),
        f"law_version values={sorted({str(e.get('law_version')) for e in law_events})} "
        f"(expected {WALL_LAW_VERSION!r}) over {len(law_events)} lifecycle events"))

    # ---- shared wall-lifecycle checks ---------------------------------
    # A4 (r3 F1): the A3 plan deferral stages the wall on the first tick
    # where the FULL roster is live (real SUMO inserts depart=0 vehicles
    # during the first step, so that tick is 1.0, not 0.0); assembly_start
    # therefore lands on the plan_submitted tick, recorded per run in the
    # plan_submitted telemetry rows.
    submitted_rows = kinds.get("plan_submitted", [])
    submitted_ts = (submitted_rows[0]["timestamp_s"]
                    if len(submitted_rows) == 1 else None)
    assembly_ok = (len(assembly) == 1 and submitted_ts is not None
                   and abs(assembly[0]["timestamp_s"] - submitted_ts) <= 1e-6
                   and assembly[0]["timestamp_s"] <= ASSEMBLY_DEADLINE_S
                   and abs(assembly[0]["wall_speed_mps"] - wall_speed) <= WALL_EPS)
    results.append(_check(
        "A1_1_assembly_start", assembly_ok,
        f"assembly_start events={len(assembly)}, "
        f"at={assembly[0]['timestamp_s'] if assembly else None} vs "
        f"plan_submitted={submitted_ts}, "
        f"speed={assembly[0]['wall_speed_mps'] if assembly else None} "
        f"(deadline {ASSEMBLY_DEADLINE_S}s)"))
    if arm in ("A1", "A2", "A3"):
        assembly_ok = (len(assembled) == 1 and members == tuple(sorted(assembled[0]["members"]))
                       and 0.0 < assembled[0]["timestamp_s"] <= ASSEMBLY_DEADLINE_S) \
            if assembled else False
        results.append(_check(
            "A1_2_assembled_event", assembly_ok,
            f"assembled events={len(assembled)}, at={assembled[0]['timestamp_s'] if assembled else None}, "
            f"members={assembled[0].get('members') if assembled else None}"))
    hold_ticks = sorted(t for t, phase in phase_of_tick.items() if phase == "hold")
    if spec["injection"] is not None:
        # A3: the post-injection hold ticks violate the 8 m/s hold BY DESIGN
        # (the scripted fault injector pins the front member); the shared
        # hold check covers the pre-injection window only.
        hold_ticks = [t for t in hold_ticks if t < spec["injection"]["start_s"]]
    hold_speed_ok, hold_detail = _hold_speed_check(responses, hold_ticks, members,
                                                   release_ts, abort_ts, wall_speed,
                                                   frame_speed, depart_speed)
    if arm != "A5":
        results.append(_check("A1_3_hold_speed_tolerance", hold_speed_ok, hold_detail))
    assembled_ts = assembled[0]["timestamp_s"] if assembled else None
    results.append(_check(
        "A1_7_wall_pacing_identity", *_wall_pacing_check(
            commands_by_tick, frame_table, members, wall_speed, assembled_ts)))

    # ---- arm-specific --------------------------------------------------
    if arm in ("A1", "A4"):
        release_ok = (len(released) == 1
                      and released[0].get("handback") == "native"
                      and spec["release_at_s"] - 1.0 <= release_ts
                      <= spec["release_at_s"] + RELEASE_WINDOW_S) if released else False
        results.append(_check(
            "A1_4_released_event", release_ok,
            f"released events={len(released)}, at={release_ts}, "
            f"handback={released[0].get('handback') if released else None}"))
        hb_ok, hb_detail = _handback_check(commands_by_tick, members, release_ts, None)
        results.append(_check("A1_5_native_handback_receipts", hb_ok, hb_detail))
        late_wall = [r["timestamp_s"] for r in commands
                     if r["owner"] == "wall" and release_ts is not None
                     and r["timestamp_s"] >= release_ts]
        results.append(_check(
            "A1_6_no_wall_after_release_no_abort",
            not late_wall and not aborts,
            f"wall_rows_after_release={len(late_wall)}, safety_abort_events={len(aborts)}"))
        if arm == "A4":
            results.extend(_a4_checks(frame_table, commands_by_tick, members,
                                      wall_speed, assembly, assembled, aborts,
                                      proximity))
    elif arm == "A2":
        release_ok = (len(released) == 1 and not aborts
                      and released[0].get("handback") == "native"
                      and spec["release_at_s"] <= release_ts
                      <= spec["release_at_s"] + RELEASE_WINDOW_S) if released else False
        results.append(_check(
            "A2_1_no_abort_release_at_300", release_ok,
            f"safety_abort_events={len(aborts)}, released_at={release_ts} "
            f"(expected {spec['release_at_s']}), handback="
            f"{released[0].get('handback') if released else None}"))
        follow_id = extras[0]
        follow_mixed = [r for r in mixed if r["vehicle_id"] == follow_id]
        follow_cmds = [r for r in commands if r["vehicle_id"] == follow_id]
        committed = {r["action"] for r in follow_mixed}
        # (a) the dispatcher must never reassign a NON-MEMBER's committed
        # action: every mixed_assignment row stays 'damping' (r3 evidence:
        # it always did - the r3 failure was check-side, not dispatcher-side)
        reassigned = committed != {"damping"}
        # (b) a native-owned follower command row is legal ONLY as the
        # registered damping-refusal fallback (the frozen controller refused
        # with a documented reason, e.g. missing_or_uncovered_leader, and
        # ownership released that tick).  'selected_native' would mean the
        # planner dropped the assignment; None would mean an undocumented
        # owner switch - both are real defects.
        fallback_rows = [r for r in follow_cmds if r["owner"] == "native"]
        illegal = [r for r in fallback_rows
                   if r["fallback_reason"] in (None, "selected_native")]
        reasons = sorted({str(r["fallback_reason"]) for r in fallback_rows})
        owners = sorted({r["owner"] for r in follow_cmds})
        results.append(_check(
            "A2_2_follower_damping_path",
            bool(follow_cmds) and committed == {"damping"} and not illegal,
            f"committed_actions={sorted(committed)} (mixed rows "
            f"{len(follow_mixed)}), owners={owners}, damping_rows="
            f"{sum(1 for r in follow_cmds if r['owner'] == 'damping')}, "
            f"native_fallback_rows={len(fallback_rows)} reasons={reasons} "
            f"(registered damping-refusal path), illegal={len(illegal)}, "
            f"reassigned={reassigned}"))
        active_ticks = sorted(wall_active_ticks)
        missing = [t for t in active_ticks
                   if not any(abs(p["timestamp_s"] - t) <= 1e-9 for p in proximity)]
        region = [p for p in proximity if p["would_abort"]]
        min_gap = min((p["rear_gap_m"] for p in proximity), default=None)
        results.append(_check(
            "A2_3_surveillance_no_region_entry",
            bool(proximity) and not missing and not region,
            f"proximity_rows={len(proximity)}, ticks_missing={len(missing)}, "
            f"region_entries={len(region)}, min_rear_gap_m={min_gap}"))
        cross_ok, cross_detail = _cross_path_check(mixed_by_tick, commands_by_tick,
                                                   members, extras, release_ts)
        results.append(_check("A2_4_cross_path_single_writer", cross_ok, cross_detail))
        hb_ok, hb_detail = _handback_check(commands_by_tick, members, release_ts, None)
        results.append(_check("A2_5_handback_receipts_at_release", hb_ok, hb_detail))
    elif arm == "A5":
        results.extend(_a5_checks(spec, injections, aborts, assembly, assembled,
                                  released, commands_by_tick, members))
    else:  # A3
        inj_ok = (bool(injections)
                  and all(r["vehicle_id"] == spec["injection"]["member"] for r in injections)
                  and all(abs(r["value_mps"] - spec["injection"]["target_mps"]) <= WALL_EPS
                          for r in injections)
                  and injections[0]["timestamp_s"] >= spec["injection"]["start_s"] - 1e-9)
        results.append(_check(
            "A3_1_injections_logged", inj_ok,
            f"injections={len(injections)}, first={injections[0]['timestamp_s'] if injections else None}, "
            f"member={spec['injection']['member']}, target={spec['injection']['target_mps']} m/s"))
        abort_ok = (len(aborts) == 1
                    and aborts[0].get("reason") == "member_span_exceeded"
                    and spec["injection"]["start_s"] < abort_ts <= A3_ABORT_LATEST_S) \
            if aborts else False
        results.append(_check(
            "A3_2_abort_reason_span", abort_ok,
            f"abort events={len(aborts)}, reason={aborts[0].get('reason') if aborts else None}, "
            f"at={abort_ts} (window ({spec['injection']['start_s']}, {A3_ABORT_LATEST_S}])"))
        hb_ok, hb_detail = _handback_check(commands_by_tick, members, None, abort_ts)
        results.append(_check("A3_3_same_tick_handback", hb_ok, hb_detail))
        late_inj = [r["timestamp_s"] for r in injections
                    if abort_ts is not None and r["timestamp_s"] > abort_ts - 1.0]
        results.append(_check(
            "A3_4_injections_cease",
            bool(injections) and not late_inj,
            f"last_injection={injections[-1]['timestamp_s'] if injections else None}, "
            f"abort_at={abort_ts}, after_bound={len(late_inj)}"))
        pre_hold = [t for t in hold_ticks if t < spec["injection"]["start_s"]]
        pre_ok, pre_detail = _hold_speed_check(responses, pre_hold, members, None, None,
                                               wall_speed, frame_speed, depart_speed,
                                               min_samples=10)
        results.append(_check("A3_5_hold_until_injection", pre_ok, pre_detail))

    expected = list(CHECK_LISTS[arm])
    got = sorted(c["id"] for c in results)
    if got != sorted(expected):
        results.append(_check("CHECK_LIST_COVERAGE", False,
                              f"evaluated {got} != registered {sorted(expected)}"))
    return dict(schema=CHECKS_SCHEMA, run=world["run"], arm=arm, seed=world["seed"],
                checks=results, ok=all(c["ok"] for c in results))


def manifest_roster_ids(manifest: dict) -> list[str]:
    return [v["vehicle_id"] for v in manifest["roster"]]


def _hold_speed_check(responses, hold_ticks, members, release_ts, abort_ts,
                      wall_speed, frame_speed, depart_speed, *,
                      min_samples: int = MIN_HOLD_SAMPLES):
    """Measured member speeds stay within the hold tolerance while held."""
    bad, samples = [], {m: 0 for m in members}
    hold_set = set(hold_ticks)
    for row in responses:
        if row["vehicle_id"] not in members:
            continue
        cmd_tick = row["command_timestamp_s"]
        if cmd_tick not in hold_set:
            continue
        if release_ts is not None and cmd_tick >= release_ts:
            continue
        if abort_ts is not None and cmd_tick >= abort_ts:
            continue
        samples[row["vehicle_id"]] += 1
        if abs(row["measured_speed_mps"] - wall_speed) > HOLD_SPEED_TOL_MPS:
            bad.append((row["vehicle_id"], row["timestamp_s"], row["measured_speed_mps"]))
    enough = all(n >= min_samples for n in samples.values())
    return (not bad and enough,
            f"violations={bad[:5]} (of {sum(samples.values())} samples), "
            f"samples_per_member={samples}, tol={HOLD_SPEED_TOL_MPS} m/s")


def _expected_wall_accel(tick_frames: dict, member: str, wall_speed: float,
                         assembling: bool) -> tuple[float, float]:
    """The wall-v2.1 law re-derived from the telemetry frame content:
    front (max s_m) or HOLD -> v_wall; rear in ASSEMBLY -> v_wall +
    min(2.0, 0.1 * gap_to_front); a = clip((target - v)/1, -3, +1.5)."""
    v = float(tick_frames[member]["speed_mps"])
    target = wall_speed
    if assembling:
        front = max(float(m["s_m"]) for m in tick_frames.values())
        gap = max(0.0, front - float(tick_frames[member]["s_m"]))
        target = wall_speed + min(CLOSE_MAX_DELTA_MPS, CLOSE_GAIN_PER_S * gap)
    a = min(max((target - v) / 1.0, -3.0), 1.5)
    return a, max(0.0, v + a)


def _is_assembling(tick: float, assembled_ts) -> bool:
    """The assembled event is emitted on the tick whose commands still used
    the assembly law; HOLD starts the next tick."""
    return assembled_ts is None or tick <= assembled_ts + 1e-9


def _wall_pacing_check(commands_by_tick, frame_table, members,
                       wall_speed, assembled_ts):
    """wall rows satisfy the wall-v2.1 law exactly, on EVERY wall row.

    Frames come from the `wall_frames` telemetry rows (the very speeds and
    downstream coordinates the session decided on), so no row is skipped
    (the v1 check skipped ticks without a same-timestamp response sample).
    A missing wall_frames row for a wall command is a failure.
    """
    bad, missing, verified = [], [], 0
    for tick, rows in sorted(commands_by_tick.items()):
        for row in rows:
            if row["owner"] != "wall" or row["vehicle_id"] not in members:
                continue
            if row["fallback_reason"] is not None:
                continue
            tick_frames = frame_table.get(tick)
            if tick_frames is None or row["vehicle_id"] not in tick_frames:
                missing.append((row["vehicle_id"], tick))
                continue
            expected_a, expected_target = _expected_wall_accel(
                tick_frames, row["vehicle_id"], wall_speed,
                _is_assembling(tick, assembled_ts))
            verified += 1
            if (abs(row["issued_acceleration_mps2"] - expected_a) > PACING_TOL
                    or abs(row["issued_target_speed_mps"] - expected_target) > PACING_TOL):
                bad.append((row["vehicle_id"], tick, row["issued_acceleration_mps2"],
                            expected_a))
    return (verified > 0 and not bad and not missing,
            f"identity violations={bad[:5]} (verified {verified} wall rows), "
            f"rows_without_wall_frames={missing[:3]}")


def _a4_checks(frame_table, commands_by_tick, members, wall_speed, assembly,
               assembled, aborts, proximity) -> list[dict]:
    out = []
    start_ts = assembly[0]["timestamp_s"] if len(assembly) == 1 else None
    assembled_ts = assembled[0]["timestamp_s"] if len(assembled) == 1 else None
    have_both = start_ts is not None and assembled_ts is not None
    elapsed = (assembled_ts - start_ts) if have_both else None
    ok1 = (have_both and members == tuple(sorted(assembled[0]["members"]))
           and 0.0 < elapsed <= ASSEMBLY_TIMEOUT_S)
    out.append(_check(
        "A4_1_assembled_within_60s", ok1,
        f"assembly_start={start_ts}, assembled={assembled_ts}, "
        f"elapsed={elapsed} (limit {ASSEMBLY_TIMEOUT_S}s)"))
    tick_frames = frame_table.get(assembled_ts) if assembled_ts is not None else None
    if tick_frames:
        pos = [float(m["s_m"]) for m in tick_frames.values()]
        span = max(pos) - min(pos)
        spd_dev = max(abs(float(m["speed_mps"]) - wall_speed) for m in tick_frames.values())
        ok2 = span <= ASSEMBLY_SPAN_TOL_M + 1e-9 and spd_dev <= HOLD_SPEED_TOL_MPS + 1e-9
        detail2 = (f"span_at_assembled={span:.3f} m (tol {ASSEMBLY_SPAN_TOL_M}), "
                   f"max speed dev={spd_dev:.3f} m/s")
    else:
        ok2, detail2 = False, "no wall_frames row at the assembled tick"
    out.append(_check("A4_2_span_at_assembled", ok2, detail2))
    front_bad, rear_bad, front_n, rear_n, closing_n = [], [], 0, 0, 0
    for tick, rows in sorted(commands_by_tick.items()):
        if not _is_assembling(tick, assembled_ts):
            continue
        tf = frame_table.get(tick)
        for row in rows:
            if row["owner"] != "wall" or row["vehicle_id"] not in members:
                continue
            if tf is None or row["vehicle_id"] not in tf:
                front_bad.append((tick, row["vehicle_id"], "no wall_frames"))
                continue
            front_s = max(float(m["s_m"]) for m in tf.values())
            is_front = float(tf[row["vehicle_id"]]["s_m"]) >= front_s - 1e-12
            v = float(tf[row["vehicle_id"]]["speed_mps"])
            paced = min(max((wall_speed - v) / 1.0, -3.0), 1.5)
            if is_front:
                front_n += 1
                if abs(row["issued_acceleration_mps2"] - paced) > PACING_TOL:
                    front_bad.append((tick, row["vehicle_id"],
                                      row["issued_acceleration_mps2"], paced))
            else:
                rear_n += 1
                gap = front_s - float(tf[row["vehicle_id"]]["s_m"])
                if gap > ASSEMBLY_SPAN_TOL_M:
                    closing_n += 1
                exp_a, _ = _expected_wall_accel(tf, row["vehicle_id"], wall_speed, True)
                if abs(row["issued_acceleration_mps2"] - exp_a) > PACING_TOL:
                    rear_bad.append((tick, row["vehicle_id"],
                                     row["issued_acceleration_mps2"], exp_a))
    out.append(_check(
        "A4_3_front_paced_every_assembly_tick", front_n > 0 and not front_bad,
        f"front rows verified={front_n}, violations={front_bad[:5]} "
        "(front target == v_wall)"))
    out.append(_check(
        "A4_4_rear_closure_identity", rear_n > 0 and closing_n > 0 and not rear_bad,
        f"rear rows verified={rear_n} (of which {closing_n} with gap > "
        f"{ASSEMBLY_SPAN_TOL_M} m, i.e. closure actually exercised), "
        f"violations={rear_bad[:5]}"))
    region = [p for p in proximity if p.get("would_abort")]
    out.append(_check(
        "A4_5_no_rear_end_region_entry", not aborts and not region,
        f"safety_abort_events={len(aborts)}, would_abort_proximity_rows={len(region)} "
        "(the A4 roster has no non-member CAV: surveillance is vacuously silent, "
        "and NO abort of any reason may fire)"))
    return out


def _a5_checks(spec, injections, aborts, assembly, assembled, released,
               commands_by_tick, members) -> list[dict]:
    out = []
    inj = spec["injection"]
    start_ts = assembly[0]["timestamp_s"] if len(assembly) == 1 else None
    inj_ok = (bool(injections)
              and all(r["vehicle_id"] == inj["member"] for r in injections)
              and all(abs(r["value_mps"] - inj["target_mps"]) <= WALL_EPS
                      for r in injections)
              and start_ts is not None
              and injections[0]["timestamp_s"] <= start_ts + 1.0 + 1e-9)
    out.append(_check(
        "A5_1_injections_logged", inj_ok,
        f"injections={len(injections)}, "
        f"first={injections[0]['timestamp_s'] if injections else None}, "
        f"assembly_start={start_ts}, member={inj['member']}, "
        f"target={inj['target_mps']} m/s"))
    abort_ts = aborts[0]["timestamp_s"] if aborts else None
    elapsed = ((abort_ts - start_ts)
               if (abort_ts is not None and start_ts is not None) else None)
    abort_ok = (len(aborts) == 1 and aborts[0].get("reason") == "assembly_timeout"
                and elapsed is not None
                and abs(elapsed - ASSEMBLY_TIMEOUT_S) <= ASSEMBLY_TIMEOUT_TOL_S + 1e-9)
    out.append(_check(
        "A5_2_abort_reason_assembly_timeout", abort_ok,
        f"abort events={len(aborts)}, "
        f"reason={aborts[0].get('reason') if aborts else None}, "
        f"elapsed_since_assembly_start={elapsed} (expected {ASSEMBLY_TIMEOUT_S} +/- "
        f"{ASSEMBLY_TIMEOUT_TOL_S})"))
    hb_ok, hb_detail = _handback_check(commands_by_tick, members, None, abort_ts)
    out.append(_check("A5_3_same_tick_handback", hb_ok, hb_detail))
    late_inj = [r["timestamp_s"] for r in injections
                if abort_ts is not None and r["timestamp_s"] > abort_ts - 1.0]
    out.append(_check(
        "A5_4_injections_cease", bool(injections) and not late_inj,
        f"last_injection={injections[-1]['timestamp_s'] if injections else None}, "
        f"abort_at={abort_ts}, after_bound={len(late_inj)}"))
    out.append(_check(
        "A5_5_never_assembled", not assembled and not released,
        f"assembled_events={len(assembled)}, released_events={len(released)}"))
    return out


def _handback_check(commands_by_tick, members, release_ts, abort_ts):
    """Same-tick native handback receipts, and native forever after."""
    ts = abort_ts if abort_ts is not None else release_ts
    if ts is None:
        return False, "no released/safety_abort event to anchor the handback"
    missing = []
    for member in members:
        rows = [r for r in commands_by_tick.get(ts, []) if r["vehicle_id"] == member]
        if not any(r["owner"] == "native" and r.get("released_to_native")
                   and r["fallback_reason"] == "selected_native" for r in rows):
            missing.append(member)
    late = [r["timestamp_s"] for r in
            (row for tick_rows in commands_by_tick.values() for row in tick_rows)
            if r["vehicle_id"] in members and r["owner"] == "wall"
            and r["timestamp_s"] > ts]
    return (not missing and not late,
            f"same_tick_receipts_missing={missing}, wall_rows_after={late[:5]}")


def _cross_path_check(mixed_by_tick, commands_by_tick, members, extras, release_ts):
    """While the wall is active: members wall+locked, extras damping+free."""
    problems = []
    for tick, rows in sorted(mixed_by_tick.items()):
        if release_ts is not None and tick >= release_ts:
            continue
        wall_here = any(c["owner"] == "wall" for c in commands_by_tick.get(tick, []))
        if not wall_here:
            continue
        for row in rows:
            if row["vehicle_id"] in members:
                if row["action"] != "wall-v2" or not row["locked"]:
                    problems.append((tick, row["vehicle_id"], row["action"], row["locked"]))
            elif row["vehicle_id"] in extras:
                if row["action"] != "damping" or row["locked"]:
                    problems.append((tick, row["vehicle_id"], row["action"], row["locked"]))
    return (not problems, f"cross-path violations={problems[:5]}")


def evaluate_smoke(world: dict, run_dir: Path, receipt: dict | None = None,
                   *, horizon_ticks: int) -> dict:
    """Standing pre-launch smoke checks over the capped smoke run.

    The smoke is never data: it only has to prove that the REAL pipeline
    (scenario/vType load, dispatcher wall assembly start, telemetry rows,
    single ownership write per tick) works through the horizon cap.  Six
    checks: S01 completes exactly the capped horizon; S02/S03 zero
    collisions/teleports inside the window; S04 the single-writer path;
    S05 assembly started at t=0 at the registered wall speed; S06 the wall
    ASSEMBLED within the cap (the A1 critical point is ~t<=30 s << 70 s).
    """
    spec = ARM_SPECS[world["arm"]]
    members = tuple(spec["members"])
    horizon = int(horizon_ticks)
    telemetry, intervals, _manifest = load_rows(run_dir)
    kinds = by_kind(telemetry)
    commands = kinds.get("command", [])
    mixed = kinds.get("mixed_assignment", [])
    events = kinds.get("wall_event", [])
    wall_speed = float(world["wall_speed_mps"])

    complete = bool(receipt.get("exit_status") == "complete") if receipt else True
    ticks = int((receipt or {}).get("summary", {}).get("ticks_executed", -1))
    results: list[dict] = []
    results.append(_check(
        "S01_smoke_horizon_complete",
        complete and len(intervals) == horizon and ticks == horizon
        and (not intervals or abs(intervals[-1]["end_s"] - horizon) <= 1e-6),
        f"receipt={receipt.get('exit_status') if receipt else 'n/a'}, "
        f"interval_rows={len(intervals)}/{horizon}, ticks_executed={ticks}"))
    collisions = sum(int(r["collision_involvements"]) for r in intervals)
    results.append(_check("S02_smoke_zero_collisions", collisions == 0,
                          f"collision_involvements={collisions} (capped window)"))
    teleports = sum(int(r["starting_teleports"]) for r in intervals)
    results.append(_check("S03_smoke_zero_teleports", teleports == 0,
                          f"starting_teleports={teleports} (capped window)"))
    dup, mismatch = _single_writer_scan(commands, mixed, members)
    results.append(_check(
        "S04_smoke_single_writer_path", not dup and not mismatch,
        f"duplicate_writes_at={dup[:5]}, path_mismatches={mismatch[:5]}"))
    starts = [e for e in events if e["event"] == "assembly_start"]
    results.append(_check(
        "S05_smoke_assembly_started",
        len(starts) == 1 and starts
        and 0.0 <= starts[0]["timestamp_s"] <= horizon
        and abs(starts[0]["wall_speed_mps"] - wall_speed) <= WALL_EPS,
        f"assembly_start events={len(starts)}, "
        f"first={starts[0] if starts else None} (A3: the plan is submitted on "
        "the first tick where the full roster is live, so the start tick may "
        "exceed 0 when SUMO insertion is still pending)"))
    assembled = [e for e in events if e["event"] == "assembled"]
    results.append(_check(
        "S06_smoke_assembled_by_cap",
        len(assembled) == 1 and members == tuple(sorted(assembled[0]["members"]))
        and 0.0 < assembled[0]["timestamp_s"] <= horizon,
        f"assembled events={len(assembled)}, "
        f"at={assembled[0]['timestamp_s'] if assembled else None} <= cap {horizon}"))
    return {"schema": SMOKE_CHECKS_SCHEMA, "run": world["run"], "arm": world["arm"],
            "horizon_ticks": horizon, "checks": results,
            "ok": all(c["ok"] for c in results)}
