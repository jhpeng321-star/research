"""E2: batched PI external-U emulator == the reference recurrence.

Two legs:
1. recurrence-level equivalence vs cv4 PIExternalU on synthetic sequences
   (random resets/invalidations, random traffic) to 1e-6;
2. full frozen-stack equivalence (MixedAssignmentSession + PISession +
   CommandOwnership + SumoLocalSensor on the fake api) vs the batched
   emulators through the emulate() API: every issued target, every
   fallback reason, and the controller_state internals across a switching
   schedule.  ZERO SIMULATION: the fake api is scripted dicts, no traci.
"""
import math

import numpy as np
import torch

import emulators
import refload

refload.load_reference()

from output.controller_verification_v4.pi_external_u import PIExternalU, UReport  # noqa: E402

from fakeapi import FakeApi  # noqa: E402
from reference_stack import ReferenceStack  # noqa: E402

U_REPORT = UReport(31.29, "fixed_registered_setpoint", "cv11-registered-u31.29")


def test_recurrence_synthetic_sequences():
    """PIExternalU (reference class) vs PIEmulator.raw_command."""
    rng = np.random.default_rng(5)
    n_cav, n_ticks = 6, 400
    pi = PIExternalU(sample_period_s=1.0)
    emu = emulators.PIEmulator()
    state = emulators.PIState.zeros(n_cav)
    ego0 = rng.uniform(5, 30, n_cav)
    for slot in range(n_cav):
        pi.reset(f"v{slot}", measured_ego_speed_mps=float(ego0[slot]))
    state.raw_prev[:] = torch.tensor(ego0)
    state.last_ts.fill_(float("nan"))
    state.active[:] = True

    for tick in range(1, n_ticks + 1):
        now = float(tick)
        ego = np.clip(rng.uniform(0, 35, n_cav), 0, None)
        leader = np.clip(ego + rng.uniform(-4, 4, n_cav), 0, None)
        gap = rng.uniform(0.5, 120, n_cav)
        invalidate = rng.random(n_cav) < 0.07  # random lifecycle drops
        # reference lifecycle: invalidate(live_ids=LIVE...) keeps the live
        # mask and drops everything else (contract of PIExternalU.invalidate)
        pi.invalidate(vehicle_ids=[f"v{s}" for s in range(n_cav)
                                   if not invalidate[s]])
        engaged = torch.tensor(~invalidate)
        raw_new, state = emu.raw_command(
            state, engaged,
            torch.full((n_cav,), now, dtype=torch.float64),
            torch.tensor(ego, dtype=torch.float64),
            torch.tensor(leader, dtype=torch.float64),
            torch.tensor(gap, dtype=torch.float64))
        for slot in range(n_cav):
            vid = f"v{slot}"
            if invalidate[slot]:
                continue  # stays invalidated; re-engages next tick
            if vid not in pi._state:
                pi.reset(vid, measured_ego_speed_mps=float(ego[slot]))
            proposal = pi.propose(vid, timestamp_s=now,
                                  ego_speed_mps=float(ego[slot]),
                                  leader_speed_mps=float(leader[slot]),
                                  gap_m=float(gap[slot]), u_report=U_REPORT)
            ref_raw = pi.commit(proposal, writer_ack=True)
            if math.isfinite(float(raw_new[slot])):
                assert abs(ref_raw - float(raw_new[slot])) <= 1e-9, \
                    f"tick {tick} slot {slot}"
        # committed reference state == emulator running target (slots still
        # holding state; an invalidation on the last tick removes it)
        for slot in range(n_cav):
            if invalidate[slot]:
                continue
            ref_state = pi._state[f"v{slot}"]
            assert abs(ref_state[0] - float(state.raw_prev[slot])) <= 1e-6


def _scenario_world():
    api = FakeApi()
    api.set_lane("main_0", length=2644.22, speed=31.29)
    api.set_lane("main_1", length=2645.67, speed=31.29)
    layout = {"c1": ("main_0", 200.0), "c2": ("main_1", 260.0),
              "c3": ("main_0", 480.0), "h1": ("main_0", 240.0),
              "h2": ("main_1", 300.0)}
    return api, layout


def _scripted_speeds(layout, n_ticks):
    phases = {"c1": 0.0, "c2": 1.3, "c3": 2.1, "h1": 3.9, "h2": 5.2}
    speeds = {}
    for vid, (_, _) in layout.items():
        seq = 24.0 + 3.5 * np.sin(np.arange(n_ticks) / 6.0 + phases[vid])
        speeds[vid] = [max(1.0, float(v)) for v in seq]
    return speeds


def _advance(api, layout, k, speeds):
    """Scripted kinematics at substep k with per-lane gap safety."""
    per_lane: dict[str, list[str]] = {}
    for vid, (lane_id, _) in layout.items():
        per_lane.setdefault(lane_id, []).append(vid)
    positions = {}
    for lane_id, vids in per_lane.items():
        lane_len = api.lanes[lane_id]["length"]
        vids = sorted(vids, key=lambda v: layout[v][1])  # start-s order
        for vid in vids:
            start = layout[vid][1]
            positions[vid] = start + sum(speeds[vid][:k])
        for rear, front in zip(vids, vids[1:]):  # adjacency in s order
            if positions[front] - positions[rear] < 25.0:
                positions[rear] = positions[front] - 25.0
        for vid in vids:
            positions[vid] = min(positions[vid], lane_len - 30.0)
    for vid, (lane_id, _) in layout.items():
        api.set_vehicle(vid, lane_id=lane_id, position=positions[vid],
                        speed=speeds[vid][k], accel=0.0,
                        lane_index=int(lane_id[-1]))


def _frame_dict(frame):
    """SensorFrame -> emulator frame dict (same captured content)."""
    return dict(now_s=frame.timestamp_s, ts_s=frame.timestamp_s,
                speed_mps=frame.speed_mps,
                allowed_speed_mps=frame.allowed_speed_mps,
                front_coverage_m=frame.front_coverage_m,
                rear_coverage_m=frame.rear_coverage_m,
                front=None if frame.front is None else dict(
                    gap_m=frame.front.gap_m,
                    relative_speed_mps=frame.front.relative_speed_mps),
                rear=None if frame.rear is None else dict(
                    gap_m=frame.rear.gap_m,
                    relative_speed_mps=frame.rear.relative_speed_mps))


def _compare(record, emu_target, emu_reason, where, problems):
    ref_reason = record.fallback_reason
    ref_target = record.issued_target_speed_mps
    if ref_reason != emu_reason:
        problems.append(f"{where}: reason {ref_reason!r} != {emu_reason!r}")
        return False
    if ref_target is None:
        if not math.isnan(emu_target):
            problems.append(f"{where}: target {emu_target} vs None")
            return False
        return True
    if math.isnan(emu_target):
        problems.append(f"{where}: target None vs {ref_target}")
        return False
    tol = 1e-5  # damping path carries the float32 policy output
    if abs(ref_target - emu_target) > tol:
        problems.append(f"{where}: target {ref_target} vs {emu_target}")
        return False
    return True


def test_pi_infeasible_bridge_resets_state():
    """Over-limit ego -> infeasible_speed_bounds + recurrence reset."""
    emu = emulators.PIEmulator()
    state = emulators.PIState.zeros(2)
    rows = [dict(now_s=10.0, ts_s=10.0, speed_mps=35.0, allowed_speed_mps=31.29,
                 front_coverage_m=100.0, rear_coverage_m=20.0,
                 front_present=True, front_gap_m=40.0, front_rel_mps=-2.0,
                 rear_present=False, rear_gap_m=0.0, rear_rel_mps=0.0,
                 frame_present=True),
            dict(now_s=10.0, ts_s=10.0, speed_mps=25.0, allowed_speed_mps=31.29,
                 front_coverage_m=100.0, rear_coverage_m=20.0,
                 front_present=True, front_gap_m=40.0, front_rel_mps=-2.0,
                 rear_present=False, rear_gap_m=0.0, rear_rel_mps=0.0,
                 frame_present=True)]
    batch = emulators.FrameBatch.stack(rows)
    engaged = torch.ones(2, dtype=torch.bool)
    target, codes, raw, new_state, reset = emu.decide(state, batch, engaged)
    assert emulators.REASON_STRINGS[int(codes[0])] == "infeasible_speed_bounds"
    assert math.isnan(float(target[0]))
    assert not bool(new_state.active[0])
    assert emulators.REASON_STRINGS[int(codes[1])] is None
    assert not math.isnan(float(target[1]))
    assert bool(new_state.active[1])
    assert bool(reset[1])  # first engagement is a fresh measured-speed reset
    assert abs(float(new_state.raw_prev[1]) - float(raw[1])) <= 1e-12


def test_fullstack_mixed_dispatcher_equivalence():
    api, layout = _scenario_world()
    roster = ["c1", "c2", "c3"]
    speeds = _scripted_speeds(layout, 40)
    stack = ReferenceStack(api, roster)
    emulator = emulators.ControllerEmulators(refload.MODEL_ONNX)
    state = emulator.init_state(len(roster))

    assignments = {vid: "damping" for vid in roster}
    problems, n_pi_rows = [], 0
    for t in range(100, 140):
        if t == 110:
            assignments["c2"] = "pi-external-u"
        if t == 118:
            assignments["c2"] = "native"     # PI exit (state invalidated)
        if t == 124:
            assignments["c2"] = "pi-external-u"  # re-entry: fresh reset
        if t == 130:
            assignments["c1"] = "pi-external-u"
            assignments["c3"] = "native"
        _advance(api, layout, t - 100, speeds)
        stack.apply_plan(assignments, f"test:p{t}", now_s=float(t))
        frames, records, telemetry = stack.step(now_s=float(t))
        telem = {row["vehicle_id"]: row for row in telemetry}
        live = sorted(frames)  # captured frames: the exact reference inputs
        batch = emulators.FrameBatch.stack([_frame_dict(frames[v])
                                            for v in live])
        kinds = torch.tensor([emulators.ACTION_CODES[assignments[v]]
                              for v in live])
        out = emulator.emulate(kinds, state, batch)
        state = out["new_state"]
        rec_by_id = {r.vehicle_id: r for r in records}
        for slot, vid in enumerate(live):
            record = rec_by_id[vid]
            ok = _compare(record, float(out["target_speed_mps"][slot]),
                          emulators.REASON_STRINGS[int(out["reason_code"][slot])],
                          f"{vid}@{t}", problems)
            if not ok:
                continue
            cs = telem[vid]["controller_state"]
            if assignments[vid] == "pi-external-u":
                if record.fallback_reason is None:
                    n_pi_rows += 1
                    ref_pi = cs["pi_running_target_mps"]
                    emu_pi = float(out["raw_command_mps"][slot])
                    assert ref_pi is not None and \
                        abs(ref_pi - emu_pi) <= 1e-9, \
                        f"pi running target {vid}@{t}: {ref_pi} vs {emu_pi}"
            elif record.owner == "damping":
                ref_dt = cs["damping_target_speed_mps"]
                assert ref_dt is not None and abs(
                    ref_dt - record.issued_target_speed_mps) <= 1e-12
    assert not problems, "\n".join(problems[:20])
    assert n_pi_rows >= 20, f"PI legs underexercised: {n_pi_rows}"
