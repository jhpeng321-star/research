"""E4: validate the emulators against logged chunk-1 TRAIN commands.

For every TRAIN world and every restored branch (native, pi, fixed1/2,
switching1-3): reconstruct each live CAV's ideal SensorFrame at each
command tick from physics_steps.jsonl (same-lane nearest-front/rear
selection, bumper gaps, lane-length/speed from the frozen network), run
the batched torch emulators + a reference-PI state machine on the SAME
reconstructed frames, and compare with the logged suffix_commands rows
and dispatcher controller_state.

Mismatch causes:
  sensing_reconstruction - the reference implementation run on the
      reconstructed frame also disagrees with the log (the frame, not the
      controller port, differs)
  controller_logic       - reference-on-reconstructed matches the log but
      the torch emulator does not (emulator defect)
  fallback_ownership     - command target agrees but owner/fallback
      reason classification differs
  timing                 - missing physics/dispatcher rows, state clock
      anomalies (PI continuity resets)
  unexplained            - none of the above

PI is additionally validated on the LOGGED pi_source_evidence frames
(exact reference inputs), isolating controller logic from sensing.

ZERO SIMULATION: jsonl/plan/net files are only read; nothing runs SUMO.
TRAIN branches only - HOLDOUT worlds are never opened.
"""
import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import emulators  # noqa: E402
import refload  # noqa: E402

refload.load_reference()
from output.controller_verification_v4.pi_external_u import PIExternalU, UReport  # noqa: E402
from traffic_shepherd.config import load_config  # noqa: E402
from traffic_shepherd.damping import CirclesController, direct_target  # noqa: E402

U_REPORT = UReport(31.29, "fixed_registered_setpoint", "cv11-registered-u31.29")
CONFIG = load_config()
VEHICLE_LENGTH_M = 5.0  # SUMO default; chunk-1 vTypes carry no length override
MATCH_TOL_MPS = 0.01
BRANCH_ARMS = ["native", "pi", "fixed1", "fixed2",
               "switching1", "switching2", "switching3"]

CAUSES = ["sensing_reconstruction", "controller_logic",
          "fallback_ownership", "timing", "unexplained"]


# ------------------------------------------------------------------ helpers

def load_lane_table():
    """lane_id -> (length, speed) for every lane of the frozen network."""
    cache = HERE / "results" / "lane_table.json"
    if cache.exists():
        raw = json.loads(cache.read_text())
        return {k: (v[0], v[1]) for k, v in raw.items()}
    with gzip.open(refload.RESEARCH_DENSE / "assets" / "research" /
                   "highway_merge_shepherd_long" / "osm.net.xml.gz", "rb") as fh:
        import xml.etree.ElementTree as ET
        tree = ET.parse(fh)
    table = {}
    for edge in tree.getroot().iter("edge"):
        for lane in edge.findall("lane"):
            table[lane.get("id")] = (float(lane.get("length")),
                                     float(lane.get("speed")))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(table), encoding="utf-8")
    return table


def train_worlds():
    plan = json.load(open(refload.WORLD_PLAN, encoding="utf-8"))
    return {w["world"]: w for w in plan["worlds"] if w["split"] == "TRAIN"}


def base_row_at_t0(world_name, t0):
    """The base run's final physics row (the state the branch restores)."""
    path = refload.CHUNK1_RUNS / f"{world_name}__base" / "run" / "physics_steps.jsonl"
    row = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if abs(float(r["time_s"]) - t0) < 1e-6:
                row = r  # keep the last row at t0
    return row


def iter_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def vehicle_states(row):
    """{vid: (lane_id, position, speed, type_id)} of one physics row."""
    return {vid: (v["lane_id"], float(v["lane_position_m"]),
                  float(v["speed_mps"]), v["type_id"])
            for vid, v in row["vehicles"].items()}


def speed_factors(world):
    """{vType id: speedFactor} of one world's scenario (default 1.0)."""
    import re
    route = (refload.CHUNK1_RUNS / f"{world}__base" / "run" / "scenario"
             / "requests.rou.xml")
    factors = {}
    for match in re.finditer(r'<vType\b[^>]*>', route.read_text(encoding="utf-8")):
        tag = match.group(0)
        vid = re.search(r'id="([^"]+)"', tag)
        sf = re.search(r'speedFactor="([^"]+)"', tag)
        if vid:
            factors[vid.group(1)] = float(sf.group(1)) if sf else 1.0
    return factors


def reconstruct(row_states, cav_ids, lanes, speed_factors, out_anomalies):
    """Ideal-sensing frames for every roster CAV live in this row.

    Mirrors SumoLocalSensor.capture: same-lane nearest front/rear by bumper
    gap, coverage = min(100 m, lane_length - position) front and
    min(100 m, position - length) rear.  getAllowedSpeed resolves to
    lane_speed * speedFactor(vType) on this fleet (verified against the
    logged evidence frames; speedDev=0 makes the factor deterministic).
    """
    frames = {}
    for cav in cav_ids:
        state = row_states.get(cav)
        if state is None:
            continue
        lane_id, position, speed, vtype = state
        lane = lanes.get(lane_id)
        if lane is None:
            out_anomalies["unknown_lane"] += 1
            continue
        lane_length, lane_speed = lane
        factor = speed_factors.get(vtype, 1.0)
        allowed = lane_speed * factor
        front_cov = min(100.0, max(0.0, lane_length - position))
        rear_cov = min(100.0, max(0.0, position - VEHICLE_LENGTH_M))
        front = rear = None
        front_best = rear_best = None
        for other, o_state in row_states.items():
            o_lane, o_pos, o_speed = o_state[0], o_state[1], o_state[2]
            if other == cav or o_lane != lane_id:
                continue
            if o_pos >= position:
                gap = o_pos - VEHICLE_LENGTH_M - position
                if gap <= front_cov and (front_best is None or
                                         (gap, other) < front_best):
                    front_best = (gap, other)
                    front = dict(gap_m=gap, relative_speed_mps=o_speed - speed)
            else:
                gap = position - VEHICLE_LENGTH_M - o_pos
                if gap <= rear_cov and (rear_best is None or
                                        (gap, other) < rear_best):
                    rear_best = (gap, other)
                    rear = dict(gap_m=gap, relative_speed_mps=o_speed - speed)
        frames[cav] = dict(now_s=None, ts_s=None, speed_mps=speed,
                           allowed_speed_mps=allowed,
                           front_coverage_m=front_cov,
                           rear_coverage_m=rear_cov,
                           front=front, rear=rear)
    return frames


class ReferencePI:
    """Reference-faithful PIExternalU lifecycle on reconstructed frames."""

    def __init__(self):
        self.pi = PIExternalU(sample_period_s=1.0)
        self.clock_resets = 0

    def step(self, live_pi, valid_pi, frames, now_s):
        """Advance; return {vid: (issued_target, reason_str_or_None, raw)}."""
        self.pi.invalidate(vehicle_ids=set(valid_pi),
                           missing_ids=set(live_pi) - set(valid_pi))
        out = {}
        for vid in sorted(valid_pi):
            frame = frames[vid]
            ego = frame["speed_mps"]
            leader = ego + frame["front"]["relative_speed_mps"]
            gap = frame["front"]["gap_m"]
            if vid not in self.pi._state:
                self.pi.reset(vid, measured_ego_speed_mps=ego)
            try:
                proposal = self.pi.propose(
                    vid, timestamp_s=now_s, ego_speed_mps=ego,
                    leader_speed_mps=leader, gap_m=gap, u_report=U_REPORT)
            except ValueError:
                self.clock_resets += 1
                self.pi.reset(vid, measured_ego_speed_mps=ego)
                proposal = self.pi.propose(
                    vid, timestamp_s=now_s, ego_speed_mps=ego,
                    leader_speed_mps=leader, gap_m=gap, u_report=U_REPORT)
            raw = proposal.command_mps
            requested = (raw - ego) / 1.0
            frame_obj = type("F", (), {"speed_mps": ego,
                                       "allowed_speed_mps": frame["allowed_speed_mps"]})()
            issued, target = direct_target(frame_obj, requested, CONFIG)
            if issued is None:
                self.pi.discard(proposal)
                self.pi.invalidate(vehicle_ids=[v for v in valid_pi if v != vid],
                                   native_ids={vid})
                out[vid] = (None, "infeasible_speed_bounds", raw)
            else:
                self.pi.commit(proposal, writer_ack=True)
                out[vid] = (target, None, raw)
        return out


def frame_is_valid(frame):
    """PI-path validity (cv7 adapter gate): front present and front
    coverage > 0; rear coverage does NOT gate the PI path."""
    if frame is None:
        return False
    if frame["front"] is None or frame["front_coverage_m"] <= 0:
        return False
    speed = frame["speed_mps"]
    leader = speed + frame["front"]["relative_speed_mps"]
    return (np.isfinite(speed) and speed >= 0 and np.isfinite(leader)
            and leader >= 0 and np.isfinite(frame["front"]["gap_m"])
            and frame["front"]["gap_m"] >= 0)


def frame_problem_reason(frame):
    """Reference frame_problem string for a reconstructed frame.

    The noncurrent_observation gate cannot fire here: a reconstructed
    frame always carries its own row's tick as timestamp."""
    if frame is None:
        return "missing_observation"
    values = (frame["speed_mps"], frame["allowed_speed_mps"],
              frame["front_coverage_m"], frame["rear_coverage_m"])
    if not all(np.isfinite(v) and v >= 0 for v in values):
        return "invalid_observation"
    if max(frame["front_coverage_m"], frame["rear_coverage_m"]) > 100.0:
        return "coverage_exceeds_contract"
    for neighbor, coverage in ((frame["front"], frame["front_coverage_m"]),
                               (frame["rear"], frame["rear_coverage_m"])):
        if neighbor is not None and (
                not np.isfinite(neighbor["gap_m"])
                or not np.isfinite(neighbor["relative_speed_mps"])
                or neighbor["gap_m"] < 0
                or neighbor["gap_m"] > coverage):
            return "invalid_neighbor_measurement"
    return None


# ----------------------------------------------------------------- per branch

def load_branch(run_dir, t0, base_row):
    """One pass over the three jsonl files -> everything E4 needs."""
    commands = {}   # (tick, vid) -> [owner, issued, fallback]
    evidence = defaultdict(dict)  # tick -> vid -> evidence row
    actions = {}    # (tick, vid) -> [action, cs_damping, cs_pi]
    physics = {}    # tick -> row_states
    cav_ids = None
    disp_stale = 0
    for row in iter_jsonl(run_dir / "suffix_commands.jsonl"):
        kind = row.get("kind")
        tick = round(float(row["timestamp_s"]), 6)
        if kind == "command":
            commands[(tick, row["vehicle_id"])] = [
                row["owner"], row["issued_target_speed_mps"],
                row["fallback_reason"]]
        elif kind == "pi_source_evidence":
            evidence[tick][row["vehicle_id"]] = row
    for row in iter_jsonl(run_dir / "dispatcher_rows.jsonl"):
        ts = float(row["timestamp_s"])
        if abs((ts * 2.0) % 2.0 - 1.0) > 1e-9:
            continue  # keep the first-substep stamp only
        cs = row.get("controller_state") or {}
        actions[(round(ts - 0.5, 6), row["vehicle_id"])] = [
            row["action"], cs.get("damping_target_speed_mps"),
            cs.get("pi_running_target_mps")]
    for row in iter_jsonl(run_dir / "physics_steps.jsonl"):
        t = float(row["time_s"])
        if abs(t - round(t)) < 1e-6:
            physics[round(t)] = row
        if cav_ids is None:
            cav_ids = list(row["cav_ids"])
    # restored branch starts at t0 with no physics row of its own: the
    # branch driver writes its first row at t0+0.5, so the restored state
    # is the base run's t0 row (exact saveState restore)
    if t0 is not None and t0 not in physics and base_row is not None:
        physics[round(t0)] = base_row
    return commands, evidence, actions, physics, cav_ids or [], disp_stale


def validate_branch(run_name, world, t0, lanes, speed_factor_map,
                    stats, frame_sanity, options):
    run_dir = refload.CHUNK1_RUNS / run_name / "run"
    receipt_path = refload.CHUNK1_RUNS / run_name / "RUN_RECEIPT.json"
    if not receipt_path.exists():
        stats["protocol"]["incomplete_receipt"][run_name] = "receipt_missing"
        return
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("exit_status") != "complete":
        stats["protocol"]["incomplete_receipt"][run_name] = \
            receipt.get("exit_status")
        return
    base_row = base_row_at_t0(world, t0) if options["use_base_row"] else None
    commands, evidence, actions, physics, cav_ids, _ = load_branch(
        run_dir, t0, base_row)
    if not commands:
        return
    commands_ticks = sorted({t for (t, _v) in commands})
    emulator = emulators.ControllerEmulators(refload.MODEL_ONNX)
    ref_pi = ReferencePI()
    pi_state = None
    # ---- arm 2: PI on the logged evidence frames (exact reference inputs)
    pi_logged = ReferencePI()
    arm2 = stats["arm2_pi_logged"]
    arm2_rows = 0
    arm2_bad = []
    # the adapter is ticked on EVERY control tick (its invalidate drops the
    # state of any CAV outside the current PI subset) - so the arm-2 stream
    # must visit every command tick, not only ticks with evidence rows
    arm2_ticks = sorted(set(commands_ticks) | set(evidence))
    for tick in arm2_ticks:
        evs = evidence.get(tick, {})
        # evidence rows exist exactly for prepared proposals (valid frame +
        # leader); every other CAV is outside this tick's PI subset and its
        # recurrence state is dropped (adapter invalidate semantics)
        pi_logged.pi.invalidate(vehicle_ids=set(evs))
        for vid, row in sorted(evs.items()):
            frame = row["frame"]
            ego = frame["speed_mps"]
            if vid not in pi_logged.pi._state:
                pi_logged.pi.reset(vid, measured_ego_speed_mps=ego)
            front = frame["front"]
            proposal = pi_logged.pi.propose(
                vid, timestamp_s=tick, ego_speed_mps=ego,
                leader_speed_mps=ego + front["relative_speed_mps"],
                gap_m=front["gap_m"], u_report=U_REPORT)
            raw = proposal.command_mps
            frame_obj = type("F", (), {"speed_mps": ego,
                                       "allowed_speed_mps": frame["allowed_speed_mps"]})()
            issued, target = direct_target(frame_obj, (raw - ego) / 1.0, CONFIG)
            if issued is None:
                pi_logged.pi.discard(proposal)
                pi_logged.pi.invalidate(vehicle_ids=[v for v in evs if v != vid],
                                        native_ids={vid})
                continue
            pi_logged.pi.commit(proposal, writer_ack=True)
            arm2_rows += 1
            if not np.isfinite(raw) or abs(raw - row["raw_proposal_mps"]) > 1e-6 \
                    or abs(target - row["issued_target_speed_mps"]) > 1e-6:
                arm2_bad.append(f"{run_name} {vid}@{tick}: raw {raw} vs "
                                f"{row['raw_proposal_mps']}")
    arm2["rows"] += arm2_rows
    arm2["mismatches"] += len(arm2_bad)
    if arm2_bad and len(arm2["examples"]) < 10:
        arm2["examples"].extend(arm2_bad[:10 - len(arm2["examples"])])

    # ---- arm 1: reconstructed frames vs every logged command row
    # per-CAV emulator state (batch composition varies with liveness)
    pi_store = {cav: [0.0, float("nan"), False] for cav in cav_ids}
    by_tick = defaultdict(list)
    for (tick, vid) in commands:
        by_tick[tick].append(vid)
    anomalies = stats["anomalies"]
    for tick in sorted(by_tick):
        row = physics.get(tick)
        if row is None:
            anomalies["missing_physics_row"] += len(by_tick[tick])
            for vid in by_tick[tick]:
                add_mismatch(stats, run_name, "timing", vid, tick)
            continue
        row_states = vehicle_states(row)
        frames = reconstruct(row_states, cav_ids, lanes, speed_factor_map, anomalies)
        live = sorted(frames)
        if not live:
            for vid in by_tick[tick]:
                add_mismatch(stats, run_name, "timing", vid, tick)
                anomalies["no_live_frame"] += 1
            continue
        batch = emulators.FrameBatch.stack([dict(frames[v], now_s=tick, ts_s=tick)
                                            for v in live])
        kinds = []
        for v in live:
            action_row = actions.get((tick, v))
            kinds.append(emulators.ACTION_CODES.get(
                action_row[0] if action_row else "", 0))
            if action_row is None:
                anomalies["missing_dispatcher_row"] += 1
        kinds_t = torch.tensor(kinds)
        pi_state = emulators.ControllerState(pi=emulators.PIState(
            raw_prev=torch.tensor([pi_store[v][0] for v in live],
                                  dtype=torch.float64),
            last_ts=torch.tensor([pi_store[v][1] for v in live],
                                 dtype=torch.float64),
            active=torch.tensor([pi_store[v][2] for v in live])))
        out = emulator.emulate(kinds_t, pi_state, batch)
        for i, v in enumerate(live):
            pi_store[v] = [float(out["new_state"].pi.raw_prev[i]),
                           float(out["new_state"].pi.last_ts[i]),
                           bool(out["new_state"].pi.active[i])]
        # reference legs on the SAME reconstructed frames
        ref_targets = {}
        live_pi = [v for v, k in zip(live, kinds) if k == 2]
        valid_pi = [v for v in live_pi if frame_is_valid(frames[v])]
        # step even when the subset is empty: routed-away CAVs must lose
        # their recurrence state exactly as the adapter's invalidate does
        ref_pi_out = ref_pi.step(live_pi, valid_pi, frames, float(tick))
        for v in live_pi:
            if v in ref_pi_out:
                ref_targets[v] = ref_pi_out[v]
        for slot, v in enumerate(live):
            logged = commands.get((tick, v))
            if logged is None:
                continue
            owner, issued, fallback = logged
            action_row = actions.get((tick, v))
            action = action_row[0] if action_row else "native"
            kind = emulators.ACTION_CODES.get(action, 0)
            emu_target = float(out["target_speed_mps"][slot])
            emu_reason = emulators.REASON_STRINGS[int(out["reason_code"][slot])]
            arm = stats["arm"][run_name.split("__")[1]]
            record_row(arm, kind)
            # frame sanity on logged evidence frames
            if v in evidence.get(tick, {}):
                check_frame_sanity(frame_sanity, frames[v],
                                   evidence[tick][v]["frame"])
            target_ok = (both_nan(issued, emu_target)
                         or (issued is not None and not np.isnan(emu_target)
                             and abs(issued - emu_target) <= MATCH_TOL_MPS))
            reason_ok = reason_matches(owner, fallback, emu_reason)
            if target_ok and reason_ok:
                record_match(stats, arm, kind, issued, emu_target)
                check_controller_state(stats, arm, action_row, out, slot, kind)
                continue
            if not reason_ok and target_ok:
                add_mismatch(stats, run_name, "fallback_ownership", v, tick,
                             detail=f"owner {owner} fallback {fallback} vs {emu_reason}")
                continue
            # reference leg on the SAME reconstructed frame decides whether
            # the reconstructed input or the controller port is at fault
            ref_target, ref_reason = reference_reference(ref_targets, frames, v, kind)
            ref_target_ok = (both_nan(issued, ref_target)
                             or (issued is not None and ref_target is not None
                                 and abs(issued - ref_target) <= MATCH_TOL_MPS))
            ref_reason_ok = reason_matches(owner, fallback, ref_reason)
            emu_ref_same = (
                (both_nan(emu_target, ref_target)
                 or (not np.isnan(emu_target) and ref_target is not None
                     and abs(emu_target - ref_target) <= MATCH_TOL_MPS))
                and ((emu_reason is None) == (ref_reason is None)))
            if emu_ref_same:
                # emulator reproduces the reference on this frame: the
                # frame itself differs from what the run saw
                add_mismatch(stats, run_name, "sensing_reconstruction", v, tick,
                             detail=f"ref {ref_target}/{ref_reason} vs log {issued}/{fallback}")
            elif ref_target_ok and ref_reason_ok:
                add_mismatch(stats, run_name, "controller_logic", v, tick,
                             detail=f"emu {emu_target}/{emu_reason} vs ref {ref_target}/{ref_reason}")
            else:
                # reference also disagrees with the log on this frame
                add_mismatch(stats, run_name, "sensing_reconstruction", v, tick,
                             detail=f"ref {ref_target}/{ref_reason} vs log {issued}/{fallback}")
    stats["arm2_pi_logged"]["clock_resets"] += pi_logged.clock_resets
    stats["protocol"]["branches_validated"].append(run_name)


def reference_reference(ref_targets, frames, vid, kind):
    """The reference implementation's result on the RECONSTRUCTED frame.

    Returns (target, reason) exactly as the frozen stack would have issued
    on this frame (the caller compares it with the log and the emulator).
    """
    frame = frames.get(vid)
    if kind == 0:  # native: owner native, no command
        return None, "selected_native"
    if kind == 1:  # damping
        if frame is None:
            return None, "missing_observation"
        reason = frame_problem_reason(frame)
        if reason is None and (frame["front"] is None
                               or min(frame["front_coverage_m"],
                                      frame["rear_coverage_m"]) <= 0):
            reason = "missing_or_uncovered_leader"
        if reason is not None:
            return None, reason
        frame_obj = to_frame_obj(frame)
        leader = frame["speed_mps"] + frame["front"]["relative_speed_mps"]
        raw = CIRCLES.infer((frame["speed_mps"], leader,
                             frame["front"]["gap_m"]))
        issued, target = direct_target(frame_obj, float(raw), CONFIG)
        return target, ("infeasible_speed_bounds" if issued is None else None)
    if kind == 2:  # pi: the reference state machine already ran
        if vid not in ref_targets:
            # invalidated slot: fallback with its frame_problem reason
            reason = frame_problem_reason(frame) if frame else "missing_observation"
            if reason is None:
                reason = "missing_or_uncovered_leader"
            return None, reason
        target, reason, _raw = ref_targets[vid]
        return target, reason
    return None, f"unsupported action kind {kind}"


def to_frame_obj(frame):
    return type("F", (), {"speed_mps": frame["speed_mps"],
                          "allowed_speed_mps": frame["allowed_speed_mps"]})()


def both_nan(a, b):
    return a is None and (b != b)


def reason_matches(owner, fallback, emu_reason):
    if fallback is None:
        return emu_reason is None and owner == "damping"
    return fallback == emu_reason and owner == "native"


def record_row(arm, kind):
    name = {0: "native", 1: "damping", 2: "pi", 3: "wall-v2"}[kind]
    arm["rows"][name] += 1


def record_match(stats, arm, kind, issued, emu_target):
    name = {0: "native", 1: "damping", 2: "pi", 3: "wall-v2"}[kind]
    arm["matched"][name] += 1
    if issued is not None and not np.isnan(emu_target):
        stats["errors"][name].append(abs(issued - emu_target))


def check_controller_state(stats, arm, action_row, out, slot, kind):
    if action_row is None:
        return
    _, cs_damping, cs_pi = action_row
    if kind == 2 and cs_pi is not None:
        raw = float(out["raw_command_mps"][slot])
        arm["cs_pi_checked"] += 1
        if not (raw != raw) and abs(cs_pi - raw) <= 1e-6:
            arm["cs_pi_match"] += 1


def check_frame_sanity(frame_sanity, reconstructed, logged):
    diffs = frame_sanity
    diffs["n"] += 1
    diffs["speed"].append(abs(reconstructed["speed_mps"] - logged["speed_mps"]))
    diffs["allowed"].append(abs(reconstructed["allowed_speed_mps"]
                                - logged["allowed_speed_mps"]))
    diffs["front_cov"].append(abs(reconstructed["front_coverage_m"]
                                  - logged["front_coverage_m"]))
    diffs["rear_cov"].append(abs(reconstructed["rear_coverage_m"]
                                 - logged["rear_coverage_m"]))
    if reconstructed["front"] is not None and logged["front"] is not None:
        diffs["front_gap"].append(abs(reconstructed["front"]["gap_m"]
                                      - logged["front"]["gap_m"]))
        diffs["front_rel"].append(abs(reconstructed["front"]["relative_speed_mps"]
                                      - logged["front"]["relative_speed_mps"]))
    elif (reconstructed["front"] is None) != (logged["front"] is None):
        diffs["front_presence_flip"] += 1


def add_mismatch(stats, run_name, cause, vid, tick, detail=""):
    stats["mismatches"][cause] += 1
    bucket = stats["mismatch_examples"][cause]
    if len(bucket) < 15:
        bucket.append(f"{run_name} {vid}@{tick}: {detail}")


# ---------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=0,
                        help="validate only the first N TRAIN worlds")
    parser.add_argument("--start", type=int, default=0,
                        help="first TRAIN world index (chunked runs)")
    parser.add_argument("--no-base-row", action="store_true")
    options = {"use_base_row": True}
    args = parser.parse_args()
    if args.no_base_row:
        options["use_base_row"] = False

    lanes = load_lane_table()
    worlds = train_worlds()
    names = sorted(worlds)
    if args.worlds:
        names = names[:args.worlds]
    names = names[args.start:]
    global CIRCLES
    CIRCLES = CirclesController()
    stats = {
        "arm": {arm: {name: 0 for name in ("rows", "matched")} for arm in BRANCH_ARMS},
        "errors": {"native": [], "damping": [], "pi": []},
        "mismatches": {c: 0 for c in CAUSES},
        "mismatch_examples": {c: [] for c in CAUSES},
        "anomalies": defaultdict(int),
        "arm2_pi_logged": {"rows": 0, "mismatches": 0, "examples": [],
                           "clock_resets": 0},
        "protocol": {"branches_validated": [], "incomplete_receipt": {}},
        "frame_sanity": {"n": 0, "speed": [], "allowed": [], "front_cov": [],
                         "rear_cov": [], "front_gap": [], "front_rel": [],
                         "front_presence_flip": 0},
    }
    # per-arm substructure
    for arm in BRANCH_ARMS:
        stats["arm"][arm] = {
            "rows": {"native": 0, "damping": 0, "pi": 0},
            "matched": {"native": 0, "damping": 0, "pi": 0},
            "cs_pi_checked": 0, "cs_pi_match": 0}

    t_start = time.time()
    for i, world in enumerate(names):
        t0 = float(worlds[world]["t0_s"])
        factors = speed_factors(world)
        for arm in BRANCH_ARMS:
            validate_branch(f"{world}__{arm}", world, t0, lanes,
                            factors, stats, stats["frame_sanity"], options)
        if (i + 1) % 5 == 0:
            print(f"[{args.start + i + 1}/{len(sorted(worlds))} worlds] "
                  f"{time.time() - t_start:.0f}s", flush=True)

    # ---- aggregate
    totals = {name: {"rows": 0, "matched": 0}
              for name in ("native", "damping", "pi")}
    per_arm_lines = []
    for arm in BRANCH_ARMS:
        a = stats["arm"][arm]
        cells = []
        for name in ("native", "damping", "pi"):
            rows, matched = a["rows"][name], a["matched"][name]
            totals[name]["rows"] += rows
            totals[name]["matched"] += matched
            rate = (matched / rows) if rows else float("nan")
            cells.append(f"{name}: {matched}/{rows} ({rate:.6f})")
        per_arm_lines.append(f"{arm:>11s}  " + "  ".join(cells))
    overall_lines = []
    for name in ("native", "damping", "pi"):
        rows = totals[name]["rows"]
        matched = totals[name]["matched"]
        errs = np.array(stats["errors"][name]) if stats["errors"][name] else np.array([0.0])
        line = (f"{name:>8s}: {matched}/{rows} matched "
                f"({(matched / rows) if rows else float('nan'):.6f})  "
                f"|diff| m/s: max {errs.max():.6g} "
                f"p50 {np.quantile(errs, 0.5):.3g} "
                f"p99 {np.quantile(errs, 0.99):.3g}")
        overall_lines.append(line)
    print("\n=== E4 per-controller match vs logged commands (TRAIN) ===")
    print("\n".join(overall_lines))
    print("\n--- per arm (matched/rows (rate)) ---")
    print("\n".join(per_arm_lines))
    print("\n--- mismatch causes ---")
    for cause in CAUSES:
        print(f"{cause:>24s}: {stats['mismatches'][cause]}")
    print("\n--- anomalies ---")
    for key, value in sorted(stats["anomalies"].items()):
        print(f"{key:>24s}: {value}")
    print("\n--- PI on logged evidence frames (arm 2) ---")
    a2 = stats["arm2_pi_logged"]
    print(f"rows {a2['rows']}  mismatches {a2['mismatches']}  "
          f"clock_resets {a2['clock_resets']}")
    print("\n--- frame sanity (reconstructed vs logged evidence) ---")
    fs = stats["frame_sanity"]
    for field in ("speed", "allowed", "front_cov", "rear_cov", "front_gap",
                  "front_rel"):
        arr = np.array(fs[field]) if fs[field] else np.array([0.0])
        print(f"{field:>10s}: max {arr.max():.6g}  mean {arr.mean():.3g}")
    print(f"front presence flips: {fs['front_presence_flip']} of {fs['n']}")

    result = {
        "experiment": "WD-01 partA_ctrl E4",
        "data_root": str(refload.CHUNK1_RUNS),
        "split": "TRAIN only (49 worlds); HOLDOUT never opened",
        "n_worlds": len(names),
        "branches_per_world": BRANCH_ARMS,
        "match_tol_mps": MATCH_TOL_MPS,
        "totals": {name: {"rows": totals[name]["rows"],
                          "matched": totals[name]["matched"],
                          "rate": (totals[name]["matched"] / totals[name]["rows"])
                          if totals[name]["rows"] else None,
                          "error_max_mps": float(max(stats["errors"][name]) if stats["errors"][name] else 0.0)}
                   for name in ("native", "damping", "pi")},
        "per_arm": {arm: {"rows": stats["arm"][arm]["rows"],
                          "matched": stats["arm"][arm]["matched"],
                          "cs_pi_checked": stats["arm"][arm]["cs_pi_checked"],
                          "cs_pi_match": stats["arm"][arm]["cs_pi_match"]}
                    for arm in BRANCH_ARMS},
        "mismatch_causes": dict(stats["mismatches"]),
        "mismatch_examples": stats["mismatch_examples"],
        "anomalies": dict(stats["anomalies"]),
        "arm2_pi_logged": {k: v for k, v in stats["arm2_pi_logged"].items()},
        "frame_sanity": {k: (v if not isinstance(v, list) else
                             {"max": float(max(v)), "mean": float(np.mean(v))})
                         for k, v in stats["frame_sanity"].items()},
        "protocol": stats["protocol"],
        "speed_mode_note": "all chunk-1 CAV speedMode 31 (pre-fix); "
                           "irrelevant to commanded-speed emulation",
        "elapsed_s": round(time.time() - t_start, 1),
    }
    out = HERE / "results" / "e4_chunk1_train_validation.json"
    out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
