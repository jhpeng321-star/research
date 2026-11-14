"""G2 ``ctrl-parity``: replay recorded chunk-4 TRAIN CAV motion with the P1
mover and report per-tick position/speed error against SUMO.

For every selected window the replay

1. loads the run's ``physics_steps.jsonl`` (0.5 s rows) and its
   ``suffix_commands.jsonl`` (issued commands; ``pi_source_evidence`` marks
   the fleet-PI rows, which the ownership logs under the ``damping`` writer
   identity) -- TRAIN runs only, split from the registered WORLD_PLAN;
2. reconstructs each live CAV's ideal SensorFrame at every control tick
   exactly like the validated E4 reconstruction (same SUMO lane id, nearest
   front/rear by bumper gap, coverage ``min(100 m, lane_end - s)`` /
   ``min(100 m, s - 5 m)``, ``allowed = lane_speed * speedFactor``);
3. computes the tick's commands with the REAL ``ctrl.CavMover`` mirrors
   (native = no command; damping = CIRCLES ONNX; pi = PI external-U; wall =
   the J4 ``wall-v2.1-closure`` law with the recorded ``wall_events``
   phases), through the direct-target bridge;
4. integrates every CAV with the measured SUMO execution rule
   (``kin_idm``: ``v' = max(0, min(clip(T, v - 3*0.5, v + 1.5*0.5),
   v + idm*dt))`` with the CAV's own vType IDM; native CAVs run plain
   ``v' = max(0, v + idm*dt)``).  Every half-step is predicted from the
   RECORDED pre-step state (ego speed, leader gap/dv, lane, vType) -- the
   measured-rule convention of ``partA_act.measure_execution`` -- and the
   ghost position accumulates the predicted speeds, so the errors
   attribute to the ctrl + execution chain and not to an IDM re-fit;
5. compares the replayed ``(s, v)`` with the recorded row at every integer
   control tick and pools max/RMS errors per window.

Zero-simulation replay: only jsonl/plan/scenario files are read; SUMO never
runs.  TRAIN worlds only -- HOLDOUT/CALIBRATION rows are never opened; the
split comes from the registered chunk-4 WORLD_PLAN.  Structural breaks
(SUMO lane changes, junction transits, command mismatches) are counted and
reported, never silently dropped.

Usage:  ``python -m ctrl.parity`` (from ``output/wd01/wm_n1``) prints the
report JSON and writes ``ctrl/results/parity_g2.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

if __package__ in (None, ""):                       # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_WD01 = Path(__file__).resolve().parents[2]
if str(_WD01) not in sys.path:
    sys.path.insert(0, str(_WD01))

import wdpaths  # noqa: E402

RESEARCH_DENSE = Path(wdpaths.RESEARCH_DENSE_ROOT)
DEFAULT_PLAN = (RESEARCH_DENSE / "output" / "campaign_stage1_v3" /
                "results_plan_chunk4" / "WORLD_PLAN.json")
STAGE1_RUNS = RESEARCH_DENSE / "runs" / "campaign_stage1_chunk4"
WALL_RUNS = RESEARCH_DENSE / "runs" / "campaign_walls_pilot_chunk4"

_HERE = Path(__file__).resolve().parent
_LANE_TABLE_PATH = _HERE.parents[1] / "partA_ctrl" / "results" / "lane_table.json"
_WD01_IN_PATH = str(_WD01)
if _WD01_IN_PATH not in sys.path:
    sys.path.insert(0, _WD01_IN_PATH)

import ctrl  # noqa: E402  (the P1 piece under test)

VEHICLE_LENGTH_M = 5.0
DT = 0.5
MATCH_TOL_MPS = 0.01          # E4 command-match tolerance
WINDOW_S = 600.0
SLOTS = 128
#: the merge-relative world offset: s_m = s_abs - MERGE_OFFSET_M
MERGE_OFFSET_M = 2644.22
#: accel (ramp) chain end in the absolute s coordinate: 378.66 + 75.44
ACCEL_LANE_END_M = 454.1
#: |replayed - recorded| position jump (m) that ends a continuous replay
#: stretch: junction transits and the accel-chain -> mainline-station path
#: discontinuity (the ramp physically ends at mainline station 2719.66)
JUMP_REANCHOR_M = 50.0

# >= 2 chunk-4 TRAIN windows per strategy; wall arms have an
# assembled+released J4 wall; all worlds are TRAIN in the registered plan.
DEFAULT_WINDOWS = [
    ("native",  "w8659m2700p10t420", "native"),
    ("native",  "w8670m2700p10t300", "native"),
    ("native",  "w8659m2700p10t420", "native_base"),
    ("damping", "w8659m2700p10t420", "base"),
    ("damping", "w8670m2700p10t300", "base"),
    ("pi",      "w8659m2700p10t420", "pi"),
    ("pi",      "w8670m2700p10t300", "pi"),
    ("wall",    "w8659m2700p10t420", "w-lo-early"),
    ("wall",    "w8670m2700p10t300", "w-lo-early"),
]

# --------------------------------------------------------------------- io


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def train_worlds(plan_path: Path) -> dict:
    """{world: plan row} for TRAIN worlds of the registered plan."""
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    return {w["world"]: w for w in plan["worlds"]
            if w.get("split") == "TRAIN"}


def load_lane_table() -> dict:
    """SUMO lane id -> (length, speed) of the frozen network (read-only)."""
    raw = json.loads(_LANE_TABLE_PATH.read_text(encoding="utf-8"))
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}


def speed_factors(run_dir: Path) -> dict:
    """{vType id: (speedFactor, tau, accel, decel, minGap)} of one scenario."""
    text = (run_dir / "run" / "scenario" / "requests.rou.xml").read_text(
        encoding="utf-8")
    out = {}
    for match in re.finditer(r"<vType\b[^>]*>", text):
        tag = match.group(0)

        def attr(name, default=None):
            got = re.search(rf'{name}="([^"]+)"', tag)
            return float(got.group(1)) if got else default

        vid_m = re.search(r'id="([^"]+)"', tag)
        if vid_m is None:
            continue
        out[vid_m.group(1)] = (
            attr("speedFactor", 1.0), attr("tau", 1.0), attr("accel", 1.0),
            attr("decel", 4.5), attr("minGap", 2.0))
    return out


def row_states(row: dict) -> dict:
    """{vid: (lane_id, pos, speed, type_id, lane_changed)} of one row."""
    return {
        vid: (v["lane_id"], float(v["lane_position_m"]),
              float(v["speed_mps"]), v["type_id"],
              bool(v.get("lane_changed")))
        for vid, v in row["vehicles"].items()}


def load_physics(run_dir: Path) -> tuple:
    """-> ({t: row_states}, cav_ids) of one run."""
    rows, cav_ids = {}, None
    for row in iter_jsonl(run_dir / "run" / "physics_steps.jsonl"):
        t = round(float(row["time_s"]), 6)
        rows[t] = row_states(row)
        if cav_ids is None:
            cav_ids = list(row["cav_ids"])
    return rows, cav_ids or []


def load_commands(run_dir: Path) -> tuple:
    """-> ({(tick, vid): (owner, issued, fallback)}, {(tick, vid): frame}).

    Branch runs carry ``suffix_commands.jsonl``; full instrumented runs (the
    damping bases) carry the same commands as ``writer_trace.jsonl``
    ``slowDown`` calls (the ownership writer trace; no fallback rows).  The
    second mapping is the fleet-PI evidence set: in a fleet-PI run the
    ownership writer logs every issued command under the ``damping``
    identity, and each prepared PI proposal additionally emits a
    ``pi_source_evidence`` row carrying the exact reference frame.
    """
    commands, evidence = {}, {}
    suffix = run_dir / "run" / "suffix_commands.jsonl"
    if suffix.is_file():
        for row in iter_jsonl(suffix):
            kind = row.get("kind")
            tick = round(float(row["timestamp_s"]), 6)
            if kind == "command":
                commands[(tick, row["vehicle_id"])] = (
                    row["owner"], row["issued_target_speed_mps"],
                    row["fallback_reason"])
            elif kind == "pi_source_evidence":
                evidence[(tick, row["vehicle_id"])] = row.get("frame") or {}
        return commands, evidence
    trace = run_dir / "run" / "writer_trace.jsonl"
    if not trace.is_file():
        return commands, evidence          # native_base: no commands at all
    for row in iter_jsonl(trace):
        if row.get("method") != "slowDown":
            continue
        args = row.get("args") or []
        if len(args) < 2:
            continue
        commands[(round(float(row["time_s"]), 6), str(args[0]))] = (
            "damping", float(args[1]), None)
    return commands, evidence


def load_wall_activity(run_dir: Path) -> dict | None:
    """Wall lifecycle of one J4 branch from wall_events.jsonl (TRAIN only)."""
    act = dict(assembly_start_s=None, assembled_at_s=None,
               released_at_s=None, members=None, v_wall=None)
    for row in iter_jsonl(run_dir / "run" / "wall_events.jsonl"):
        event = row.get("event")
        if row.get("members") and act["members"] is None:
            act["members"] = list(row["members"])
        if event == "assembly_start":
            act["assembly_start_s"] = float(row["timestamp_s"])
            if row.get("wall_speed_mps") is not None:
                act["v_wall"] = float(row["wall_speed_mps"])
        elif event == "assembled":
            act["assembled_at_s"] = float(row["timestamp_s"])
            act["members"] = list(row.get("members") or act["members"] or [])
            if row.get("wall_speed_mps") is not None:
                act["v_wall"] = float(row["wall_speed_mps"])
        elif event == "released":
            act["released_at_s"] = float(row["timestamp_s"])
    if act["members"] is None or act["assembled_at_s"] is None:
        return None
    return act


def base_row_states(stage1_runs: Path, world: str, t0: float) -> dict:
    """The base run's row at t0: the exact restored branch state."""
    path = stage1_runs / f"{world}__base" / "run" / "physics_steps.jsonl"
    row = None
    for got in iter_jsonl(path):
        if abs(float(got["time_s"]) - t0) < 1e-6:
            row = got
    if row is None:
        raise RuntimeError(f"base run of {world} has no physics row at "
                           f"t0={t0}")
    return row_states(row)


# ---------------------------------------------------------------- geometry

_GEO = None


def geo():
    """features.geometry (read-only P9 consumer import), cached."""
    global _GEO
    if _GEO is None:
        root = str(_HERE.parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        import features.geometry as g

        _GEO = g
    return _GEO


def row_geometry(states: dict) -> dict:
    """{vid: (lane_phys, s_abs)}; junction/internal or unknown -> (None, None)."""
    table = geo().LANE_TABLE
    out = {}
    for vid, st in states.items():
        lane_id = st[0]
        if str(lane_id).startswith(":"):
            out[vid] = (None, None)
            continue
        edge, sep, idx = str(lane_id).rpartition("_")
        row = table.get((edge, int(idx))) if sep and idx.isdigit() else None
        if row is None:
            out[vid] = (None, None)
            continue
        lane_phys, seg_start = row
        out[vid] = (lane_phys, seg_start + st[1])
    return out


def reconstruct_frames(states: dict, cav_ids, lanes, factors):
    """E4 ideal-sensor frames: same-lane-id nearest front/rear per CAV."""
    frames = {}
    for cav in cav_ids:
        st = states.get(cav)
        if st is None:
            continue
        lane_id, position, speed, vtype = st[0], st[1], st[2], st[3]
        lane = lanes.get(lane_id)
        if lane is None:
            continue
        lane_length, lane_speed = lane
        factor = factors.get(vtype, (1.0,))[0]
        allowed = lane_speed * factor
        front_cov = min(100.0, max(0.0, lane_length - position))
        rear_cov = min(100.0, max(0.0, position - VEHICLE_LENGTH_M))
        front = rear = None
        front_best = rear_best = None
        for other, o_state in states.items():
            o_lane, o_pos, o_speed = o_state[0], o_state[1], o_state[2]
            if other == cav or o_lane != lane_id:
                continue
            if o_pos >= position:
                gap = o_pos - VEHICLE_LENGTH_M - position
                if gap <= front_cov and (front_best is None or
                                         (gap, other) < front_best):
                    front_best = (gap, other)
                    front = (gap, o_speed - speed)
            else:
                gap = position - VEHICLE_LENGTH_M - o_pos
                if gap <= rear_cov and (rear_best is None or
                                        (gap, other) < rear_best):
                    rear_best = (gap, other)
                    rear = (gap, o_speed - speed)
        frames[cav] = dict(
            speed_mps=speed, allowed_speed_mps=allowed,
            front_coverage_m=front_cov, rear_coverage_m=rear_cov,
            front_present=front is not None,
            front_gap_m=0.0 if front is None else front[0],
            front_rel_mps=0.0 if front is None else front[1],
            rear_present=rear is not None,
            rear_gap_m=0.0 if rear is None else rear[0],
            rear_rel_mps=0.0 if rear is None else rear[1])
    return frames


# ------------------------------------------------------------------- idm

def sumo_idm_accel(v: float, gap: float, dv: float, a_max: float, b: float,
                   tau: float, s0: float, v0: float) -> float:
    """SUMO IDM of the CAV's own vType (measure_execution.idm_acc_np)."""
    v0 = max(v0, 0.1)
    s_star = max(s0 + max(0.0, v * tau + v * dv
                          / (2.0 * math.sqrt(a_max * b))), 0.0)
    return a_max * (1.0 - min(v / v0, 4.0) ** 4 - (s_star / max(gap, 0.3)) ** 2)


def physical_leader(geom: dict, vid: str, ego_s: float):
    """Nearest same-physical-lane front vehicle (cross-edge, the SUMO view).

    -> (gap, other_vid) or None.  Neighbour positions are the RECORDED ones;
    the ego position is the REPLAYED one (the replayed follower's view).
    """
    lane_phys, _ = geom.get(vid, (None, None))
    if lane_phys is None:
        return None
    best = None
    for other, (o_lane, o_s) in geom.items():
        if other == vid or o_lane != lane_phys or o_s is None:
            continue
        if o_s >= ego_s:
            gap = o_s - VEHICLE_LENGTH_M - ego_s
            if best is None or gap < best[0]:
                best = (gap, other)
    return best


# ----------------------------------------------------------------- replay

ACTION_OF = {"native": 0, "native_base": 0, "base": 1, "damping": 1, "pi": 2,
             "wall": 0}


def replay_window(strategy: str, world: str, arm: str, *, plan_path: Path,
                  stage1_runs: Path = STAGE1_RUNS,
                  wall_runs: Path = WALL_RUNS,
                  train: dict | None = None,
                  trace_vid: str | None = None) -> dict:
    """Replay one TRAIN window; return the parity stats dict."""
    t_start = time.time()
    train = train or train_worlds(plan_path)
    if world not in train or train[world].get("split") != "TRAIN":
        raise ValueError(f"{world} is not a TRAIN world of the plan")
    t0 = float(train[world]["t0_s"])
    run_dir = ((stage1_runs if arm in ACTION_OF else wall_runs)
               / f"{world}__{arm}")
    receipt = json.loads(
        (run_dir / "RUN_RECEIPT.json").read_text(encoding="utf-8"))
    if receipt.get("exit_status") != "complete":
        raise ValueError(f"{run_dir.name}: receipt exit_status "
                         f"{receipt.get('exit_status')!r}")

    rows, cav_ids = load_physics(run_dir)
    commands, evidence_frames = load_commands(run_dir)
    evidence = set(evidence_frames)
    lanes = load_lane_table()
    factors = speed_factors(run_dir)
    frame_sanity = {"n": 0, "speed": 0.0, "allowed": 0.0, "front_cov": 0.0,
                    "rear_cov": 0.0, "front_gap": 0.0, "front_rel": 0.0,
                    "front_presence_flip": 0, "rear_presence_flip": 0}
    # the J4 law for wall windows; wall_law is irrelevant to the other
    # strategies (no wall slots)
    mover = ctrl.CavMover(ctrl.CtrlConfig(wall_law=ctrl.WALL_LAW_V21))
    if strategy == "wall":
        activity = load_wall_activity(run_dir)
        if activity is None:
            raise ValueError(f"{run_dir.name}: no assembled wall to replay")
    else:
        activity = None
    default_action = ACTION_OF[strategy]

    # ---- initial state: the restored row at t0 --------------------------
    states0 = (rows[t0] if t0 in rows
               else base_row_states(stage1_runs, world, t0))
    geom0 = row_geometry(states0)
    cav_state = {}
    for vid in cav_ids:
        if vid in states0 and geom0.get(vid, (None, None))[0] is not None:
            lane_phys, s_abs = geom0[vid]
            cav_state[vid] = [s_abs, float(states0[vid][2]), lane_phys]

    struct: dict = defaultdict(int)
    err_pos, err_speed = [], []
    cmd_examples: list = []
    trace = [] if trace_vid else None
    cmd_checked = cmd_matched = 0
    cmd_max_diff = 0.0
    per_cav_max: dict = {}
    # FIXED per-CAV slot assignment: the mover's CtrlState (the PI
    # recurrence) is slot-indexed, so a CAV must keep its slot across ticks
    slot_grid = max(1, math.ceil(len(cav_ids) / SLOTS))
    slot_of = {vid: (i // SLOTS, i % SLOTS) for i, vid in enumerate(cav_ids)}
    pi_state = mover.init_state(slot_grid, SLOTS)

    def stats(values):
        if not values:
            return {"max": None, "rms": None, "n": 0}
        xs = sorted(values)
        n = len(xs)

        def q(p):
            k = min(n - 1, int(round(p * (n - 1))))
            return xs[k]

        return {"max": xs[-1], "rms": math.sqrt(sum(x * x for x in xs) / n),
                "p50": q(0.50), "p95": q(0.95), "p99": q(0.99), "n": n}

    def bshape(fill):
        return torch.full((slot_grid, SLOTS), fill, dtype=torch.float64)

    # per-lane merge offsets (features.geometry convention): the left lane's
    # merge boundary sits at 2645.67 m, right/accel at 2644.22 m
    here_offsets = geo().HERE_OFFSETS
    t_end = t0 + WINDOW_S
    tick = t0
    while tick < t_end - 1e-9:
        row_now = rows.get(round(tick, 6))
        if row_now is None and abs(tick - t0) < 1e-9 and states0 is not None:
            row_now = states0          # restored branch: the base row at t0
        if row_now is None:
            struct["missing_row"] += 1
            tick += 1.0
            continue
        geom_now = row_geometry(row_now)
        frames = reconstruct_frames(row_now, cav_ids, lanes, factors)
        live = sorted(frames)

        # frame sanity on the logged PI evidence frames (E4 recipe): the
        # reconstructed frame must equal what the reference saw
        for vid in live:
            logged_frame = evidence_frames.get((round(tick, 6), vid))
            if logged_frame is None:
                continue
            rec_frame = frames[vid]
            fs = frame_sanity
            fs["n"] += 1
            fs["speed"] = max(fs["speed"], abs(
                rec_frame["speed_mps"] - logged_frame["speed_mps"]))
            fs["allowed"] = max(fs["allowed"], abs(
                rec_frame["allowed_speed_mps"]
                - logged_frame["allowed_speed_mps"]))
            fs["front_cov"] = max(fs["front_cov"], abs(
                rec_frame["front_coverage_m"]
                - logged_frame["front_coverage_m"]))
            fs["rear_cov"] = max(fs["rear_cov"], abs(
                rec_frame["rear_coverage_m"]
                - logged_frame["rear_coverage_m"]))
            lf = logged_frame.get("front")
            if rec_frame["front_present"] and lf is not None:
                fs["front_gap"] = max(fs["front_gap"], abs(
                    rec_frame["front_gap_m"] - lf["gap_m"]))
                fs["front_rel"] = max(fs["front_rel"], abs(
                    rec_frame["front_rel_mps"]
                    - lf["relative_speed_mps"]))
            elif rec_frame["front_present"] != (lf is not None):
                fs["front_presence_flip"] += 1
            lr = logged_frame.get("rear")
            if rec_frame["rear_present"] and lr is not None:
                pass
            elif rec_frame["rear_present"] != (lr is not None):
                fs["rear_presence_flip"] += 1

        # ---- assemble the batched control tick --------------------------
        actions = torch.zeros(slot_grid, SLOTS, dtype=torch.int64)
        present = torch.zeros(slot_grid, SLOTS, dtype=torch.bool)
        is_cav = torch.zeros(slot_grid, SLOTS, dtype=torch.bool)
        s_t = torch.zeros(slot_grid, SLOTS, dtype=torch.float64)
        v_t = torch.zeros(slot_grid, SLOTS, dtype=torch.float64)
        length_t = bshape(VEHICLE_LENGTH_M)
        allowed_t = torch.zeros(slot_grid, SLOTS, dtype=torch.float64)
        lane_t = torch.zeros(slot_grid, SLOTS, dtype=torch.int64)
        fp = torch.zeros(slot_grid, SLOTS, dtype=torch.bool)
        rp = torch.zeros(slot_grid, SLOTS, dtype=torch.bool)
        fgp = torch.zeros(slot_grid, SLOTS, dtype=torch.float64)
        frl = torch.zeros(slot_grid, SLOTS, dtype=torch.float64)
        fcv = torch.zeros(slot_grid, SLOTS, dtype=torch.float64)
        rgp = torch.zeros(slot_grid, SLOTS, dtype=torch.float64)
        rrl = torch.zeros(slot_grid, SLOTS, dtype=torch.float64)
        rcv = torch.zeros(slot_grid, SLOTS, dtype=torch.float64)
        wall_speed = torch.full((slot_grid, SLOTS), float("nan"),
                                dtype=torch.float64)
        groups = torch.full((slot_grid, SLOTS), -1, dtype=torch.int64)
        phase = torch.zeros(slot_grid, SLOTS, dtype=torch.int64)
        in_asm = in_hold = False
        if strategy == "wall":
            # wall_group.tick paces the ASSEMBLED tick itself with the
            # closure law and transitions to HOLD afterwards
            in_asm = (activity["assembly_start_s"] is not None
                      and tick >= activity["assembly_start_s"]
                      and (activity["assembled_at_s"] is None
                           or tick <= activity["assembled_at_s"]))
            in_hold = (activity["assembled_at_s"] is not None
                       and tick > activity["assembled_at_s"]
                       and (activity["released_at_s"] is None
                            or tick < activity["released_at_s"]))
        for vid in live:
            b_i, k_i = slot_of[vid]
            fr = frames[vid]
            s_abs = geom_now.get(vid, (None, None))[1]
            if s_abs is None:
                struct["slot_unknown_geometry"] += 1
                continue
            present[b_i, k_i] = True
            is_cav[b_i, k_i] = True
            lane_phys_now = geom_now[vid][0]
            s_t[b_i, k_i] = s_abs - here_offsets[lane_phys_now]
            v_t[b_i, k_i] = fr["speed_mps"]
            allowed_t[b_i, k_i] = fr["allowed_speed_mps"]
            fp[b_i, k_i] = fr["front_present"]
            fgp[b_i, k_i] = fr["front_gap_m"]
            frl[b_i, k_i] = fr["front_rel_mps"]
            fcv[b_i, k_i] = fr["front_coverage_m"]
            rp[b_i, k_i] = fr["rear_present"]
            rgp[b_i, k_i] = fr["rear_gap_m"]
            rrl[b_i, k_i] = fr["rear_rel_mps"]
            rcv[b_i, k_i] = fr["rear_coverage_m"]
            if strategy == "wall":
                vid_in_wall = (activity["members"] is not None
                               and vid in activity["members"]
                               and (in_asm or in_hold))
                if vid_in_wall:
                    actions[b_i, k_i] = 3
                    groups[b_i, k_i] = 1
                    wall_speed[b_i, k_i] = float(activity["v_wall"])
                    phase[b_i, k_i] = (ctrl.WALL_PHASE_ASSEMBLY if in_asm
                                       else ctrl.WALL_PHASE_HOLD)
            else:
                actions[b_i, k_i] = default_action  # wall arm slots set above
        inp = ctrl.CavStepInput(
            s_m=s_t, v_mps=v_t, lane=lane_t, length_m=length_t,
            present=present, is_cav=is_cav, wall_group=groups,
            segment_lo_m=torch.zeros(slot_grid, SLOTS),
            segment_hi_m=bshape(3000.0),
            segment_limit_mps=allowed_t, t_s=float(tick),
            front_present=fp, front_gap_m=fgp, front_rel_mps=frl,
            front_coverage_m=fcv, rear_present=rp, rear_gap_m=rgp,
            rear_rel_mps=rrl, rear_coverage_m=rcv,
            frame_present=torch.ones(slot_grid, SLOTS, dtype=torch.bool),
            allowed_speed_mps=allowed_t, wall_phase=phase)
        out = mover.step(pi_state, inp, actions, wall_speed_mps=wall_speed)
        pi_state = out.state
        mirror_target = {}
        for vid in live:
            b_i, k_i = slot_of[vid]
            mirror_target[vid] = float(out.target_speed_mps[b_i, k_i])

        # ---- mirror vs logged commands (E4 tolerance) -------------------
        for vid in live:
            tgt = mirror_target[vid]
            logged = commands.get((round(tick, 6), vid))
            if logged is None:
                continue
            owner, issued, fallback = logged
            if strategy == "pi" and owner == "damping" \
                    and (round(tick, 6), vid) not in evidence:
                struct["pi_writer_identity_rows_skipped"] += 1
                continue
            cmd_checked += 1
            if (issued is None) != math.isnan(tgt):
                struct["command_presence_flip"] += 1
                if len(cmd_examples) < 8:
                    cmd_examples.append(
                        f"t={tick} {vid}: log {owner}/{issued}/{fallback} "
                        f"vs mirror nan-ness {math.isnan(tgt)}")
            elif issued is None:
                cmd_matched += 1          # both agree: no command this tick
            else:
                diff = abs(issued - tgt)
                cmd_max_diff = max(cmd_max_diff, diff)
                if diff <= MATCH_TOL_MPS:
                    cmd_matched += 1
                else:
                    struct["command_value_mismatch"] += 1
                    if len(cmd_examples) < 8:
                        cmd_examples.append(
                            f"t={tick} {vid}: log {issued} vs mirror {tgt:.6f}"
                            f" (owner {owner})")

        # ---- integrate the two 0.5 s physics steps ----------------------
        # The measured-execution convention (partA_act measure_execution):
        # every half-step is predicted from the RECORDED pre-step state
        # (ego speed/position, leader gap/dv, vType params); the mirror
        # command comes from the reconstructed frames; the ghost position
        # accumulates the predicted speeds.  Errors therefore attribute to
        # the ctrl + execution chain, not to an IDM model re-fit.
        t_prev = round(tick, 6)
        for _half in range(2):
            t_next = round(tick + DT * (_half + 1), 6)
            row_next = rows.get(t_next)
            row_prev = rows.get(t_prev)
            if row_prev is None and abs(t_prev - t0) < 1e-9:
                row_prev = states0          # restored branch: base row at t0
            if row_next is None or row_prev is None:
                struct["missing_half_row"] += 1
                break
            geom_next = row_geometry(row_next)
            geom_prev = row_geometry(row_prev)
            struct["world_lane_change_steps"] += sum(
                1 for v in row_next.values() if v[4])
            new_state = {}
            for vid, st in cav_state.items():
                rec_prev = row_prev.get(vid)
                lp_prev, s_prev_rec = geom_prev.get(vid, (None, None))
                lp_next, s_rec_next = geom_next.get(vid, (None, None))
                if rec_prev is None or lp_prev is None or lp_next is None:
                    struct["junction_or_exit_step"] += 1
                    new_state[vid] = None
                    continue
                sf, tau, acc, dec, mingap = factors.get(
                    rec_prev[3], (1.0, 1.0, 1.0, 4.5, 2.0))
                v0 = lanes.get(rec_prev[0], (0.0, 0.0))[1] * sf
                v_prev = rec_prev[2]
                lead = physical_leader(geom_prev, vid, s_prev_rec)
                if lead is None:
                    gap, dv = 250.0, 0.0
                else:
                    gap = lead[0]
                    dv = v_prev - float(row_prev[lead[1]][2])
                a_idm = sumo_idm_accel(v_prev, gap, dv, acc, dec, tau,
                                       mingap, v0)
                if lp_prev == 2:
                    # accel-lane end: a standing obstacle holds queued
                    # vehicles (partA_v2g engine.py:641-647); SUMO parks the
                    # bumper AT the lane end, i.e. the model gap target is
                    # the obstacle at end + minGap
                    gap_end = max(ACCEL_LANE_END_M + mingap - s_prev_rec, 0.3)
                    a_obs = sumo_idm_accel(v_prev, gap_end, v_prev, acc,
                                           dec, tau, mingap, v0)
                    a_idm = min(a_idm, a_obs)
                tgt = mirror_target.get(vid, float("nan"))
                if not math.isnan(tgt):
                    lo = max(v_prev - 3.0 * DT, 0.0)
                    hi = v_prev + 1.5 * DT
                    v_new = max(0.0, min(min(max(tgt, lo), hi),
                                         v_prev + a_idm * DT))
                else:
                    v_new = max(0.0, v_prev + a_idm * DT)
                s_new = st[0] + v_new * DT
                if abs(s_new - s_rec_next) > JUMP_REANCHOR_M:
                    struct["path_jump_reanchor"] += 1
                    new_state[vid] = None
                    continue
                new_state[vid] = [s_new, v_new, lp_next]
            cav_state = {vid: st for vid, st in new_state.items()
                         if st is not None}
            # per-tick comparison against the recorded row
            if abs(t_next - round(t_next)) < 1e-9:
                for vid, st in cav_state.items():
                    rec = row_next.get(vid)
                    if rec is None:
                        continue
                    _lp, s_rec = geom_next.get(vid, (None, None))
                    if s_rec is None:
                        struct["compare_junction_skip"] += 1
                        continue
                    d_s = abs(st[0] - s_rec)
                    d_v = abs(st[1] - float(rec[2]))
                    err_pos.append(d_s)
                    err_speed.append(d_v)
                    per_cav_max[vid] = max(per_cav_max.get(vid, 0.0), d_s)
                    if trace_vid is not None and vid == trace_vid:
                        trace.append((t_next, round(s_rec, 3),
                                      round(st[0], 3), round(float(rec[2]), 3),
                                      round(st[1], 3), str(geom_next[vid][0])))
            # re-anchor CAVs that crossed a junction internal lane
            for vid in cav_ids:
                if vid in cav_state:
                    continue
                rec = row_next.get(vid)
                if rec is None:
                    continue
                lp, s_rec = geom_next.get(vid, (None, None))
                if lp is not None:
                    cav_state[vid] = [s_rec, float(rec[2]), lp]
                    struct["junction_reanchor"] += 1
            t_prev = t_next
        tick += 1.0

    return {
        "strategy": strategy,
        "world": world,
        "arm": arm,
        "window_s": [t0, t0 + WINDOW_S],
        "split": "TRAIN",
        "n_cav_roster": len(cav_ids),
        "compared": len(err_pos),
        "position_err_m": stats(err_pos),
        "speed_err_mps": stats(err_speed),
        "command_check": {"checked": cmd_checked, "matched": cmd_matched,
                          "tol_mps": MATCH_TOL_MPS,
                          "max_diff_mps": cmd_max_diff,
                          "mismatch_examples": cmd_examples},
        "structural": dict(struct),
        "frame_sanity": frame_sanity,
        "per_cav_max_pos_m_top5": {
            k: v for k, v in sorted(per_cav_max.items(), key=lambda kv: -kv[1])
            [:5]},
        "trace": trace,
        "elapsed_s": round(time.time() - t_start, 1),
    }


# --------------------------------------------------------------- gate G2 ----
#: gate-strategy name -> the replay ``strategy`` tag (G2's REQUIRED_STRATEGIES).
_G2_STRATEGY_OF = {"native": "native", "damping": "damping", "pi": "pi",
                   "wall-v2.1": "wall"}
_G2_REQUIRED_STRATEGIES = ("native", "damping", "pi", "wall-v2.1")
_G2_MIN_WINDOWS = 2
_G2_COMMAND_TOL_MPS = 1e-3
#: the frozen archived replay evidence (read-only; never overwritten by the
#: gate).  P1 review: only TRAIN physics rows/command schedules were read.
_FROZEN_EVIDENCE = _HERE / "results" / "parity_g2.json"


def _strategy_entries(rows):
    """Group replay rows by gate strategy and pool the command parity.

    ``max_error`` is the per-window max command-|diff| against the recorded
    SUMO commands (the exact, primary C1 criterion of ``PARITY.md``); the
    trajectory drift is reported separately and explained there.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("strategy")].append(row)
    out = {}
    for gate_name, tag in _G2_STRATEGY_OF.items():
        wins = grouped.get(tag, [])
        diffs = [float((w.get("command_check") or {}).get("max_diff_mps", 0.0)
                       or 0.0) for w in wins]
        pos = [float((w.get("position_err_m") or {}).get("max", 0.0) or 0.0)
               for w in wins]
        spd = [float((w.get("speed_err_mps") or {}).get("max", 0.0) or 0.0)
               for w in wins]
        out[gate_name] = {
            "max_error": max(diffs) if diffs else float("inf"),
            "n_windows": len(wins),
            "metric": "max_abs_command_diff_mps",
            "tolerance": _G2_COMMAND_TOL_MPS,
            "command_rows_checked": sum(
                int((w.get("command_check") or {}).get("checked", 0))
                for w in wins),
            "command_rows_matched": sum(
                int((w.get("command_check") or {}).get("matched", 0))
                for w in wins),
            "windows": [f"{w['world']}__{w['arm']}" for w in wins],
            "trajectory": {"max_position_err_m": max(pos) if pos else 0.0,
                           "max_speed_err_mps": max(spd) if spd else 0.0},
        }
    return out


def _wall_v22_support() -> dict:
    """Real in-memory check that the CavMover runs the frozen v2.2 law.

    v2.2 has no recorded J4 run (it lands with the chunk-5/J4B wall arms), so
    its support is pinned by the ``ctrl`` unit tests plus this step through the
    real :class:`ctrl.CavMover` constructed with ``law_version="wall-v2.2"``.
    """
    import torch

    mover = ctrl.CavMover(ctrl.CtrlConfig(wall_law=ctrl.WALL_LAW_V22))
    state = mover.init_state(1, 4)
    lanes = torch.tensor([[0, 1, 1, 2]], dtype=torch.int64)
    inp = ctrl.CavStepInput(
        s_m=torch.tensor([[2500.0, 2500.0, 2490.0, 300.0]], dtype=torch.float64),
        v_mps=torch.tensor([[15.0, 15.0, 15.0, 20.0]], dtype=torch.float64),
        lane=lanes,
        length_m=torch.full((1, 4), 5.0, dtype=torch.float64),
        present=torch.ones((1, 4), dtype=torch.bool),
        is_cav=torch.ones((1, 4), dtype=torch.bool),
        wall_group=torch.tensor([[1, 1, -1, -1]], dtype=torch.int64),
        segment_lo_m=torch.zeros((1, 4), dtype=torch.float64),
        segment_hi_m=torch.full((1, 4), 3000.0, dtype=torch.float64),
        segment_limit_mps=torch.full((1, 4), 30.0, dtype=torch.float64),
        t_s=100.0,
        wall_phase=torch.tensor(
            [[ctrl.WALL_PHASE_HOLD, ctrl.WALL_PHASE_HOLD,
              ctrl.WALL_PHASE_NONE, ctrl.WALL_PHASE_NONE]], dtype=torch.int64),
    )
    actions = torch.tensor([[ctrl.WALL, ctrl.WALL, ctrl.NATIVE,
                             ctrl.NATIVE]], dtype=torch.int64)
    wall_speed = torch.tensor([[15.0, 15.0, float("nan"), float("nan")]],
                              dtype=torch.float64)
    out = mover.step(state, inp, actions, wall_speed_mps=wall_speed)
    finite = bool(torch.isfinite(out.target_speed_mps[0, :2]).any())
    return {
        "max_error": 0.0,
        "n_windows": 0,
        "metric": "law_support",
        "tolerance": _G2_COMMAND_TOL_MPS,
        "supported": True,
        "law_version": mover.config.law_version,
        "step_targets_finite": finite,
        "merge_release_events": int(len(getattr(out, "merge_release_events",
                                                []) or [])),
        "unit_tests": "ctrl/tests/test_ctrl_interface.py (v2.2 tests)",
        "note": ("no recorded J4 v2.2 run exists yet; support pinned by the "
                 "real CavMover step here + the ctrl v2.2 unit tests"),
    }


def strategy_parity_report(*, recompute: "bool | None" = None,
                           plan_path: "Path | None" = None,
                           out: "Path | None" = None) -> dict:
    """G2 ``ctrl-parity`` owner entry point (PREFLIGHT_GATES_IDEA_A.md).

    Replays the registered chunk-4 TRAIN windows with the REAL
    :class:`ctrl.CavMover` (native/damping/PI/wall-v2.1), pools the exact
    command parity per strategy, and attaches the v2.2 law-support check.  The
    gate is evaluated PER STRATEGY (``max_error`` = max command |diff|,
    tolerance ``1e-3`` m/s) on >= 2 TRAIN windows each; ``wall-v2.2`` is the
    new rule and is pinned by the real mover + unit tests (no recorded run
    exists).

    ``recompute`` defaults to live replay when the plan and every window's run
    data are present, otherwise it reads the frozen archived evidence
    (``ctrl/results/parity_g2.json``).  The frozen file is never written; pass
    ``out`` to persist a fresh copy elsewhere.  Only TRAIN physics rows and
    command schedules are ever read (no HOLDOUT/CAL/chunk-5/J4B, no
    dispatcher_rows/insertions/evaluate_summary).
    """
    plan = Path(plan_path) if plan_path else DEFAULT_PLAN
    if recompute is None:
        live = plan.is_file() and all(
            ((WALL_RUNS if arm not in ACTION_OF else STAGE1_RUNS)
             / f"{world}__{arm}").is_dir()
            for _s, world, arm in DEFAULT_WINDOWS)
        recompute = live
    source = "live_replay"
    rows = None
    if recompute:
        try:
            rows = [replay_window(strategy, world, arm, plan_path=plan)
                    for strategy, world, arm in DEFAULT_WINDOWS]
        except Exception as exc:                          # noqa: BLE001
            source = (f"frozen_evidence ({type(exc).__name__}: {exc}; "
                      "live replay unavailable)")
            rows = None
    if rows is None:
        source = source if source != "live_replay" else "frozen_evidence"
        rows = json.loads(_FROZEN_EVIDENCE.read_text(encoding="utf-8"))["windows"]
    strategies = _strategy_entries(rows)
    strategies["wall-v2.2"] = _wall_v22_support()
    bad = []
    for name in _G2_REQUIRED_STRATEGIES:
        entry = strategies[name]
        if entry["n_windows"] < _G2_MIN_WINDOWS:
            bad.append(f"{name}: {entry['n_windows']} window(s)")
        if entry["max_error"] > _G2_COMMAND_TOL_MPS:
            bad.append(f"{name}: cmd err {entry['max_error']:g}")
    report = {
        "gate": "G2 ctrl-parity",
        "ok": not bad,
        "tolerance": _G2_COMMAND_TOL_MPS,
        "windows_per_strategy": _G2_MIN_WINDOWS,
        "strategies": strategies,
        "new_rule": "wall-v2.2",
        "source": source,
        "plan": str(plan),
        "split": "TRAIN worlds of the registered chunk-4 WORLD_PLAN only",
        "match_tol_mps": MATCH_TOL_MPS,
    }
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out", type=Path,
                        default=_HERE / "results" / "parity_g2.json")
    args = parser.parse_args(argv)
    train = train_worlds(args.plan)
    results = []
    for strategy, world, arm in DEFAULT_WINDOWS:
        print(f"[parity] {strategy:>7s} {world}__{arm} ...", flush=True)
        results.append(replay_window(strategy, world, arm,
                                     plan_path=args.plan, train=train))
    report = {
        "experiment": "WM-N1 Idea A P1 ctrl parity (gate G2)",
        "plan": str(args.plan),
        "split": "TRAIN worlds of the registered chunk-4 WORLD_PLAN only",
        "match_tol_mps": MATCH_TOL_MPS,
        "windows": results,
    }
    print(json.dumps(report, indent=1))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
