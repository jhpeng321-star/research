"""WALLREGEN: regenerate the unseen-wall evaluation states in a (new) SUMO driver population.

The world model's key evaluation is prediction on UNSEEN wall actions (walls never enter WM
training).  The original state sets (``data_cache/wallc1_states_h60{,_real}``) come from the
WM-59 walls pilot (chunk-4 worlds, legacy 3-type population).  This driver re-creates
equivalent sets in any population supported by :class:`traffic_shepherd.c4.env.CavSumoEnv`
(``population="textbook"`` once c4/popfix is merged), on a registered fresh seed block.

One plan world = one fresh SUMO episode:

1. native warmup ``[0, t_w]`` (all CAVs released to SUMO), entry times tracked for the RSU
   entrance counter, pre-``t_w`` history kept (last 120 s);
2. ``saveState`` at ``t_w`` (precision 17 + RNG, the branching convention) and a deep copy of
   the env's Python state (:mod:`traffic_shepherd.c4.branching` save/restore);
3. one branch per arm, each loaded into a FRESH SUMO process from the same state (common random
   numbers): ``native`` (every CAV native) and the four WM-59 wall arms
   ``w-{lo,mid}-{early,late}``; optional ``native_repeat`` determinism check.  Each branch runs
   ``branch_s`` seconds and its 0.5 s slot state is taped (``arm_<name>.npz``).

Wall arms reproduce the WM-59 pilot exactly as registered and run (law ``wall-v2.1-closure`` =
:class:`traffic_shepherd.wall_group.WallGroupSession`, ledger WM-59 Amendment B; request rule =
``output/campaign_walls_v1/tests/wall_v21_pre/wall_request.py``):

* request: early = first control tick >= t_w with a feasible pair, late = first feasible tick
  >= t_w + 60; ``not_feasible`` when none appears within 120 s after the target;
* feasible pair: two live CAVs on the entry edge 1191554636 in different lanes, span <= 75 m,
  road_left(front member) >= v_wall x 80 s, from CAV report frames only; member choice =
  minimal span, ties by most road left, then ids;
* wall speed = 0.60 (lo) / 0.85 (mid) x the members' mean reported speed, floor 2.5 m/s;
* hold H in {60, 120, 240} s (drawn per world/arm from the plan seed) counted from assembly
  completion, then release to NATIVE; road-end release in HOLD when any member's road left
  < v_wall x 10 s; safety aborts (any phase) hand back to native;
* non-members NATIVE for the whole branch; every CAV command through ``CommandOwnership``
  (speedMode 27) via the env's ``direct`` path with the adopted 1.5 m/s^2 ceiling (the law
  clips to [-3, 1.5], so the ceiling never binds - identical to the dispatcher write).

The dispatcher order per tick is kept: request -> road-end release -> hold-term release ->
``session.tick`` (commands of a wall that hands back this tick are dropped).

CLI::

    python -m traffic_shepherd.c4.wallregen plan --out PLAN.json [--n-worlds 40 --population textbook]
    python -m traffic_shepherd.c4.wallregen run --plan PLAN.json --out RUN_DIR --workers 46
    python -m traffic_shepherd.c4.wallregen status --out RUN_DIR

``run`` is resumable (complete worlds are skipped; an incomplete world directory is preserved
as ``<world>.incomplete-<n>`` and redone) and parallel (one fresh process per world, spawn).
Convert the tapes with :mod:`traffic_shepherd.c4.wallregen_states`.
"""
from __future__ import annotations

import os as _os  # noqa: E402

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from math import isfinite  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

# ============================================================ DEMAND LEVELS
#: Mainline demand levels (veh/h) of the regenerated worlds; the ONE variable to set.  None =
#: undecided: ``plan`` refuses unless ``--demands`` (or ``DEMANDS=`` in run_wallregen.sh) is
#: given.  The old sets used 2400/2600/2700 (legacy 3-type population); the textbook population
#: saturates at ~1930 veh/h mainline (coordinator 2026-10-14), so those levels give unbounded
#: source queues there.  Any multiple of 50 in [1000, 2800] passes the seed guard.
DEMANDS_VPH: tuple | None = None

SCHEMA_PLAN = "c4_wallregen_plan/1"
SCHEMA_WORLD = "c4_wallregen_world/1"
FORBIDDEN = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1", re.I)

# --------------------------------------------------------------------------- WM-59 v2.1 rules
#: pinned from output/campaign_walls_v1/tests/wall_v21_pre/common.py (the pilot as run)
ARMS = ("w-lo-early", "w-lo-late", "w-mid-early", "w-mid-late")
ARM_SPEED_FACTORS = {"w-lo": 0.60, "w-mid": 0.85}
TIMING_OFFSET_S = {"early": 0.0, "late": 60.0}
REQUEST_WINDOW_S = 120.0
SPEED_FLOOR_MPS = 2.5
HOLD_CHOICES_S = (60.0, 120.0, 240.0)
MEMBER_EDGES = frozenset({"1191554636"})
REQUEST_SPAN_CAP_M = 75.0
UPSTREAM_EDGE_LENGTH_M = 2644.2
DOWNSTREAM_ROAD_M = 75.4 + 477.9            # 1487606179 + 1487606178
ROAD_LEFT_REQUEST_S = 80.0
ROAD_END_RELEASE_S = 10.0
WALL_LAW = "wall-v2.1-closure"
#: the adopted CAV acceleration ceiling (config bound) for the wall arm's direct write
WALL_ACCEL_CEILING_MPS2 = 1.5

# --------------------------------------------------------------------------- plan defaults
DEFAULT_SEEDS = "962000-962035"
DEFAULT_PENETRATIONS = (0.1, 0.2, 0.4)
DEFAULT_T_CHOICES = (240, 300, 360, 420, 480, 540)
DEFAULT_PLAN_SEED = 20261015
DEFAULT_BRANCH_S = 600                       # = the WM-59 pilot BRANCH_LENGTH_S
HISTORY_TICKS = 241                          # 120 s of 0.5 s ticks kept before t_w


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable), encoding="utf-8")
    tmp.replace(path)


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (set, frozenset, tuple)):
        return sorted(value) if isinstance(value, (set, frozenset)) else list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def checked(path) -> Path:
    path = Path(path)
    if FORBIDDEN.search(str(path).replace("\\", "/")):
        raise ValueError(f"restricted path token in {path}")
    return path


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


# =========================================================================== request rule
def arm_speed_factor(arm: str) -> float:
    return ARM_SPEED_FACTORS["-".join(arm.split("-")[:2])]


def arm_timing(arm: str) -> str:
    timing = arm.rsplit("-", 1)[-1]
    if timing not in TIMING_OFFSET_S:
        raise ValueError(f"unknown arm timing in {arm!r}")
    return timing


def edge_of_lane(lane_id) -> str | None:
    if not isinstance(lane_id, str) or not lane_id or lane_id.startswith(":"):
        return None
    edge, sep, index = lane_id.rpartition("_")
    if not sep or not index.isdigit():
        return None
    return edge


def wall_speed_for(factor: float, member_speeds: dict) -> float:
    speeds = [float(s) for s in member_speeds.values()]
    if not speeds or any(not isfinite(s) or s < 0 for s in speeds):
        raise ValueError("member speeds must be finite and nonnegative")
    return round(max(SPEED_FLOOR_MPS, factor * (sum(speeds) / len(speeds))), 6)


def road_left_m(frame) -> float | None:
    """(2644.2 - position on 1191554636) + 75.4 + 477.9 from the REPORT frame (pilot rule)."""
    position = getattr(frame, "position_m", None)
    if position is not None and isfinite(float(position)):
        return round(UPSTREAM_EDGE_LENGTH_M - float(position) + DOWNSTREAM_ROAD_M, 6)
    distance = getattr(frame, "distance_to_merge_m", None)
    if distance is not None and isfinite(float(distance)):
        return round(float(distance) + DOWNSTREAM_ROAD_M, 6)
    return None


def feasible_pairs(frames: dict, factor: float | None = None) -> list[dict]:
    """Pilot ``feasible_pairs`` (v2.1 pre copy): entry edge only, different lanes, span <= 75 m,
    road_left(front) >= v_wall x 80 s; order: minimal span, most road left, member ids."""
    from traffic_shepherd.wall_group import downstream_position_m

    by_edge_lane: dict[tuple, list[dict]] = {}
    for vehicle_id, frame in frames.items():
        if frame is None:
            continue
        edge = edge_of_lane(getattr(frame, "lane_id", None))
        if edge is None or edge not in MEMBER_EDGES:
            continue
        s_m = downstream_position_m(frame)
        left = road_left_m(frame)
        lane = getattr(frame, "lane_index", None)
        if s_m is None or left is None or lane is None:
            continue
        by_edge_lane.setdefault((edge, int(lane)), []).append({
            "vehicle_id": str(vehicle_id), "s_m": float(s_m),
            "distance_to_merge_m": -float(s_m),
            "speed_mps": float(getattr(frame, "speed_mps", 0.0) or 0.0),
            "lane_index": int(lane), "road_left_m": float(left)})
    pairs: list[dict] = []
    for edge in sorted({e for (e, _l) in by_edge_lane}):
        lanes = sorted(lane for (e, lane) in by_edge_lane if e == edge)
        for ia, lane_a in enumerate(lanes):
            for lane_b in lanes[ia + 1:]:
                for a in by_edge_lane[(edge, lane_a)]:
                    for b in by_edge_lane[(edge, lane_b)]:
                        span = abs(a["s_m"] - b["s_m"])
                        if span > REQUEST_SPAN_CAP_M + 1e-9:
                            continue
                        members = sorted([a, b], key=lambda m: m["lane_index"])
                        front_left = min(a["road_left_m"], b["road_left_m"])
                        speeds = {m["vehicle_id"]: m["speed_mps"] for m in members}
                        if factor is not None and front_left < (
                                wall_speed_for(factor, speeds) * ROAD_LEFT_REQUEST_S):
                            continue
                        pairs.append({
                            "edge": edge,
                            "members": tuple(m["vehicle_id"] for m in members),
                            "lanes": [m["lane_index"] for m in members],
                            "span_m": round(span, 6),
                            "mean_distance_to_merge_m": round(
                                (a["distance_to_merge_m"] + b["distance_to_merge_m"]) / 2.0, 6),
                            "road_left_m": front_left,
                            "member_speeds": speeds})
    pairs.sort(key=lambda p: (p["span_m"], -p["road_left_m"], p["members"]))
    return pairs


def choose_pair(frames: dict, factor: float | None = None) -> dict | None:
    pairs = feasible_pairs(frames, factor)
    return pairs[0] if pairs else None


class WallArmRunner:
    """One WM-59 wall arm: request state machine + the v2.1 ``WallGroupSession`` lifecycle
    with the dispatcher's tick order (``MixedAssignmentSession._tick_walls``)."""

    def __init__(self, *, arm: str, run_name: str, t_branch_s: float, hold_s: float):
        self.arm = arm
        self.run_name = run_name
        self.factor = arm_speed_factor(arm)
        self.timing = arm_timing(arm)
        self.t_branch_s = float(t_branch_s)
        self.target_s = self.t_branch_s + TIMING_OFFSET_S[self.timing]
        self.deadline_s = self.target_s + REQUEST_WINDOW_S
        self.hold_s = float(hold_s)
        self.wall_id = f"{run_name}:wall1"
        self.state = "waiting"
        self.session = None
        self.pair: dict | None = None
        self.wall_speed_mps: float | None = None
        self.requested_at: float | None = None
        self.assembled_at: float | None = None
        self.finished_at: float | None = None
        self.finished_how: str | None = None       # released | safety_abort
        self.phase_at_finish: str | None = None
        self.abort_reason: str | None = None
        self.road_end_release_at: float | None = None
        self.road_end_road_left_m: float | None = None
        self._staged_release = False
        self.events: list[dict] = []
        self.member_lanes: list[dict] = []

    # ------------------------------------------------------------------ per tick
    def tick(self, frames: dict, now_s: float, *, live_ids=(), arrived_ids=()) -> dict:
        from traffic_shepherd.wall_group import (PHASE_HANDBACK, PHASE_HOLD, WallGroupSession,
                                                 validate_wall_request)

        now_s = float(now_s)
        if self.state == "waiting" and now_s >= self.target_s - 1e-9:
            pair = choose_pair(frames, self.factor)
            if pair is not None:
                speed = wall_speed_for(self.factor, pair["member_speeds"])
                members = validate_wall_request(pair["members"], speed, {})
                self.session = WallGroupSession(self.wall_id, members=members,
                                                wall_speed_mps=speed)
                self.state, self.pair, self.wall_speed_mps = "requested", pair, speed
                self.requested_at = now_s
                self.events.append({"kind": "wall_request", "timestamp_s": now_s,
                                    "members": list(pair["members"]), "lanes": pair["lanes"],
                                    "span_m": pair["span_m"], "road_left_m": pair["road_left_m"],
                                    "speed_factor": self.factor, "wall_speed_mps": speed,
                                    "hold_s": self.hold_s, "member_speeds": pair["member_speeds"]})
            elif now_s > self.deadline_s + 1e-9:
                self.state = "not_feasible"
                self.events.append({"kind": "not_feasible", "timestamp_s": now_s,
                                    "target_time_s": self.target_s, "deadline_s": self.deadline_s})
        session = self.session
        if session is None or self.finished_at is not None:
            return {}
        # A4 road-end release (planner release, HOLD only, once)
        if (session.phase == PHASE_HOLD and self.road_end_release_at is None):
            lefts = {m: road_left_m(frames[m]) for m in session.members
                     if frames.get(m) is not None}
            lefts = {m: v for m, v in lefts.items() if v is not None}
            if lefts and min(lefts.values()) < self.wall_speed_mps * ROAD_END_RELEASE_S:
                self._staged_release = True
                self.road_end_release_at = now_s
                self.road_end_road_left_m = float(min(lefts.values()))
                self.events.append({"kind": "road_end_release", "timestamp_s": now_s,
                                    "road_left_m": lefts})
        # member lanes (reported) before the step, as the pilot logs them
        self.member_lanes.append({"t": now_s, "lanes": {
            m: (getattr(frames[m], "lane_id", None) if frames.get(m) is not None else None)
            for m in sorted(session.members)}})
        phase_before = "hold" if session.assembled_at_s is not None else "assembly"
        if session.phase == PHASE_HOLD:
            due = self._staged_release or (
                session.assembled_at_s is not None
                and now_s - session.assembled_at_s >= self.hold_s)
            if due:
                session.release(now_s)
        self._staged_release = False
        commands = session.tick(frames, now_s)
        if session.assembled_at_s is not None and self.assembled_at is None:
            self.assembled_at = float(session.assembled_at_s)
        for event in session.events:
            self.events.append(dict(event))
        session.events.clear()
        if session.phase == PHASE_HANDBACK:
            self.finished_at = now_s
            self.finished_how = "safety_abort" if session.abort_reason else "released"
            self.phase_at_finish = ("hold" if session.assembled_at_s is not None
                                    else phase_before)
            reason = session.abort_reason
            if reason == "member_frame_missing" and any(
                    m in set(arrived_ids) for m in session.members):
                reason = "member_left_network"
            self.abort_reason = reason
            return {}
        return commands

    # ------------------------------------------------------------------ outcome
    def outcome(self) -> str:
        if self.state == "not_feasible":
            return "not_feasible"
        if self.state == "waiting":
            return "waiting_at_end"
        if self.finished_at is None:
            return "in_hold_at_end" if self.assembled_at is not None else "in_assembly_at_end"
        if self.finished_how == "released":
            return "released_road_end" if self.road_end_release_at is not None \
                else "assembled_released"
        return "aborted_in_hold" if self.phase_at_finish == "hold" else "aborted_in_assembly"

    def timeline(self, t_end_s: float) -> dict:
        tb = self.t_branch_s

        def rel(x):
            return None if x is None else round(float(x) - tb, 6)

        owner_end = (self.finished_at if self.finished_at is not None
                     else (float(t_end_s) if self.requested_at is not None else None))
        outcome = self.outcome()
        achieved = None
        if self.assembled_at is not None:
            achieved = round((self.finished_at if self.finished_at is not None
                              else float(t_end_s)) - self.assembled_at, 6)
        return {
            "arm": self.arm, "run": self.run_name, "wall_id": self.wall_id,
            "law_version": WALL_LAW, "timing": self.timing, "speed_factor": self.factor,
            "t0_s": tb, "target_s": self.target_s, "deadline_s": self.deadline_s,
            "members": list(self.pair["members"]) if self.pair else [],
            "member_lanes_at_request": list(self.pair["lanes"]) if self.pair else [],
            "span_m": self.pair["span_m"] if self.pair else None,
            "wall_speed_mps": self.wall_speed_mps,
            "request_s": self.requested_at, "request_rel_s": rel(self.requested_at),
            "assembled_at_s": self.assembled_at, "assembled_rel_s": rel(self.assembled_at),
            "released_at_s": self.finished_at if self.finished_how == "released" else None,
            "released_rel_s": rel(self.finished_at) if self.finished_how == "released" else None,
            "aborted_at_s": self.finished_at if self.finished_how == "safety_abort" else None,
            "aborted_rel_s": rel(self.finished_at) if self.finished_how == "safety_abort" else None,
            "abort_reason": self.abort_reason,
            "road_end_release_s": self.road_end_release_at,
            "road_end_min_road_left_m": self.road_end_road_left_m,
            "handback_s": self.finished_at, "handback_to": "native" if self.finished_at else None,
            "owner_end_s": owner_end, "owner_end_rel_s": rel(owner_end),
            "owned": self.requested_at is not None,
            "engaged": self.assembled_at is not None,
            "requested_hold_s": self.hold_s, "achieved_hold_s": achieved,
            "outcome": outcome, "status": outcome,
            "branch_end_s": float(t_end_s), "n_events": len(self.events),
        }


# =========================================================================== plan
def _hold_for(plan_seed: int, seed: int, arm: str) -> float:
    digest = hashlib.sha256(f"{plan_seed}:{seed}:{arm}:hold".encode()).digest()
    return HOLD_CHOICES_S[digest[0] % len(HOLD_CHOICES_S)]


def world_name(seed: int, demand: int, penetration: float, t_w: int) -> str:
    """``w<seed>m<demand>p<pct>t<t0>`` (``wms.eval.states.world_meta`` grammar)."""
    pct = float(penetration) * 100.0
    if abs(pct - round(pct)) > 1e-6:
        raise ValueError(f"penetration {penetration} is not a whole percent")
    return f"w{int(seed)}m{int(demand)}p{int(round(pct)):02d}t{int(t_w)}"


def code_shas() -> dict:
    """LF-normalised sha256 of the code that shapes the simulated data (provenance)."""
    root = Path(__file__).resolve().parents[2]
    out = {}
    for rel in ("traffic_shepherd/c4/wallregen.py", "traffic_shepherd/wall_group.py",
                "traffic_shepherd/c4/env.py", "traffic_shepherd/scenarios.py",
                "traffic_shepherd/ownership.py", "traffic_shepherd/c4/population.py"):
        p = root / rel
        if p.is_file():
            out[rel] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return out


def make_plan(*, seeds: list[int], penetrations, demands, t_choices, plan_seed: int,
              branch_s: int, arms=ARMS, population: str = "textbook",
              repeat_every: int = 10) -> dict:
    """Deterministic world list: penetration cycles fastest, then demand; t_w drawn per seed."""
    from traffic_shepherd.c4.seeds import check_case

    rng = random.Random(int(plan_seed))
    worlds = []
    for i, seed in enumerate(seeds):
        pen = float(penetrations[i % len(penetrations)])
        demand = int(demands[(i // len(penetrations)) % len(demands)])
        t_w = int(rng.choice(list(t_choices)))
        check_case(int(seed), demand)
        name = world_name(seed, demand, pen, t_w)
        branch_arms = ["native"] + list(arms)
        if repeat_every and i % int(repeat_every) == 0:
            branch_arms.append("native_repeat")
        worlds.append({"world": name, "seed": int(seed), "demand_vph": demand,
                       "penetration": pen, "t_w_s": t_w, "branch_s": int(branch_s),
                       "arms": branch_arms,
                       "holds_s": {arm: _hold_for(plan_seed, seed, arm) for arm in arms}})
    return {"schema": SCHEMA_PLAN, "created": _now(), "plan_seed": int(plan_seed),
            "population": population, "wall_law": WALL_LAW,
            "request_rule": "WM-59 pilot v2.1 (output/campaign_walls_v1/tests/wall_v21_pre)",
            "seed_block": "C4_WALLREGEN_EVAL_SEEDS (traffic_shepherd.seeds)",
            "branch_s": int(branch_s), "n_worlds": len(worlds), "worlds": worlds}


# =========================================================================== tape
class Tape:
    """0.5 s slot state of one branch (compact columns; ``kin`` is rebuilt by the converter)."""

    def __init__(self):
        self.t, self.s, self.lane, self.v, self.a, self.mask, self.vid = ([] for _ in range(7))

    def add_raw(self, raw: dict, times) -> None:
        kin = raw["kin"]
        for j, t in enumerate(times):
            k = kin[j]
            onehot = k[:, 1:4]
            lane = np.where(onehot.sum(-1) > 0, onehot.argmax(-1), -1).astype(np.int8)
            self.t.append(float(t))
            self.s.append(k[:, 0].astype(np.float32))
            self.lane.append(lane)
            self.v.append(k[:, 4].astype(np.float32))
            self.a.append(k[:, 5].astype(np.float32))
            self.mask.append(raw["mask"][j].astype(bool))
            self.vid.append(raw["vid_index"][j].astype(np.int32))

    def arrays(self) -> dict:
        return {"time_s": np.asarray(self.t, np.float64), "s": np.stack(self.s),
                "lane": np.stack(self.lane), "v": np.stack(self.v), "a": np.stack(self.a),
                "mask": np.stack(self.mask), "vid": np.stack(self.vid)}


class FirstSeen:
    """Per-vehicle first occupied tick and entrance (1 = ramp lane 2, else 0)."""

    def __init__(self):
        self.d: dict[int, tuple[float, int]] = {}

    def update(self, raw: dict, times) -> None:
        kin, mask, vid = raw["kin"], raw["mask"], raw["vid_index"]
        for j, t in enumerate(times):
            for slot in np.flatnonzero(mask[j] & (vid[j] >= 0)).tolist():
                v = int(vid[j][slot])
                if v not in self.d:
                    self.d[v] = (float(t), 1 if kin[j][slot, 3] > 0.5 else 0)

    def arrays(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        ts = np.full(n, np.nan, np.float64)
        ent = np.full(n, -1, np.int8)
        for v, (t, e) in self.d.items():
            if v < n:
                ts[v], ent[v] = t, e
        return ts, ent


def _last_ticks(env, n: int):
    raw = env.history.raw(n)
    now = env.now
    times = [now - 0.5 * (raw["kin"].shape[0] - 1 - j) for j in range(raw["kin"].shape[0])]
    return raw, times


def _capture_frames(env) -> tuple[dict, set]:
    now = env.now
    live = set(env.api.vehicle.getIDList())
    frames = {}
    env.sensor.bind(env._table)
    for cav in sorted(live & env.cav_ids):
        try:
            frames[cav] = env.sensor.capture(cav, now_s=now)
        except (ValueError, KeyError):
            pass
    return frames, live


def _accel_slots(env, commands: dict):
    if not commands:
        return None
    acc = np.full((1, env.k), np.nan, np.float64)
    f = env.frame
    for vid, a in commands.items():
        slot = env.builder.pool.slot_of(vid)
        if slot is not None and f.present[slot]:
            acc[0, slot] = float(a)
    return acc


# =========================================================================== one world
def make_env(population: str, scenario_root: Path, cav_vtype=None):
    from traffic_shepherd.c4.branching import SAVE_STATE_CLI
    from traffic_shepherd.c4.env import CavSumoEnv

    kwargs = dict(controller="direct", scenario_root=scenario_root, extra_sumo_cli=SAVE_STATE_CLI,
                  history_ticks=HISTORY_TICKS, physics_log=False, ttc=True,
                  direct_accel_max_mps2=WALL_ACCEL_CEILING_MPS2)
    params = inspect.signature(CavSumoEnv.__init__).parameters
    if "population" in params:
        kwargs["population"] = population
    elif population not in ("legacy", "legacy3"):
        raise RuntimeError(f"population={population!r} requested but this tree's CavSumoEnv has no "
                           "population argument (c4/popfix not merged)")
    if cav_vtype is not None:
        kwargs["cav_vtype"] = cav_vtype
    return CavSumoEnv(**kwargs)


def run_world(world: dict, out_dir: Path, *, population: str, plan_seed: int,
              cav_vtype=None) -> dict:
    """Base episode + all branches of one plan world; writes the world directory."""
    from traffic_shepherd.c4.branching import load_state, save_state, snapshot_env

    out_dir = checked(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    t_start = time.monotonic()
    name, t_w, branch_s = world["world"], int(world["t_w_s"]), int(world["branch_s"])
    env = make_env(population, out_dir / "scenario", cav_vtype)
    first = FirstSeen()

    def hook(e):
        raw, times = _last_ticks(e, 2)
        first.update(raw, times)

    record: dict = {"schema": SCHEMA_WORLD, "world": name, "plan_world": world,
                    "population": population, "plan_seed": plan_seed, "started": _now(),
                    "code_sha256": code_shas(),
                    "arms": {}, "complete": False}
    try:
        env.reset(seed=int(world["seed"]), demand_vph=int(world["demand_vph"]),
                  penetration=float(world["penetration"]), warmup_s=t_w,
                  horizon_s=branch_s + 2, controller="direct", warmup_hook=hook)
        record["population_manifest"] = {k: env.scenario_manifest.get(k) for k in
                                         ("driver_population", "driver_population_version",
                                          "cav_vtype") if isinstance(env.scenario_manifest, dict)}
        record["cav_vtype"] = env.cav_vtype
        record["n_designated_cavs"] = len(env.cav_ids)
        raw0, times0 = _last_ticks(env, HISTORY_TICKS)
        base = Tape()
        base.add_raw(raw0, times0)
        registry0 = list(env.builder.vehicle_ids)
        fs_t, fs_e = first.arrays(len(registry0))
        np.savez_compressed(out_dir / "base.npz", **base.arrays(), first_seen_s=fs_t,
                            first_entrance=fs_e)
        record["base"] = {"t_w_s": env.now, "registry_n": len(registry0),
                          "history_ticks": len(base.t), "wall_s": time.monotonic() - t_start}
        state_path = out_dir / "state_tw.xml"
        save_state(env, state_path)
        snap = snapshot_env(env)
        first_snap = copy.deepcopy(first.d)
        for arm in world["arms"]:
            t_arm = time.monotonic()
            note = load_state(env, state_path, snap)
            first.d = copy.deepcopy(first_snap)
            runner = None
            if arm.startswith("w-"):
                runner = WallArmRunner(arm=arm, run_name=f"{name}__{arm}", t_branch_s=env.now,
                                       hold_s=float(world["holds_s"][arm]))
            tape = Tape()
            raw, times = _last_ticks(env, 1)
            tape.add_raw(raw, times)
            totals0 = dict(env.totals)
            commanded_ticks = 0
            for _ in range(branch_s):
                accel = None
                if runner is not None:
                    frames, live = _capture_frames(env)
                    cmds = runner.tick(frames, env.now, live_ids=live,
                                       arrived_ids=env.ledger.arrived)
                    accel = _accel_slots(env, cmds)
                    commanded_ticks += int(bool(cmds))
                _, info = env.step(accel, fallback="native")
                if info["done"]:
                    raise RuntimeError("episode ended inside a branch")
                raw, times = _last_ticks(env, 2)
                tape.add_raw(raw, times)
                first.update(raw, times)
            registry = list(env.builder.vehicle_ids)
            cav_flags = np.asarray([v in env.cav_ids for v in registry], bool)
            fs_t, fs_e = first.arrays(len(registry))
            np.savez_compressed(out_dir / f"arm_{arm}.npz", **tape.arrays(),
                                first_seen_s=fs_t, first_entrance=fs_e, is_cav=cav_flags)
            totals = {k: env.totals.get(k, 0) - totals0.get(k, 0) for k in
                      ("collision_involvements", "starting_teleports", "arrivals",
                       "departures", "dropped_vehicle_steps")}
            arm_rec = {"registry": registry, "t_start_s": t_w, "t_end_s": env.now,
                       "n_ticks": len(tape.t), "wall_s": time.monotonic() - t_arm,
                       "load_note": note, "totals": totals, "commanded_ticks": commanded_ticks}
            if runner is not None:
                arm_rec["wall_timeline"] = runner.timeline(env.now)
                arm_rec["wall_events"] = runner.events
                arm_rec["member_lanes"] = runner.member_lanes
            record["arms"][arm] = arm_rec
        env.close()
        try:
            state_path.unlink()
        except OSError:
            pass
        if "native_repeat" in record["arms"]:
            a = np.load(out_dir / "arm_native.npz")
            b = np.load(out_dir / "arm_native_repeat.npz")
            same_shape = a["s"].shape == b["s"].shape
            record["determinism_native_vs_repeat"] = {
                "same_shape": same_shape,
                "max_abs_ds_m": float(np.abs(a["s"] - b["s"]).max()) if same_shape else None,
                "vid_equal": bool(same_shape and np.array_equal(a["vid"], b["vid"]))}
        record.update(complete=True, finished=_now(), wall_s=time.monotonic() - t_start,
                      sim_s=t_w + branch_s * len(world["arms"]))
        _write_json(out_dir / "world.json", record)
        return {"world": name, "ok": True, "wall_s": record["wall_s"], "sim_s": record["sim_s"],
                "outcomes": {a: r.get("wall_timeline", {}).get("outcome")
                             for a, r in record["arms"].items()}}
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


# =========================================================================== runner
def _world_complete(path: Path) -> bool:
    p = path / "world.json"
    if not p.is_file():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("complete"))
    except (OSError, json.JSONDecodeError):
        return False


def _job(args: tuple) -> dict:
    world, out_root, population, plan_seed, cav_vtype = args
    target = Path(out_root) / "worlds" / world["world"]
    t0 = time.monotonic()
    try:
        return run_world(world, target, population=population, plan_seed=plan_seed,
                         cav_vtype=cav_vtype)
    except Exception as exc:  # noqa: BLE001 - recorded, never silently retried
        fail = {"world": world["world"], "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-3000:], "wall_s": time.monotonic() - t0}
        try:
            target.mkdir(parents=True, exist_ok=True)
            _write_json(target / "FAILED.json", fail)
        except OSError:
            pass
        return fail


def run_plan(plan_path: Path, out: Path, *, workers: int, only=None, limit=None,
             cav_vtype=None) -> dict:
    plan_path, out = checked(plan_path), checked(out)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan.get("schema") != SCHEMA_PLAN:
        raise SystemExit(f"not a wallregen plan: {plan_path}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "worlds").mkdir(exist_ok=True)
    plan_copy = out / "PLAN.json"
    if plan_copy.is_file():
        if json.loads(plan_copy.read_text(encoding="utf-8"))["worlds"] != plan["worlds"]:
            raise SystemExit(f"{out} holds a different plan; use a new --out")
    else:
        _write_json(plan_copy, plan)
    worlds = plan["worlds"]
    if only:
        keep = set(only)
        worlds = [w for w in worlds if w["world"] in keep]
    if limit:
        worlds = worlds[:int(limit)]
    todo = []
    for w in worlds:
        d = out / "worlds" / w["world"]
        if _world_complete(d):
            continue
        if d.exists():
            n = 1
            while (out / "worlds" / f"{w['world']}.incomplete-{n}").exists():
                n += 1
            d.rename(out / "worlds" / f"{w['world']}.incomplete-{n}")
        todo.append(w)
    print(json.dumps({"plan_worlds": len(plan["worlds"]), "selected": len(worlds),
                      "todo": len(todo), "workers": workers}), flush=True)
    t0 = time.monotonic()
    results = []
    log = out / "run_log.jsonl"
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=max(1, min(int(workers), len(todo))), maxtasksperchild=1) as pool:
            jobs = [(w, str(out), plan["population"], int(plan["plan_seed"]), cav_vtype)
                    for w in todo]
            for res in pool.imap_unordered(_job, jobs):
                res["at"] = _now()
                results.append(res)
                with log.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({k: v for k, v in res.items() if k != "traceback"},
                                        default=_jsonable) + "\n")
                print(json.dumps({k: res.get(k) for k in ("world", "ok", "wall_s", "error")}),
                      flush=True)
    summary = status(out)
    summary["this_invocation"] = {"todo": len(todo), "ok": sum(r["ok"] for r in results),
                                  "failed": sum(not r["ok"] for r in results),
                                  "wall_s": time.monotonic() - t0, "workers": workers}
    _write_json(out / "RUN_SUMMARY.json", summary)
    return summary


def status(out: Path) -> dict:
    out = Path(out)
    plan = json.loads((out / "PLAN.json").read_text(encoding="utf-8"))
    done, failed, outcomes, wall, sim = [], [], {}, 0.0, 0.0
    for w in plan["worlds"]:
        d = out / "worlds" / w["world"]
        if _world_complete(d):
            rec = json.loads((d / "world.json").read_text(encoding="utf-8"))
            done.append(w["world"])
            wall += float(rec.get("wall_s", 0.0))
            sim += float(rec.get("sim_s", 0.0))
            for arm, r in rec["arms"].items():
                oc = (r.get("wall_timeline") or {}).get("outcome")
                if oc:
                    outcomes[oc] = outcomes.get(oc, 0) + 1
        elif (d / "FAILED.json").is_file():
            failed.append(w["world"])
    return {"schema": "c4_wallregen_status/1", "at": _now(), "population": plan["population"],
            "n_plan": len(plan["worlds"]), "n_complete": len(done), "n_failed": len(failed),
            "failed": failed, "wall_outcomes": outcomes,
            "cpu_wall_s_total": round(wall, 1), "sim_s_total": round(sim, 1),
            "sim_s_per_cpu_s": round(sim / wall, 2) if wall else None,
            "cpu_s_per_world_mean": round(wall / len(done), 1) if done else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default=DEFAULT_SEEDS)
    p.add_argument("--penetrations", default=",".join(str(x) for x in DEFAULT_PENETRATIONS))
    p.add_argument("--demands", default=(",".join(str(x) for x in DEMANDS_VPH)
                                         if DEMANDS_VPH else None),
                   help="comma-separated veh/h (default: DEMANDS_VPH at the top of this file)")
    p.add_argument("--t-choices", default=",".join(str(x) for x in DEFAULT_T_CHOICES))
    p.add_argument("--plan-seed", type=int, default=DEFAULT_PLAN_SEED)
    p.add_argument("--branch-s", type=int, default=DEFAULT_BRANCH_S)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--population", default="textbook")
    p.add_argument("--repeat-every", type=int, default=10,
                   help="add a native_repeat determinism arm to every N-th world (0 = never)")
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    r.add_argument("--only", default=None, help="comma-separated world names")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--cav-vtype", default=None, help="legacy population only (env cav_vtype)")
    s = sub.add_parser("status")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        out = checked(a.out)
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        arms = [x for x in a.arms.split(",") if x]
        bad = [x for x in arms if x not in ARMS]
        if bad:
            raise SystemExit(f"unknown arms {bad}")
        if not a.demands:
            raise SystemExit("demand levels undecided: pass --demands (or set DEMANDS_VPH)")
        plan = make_plan(seeds=parse_seeds(a.seeds),
                         penetrations=[float(x) for x in a.penetrations.split(",")],
                         demands=[int(x) for x in a.demands.split(",")],
                         t_choices=[int(x) for x in a.t_choices.split(",")],
                         plan_seed=a.plan_seed, branch_s=a.branch_s, arms=arms,
                         population=a.population, repeat_every=a.repeat_every)
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out, plan)
        print(json.dumps({"plan": str(out), "n_worlds": plan["n_worlds"],
                          "population": plan["population"]}))
        return 0
    if a.cmd == "run":
        summary = run_plan(Path(a.plan), Path(a.out), workers=a.workers,
                           only=a.only.split(",") if a.only else None, limit=a.limit,
                           cav_vtype=a.cav_vtype)
        print(json.dumps(summary, indent=1, default=_jsonable))
        return 0 if summary["n_failed"] == 0 else 1
    print(json.dumps(status(Path(a.out)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
