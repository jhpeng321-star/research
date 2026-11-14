"""Synthetic-scenario helpers for the inc1_gen tests (zero SUMO, seeded).

One synthetic truth world (plain python floats) is rendered twice by the
adapter tests: traci-like dicts (SUMO side) and tensor-like arrays (WM side).
"""
from __future__ import annotations

import math

import numpy as np

from inc1_gen.states import ObstacleRecord, VehState

OBSTACLE = ObstacleRecord(obstacle_id="incident.obstacle", lane_phys=1,
                          s_down=40.0, v=0.0, a=0.0, length_m=5.0)
T0_S = 300.0
DT_S = 1.0
LEN_M = 5.0


def base_world() -> list[dict]:
    """A synthetic t0 world exercising every membership boundary.

    d = s_obs - s_down; the obstacle sits at lane 1, s_down 40.
    """
    def veh(vid, *, cav=True, lane, d, v=20.0, a=0.0, gap=None, lv=None,
            native=1.0, present=True):
        return dict(vehicle_id=vid, is_cav=cav, lane_phys=lane,
                    s_down=OBSTACLE.s_down - float(d), v=float(v), a=float(a),
                    leader_gap=(None if gap is None else float(gap)),
                    leader_v=(None if lv is None else float(lv)),
                    native_accel=float(native), present=present)

    return [
        veh("cav.a", lane=0, d=100.0),            # R5 edge in; R6 opener
        veh("cav.b", lane=1, d=30.0),             # R6 approacher edge in
        veh("cav.c", lane=0, d=1499.999),         # R5 member, far
        veh("cav.d", lane=1, d=1500.0),           # R5 edge out
        veh("cav.e", lane=0, d=600.0),            # R5 member; opener edge out
        veh("cav.f", lane=1, d=29.999),           # approacher edge out
        veh("cav.g", lane=0, d=0.0),              # opener edge in
        veh("cav.h", lane=2, d=500.0),            # ramp lane: never a member
        veh("hdv.1", cav=False, lane=1, d=200.0),  # HDV: never a member
        veh("cav.gone", lane=0, d=400.0, present=False),  # already left
    ]


def as_states(world) -> list[VehState]:
    return [VehState(**row) for row in world]


def to_sumo_rows(world) -> dict[str, dict]:
    """The traci-like rendering: dict of per-vehicle state dicts."""
    rows = {}
    for row in world:
        entry = {
            "speed": row["v"],
            "lane_position": row["s_down"],
            "lane_index": row["lane_phys"],
            "acceleration": row["a"],
            "type": "cav_passenger" if row["is_cav"] else "passenger",
            "present": row["present"],
            "native_accel": row["native_accel"],
        }
        if row["leader_gap"] is not None:
            entry["leader"] = (row["leader_gap"], row["leader_v"])
        rows[row["vehicle_id"]] = entry
    return rows


def to_wm_frame(world) -> tuple[list[str], dict]:
    """The tensor-like rendering: (ids, array kwargs) for the WM shim."""
    ids = [row["vehicle_id"] for row in world]

    def col(key, none_value=math.nan):
        return np.asarray(
            [none_value if row[key] is None else row[key] for row in world],
            dtype=float)

    return ids, {
        "s_down": col("s_down"), "v": col("v"), "a": col("a"),
        "lane_phys": col("lane_phys"), "is_cav": col("is_cav"),
        "present": col("present"), "native_accel": col("native_accel"),
        "leader_gap": col("leader_gap"), "leader_v": col("leader_v"),
    }


def with_leaders(world, obstacle=OBSTACLE) -> list[dict]:
    """Recompute same-lane leader gaps/velocities from vehicle positions."""
    blockers = [dict(row) for row in world if row["present"]]
    if obstacle is not None:
        blockers.append(dict(vehicle_id=obstacle.obstacle_id, is_cav=False,
                             lane_phys=obstacle.lane_phys,
                             s_down=obstacle.s_down, v=obstacle.v, a=obstacle.a,
                             leader_gap=None, leader_v=None,
                             native_accel=None, present=True))
    out = []
    for row in world:
        row = dict(row)
        ahead = [b for b in blockers
                 if b["lane_phys"] == row["lane_phys"]
                 and b["vehicle_id"] != row["vehicle_id"]
                 and b["s_down"] > row["s_down"]]
        if ahead:
            lead = min(ahead, key=lambda b: b["s_down"])
            row["leader_gap"] = lead["s_down"] - row["s_down"] - LEN_M
            row["leader_gap"] = max(0.0, row["leader_gap"])
            row["leader_v"] = lead["v"]
        else:
            row["leader_gap"] = None
            row["leader_v"] = None
        out.append(row)
    return out


def advance(world, commands: dict[str, float], *, obstacle=OBSTACLE,
            dt_s: float = DT_S) -> list[dict]:
    """One closed-loop tick: commanded accel wins, else native IDM accel."""
    out = []
    for row in world:
        row = dict(row)
        a_eff = (commands[row["vehicle_id"]]
                 if row["vehicle_id"] in commands
                 else (0.0 if row["native_accel"] is None
                       else row["native_accel"]))
        row["v"] = max(0.0, row["v"] + float(a_eff) * dt_s)
        row["s_down"] = row["s_down"] + row["v"] * dt_s
        row["a"] = float(a_eff)
        out.append(row)
    return out


def random_world(rng: np.random.Generator, n: int = 12) -> list[dict]:
    """A seeded random synthetic world for the limits property test."""
    world = []
    for i in range(n):
        cav = bool(rng.random() < 0.8)
        has_leader = rng.random() < 0.7
        world.append(dict(
            vehicle_id=f"v{i:02d}",
            is_cav=cav,
            lane_phys=int(rng.integers(0, 3)),
            s_down=OBSTACLE.s_down - float(rng.uniform(-400.0, 1700.0)),
            v=float(rng.uniform(0.0, 30.0)),
            a=float(rng.uniform(-2.0, 1.5)),
            leader_gap=(None if not has_leader
                        else float(rng.uniform(0.5, 250.0))),
            leader_v=(None if not has_leader
                      else float(rng.uniform(0.0, 30.0))),
            native_accel=float(rng.uniform(-4.0, 2.5)),
            present=bool(rng.random() < 0.95),
        ))
    return world


def drop_world_row(world, vid: str) -> list[dict]:
    return [dict(row, present=False) if row["vehicle_id"] == vid else dict(row)
            for row in world]
