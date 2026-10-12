"""Subscription-based per-step SUMO reads and the WM-41 physics-row recorder schema.

One TraCI ``getAllSubscriptionResults`` call per 0.5 s step replaces ~10 getters per
vehicle.  The physics row is the frozen ``wm41_logging/1`` contract (port of
``traffic_shepherd/physics_log.py`` on branch ``wms/swgen``, itself a port of
``output/campaign_pilot_v1``) so the WM-N1 featurizer / hetpack pack builders read
these runs unchanged: ``time_s``, ``vehicles`` (road_id, lane_id, lane_position_m,
speed_mps, acceleration_mps2, type_id, speed_mode, lane_changed), ``departed_ids``,
``arrived_ids``, ``colliding_ids``, ``cav_ids``, ``speed_modes``, ``cav_actions``,
``boundary_crossings`` (+ ``starting_teleports``/``ending_teleports`` as in the
campaign branch arms).
"""
from __future__ import annotations

import json

LOG_SCHEMA = "wm41_logging/1"
LEADER_RANGE_M = 200.0          # metrics.ttc_exposure outcome-only leader query
BOUNDARY_KEYS = ("mainline_entry", "mainline_exit", "ramp_entry")
CAV_ACTION_FIELDS = ("owner", "action", "commanded_speed_mps",
                     "controller_state_digest", "wall_id", "wall_membership",
                     "wall_phase")


def _tc():
    import traci.constants as tc
    return tc


def vehicle_vars(cav: bool = False):
    """Dynamic per-step variables (CAVs additionally need the allowed speed)."""
    tc = _tc()
    base = (tc.VAR_LANE_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_ACCELERATION,
            tc.VAR_SPEEDSETMODE, tc.VAR_LEADER)
    return base + ((tc.VAR_ALLOWED_SPEED,) if cav else ())


def split_lane(lane_id: str) -> tuple[str, int]:
    """SUMO lane id -> (edge/road id, lane index); internal ':J_0_1' -> (':J_0', 1)."""
    edge, _, idx = str(lane_id).rpartition("_")
    return edge, int(idx)


class VehicleReader:
    """Per-step live vehicle table from TraCI variable subscriptions.

    ``read()`` -> ``{vid: {road_id, lane_id, lane_position_m, speed_mps,
    acceleration_mps2, type_id, speed_mode, length_m, min_gap_m, allowed_speed_mps,
    lane_index, leader}}`` with ``leader`` = ``(leader_id, gap_without_min_gap)`` or
    ``None`` (same query as ``api.vehicle.getLeader(v, 200)``).  ``road_id`` and
    ``lane_index`` are parsed from the lane id; type/length/minGap are read once per
    vehicle (constant per vType); ``allowed_speed_mps`` is subscribed for CAVs only
    (``nan`` for HDVs, which are never sensed or commanded).
    """

    def __init__(self, api, *, cav_ids=(), use_subscriptions: bool = True):
        self.api = api
        self.cav_ids = frozenset(cav_ids)
        self.use_subscriptions = bool(use_subscriptions)
        self._subscribed: set[str] = set()
        self._static: dict[str, tuple] = {}

    def _static_of(self, vid: str) -> tuple:
        row = self._static.get(vid)
        if row is None:
            v = self.api.vehicle
            row = self._static[vid] = (str(v.getTypeID(vid)), float(v.getLength(vid)),
                                       float(v.getMinGap(vid)))
        return row

    def _direct(self, vid: str) -> dict:
        v = self.api.vehicle
        leader = v.getLeader(vid, LEADER_RANGE_M)
        lane_id = str(v.getLaneID(vid))
        type_id, length, min_gap = self._static_of(vid)
        return dict(road_id=split_lane(lane_id)[0], lane_id=lane_id,
                    lane_position_m=float(v.getLanePosition(vid)),
                    speed_mps=float(v.getSpeed(vid)),
                    acceleration_mps2=float(v.getAcceleration(vid)),
                    type_id=type_id, speed_mode=int(v.getSpeedMode(vid)),
                    length_m=length, min_gap_m=min_gap,
                    allowed_speed_mps=(float(v.getAllowedSpeed(vid)) if vid in self.cav_ids
                                       else float("nan")),
                    lane_index=split_lane(lane_id)[1],
                    leader=(None if not leader or not leader[0]
                            else (str(leader[0]), float(leader[1]))))

    def read(self, live_ids=None) -> dict[str, dict]:
        live = sorted(str(v) for v in (self.api.vehicle.getIDList() if live_ids is None
                                        else live_ids))
        if not self.use_subscriptions:
            return {vid: self._direct(vid) for vid in live}
        tc = _tc()
        for vid in live:
            if vid not in self._subscribed:
                self.api.vehicle.subscribe(vid, vehicle_vars(vid in self.cav_ids),
                                           parameters={tc.VAR_LEADER: ("d", LEADER_RANGE_M)})
                self._subscribed.add(vid)
        live_set = set(live)
        self._subscribed &= live_set
        self._static = {v: r for v, r in self._static.items() if v in live_set}
        results = self.api.vehicle.getAllSubscriptionResults()
        table = {}
        for vid in live:
            r = results.get(vid)
            cav = vid in self.cav_ids
            if not r or any(name not in r for name in vehicle_vars(cav)):
                table[vid] = self._direct(vid)
                continue
            leader = r[tc.VAR_LEADER]
            lane_id = str(r[tc.VAR_LANE_ID])
            road_id, lane_index = split_lane(lane_id)
            type_id, length, min_gap = self._static_of(vid)
            table[vid] = dict(
                road_id=road_id, lane_id=lane_id,
                lane_position_m=float(r[tc.VAR_LANEPOSITION]),
                speed_mps=float(r[tc.VAR_SPEED]),
                acceleration_mps2=float(r[tc.VAR_ACCELERATION]),
                type_id=type_id, speed_mode=int(r[tc.VAR_SPEEDSETMODE]),
                length_m=length, min_gap_m=min_gap,
                allowed_speed_mps=(float(r[tc.VAR_ALLOWED_SPEED]) if cav else float("nan")),
                lane_index=lane_index,
                leader=(None if not leader or not leader[0]
                        else (str(leader[0]), float(leader[1]))))
        return table


def boundary_edges(config) -> dict:
    edges = getattr(config, "edge_ids", None) or {}
    return {"mainline_entry": str(edges.get("pre_merge", "1191554636")),
            "mainline_exit": str(edges.get("downstream", "1487606178")),
            "ramp_entry": str(edges.get("ramp", "11553048#1"))}


def crossings_between(prev_vehicles: dict, vehicles: dict, edges: dict) -> dict:
    counts = {key: 0 for key in BOUNDARY_KEYS}
    for vid, veh in vehicles.items():
        road = veh.get("road_id")
        previous = (prev_vehicles.get(vid) or {}).get("road_id")
        for key in BOUNDARY_KEYS:
            if road == edges[key] and previous != road:
                counts[key] += 1
    return counts


def build_physics_row(now_s: float, table: dict, *, previous_row, cav_ids, cav_actions,
                      edges, departed, arrived, colliding, starting_teleports=(),
                      ending_teleports=()) -> dict:
    """One ``wm41_logging/1`` row from a :class:`VehicleReader` table."""
    prev = (previous_row or {}).get("vehicles", {})
    vehicles = {}
    for vid in sorted(table):
        veh = table[vid]
        before = prev.get(vid)
        vehicles[vid] = {
            "road_id": veh["road_id"], "lane_id": veh["lane_id"],
            "lane_position_m": veh["lane_position_m"], "speed_mps": veh["speed_mps"],
            "acceleration_mps2": veh["acceleration_mps2"], "type_id": veh["type_id"],
            "speed_mode": veh["speed_mode"],
            "lane_changed": before is not None and before.get("lane_id") != veh["lane_id"],
        }
    cav_ids = set(cav_ids)
    return {
        "time_s": round(float(now_s), 6),
        "live_ids": sorted(vehicles),
        "departed_ids": sorted(str(v) for v in departed),
        "arrived_ids": sorted(str(v) for v in arrived),
        "colliding_ids": sorted(str(v) for v in colliding),
        "cav_ids": sorted(cav_ids),
        "speed_modes": {v: vehicles[v]["speed_mode"] for v in vehicles if v in cav_ids},
        "cav_actions": {str(v): dict(a) for v, a in cav_actions.items()},
        "boundary_crossings": crossings_between(prev, vehicles, edges),
        "vehicles": vehicles,
        "starting_teleports": sorted(str(v) for v in starting_teleports),
        "ending_teleports": sorted(str(v) for v in ending_teleports),
    }


def action_entry(owner: str, action: str, commanded_speed_mps) -> dict:
    return {"owner": str(owner), "action": str(action),
            "commanded_speed_mps": (None if commanded_speed_mps is None
                                    else float(commanded_speed_mps)),
            "controller_state_digest": None, "wall_id": None, "wall_membership": None,
            "wall_phase": None}


def validate_physics_row(row: dict) -> dict:
    missing = [f for f in ("time_s", "cav_ids", "speed_modes", "cav_actions",
                           "boundary_crossings", "vehicles") if f not in row]
    if missing:
        raise ValueError(f"physics row missing required fields: {missing}")
    for cav, action in row["cav_actions"].items():
        if cav not in row["cav_ids"]:
            raise ValueError(f"cav_actions entry {cav!r} not in cav_ids")
        if [f for f in CAV_ACTION_FIELDS if f not in action]:
            raise ValueError(f"cav_actions[{cav!r}] incomplete")
    return row


def dumps_row(row: dict) -> str:
    return json.dumps(row, allow_nan=False) + "\n"
