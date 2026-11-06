"""Deterministic synthetic ``c4_sumo_episode/1`` episodes in the WM model frame (WM-A, no SUMO).

The generator only needs the frame geometry, so the same code writes merge-map episodes (the
golden fixture) and wz3to2 lane-drop episodes (map-port tests).  Traffic is kinematic, not a
simulator: per-lane spawn streams at the lane starts, per-vehicle speeds with a small sinusoid,
closing-lane vehicles that change into the target lane at a drawn position inside the road's
closing-lane change zone, a few lane-1 -> lane-0 overtakes, optional lane-1 -> closing-lane
entries (lane-drop map only), exits past the through-lane end, CAVs with direct commands after
the warmup.  Every number comes from ``numpy.random.default_rng(seed)``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONTROLLERS = ("native", "damping", "pi-external-u", "direct")
NATIVE, DIRECT = 0, 3


@dataclass(frozen=True)
class Geometry:
    """Frame geometry needed by the generator (WM lanes 0/1 through, 2 closing)."""

    entry_s_m: tuple            # per WM lane insertion s
    road_end_m: float           # through-lane exit
    closing_end_m: float        # closing-lane end (75.44 on both maps)
    change_zone_s_m: tuple      # where closing-lane cars may change into the target lane
    into_closing: bool = False  # lane 1 -> closing lane changes happen
    headways_s: tuple = (9.0, 8.0, 5.0)   # per-lane spawn headway (keeps K slots unsaturated)


MERGE = Geometry(entry_s_m=(-2645.67, -2644.22, -378.66), road_end_m=553.33,
                 closing_end_m=75.44, change_zone_s_m=(0.0, 75.44))
WZ3TO2 = Geometry(entry_s_m=(-2620.56, -2620.56, -2620.56), road_end_m=1379.44,
                  closing_end_m=75.44, change_zone_s_m=(-2620.56, 75.44), into_closing=True,
                 headways_s=(9.0, 8.0, 10.0))


def simulate(geo: Geometry, *, seed: int, k: int = 32, total_s: float = 160.0,
             warmup_s: float = 90.0, headways_s=None, speeds=(27.0, 24.0, 18.0),
             cav_every: int = 5, preload: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    headways_s = tuple(geo.headways_s if headways_s is None else headways_s)
    dt = 0.5
    n_t = int(round(total_s / dt)) + 1
    times = np.arange(n_t) * dt
    vehicles = []                     # dicts: id, lane, s, v0, phase, born, merge_s, over_s, cav

    def new(lane, s, born):
        vid = len(vehicles)
        merge_s = None
        if lane == 2:
            z0, z1 = geo.change_zone_s_m
            lo = max(z0 + 5.0, s + 20.0) if z0 > -1e6 else s + 20.0
            merge_s = float(rng.uniform(lo, z1 - 5.0)) if lo < z1 - 5.0 else z1 - 5.0
        over_s = float(rng.uniform(-2000.0, 200.0)) if lane == 1 and rng.random() < 0.25 else None
        into_s = (float(rng.uniform(geo.entry_s_m[2] + 100.0, -300.0))
                  if geo.into_closing and lane == 1 and over_s is None and rng.random() < 0.15
                  else None)
        vehicles.append(dict(id=vid, lane=lane, s=float(s), v0=float(speeds[lane] + rng.normal(0, 1.0)),
                             phase=float(rng.uniform(0, 2 * np.pi)), born=born, merge_s=merge_s,
                             over_s=over_s, into_s=into_s, cav=(vid % cav_every == 2),
                             alive=True, slot=-1))
        veh = vehicles[-1]
        veh["v"], veh["a"] = max(0.0, veh["v0"] + 0.6 * np.sin(0.15 * born + veh["phase"])), 0.0
        return veh

    if preload:
        for lane in range(3):
            end = geo.road_end_m if lane < 2 else geo.closing_end_m - 10.0
            gap = headways_s[lane] * speeds[lane]
            s = geo.entry_s_m[lane] + 10.0
            while s < end - 5.0:
                new(lane, s, 0.0)
                s += gap * float(rng.uniform(0.8, 1.3))
    next_spawn = [float(rng.uniform(0.5, h)) for h in headways_s]
    slots = [-1] * k
    cols = {name: np.zeros((n_t, k), dt_) for name, dt_ in
            (("s_m", np.float32), ("v_mps", np.float32), ("a_mps2", np.float32),
             ("length_true_m", np.float32))}
    lane_a = np.full((n_t, k), -1, np.int8)
    present = np.zeros((n_t, k), bool)
    is_cav = np.zeros((n_t, k), bool)
    vid_a = np.full((n_t, k), -1, np.int32)
    first_lane = {}
    for t, now in enumerate(times):
        if t:
            for lane in range(3):
                while next_spawn[lane] <= now:
                    new(lane, geo.entry_s_m[lane], now)
                    next_spawn[lane] += headways_s[lane] * float(rng.uniform(0.7, 1.4))
        for veh in vehicles:
            if not veh["alive"]:
                continue
            if t and veh["born"] < now:
                v_prev = veh["v"]
                v_new = max(0.0, veh["v0"] + 0.6 * np.sin(0.15 * now + veh["phase"]))
                veh["a"] = (v_new - v_prev) / dt
                veh["s"] += 0.5 * (v_prev + v_new) * dt
                veh["v"] = v_new
                if veh["lane"] == 2 and veh["merge_s"] is not None and veh["s"] >= veh["merge_s"]:
                    veh["lane"], veh["v0"] = 1, max(veh["v0"], speeds[1] - 1.0)
                elif veh["lane"] == 1 and veh["over_s"] is not None and veh["s"] >= veh["over_s"]:
                    veh["lane"], veh["over_s"] = 0, None
                elif (veh["lane"] == 1 and veh["into_s"] is not None and veh["s"] >= veh["into_s"]):
                    veh["lane"], veh["into_s"] = 2, None
                    z1 = geo.change_zone_s_m[1]
                    veh["merge_s"] = float(rng.uniform(veh["s"] + 50.0, z1 - 5.0))
                if veh["lane"] == 2 and veh["s"] > geo.closing_end_m - 0.5:
                    veh["lane"] = 1                        # never past the closing-lane end
            end = geo.road_end_m
            if veh["s"] > end:
                veh["alive"] = False
                if veh["slot"] >= 0:
                    slots[veh["slot"]] = -1
                continue
            if veh["slot"] < 0:
                free = [i for i, x in enumerate(slots) if x < 0]
                if not free:
                    continue                                # pool full: not observed this tick
                veh["slot"] = free[0]
                slots[free[0]] = veh["id"]
                first_lane.setdefault(veh["id"], veh["lane"])
            j = veh["slot"]
            present[t, j], vid_a[t, j], lane_a[t, j] = True, veh["id"], veh["lane"]
            cols["s_m"][t, j], cols["v_mps"][t, j] = veh["s"], veh["v"]
            cols["a_mps2"][t, j], cols["length_true_m"][t, j] = veh["a"], 5.0
            is_cav[t, j] = veh["cav"]
    n_c = int(round(total_s))
    ctrl_t = np.arange(n_c, dtype=np.float64)
    controller = np.zeros((n_c, k), np.int8)
    acc = np.zeros((n_c, k), np.float32)
    tgt = np.zeros((n_c, k), np.float32)
    for c in range(n_c):
        if ctrl_t[c] < warmup_s:
            continue
        cav = is_cav[2 * c]
        controller[c, cav] = DIRECT
        acc[c, cav] = (0.4 * np.sin(0.3 * ctrl_t[c] + np.arange(k)) )[cav].astype(np.float32)
    used = sorted({int(v) for v in vid_a[present]})
    return dict(time_s=times, **cols, lane=lane_a, lane_carry=lane_a.copy(), present=present,
                is_cav=is_cav, junction=np.zeros_like(present), vid_index=vid_a,
                vehicle_ids=np.asarray([f"veh{i}" for i in range(len(vehicles))]),
                cav_vehicle_ids=np.asarray([f"veh{v['id']}" for v in vehicles if v["cav"]]),
                control_time_s=ctrl_t, controller=controller, issued_acceleration_mps2=acc,
                issued_target_speed_mps=tgt,
                requested_acceleration_mps2=np.where(controller == DIRECT, acc, np.nan).astype(np.float32),
                lane_command_target=np.full((n_c, k), -1, np.int8),
                lane_request_target=np.full((n_c, k), -1, np.int8),
                controller_vocab=np.asarray(CONTROLLERS), _used=np.asarray(used))


def write_episode(root: Path, geo: Geometry, *, seed: int, demand: int = 2700,
                  penetration: float = 0.2, policy: str = "native", warmup_s: float = 90.0,
                  total_s: float = 160.0, k: int = 48, extra_meta: dict | None = None) -> Path:
    arrays = simulate(geo, seed=seed, k=k, total_s=total_s, warmup_s=warmup_s)
    arrays.pop("_used")
    ep = Path(root) / f"s{seed}_m{demand}_p{int(round(penetration * 1000)):04d}"
    ep.mkdir(parents=True)
    meta = dict(schema="c4_sumo_episode/1", seed=seed, demand_vph=demand, penetration=penetration,
                warmup_s=warmup_s, horizon_s=total_s - warmup_s, controller="direct", k_slots=k,
                **(extra_meta or {}))
    np.savez_compressed(ep / "episode.npz", meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
                        **arrays)
    (ep / "manifest.json").write_text(json.dumps({**meta, "status": "complete", "policy": policy},
                                                 sort_keys=True))
    (ep / "summary.json").write_text("{}")
    return ep
