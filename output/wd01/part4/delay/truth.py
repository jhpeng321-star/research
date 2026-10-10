"""Truth-log adapter: partA_data root npz -> delay windows (part 4).

Read-only over RD_WT_ROOT/wd01/output/wd01/partA_data (the Part A v0
NATIVE dataset; MANIFEST sha256-verified) and the regenerated runs'
requests.rou.xml (scheduled departures per vehicle id, matched by id to
the logged insertion records).  Zero simulation.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import wdpaths  # noqa: E402

from . import core  # noqa: E402
from .geometry import ENTRANCE_MAINLINE, ENTRANCE_RAMP, SectionGeometry  # noqa: E402

PARTA_DATA_DEFAULT = str(Path(wdpaths.RD_WT_ROOT) / "wd01" / "output" /
                         "wd01" / "partA_data")
ROUTE_ENTRANCE = {"mainline_route": ENTRANCE_MAINLINE,
                  "ramp_route": ENTRANCE_RAMP}


def partA_root() -> Path:
    import os
    return Path(wdpaths.remap(os.environ.get("WD01_PARTA_DATA",
                                             PARTA_DATA_DEFAULT)))


@lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads((partA_root() / "MANIFEST.json").read_text("utf-8"))


@lru_cache(maxsize=1)
def run_map() -> dict:
    """tag -> row from part1's tag_run_map.json (authoritative copy)."""
    p = partA_root().parent / "part1" / "results" / "tag_run_map.json"
    return {r["tag"]: r for r in json.loads(p.read_text("utf-8"))["rows"]}


def sha256_file(path, block=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def roots_by_split() -> dict:
    man = manifest()
    out = {"train": [], "holdout": []}
    for name, r in sorted(man["roots"].items()):
        out[r["split"]].append(name)
    return out


def loading_of(root: str):
    """Demand loading (veh/h) parsed from the root name, or None."""
    import re
    m = re.search(r"m(\d+)$", root)
    return int(m.group(1)) if m else None


def root_run_dir(root: str) -> str:
    man = manifest()
    tag = man["roots"][root]["representative_tag"]
    return wdpaths.remap(run_map()[tag]["run_dir"])


@lru_cache(maxsize=32)
def load_schedule(run_dir: str) -> dict:
    """{vehicle id: (t_dep_s, entrance)} from run/scenario/requests.rou.xml.

    The route determines the entrance; the id prefix is the fallback.
    """
    path = Path(run_dir) / "scenario" / "requests.rou.xml"
    out = {}
    for veh in ET.parse(path).getroot().iter("vehicle"):
        route = veh.get("route")
        if route in ROUTE_ENTRANCE:
            ent = ROUTE_ENTRANCE[route]
        else:
            ent = (ENTRANCE_MAINLINE if veh.get("id").startswith("mainline")
                   else ENTRANCE_RAMP)
        out[veh.get("id")] = (float(veh.get("depart")), ent)
    return out


@lru_cache(maxsize=32)
def vtype_speed_factors(run_dir: str) -> dict:
    """{vType id: speedFactor} from the same rou.xml (oracle variant)."""
    path = Path(run_dir) / "scenario" / "requests.rou.xml"
    out = {}
    for vt in ET.parse(path).getroot().iter("vType"):
        out[vt.get("id")] = float(vt.get("speedFactor", "1.0"))
    return out


@dataclass
class RootData:
    root: str
    window_t0: float
    dt: float
    n_ticks: int
    vid_table: np.ndarray            # (n_vehicles,) object/str ids
    start: np.ndarray                # (n_vehicles,) row index per vid
    end: np.ndarray                  # exclusive
    tick: np.ndarray                 # row-sorted-by-(vid,tick) int16
    lane: np.ndarray                 # int8, -1 junction
    s: np.ndarray                    # float64 physical-lane s
    type_label: np.ndarray           # int8 per row (constant per vid)
    b_tick: np.ndarray               # birth tick per vid (aligned vid order)
    b_departed: np.ndarray           # 1 iff departed within the log window
    ar_tick: np.ndarray              # insertion ticks (departed births)
    ar_entrance: np.ndarray          # 0 mainline / 1 ramp
    ar_v: np.ndarray
    ar_type: np.ndarray
    ar_vid: np.ndarray               # vid index per arrival row


def load_root_data(root: str) -> RootData:
    """All window arrays of one root, rows sorted by (vid, tick)."""
    man = manifest()
    info = man["roots"][root]
    z = np.load(wdpaths.remap(info["file"]), allow_pickle=False)
    dt = float(man.get("dt_s", 0.5))
    t0w = float(man.get("window_s", [60.0, 600.0])[0])
    n_ticks = int(round((man["window_s"][1] - man["window_s"][0]) / dt)) + 1

    vid = z["vid"].astype(np.int64)
    tick = z["tick"].astype(np.int64)
    order = np.lexsort((tick, vid))
    vid, tick = vid[order], tick[order]
    n_vehicles = len(z["vid_table"])
    starts = np.flatnonzero(np.r_[True, vid[1:] != vid[:-1]])
    ends = np.r_[starts[1:], len(vid)]

    # per-vid birth tick / departed flag, rows sorted by vid index
    b_vid = z["b_vid"].astype(np.int64)
    assert np.array_equal(np.sort(b_vid), np.arange(n_vehicles)), \
        "one birth row per vehicle expected"
    bo = np.argsort(b_vid, kind="stable")
    b_tick = z["b_tick"].astype(np.int64)[bo]
    b_departed = z["b_departed"].astype(np.int8)[bo]
    # a vehicle never departed inside [60,600] and first seen at tick 0 was
    # inserted before the log window (asserted on all roots in validate)
    dep_vid = b_vid[bo][b_departed == 1]      # vid index per departed birth
    dep_tick = b_tick[b_departed == 1]

    # arrival rows (contract format): an arrival row IS a departed birth.
    # Both multisets have the same ticks (asserted), so sorting both by
    # tick aligns each arrival row with its vehicle id.
    dep_order = np.argsort(dep_tick, kind="stable")
    ar_sorted = np.argsort(z["ar_tick"].astype(np.int64), kind="stable")
    ar_tick = z["ar_tick"].astype(np.int64)[ar_sorted]
    assert np.array_equal(ar_tick, dep_tick[dep_order]), \
        "arrival rows must be exactly the departed births (by tick)"
    ar_vid = dep_vid[dep_order]              # vid index per arrival row
    ar_entrance = z["ar_entrance"].astype(np.int8)[ar_sorted]
    ar_v = z["ar_v"].astype(np.float64)[ar_sorted]
    ar_type = z["ar_type"].astype(np.int8)[ar_sorted]

    return RootData(
        root=root, window_t0=t0w, dt=dt, n_ticks=n_ticks,
        vid_table=z["vid_table"],
        start=starts, end=ends,
        tick=tick, lane=z["lane"].astype(np.int64)[order],
        s=z["s"].astype(np.float64)[order],
        type_label=z["type_label"].astype(np.int64)[order],
        b_tick=b_tick, b_departed=b_departed,
        ar_tick=ar_tick, ar_entrance=ar_entrance, ar_v=ar_v,
        ar_type=ar_type, ar_vid=ar_vid)


def insertion_map(rd: RootData, kH: int) -> dict:
    """{vehicle id: t_ins} for every vehicle born by tick kH.

    Departed births carry their exact insertion time; births never departed
    inside the log window were inserted before it (t_ins = -inf; they are
    all tick-0 records, asserted in validation).  The full map -- not just
    in-window insertions -- is what entry waiting needs: a vehicle already
    on the section at t0 must NOT be counted as queued at the entrance.
    """
    t = rd.window_t0 + rd.ar_tick.astype(np.float64) * rd.dt
    out = {str(rd.vid_table[vi]): float(ti)
           for vi, ti in zip(rd.ar_vid, t) if ti <= rd.window_t0 + kH * rd.dt
           + core.EPS}
    pre = np.flatnonzero(rd.b_departed == 0)
    for vi in pre:
        if rd.window_t0 + rd.b_tick[vi] * rd.dt <= rd.window_t0 + kH * rd.dt:
            out[str(rd.vid_table[vi])] = float("-inf")
    return out


def window_truth(rd: RootData, t0: float, horizon_s: float,
                 geom: SectionGeometry, schedule: dict | None = None,
                 speed_factors: dict | None = None,
                 strict: bool = True) -> core.WindowDelay:
    """Truth delay for [t0, t0+horizon] on one root (contract §4.1)."""
    dt = rd.dt
    k0 = int(round((t0 - rd.window_t0) / dt))
    kH = k0 + int(round(horizon_s / dt))
    assert abs(t0 - (rd.window_t0 + k0 * dt)) < 1e-6, "t0 off the grid"
    assert 0 <= k0 <= kH < rd.n_ticks, "window outside the stored data"
    if schedule is None:
        schedule = load_schedule(root_run_dir(rd.root))
    sched_w = {vid: sd for vid, sd in schedule.items()
               if sd[0] <= t0 + horizon_s + core.EPS}
    insertion = insertion_map(rd, kH)
    waits, wdiag = core.waiting_component(sched_w, insertion, t0, horizon_s)

    t1 = t0 + horizon_s
    veh, diags = [], dict(n_gapped=0, n_junction_records=0,
                          n_boundary_change_steps=0)
    for vi in range(len(rd.vid_table)):
        lo, hi = rd.start[vi], rd.end[vi]
        a = np.searchsorted(rd.tick[lo:hi], k0, "left")
        b = np.searchsorted(rd.tick[lo:hi], kH, "right")
        if b <= a:
            continue
        sel = slice(lo + a, lo + b)
        trace = core.VehicleTrace(
            vid=str(rd.vid_table[vi]),
            t=rd.window_t0 + rd.tick[sel] * rd.dt,
            lane=rd.lane[sel], s=rd.s[sel],
            vehicle_type=int(rd.type_label[sel][0]))
        v = core.vehicle_delay(trace, t0, horizon_s, geom, dt,
                               speed_factors=speed_factors, strict=strict)
        veh.append(v)
        diags["n_gapped"] += int(v.gapped)
        diags["n_junction_records"] += v.n_junction_records
        diags["n_boundary_change_steps"] += v.n_boundary_change_steps
    wd = core.WindowDelay(t0=t0, horizon_s=horizon_s, vehicles=veh,
                          waits=waits, diagnostics={**diags, **wdiag})
    return wd


def trace_ids(rd: RootData, k0: int, kH: int):
    """(ids present at k0, ids inserted in (k0, kH]) for label assembly."""
    present = [str(rd.vid_table[vi]) for vi in range(len(rd.vid_table))
               if rd.tick[rd.start[vi]] <= k0]
    ins = {vid: ti for vid, ti in insertion_map(rd, kH).items()
           if k0 * rd.dt < ti - rd.window_t0 <= kH * rd.dt}
    return present, sorted(ins.keys())
