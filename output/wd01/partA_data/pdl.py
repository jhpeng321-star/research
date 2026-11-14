"""Part A v0 NATIVE dataset library (WD-01 part 2C).

Builds the 0.5 s physics-step training dataset on the WM-30 physical lanes
from the regenerated run logs.  ZERO SIMULATION: the regenerated
physics_steps.jsonl / telemetry.jsonl logs are read as data; the network is
read as XML only (via hybrid_state_v1's topology module, imported read-only,
plus a local parse of the same net XML for the junction internal-lane map).

Physical lanes (hybrid_state_v1/topology.py, imported not copied):
  main_left  = 1191554636_1 -> 1487606179_2 -> 1487606178_1   (id 0)
  main_right = 1191554636_0 -> 1487606179_1 -> 1487606178_0   (id 1)
  accel      = 11553048#1_0 -> 1487606179_0                   (id 2)
s = physical-lane coordinate from the lane's upstream end; d = OFFSET - s
is merge-relative and comparable ACROSS lanes (that is what the neighbour
builder uses for leader/follower search).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))         # output/wd01 (wdpaths)
import wdpaths  # noqa: E402

# Untracked data/weights/results live in a populated copy of the output
# tree (e.g. RD_WT_ROOT/wd01/output/wd01).  Set WD01_DATA_ROOT to that
# tree to read them; UNSET (default) keeps the historical in-repo layout
# (partA_data/data, ../part1, ...), byte-identical to pre-3k behaviour.
DATA_ROOT_ENV = "WD01_DATA_ROOT"
_DATA_ROOT = Path(wdpaths.remap(os.environ[DATA_ROOT_ENV])) \
    if os.environ.get(DATA_ROOT_ENV, "").strip() else None
PART1 = (_DATA_ROOT / "part1") if _DATA_ROOT else (HERE.parent / "part1")
DATA = (_DATA_ROOT / "partA_data" / "data") if _DATA_ROOT else (HERE / "data")
RESULTS = HERE / "results"
SRC = Path(wdpaths.remap(os.environ.get(
    "WD01_SRC", wdpaths.RESEARCH_DENSE_ROOT)))
SURV_CKPT = SRC / "output" / "worldmodel_v3_vehicle" / "vehicle_dynamics.pt"
SURV_CKPT_SHA256 = ("ded83907b8d218574d3f5359c332408f17167da055574afbe"
                    "29eaecb59d4a8f5")
HYBRID = SRC / "output" / "hybrid_state_v1"

WINDOW_T0 = 60.0
WINDOW_T1 = 600.0
DT = 0.5
N_TICKS = int(round((WINDOW_T1 - WINDOW_T0) / DT)) + 1      # 1081
GAP_MAX = 250.0

LANE_IDS = {"main_left": 0, "main_right": 1, "accel": 2}
# vType labels are SUPERVISION ONLY (ideal sensing does not reveal vType);
# they must never appear in input-feature builders (tested).
TYPE_IDS = {"aggressive_car": 0, "normal_car": 1, "cautious_car": 2}
ARM_SUFFIXES = ("-pi-external-u", "-native", "-retain", "-wall")


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tag_stem(tag):
    """cv19_4107-native -> cv19_4107 (separator may be '-' or '_')."""
    import re
    return re.sub(r"[_-](pi-external-u|native|retain|wall)$", "", tag)


def group_name(stems, h):
    """Root name for a dedup group: the single stem if unambiguous, else
    the longest common suffix (the (seed, loading) part, e.g. 7012m2700),
    else a hash fallback.  Cross-fold groups share identical traffic, so
    the common suffix IS the traffic identity."""
    stems = sorted(set(stems))
    if len(stems) == 1:
        return stems[0]
    suf = stems[0]
    for st in stems[1:]:
        while suf and not st.endswith(suf):
            suf = suf[1:]
        if not suf:
            break
    return f"seed{suf}" if suf else f"root_{h[:12]}"


# ------------------------------------------------------------- topology ----

_TOPO = None


def _sys_path_add(p):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)


def topology():
    """(PhysicalTopology, net_path, internal_lane_map) -- read-only.

    internal_lane_map: {internal lane id: (from_lane_id, to_lane_id)}
    parsed from the same net XML (the connections' via attribute).  A
    junction internal lane occupies ZERO s in the physical coordinate: the
    traversal boundary is segments[to].s0 == segments[from].s1.
    """
    global _TOPO
    if _TOPO is not None:
        return _TOPO
    _sys_path_add(HYBRID)
    import repo                                   # read-only source module
    # repo.py (reference tree) may add D:-prefixed entries to sys.path and
    # expose D:-prefixed constants; rewrite them to the live roots here.
    sys.path[:] = [wdpaths.remap(p) if isinstance(p, str) else p
                   for p in sys.path]
    import topology as T
    net_path = str(wdpaths.remap(repo.NETWORK))
    topo = T.PhysicalTopology.from_net_xml(net_path)

    internal = {}
    opener = gzip.open if net_path.endswith(".gz") else open
    with opener(net_path, "rb") as fh:
        import xml.etree.ElementTree as ET
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag == "connection":
                via = el.get("via")
                if via:
                    internal[via] = (f"{el.get('from')}_{el.get('fromLane')}",
                                     f"{el.get('to')}_{el.get('toLane')}")
            elif el.tag not in ("connection", "edge", "junction", "tlLogic"):
                el.clear()
    _TOPO = (topo, net_path, internal)
    return _TOPO


def junction_boundary_s(internal_lane_id):
    """Physical s of a junction traversal, or None if unresolvable."""
    topo, _, internal = topology()
    ft = internal.get(str(internal_lane_id))
    if ft is None:
        return None
    found = topo._segments_by_lane.get(ft[1])
    if found is None:
        return None
    return found[1].s0


def map_step(lane_id, pos, prev_s=None, prev_v=None):
    """One physics row -> (lane, s, in_junction).

    Physical lanes: lane 0/1/2 and exact s from the topology.  Junction
    internal lanes (':'-prefixed): lane=-1, in_junction=1, s = the shared
    boundary s of the traversal (internal lanes span zero physical s).
    Unresolvable/undocumented lanes: lane=-1, in_junction=1, s
    dead-reckoned from (prev_s, prev_v) -- expected ~0 rows.
    """
    topo, _, _ = topology()
    name, s = topo.lane_of(lane_id, pos)
    if name is not None:
        return LANE_IDS[name], float(s), 0
    b = junction_boundary_s(lane_id) if str(lane_id).startswith(":") else None
    if b is not None:
        return -1, float(b), 1
    s = (prev_s + prev_v * DT) if prev_s is not None else float("nan")
    return -1, float(s), 1


# ------------------------------------------------------- run window scan ----

def _rows_in_window(run_dir):
    """Yield (t, row_dict) for physics rows with WINDOW_T0 <= t <= WINDOW_T1."""
    with open(Path(run_dir) / "physics_steps.jsonl", encoding="utf-8") as fh:
        for line in fh:
            i = line.find('"time_s": ')
            if i < 0:
                continue
            j = i + len('"time_s": ')
            k = line.find(",", j)
            try:
                t = float(line[j:k])
            except ValueError:
                t = None
            if t is None or not (WINDOW_T0 - 1e-9 <= t <= WINDOW_T1 + 1e-9):
                continue
            yield t, json.loads(line)


def window_hash(run_dir):
    """Deterministic content hash of the [60, 600] window.

    Canonical bytes: per tick (int(t*2)), per vehicle sorted by id:
    (lane_id, round(pos,6), round(speed,6), type_id).  Byte-identical arms
    hash equal; the hash is stable across processes.
    """
    h = hashlib.sha256()
    for t, row in _rows_in_window(run_dir):
        h.update(struct.pack("<i", int(round(t * 2))))
        h.update(struct.pack("<i", len(row["vehicles"])))
        for vid in sorted(row["vehicles"]):
            v = row["vehicles"][vid]
            enc = (vid.encode() + b"\x00" + v["lane_id"].encode() + b"\x00"
                   + repr(round(float(v["lane_position_m"]), 6)).encode()
                   + b"\x00"
                   + repr(round(float(v["speed_mps"]), 6)).encode() + b"\x00"
                   + str(v["type_id"]).encode())
            h.update(struct.pack("<i", len(enc)))
            h.update(enc)
    return h.hexdigest()


def cav_ids(run_dir):
    """Distinct CAV ids from telemetry cav_report frames (the controller's
    own fleet reporting; see DATA_CARD for the coverage caveat)."""
    out = set()
    with open(Path(run_dir) / "telemetry.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if '"cav_report"' not in line:
                continue
            r = json.loads(line)
            out.add(r["frame"]["vehicle_id"])
    return out


def extract_run(tag, run_dir):
    """Extract the [60, 600] window of one run into compact arrays + stats.

    Returns dict(npz_arrays=..., stats=...).  Zero simulation.
    """
    topo, net_path, internal = topology()
    per_step = [None] * N_TICKS
    missing = []
    for t, row in _rows_in_window(run_dir):
        k = int(round((t - WINDOW_T0) / DT))
        if not (0 <= k < N_TICKS) or abs(WINDOW_T0 + k * DT - t) > 1e-6:
            missing.append(t)
            continue
        per_step[k] = row
    for k, row in enumerate(per_step):
        if row is None:
            missing.append(WINDOW_T0 + k * DT)

    vid_idx = {}
    vids = []
    cols = dict(vid=[], tick=[], lane=[], s=[], v=[], a=[],
                cav=[], inj=[], typ=[])

    def idx_of(vid):
        i = vid_idx.get(vid)
        if i is None:
            i = vid_idx[vid] = len(vids)
            vids.append(vid)
        return i

    for k, row in enumerate(per_step):
        if row is None:
            continue
        for vid in sorted(row["vehicles"]):
            v = row["vehicles"][vid]
            lane, s, inj = map_step(v["lane_id"], v["lane_position_m"])
            cols["vid"].append(idx_of(vid))
            cols["tick"].append(k)
            cols["lane"].append(lane)
            cols["s"].append(s)
            cols["v"].append(float(v["speed_mps"]))
            cols["a"].append(float(v["acceleration_mps2"]))
            cols["cav"].append(0)                    # filled post-hoc
            cols["inj"].append(inj)
            cols["typ"].append(TYPE_IDS.get(v["type_id"], -1))

    arr = {k: np.asarray(v) for k, v in cols.items()}
    n_steps = sum(1 for r in per_step if r is not None)

    # ---- per-vehicle temporal structure: births / exits / lane changes
    by_vid = {}
    for i in range(len(arr["vid"])):
        by_vid.setdefault(int(arr["vid"][i]), []).append(i)

    births, exits, changes, lost, gaps = [], [], [], [], []
    departed = {v for r in per_step if r for v in r["departed_ids"]}
    arrived = {v for r in per_step if r for v in r["arrived_ids"]}
    arrived_t = {}
    for r in per_step:
        if not r:
            continue
        for v in r["arrived_ids"]:
            arrived_t.setdefault(v, r["time_s"])
    for vi, idxs in by_vid.items():
        idxs.sort(key=lambda i: int(arr["tick"][i]))
        ticks = [int(arr["tick"][i]) for i in idxs]
        if ticks != list(range(ticks[0], ticks[-1] + 1)):
            gaps.append(vi)                # non-contiguous presence
        first, last = idxs[0], idxs[-1]
        vid = vids[vi]
        births.append((vi, ticks[0], int(arr["lane"][first]),
                       float(arr["s"][first]), float(arr["v"][first]),
                       1 if vid in departed else 0))
        exits.append((vi, ticks[-1], float(arr["s"][last]),
                      1 if vid in arrived else 0))
        # lane changes: compare consecutive PHYSICAL rows per vehicle, so a
        # change that coincides with a junction crossing is still caught;
        # same-lane junction crossings produce no event
        prev_phys = None
        for i in idxs:
            li = int(arr["lane"][i])
            if li < 0:
                continue
            if prev_phys is not None:
                j, lj = prev_phys
                if lj != li:
                    changes.append((vi, int(arr["tick"][i]), lj, li,
                                    float(arr["s"][i])))
            prev_phys = (i, li)

        # exits: a vehicle whose last row has arrived=1 truly left the
        # corridor; last row at the window end is a truncation, not an exit

    # ---- arrivals in the contract's explicit format
    cav_set = cav_ids(run_dir)
    manifest = load_json(Path(run_dir) / "manifest.json")
    pop_cav = manifest["scenario"]["cav_population"]
    arrivals = []
    for vi, tk, lane0, s0, v0, dep in births:
        vid = vids[vi]
        if vid not in departed:
            continue
        arrivals.append((WINDOW_T0 + tk * DT,
                         0 if vid.startswith("mainline") else 1,
                         v0, 1 if vid in cav_set else 0,
                         int(arr["typ"][by_vid[vi][0]])))

    # ---- is_cav post-hoc
    cav_flag = np.isin(arr["vid"], sorted(
        vid_idx[c] for c in cav_set if c in vid_idx))
    arr["cav"] = cav_flag.astype(np.int8)

    # ---- sanity: s-advance residual vs v*dt for every non-lane-change
    # consecutive pair of a vehicle, INCLUDING junction bridges (this is the
    # cross-segment-boundary continuity check; junction rows carry the exact
    # boundary s, so a residual there means the bridge itself is off)
    resids_phys, resids_bridge = [], []
    change_at = {}
    for c in changes:
        change_at.setdefault(c[0], set()).add(c[1])
    for vi, idxs in by_vid.items():
        ch = change_at.get(vi, ())
        for a_, b_ in zip(idxs, idxs[1:]):
            ta, tb = int(arr["tick"][a_]), int(arr["tick"][b_])
            if any(ta < t <= tb for t in ch):
                continue                       # declared lane change inside
            r = ((float(arr["s"][b_]) - float(arr["s"][a_]))
                 - float(arr["v"][a_]) * DT)
            if arr["inj"][a_] or arr["inj"][b_]:
                resids_bridge.append(r)        # junction bridge (boundary-s
            else:                              # approximation, documented)
                resids_phys.append(r)
    resids_phys = np.asarray(resids_phys, dtype=np.float64)
    resids_bridge = np.asarray(resids_bridge, dtype=np.float64)

    flagged = dict(junction_rows=int((arr["inj"] == 1).sum()),
                   unmapped=int(((arr["lane"] < 0)
                                 & (arr["inj"] == 0)).sum()))
    npz = dict(
        vid=np.asarray(arr["vid"], dtype=np.int32),
        tick=np.asarray(arr["tick"], dtype=np.int16),
        lane=np.asarray(arr["lane"], dtype=np.int8),
        s=np.asarray(arr["s"], dtype=np.float64),
        v=np.asarray(arr["v"], dtype=np.float32),
        a=np.asarray(arr["a"], dtype=np.float32),
        is_cav=np.asarray(arr["cav"], dtype=np.int8),
        in_junction=np.asarray(arr["inj"], dtype=np.int8),
        type_label=np.asarray(arr["typ"], dtype=np.int8),
        vid_table=np.asarray(vids),
    )
    ev = dict(
        lc_vid=np.asarray([c[0] for c in changes], dtype=np.int32),
        lc_tick=np.asarray([c[1] for c in changes], dtype=np.int16),
        lc_from=np.asarray([c[2] for c in changes], dtype=np.int8),
        lc_to=np.asarray([c[3] for c in changes], dtype=np.int8),
        lc_s=np.asarray([c[4] for c in changes], dtype=np.float64),
        b_vid=np.asarray([b[0] for b in births], dtype=np.int32),
        b_tick=np.asarray([b[1] for b in births], dtype=np.int16),
        b_lane=np.asarray([b[2] for b in births], dtype=np.int8),
        b_s=np.asarray([b[3] for b in births], dtype=np.float64),
        b_v=np.asarray([b[4] for b in births], dtype=np.float32),
        b_departed=np.asarray([b[5] for b in births], dtype=np.int8),
        e_vid=np.asarray([e[0] for e in exits], dtype=np.int32),
        e_tick=np.asarray([e[1] for e in exits], dtype=np.int16),
        e_s=np.asarray([e[2] for e in exits], dtype=np.float64),
        e_arrived=np.asarray([e[3] for e in exits], dtype=np.int8),
        ar_tick=np.asarray([int(round((a[0] - WINDOW_T0) / DT))
                            for a in arrivals], dtype=np.int16),
        ar_entrance=np.asarray([a[1] for a in arrivals], dtype=np.int8),
        ar_v=np.asarray([a[2] for a in arrivals], dtype=np.float32),
        ar_cav=np.asarray([a[3] for a in arrivals], dtype=np.int8),
        ar_type=np.asarray([a[4] for a in arrivals], dtype=np.int8),
    )
    npz.update(ev)

    
    if len(resids_phys):
        rq = np.percentile(np.abs(resids_phys), [50, 95, 99, 100])
    else:
        rq = [float("nan")] * 4
    bmax = (float(np.abs(resids_bridge).max())
            if len(resids_bridge) else float("nan"))
    stats = dict(
        tag=tag, run_dir=str(run_dir),
        n_ticks_present=n_steps, missing_ticks=len(missing),
        n_vehicle_steps=int(len(arr["vid"])),
        n_vehicles=len(vids), n_cav_ids=len(cav_set),
        manifest_cav_population=pop_cav,
        n_lane_changes=len(changes), n_births=len(births),
        n_exits=len(exits), n_arrivals=len(arrivals),
        n_gapped_vids=len(gaps),
        flagged=dict(flagged),
        resid_abs_p50_p95_p99_max=[float(x) for x in rq],
        n_resid=int(len(resids_phys)),
        resid_bridge_abs_max_m=bmax,
        n_bridge_pairs=int(len(resids_bridge)),
    )
    return dict(npz=npz, stats=stats)


# --------------------------------------------------------------- loaders ----

def load_run_map():
    m = load_json(PART1 / "results" / "tag_run_map.json")
    return [(r["tag"], wdpaths.remap(r["run_dir"]), r["split"])
            for r in m["rows"]], m


def load_v3_splits():
    m = load_json(SRC / "output" / "worldmodel_v3_vehicle" /
                  "build_tracks_manifest.json")
    return {r["tag"]: r["split"] for r in m["runs"]}
