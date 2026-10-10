"""V0 featurizer (DESIGN_PARTB_V2.md §3 inputs; v0.4 action/wall inputs per
BRIEF_PARTB_WALL_ACTION_INPUTS_DRAFT.md items 1-2).

Per decision time t, from one RunTrace snapshot plus the shared schedule:

  lane_bins   [3, LMAX_BINS, 3]   count / mean v / std v per 10 m bin
                                  (part4/partB skeleton lane_bin_field)
  lane_mask   [3, LMAX_BINS]      occupancy mask (skeleton convention)
  lane_geom   [3, LMAX_BINS]      static geometry-validity mask (1 = bin
                                  inside the physical lane's length)
  cav_tokens  [MAX_CAVS, CAV_RAW] skeleton slot layout; action one-hot
                                  over {native, damping, pi, wall, unknown}
                                  (+ phase, wall speed, time in phase,
                                  member flag, planned hold remaining).
                                  Actions come ONLY from the v0.4
                                  reconstruction (switch_schedule.json /
                                  suffix_commands.jsonl owners / fleet
                                  manifest policy); with no evidence a CAV
                                  is "unknown" (its own one-hot slot) --
                                  never silently "native".
  wall_tokens [MAX_WALLS, WALL_RAW]  planner-known wall state from the
                                  wall pilot's wall_events.jsonl (v0.4);
                                  all-zero when no wall episode set is
                                  given (no pilot data yet)
  demand      [DEMAND_RAW]        scheduled arrivals per minute per entrance
                                  over (t, t+600] (/50) + designated-CAV
                                  share; tau-conditioning masks minutes
                                  >= tau/60 downstream (tau <= 570 < 600)
  context     [CTX_RAW]           [t/1000, (t-t0)/600, time-to-branch/600,
                                  loading/2700, 0, 0, q_main/50, q_ramp/50]

The raw snapshot rows (vid index, lane, s, v, a, in-junction, speed mode,
is_cav) are stored flat alongside, so features can be rebuilt or extended
without re-reading the physics logs.  Nothing here reveals vType.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "part4",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Load the part4/partB skeleton featurizer by file path: its blocks and
# scaling conventions are the reuse contract; the package __init__ pulls
# torch, which the dataset build does not need.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "partb_featurize_skeleton",
    HERE.parent / "part4" / "partB" / "featurize.py")
_SK = importlib.util.module_from_spec(_spec)
sys.modules["partb_featurize_skeleton"] = _SK   # dataclass machinery needs it
_spec.loader.exec_module(_SK)

from actions_v04 import (  # noqa: E402  (same directory; v0.4 inputs)
    ACTIONS_V04, ACTION_UNKNOWN)
from wall_tokens_v04 import (  # noqa: E402
    ACTIVE_WALL_PHASES, WALL_PHASES)

CAV_RAW = _SK.CAV_RAW
CTX_RAW = _SK.CTX_RAW
DEMAND_RAW = _SK.DEMAND_RAW
LMAX_BINS = _SK.LMAX_BINS
N_LANES = _SK.N_LANES
WALL_RAW = _SK.WALL_RAW
lane_bin_field = _SK.lane_bin_field
_action_onehot = _SK._action_onehot
_phase_onehot = _SK._phase_onehot

# single-vocabulary guard: the skeleton's action slot order IS the v0.4
# token order (partB_v0/model.py sizes its projections from CAV_RAW)
assert tuple(_SK.ACTIONS) == ACTIONS_V04, \
    f"skeleton action slots {(_SK.ACTIONS)} drifted from {ACTIONS_V04}"
assert tuple(ACTIVE_WALL_PHASES) == ("requested", "assembling", "holding")
assert len(WALL_PHASES) == len(_SK.PHASES) - 1

from v0lib import CAV_SPEED_MODE_CHUNK1  # noqa: E402

MAX_CAVS = 64                       # skeleton featurize() default
MAX_WALLS = 4                       # skeleton featurize() default


def geometry_mask(geom) -> np.ndarray:
    """(3, LMAX_BINS) float32: 1 where the 10 m bin is inside the lane."""
    m = np.zeros((N_LANES, LMAX_BINS), dtype=np.float32)
    for lane in (0, 1, 2):
        nbin = int(np.ceil(geom.length(lane) / 10.0))
        m[lane, :nbin] = 1.0
    return m


def _demand_vector(rt, t: float) -> np.ndarray:
    fut = (rt.sched_t > t + 1e-9) & (rt.sched_t <= t + 600.0 + 1e-9)
    grid = np.zeros((10, 2), dtype=np.float64)
    if fut.any():
        mins = np.clip(((rt.sched_t[fut] - t) / 60.0).astype(np.int64), 0, 9)
        np.add.at(grid, (mins, rt.sched_ent[fut].astype(np.int64)), 1.0)
    share = float(rt.sched_is_cav[fut].mean()) if fut.any() else 0.0
    return np.concatenate([grid.reshape(-1) / 50.0,
                           np.array([share])]).astype(np.float32)


def _context_vector(rt, t: float, q: tuple) -> np.ndarray:
    t0 = rt.world_t0
    time_to_branch = max(0.0, t0 - t) if rt.source == "base" else 0.0
    return np.array([
        t / 1000.0, (t - t0) / 600.0, time_to_branch / 600.0,
        rt.demand_loading / 2700.0, 0.0, 0.0,
        q[0] / 50.0, q[1] / 50.0,
    ], dtype=np.float32)


def _empty_blocks() -> dict:
    return dict(
        lane_bins=np.zeros((N_LANES, LMAX_BINS, 3), np.float32),
        lane_mask=np.zeros((N_LANES, LMAX_BINS), np.float32),
        cav_tokens=np.zeros((MAX_CAVS, CAV_RAW), np.float32),
        cav_mask=np.zeros((MAX_CAVS,), np.float32),
        wall_tokens=np.zeros((MAX_WALLS, WALL_RAW), np.float32),
        wall_mask=np.zeros((MAX_WALLS,), np.float32))


def _wall_token_row(w: dict) -> np.ndarray:
    """WALL_RAW row from one walls_at() state dict (field table in
    wall_tokens_v04.WALL_TOKEN_FIELDS)."""
    row = np.zeros(WALL_RAW, np.float32)
    ph = np.zeros(len(WALL_PHASES), np.float32)
    ph[WALL_PHASES.index(w["phase"])] = 1.0
    off = 0
    row[off:off + len(WALL_PHASES)] = ph
    off += len(WALL_PHASES)
    row[off] = float(w["speed_mps"]) / 40.0
    row[off + 1] = float(w["time_in_phase_s"]) / 600.0
    row[off + 2] = float(w["planned_hold_remaining_s"]) / 600.0
    row[off + 3] = float(w["n_members"]) / 4.0
    row[off + 4] = float(w["mean_s_down_m"]) / 1000.0
    row[off + 5] = float(w["lane_span"]) / 2.0
    return row


def featurize_at(rt, t: float, geom, actions=None, walls=None) -> dict:
    """All V0 feature blocks + raw snapshot rows for one decision time.

    actions: v0.4 per-CAV action input (actions_v04.ActionTimeline-like:
    action_at(vid, t) -> canonical token).  None = no evidence -- every
    CAV carries the explicit "unknown" action slot, never a silent
    "native".
    walls: v0.4 wall input (wall_tokens_v04.WallEpisodeSet-like:
    walls_at(t, positions, cav_roster)).  None = no wall pilot data
    (all-zero wall tokens, as in every existing dataset).
    """
    from v0lib import queue_lengths
    snap = rt.snapshot_at(t)
    q = queue_lengths(rt, t)
    out = dict(t=float(t), q_main=float(q[0]), q_ramp=float(q[1]),
               n_present=0, n_cav=0)
    if snap is None:
        out.update(_empty_blocks())
        out["demand"] = _demand_vector(rt, t)
        out["context"] = _context_vector(rt, t, q)
        out["raw"] = {k: np.zeros(0, np.int64) for k in
                      ("vid", "lane", "inj", "is_cav")}
        out["raw"].update({k: np.zeros(0, np.float64)
                           for k in ("s", "v", "a", "smode")})
        return out

    rows = np.stack([snap["lane"].astype(np.float64), snap["s"],
                     snap["v"].astype(np.float64)], axis=1)
    ok = np.isfinite(rows).all(axis=1)
    bins, mask = lane_bin_field(rows[ok], LMAX_BINS)
    out["lane_bins"], out["lane_mask"] = bins, mask

    cav_sel = np.flatnonzero(snap["is_cav"] == 1)
    cav_tok = np.zeros((MAX_CAVS, CAV_RAW), np.float32)
    cav_msk = np.zeros((MAX_CAVS,), np.float32)
    order = cav_sel[np.argsort(snap["vid"][cav_sel], kind="stable")]
    # wall input: planner-known state at t; member geometry from the CAV
    # self-report rows of THIS snapshot only (positions of every vehicle
    # are passed; the wall set consults members = designated CAVs).
    # snap["vid"] holds vid_table INDICES: resolve them to vehicle ids.
    wall_states = []
    member_of: dict = {}
    if walls is not None:
        vid_str = [str(rt.vid_table[int(k)]) for k in snap["vid"]]
        pos = {vid_str[i]: (float(snap["s"][i]), int(snap["lane"][i]))
               for i in range(len(vid_str))}
        roster = {vid_str[i] for i in cav_sel}
        wall_states = walls.walls_at(t, positions=pos, cav_roster=roster)
        for w in wall_states:
            if w["phase"] in ACTIVE_WALL_PHASES:
                for m in w["members"]:
                    member_of[str(m)] = w
    for i, j in enumerate(order[:MAX_CAVS]):
        row = cav_tok[i]
        vid = str(rt.vid_table[int(snap["vid"][j])])
        row[0] = snap["s"][j] / 1000.0
        row[1:4] = np.eye(3, dtype=np.float32)[int(snap["lane"][j])]
        row[4] = snap["v"][j] / 40.0
        off = 5
        w = member_of.get(vid)
        action = ("wall" if w is not None else
                  (actions.action_at(vid, t) if actions is not None
                   else ACTION_UNKNOWN))
        row[off:off + len(ACTIONS_V04)] = _action_onehot(action)
        off += len(ACTIONS_V04)
        row[off:off + len(_SK.PHASES)] = _phase_onehot(
            w["phase"] if w is not None else None)
        off += len(_SK.PHASES)
        row[off] = (w["speed_mps"] / 40.0) if w is not None else 0.0
        row[off + 1] = (w["time_in_phase_s"] / 600.0) \
            if w is not None else 0.0
        row[off + 2] = 1.0 if w is not None else 0.0
        row[off + 3] = (w["planned_hold_remaining_s"] / 600.0) \
            if w is not None else 0.0
        cav_msk[i] = 1.0
    out["cav_tokens"], out["cav_mask"] = cav_tok, cav_msk
    out["n_cav"] = int(len(order))

    wall_tok = np.zeros((MAX_WALLS, WALL_RAW), np.float32)
    wall_msk = np.zeros((MAX_WALLS,), np.float32)
    for wi, w in enumerate(wall_states[:MAX_WALLS]):
        wall_tok[wi] = _wall_token_row(w)
        wall_msk[wi] = 1.0
    out["wall_tokens"], out["wall_mask"] = wall_tok, wall_msk
    out["demand"] = _demand_vector(rt, t)
    out["context"] = _context_vector(rt, t, q)
    out["n_present"] = int(len(snap["vid"]))
    out["raw"] = dict(
        vid=snap["vid"].astype(np.int64), lane=snap["lane"].astype(np.int64),
        s=snap["s"].astype(np.float64), v=snap["v"].astype(np.float64),
        a=snap["a"].astype(np.float64), inj=snap["inj"].astype(np.int64),
        smode=np.nan_to_num(snap["smode"], nan=float("nan")).astype(np.float64),
        is_cav=snap["is_cav"].astype(np.int64))
    return out


def build_features(rt, ts: np.ndarray, tau_avail: np.ndarray, geom,
                   actions=None, walls=None) -> dict:
    """Stacked npz payload for one source: features per decision time plus
    the flat raw snapshots and the per-source identity scalars."""
    per = [featurize_at(rt, t, geom, actions=actions, walls=walls)
           for t in ts]
    n = len(per)
    raw_off = np.zeros(n + 1, np.int64)
    for i, p in enumerate(per):
        raw_off[i + 1] = raw_off[i] + p["n_present"]
    m_raw = int(raw_off[-1])
    raw = {k: np.zeros(m_raw, np.int64) for k in
           ("vid", "lane", "inj", "is_cav")}
    raw.update({k: np.zeros(m_raw, np.float64)
                for k in ("s", "v", "a", "smode")})
    for i, p in enumerate(per):
        sl = slice(int(raw_off[i]), int(raw_off[i + 1]))
        for k, arr in p["raw"].items():
            raw[k][sl] = arr

    return dict(
        t=np.asarray([p["t"] for p in per], np.float64),
        tau_available=tau_avail.astype(np.float64),
        lane_bins=np.stack([p["lane_bins"] for p in per]).astype(np.float32),
        lane_mask=np.stack([p["lane_mask"] for p in per]).astype(np.float32),
        lane_geom=geometry_mask(geom),
        cav_tokens=np.stack([p["cav_tokens"] for p in per]).astype(np.float32),
        cav_mask=np.stack([p["cav_mask"] for p in per]).astype(np.float32),
        wall_tokens=np.stack([p["wall_tokens"] for p in per]).astype(np.float32),
        wall_mask=np.stack([p["wall_mask"] for p in per]).astype(np.float32),
        demand=np.stack([p["demand"] for p in per]).astype(np.float32),
        context=np.stack([p["context"] for p in per]).astype(np.float32),
        q_main=np.asarray([p["q_main"] for p in per], np.float32),
        q_ramp=np.asarray([p["q_ramp"] for p in per], np.float32),
        n_cav=np.asarray([p["n_cav"] for p in per], np.int32),
        raw_off=raw_off,
        raw_vid=raw["vid"].astype(np.int32),
        raw_lane=raw["lane"].astype(np.int8),
        raw_s=raw["s"].astype(np.float64),
        raw_v=raw["v"].astype(np.float32),
        raw_a=raw["a"].astype(np.float32),
        raw_inj=raw["inj"].astype(np.int8),
        raw_smode=raw["smode"].astype(np.float32),
        raw_is_cav=raw["is_cav"].astype(np.int8),
        vid_table=rt.vid_table,
        cav_speed_mode=np.int32(CAV_SPEED_MODE_CHUNK1),
    )


def attach_meta(rt, payload: dict) -> dict:
    """Per-source scalars stored alongside the arrays (small npz entries)."""
    payload = dict(payload)
    payload.update(
        world=np.str_(rt.world), source=np.str_(rt.source),
        world_t0=np.float64(rt.world_t0), data_end=np.float64(rt.data_end))
    return payload


def build_features_multi(items, main_rt, tau_avail: np.ndarray, geom,
                         actions=None, walls=None) -> dict:
    """build_features over decision times that may come from DIFFERENT
    RunTraces (v0.4a): items = [(rt, t), ...].  A restored branch trace
    has no snapshot at its own t0 (its grid starts at t0 + 0.5 s), so the
    branch's t0 decision row is featurized from the shared base trace
    (identical saved state by the CRN restore) while every later row uses
    the branch trace.  Raw snapshot vid indices are re-mapped onto ONE
    unified vid_table (main_rt's table first, then any extra ids).  With
    every item on main_rt this is exactly build_features."""
    per = [(featurize_at(rt, t, geom, actions=actions, walls=walls), rt)
           for rt, t in items]
    names = [str(v) for v in main_rt.vid_table]
    pos = {v: i for i, v in enumerate(names)}
    remap: dict = {}
    for _p, rt in per:
        if id(rt) in remap:
            continue
        m = np.zeros(len(rt.vid_table), np.int64)
        for i, v in enumerate(rt.vid_table):
            v = str(v)
            if v not in pos:
                pos[v] = len(names)
                names.append(v)
            m[i] = pos[v]
        remap[id(rt)] = m
    n = len(per)
    raw_off = np.zeros(n + 1, np.int64)
    for i, (p, _rt) in enumerate(per):
        raw_off[i + 1] = raw_off[i] + p["n_present"]
    m_raw = int(raw_off[-1])
    raw = {k: np.zeros(m_raw, np.int64) for k in
           ("vid", "lane", "inj", "is_cav")}
    raw.update({k: np.zeros(m_raw, np.float64)
                for k in ("s", "v", "a", "smode")})
    for i, (p, rt) in enumerate(per):
        sl = slice(int(raw_off[i]), int(raw_off[i + 1]))
        for k, arr in p["raw"].items():
            raw[k][sl] = remap[id(rt)][arr] if k == "vid" and len(arr) \
                else arr
    ps = [p for p, _rt in per]
    return dict(
        t=np.asarray([p["t"] for p in ps], np.float64),
        tau_available=tau_avail.astype(np.float64),
        lane_bins=np.stack([p["lane_bins"] for p in ps]).astype(np.float32),
        lane_mask=np.stack([p["lane_mask"] for p in ps]).astype(np.float32),
        lane_geom=geometry_mask(geom),
        cav_tokens=np.stack([p["cav_tokens"] for p in ps]).astype(np.float32),
        cav_mask=np.stack([p["cav_mask"] for p in ps]).astype(np.float32),
        wall_tokens=np.stack([p["wall_tokens"] for p in ps]).astype(np.float32),
        wall_mask=np.stack([p["wall_mask"] for p in ps]).astype(np.float32),
        demand=np.stack([p["demand"] for p in ps]).astype(np.float32),
        context=np.stack([p["context"] for p in ps]).astype(np.float32),
        q_main=np.asarray([p["q_main"] for p in ps], np.float32),
        q_ramp=np.asarray([p["q_ramp"] for p in ps], np.float32),
        n_cav=np.asarray([p["n_cav"] for p in ps], np.int32),
        raw_off=raw_off,
        raw_vid=raw["vid"].astype(np.int32),
        raw_lane=raw["lane"].astype(np.int8),
        raw_s=raw["s"].astype(np.float64),
        raw_v=raw["v"].astype(np.float32),
        raw_a=raw["a"].astype(np.float32),
        raw_inj=raw["inj"].astype(np.int8),
        raw_smode=raw["smode"].astype(np.float32),
        raw_is_cav=raw["is_cav"].astype(np.int8),
        vid_table=np.asarray(names, dtype=np.str_),
        cav_speed_mode=np.int32(CAV_SPEED_MODE_CHUNK1),
    )
