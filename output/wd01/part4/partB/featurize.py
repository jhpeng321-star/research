"""Part B featurizer (DESIGN_PARTS_A_B.md §3.2).

Turns an end-of-Part-A state -- contract trajectory layout
(PREDICT_CONTRACT_v1.md §4.3: per-vehicle [T,4] (t_s, s_m, lane_id,
speed_mps), NaN trailing suffix) at t_end = t0 + A, plus the plan, the
arrival draw and context scalars -- into fixed-size numpy feature arrays
for the DelayEstimator (partB.model).  Pure functions, deterministic,
no torch here.

Feature blocks (raw, unnormalised sizes declared in RAW):
  lane bins   [3, LMAX_BINS, 3]   count, mean v, std v per 10 m bin
  cav tokens  [MAX_CAVS, CAV_RAW] per-CAV state + action/wall tokens
  wall tokens [MAX_WALLS, WALL_RAW]
  demand      [DEMAND_RAW]        per-minute counts/entrance + CAV share
  context     [CTX_RAW]           t0, A, branch context, Part A stats

Nothing here reveals vType (supervision-only); ideal-sensing inputs only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# physical lanes (WM-30)
LANE_MAIN_LEFT, LANE_MAIN_RIGHT, LANE_ACCEL = 0, 1, 2
N_LANES = 3
BIN_WIDTH_M = 10.0
LMAX_BINS = 320                      # covers 3199.1 m main lanes; padded

ACTIONS = ("native", "damping", "pi", "wall", "unknown")
# legacy controller/owner spellings -> the canonical action vocabulary
ACTION_ALIASES = {"pi-external-u": "pi", "wall-member": "wall",
                  "wall-v2": "wall"}
PHASES = ("none", "requested", "assembling", "holding", "released",
          "aborted")
# legacy plan/dispatcher spellings -> the canonical phase vocabulary
PHASE_ALIASES = {"assemble": "assembling", "assembly": "assembling",
                 "hold": "holding", "release": "released",
                 "abort": "aborted", "safety_abort": "aborted"}

# v0.4 token layout (BRIEF_PARTB_WALL_ACTION_INPUTS_DRAFT.md items 1-2;
# bump widths ONLY here -- featurize_v0 and partB_v0/model.py import them):
#   cav: s, lane(3), v, action(5), phase(6), wall speed, t-in-phase,
#        member, planned hold remaining
#   wall: phase(5), speed, t-in-phase, planned hold remaining (planner-
#         owned, flagged), n_members, members' mean s_down, lane span
CAV_RAW = 20
WALL_RAW = 11
DEMAND_RAW = 21    # 10 min x 2 entrances counts + cav share
CTX_RAW = 8        # t0, A, since-branch, regime, A-delay, ttc, queue main, queue ramp


@dataclass
class PartBFeatures:
    lane_bins: np.ndarray          # [3, LMAX, 3] float32
    lane_mask: np.ndarray          # [3, LMAX] float32 (1 = valid bin)
    cav_tokens: np.ndarray         # [MAX_CAVS, CAV_RAW] float32
    cav_mask: np.ndarray           # [MAX_CAVS] float32
    wall_tokens: np.ndarray        # [MAX_WALLS, WALL_RAW] float32
    wall_mask: np.ndarray          # [MAX_WALLS] float32
    demand: np.ndarray             # [DEMAND_RAW] float32
    context: np.ndarray            # [CTX_RAW] float32
    meta: dict = field(default_factory=dict)


def _snapshot(traj: np.ndarray, t_end: float, dt: float):
    """(lane, s, v) at the vehicle's last valid row at or before t_end.

    None if the vehicle is not on the section at t_end: rows after t_end
    are ignored, and a vehicle whose last row is older than one step has
    left the section (NaN suffix from the exit step, contract §4.3)."""
    valid = (np.isfinite(traj[:, :3]).all(axis=1)
             & (traj[:, 0] <= t_end + 1e-9))
    if not valid.any():
        return None
    last = int(np.flatnonzero(valid)[-1])
    if traj[last, 0] < t_end - dt - 1e-9:
        return None                            # left the section before t_end
    return traj[last, 2], traj[last, 1], traj[last, 3]


def lane_bin_field(rows: np.ndarray, lmax: int = LMAX_BINS):
    """(count, mean v, std v) per (lane, 10 m bin) from snapshot rows
    [n, 3] of (lane, s, v).  Empty bins are zero with mask 0."""
    bins = np.zeros((N_LANES, lmax, 3), dtype=np.float64)
    cnt = np.zeros((N_LANES, lmax), dtype=np.float64)
    m = np.isfinite(rows).all(axis=1)
    rows = rows[m]
    lane = rows[:, 0].astype(np.int64)
    kb = np.clip((rows[:, 1] / BIN_WIDTH_M).astype(np.int64), 0, lmax - 1)
    np.add.at(cnt, (lane, kb), 1.0)
    np.add.at(bins[:, :, 0], (lane, kb), rows[:, 2])
    np.add.at(bins[:, :, 1], (lane, kb), rows[:, 2] ** 2)
    out = np.zeros((N_LANES, lmax, 3), dtype=np.float32)
    mask = (cnt > 0).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, bins[:, :, 0] / np.maximum(cnt, 1), 0.0)
        var = np.where(cnt > 1,
                       bins[:, :, 1] / np.maximum(cnt, 1) - mean ** 2, 0.0)
    out[:, :, 0] = cnt.astype(np.float32)
    out[:, :, 1] = mean.astype(np.float32)
    out[:, :, 2] = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    return out, mask


def _action_onehot(action: str) -> np.ndarray:
    v = np.zeros(len(ACTIONS), dtype=np.float32)
    a = ACTION_ALIASES.get(str(action), str(action))
    if a in ACTIONS:
        v[ACTIONS.index(a)] = 1.0
    return v


def _phase_onehot(phase) -> np.ndarray:
    v = np.zeros(len(PHASES), dtype=np.float32)
    p = PHASES[0] if phase is None else str(phase)
    p = PHASE_ALIASES.get(p, p)
    if p in PHASES:
        v[PHASES.index(p)] = 1.0
    return v


def featurize(state: dict, plan: dict, arrivals: dict, ctx: dict | None = None,
              lmax: int = LMAX_BINS, max_cavs: int = 64,
              max_walls: int = 4) -> PartBFeatures:
    """End-of-Part-A state -> PartBFeatures.

    state: {"t0": s, "part_a_horizon_s": A, "trajectories": {vid: [T,4]},
            "controller_state": {cav: {action, wall_id, wall_phase,
                                       time_in_phase_s, wall_speed_mps}}}
    plan:  {"per_cav": {cav: action}, "walls": [WallSpec-like dicts with
            wall_id, members, speed_mps, phase, release_at_s|hold_s]}
    arrivals: {"t_s": (n,), "entrance": (n,) 0/1 or "mainline"/"ramp",
               "is_cav": (n,)} -- the draw for [t0, t0+600]
    ctx:   {"time_since_last_native_branch_s", "demand_regime",
            "part_a_delay_s", "part_a_ttc_lt3_exposure_s"} (defaults 0)
    """
    t0 = float(state["t0"])
    A = float(state["part_a_horizon_s"])
    t_end = t0 + A
    traj = state.get("trajectories", {}) or {}
    cstate = state.get("controller_state", {}) or {}

    # ---- snapshot of every vehicle at t_end -----------------------------
    snaps = {}
    for vid, arr in traj.items():
        sn = _snapshot(np.asarray(arr, dtype=np.float64), t_end, A)
        if sn is not None:
            snaps[str(vid)] = sn
    snap_rows = np.array([(ln, s, v) for (ln, s, v) in snaps.values()],
                         dtype=np.float64) if snaps else \
        np.zeros((0, 3), dtype=np.float64)
    lane_bins, lane_mask = lane_bin_field(snap_rows, lmax)

    # ---- CAV tokens ------------------------------------------------------
    per_cav = dict(plan.get("per_cav", {}) or {})
    walls = list(plan.get("walls", []) or [])
    member_of = {m: wi for wi, w in enumerate(walls)
                 for m in (w.get("members") or ())}
    cav_ids = sorted(set(per_cav) | set(member_of) | set(cstate),
                     key=str)[:max_cavs]
    cav_tok = np.zeros((max_cavs, CAV_RAW), dtype=np.float32)
    cav_mask = np.zeros(max_cavs, dtype=np.float32)
    for i, cid in enumerate(cav_ids):
        sn = snaps.get(cid)
        cs = cstate.get(cid, {}) or {}
        row = cav_tok[i]
        if sn is not None:
            row[0] = sn[1] / 1000.0
            row[1:4] = np.eye(3, dtype=np.float32)[int(sn[0])]
            row[4] = sn[2] / 40.0
        off = 5
        action = ("wall" if cid in member_of
                  else str(per_cav.get(cid, cs.get("action", "unknown"))))
        row[off:off + len(ACTIONS)] = _action_onehot(action)
        off += len(ACTIONS)
        row[off:off + len(PHASES)] = _phase_onehot(cs.get("wall_phase"))
        off += len(PHASES)
        row[off] = (cs.get("wall_speed_mps") or 0.0) / 40.0
        row[off + 1] = (cs.get("time_in_phase_s") or 0.0) / 600.0
        row[off + 2] = 1.0 if cid in member_of else 0.0
        remaining = _wall_remaining(walls[member_of[cid]], t_end) \
            if cid in member_of else 0.0
        row[off + 3] = remaining / 600.0
        cav_mask[i] = 1.0

    # ---- wall tokens ------------------------------------------------------
    wall_tok = np.zeros((max_walls, WALL_RAW), dtype=np.float32)
    wall_mask = np.zeros(max_walls, dtype=np.float32)
    for wi, w in enumerate(walls[:max_walls]):
        members = list(w.get("members") or ())
        present = [snaps[m] for m in members if m in snaps]
        row = wall_tok[wi]
        off = 0
        row[off:off + len(PHASES) - 1] = _phase_onehot(w.get("phase"))[1:]
        off += len(PHASES) - 1
        row[off] = (w.get("speed_mps") or 0.0) / 40.0
        row[off + 1] = (w.get("time_in_phase_s") or 0.0) / 600.0
        row[off + 2] = _wall_remaining(w, t_end) / 600.0   # planner-owned
        row[off + 3] = len(members) / 4.0
        row[off + 4] = (float(np.mean([sn[1] for sn in present])) / 1000.0
                        if present else 0.0)
        lanes = {int(sn[0]) for sn in present if int(sn[0]) >= 0}
        row[off + 5] = ((max(lanes) - min(lanes)) / 2.0
                        if len(lanes) >= 2 else 0.0)
        wall_mask[wi] = 1.0

    # ---- demand: per-minute counts per entrance over (t_end, t0+600] ----
    a_t = np.asarray(arrivals.get("t_s", []), dtype=np.float64)
    a_ent = np.asarray(arrivals.get("entrance", []))
    a_cav = np.asarray(arrivals.get("is_cav", np.zeros(len(a_t))), )
    ent_code = np.where(np.isin(a_ent, (1, "ramp")), 1, 0)
    fut = a_t > t_end + 1e-9
    grid = np.zeros((10, 2), dtype=np.float32)
    if fut.any():
        mins = np.clip(((a_t[fut] - t_end) / 60.0).astype(np.int64), 0, 9)
        np.add.at(grid, (mins, ent_code[fut]), 1.0)
    n_fut = int(fut.sum())
    share = float(np.mean(a_cav[fut])) if n_fut and len(a_cav) == len(a_t) \
        else 0.0
    demand = np.concatenate([grid.reshape(-1) / 50.0,
                             np.array([share], np.float32)]).astype(np.float32)

    # ---- queue lengths at the entrances at t_end ------------------------
    # Skeleton approximation: draw rows at/before t_end minus vehicles
    # actually inserted by t_end (first valid trajectory row <= t_end);
    # the caller may override both via ctx queue_* scalars.
    q = np.zeros(2, dtype=np.float32)
    passed = a_t <= t_end + 1e-9
    n_pass = int(passed.sum())
    n_inserted = 0
    for arr in traj.values():
        arr = np.asarray(arr, dtype=np.float64)
        valid = np.isfinite(arr[:, :3]).all(axis=1)
        if valid.any() and arr[int(np.flatnonzero(valid)[0]), 0] <= t_end + 1e-9:
            n_inserted += 1
    queued_total = max(0, n_pass - n_inserted)
    q[0] = queued_total                        # split unknown without ids;
    # both-entrance split should come from the predictor's queue state
    ctx = ctx or {}

    context = np.array([
        t0 / 600.0, A / 600.0,
        float(ctx.get("time_since_last_native_branch_s", 0.0)) / 600.0,
        float(ctx.get("demand_regime", 0.0)),
        float(ctx.get("part_a_delay_s", 0.0)) / 1000.0,
        float(ctx.get("part_a_ttc_lt3_exposure_s", 0.0)) / 100.0,
        float(ctx.get("queue_mainline", q[0])) / 50.0,
        float(ctx.get("queue_ramp", q[1])) / 50.0,
    ], dtype=np.float32)

    return PartBFeatures(
        lane_bins=lane_bins, lane_mask=lane_mask,
        cav_tokens=cav_tok, cav_mask=cav_mask,
        wall_tokens=wall_tok, wall_mask=wall_mask,
        demand=demand, context=context,
        meta=dict(t0=t0, A=A, n_snapshot=len(snaps), n_cav=len(cav_ids),
                  n_walls=len(walls)))


def _wall_remaining(w: dict, t_end: float) -> float:
    """Remaining hold/release time of a wall relative to the plan's t0,
    from release_at_s / hold_s (contract §3.3: exactly one is set)."""
    if w.get("release_at_s") is not None:
        return float(np.clip(float(w["release_at_s"]), 0.0, 600.0))
    if w.get("hold_s") is not None:
        return float(np.clip(float(w["hold_s"]), 0.0, 600.0))
    return 0.0


def collate(feats: list[PartBFeatures]) -> dict:
    """Stack a list of featurized samples into batch tensors (numpy).
    All arrays in PartBFeatures are already fixed-size."""
    keys = ("lane_bins", "lane_mask", "cav_tokens", "cav_mask",
            "wall_tokens", "wall_mask", "demand", "context")
    return {k: np.stack([getattr(f, k) for f in feats]).astype(np.float32)
            for k in keys}
