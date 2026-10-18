"""Shared helpers of the C4 world-model evaluation tools (wm_merge_check, wm_controller_ranking).

* checkpoint specs ``NAME=path[@MODE]`` (MODE: ON = merge rules on (v5m physics), OFF = both
  merge switches off, AS_IS = the checkpoint's own physics.json);
* a predictor loader that captures the raw rollout tensors and (``--device cuda``) moves every
  rollout input to the device and the outputs back to the CPU, so ``WmsPredictor.predict_states``
  is reused unchanged;
* :func:`roll` -> predicted and SUMO-recorded ``[T+1, N]`` arrays of the t0 roster;
* HDV metrics on those arrays and bootstrap / Spearman statistics.

No SUMO; CPU by default.  Paths with restricted-split tokens are refused.
"""
from __future__ import annotations

import dataclasses
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
DT_S = 0.5
MERGE_END_M = 75.44               # merge-zone end (s = 0 is the merge start)
SPEED_LIMIT_MPS = 31.29           # the collector's speed-deficit reference (traffic_shepherd.c4.env)
DELAY_END_M = 500.0               # delay is counted upstream of this line only (road end ~553 m):
                                  # SUMO exits (exogenous presence in the WM rollout) then do not
                                  # set the WM's exposure
MODES = {"OFF": (False, False), "ON": (True, True), "WAIT": (True, False), "YIELD": (False, True)}
MODE_FIELDS = ("merge_end_wait", "mainline_merge_yield")
#: V9ENTRY entry modes of :func:`load_predictor` (``off`` = the frozen default): ``queue`` = entries
#: v2 seeded with the recorded SUMO entrance backlog and demand (``wms.eval.entry_queue``).
ENTRY_MODES = {"off": {}, "v1": {"entries": True},
               "v2": {"entries": True, "entries_v2": True},
               "queue": {"entries": True, "entries_v2": True, "entry_queue_state": True}}


def guard(path):
    s = str(path).replace("\\", "/").lower()
    if any(t in s for t in FORBIDDEN):
        raise SystemExit(f"refusing restricted path {path}")
    return path


def parse_ckpt(spec: str, default_mode: str = "ON") -> tuple[str, str, str]:
    """``NAME=path[@MODE]`` -> (name, path, mode)."""
    if "=" not in spec:
        raise SystemExit(f"--ckpt needs NAME=path[@MODE], got {spec!r}")
    name, rest = spec.split("=", 1)
    mode = default_mode
    if "@" in rest and rest.rsplit("@", 1)[1].upper() in (*MODES, "AS_IS"):
        rest, mode = rest.rsplit("@", 1)
    mode = mode.upper()
    if mode not in (*MODES, "AS_IS"):
        raise SystemExit(f"unknown mode {mode}")
    return name, guard(rest), mode


# ----------------------------------------------------------------------------- physics switch
def _physics_holders(model):
    from wms.physics.config import PhysicsConfig
    found, seen = [], set()

    def visit(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        for name, val in list(vars(obj).items()):
            if isinstance(val, PhysicsConfig):
                found.append((obj, name))
        for m in getattr(obj, "_modules", {}).values():
            if m is not None:
                visit(m)

    visit(model)
    return found


def set_mode(model, mode: str) -> dict:
    """Replace every PhysicsConfig reference of ``model`` by a copy with the mode's two switches
    (same rule as tools/v6merge_probe.py, but the base physics is kept per model)."""
    from wms.physics.config import PhysicsConfig
    if mode == "AS_IS":
        return {"applied": {}}
    have = PhysicsConfig.__dataclass_fields__
    want = dict(zip(MODE_FIELDS, MODES[mode]))
    missing = [f for f in MODE_FIELDS if f not in have]
    if mode != "OFF" and missing:
        raise SystemExit(f"mode {mode} needs PhysicsConfig fields {missing}")
    changes = {k: v for k, v in want.items() if k in have}
    holders = _physics_holders(model)
    if not holders:
        raise SystemExit("no PhysicsConfig reference found on the model")
    if not hasattr(model, "_wm_eval_base_physics"):
        model._wm_eval_base_physics = model.physics
    new = dataclasses.replace(model._wm_eval_base_physics, **changes)
    for owner, name in holders:
        setattr(owner, name, new)
    return {"applied": changes, "holders": len(holders)}


# ----------------------------------------------------------------------------- predictor
def _to(obj, device):
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to(v, device) for v in obj)
    return obj


def load_predictor(path: str, mode: str, device: str = "cpu", threads: int = 1,
                   hdv_exits: str = "sumo", entries: str = "off"):
    """-> (predictor, capture dict); ``capture['last']`` holds the last raw rollout (CPU).

    ``hdv_exits``: ``sumo`` = the predictor's standard semantics (recorded SUMO presence ends
    every slot, HDVs included, so a SUMO exit can remove a vehicle the WM still has queued
    upstream); ``none`` = recorded exits end CAV slots only, HDVs stay until the model itself
    drops them (no SUMO outcome leaks into the predicted HDV state).

    ``entries`` (:data:`ENTRY_MODES`, default ``off``): vehicles entering after t0; ``queue`` =
    V9ENTRY entries ON with the recorded entrance queue (HDV entrants, sumo_like_v2 insertion).
    Entrants occupy slots after the t0 roster, so :func:`roll` (``[:n]``) is unchanged."""
    if hdv_exits not in ("sumo", "none"):
        raise SystemExit(f"hdv_exits must be sumo or none, not {hdv_exits!r}")
    if entries not in ENTRY_MODES:
        raise SystemExit(f"entries must be one of {sorted(ENTRY_MODES)}, not {entries!r}")
    import torch
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    torch.set_num_threads(threads)
    p = WmsPredictor.from_checkpoint(guard(path), RolloutFlags(**ENTRY_MODES[entries]),
                                     device=device)
    info = set_mode(p.model, mode)
    cap = {"mode_info": info, "hdv_exits": hdv_exits, "entries": entries}
    orig = p.model.rollout

    def rollout(state, **kw):
        if hdv_exits == "none" and kw.get("present_seq") is not None:
            ps = kw["present_seq"].clone()
            hdv = ~state["is_cav"].bool()                       # [B, N] slots at t0
            ps[:, :, :hdv.shape[1]] |= hdv[:, None, :] & state["present"].bool()[:, None, :]
            kw["present_seq"] = ps
        if device != "cpu":
            state, kw = _to(state, device), _to(kw, device)
        out = orig(state, **kw)
        if device != "cpu":
            out = _to(out, "cpu")
        cap["last"] = out
        return out

    p.model.rollout = rollout
    return p, cap


def roll(p, cap, rec: dict, arm: str, horizon_s: float = 120.0):
    """Roll ``rec`` with the CAVs teacher-forced from ``rec['recorded'][arm]``.

    Returns ``None`` when the arm has no recorded trajectory, else a dict of ``[T+1, N]`` arrays
    (row i = t0 + i * 0.5 s) for the t0 roster: ``wm_pos, wm_vel, wm_msk`` (predicted; row 0 = t0
    snapshot) and ``rec_pos, rec_vel, rec_msk`` (SUMO; exits end presence, junction transits are
    carried), plus ``is_cav [N]`` and ``ids``.  Entries are OFF: vehicles entering after t0 are in
    neither array.
    """
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
    n_steps = int(round(horizon_s / DT_S))
    t0, ids, kin, is_cav = t0_vehicles(rec)
    teacher, present_seq, _ = recorded_teacher(rec, t0, ids, kin, n_steps, arm, p.backend.tools)
    if teacher is None:
        return None
    tr = teacher[0].numpy()
    rp = present_seq[0].numpy().astype(bool)
    p.predict_states(rec, {**rec.get("cav_controllers", {}), "arm": arm}, horizon_s, 0)
    r = cap["last"]
    n = len(ids)
    pos = np.concatenate([tr[:1, :, 0], r["positions"][0, :, :n].numpy()], 0)
    msk = np.concatenate([rp[:1], r["masks"][0, :, :n].numpy().astype(bool)], 0)
    if "speeds" in r:
        vel = np.concatenate([tr[:1, :, 2], r["speeds"][0, :, :n].numpy()], 0)
    else:
        vel = np.vstack([tr[:1, :, 2], np.diff(pos, axis=0) / DT_S])
    return {"wm_pos": pos, "wm_vel": vel, "wm_msk": msk, "rec_pos": tr[:, :, 0],
            "rec_vel": tr[:, :, 2], "rec_msk": rp, "is_cav": np.asarray(is_cav, bool),
            "ids": list(ids), "t0_s": float(t0)}


# ----------------------------------------------------------------------------- metrics
def discharge(pos, msk, sel, horizon_s: float, end: float = MERGE_END_M) -> int:
    """Vehicles in ``sel`` first seen upstream of ``end`` that reach ``s >= end`` within
    ``horizon_s`` (rows 0..horizon/dt)."""
    rows = int(round(horizon_s / DT_S)) + 1
    pos, msk = pos[:rows], msk[:rows]
    n = 0
    for k in np.flatnonzero(sel):
        seen = np.flatnonzero(msk[:, k])
        if seen.size and pos[seen[0], k] < end and (pos[seen, k] >= end).any():
            n += 1
    return n


def delay_proxy(vel, msk, sel, horizon_s: float, v_ref: float = SPEED_LIMIT_MPS, pos=None,
                end: float = DELAY_END_M) -> float:
    """HDV delay proxy [veh-s]: sum over steps 1..H/dt and vehicles in ``sel`` present at the step
    (and, when ``pos`` is given, with ``s < end``) of ``dt * max(0, 1 - v / v_ref)`` (the
    collector's speed-deficit definition, t0 roster only)."""
    rows = int(round(horizon_s / DT_S)) + 1
    v, m = vel[1:rows], msk[1:rows] & sel[None, :]
    if pos is not None:
        m = m & (pos[1:rows] < end)
    return float(DT_S * np.clip(1.0 - v / v_ref, 0.0, None)[m].sum())


def exit_leaks(wm_pos, rec_msk, sel, horizon_s: float, end: float = DELAY_END_M) -> int:
    """Vehicles in ``sel`` whose recorded SUMO exit (presence is exogenous in the WM rollout)
    removed them from the WM while the WM still had them upstream of ``end``."""
    rows = int(round(horizon_s / DT_S)) + 1
    n = 0
    for k in np.flatnonzero(sel):
        gone = np.flatnonzero(~rec_msk[1:rows, k])
        if gone.size and wm_pos[gone[0], k] < end:     # WM position one row before the exit
            n += 1
    return n


def region_speed(pos, vel, msk, sel, horizon_s: float, lo: float, hi: float) -> float:
    """Vehicle-step mean speed of ``sel`` vehicles with ``lo <= s < hi`` over steps 1..H/dt."""
    rows = int(round(horizon_s / DT_S)) + 1
    p, v, m = pos[1:rows], vel[1:rows], msk[1:rows] & sel[None, :]
    m = m & (p >= lo) & (p < hi)
    return float(v[m].mean()) if m.any() else float("nan")


def snapshot_speed(pos, vel, msk, sel, t_s: float, lo: float, hi: float) -> float:
    i = int(round(t_s / DT_S))
    m = msk[i] & sel & (pos[i] >= lo) & (pos[i] < hi)
    return float(vel[i][m].mean()) if m.any() else float("nan")


# ----------------------------------------------------------------------------- statistics
def bootstrap_mean(values, n_boot: int = 2000, seed: int = 0, groups=None):
    """Mean and 95 % percentile CI; ``groups`` (same length) = cluster labels resampled whole."""
    x = np.asarray(values, float)
    ok = np.isfinite(x)
    if groups is None:
        x = x[ok]
        if x.size == 0:
            return float("nan"), float("nan"), float("nan")
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, x.size, size=(n_boot, x.size))
        bs = x[idx].mean(1)
        return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    g = np.asarray(groups)[ok]
    x = x[ok]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    labels, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=x, minlength=labels.size)
    counts = np.bincount(inv, minlength=labels.size).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, labels.size, size=(n_boot, labels.size))
    bs = sums[pick].sum(1) / counts[pick].sum(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def spearman(a, b) -> float:
    """Spearman rank correlation (average ranks for ties); NaN if fewer than 3 finite pairs or
    a constant vector."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")

    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(x.size)
        r[order] = np.arange(x.size, dtype=float)
        for v in np.unique(x):
            sel = x == v
            r[sel] = r[sel].mean()
        return r

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def fmt_ci(t, nd: int = 2) -> str:
    m, lo, hi = t
    if not np.isfinite(m):
        return "n/a"
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"
