"""Shared library for WD-01 part 1 (world-model identity + rollout drift).

All writes stay under output/wd01/part1/.  The source repo (wdpaths.
RESEARCH_DENSE_ROOT, default E:/research-dense) is opened read-only.
No SUMO/TraCI anywhere.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wdpaths  # noqa: E402

# ---------------------------------------------------------------- paths -----
SRC = Path(wdpaths.remap(os.environ.get(
    "WD01_SRC", wdpaths.RESEARCH_DENSE_ROOT)))
WM = SRC / "output" / "worldmodel_v3_vehicle"
RUNS = SRC / "runs"
REGEN = RUNS / "regenerated_v1"
RETRY = RUNS / "regenerated_v1_retry1"
EVID = SRC / "output" / "evidence_regeneration_v1"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
IDENTITY = HERE / "identity"
SURV_CKPT = WM / "vehicle_dynamics.pt"          # pinned, never modified
SURV_CKPT_SHA256 = ("ded83907b8d218574d3f5359c332408f17167da055574afbe"
                    "29eaecb59d4a8f5")
ENSEMBLE = SRC / "output" / "worldmodel_dynamics_v1" / "dynamics_ensemble.pt"
ENSEMBLE_SHA256 = ("d017d17e7d5ae6f722f2c78c702bc715f09b91631861749b45ce"
                   "5e2d0802a934")

HORIZONS = (30, 60, 90, 120)
START_TICKS = tuple(range(120, 481, 60))        # 120..480 step 60
GAP_MAX = 250.0

# ------------------------------------------------------------- helpers ------

def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def device() -> torch.device:
    """Resolve the torch device once (env WD01_DEVICE > cuda-if-ok > cpu)."""
    want = os.environ.get("WD01_DEVICE", "").strip().lower()
    if want:
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_tag(tag, retry_root, regen_root):
    """Resolve a cvNN_<job> tag to a regenerated run dir; retry1 wins if its
    directory exists.  Returns (pref, run_dir) or (None, None)."""
    m = re.match(r"^cv(\d+)_(.+)$", tag)
    if not m:
        return None, None
    for root, pref in ((Path(retry_root), "retry1"), (Path(regen_root), "regen1")):
        cand = root / f"cv{m.group(1)}" / m.group(2) / "run"
        if cand.is_dir():
            return pref, cand
    return None, None


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------- source module loading ----

def load_src_module(stem, name=None):
    """Import a module from the read-only source dir by explicit path."""
    path = WM / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(name or stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wm():
    return load_src_module("wm_common", name="wd01_wm_common")


def train_mod():
    return load_src_module("train_vehicle_dynamics", name="wd01_train_mod")


# --------------------------------------------------------- state caching ----

def states_to_cache(run_dir, out_path, w=None):
    """Extract per-tick whole-second states and store compact npz.

    Columns: tick (int32 per vehicle), d (f64), v (f32), g (int8), vid (str).
    Raises on missing ticks exactly like wm_common.load_tick_states.
    """
    w = w or wm()
    offsets, _ = w.fit_lane_offsets(run_dir)
    states, diag = w.load_tick_states(run_dir, offsets)
    ticks, dsv, vsv, gsv, vidsv = [], [], [], [], []
    for t in sorted(states):
        snap = states[t]
        for vid in sorted(snap):
            d, v, g = snap[vid]
            ticks.append(t)
            dsv.append(d)
            vsv.append(v)
            gsv.append(g)
            vidsv.append(str(vid))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        tick=np.asarray(ticks, dtype=np.int32),
        d=np.asarray(dsv, dtype=np.float64),
        v=np.asarray(vsv, dtype=np.float64),
        g=np.asarray(gsv, dtype=np.int8),
        vid=np.asarray(vidsv),
        diag_missing=np.asarray(diag["missing_ticks"], dtype=np.int32),
        n_offsets=np.int64(len(offsets)),
    )
    return dict(n_ticks=len(states), n_offsets=len(offsets),
                missing=diag["missing_ticks"])


def cache_to_states(path):
    """Inverse of states_to_cache: {tick: {vid: (d, v, g)}}."""
    z = np.load(path, allow_pickle=False)
    tick, d, v, g, vid = z["tick"], z["d"], z["v"], z["g"], z["vid"]
    states = {}
    for i in range(len(tick)):
        states.setdefault(int(tick[i]), {})[str(vid[i])] = (
            float(d[i]), float(v[i]), int(g[i]))
    return states


# ------------------------------------------------------------- rollouts -----

def raw_propagate(d, v, dv, clamp=True):
    """Original v3 integrator: v' = max(v + dv, 0) [if clamp]; d' = d - (v+v')/2.

    clamp=False is the diagnostics variant used by the source verify script.
    """
    v_raw = v + dv
    v_next = np.maximum(v_raw, 0.0) if clamp else v_raw
    return d - (v + v_next) / 2.0, v_next, v_raw


def constrained_propagate(d, v, g, dv, v_cap, dt=1.0,
                          a_min=-9.0, a_max=3.0, back_gap=5.0):
    """Hard-constraint integrator (DESIGN Part A 2.3).

    Per lane group, front-to-back (d ascending): a = clip(dv, a_min, a_max);
    v' = clip(v + a*dt, 0, v_cap[g]); d' = d - (v + v')/2 * dt.  If d' would
    pass the in-lane leader's new position (d'_self <= d'_leader), flag an
    overlap and place the follower at d'_leader + back_gap (5 m behind).
    Overlaps are counted, not hidden.

    Returns (d_next, v_next, overlap_flags).
    """
    d = np.asarray(d, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    g = np.asarray(g)
    dv = np.asarray(dv, dtype=np.float64)
    a = np.clip(dv, a_min, a_max)
    v_next = np.clip(v + a * dt, 0.0, np.asarray(v_cap, dtype=np.float64)[g])
    d_next = d - (v + v_next) / 2.0 * dt
    overlap = np.zeros(len(d), dtype=bool)
    for grp in np.unique(g):
        idx = np.argsort(d[g == grp], kind="stable")     # front (small d) first
        members = np.where(g == grp)[0][idx]
        for j in range(1, len(members)):
            i, lead = int(members[j]), int(members[j - 1])
            if d_next[i] <= d_next[lead]:
                overlap[i] = True
                d_next[i] = d_next[lead] + back_gap
    return d_next, v_next, overlap


# ----------------------------------------------------------- event logic ----

def follower_leader(snap):
    """Per-tick follower->leader map within lane groups.

    Returns {vid: (leader_vid, gap)} for vehicles with an in-group leader at
    gap <= GAP_MAX (free drivers omitted).
    """
    by_group = {}
    for vid, (d, v, g) in snap.items():
        by_group.setdefault(g, []).append((d, v, vid))
    out = {}
    for grp, rows in by_group.items():
        rows.sort(key=lambda r: -r[0])                   # rearmost first
        for j in range(len(rows) - 1):
            d_self, v_self, vid = rows[j]
            d_lead, v_lead, vid_lead = rows[j + 1]
            gap = d_self - d_lead
            if np.isfinite(gap) and 0.0 <= gap <= GAP_MAX:
                out[vid] = (vid_lead, gap)
    return out


def ttc_events_at(snap):
    """Set of (vid,) with closing TTC < 3 s at this tick (per vehicle-second).

    closing TTC = gap / (v_follow - v_leader), needs v_follow > v_leader and
    an in-group leader (free-driving vehicles have no leader -> no event).
    """
    out = set()
    for vid, (vid_lead, gap) in follower_leader(snap).items():
        vf, vl = snap[vid][1], snap[vid_lead][1]
        closing = vf - vl
        if closing > 0.0 and gap / closing < 3.0:
            out.add(vid)
    return out


def brake_events(states, t0, t1):
    """Set of (vid,) with v(t+1) - v(t) < -3 m/s over 1 s, t in [t0, t1).

    Vehicle must be present at t and t+1 (lane group change allowed).
    """
    out = set()
    for t in range(t0, t1):
        cur, nxt = states.get(t), states.get(t + 1)
        if not cur or not nxt:
            continue
        for vid, (d, v, g) in cur.items():
            nx = nxt.get(vid)
            if nx is not None and nx[1] - v < -3.0:
                out.add(vid)
    return out


def match_events(truth_events, pred_events, tol=2):
    """Match vehicle-second events by vehicle id within +-tol ticks.

    Each truth event counts at most one matched pred event (1-1 greedy by
    smallest |dt|).  Returns (n_truth_matched, n_truth, n_pred_matched,
    n_pred); recall/precision derived from these.
    """
    truth_by_vid = {}
    for vid, t in truth_events:
        truth_by_vid.setdefault(vid, []).append(t)
    pred_by_vid = {}
    for vid, t in pred_events:
        pred_by_vid.setdefault(vid, []).append(t)
    n_truth = len(truth_events)
    n_pred = len(pred_events)
    n_truth_matched = 0
    n_pred_matched = 0
    for vid, tlist in truth_by_vid.items():
        plist = pred_by_vid.get(vid)
        if not plist:
            continue
        pairs = sorted((abs(t - p), t, p) for t in tlist for p in plist)
        used_t, used_p = set(), set()
        for dt, t, p in pairs:
            if dt > tol:
                break
            if t in used_t or p in used_p:
                continue
            used_t.add(t)
            used_p.add(p)
            n_truth_matched += 1
            n_pred_matched += 1
    return n_truth_matched, n_truth, n_pred_matched, n_pred


# -------------------------------------------------------- modern ckpt IO ----

def save_modern_ckpt(path, model, metadata):
    """State_dict + plain python metadata -> torch.load(weights_only=True)."""
    payload = {"model_state": {k: v.detach().cpu() for k, v in
                               model.state_dict().items()},
               "meta": _to_plain(metadata)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _to_plain(obj):
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_plain(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def load_modern_ckpt(path):
    """weights_only=True load; returns (state_dict, meta)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["model_state"], payload["meta"]
