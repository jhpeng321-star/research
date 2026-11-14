"""Part 2 shared library: paths, hashing, part1 imports, topology, and the
corrected RAW pass-through + contract-level aggregate safety metrics.

Read-only on the research-dense source repo (wdpaths.RESEARCH_DENSE_ROOT,
default E:/research-dense).  Zero simulation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wdpaths  # noqa: E402

HERE = Path(__file__).resolve().parent
PART1 = HERE.parent / "part1"
PREDICT = HERE / "predict"
RESULTS = HERE / "results"
DATA = HERE / "data"
SRC = Path(wdpaths.remap(os.environ.get(
    "WD01_SRC", wdpaths.RESEARCH_DENSE_ROOT)))
WM1 = SRC / "output" / "worldmodel_v3_vehicle"
SURV_CKPT = WM1 / "vehicle_dynamics.pt"
SURV_CKPT_SHA256 = ("ded83907b8d218574d3f5359c332408f17167da055574afbe"
                    "29eaecb59d4a8f5")
RETRAINED_PT = PART1 / "identity" / "retrained" / "vehicle_dynamics.pt"
RETRAINED_SHA256 = ("b5836346d0d615f1c0433bcfa87d6b33f48d98e8d729c0117ca"
                    "44e7669042db1")
GAP_MAX = 250.0
HORIZONS = (30, 60, 90, 120)
START_TICKS = tuple(range(120, 481, 60))


# ------------------------------------------------------------- basics ------

def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def device() -> torch.device:
    want = os.environ.get("WD_DEVICE", "").strip().lower()
    if want:
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_part1_module(stem, name):
    """Import a part1 module by explicit path (part1 is never modified)."""
    path = PART1 / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def part1_lib():
    return load_part1_module("wd1lib", "wd01_part1_lib")


@lru_cache(maxsize=1)
def part1_t2():
    sys_path_add(PART1)          # t2_run does `import wd1lib` at module level
    return load_part1_module("t2_run", "wd01_part1_t2")


@lru_cache(maxsize=1)
def part1_wm():
    return part1_lib().load_src_module("wm_common", "wd01_part2_wm")


# ------------------------------------------------ WM-30 physical topology --

@lru_cache(maxsize=1)
def topology():
    """PhysicalTopology from the read-only source (network XML only)."""
    sys_path_add(SRC / "output" / "hybrid_state_v1")
    import topology as T  # noqa: PLC0415 - source module, read-only
    import repo  # noqa: PLC0415
    return T.PhysicalTopology.from_net_xml(repo.NETWORK), T


def sys_path_add(p):
    import sys
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)


IN_JUNCTION = -1
PHYS_LANE_IDS = {"main_left": 0, "main_right": 1, "accel": 2}


def map_vehicle_state(topo_w, lane_id, pos_sumo):
    """(phys_lane_id 0/1/2 or -1 in-junction/unknown, s or nan)."""
    t, _ = topo_w
    name, s = t.lane_of(lane_id, pos_sumo)
    if name is None:
        if str(lane_id).startswith(":"):
            return IN_JUNCTION, float("nan")
        return IN_JUNCTION, float("nan")
    return PHYS_LANE_IDS[name], float(s)


# --------------------------------------- corrected RAW pass-through --------

def corrected_passthrough(d_prev, d_next, g):
    """Corrected RAW pass-through flags for one 1 s step.

    A violation is a follower/leader pair ADJACENT in the d-order at t
    (same lane group; leader immediately ahead: smaller d) whose order is
    inverted at t+1: d_follower(t+1) <= d_leader(t+1).  Flags the follower.

    The defective part-1 variant sorted by CURRENT d and compared along
    that same order, which is false by construction and always returned 0.
    """
    d_prev = np.asarray(d_prev, dtype=np.float64)
    d_next = np.asarray(d_next, dtype=np.float64)
    g = np.asarray(g)
    flags = np.zeros(len(d_prev), dtype=bool)
    for grp in np.unique(g):
        members = np.where(g == grp)[0]
        order = members[np.argsort(d_prev[members], kind="stable")]
        for j in range(1, len(order)):
            i, lead = int(order[j]), int(order[j - 1])
            if d_next[i] <= d_next[lead]:
                flags[i] = True
    return flags


# ------------------------------------ contract-level safety aggregates ----

def closing_ttc(d, v, g, w):
    """(lt1p5 idx set, lt3 idx set, min closing TTC, per-vehicle TTC array).

    In-group leader from the snapshot's own d-order (wm_common
    leader_features; free iff no leader within GAP_MAX).  closing =
    v_self - v_leader > 0; TTC = gap / closing.  ttc[i] is nan where the
    vehicle is free or not closing; min_ttc is inf when nobody is closing
    (caller serialises as null).
    """
    v_lead, gap, free = w.leader_features(d, v, g)
    closing = v - v_lead
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = gap / closing
    ok = (free == 0) & (closing > 0.0)
    ttc = np.where(ok, ttc, np.nan)
    lt15 = {int(i) for i in np.where(ok & (ttc < 1.5))[0]}
    lt3 = {int(i) for i in np.where(ok & (ttc < 3.0))[0]}
    min_ttc = float(np.nanmin(ttc)) if np.isfinite(ttc).any() else float("inf")
    return lt15, lt3, min_ttc, ttc


CONTINUOUS_KEYS = ("ttc_lt3_exposure_s", "ttc_lt1p5_exposure_s",
                   "hard_brake_s", "min_ttc_s")
BOOLEAN_KEYS = ("any_ttc_lt1p5", "any_ttc_lt3", "any_hard_brake")
# correlations registered in REGISTRATION_ADDENDUM.md section 3
REGISTERED_CORR_KEYS = ("ttc_lt3_exposure_s", "hard_brake_s", "min_ttc_s")


def detection_table(rows):
    """Per (H, variant) pooled diagnostics from per-eval rows (truth vs pred).

    Window-level recall/precision for the three booleans; for each continuous
    metric: n, mean bias (pred - truth), fraction of evals with pred >= truth
    (exposure-type metrics only), Pearson and Spearman (registered keys over
    evals where both sides are finite for min_ttc).
    """
    out = {}
    for key in BOOLEAN_KEYS:
        t = [r[f"truth_{key}"] for r in rows]
        p = [r[f"pred_{key}"] for r in rows]
        tp = sum(1 for a, b in zip(t, p) if a and b)
        fn = sum(1 for a, b in zip(t, p) if a and not b)
        fp = sum(1 for a, b in zip(t, p) if not a and b)
        out[key] = dict(
            n_evals=len(rows), truth_pos=sum(t), pred_pos=sum(p),
            recall=(tp / (tp + fn)) if (tp + fn) else None,
            precision=(tp / (tp + fp)) if (tp + fp) else None)
    for key in CONTINUOUS_KEYS:
        t_raw = [r[f"truth_{key}"] for r in rows]
        p_raw = [r[f"pred_{key}"] for r in rows]
        if key == "min_ttc_s":
            pair = [(a, b) for a, b in zip(t_raw, p_raw)
                    if a is not None and b is not None]
            tt = np.array([a for a, _ in pair], dtype=np.float64)
            pp = np.array([b for _, b in pair], dtype=np.float64)
        else:
            tt = np.array([float(x) for x in t_raw], dtype=np.float64)
            pp = np.array([float(x) for x in p_raw], dtype=np.float64)
        entry = dict(n=len(tt),
                     mean_bias=float(np.mean(pp - tt)) if len(tt) else None)
        if key != "min_ttc_s":
            entry["frac_pred_ge_truth"] = (float(np.mean(pp >= tt))
                                           if len(tt) else None)
        if len(tt) > 2 and np.std(tt) > 0 and np.std(pp) > 0:
            entry["pearson"] = float(np.corrcoef(tt, pp)[0, 1])
            entry["spearman"] = _spearman(tt, pp)
        else:
            entry["pearson"] = entry["spearman"] = None
        out[key] = entry
    return out


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])
