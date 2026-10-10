"""WD-01 part 3 library: NATIVE Part A v0 model, rollouts, baselines, metrics.

Zero simulation.  Data: the p2C dataset (output/wd01/partA_data, read-only)
via its loader.  Device-agnostic torch (env WD_DEVICE, default
cuda-if-available); tensors created with device=.

Integrator (MEASURED on 405k real same-lane pairs across 4 roots,
tests/test_integrator_form.py): SUMO advances position by the NEW speed,
s[t+1] = s[t] + v[t+1]*dt (p50 = p95 = p99 residual exactly 0.0).  The
model uses the same form.

Model world: junction rows are eliminated by construction -- the physical
lanes span the junctions, so a crossing vehicle keeps its lane and s
continues (the documented 'map junction steps to the downstream lane' for
modelling).  On the truth side of evaluations, junction rows are
retrospectively assigned the vehicle's next physical lane (documented).
vType is SUPERVISION ONLY, never an input (tested); is_cav is an input.
NO pass-through repair: per-lane order inversions after a step are
PREDICTED COLLISIONS, counted, never repaired.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
P2C = HERE.parent / "partA_data"
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
# Untracked data/weights root override (part 3k): WD01_DATA_ROOT points at
# a populated copy of output/wd01; unset keeps the in-repo layout.
_DR = os.environ.get("WD01_DATA_ROOT", "").strip()
if _DR:
    RESULTS = Path(_DR) / "part3" / "results"
    WEIGHTS = Path(_DR) / "part3" / "weights"
sys.path.insert(0, str(P2C))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))            # output/wd01 (wdpaths)

import wdpaths  # noqa: E402
import pdl as PD  # noqa: E402  (part2C library, read-only)
import loader as LD  # noqa: E402

DT = 0.5
CKPT = Path(wdpaths.RESEARCH_DENSE_ROOT) / "output" / \
    "worldmodel_v3_vehicle" / "vehicle_dynamics.pt"
CKPT_SHA256 = ("ded83907b8d218574d3f5359c332408f17167da055574afbe"
               "29eaecb59d4a8f5")
P2C_MANIFEST_SHA256 = "2439004d528a860c701bfb147afcda99ab3f864a29e6187be721db81cc82dd27"
LANE_LIMIT = (33.79, 33.79, 24.0)              # m/s per lane 0/1/2
ACCEL_LEN = 454.10
MERGE_ZONE_S0 = 378.66
HORIZONS = (30, 60, 90, 120)
EVAL_T0S = tuple(range(120, 481, 30))
ROLL_TICKS = 240                               # 120 s at 0.5 s
N_HIST = 20                                    # <= 10 s history ticks
GAP_MAX = 250.0
OFFSETS = (2645.67, 2644.22, 378.66)           # measured, test-anchored
SLOT_NAMES = ("own_leader", "own_follower", "left_leader", "left_follower",
              "right_leader", "right_follower")
Z_DIM = 8
RES_SCALE = 0.5   # residual authority, m/s^2 (3c: reduced from 2 after closed-loop saturation diagnosis)

# SUMO vType params from the regenerated requests.rou.xml (B3 oracle ONLY)
VTYPES = {
    0: dict(tau=0.95, a_max=3.0, b=4.5, s0=1.5, sf=1.08),   # aggressive
    1: dict(tau=1.1, a_max=2.6, b=4.5, s0=2.5, sf=1.00),    # normal
    2: dict(tau=1.5, a_max=2.0, b=3.0, s0=3.5, sf=0.92),    # cautious
}
ADJ_MASK = torch.tensor([[1.0, 1.0, 0.0],
                         [1.0, 1.0, 1.0],
                         [0.0, 1.0, 1.0]])   # rows: from lane; stay allowed


def device():
    want = os.environ.get("WD_DEVICE", "").strip().lower()
    if want:
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str),
                          encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def guards():
    assert sha256_file(CKPT) == CKPT_SHA256, "surviving checkpoint changed"
    man_sha = sha256_file(P2C / "MANIFEST.json")
    assert man_sha == P2C_MANIFEST_SHA256, "p2C MANIFEST changed"
    man = load_json(P2C / "MANIFEST.json")
    return man, man_sha


@lru_cache(maxsize=1)
def lane_lengths():
    topo, _, _ = PD.topology()
    return {PD.LANE_IDS[n]: float(topo.length_of(n)) for n in PD.LANE_IDS}


@lru_cache(maxsize=1)
def lane_lengths():
    topo, _, _ = PD.topology()
    return {PD.LANE_IDS[n]: float(topo.length_of(n)) for n in PD.LANE_IDS}


def roots_by_split():
    """Deterministic splits.  VALIDATION = 4 of the 21 train roots chosen
    by loading class parsed from the root name: the two median elements of
    the m2000 class, the first of m2400, the median element of m2700.
    Frozen rule, no randomness; holdout untouched until the registered
    evaluation."""
    man, _ = guards()
    train, holdout = [], []
    for name, r in man["roots"].items():
        (holdout if r["split"] == "holdout" else train).append(name)
    train.sort()
    holdout.sort()

    def cls(n):
        for c in ("m2400", "m2000", "m2700"):
            if c in n:
                return c
        return "other"

    classes = defaultdict(list)
    for n in train:
        classes[cls(n)].append(n)
    val = sorted({classes["m2000"][len(classes["m2000"]) // 2],
                  classes["m2000"][(len(classes["m2000"]) + 1) // 2],
                  classes["m2400"][0],
                  classes["m2700"][len(classes["m2700"]) // 2]})
    assert len(val) == 4 and set(val) <= set(train)
    tr = [n for n in train if n not in val]
    return dict(train=tr, val=val, holdout=holdout)


# --------------------------------------------------------------- physics ----

def idm_acc(v, gap, dv, a_max, b, tau, s0, v0):
    """IDM acceleration (torch).  gap >= 0.3; dv = v - v_leader;
    free driving approximated by gap = 250 (interaction ~ 0)."""
    gap = gap.clamp(min=0.3)
    s_star = (s0 + (v * tau + v * dv / (2.0 * (a_max * b).sqrt()))
              .clamp(min=0.0))
    return a_max * (1.0 - (v / v0).clamp(max=4.0) ** 4
                    - (s_star / gap) ** 2)


def bounded(x, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(x)


HIST_FEATS = 11
EGO_DIM = 8


def hist_features(hist):
    """hist [n, 20, 3] (v, a, ds; zero-padded at the old end) -> [n, 11]."""
    v, a = hist[..., 0], hist[..., 1]
    return torch.stack([v[:, -1], a[:, -1], v.mean(1), v.std(1), a.mean(1),
                        a.std(1), v.min(1).values, v.max(1).values,
                        v[:, -4:].mean(1), a[:, -4:].mean(1),
                        v[:, -1] - v[:, 0]], dim=1)


def prior_from_raw(arr):
    """[n,5] raw (lane, v, gap, rel, cav) -> [n,7] prior input."""
    n = arr.shape[0]
    oneh = torch.zeros(n, 3, device=arr.device)
    oneh[torch.arange(n, device=arr.device), arr[:, 0].long()] = 1.0
    return torch.cat([oneh, arr[:, 1:2] / 40.0, arr[:, 2:3] / 250.0,
                      arr[:, 3:4] / 30.0, arr[:, 4:5]], dim=1)


def ego_features(lane, s, v, a_prev, cav):
    dev = v.device
    oneh = torch.zeros(len(v), 3, device=dev)
    oneh[torch.arange(len(v), device=dev), lane] = 1.0
    dist_end = ((ACCEL_LEN - s) / 500.0).clamp(0.0, 1.0)
    dist_end = torch.where(lane == 2, dist_end, torch.zeros_like(s))
    merge = ((lane == 2) & (s >= MERGE_ZONE_S0)).float().unsqueeze(1)
    return torch.cat([v.unsqueeze(1) / 40.0, a_prev.unsqueeze(1) / 9.0,
                      oneh, dist_end.unsqueeze(1), merge,
                      cav.unsqueeze(1)], dim=1)


def neighbour_slots(lane, s, v, cav):
    """numpy [n] -> [n, 6, 4] float32 (present, gap, rel_v, is_cav):
    own/left/right lane leader+follower, adjacency 0<->1<->2 (accel is
    never adjacent to main_left), d-gap capped at 250 m."""
    n = len(lane)
    off = np.array([OFFSETS[0], OFFSETS[1], OFFSETS[2]])
    d = off[lane] - s
    out = np.zeros((n, 6, 4), dtype=np.float32)
    out[:, :, 1] = GAP_MAX
    col = {nm: j for j, nm in enumerate(SLOT_NAMES)}
    by = {li: np.where(lane == li)[0] for li in (0, 1, 2)}
    adj = {0: (None, 1), 1: (0, 2), 2: (1, None)}

    def put(i, c, j):
        raw = abs(d[i] - d[j])
        if raw > GAP_MAX:
            return          # leader beyond 250 m: FREE (present = 0)
        out[i, c, 0] = 1.0
        out[i, c, 1] = raw
        out[i, c, 2] = v[j] - v[i]
        out[i, c, 3] = cav[j]

    for i in range(n):
        li = lane[i]
        same = by[li]
        if len(same):
            o = same[np.argsort(d[same])]
            p = int(np.searchsorted(d[o], d[i]))
            if p > 0:
                put(i, col["own_leader"], o[p - 1])
            if p < len(o) - 1:
                put(i, col["own_follower"], o[p + 1])
        for side, other in zip(("left", "right"), adj[li]):
            if other is None or not len(by[other]):
                continue
            ot = by[other]
            do = d[ot]
            am = do <= d[i]
            if am.any():
                put(i, col[f"{side}_leader"], ot[am][np.argmax(do[am])])
            if (~am).any():
                put(i, col[f"{side}_follower"], ot[~am][np.argmin(do[~am])])
    return out


# ----------------------------------------------------------------- model ----

class PartAModel(nn.Module):
    """z-encoder (from <=10 s history) + learned prior (newborns),
    bounded IDM theta(z), residual + lane head + 2 aux heads."""

    def __init__(self, hidden=128):
        super().__init__()
        self.z_enc = nn.Sequential(nn.Linear(HIST_FEATS, 64), nn.SiLU(),
                                   nn.Linear(64, Z_DIM), nn.Tanh())
        self.z_prior = nn.Sequential(nn.Linear(7, 64), nn.SiLU(),
                                     nn.Linear(64, Z_DIM), nn.Tanh())
        self.theta = nn.Sequential(nn.Linear(Z_DIM, 32), nn.SiLU(),
                                   nn.Linear(32, 5))
        in_dim = EGO_DIM + 6 * 4 + Z_DIM
        self.res = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 6))   # da + lc3 + aux2

    @staticmethod
    def prior_input(lane, v, gap, rel, cav):
        oneh = torch.zeros(len(v), 3, device=v.device)
        oneh[torch.arange(len(v), device=v.device), lane] = 1.0
        return torch.cat([oneh, (v / 40.0).unsqueeze(1),
                          (gap / GAP_MAX).unsqueeze(1),
                          (rel / 30.0).unsqueeze(1),
                          cav.unsqueeze(1)], dim=1)

    def z_of(self, hist, prior_in):
        """hist [n,20,3] (may be all-zero -> prior), prior_in [n,7]."""
        has = hist.abs().sum((1, 2)) > 0
        z = torch.where(has.unsqueeze(1), self.z_enc(hist_features(hist)),
                        self.z_prior(prior_in))
        return z

    def theta_of(self, z):
        p = self.theta(z)
        return dict(a_max=bounded(p[:, 0], 2.0, 3.0),
                    b=bounded(p[:, 1], 3.0, 4.5),
                    tau=bounded(p[:, 2], 0.95, 1.5),
                    s0=bounded(p[:, 3], 1.5, 3.5),
                    sf=bounded(p[:, 4], 0.92, 1.08))

    def forward(self, ego, slots, z):
        x = torch.cat([ego, slots.reshape(len(ego), -1), z], dim=1)
        out = self.res(x)
        return out[:, 0], out[:, 1:4], out[:, 4:]   # da, lc_logits, aux


# ------------------------------------------------------- root model view ----

class RootArrays:
    """Per-root model view of the p2C arrays.

    lane_m: junction rows retrospectively assigned the vehicle's next
    physical lane (documented); junction rows keep in_junction=True for
    truth-side exclusion of lane/TTC checks.  Only rows with a resolvable
    downstream lane (lane_m >= 0) participate in the model world.
    """

    def __init__(self, root, registration_sha256=None):
        from holdout_block import assert_holdout_allowed
        assert_holdout_allowed(root, registration_sha256)
        z = LD.load_root(root)
        self.root = root
        self.vid = z["vid"].astype(np.int64)
        self.tick = z["tick"].astype(np.int64)
        self.lane = z["lane"].astype(np.int64)
        self.s = z["s"].astype(np.float64)
        self.v = z["v"].astype(np.float64)
        self.a = z["a"].astype(np.float64)
        self.cav = z["is_cav"].astype(np.float64)
        self.typ = z["type_label"].astype(np.int64)   # SUPERVISION ONLY
        self.inj = z["in_junction"].astype(bool)
        n = len(self.vid)
        rows = defaultdict(list)
        for i in range(n):
            rows[int(self.vid[i])].append(i)
        lane_m = self.lane.copy()
        for vi, idxs in rows.items():
            idxs.sort(key=lambda i: self.tick[i])
            nxt = -1
            for i in reversed(idxs):
                if self.lane[i] >= 0:
                    nxt = int(self.lane[i])
                elif nxt >= 0:
                    lane_m[i] = nxt
        keep = lane_m >= 0
        self.vid, self.tick, self.lane_m = self.vid[keep], self.tick[keep], \
            lane_m[keep]
        self.s, self.v, self.a = self.s[keep], self.v[keep], self.a[keep]
        self.cav, self.typ, self.inj = (self.cav[keep], self.typ[keep],
                                        self.inj[keep])
        self.n = len(self.vid)
        rows2 = defaultdict(list)
        for i in range(self.n):
            rows2[int(self.vid[i])].append(i)
        for vi in rows2:
            rows2[vi].sort(key=lambda i: self.tick[i])
        self._rows = rows2
        self._by_tick = defaultdict(list)
        self._rowmap = {}
        for i in range(self.n):
            self._rowmap[(int(self.vid[i]), int(self.tick[i]))] = i
        for i, t in enumerate(self.tick):
            self._by_tick[int(t)].append(i)
        # departed births (model-world insertions)
        self.b_tick = z["b_tick"].astype(np.int64)
        self.b_lane = z["b_lane"].astype(np.int64)
        self.b_s = z["b_s"].astype(np.float64)
        self.b_v = z["b_v"].astype(np.float64)
        self.b_dep = z["b_departed"].astype(bool)
        self.b_vid = z["b_vid"].astype(np.int64)
        # truth lane changes (declared, from p2C events)
        self.lc_vid = z["lc_vid"].astype(np.int64)
        self.lc_tick = z["lc_tick"].astype(np.int64)
        self.lc_from = z["lc_from"].astype(np.int8)   # part 3k (fit stats)
        self.lc_to = z["lc_to"].astype(np.int8)
        self.lc_s = z["lc_s"].astype(np.float64)
        # contract-format arrivals
        self.arr_tick = z["ar_tick"].astype(np.int64)
        self.arr_entrance = z["ar_entrance"].astype(np.int64)
        self.arr_v = z["ar_v"].astype(np.float64)
        self.arr_cav = z["ar_cav"].astype(np.int64)
        self.arr_type = z["ar_type"].astype(np.int64)
        self.vid_table = z["vid_table"]
        self._rowmap0 = {}
        for i in range(len(z["vid"])):
            self._rowmap0[(int(z["vid"][i]), int(z["tick"][i]))] = i
        self.b_cav = np.array([
            float(z["is_cav"][self._rowmap0[(int(v), int(t))]])
            if (int(v), int(t)) in self._rowmap0 else 0.0
            for v, t in zip(self.b_vid, self.b_tick)])

    def tick_rows(self, k):
        return np.asarray(self._by_tick.get(int(k), ()), dtype=np.int64)

    def rows_of(self, vi):
        return self._rows.get(int(vi), ())


def cav_at(ra, vid, tick):
    i = ra._rowmap.get((int(vid), int(tick)))
    return float(ra.cav[i]) if i is not None else 0.0


@lru_cache(maxsize=8)
def root_arrays(root, registration_sha256=None):
    return RootArrays(root, registration_sha256)


def history_at(ra, k0):
    """Truth history for every vehicle alive at tick k0:
    vid -> [N_HIST, 3] (v, a, ds; zero-padded at the old end)."""
    out = {}
    for vi, idxs in ra._rows.items():
        upto = [i for i in idxs if ra.tick[i] <= k0]
        if not upto:
            continue
        h = np.zeros((N_HIST, 3))
        h[-1] = (ra.v[upto[-1]], ra.a[upto[-1]], 0.0)
        j, p = N_HIST - 2, len(upto) - 2
        cur = upto[-1]
        while j >= 0 and p >= 0:
            i = upto[p]
            gap_t = ra.tick[cur] - ra.tick[i]
            if gap_t != (N_HIST - 1 - j):
                break
            h[j] = (ra.v[i], ra.a[i], max(ra.s[cur] - ra.s[i], 0.0))
            cur, j, p = i, j - 1, p - 1
        out[vi] = h
    return out
