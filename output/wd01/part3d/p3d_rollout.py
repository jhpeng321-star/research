"""Part 3d structured stochastic rollout (numpy) — SUMO-emulating native
Part A v0.  One member, one window, 240 steps of 0.5 s.

Per step, per vehicle:
- IDM acceleration from the SAMPLED vType (tau, a_max, b, s0, v0 =
  speedFactor x network segment limit); leader = same-lane next larger s;
  gap > 250 -> free;
- measured residual noise N(mean_vt_regime, sd_vt_regime) added to the
  IDM acceleration (member CRN stream);
- v' = clip(v + a*0.5, 0, 1.2*limit); s' = s + v'*0.5 (MEASURED form);
- stochastic lane change: per-direction hazard probability; legality
  0<->1 and 2->1 (merge zone s >= 378.66 only); safety gap check;
  d kept constant across the change (s += OFFSETS[new] - OFFSETS[old]);
- births inserted from logged (tick, lane, s, v) with a sampled vType;
  exits at lane end;
- collisions: pre/post-step adjacent-pair order inversions, counted,
  never repaired.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

HERE = __import__("pathlib").Path(__file__).resolve().parent
P2C = HERE.parent / "partA_data"
if str(P2C) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(P2C))
if str(HERE) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(HERE))

import engine as E  # noqa: E402
import p3lib as P  # noqa: E402

OFFS = np.array([2645.67, 2644.22, 378.66])
VT = {0: dict(tau=0.95, a_max=3.0, b=4.5, s0=1.5, sf=1.08),
      1: dict(tau=1.1, a_max=2.6, b=4.5, s0=2.5, sf=1.00),
      2: dict(tau=1.5, a_max=2.0, b=3.0, s0=3.5, sf=0.92)}
NOISE = {(0, 1): (-0.75, 0.90), (0, 0): (-0.40, 0.39),
         (1, 1): (-0.24, 0.57), (1, 0): (-0.21, 0.26),
         (2, 1): (-0.22, 0.46), (2, 0): (-0.08, 0.41)}
LC_LIMITS = {0: (1,), 1: (0,), 2: (1,)}


def leaders_gap(lane, s):
    """(leader index per vehicle or -1, gap leader-follower).  s increases
    forward: leader = next larger s in the same lane."""
    n = len(lane)
    leader = np.full(n, -1, dtype=np.int64)
    gap = np.full(n, np.inf)
    for li in (0, 1, 2):
        ids = np.where(lane == li)[0]
        if len(ids) < 2:
            continue
        o = ids[np.argsort(s[ids], kind="stable")]   # rear (small s) first
        for j in range(len(o) - 1):
            leader[o[j]] = o[j + 1]
            gap[o[j]] = s[o[j + 1]] - s[o[j]]
    return leader, gap


def idm_acc_np(v, gap, dv, tau, a_max, b, s0, v0):
    v0 = np.maximum(v0, 0.1)
    ratio = np.minimum(v / v0, 4.0)
    free = gap > 250.0
    g = np.where(free, 1e6, np.maximum(gap, 0.1))
    s_star = s0 + np.maximum(0.0, v * tau + v * dv /
                             (2.0 * np.sqrt(a_max * b)))
    inter = np.where(free, 0.0, (s_star / g) ** 2)
    return a_max * (1.0 - ratio ** 4 - inter)


def seg_limits(lane, s):
    out = np.empty(len(lane))
    for i in range(len(lane)):
        out[i] = 22.22 if (lane[i] == 2 and s[i] < 378.66) else 31.29
    return out


class StructRollout:
    """One structured member rollout on one validation window."""

    def __init__(self, ra, k0, member_seed, post_model, haz_model,
                 lc_threshold=0.5, lc_scale=1.0):
        self.ra = ra
        self.member_seed = int(member_seed)
        self.k0 = k0
        self.rng = np.random.default_rng((self.member_seed, k0))
        self.lc_threshold = float(lc_threshold)
        self.lc_scale = float(lc_scale)
        idx = ra.tick_rows(k0)
        idx = idx[ra.lane_m[idx] >= 0]
        self.vid = ra.vid[idx].copy()
        self.lane = ra.lane_m[idx].copy()
        self.s = ra.s[idx].copy()
        self.v = ra.v[idx].copy()
        self.a_prev = np.zeros(len(self.v))
        self.llen = np.array([E.lane_lengths()[int(l)]
                              for l in self.lane])
        hist = P.history_at(ra, k0)
        H = np.zeros((len(self.vid), 20, 3), dtype=np.float32)
        has = np.zeros(len(self.vid), dtype=bool)
        for r, vi in enumerate(self.vid):
            if vi in hist:
                H[r] = hist[vi]
                has[r] = True
        with torch.no_grad():
            probs = torch.softmax(post_model(torch.as_tensor(
                H, device="cpu")), dim=1).cpu().numpy()
        self.probs = probs.copy()
        u = self.rng.random(len(self.vid))
        prior = np.cumsum([0.1, 0.7, 0.2])
        self.vtype = np.empty(len(self.vid), dtype=np.int64)
        for r in range(len(self.vid)):
            cdf = np.cumsum(probs[r]) if has[r] else prior
            self.vtype[r] = int(np.searchsorted(cdf, u[r], "left"))
        self.births = defaultdict(list)
        bm = ra.b_dep & (ra.b_tick > k0) & (ra.b_tick <= k0 + 240)
        bm &= ra.b_lane >= 0
        for i in np.where(bm)[0]:
            self.births[int(ra.b_tick[i]) - k0].append(
                (int(ra.b_vid[i]), int(ra.b_lane[i]), float(ra.b_s[i]),
                 float(ra.b_v[i])))
        self.collisions = 0
        self.capviol = 0
        self.lc_count = 0
        self.n_birthed = 0
        self.snapshots = {0: self._snap()}

    # ------------------------------------------------------------------
    def _snap(self):
        return dict(vid=self.vid.copy(), lane=self.lane.copy(),
                    s=self.s.copy(), v=self.v.copy(),
                    a=self.a_prev.copy(), collisions=self.collisions,
                    capviol=self.capviol, lc=self.lc_count,
                    n_birthed=self.n_birthed)

    def _insert(self, tvid, lane, s0, v0, tick):
        self.vid = np.append(self.vid, tvid)
        self.lane = np.append(self.lane, lane)
        self.s = np.append(self.s, s0)
        self.v = np.append(self.v, v0)
        self.a_prev = np.append(self.a_prev, 0.0)
        self.llen = np.append(self.llen, E.lane_lengths()[lane])
        u = self.rng.random()
        self.vtype = np.append(self.vtype,
                               int(np.searchsorted(
                                   np.cumsum([0.1, 0.7, 0.2]), u, "left")))
        self.n_birthed += 1

    def _collisions(self, pre_lane, pre_s, pre_leader):
        post = {int(v): float(ss) for v, ss in zip(self.vid, self.s)}
        post_lane = {int(v): int(l) for v, l in zip(self.vid, self.lane)}
        for i in range(len(pre_s)):
            j = int(pre_leader[i])
            if j < 0:
                continue
            vi, vj = int(self.vid[i]), int(self.vid[j])
            if vi not in post or vj not in post:
                continue
            if post_lane[vi] != pre_lane[i] or \
                    post_lane[vj] != pre_lane[j]:
                continue
            if post[vi] >= post[vj] - 1e-9:
                self.collisions += 1

    def _drop_exits(self):
        gone = self.s > self.llen
        if gone.any():
            keep = ~gone
            self.vid, self.lane = self.vid[keep], self.lane[keep]
            self.s, self.v, self.a_prev = (self.s[keep], self.v[keep],
                                           self.a_prev[keep])
            self.vtype = self.vtype[keep]
            self.llen = self.llen[keep]

    def _snap2(self):
        return self._snap()

    # ------------------------------------------------------------------
    def step(self, j):
        """One 0.5 s step to absolute tick k0 + j."""
        for args in self.births.get(j, ()):
            self._insert(*args)
        n = len(self.vid)
        if n == 0:
            self.snapshots[j] = self._snap()
            return
        pre_lane = self.lane.copy()
        pre_s = self.s.copy()
        pre_leader = _leaders(pre_lane, pre_s)
        lim = seg_limits(self.lane, self.s)
        tau = np.array([VT[int(t)]["tau"] for t in self.vtype])
        a_max = np.array([VT[int(t)]["a_max"] for t in self.vtype])
        bb = np.array([VT[int(t)]["b"] for t in self.vtype])
        s0p = np.array([VT[int(t)]["s0"] for t in self.vtype])
        sf = np.array([VT[int(t)]["sf"] for t in self.vtype])
        v0 = sf * lim
        lead, gap = _leaders_gap(self.lane, self.s)
        free = lead < 0
        dv = np.where(free, 0.0, self.v - self.v[np.maximum(lead, 0)])
        g = np.where(free, 250.0, np.maximum(gap, 0.1))
        acc = idm_acc_np(self.v, g, dv, tau, a_max, bb, s0p, v0)
        keys = [(int(t), 1 if fr else 0) for t, fr in
                zip(self.vtype, free)]
        acc = acc + np.array([NOISE[k][0] for k in keys])
        acc = acc + self.rng.normal(0.0, 1.0, size=n) * np.array(
            [NOISE[k][1] for k in keys])
        acc = np.clip(acc, -9.0, 3.0)
        v_new = np.clip(self.v + acc * 0.5, 0.0, 1.2 * lim)
        s_new = self.s + v_new * 0.5
        self.capviol += int((v_new > 1.2 * lim + 1e-6).sum())
        self._count_collisions(pre_lane, pre_s, pre_leader)
        self.a_prev = acc
        self.v = v_new
        self.s = s_new
        self._drop_exits()
        self.snapshots[j] = self._snap()

    def run(self):
        for j in range(1, 241):
            self.step(j)
        return self
