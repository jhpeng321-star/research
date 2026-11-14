"""Calibrated SUMO-like mandatory-merge physics for the A11 rollout.

Flag-gated, default OFF (``Rollout2Config.merge_cal``).  The F2 geometry fix
(``sumo_fidelity``) makes the mandatory merge fire, but the frozen v2g merge
rule (:meth:`rollout2.baseline.V2GCore._merge_eval`) is far too strict: it
demands a lead gap of ``s0 + v*T`` (~39 m at 25 m/s) and a hard follower
required-braking ceiling, so almost no ramp vehicle merges and lane 2
over-queues ~7x the recorded SUMO merge stopped vehicle-ticks (FIDELITY.md).

This module replaces it with an lc2013-style strategic merge on TRAIN-only
fitted parameters (``merge_cal_params.json``):

* **speed/urgency safe gaps** -- mandatory changes are taken whenever
  physically safe; the lead safe gap is ``s0 + v*T_lead(u)`` and the follower
  required deceleration is bounded by ``b_accept(u)``, both relaxing toward
  the lane end (``u`` = the frozen merge urgency).  There is no MOBIL-gain
  gate: a ramp vehicle on a lane that ends is *strategic*, exactly as in
  SUMO's ``LC2013``.
* **cooperative mainline yielding** -- a lane-1 follower adjacent to an urgent
  lane-2 vehicle brakes (bounded) toward the merging vehicle treated as a
  virtual leader, opening the gap the merger needs.
* **correct exits** -- the caller keeps ``physical_exits``; the merge lane-end
  clamp is unchanged.

Parameters are fitted on the recorded chunk-4 TRAIN merge statistics only
(``_out/merge_cal_recorded.json``); no held-out or sealed data is read.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .baseline import GAP_MAX, WorldRows, V2GCore


@dataclass
class MergeCalParams:
    """Fitted physics knobs (defaults = the registered fit)."""

    t_lead_base_s: float = 1.0        # lead safe-gap time at u = 0
    t_lead_end_s: float = 0.10        # lead safe-gap time at u = 1
    b_accept_base_mps2: float = 4.0   # follower/lead required-decel ceiling at u = 0
    b_accept_end_mps2: float = 6.0    # ... at u = 1
    gap_floor_m: float = 0.3
    u_yield: float = 0.35             # urgency above which mainline yields
    yield_lookahead_m: float = 120.0  # virtual-leader lookahead for yielding
    b_yield_max_mps2: float = 2.5     # bounded cooperative deceleration
    yield_min_speed_mps: float = 0.5  # do not yield below this own speed
    yield_min_merger_speed_mps: float = 3.0  # only yield to MOVING mergers
    end_obstacle_range_m: float = 25.0  # lane-2 end obstacle only within last X m
    speed_factor: Optional[float] = None  # free-speed multiplier (exits/over-speed)

    @classmethod
    def load(cls, path=None) -> "MergeCalParams":
        p = Path(path) if path is not None else default_params_path()
        if not p.is_file():
            return cls()
        raw = json.loads(p.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def default_params_path() -> Path:
    return Path(__file__).resolve().parent / "merge_cal_params.json"


class MergeCalCore(V2GCore):
    """Mandatory-merge core with calibrated acceptance + cooperative yielding."""

    def __init__(self, params, merge_params: Optional[MergeCalParams] = None,
                 *, lane_changes_enabled: bool = True) -> None:
        super().__init__(params, lane_changes_enabled=lane_changes_enabled)
        self.M = merge_params if merge_params is not None else MergeCalParams()

    # ------------------------------------------------- strategic merge
    def a_stay(self, w: WorldRows, r: int) -> float:
        """Own-lane IDM with the lane-2 end obstacle only near the lane end.

        The v2g nominal treats the accel-lane end as a standing obstacle over
        the whole merge zone, which brakes ramp vehicles far from the end and
        is a major over-queue contributor.  SUMO enforces the lane change
        strategically instead; here the end obstacle binds only within
        ``end_obstacle_range_m`` of the end (the accel-lane-end clamp in
        ``step`` still catches a vehicle that reaches it).
        """
        P = self.P
        lo, _ = w.own_leader_follower(r)
        li = int(w.lane[r])
        v0 = P.seg_limit(li, float(w.s[r]))
        if lo >= 0:
            gap = float(w.s[lo] - w.length[lo] - w.s[r])
            dv = float(w.v[r] - w.v[lo])
        else:
            gap, dv = GAP_MAX, 0.0
        a = P.idm1(float(w.v[r]), gap, dv, v0)
        if li == 2 and (P.s1z - float(w.s[r])) <= self.M.end_obstacle_range_m:
            gap_end = max(P.s1z - float(w.s[r]), 0.3)
            a = min(a, P.idm1(float(w.v[r]), gap_end, float(w.v[r]), v0))
        return P.aclip1(a)

    def _merge_accept(self, safe: bool, margin: float, delta) -> bool:
        # Strategic mandatory merge: accept whenever physically safe.  The
        # residual lane logit (``delta``) is still allowed to *unlock* an
        # otherwise-unsafe change?  No -- safety is never residual-adjustable,
        # so the merge is exactly ``safe``.
        return bool(safe)

    def _merge_eval(self, w: WorldRows, r: int, d_sorted1,
                    rows1) -> Tuple[bool, float]:
        P = self.P
        mp = P.merge_p
        M = self.M
        s1z = P.s1z
        zl = float(mp["zone_len_m"])
        u = min(max(1.0 - (s1z - float(w.s[r])) / zl, 0.0), 1.0)
        T_lead = M.t_lead_base_s + (M.t_lead_end_s - M.t_lead_base_s) * u
        b_acc = M.b_accept_base_mps2 + (
            M.b_accept_end_mps2 - M.b_accept_base_mps2) * u
        d_i = float(P.off[2] - w.s[r])
        lrow, frow = w.d_neighbors(d_sorted1, rows1, d_i)
        s_ego_t = float(w.s[r]) + float(P.off[1] - P.off[2])
        v0_t = P.seg_limit(1, s_ego_t)
        if lrow >= 0:
            gap_lead = float(w.s[lrow] - w.length[lrow] - s_ego_t)
            v_lead = float(w.v[lrow])
        else:
            gap_lead, v_lead = GAP_MAX, 0.0
        if frow >= 0:
            gap_foll = float(s_ego_t - w.length[r] - w.s[frow])
            v_foll = float(w.v[frow])
        else:
            gap_foll, v_foll = GAP_MAX, 0.0
        s0i = float(P.idm_p["s0_m"])
        floor = float(M.gap_floor_m)
        v = float(w.v[r])
        safe = True
        # lead: speed/urgency-dependent safe headway (bumper gap)
        if lrow >= 0 and gap_lead < max(floor, s0i + max(0.0, v * T_lead)):
            safe = False
        # follower: required braking ceiling (relaxes toward the lane end)
        if frow >= 0:
            b_req_foll = (v_foll ** 2 - v ** 2) / \
                (2.0 * max(gap_foll, floor))
            if b_req_foll > b_acc:
                safe = False
        if lrow >= 0:
            b_req_lead = (v ** 2 - v_lead ** 2) / \
                (2.0 * max(gap_lead, floor))
            if b_req_lead > b_acc:
                safe = False
        if gap_lead < floor or gap_foll < floor:
            safe = False
        # MOBIL incentive is reported for telemetry only; strategic acceptance
        # is handled in _merge_accept (safe => accept).
        a_ego_new = P.aclip1(P.idm1(v, gap_lead, v - v_lead, v0_t))
        gain = a_ego_new - self.a_stay(w, r)
        return safe, gain

    # ------------------------------------------- cooperative yielding
    def accelerations(self, w: WorldRows, v0_override=None):
        acc, idm_raw, acc_pre, gap_eff, dv, v0 = super().accelerations(
            w, v0_override=v0_override)
        M = self.M
        # The v2g standing-obstacle brake at the accel-lane end binds over the
        # WHOLE lane (even behind the merge zone); with the SUMO IDM its
        # stopping distance exceeds the ramp length, so ramp traffic crawls
        # from the insertion point (the dominant over-queue term).  The
        # calibrated physics brakes for the end only within
        # ``end_obstacle_range_m`` (SUMO has no such obstacle: it changes
        # lanes strategically and the lane-end clamp catches failures).
        on2 = w.lane == 2
        if on2.any():
            far = on2 & ((self.P.s1z - w.s) > float(M.end_obstacle_range_m))
            if far.any():
                acc[far] = np.clip(idm_raw[far],
                                   self.P.a_clip[0], self.P.a_clip[1])
                acc_pre[far] = idm_raw[far]
        if not bool(getattr(M, "yield_lookahead_m", 0.0)) or w.n == 0:
            return acc, idm_raw, acc_pre, gap_eff, dv, v0
        lane = w.lane
        rows2 = np.flatnonzero(lane == 2)
        rows1 = np.flatnonzero(lane == 1)
        if rows2.size == 0 or rows1.size == 0:
            return acc, idm_raw, acc_pre, gap_eff, dv, v0
        s2 = w.s[rows2]
        u2 = np.array([self.P.merge_urgency(float(s)) for s in s2])
        urg = rows2[u2 >= M.u_yield]
        if urg.size == 0:
            return acc, idm_raw, acc_pre, gap_eff, dv, v0
        su, vu = w.s[urg], w.v[urg]
        for f in rows1:
            vf = float(w.v[f])
            if vf < M.yield_min_speed_mps:
                continue
            ds = su - float(w.s[f])
            m = (ds > 0.0) & (ds <= M.yield_lookahead_m)
            if not m.any():
                continue
            k = int(np.argmin(np.where(m, ds, np.inf)))
            mrow = int(urg[k])
            if float(vu[k]) < M.yield_min_merger_speed_mps:
                continue  # never brake for a (nearly) stopped merger
            gap_v = float(su[k] - w.length[mrow] - w.s[f])
            v0f = float(self.P.seg_limit(1, float(w.s[f])))
            a_v = self.P.idm1(vf, gap_v, vf - float(vu[k]), v0f)
            a_v = max(a_v, -float(M.b_yield_max_mps2))
            if a_v < acc[f]:
                acc[f] = a_v
                acc_pre[f] = min(acc_pre[f], a_v)
        return acc, idm_raw, acc_pre, gap_eff, dv, v0
