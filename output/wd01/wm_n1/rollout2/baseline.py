"""P3 batched Part A v2g baseline (Idea A step (b), zero-residual form).

A math-identical reimplementation of ``partA_v2g.engine`` (imported
READ-ONLY; ``params_nominal.json`` is used unchanged) over the batched
``[B, K]`` rollout state.  INTERFACES_V2 Mismatch 5: the v2g engine is
scalar with a ballistic integrator and per-vehicle loops; P3 reimplements
it batched (per-world compact numpy rows, slot order = vehicle order) so
the G1 zero-residual comparison is like for like, and swaps the
integrator through :mod:`rollout2.validity`.

Faithfulness contract (pinned by rollout2/tests/test_engine_parity.py):

* the same rule functions and evaluation order as
  ``PartAV2GEngine.step`` (lane changes front-most-first, merges before
  discretionary, occupancy refreshed after every accepted change);
* the scalar decision forms with the gap/v0 floors (``_idm1`` mirror)
  and the vectorized ``idm_accel`` for the dynamics path;
* safety gates are NEVER residual-adjustable: the P2 residual only adds
  to the MOBIL incentive margins;
* commanded CAVs: ``ctrl.execution_speed`` with the RAW (unclipped,
  no-obstacle) own-lane IDM term, exactly engine.py:649-658;
* the accel-lane end clamp, collision counting (never repaired) and the
  ``s > lane length`` exit are the engine's (step.py), not the caller's.

The LC2 structural variants (``lc_variants``) are a labelled v2g-side
diagnostic path and are refused here: Idea A runs the nominal set only.
"""
from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROLLOUT2 = Path(__file__).resolve().parent
_WD01 = _ROLLOUT2.parents[1]
if str(_WD01) not in sys.path:
    sys.path.insert(0, str(_WD01))

from partA_v2g.engine import GAP_MAX, LANES, idm_accel  # noqa: E402  read-only


def default_params_path() -> Path:
    """The frozen ``partA_v2g/params_nominal.json`` (read-only)."""
    return _WD01 / "partA_v2g" / "params_nominal.json"


def load_params(path: Optional[Path] = None) -> dict:
    """Load the nominal parameter set (the v2g file, never copied)."""
    p = Path(path) if path is not None else default_params_path()
    return json.loads(p.read_text(encoding="utf-8"))


#: Authoritative SUMO demand driver types (``osm.demand.*.rou.xml``
#: ``vTypeDistribution hdv_mix``; mirrored in the legacy
#: ``hdv_types.add.xml``).  Probabilities are the campaign's mix.
SUMO_VTYPES = {
    "aggressive_car": dict(probability=0.10, tau=0.95, sigma=0.6,
                           speedFactor=1.08, accel=3.0, decel=4.5, minGap=1.5),
    "normal_car": dict(probability=0.70, tau=1.1, sigma=0.4,
                       speedFactor=1.0, accel=2.6, decel=4.5, minGap=2.5),
    "cautious_car": dict(probability=0.20, tau=1.5, sigma=0.2,
                         speedFactor=0.92, accel=2.0, decel=3.0, minGap=3.5),
}


def sumo_idm_mean() -> dict:
    """Probability-weighted SUMO vType means mapped onto IDM knobs.

    The rollout runs textbook IDM (one set for all vehicles), while the TRAIN
    worlds are Krauss with the ``hdv_mix`` distribution above.  These are the
    mix-weighted means of the vType car-following parameters (``tau`` -> IDM
    ``T_s``, ``accel`` -> ``a_max_mps2``, ``decel`` -> ``b_comf_mps2``,
    ``minGap`` -> ``s0_m``; ``delta`` stays 4).  ``speedFactor`` is kept as a
    free-speed multiplier via :func:`model_frame_params`' ``speed_factor``.
    """
    wt = {k: v["probability"] for k, v in SUMO_VTYPES.items()}
    tot = sum(wt.values())
    return dict(
        T_s=sum(SUMO_VTYPES[k]["tau"] * wt[k] for k in wt) / tot,
        a_max_mps2=sum(SUMO_VTYPES[k]["accel"] * wt[k] for k in wt) / tot,
        b_comf_mps2=sum(SUMO_VTYPES[k]["decel"] * wt[k] for k in wt) / tot,
        s0_m=sum(SUMO_VTYPES[k]["minGap"] * wt[k] for k in wt) / tot,
        delta=4,
    )


def sumo_speed_factor_mean() -> float:
    """Probability-weighted SUMO ``speedFactor`` (free-speed multiplier)."""
    wt = {k: v["probability"] for k, v in SUMO_VTYPES.items()}
    tot = sum(wt.values())
    return sum(SUMO_VTYPES[k]["speedFactor"] * wt[k] for k in wt) / tot


def model_frame_params(params: Optional[dict] = None, *,
                       sumo_idm: bool = False,
                       speed_factor: Optional[float] = None) -> dict:
    """The v2g parameter set translated into the rollout's ``s_down`` frame.

    The P3 world ``s_m`` is the model ``s_down`` coordinate
    (``features.geometry.s_down_of`` = physical s - lane offset), while the
    frozen v2g ``params_nominal.json`` geometry is in PHYSICAL lane s.  The
    nominal rule set therefore never sees the accel-lane merge zone
    (``s0z..s1z`` 378.66..454.1 physically vs lane-2 ``s_down`` in
    -378.66..75.44) or the true segment starts: mandatory merges never fire
    and the ramp lane-end obstacle never binds.  This view subtracts each
    lane's offset from its segment starts / lane length, shifts the accel
    merge zone by ``-off[2]`` and zeroes the offsets, so ``d = -s_down`` is
    already the shared cross-lane downstream axis (no per-lane shift).

    ``sumo_idm=True`` additionally replaces the IDM block by the
    ``hdv_mix``-weighted SUMO vType means (:func:`sumo_idm_mean`) and, when
    ``speed_factor`` is not ``None``, multiplies every segment limit by it
    (the observed/free-speed multiplier).  Default ``sumo_idm=False`` and
    ``speed_factor=None`` return an unmodified copy.
    """
    import copy

    p = copy.deepcopy(params) if params is not None else load_params()
    off = [float(x) for x in p["geometry"]["lane_offsets_m"]]
    p["geometry"]["lane_segment_starts_m"] = [
        [float(s) - off[li] for s in p["geometry"]["lane_segment_starts_m"][li]]
        for li in LANES]
    p["geometry"]["lane_lengths_m"] = [
        float(p["geometry"]["lane_lengths_m"][li]) - off[li] for li in LANES]
    p["geometry"]["lane_offsets_m"] = [0.0 for _ in LANES]
    p["merge"]["zone_s0_m"] = float(p["merge"]["zone_s0_m"]) - off[2]
    p["merge"]["zone_s1_m"] = float(p["merge"]["zone_s1_m"]) - off[2]
    if sumo_idm:
        p["idm"].update(sumo_idm_mean())
    if speed_factor is not None:
        sf = float(speed_factor)
        p["geometry"]["lane_segment_limits_mps"] = [
            [float(x) * sf for x in p["geometry"]["lane_segment_limits_mps"][li]]
            for li in LANES]
    return p


class Params:
    """Validated read-only view of the v2g parameter set."""

    def __init__(self, params=None) -> None:
        if params is None:
            params = load_params()
        self.raw = dict(params)
        variants = self.raw.get("lc_variants", {}) or {}
        if any(bool(variants.get(k)) for k in
               ("va_enabled", "vb_enabled", "vc_enabled")):
            raise ValueError(
                "lc_variants are a labelled partA_v2g diagnostic path; "
                "the Idea A baseline runs the nominal set only")
        self.dt = float(self.raw["dt_s"])
        self.idm_p = self.raw["idm"]
        self.mobil_p = self.raw["mobil"]
        self.merge_p = self.raw["merge"]
        self.a_clip = tuple(float(x)
                            for x in self.raw["dynamics"]["a_clip_mps2"])
        self.exec_p = self.raw["cav_execution"]
        self.llen = np.asarray(self.raw["geometry"]["lane_lengths_m"],
                               dtype=np.float64)
        self.off = np.asarray(self.raw["geometry"]["lane_offsets_m"],
                              dtype=np.float64)
        self.s1z = float(self.merge_p["zone_s1_m"])
        self.s0z = float(self.merge_p["zone_s0_m"])
        self.lc_cooldown = float(self.mobil_p.get("cooldown_s", 0.0))
        self.disc_cooldown = max(
            self.lc_cooldown, float(self.mobil_p.get("T_lc_s", 0.0)))
        self.lc_min_speed = float(self.mobil_p.get("lc_min_speed_mps", 0.0))
        self._seg_starts = [np.asarray(
            self.raw["geometry"]["lane_segment_starts_m"][li],
            dtype=np.float64) for li in LANES]
        self._seg_lims = [np.asarray(
            self.raw["geometry"]["lane_segment_limits_mps"][li],
            dtype=np.float64) for li in LANES]

    def seg_limit(self, li: int, s: float) -> float:
        """Scalar segment limit (mirror of partA_v2g.engine._seg_limit1)."""
        idx = 0
        starts = self._seg_starts[li]
        for k in range(len(starts) - 1, -1, -1):
            if s >= starts[k]:
                idx = int(k)
                break
        return float(self._seg_lims[li][idx])

    def seg_limit_vec(self, lane, s):
        """Vectorized per-vehicle limit (mirror of segment_limits)."""
        out = np.full(len(lane), np.nan, dtype=np.float64)
        for li in LANES:
            m = lane == li
            if not m.any():
                continue
            idx = np.searchsorted(self._seg_starts[li], s[m],
                                  side="right") - 1
            idx = np.clip(idx, 0, len(self._seg_lims[li]) - 1)
            out[m] = self._seg_lims[li][idx]
        return out

    def idm1(self, v: float, gap: float, dv: float, v0: float) -> float:
        """Scalar IDM (mirror of partA_v2g.engine._idm1, floors included)."""
        a = float(self.idm_p["a_max_mps2"])
        b = float(self.idm_p["b_comf_mps2"])
        big_t = float(self.idm_p["T_s"])
        s0 = float(self.idm_p["s0_m"])
        delta = int(self.idm_p["delta"])
        gap = gap if gap > 1e-6 else 1e-6
        v0 = v0 if v0 > 1e-6 else 1e-6
        s_star = s0 + max(0.0, v * big_t + v * dv / (2.0 * np.sqrt(a * b)))
        return a * (1.0 - (v / v0) ** delta - (s_star / gap) ** 2)

    def idm_vec(self, v, gap, dv, v0):
        """Vectorized IDM -> partA_v2g.engine.idm_accel (read-only reuse)."""
        return idm_accel(v, gap, dv, v0, self.idm_p)

    def aclip1(self, a: float) -> float:
        lo, hi = self.a_clip
        return hi if a > hi else (lo if a < lo else a)

    def merge_urgency(self, s: float) -> float:
        zl = float(self.merge_p["zone_len_m"])
        return min(max(1.0 - (self.s1z - float(s)) / zl, 0.0), 1.0)


class WorldRows:
    """One world's compact vehicle rows (slot order == vehicle order).

    ``rows[i]`` corresponds to global slot ``slots[i]``; only occupied
    slots are carried, exactly like the v2g engine's per-step arrays.
    Ties in s keep slot order (stable sorts), so the equivalence harness
    assigns slots in sorted-vid order to mirror the v2g name order.
    """

    def __init__(self, off: np.ndarray, lane, s, v, length, slots,
                 lc_last=None) -> None:
        self.off = off
        self.lane = np.asarray(lane, dtype=np.int64)
        self.s = np.asarray(s, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        self.length = np.asarray(length, dtype=np.float64)
        self.slots = np.asarray(slots, dtype=np.int64)
        self.lc_last = (np.full(len(self.slots), -1e9, dtype=np.float64)
                        if lc_last is None
                        else np.asarray(lc_last, dtype=np.float64))
        self._nb = None

    @property
    def n(self) -> int:
        return len(self.slots)

    def invalidate(self) -> None:
        self._nb = None

    def neighbors(self):
        """(leader, follower) rows per lane by ascending s (stable); the
        v2g WD-01-PA2G leader/follower orientation."""
        if self._nb is None:
            n = self.n
            leader = np.full(n, -1, dtype=np.int64)
            follower = np.full(n, -1, dtype=np.int64)
            for li in LANES:
                rows = np.where(self.lane == li)[0]
                rows = rows[np.argsort(self.s[rows], kind="stable")]
                for j in range(len(rows) - 1):
                    leader[int(rows[j])] = int(rows[j + 1])
                    follower[int(rows[j + 1])] = int(rows[j])
            self._nb = (leader, follower)
        return self._nb

    def own_leader_follower(self, r: int) -> Tuple[int, int]:
        leader, follower = self.neighbors()
        return int(leader[r]), int(follower[r])

    def lane_d_sorted(self, li: int) -> Tuple[List[float], List[int]]:
        """Lane li rows by ascending d (d = off - s; shared cross-lane
        alignment axis, mirror of partA_v2g.engine._lane_d_sorted)."""
        rows = [r for r in range(self.n) if self.lane[r] == li]
        rows.sort(key=lambda r: float(self.off[li] - self.s[r]))
        return [float(self.off[li] - self.s[r]) for r in rows], rows

    def d_neighbors(self, d_sorted: List[float], rows: List[int],
                    d_i: float) -> Tuple[int, int]:
        k = bisect.bisect_left(d_sorted, d_i)
        leader = rows[k - 1] if k >= 1 else -1
        follower = rows[k] if k < len(rows) else -1
        return leader, follower

    def overlaps_at(self, r: int, tgt: int) -> bool:
        """True iff row r, shifted into lane ``tgt``, would overlap a
        target-lane vehicle (bumper gap < 0 at the shifted position)."""
        s_shift = float(self.s[r] + (self.off[tgt] - self.off[int(
            self.lane[r])]))
        for j in range(self.n):
            if j == r or int(self.lane[j]) != tgt:
                continue
            if abs(s_shift - float(self.s[j])) < \
                    0.5 * (float(self.length[r]) + float(self.length[j])):
                return True
        return False


class V2GCore:
    """The per-world rule engine (one batch element's rows per call)."""

    def __init__(self, params: Params, *, lane_changes_enabled: bool = True
                 ) -> None:
        self.P = params
        self.lane_changes_enabled = bool(lane_changes_enabled)

    # ---------------------------------------------------------- helpers
    def leaders(self, w: WorldRows):
        """Own-lane leader rows, bumper gaps (GAP_MAX free) and leader
        speeds (mirror of partA_v2g.engine._leaders_vector)."""
        leader = np.full(w.n, -1, dtype=np.int64)
        gap = np.full(w.n, GAP_MAX, dtype=np.float64)
        lv = np.zeros(w.n, dtype=np.float64)
        for li in LANES:
            rows = np.where(w.lane == li)[0]
            rows = rows[np.argsort(w.s[rows], kind="stable")]
            for j in range(len(rows) - 1):
                f, l_ = int(rows[j]), int(rows[j + 1])
                leader[f] = l_
                gap[f] = w.s[l_] - w.length[l_] - w.s[f]
                lv[f] = w.v[l_]
        return leader, gap, lv

    def a_stay(self, w: WorldRows, r: int) -> float:
        """Mirror of partA_v2g.engine._a_stay."""
        lo, _ = w.own_leader_follower(r)
        li = int(w.lane[r])
        v0 = self.P.seg_limit(li, float(w.s[r]))
        if lo >= 0:
            gap = float(w.s[lo] - w.length[lo] - w.s[r])
            dv = float(w.v[r] - w.v[lo])
        else:
            gap, dv = GAP_MAX, 0.0
        a = self.P.idm1(float(w.v[r]), gap, dv, v0)
        if li == 2:
            gap_end = max(self.P.s1z - float(w.s[r]), 0.3)
            a = min(a, self.P.idm1(float(w.v[r]), gap_end, float(w.v[r]),
                                   v0))
        return self.P.aclip1(a)

    # ------------------------------------------------------ lane changes
    def _merge_eval(self, w: WorldRows, r: int, d_sorted1: List[float],
                    rows1: List[int]) -> Tuple[bool, float]:
        """Merge decision split into (safety ok, incentive margin).

        Mirror of partA_v2g.engine._merge_decision: ``safe`` is the
        safe-gap conjunction (the v1 kinematic form on d-gaps + the
        b_safe(u) required-deceleration tests); ``margin`` is the MOBIL
        gain minus accel_threshold*(1 - u).  Accept = safe AND margin >= 0.
        """
        P = self.P
        mp = P.merge_p
        s1z = P.s1z
        zl = float(mp["zone_len_m"])
        u = min(max(1.0 - (s1z - float(w.s[r])) / zl, 0.0), 1.0)
        bsafe = float(mp["b_safe_base_mps2"]) + u * (
            float(mp["b_safe_end_mps2"]) - float(mp["b_safe_base_mps2"]))
        d_i = float(P.off[2] - w.s[r])
        lrow, frow = w.d_neighbors(d_sorted1, rows1, d_i)
        s_ego_t = float(w.s[r]) + float(P.off[1] - P.off[2])
        v0_t = P.seg_limit(1, s_ego_t)
        if lrow >= 0:
            gap_lead = float(w.s[lrow] - w.length[lrow] - s_ego_t)
            d_gap_lead = float(w.s[lrow] - s_ego_t)
            v_lead = float(w.v[lrow])
        else:
            gap_lead, d_gap_lead, v_lead = GAP_MAX, GAP_MAX, 0.0
        if frow >= 0:
            gap_foll = float(s_ego_t - w.length[r] - w.s[frow])
            d_gap_foll = float(s_ego_t - w.s[frow])
            v_foll = float(w.v[frow])
        else:
            gap_foll, d_gap_foll, v_foll = GAP_MAX, GAP_MAX, 0.0
        s0i = float(P.idm_p["s0_m"])
        big_t = float(P.idm_p["T_s"])
        floor = float(mp["gap_floor_m"])
        safe = True
        if lrow >= 0 and d_gap_lead < s0i + max(0.0, float(w.v[r]) * big_t):
            safe = False
        if frow >= 0 and d_gap_foll < s0i:
            safe = False
        if lrow >= 0:
            b_req_lead = (float(w.v[r]) ** 2 - v_lead ** 2) / \
                (2.0 * max(d_gap_lead - s0i, floor))
            if b_req_lead > bsafe:
                safe = False
        if frow >= 0:
            b_req_foll = (v_foll ** 2 - float(w.v[r]) ** 2) / \
                (2.0 * max(d_gap_foll - s0i, floor))
            if b_req_foll > bsafe:
                safe = False
        a_ego_new = P.aclip1(P.idm1(float(w.v[r]), gap_lead,
                                    float(w.v[r]) - v_lead, v0_t))
        d_new_foll = 0.0
        if frow >= 0:
            v0_f = P.seg_limit(1, float(w.s[frow]))
            a_f_after = P.aclip1(P.idm1(
                v_foll, gap_foll, v_foll - float(w.v[r]), v0_f))
            lo, _ = w.own_leader_follower(frow)
            if lo >= 0:
                cg = float(w.s[lo] - w.length[lo] - w.s[frow])
                cv = float(w.v[lo])
            else:
                cg, cv = GAP_MAX, 0.0
            a_f_before = P.aclip1(P.idm1(v_foll, cg, v_foll - cv, v0_f))
            d_new_foll = a_f_after - a_f_before
        d_old_foll = 0.0
        lo2, fo2 = w.own_leader_follower(r)
        if fo2 >= 0:
            v_of = float(w.v[fo2])
            gap_of = float(w.s[r] - w.length[r] - w.s[fo2])
            v0_o = P.seg_limit(2, float(w.s[fo2]))
            a_of_before = P.aclip1(P.idm1(
                v_of, gap_of, v_of - float(w.v[r]), v0_o))
            if lo2 >= 0:
                gap_ol = float(w.s[lo2] - w.length[lo2] - w.s[fo2])
                v_ol = float(w.v[lo2])
            else:
                gap_ol, v_ol = GAP_MAX, 0.0
            a_of_after = P.aclip1(P.idm1(v_of, gap_ol, v_of - v_ol, v0_o))
            d_old_foll = a_of_after - a_of_before
        gain = (a_ego_new - self.a_stay(w, r)) + \
            float(P.mobil_p["politeness_p"]) * (d_new_foll + d_old_foll)
        thr = float(P.mobil_p["accel_threshold_mps2"]) * (1.0 - u)
        return safe, gain - thr

    def _merge_accept(self, safe: bool, margin: float, delta) -> bool:
        """Merge acceptance from (safe, incentive margin, residual delta).

        The default is the exact v2g rule (``safe and margin + delta >= 0``);
        calibrated subclasses override this to make the mandatory merge
        strategic (accept whenever physically safe).
        """
        eff = margin if delta is None else margin + float(delta)
        return bool(safe and eff >= 0.0)

    def _disc_eval(self, w: WorldRows, r: int, tgt: int, to_right: bool,
                   d_sorted_t: List[float], rows_t: List[int]
                   ) -> Tuple[bool, float]:
        """Discretionary decision split into (safety ok, incentive
        margin).  Mirror of partA_v2g.engine._disc_decision: ``safe`` is
        the gap_floor lead gate + the new-follower b_safe gate + the
        (nominal-off) ego gain floor; ``margin`` = gain - effective
        threshold (keep-right bias included)."""
        P = self.P
        li = int(w.lane[r])
        d_i = float(P.off[li] - w.s[r])
        lrow, frow = w.d_neighbors(d_sorted_t, rows_t, d_i)
        s_ego_t = float(w.s[r]) + float(P.off[tgt] - P.off[li])
        v0_t = P.seg_limit(tgt, s_ego_t)
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
        if gap_lead < float(P.merge_p["gap_floor_m"]):
            return False, -np.inf
        a_ego_new = P.aclip1(P.idm1(float(w.v[r]), gap_lead,
                                    float(w.v[r]) - v_lead, v0_t))
        a_ego_old = self.a_stay(w, r)
        _floor = float(P.mobil_p.get("ego_gain_floor_mps2", 0.0))
        if _floor > 0.0 and (a_ego_new - a_ego_old) < _floor:
            return False, -np.inf
        a_new_after = a_new_before = 0.0
        if frow >= 0:
            v0_f = P.seg_limit(tgt, float(w.s[frow]))
            a_new_after = P.aclip1(P.idm1(
                v_foll, gap_foll, v_foll - float(w.v[r]), v0_f))
            if a_new_after < -float(P.mobil_p["b_safe_mps2"]):
                return False, -np.inf
            lo, _ = w.own_leader_follower(frow)
            if lo >= 0:
                cg = float(w.s[lo] - w.length[lo] - w.s[frow])
                cv = float(w.v[lo])
            else:
                cg, cv = GAP_MAX, 0.0
            a_new_before = P.aclip1(P.idm1(v_foll, cg, v_foll - cv, v0_f))
        lo_s, fo_s = w.own_leader_follower(r)
        a_old_after = a_old_before = 0.0
        if fo_s >= 0:
            v_of = float(w.v[fo_s])
            gap_of = float(w.s[r] - w.length[r] - w.s[fo_s])
            v0_o = P.seg_limit(li, float(w.s[fo_s]))
            a_old_before = P.aclip1(P.idm1(
                v_of, gap_of, v_of - float(w.v[r]), v0_o))
            if lo_s >= 0:
                gap_ol = float(w.s[lo_s] - w.length[lo_s] - w.s[fo_s])
                v_ol = float(w.v[lo_s])
            else:
                gap_ol, v_ol = GAP_MAX, 0.0
            a_old_after = P.aclip1(P.idm1(v_of, gap_ol, v_of - v_ol, v0_o))
        pol = float(P.mobil_p["politeness_p"])
        gain = (a_ego_new - a_ego_old) + pol * (
            (a_new_after - a_new_before) + (a_old_after - a_old_before))
        th = float(P.mobil_p["accel_threshold_mps2"])
        bias = float(P.mobil_p["keep_right_bias_mps2"])
        eff = th - bias if to_right else th + bias
        return True, gain - eff

    def _merge_cand(self, w: WorldRows, t_change: float) -> List[int]:
        P = self.P
        return [r for r in range(w.n)
                if w.lane[r] == 2 and P.s0z <= w.s[r] <= P.s1z
                and t_change - w.lc_last[r] >= P.lc_cooldown - 1e-9]

    def _disc_cand(self, w: WorldRows, src: int, t_change: float
                   ) -> List[int]:
        P = self.P
        return [r for r in range(w.n)
                if w.lane[r] == src and w.v[r] >= P.lc_min_speed
                and t_change - w.lc_last[r] >= P.disc_cooldown - 1e-9]

    def lane_margins(self, w: WorldRows, *, t_change: float
                     ) -> Dict[str, np.ndarray]:
        """Per-vehicle baseline LC margins (telemetry / P9 hdv_base).

        ``margin_left`` is the merge margin on lane 2 and the
        discretionary 1->0 margin on lane 1; ``margin_right`` is the
        discretionary 0->1 margin; ``accept_left``/``accept_right``
        include the safety gates.  Computed against the CURRENT
        occupancy (the start-of-stage view, no sequential refresh).
        """
        out: Dict = {"margin_left": np.full(w.n, -np.inf),
                     "margin_right": np.full(w.n, -np.inf),
                     "merge_urgency": np.zeros(w.n),
                     "accept_left": np.zeros(w.n, dtype=bool),
                     "accept_right": np.zeros(w.n, dtype=bool)}
        if not self.lane_changes_enabled:
            return out
        for r in range(w.n):
            if w.lane[r] >= 0:
                out["merge_urgency"][r] = self.P.merge_urgency(w.s[r])
        if self._merge_cand(w, t_change):
            d_sorted1, rows1 = w.lane_d_sorted(1)
            for r in self._merge_cand(w, t_change):
                safe, margin = self._merge_eval(w, r, d_sorted1, rows1)
                out["margin_left"][r] = margin
                out["accept_left"][r] = bool(safe and margin >= 0.0)
        for src, tgt, to_right, mkey, akey in (
                (1, 0, False, "margin_left", "accept_left"),
                (0, 1, True, "margin_right", "accept_right")):
            cand = self._disc_cand(w, src, t_change)
            if not cand:
                continue
            d_sorted_t, rows_t = w.lane_d_sorted(tgt)
            for r in cand:
                safe, margin = self._disc_eval(w, r, tgt, to_right,
                                               d_sorted_t, rows_t)
                out[mkey][r] = margin
                out[akey][r] = bool(safe and margin >= 0.0)
        return out

    def lane_change_stage(self, w: WorldRows, *, t_change: float,
                          delta_left=None, delta_right=None,
                          enforce_legality: bool = False,
                          counts=None) -> Dict[str, int]:
        """Execute the v2g lane-change stage IN PLACE (engine.py:576-633).

        Merges first (front-most first, occupancy refreshed after every
        acceptance), then discretionary left (1->0) and right (0->1).
        ``delta_left``/``delta_right`` are the P2 residual lane-logit
        deltas for the stage direction; they shift ONLY the incentive
        margin (safety gates are never residual-adjustable).
        ``enforce_legality`` additionally rejects an executed change that
        would overlap a target-lane vehicle at the shifted position (the
        Idea A 'd1' validity; a no-op for the exact v2g rules, whose
        safety gates already preclude overlaps).

        Returns ``{"merge": int, "disc_left": int, "disc_right": int,
        "lane_reject": int}``.
        """
        P = self.P
        out = {"merge": 0, "disc_left": 0, "disc_right": 0, "lane_reject": 0}
        if not self.lane_changes_enabled:
            return out
        cand2 = self._merge_cand(w, t_change)
        cand2.sort(key=lambda r: -w.s[r])
        d_sorted1, rows1 = w.lane_d_sorted(1)
        for r in cand2:
            safe, margin = self._merge_eval(w, r, d_sorted1, rows1)
            accept = self._merge_accept(
                safe, margin, None if delta_left is None else float(delta_left[r]))
            if accept and enforce_legality and w.overlaps_at(r, 1):
                out["lane_reject"] += 1
                if counts is not None:
                    counts.lane_reject += 1
                continue
            if accept:
                w.lane[r] = 1
                w.s[r] = w.s[r] + P.off[1] - P.off[2]
                w.lc_last[r] = t_change
                out["merge"] += 1
                w.invalidate()
                d_sorted1, rows1 = w.lane_d_sorted(1)
        for src, tgt, to_right, key in ((1, 0, False, "disc_left"),
                                        (0, 1, True, "disc_right")):
            cand = self._disc_cand(w, src, t_change)
            cand.sort(key=lambda r: -w.s[r])
            d_sorted_t, rows_t = w.lane_d_sorted(tgt)
            d_delta = delta_left if src == 1 else delta_right
            for r in cand:
                safe, margin = self._disc_eval(w, r, tgt, to_right,
                                               d_sorted_t, rows_t)
                eff = margin if d_delta is None else \
                    margin + float(d_delta[r])
                accept = bool(safe and eff >= 0.0)
                if accept and enforce_legality and w.overlaps_at(r, tgt):
                    out["lane_reject"] += 1
                    if counts is not None:
                        counts.lane_reject += 1
                    continue
                if accept:
                    w.lane[r] = tgt
                    w.s[r] = w.s[r] + P.off[tgt] - P.off[src]
                    w.lc_last[r] = t_change
                    out[key] += 1
                    w.invalidate()
                    d_sorted_t, rows_t = w.lane_d_sorted(tgt)
        return out

    # ------------------------------------------------------ accelerations
    def accelerations(self, w: WorldRows, v0_override=None):
        """Mirror of engine.py:634-648.

        Returns ``(acc_clipped, idm_raw, acc_pre, gap_eff, dv, v0)``
        where ``idm_raw`` is the pre-obstacle pre-clip own-lane IDM (the
        term the commanded-CAV execution rule consumes,
        engine.py:559) and ``acc_pre`` is the post-obstacle pre-clip
        acceleration (for the accel-clip counter).

        ``v0_override`` (WM-S T3, default ``None`` = OFF) is an optional
        per-row desired speed aligned to ``w``'s compact rows; it replaces
        the segment limit in the own-lane IDM term (and its obstacle copy).
        """
        P = self.P
        leader, gap, lv = self.leaders(w)
        v0 = P.seg_limit_vec(w.lane, w.s)
        if v0_override is not None:
            v0 = np.asarray(v0_override, dtype=np.float64).copy()
        dv = np.where(leader >= 0, w.v - lv, 0.0)
        gap_eff = np.where(leader >= 0, gap, GAP_MAX)
        acc = P.idm_vec(w.v, gap_eff, dv, v0)
        idm_raw = acc.copy()
        on2 = w.lane == 2
        if on2.any():
            gap_end = np.maximum(P.s1z - w.s, 0.3)
            acc_obs = P.idm_vec(w.v, gap_end, w.v.copy(), v0)
            acc = np.where(on2, np.minimum(acc, acc_obs), acc)
        acc_pre = acc.copy()
        acc = np.clip(acc, P.a_clip[0], P.a_clip[1])
        return acc, idm_raw, acc_pre, gap_eff, dv, v0

    def collisions(self, w: WorldRows) -> int:
        """Bumper gap < 0 over same-lane adjacent pairs after the update
        (engine.py:672-685); counted, never repaired."""
        step_coll = 0
        for li in LANES:
            o = np.where(w.lane == li)[0]
            o = o[np.argsort(w.s[o], kind="stable")]
            if len(o) < 2:
                continue
            sgap = w.s[o[1:]] - w.length[o[1:]] - w.s[o[:-1]]
            step_coll += int((sgap < 0.0).sum())
        return step_coll
