"""WD-01 partA_v2g: the general rule-based Part A v2 engine.

PARTA_V2_GENERAL_DESIGN.md (owner decision 2026-09-29): generic textbook
rules with ONE nominal parameter set (params_nominal.json), shared by all
vehicles, deterministic, no vType anywhere, no insertions, no teleports.

- Car-following: Treiber IDM (v0 = the containing segment's speed limit).
- Lane changes: MOBIL for discretionary 0<->1 changes; a mandatory merge
  rule on the accel lane (urgency rises toward the lane end, b_safe
  relaxed linearly 4 -> 6 m/s^2, stop at the lane end if no gap).
- CAVs follow the commanded action through the partA_ctrl emulators +
  the measured execution rule (cavctrl.py hook; this module is
  torch-free).
- Ballistic update at dt = 0.5 s (v' = max(0, v + a*dt); s' = s + v'*dt,
  the project's measured SUMO integrator form).
- Collisions are counted when the bumper gap < 0 after an update; the
  model never teleports and never repairs inversions.
- Every input vehicle keeps its identity until it leaves the modelled
  road (s beyond the lane length); no vehicle is ever created.

Pure numpy + scalar inner loops. ZERO SIMULATION: no SUMO/TraCI of any
kind.  The engine NEVER reads a vType/driver-type key (test-enforced).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
PARAMS_PATH = HERE / "params_nominal.json"
GAP_MAX = 250.0          # free-road gap sentinel (p3lib convention)
LANES = (0, 1, 2)


def load_params(path: Path = PARAMS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def param_set_id(path: Path = PARAMS_PATH) -> str:
    """sha256 of the parameter file (contract v2.1 meta.param_set_id)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------- pure rules
# The public vectorized rule functions are the tested contract; the
# engine's inner loops use the math-identical scalar forms below.

def idm_accel(v, gap, dv, v0, p: Mapping) -> np.ndarray:
    """Treiber IDM acceleration, vectorized.

    v, gap, dv (= v - v_leader, closing positive), v0: arrays.  gap is
    floored at 1e-6 (never NaN); free road is gap = GAP_MAX, dv = 0.
    """
    v = np.asarray(v, dtype=np.float64)
    gap = np.maximum(np.asarray(gap, dtype=np.float64), 1e-6)
    dv = np.asarray(dv, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    a = float(p["a_max_mps2"])
    b = float(p["b_comf_mps2"])
    T = float(p["T_s"])
    s0 = float(p["s0_m"])
    delta = int(p["delta"])
    s_star = s0 + np.maximum(0.0, v * T + v * dv / (2.0 * math.sqrt(a * b)))
    return a * (1.0 - (v / v0) ** delta - (s_star / gap) ** 2)


def _idm1(v: float, gap: float, dv: float, v0: float, p: Mapping) -> float:
    """Scalar IDM (math-identical to idm_accel; hot inner loop)."""
    a = float(p["a_max_mps2"])
    b = float(p["b_comf_mps2"])
    T = float(p["T_s"])
    s0 = float(p["s0_m"])
    delta = int(p["delta"])
    gap = gap if gap > 1e-6 else 1e-6
    v0 = v0 if v0 > 1e-6 else 1e-6
    s_star = s0 + max(0.0, v * T + v * dv / (2.0 * math.sqrt(a * b)))
    return a * (1.0 - (v / v0) ** delta - (s_star / gap) ** 2)


def mobil_gain(a_ego_new, a_ego_old, a_new_foll_after, a_new_foll_before,
               a_old_foll_after, a_old_foll_before, p: Mapping) -> np.ndarray:
    """MOBIL incentive gain: da_ego + p*(da_new_foll + da_old_foll)."""
    pol = float(p["politeness_p"])
    d_ego = np.asarray(a_ego_new, dtype=np.float64) - np.asarray(a_ego_old)
    d_new = np.asarray(a_new_foll_after) - np.asarray(a_new_foll_before)
    d_old = np.asarray(a_old_foll_after) - np.asarray(a_old_foll_before)
    return d_ego + pol * (d_new + d_old)


def mobil_passes(gain, p: Mapping, *, to_right: bool) -> np.ndarray:
    """Discretionary MOBIL incentive test with the keep-right bias:
    rightward changes earn the bias, leftward changes pay it."""
    th = float(p["accel_threshold_mps2"])
    bias = float(p["keep_right_bias_mps2"])
    eff = th - bias if to_right else th + bias
    return np.asarray(gain) >= eff


def mobil_safe(a_new_foll_after, p: Mapping) -> np.ndarray:
    """Textbook MOBIL safety: the new follower must not brake harder than
    b_safe after the change."""
    return np.asarray(a_new_foll_after) >= -float(p["b_safe_mps2"])


def merge_urgency(s, p: Mapping) -> np.ndarray:
    """u = 1 - (zone_s1 - s)/zone_len, clipped to [0, 1]."""
    s1 = float(p["zone_s1_m"])
    L = float(p["zone_len_m"])
    return np.clip(1.0 - (s1 - np.asarray(s, dtype=np.float64)) / L, 0.0, 1.0)


def merge_b_safe(u, p: Mapping) -> np.ndarray:
    """b_safe relaxed linearly from b_safe_base to b_safe_end with
    urgency u in [0, 1]."""
    b0 = float(p["b_safe_base_mps2"])
    b1 = float(p["b_safe_end_mps2"])
    return b0 + (b1 - b0) * np.asarray(u, dtype=np.float64)


def merge_threshold(u, p: Mapping) -> np.ndarray:
    """The MOBIL threshold replaced by urgency: it falls linearly to 0 at
    the lane end (p = the MOBIL section: the threshold is MOBIL's)."""
    return float(p["accel_threshold_mps2"]) * (1.0 - np.asarray(u))


def lane_end_obstacle(v, gap_end, v0, p: Mapping) -> np.ndarray:
    """IDM acceleration against the virtual standing obstacle at the
    accel-lane end (standing leader: dv = v)."""
    v = np.asarray(v, dtype=np.float64)
    return idm_accel(v, gap_end, v, v0, p)


def segment_limits(lane, s, params: Mapping) -> np.ndarray:
    """v0 per vehicle = the containing segment's speed limit (params
    geometry table; no external dependency)."""
    lane = np.asarray(lane)
    s = np.asarray(s, dtype=np.float64)
    starts = params["geometry"]["lane_segment_starts_m"]
    lims = params["geometry"]["lane_segment_limits_mps"]
    out = np.full(len(lane), np.nan, dtype=np.float64)
    for li in LANES:
        m = lane == li
        if not m.any():
            continue
        idx = np.searchsorted(np.asarray(starts[li]), s[m], side="right") - 1
        idx = np.clip(idx, 0, len(lims[li]) - 1)
        out[m] = np.asarray(lims[li])[idx]
    return out


def _seg_limit1(li: int, s: float, params: Mapping) -> float:
    starts = params["geometry"]["lane_segment_starts_m"][li]
    lims = params["geometry"]["lane_segment_limits_mps"][li]
    idx = 0
    for k in range(len(starts) - 1, -1, -1):
        if s >= starts[k]:
            idx = k
            break
    return float(lims[idx])


# ------------------------------------------------------------- engine

class PartAV2GEngine:
    """One deterministic rollout of the general rules from a recon-sample
    vehicle table.  No insertions; identities kept until the vehicle
    leaves the modelled road (s > lane length)."""

    def __init__(self, params: Mapping, *, t0: float,
                 vehicles: Mapping[str, Mapping],
                 control_hook: Optional[Callable] = None,
                 lane_changes_enabled: bool = True):
        """vehicles: {vid: {lane, s, v, a (optional), is_cav (optional),
        length_m (optional)}}.  control_hook(t_now, engine) -> {vid:
        commanded target speed} is called on 1 s control ticks (cavctrl).
        lane_changes_enabled=False is a test/diagnostic switch that
        freezes the lane assignment (car-following only).
        """
        self.p = dict(params)
        self.dt = float(params["dt_s"])
        self.t0 = float(t0)
        self.control_hook = control_hook
        self.lane_changes_enabled = bool(lane_changes_enabled)
        self.merge_p = params["merge"]
        self.idm_p = params["idm"]
        self.mobil_p = params["mobil"]
        self.a_clip = tuple(params["dynamics"]["a_clip_mps2"])
        self.exec_p = params["cav_execution"]
        self.llen = np.asarray(params["geometry"]["lane_lengths_m"],
                               dtype=np.float64)
        self.off = np.asarray(params["geometry"]["lane_offsets_m"],
                              dtype=np.float64)
        names = sorted(vehicles)
        self.names: List[str] = list(names)
        self.lane = np.array([int(vehicles[n]["lane"]) for n in names],
                             dtype=np.int64)
        self.s = np.array([float(vehicles[n]["s"]) for n in names],
                          dtype=np.float64)
        self.v = np.array([float(vehicles[n].get("v", 0.0)) for n in names],
                          dtype=np.float64)
        self.a_prev = np.array(
            [float(vehicles[n].get("a", 0.0)) for n in names],
            dtype=np.float64)
        self.cav = np.array(
            [bool(vehicles[n].get("is_cav", False)) for n in names],
            dtype=bool)
        self.length = np.array(
            [float(vehicles[n].get("length_m",
                                   params["vehicle"]["length_m"]))
             for n in names], dtype=np.float64)
        self.n_input = len(names)
        self.exited: Dict[str, float] = {}     # vid -> exit time
        self.collisions = 0                    # pair-step events
        self.colliding_steps = 0
        self.collisions_by_step: Dict[int, int] = {}
        self.lc_counts = {"merge": 0, "disc_left": 0, "disc_right": 0}
        self.anomalies: Dict[str, int] = {}
        self._lc_last: Dict[str, float] = {}    # vid -> last change time
        self._lc_cooldown = float(
            params["mobil"].get("cooldown_s", 0.0))
        # -- WD-01-PA2G-LC2 structural LC variants (OFF by default; the
        #    section is absent from params_nominal.json, only
        #    variants.variant_params() injects it; with every flag off
        #    the engine path below is identical to the nominal one)
        _lcv = params.get("lc_variants", {}) or {}
        self._va_n = 0
        if _lcv.get("va_enabled"):
            # V-a persistence: a discretionary change fires only after
            # the MOBIL criterion held for T_persist_s = n * dt
            # consecutive qualifying steps.
            self._va_n = max(1, int(round(
                float(_lcv.get("T_persist_s", 0.0)) / self.dt)))
        # V-b manoeuvre time: discretionary candidacy is blocked for
        # T_lc_s after ANY change (mandatory merges keep the nominal
        # cooldown gating only).
        self._vb_tlc = float(_lcv.get("T_lc_s", 0.0)) \
            if _lcv.get("vb_enabled") else 0.0
        self._disc_cooldown = max(self._lc_cooldown, self._vb_tlc)
        self._vc_on = bool(_lcv.get("vc_enabled"))
        self._vc_thr_zero = bool(_lcv.get("vc_threshold_zero", True))
        self._vc_bmerge = float(_lcv.get("b_merge_mps2", 0.0))
        self._persist: Dict[tuple, int] = {}  # (vid, tgt) -> consec count
        self.snapshots: Dict[int, dict] = {}
        self.cmd: Dict[str, float] = {}
        self._snap(0, self.t0)
        self.step_i = 0

    # ----------------------------------------------------------- helpers

    def _snap(self, j: int, t: float) -> None:
        snap = {self.names[r]: (int(self.lane[r]), float(self.s[r]),
                                float(self.v[r]), float(self.a_prev[r]))
                for r in range(len(self.names))}
        snap["__t__"] = t
        self.snapshots[j] = snap

    def _lane_sorted(self) -> Dict[int, np.ndarray]:
        out = {}
        for li in LANES:
            idx = np.where(self.lane == li)[0]
            out[li] = idx[np.argsort(self.s[idx], kind="stable")]
        return out

    def _leaders_vector(self, order) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray]:
        """Own-lane leader row (-1 free), bumper gap (GAP_MAX free),
        leader speed (0 free) -- numpy, once per step."""
        n = len(self.names)
        leader = np.full(n, -1, dtype=np.int64)
        gap = np.full(n, GAP_MAX, dtype=np.float64)
        lv = np.zeros(n, dtype=np.float64)
        for li in LANES:
            o = order[li]
            for j in range(len(o) - 1):
                f, l_ = int(o[j]), int(o[j + 1])
                leader[f] = l_
                gap[f] = self.s[l_] - self.length[l_] - self.s[f]
                lv[f] = self.v[l_]
        return leader, gap, lv

    def _lane_d_sorted(self, li: int) -> Tuple[List[float], List[int]]:
        """Lane li rows sorted by ascending d (d = off - s, downstream
        decreasing): the shared cross-lane alignment axis."""
        rows = [r for r in range(len(self.names)) if self.lane[r] == li]
        rows.sort(key=lambda r: self.off[li] - self.s[r])
        return [float(self.off[li] - self.s[r]) for r in rows], rows

    @staticmethod
    def _d_neighbors(d_sorted: List[float], rows: List[int], d_i: float
                     ) -> Tuple[int, int]:
        """(leader_row, follower_row) in a d-sorted lane (-1 free).
        Leader = largest d < d_i (physically ahead); follower = smallest
        d > d_i."""
        import bisect
        k = bisect.bisect_left(d_sorted, d_i)
        leader = rows[k - 1] if k >= 1 else -1
        follower = rows[k] if k < len(rows) else -1
        return leader, follower

    # ------------------------------------------------------ lane changes

    def _aclip1(self, a: float) -> float:
        """Decision accelerations are the model's OWN accelerations: the
        dynamics path clips IDM output to dynamics.a_clip_mps2 before it
        is used, so the MOBIL/merge incentive terms must see exactly the
        same clipped values (WD-01-PA2G fix: an overlap/collision state
        drives the raw IDM to ~-1e15 through the 1e-6 gap floor, which
        poisoned the incentive gains; bounded accelerations keep the
        textbook after-vs-before differences meaningful)."""
        lo, hi = self.a_clip
        return hi if a > hi else (lo if a < lo else a)

    def _merge_decision(self, r: int, d_sorted1: List[float], rows1: List[int]
                        ) -> Tuple[bool, float]:
        """One mandatory-merge candidate decision (accel -> main_right).
        Returns (accept, d_target).  Safety: ego toward the new leader
        AND the new follower toward ego, both with b_safe(u); incentive:
        MOBIL gain >= accel_threshold*(1 - u)."""
        mp = self.merge_p
        s1z = float(mp["zone_s1_m"])
        Lz = float(mp["zone_len_m"])
        u = min(max(1.0 - (s1z - float(self.s[r])) / Lz, 0.0), 1.0)
        bsafe = float(mp["b_safe_base_mps2"]) + u * (
            float(mp["b_safe_end_mps2"]) - float(mp["b_safe_base_mps2"]))
        d_i = float(self.off[2] - self.s[r])
        lrow, frow = self._d_neighbors(d_sorted1, rows1, d_i)
        # the ego's position expressed in the TARGET lane's s (the d
        # axis is shared; per-lane s offsets differ by up to 2265 m)
        s_ego_t = float(self.s[r]) + float(self.off[1] - self.off[2])
        v0_t = _seg_limit1(1, s_ego_t, self.p)
        if lrow >= 0:
            gap_lead = float(self.s[lrow] - self.length[lrow] - s_ego_t)
            d_gap_lead = float(self.s[lrow] - s_ego_t)
            v_lead = float(self.v[lrow])
        else:
            gap_lead, d_gap_lead, v_lead = GAP_MAX, GAP_MAX, 0.0
        if frow >= 0:
            gap_foll = float(s_ego_t - self.length[r] - self.s[frow])
            d_gap_foll = float(s_ego_t - self.s[frow])
            v_foll = float(self.v[frow])
        else:
            gap_foll, d_gap_foll, v_foll = GAP_MAX, GAP_MAX, 0.0
        # -- safe-gap tests (the v1 merge_acceptance kinematic form on
        #    d-gaps): the lead side needs the IDM headway s0 + v*T; each
        #    side's required deceleration must stay within the
        #    urgency-relaxed b_safe (binding only for the faster side)
        s0i = float(self.idm_p["s0_m"])
        Ti = float(self.idm_p["T_s"])
        floor = float(mp["gap_floor_m"])
        if lrow >= 0 and d_gap_lead < s0i + max(0.0, float(self.v[r]) * Ti):
            return False, d_i
        if frow >= 0 and d_gap_foll < s0i:
            return False, d_i
        if lrow >= 0:
            b_req_lead = (float(self.v[r]) ** 2 - v_lead ** 2) / \
                (2.0 * max(d_gap_lead - s0i, floor))
            if b_req_lead > bsafe:
                return False, d_i
        if frow >= 0:
            b_req_foll = (v_foll ** 2 - float(self.v[r]) ** 2) / \
                (2.0 * max(d_gap_foll - s0i, floor))
            if b_req_foll > bsafe:
                return False, d_i
        a_ego_new = self._aclip1(_idm1(
            float(self.v[r]), gap_lead, float(self.v[r]) - v_lead, v0_t,
            self.idm_p))
        d_new_foll = 0.0
        if frow >= 0:
            v0_f = _seg_limit1(1, float(self.s[frow]), self.p)
            a_f_after = self._aclip1(_idm1(
                v_foll, gap_foll, v_foll - float(self.v[r]), v0_f,
                self.idm_p))
            # follower's current leader for its 'before' term
            lo, hi = self._own_leader_follower(frow)
            if lo >= 0:
                cg = float(self.s[lo] - self.length[lo] - self.s[frow])
                cv = float(self.v[lo])
            else:
                cg, cv = GAP_MAX, 0.0
            a_f_before = self._aclip1(_idm1(
                v_foll, cg, v_foll - cv, v0_f, self.idm_p))
            d_new_foll = a_f_after - a_f_before
        # old follower in lane 2
        d_old_foll = 0.0
        lo2, fo2 = self._own_leader_follower(r)
        if fo2 >= 0:
            v_of = float(self.v[fo2])
            gap_of = float(self.s[r] - self.length[r] - self.s[fo2])
            v0_o = _seg_limit1(2, float(self.s[fo2]), self.p)
            a_of_before = self._aclip1(_idm1(
                v_of, gap_of, v_of - float(self.v[r]), v0_o, self.idm_p))
            if lo2 >= 0:
                gap_ol = float(self.s[lo2] - self.length[lo2] - self.s[fo2])
                v_ol = float(self.v[lo2])
            else:
                gap_ol, v_ol = GAP_MAX, 0.0
            a_of_after = self._aclip1(_idm1(
                v_of, gap_ol, v_of - v_ol, v0_o, self.idm_p))
            d_old_foll = a_of_after - a_of_before
        gain = (a_ego_new - self._a_stay(r)) + \
            float(self.mobil_p["politeness_p"]) * (d_new_foll + d_old_foll)
        thr = float(self.mobil_p["accel_threshold_mps2"]) * (1.0 - u)
        if self._vc_on:
            # WD-01-PA2G-LC2 V-c (strategic early merge): zero incentive
            # threshold (the mandatory-merge rule is applied from the
            # zone start, which the candidacy window already spans) plus
            # the b_merge bias on the MOBIL gain toward main_right.  The
            # safe-gap tests above are UNCHANGED.
            if self._vc_thr_zero:
                thr = 0.0
            gain += self._vc_bmerge
        return gain >= thr, d_i

    def _a_stay(self, r: int) -> float:
        """The 'stay in lane' acceleration: own-lane IDM, min-clamped by
        the lane-end obstacle for accel-lane vehicles (the true stay
        counterfactual)."""
        lo, _ = self._own_leader_follower(r)
        li = int(self.lane[r])
        v0 = _seg_limit1(li, float(self.s[r]), self.p)
        if lo >= 0:
            gap = float(self.s[lo] - self.length[lo] - self.s[r])
            dv = float(self.v[r] - self.v[lo])
        else:
            gap, dv = GAP_MAX, 0.0
        a = _idm1(float(self.v[r]), gap, dv, v0, self.idm_p)
        if li == 2:
            s1z = float(self.merge_p["zone_s1_m"])
            gap_end = max(s1z - float(self.s[r]), 0.3)
            a = min(a, _idm1(float(self.v[r]), gap_end, float(self.v[r]),
                             v0, self.idm_p))
        return self._aclip1(a)

    def _invalidate_nb_cache(self) -> None:
        self._nb = None

    def _neighbors(self):
        """Cached (leader_row, follower_row) arrays (-1 free), rebuilt
        only after a state change (the per-step LC fast path).  Rows are
        sorted by ASCENDING s (s increases downstream): rows[j+1] is
        ahead of rows[j], so rows[j+1] is the LEADER of rows[j].

        WD-01-PA2G fix (2026-09-30): the two assignments were swapped,
        which made every vehicle's 'leader' the vehicle BEHIND it.  The
        decision helpers then evaluated bumper gaps <= -length, the IDM
        gap floor (1e-6) blew them up to ~-1e15 m/s^2 accelerations, and
        the MOBIL incentive became astronomically positive in BOTH
        directions -- the PA2G-A discretionary LC churn (68,614 predicted
        disc LCs vs 243 true at H10)."""
        if getattr(self, "_nb", None) is None:
            n = len(self.names)
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

    def _own_leader_follower(self, r: int) -> Tuple[int, int]:
        """(leader_row, follower_row) of r in its own lane by s order."""
        leader, follower = self._neighbors()
        return int(leader[r]), int(follower[r])

    def _disc_decision(self, r: int, tgt: int, to_right: bool,
                       d_sorted_t: List[float], rows_t: List[int]
                       ) -> Tuple[bool, float]:
        """One discretionary MOBIL candidate (0<->1).  Textbook safety on
        the new follower; keep-right bias in the incentive."""
        li = int(self.lane[r])
        d_i = float(self.off[li] - self.s[r])
        lrow, frow = self._d_neighbors(d_sorted_t, rows_t, d_i)
        # the ego's position in the TARGET lane's s (shared d axis)
        s_ego_t = float(self.s[r]) + float(self.off[tgt] - self.off[li])
        v0_t = _seg_limit1(tgt, s_ego_t, self.p)
        if lrow >= 0:
            gap_lead = float(self.s[lrow] - self.length[lrow] - s_ego_t)
            v_lead = float(self.v[lrow])
        else:
            gap_lead, v_lead = GAP_MAX, 0.0
        if frow >= 0:
            gap_foll = float(s_ego_t - self.length[r] - self.s[frow])
            v_foll = float(self.v[frow])
        else:
            gap_foll, v_foll = GAP_MAX, 0.0
        if gap_lead < float(self.merge_p["gap_floor_m"]):
            return False, d_i
        a_ego_new = self._aclip1(_idm1(
            float(self.v[r]), gap_lead, float(self.v[r]) - v_lead, v0_t,
            self.idm_p))
        a_ego_old = self._a_stay(r)
        # stability guard (params mobil.ego_gain_floor): a discretionary
        # change must not slow the ego itself.  DESIGN DEVIATION (not
        # textbook MOBIL): only active when the floor is > 0; the nominal
        # set keeps it OFF (0.0) so the textbook incentive (threshold +
        # politeness) alone decides.
        _floor = float(self.mobil_p.get("ego_gain_floor_mps2", 0.0))
        if _floor > 0.0 and (a_ego_new - a_ego_old) < _floor:
            return False, d_i
        a_new_after = a_new_before = 0.0
        if frow >= 0:
            v0_f = _seg_limit1(tgt, float(self.s[frow]), self.p)
            a_new_after = self._aclip1(_idm1(
                v_foll, gap_foll, v_foll - float(self.v[r]), v0_f,
                self.idm_p))
            if a_new_after < -float(self.mobil_p["b_safe_mps2"]):
                return False, d_i
            lo, _ = self._own_leader_follower(frow)
            if lo >= 0:
                cg = float(self.s[lo] - self.length[lo] - self.s[frow])
                cv = float(self.v[lo])
            else:
                cg, cv = GAP_MAX, 0.0
            a_new_before = self._aclip1(_idm1(
                v_foll, cg, v_foll - cv, v0_f, self.idm_p))
        lo_s, fo_s = self._own_leader_follower(r)
        a_old_after = a_old_before = 0.0
        if fo_s >= 0:
            v_of = float(self.v[fo_s])
            gap_of = float(self.s[r] - self.length[r] - self.s[fo_s])
            v0_o = _seg_limit1(li, float(self.s[fo_s]), self.p)
            a_old_before = self._aclip1(_idm1(
                v_of, gap_of, v_of - float(self.v[r]), v0_o, self.idm_p))
            if lo_s >= 0:
                gap_ol = float(self.s[lo_s] - self.length[lo_s] - self.s[fo_s])
                v_ol = float(self.v[lo_s])
            else:
                gap_ol, v_ol = GAP_MAX, 0.0
            a_old_after = self._aclip1(_idm1(
                v_of, gap_ol, v_of - v_ol, v0_o, self.idm_p))
        pol = float(self.mobil_p["politeness_p"])
        gain = (a_ego_new - a_ego_old) + pol * (
            (a_new_after - a_new_before) + (a_old_after - a_old_before))
        th = float(self.mobil_p["accel_threshold_mps2"])
        bias = float(self.mobil_p["keep_right_bias_mps2"])
        eff = th - bias if to_right else th + bias
        return gain >= eff, d_i

    # ------------------------------------------------------------- step

    def execution_speed(self, T_t, v_t, gap, dv, v0):
        """The measured SUMO execution rule (partA_act
        results/exec_rule_fit.json, speedMode 31): v' = max(0,
        min(clip(T, v - b_kin*dt, v + a_kin*dt), v + idm_acc*dt)) --
        numpy, vectorized."""
        dt = self.dt
        a_kin = float(self.exec_p["a_kin_max_mps2"])
        b_kin = float(self.exec_p["b_kin_max_mps2"])
        kin = np.clip(T_t, np.maximum(v_t - b_kin * dt, 0.0),
                      v_t + a_kin * dt)
        acc = idm_accel(v_t, gap, dv, v0, self.idm_p)
        return np.maximum(np.minimum(kin, v_t + acc * dt), 0.0)

    def step(self) -> None:
        """One 0.5 s physics step."""
        self.step_i += 1
        j = self.step_i
        t_now = self.t0 + (j - 1) * self.dt
        self._invalidate_nb_cache()
        # -- control tick (1 s cadence, integer seconds, like SUMO);
        #    a command holds for the TWO physics steps of its tick
        if self.control_hook is not None and \
                abs(t_now - round(t_now)) < 1e-9:
            self.cmd = self.control_hook(round(t_now, 6), self) or {}
        # -- lane changes: mandatory merge first (front-most first,
        #    accepted mergers occupy the target lane), then MOBIL
        #    discretionary (front-most first per direction)
        s1z = float(self.merge_p["zone_s1_m"])
        t_change = self.t0 + (j - 1) * self.dt
        if self.lane_changes_enabled:
            s0z = float(self.merge_p["zone_s0_m"])
            cand2 = [r for r in range(len(self.names))
                     if self.lane[r] == 2 and s0z <= self.s[r] <= s1z
                     and t_change - self._lc_last.get(self.names[r],
                                                      -1e9)
                     >= self._lc_cooldown - 1e-9]
            cand2.sort(key=lambda r: -self.s[r])
            d_sorted1, rows1 = self._lane_d_sorted(1)
            for r in cand2:
                accept, _ = self._merge_decision(r, d_sorted1, rows1)
                if accept:
                    self.lane[r] = 1
                    self.s[r] = self.s[r] + self.off[1] - self.off[2]
                    self.lc_counts["merge"] += 1
                    self._lc_last[self.names[r]] = t_change
                    self._invalidate_nb_cache()
                    d_sorted1, rows1 = self._lane_d_sorted(1)
            for src, tgt, to_right in ((1, 0, False), (0, 1, True)):
                v_min = float(self.mobil_p.get("lc_min_speed_mps", 0.0))
                cand = [r for r in range(len(self.names))
                        if self.lane[r] == src
                        and self.v[r] >= v_min
                        and t_change - self._lc_last.get(self.names[r],
                                                         -1e9)
                        >= self._disc_cooldown - 1e-9]
                cand.sort(key=lambda r: -self.s[r])
                d_sorted_t, rows_t = self._lane_d_sorted(tgt)
                for r in cand:
                    accept, _ = self._disc_decision(r, tgt, to_right,
                                                    d_sorted_t, rows_t)
                    if self._va_n > 0:
                        # WD-01-PA2G-LC2 V-a (persistence): fire only
                        # after _va_n consecutive qualifying steps; a
                        # failing step resets the counter.
                        key = (self.names[r], tgt)
                        if not accept:
                            self._persist[key] = 0
                            continue
                        self._persist[key] = \
                            self._persist.get(key, 0) + 1
                        if self._persist[key] < self._va_n:
                            continue
                    if accept:
                        self.lane[r] = tgt
                        self.s[r] = self.s[r] + self.off[tgt] - \
                            self.off[src]
                        key = "disc_left" if tgt == 0 else "disc_right"
                        self.lc_counts[key] += 1
                        self._lc_last[self.names[r]] = t_change
                        if self._va_n > 0:
                            for k2 in [k for k in self._persist
                                       if k[0] == self.names[r]]:
                                self._persist[k2] = 0
                        self._invalidate_nb_cache()
                        d_sorted_t, rows_t = self._lane_d_sorted(tgt)
        # -- IDM accelerations (vectorized)
        order = self._lane_sorted()
        leader, gap, lv = self._leaders_vector(order)
        v0 = segment_limits(self.lane, self.s, self.p)
        dv = np.where(leader >= 0, self.v - lv, 0.0)
        gap_eff = np.where(leader >= 0, gap, GAP_MAX)
        acc = idm_accel(self.v, gap_eff, dv, v0, self.idm_p)
        # -- lane-end standing obstacle on the accel lane
        on2 = self.lane == 2
        if on2.any():
            gap_end = np.maximum(s1z - self.s, 0.3)
            acc_obs = idm_accel(self.v, gap_end, self.v.copy(), v0,
                                self.idm_p)
            acc = np.where(on2, np.minimum(acc, acc_obs), acc)
        acc = np.clip(acc, self.a_clip[0], self.a_clip[1])
        # -- commanded CAVs: the measured execution rule
        if self.cmd:
            rows = [r for r in range(len(self.names))
                    if self.names[r] in self.cmd]
            v_new = np.maximum(self.v + acc * self.dt, 0.0)
            if rows:
                rr = np.asarray(rows)
                T_t = np.array([self.cmd[self.names[r]] for r in rows])
                v_new[rr] = self.execution_speed(
                    T_t, self.v[rr], gap_eff[rr], dv[rr], v0[rr])
        else:
            v_new = np.maximum(self.v + acc * self.dt, 0.0)
        s_new = self.s + v_new * self.dt
        # -- accel-lane end guard (the obstacle rule should have stopped
        #    the vehicle; clamp numerics, never teleport otherwise)
        past = (self.lane == 2) & (s_new > s1z)
        if past.any():
            self.anomalies["accel_end_clamped"] = \
                self.anomalies.get("accel_end_clamped", 0) + int(past.sum())
            s_new = np.where(past, s1z, s_new)
            v_new = np.where(past, 0.0, v_new)
        self.a_prev = (v_new - self.v) / self.dt
        self.v, self.s = v_new, s_new
        # -- collisions: bumper gap < 0 after the update (same-lane
        #    adjacent pairs), counted, never repaired
        order = self._lane_sorted()
        step_coll = 0
        for li in LANES:
            o = order[li]
            if len(o) < 2:
                continue
            sgap = self.s[o[1:]] - self.length[o[1:]] - self.s[o[:-1]]
            step_coll += int((sgap < 0.0).sum())
        if step_coll:
            self.collisions += step_coll
            self.colliding_steps += 1
            self.collisions_by_step[j] = step_coll
        # -- exits: s beyond the lane length leaves the modelled road
        gone = self.s > self.llen[self.lane]
        if gone.any():
            for r in np.where(gone)[0]:
                self.exited[self.names[r]] = self.t0 + j * self.dt
            keep = ~gone
            self.names = [n for n, k in zip(self.names, keep) if k]
            self.lane = self.lane[keep]
            self.s, self.v = self.s[keep], self.v[keep]
            self.a_prev, self.cav = self.a_prev[keep], self.cav[keep]
            self.length = self.length[keep]
        self._snap(j, self.t0 + j * self.dt)

    def run(self, n_steps: int) -> "PartAV2GEngine":
        for _ in range(n_steps):
            if len(self.names) == 0:
                for k in range(self.step_i + 1, n_steps + 1):
                    self.snapshots[k] = {"__t__": self.t0 + k * self.dt}
                self.step_i = n_steps
                return self
            self.step()
        return self

    def insert_vehicle(self, vid: str, lane: int, s: float, v: float,
                       length_m: Optional[float] = None) -> None:
        """WD-01-PA2G-CNT1 arrivals hook: append one vehicle to the live
        state arrays so it participates in the CURRENT step (the caller
        must invoke this between steps, before ``step()``; the vehicle's
        first snapshot is the one that step produces, mirroring a truth
        vehicle's first physical row).  ONLY the CNT1 arrivals injector
        (arrivals.py) calls this -- the nominal path never does, which the
        byte-identity tests enforce."""
        self.names.append(str(vid))
        self.lane = np.append(self.lane, np.int64(lane))
        self.s = np.append(self.s, np.float64(s))
        self.v = np.append(self.v, np.float64(v))
        self.a_prev = np.append(self.a_prev, np.float64(0.0))
        self.cav = np.append(self.cav, False)
        self.length = np.append(
            self.length,
            float(length_m if length_m is not None
                  else self.p["vehicle"]["length_m"]))
        self._invalidate_nb_cache()

    # ----------------------------------------------------------- outputs

    def trajectory_table(self):
        """vid -> list of (lane, s, v, a) or None after the exit step."""
        n_steps = max(k for k in self.snapshots if k != "__t__")
        vids = [n for n in self.snapshots[0] if n != "__t__"]
        out = {vid: [] for vid in vids}
        for k in range(n_steps + 1):
            snap = self.snapshots.get(k, {})
            for vid in vids:
                out[vid].append(snap.get(vid))
        return out

    def n_exited_before(self, t_abs: float) -> int:
        return sum(1 for _, te in self.exited.items() if te <= t_abs)


def free_road_flags(vehicles: Mapping[str, Mapping], zone_extent:
                    Optional[Mapping[int, Sequence[float]]],
                    lane_lengths: Sequence[float]) -> List[str]:
    """Predictor-side n_free_road_assumed rule (params free_road_flag):
    the frontmost tracked vehicle of each lane is flagged when its
    same-lane leader is missing from the sample AND the zone downstream
    boundary of its lane lies short of the lane end (the missing leader
    may lie beyond the zone; optimistic free-road assumption)."""
    out = []
    by_lane: Dict[int, list] = {}
    for vid, veh in vehicles.items():
        li = int(veh["lane"])
        by_lane.setdefault(li, []).append((float(veh["s"]), vid))
    for li, rows in by_lane.items():
        rows.sort()
        frontmost = rows[-1][1]
        zone_end = None if zone_extent is None else float(zone_extent[li][1])
        lane_end = float(lane_lengths[li])
        if zone_end is None or zone_end < lane_end - 1e-6:
            out.append(frontmost)
    return sorted(out)
