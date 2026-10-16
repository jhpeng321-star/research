"""RSU belief state from CAV reports only (C4 Option B world-model check, data assimilation).

The RSU never sees the full traffic state.  Each control tick it receives, per reporting
CAV, a timestamped self report (s, lane, v, a) and the CAV's anonymous same-lane front /
rear neighbour detections within ``SENSOR_RANGE_M`` (bumper gap, relative speed), each
with a receipt age.  :class:`BeliefTracker` turns that stream into a WM-S-frame state:

* CAV tracks are keyed by CAV id (CAVs identify themselves).
* HDV detections carry NO identity.  They are converted to absolute (s, v), de-duplicated
  against CAV reports (a detected vehicle at a reported CAV position is that CAV),
  clustered across reporters, and associated to HDV tracks by a gated Mahalanobis cost
  on (s, v) with a lane-change penalty (Hungarian assignment; greedy fallback).
* Between sightings every track is propagated by an injected ``propagate(state) -> state``
  (default: physics IDM step; :func:`wms_step_propagator` wraps a WM-S model) and an
  EKF-style (s, v) covariance grows with a white-acceleration process noise.
* Tracks are dropped on exit, when their position std exceeds a threshold, or when a CAV
  currently observes the place they are predicted to be as empty (negative information).

Limitation (by construction): a vehicle that no CAV has ever detected does not exist in
the belief.  :meth:`BeliefTracker.coverage` reports which part of the zone was observed
recently and a CAV-only density / flow estimate per lane (an optional entry prior; it is
never used to spawn vehicles here).  Inputs are restricted to the RSU runtime contract
(docs/OBSERVATION_DECISION_CONTRACT.md): no penetration, driver type or global state.
``Detection.debug_id`` / ``CavReport`` fields named ``debug_*`` exist only for offline
evaluation and are never read by association or estimation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from wms.control.contract import (
    MERGE_S0_M,
    MERGE_S1_M,
    PHYSICS_DT_S,
    RSU_MAINLINE_S_M,
    RSU_RAMP_S_M,
    SENSOR_RANGE_M,
    SPEED_LIMIT_MPS,
)
from wms.core.constants import DEFAULT_LENGTH_M, HISTORY_F, HISTORY_H, KIN_F

try:  # optional: exact assignment; greedy nearest-neighbour otherwise
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover - scipy is present in rl_driving
    _lsa = None

Propagator = Callable[[Mapping[str, Tensor]], Mapping[str, Tensor]]

#: Model-frame lane extents (s_down frame, upstream negative): mainline lanes run from the
#: physical lane start (-2645.67) to their end (553.33); the ramp/accel lane ends at 75.44.
DEFAULT_LANE_EXTENTS = ((-2645.67, 553.33), (-2645.67, 553.33), (-378.66, MERGE_S1_M))
GAP_FREE_M = 250.0
GAP_FLOOR_M = 0.3
_INF = 1.0e9
#: Allowed (previous lane, new lane) transitions for association across lanes.
_LANE_MOVES = {(0, 1), (1, 0), (2, 1)}


# ---------------------------------------------------------------------------- inputs
@dataclass
class Detection:
    """One anonymous same-lane neighbour seen by a CAV sensor."""

    gap_m: float                      # bumper gap (front: leader rear to ego front)
    dv_mps: float                     # neighbour speed minus ego speed
    debug_id: Any = None              # EVALUATION ONLY (truth id); never used by the tracker


@dataclass
class CavReport:
    """Timestamped CAV self state + same-lane detections, as received by the RSU."""

    cav_id: str
    s_m: float
    lane: int
    v_mps: float
    a_mps2: float
    age_s: float = 0.0                # receipt age at the RSU tick (measurement at t - age)
    length_m: float = DEFAULT_LENGTH_M
    front: Optional[Detection] = None
    rear: Optional[Detection] = None
    sensor_range_m: float = SENSOR_RANGE_M


@dataclass
class RsuDetection:
    """One vehicle seen by the RSU's own sensors inside an RSU zone (anonymous: no id
    across frames; CAVs are recognised only by matching their own reports)."""

    s_m: float
    lane: int
    v_mps: float
    a_mps2: float = 0.0
    age_s: float = 0.0
    debug_id: Any = None              # EVALUATION ONLY (truth id); never used by the tracker


@dataclass(frozen=True)
class RsuZones:
    """RSU sensor coverage per WM-frame lane, half-open ``[lo, hi)`` (contract constants:
    800 m upstream .. ~150 m downstream on the mainline, the whole ramp on lane 2)."""

    by_lane: tuple = ((0, RSU_MAINLINE_S_M), (1, RSU_MAINLINE_S_M), (2, RSU_RAMP_S_M))

    def contains(self, lane: int, s: float) -> bool:
        return any(ln == lane and lo <= s < hi for ln, (lo, hi) in self.by_lane)


@dataclass
class BeliefConfig:
    max_slots: int = 256
    dt_s: float = PHYSICS_DT_S
    lane_extents: tuple = DEFAULT_LANE_EXTENTS
    exit_s_m: float = DEFAULT_LANE_EXTENTS[0][1]
    assumed_length_m: float = DEFAULT_LENGTH_M
    # measurement noise (1 sigma)
    self_pos_std_m: float = 0.3
    self_speed_std_mps: float = 0.1
    det_pos_std_m: float = 0.6        # gap noise + unknown neighbour length
    det_speed_std_mps: float = 0.3
    rsu_pos_std_m: float = 0.5
    rsu_speed_std_mps: float = 0.2
    # RSU own sensors; False = CAV-reports-only ablation
    rsu_sensors: bool = True
    rsu_zones: RsuZones = field(default_factory=RsuZones)
    age_accel_std_mps2: float = 1.5   # unknown accel over the receipt age
    # process noise: continuous white acceleration spectral density (m^2/s^3)
    q_hdv: float = 0.6
    q_cav: float = 0.6
    init_speed_std_mps: float = 1.0
    # association
    gate_chi2: float = 13.8           # ~99.9 % for 2 dof
    lane_change_cost: float = 6.0
    cluster_ds_m: float = 3.0
    cluster_dv_mps: float = 2.0
    cav_dedupe_m: float = 4.0
    negative_margin_m: float = 3.0
    # track management
    max_pos_std_m: float = 40.0
    cav_timeout_s: float = 5.0
    # history / coverage
    history_steps: int = 241          # 120 s at 0.5 s (incl. t0)
    history_valid_age_s: float = 1.0  # history sample valid if seen within this age
    coverage_cell_m: float = 5.0
    coverage_window_s: float = 5.0


# ---------------------------------------------------------------------------- tracks
@dataclass
class Track:
    tid: int
    slot: int
    is_cav: bool
    s: float
    v: float
    a: float
    lane: int
    length: float
    P: np.ndarray
    born_t: float
    last_seen_t: float
    cav_id: Optional[str] = None
    hits: int = 1
    debug_id: Any = None              # evaluation only: last associated truth id
    debug_switches: int = 0           # evaluation only: identity switches observed

    @property
    def pos_std(self) -> float:
        return float(math.sqrt(max(self.P[0, 0], 0.0)))


@dataclass
class _Meas:
    s: float
    v: float
    lane: int
    R: np.ndarray
    debug_ids: list = field(default_factory=list)


# ---------------------------------------------------------------------------- physics
def idm_propagate(state: Mapping[str, Tensor], *, dt_s: float = PHYSICS_DT_S,
                  v0: float = SPEED_LIMIT_MPS, a_max: float = 2.49, b_comf: float = 4.2,
                  headway_s: float = 1.135, s0_m: float = 2.6,
                  merge_s0_m: float = MERGE_S0_M, merge_s1_m: float = MERGE_S1_M
                  ) -> dict[str, Tensor]:
    """Default belief propagator: one IDM step (SUMO vType means) on a ``[B,K]`` state.

    Same-lane leader by front-bumper position; the ramp end is a standing obstacle;
    ramp vehicles inside the merge zone move to lane 1 when both lane-1 bumper gaps
    exceed 3 m.  No discretionary lane changes.
    """
    s, v = state["s_m"], state["v_mps"]
    lane = state["lane"].long()
    present = state.get("present", torch.ones_like(s, dtype=torch.bool)).bool()
    length = state.get("length_m", torch.full_like(s, DEFAULT_LENGTH_M))

    def neighbour(query_lane: Tensor, ahead: bool) -> tuple[Tensor, Tensor]:
        ds = s.unsqueeze(-2) - s.unsqueeze(-1)                  # [B,i,j] = s_j - s_i
        other = present.unsqueeze(-2) & (lane.unsqueeze(-2) == query_lane.unsqueeze(-1))
        if ahead:
            ok, bump = other & (ds > 0), ds - length.unsqueeze(-2)
        else:
            ok, bump = other & (ds < 0), -ds - length.unsqueeze(-1)
        gaps = torch.where(ok, bump, torch.full_like(ds, _INF))
        gap, idx = gaps.min(-1)
        has = gap < _INF / 2
        return (torch.where(has, gap.clamp_min(GAP_FLOOR_M), torch.full_like(gap, GAP_FREE_M)),
                torch.where(has, torch.gather(v, -1, idx), v))

    gap, lead_v = neighbour(lane, True)
    end_gap = (merge_s1_m - s).clamp_min(GAP_FLOOR_M)
    obstacle = (lane == 2) & (end_gap < gap)
    gap = torch.where(obstacle, end_gap, gap)
    lead_v = torch.where(obstacle, torch.zeros_like(lead_v), lead_v)
    desired = s0_m + (v * headway_s + v * (v - lead_v)
                      / (2.0 * math.sqrt(a_max * b_comf))).clamp_min(0.0)
    accel = a_max * (1.0 - (v / v0).pow(4) - (desired / gap).pow(2))
    accel = torch.where(present, accel.clamp(-9.0, 3.0), torch.zeros_like(accel))
    next_v = (v + accel * dt_s).clamp_min(0.0)
    next_s = s + 0.5 * (v + next_v) * dt_s
    one = torch.ones_like(lane)
    next_lane = lane
    if bool((present & (lane == 2)).any()):
        f_gap, _ = neighbour(one, True)
        r_gap, _ = neighbour(one, False)
        merge = (present & (lane == 2) & (s >= merge_s0_m) & (f_gap > 3.0) & (r_gap > 3.0))
        next_lane = torch.where(merge, one, lane)
    out = dict(state)
    out.update({"s_m": next_s, "v_mps": next_v, "a_mps2": accel, "lane": next_lane,
                "present": present})
    return out


def hold_speed_cav_controller(dt_s: float = PHYSICS_DT_S) -> Callable:
    """WM-S ``cav_controller`` for belief propagation: unknown CAV commands between reports
    are extrapolated at constant speed in the current lane -> ``[B,K,4]`` (s, lane, v, a)."""
    def controller(state: Mapping[str, Tensor]) -> Tensor:
        s, v = state["s_m"], state["v_mps"]
        return torch.stack((s + v * dt_s, state["lane"].to(s.dtype), v,
                            torch.zeros_like(v)), dim=-1)
    return controller


def wms_step_propagator(model: Any, **step_kwargs: Any) -> Propagator:
    """Wrap a WM-S model's ``step`` (duck-typed; no import of ``wms.model``) as a propagator.

    ``step_kwargs`` are forwarded (e.g. ``driver_params``, ``cav_controller``; default
    :func:`hold_speed_cav_controller`).  Histories are not threaded, so
    history-conditioned checkpoints fall back to their empty-history defaults.
    """
    step_kwargs.setdefault("cav_controller", hold_speed_cav_controller())

    def propagate(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
        with torch.no_grad():
            out = model.step(dict(state), **step_kwargs)
        nxt = dict(state)
        nxt.update({"s_m": out["next_s_m"], "v_mps": out["next_v_mps"],
                    "a_mps2": out["accel_mps2"], "lane": out["next_lane"].long(),
                    "present": out["next_present"].bool()})
        return nxt
    return propagate


# ---------------------------------------------------------------------------- tracker
class BeliefTracker:
    """Multi-vehicle belief over a fixed set of WM-S slots, fed by CAV reports only."""

    def __init__(self, config: Optional[BeliefConfig] = None,
                 propagate: Optional[Propagator] = None) -> None:
        self.cfg = config or BeliefConfig()
        self.propagate = propagate or (lambda st: idm_propagate(st, dt_s=self.cfg.dt_s))
        self.tracks: dict[int, Track] = {}
        self.t_s: Optional[float] = None
        self._next_tid = 0
        self._free_slots = list(range(self.cfg.max_slots))
        self._history: list[dict[str, np.ndarray]] = []
        self.stats = {"births": 0, "drops_exit": 0, "drops_uncertain": 0,
                      "drops_negative": 0, "drops_cav_timeout": 0, "slot_overflow": 0,
                      "associations": 0, "reports": 0}
        cells = []
        for lo, hi in self.cfg.lane_extents:
            n = max(0, int(math.ceil((hi - lo) / self.cfg.coverage_cell_m)))
            cells.append(np.full(n, -np.inf))
        self._cover = cells
        self._observed_intervals: list[tuple[int, float, float]] = []

    # ------------------------------------------------------------------ public API
    def update(self, t_s: float, reports: Sequence[CavReport],
               rsu_detections: Optional[Sequence[RsuDetection]] = None) -> None:
        """Assimilate one RSU tick: propagate to ``t_s`` then fuse CAV ``reports`` and
        (if ``config.rsu_sensors``) the RSU's own frame ``rsu_detections``.  ``None`` =
        no RSU frame this tick; an empty list = the RSU saw its zones empty."""
        cfg = self.cfg
        if self.t_s is not None:
            n = int(round((t_s - self.t_s) / cfg.dt_s))
            if n < 0:
                raise ValueError("belief ticks must be non-decreasing")
            for i in range(n):
                self._predict()
                if i < n - 1:
                    self._record(self.t_s + (i + 1) * cfg.dt_s)
        self.t_s = float(t_s)
        self.stats["reports"] += len(reports)
        self._fuse(reports, rsu_detections if cfg.rsu_sensors else None)
        self._record(self.t_s)

    def state(self) -> dict[str, Tensor]:
        """WM-S-frame state ``[1, K]`` (keys of the contract; rollout-ready).  Slot ``j``
        holds the track with ``track.slot == j``; free slots are absent."""
        return self._pack({tr.slot: tr for tr in self.tracks.values()}, self.cfg.max_slots)

    def _pack(self, by_index: Mapping[int, Track], k: int) -> dict[str, Tensor]:
        s = torch.zeros(1, k)
        v = torch.zeros(1, k)
        a = torch.zeros(1, k)
        lane = torch.full((1, k), -1, dtype=torch.long)
        present = torch.zeros(1, k, dtype=torch.bool)
        is_cav = torch.zeros(1, k, dtype=torch.bool)
        length = torch.full((1, k), DEFAULT_LENGTH_M)
        for j, tr in by_index.items():
            s[0, j], v[0, j], a[0, j] = tr.s, tr.v, tr.a
            lane[0, j], present[0, j], is_cav[0, j] = tr.lane, True, tr.is_cav
            length[0, j] = tr.length
        return {"s_m": s, "v_mps": v, "a_mps2": a, "lane": lane, "present": present,
                "is_cav": is_cav, "length_m": length,
                "exit_s_m": torch.full((1, k), float(self.cfg.exit_s_m))}

    def uncertainty(self) -> dict[str, Tensor]:
        """Per-slot ``pos_std_m``, ``speed_std_mps``, ``last_seen_age_s``, ``track_id``."""
        k = self.cfg.max_slots
        pos = torch.zeros(1, k)
        spd = torch.zeros(1, k)
        age = torch.full((1, k), float("inf"))
        tid = torch.full((1, k), -1, dtype=torch.long)
        for tr in self.tracks.values():
            pos[0, tr.slot] = tr.pos_std
            spd[0, tr.slot] = math.sqrt(max(tr.P[1, 1], 0.0))
            age[0, tr.slot] = (self.t_s or 0.0) - tr.last_seen_t
            tid[0, tr.slot] = tr.tid
        return {"pos_std_m": pos, "speed_std_mps": spd, "last_seen_age_s": age,
                "track_id": tid}

    def history(self, *, history_s: Optional[float] = None,
                history_stride_s: float = 1.0) -> dict[str, Any]:
        """Slot-indexed pre-t0 history (schema of :mod:`wms.eval.history_inputs`).

        ``kin [T,K,7]`` (s, lane one-hot, v, a, length), ``mask [T,K]`` (slot holds a
        track seen within ``history_valid_age_s``: propagated-only samples are masked so
        the driver encoder never learns from the belief's own IDM guesses),
        ``target_mask [T-1,K]`` (same track continues in the slot), ``ids``
        (``slotN``), and the duration metadata required by long-history checkpoints.
        """
        recs = self._history
        if not recs:
            raise ValueError("no history recorded yet")
        kin = np.stack([r["kin"] for r in recs])
        mask = np.stack([r["valid"] for r in recs])
        tids = np.stack([r["tid"] for r in recs])
        target = (tids[1:] == tids[:-1]) & (tids[1:] >= 0)
        span = (len(recs) - 1) * self.cfg.dt_s
        want = span if history_s is None else float(history_s)
        return {"kin": kin, "mask": mask, "target_mask": target,
                "ids": [f"slot{j}" for j in range(self.cfg.max_slots)],
                "history_s": want, "history_stride_s": float(history_stride_s),
                "dt_s": self.cfg.dt_s, "available_span_s": span,
                "run_start_limited": bool(span + 1e-9 < want), "gap": np.stack(
                    [r["gap"] for r in recs])}

    def driver_history(self, history_len: int = HISTORY_H) -> tuple[Tensor, Tensor]:
        """Short rolling history ``[1,K,H,HISTORY_F]`` + mask ``[1,K,H]`` for
        ``WmSModel.rollout(history=..., history_mask=...)`` /
        ``driver_history=...`` (features as ``wms.model.history.history_step_features``:
        v/30, a/3, lane one-hot, min(gap, 250)/100; continuity via the track id)."""
        recs = self._history[-history_len:]
        k = self.cfg.max_slots
        feat = torch.zeros(1, k, history_len, HISTORY_F)
        hmask = torch.zeros(1, k, history_len, dtype=torch.bool)
        if not recs:
            return feat, hmask
        now_tid = recs[-1]["tid"]
        same = np.ones(k, dtype=bool)
        for back, rec in enumerate(reversed(recs)):
            same &= rec["tid"] == now_tid
            col = history_len - 1 - back
            kin = torch.as_tensor(rec["kin"])
            gap = torch.as_tensor(rec["gap"]).clamp(GAP_FLOOR_M, GAP_FREE_M)
            feat[0, :, col] = torch.cat((kin[:, 4:5] / 30.0, kin[:, 5:6] / 3.0, kin[:, 1:4],
                                         (gap / 100.0).unsqueeze(-1)), dim=-1)
            hmask[0, :, col] = torch.as_tensor(rec["valid"] & same & (now_tid >= 0))
        return feat, hmask

    def rollout_inputs(self) -> dict[str, Any]:
        """``state`` + ``history``/``history_mask`` ready for ``WmSModel.rollout``."""
        hist, hmask = self.driver_history()
        return {"state": self.state(), "history": hist, "history_mask": hmask}

    def coverage(self, *, window_s: Optional[float] = None,
                 zone: Optional[tuple[float, float]] = None) -> dict[str, Any]:
        """Fraction of each lane's length observed within ``window_s`` + CAV-only density.

        ``zone`` restricts the mainline/ramp cells to ``[lo, hi]``.  Density per lane is
        (recently seen tracks inside currently observed intervals) / observed length;
        flow = density x mean observed speed.  Optional entry prior; not a census.
        """
        cfg = self.cfg
        window = cfg.coverage_window_s if window_s is None else float(window_s)
        t = self.t_s if self.t_s is not None else 0.0
        per_lane, tot_obs, tot = [], 0, 0
        for ln, (lo, hi) in enumerate(cfg.lane_extents):
            cells = self._cover[ln]
            if cells.size == 0:
                per_lane.append(float("nan"))
                continue
            centres = lo + (np.arange(cells.size) + 0.5) * cfg.coverage_cell_m
            keep = np.ones(cells.size, dtype=bool) if zone is None else \
                (centres >= zone[0]) & (centres <= zone[1])
            if not keep.any():
                per_lane.append(float("nan"))
                continue
            obs = (t - cells[keep]) <= window + 1e-9
            per_lane.append(float(obs.mean()))
            tot_obs += int(obs.sum())
            tot += int(keep.sum())
        density, flow = [], []
        for ln in range(len(cfg.lane_extents)):
            ivs = [(lo, hi) for lane, lo, hi in self._observed_intervals if lane == ln]
            obs_len = _union_length(ivs)
            if obs_len <= 0:
                density.append(float("nan"))
                flow.append(float("nan"))
                continue
            inside = [tr for tr in self.tracks.values() if tr.lane == ln
                      and t - tr.last_seen_t <= cfg.history_valid_age_s
                      and any(lo <= tr.s <= hi for lo, hi in ivs)]
            k_vpm = len(inside) / obs_len
            vbar = float(np.mean([tr.v for tr in inside])) if inside else float("nan")
            density.append(1000.0 * k_vpm)
            flow.append(3600.0 * k_vpm * vbar if inside else 0.0)
        ages = [t - tr.last_seen_t for tr in self.tracks.values() if not tr.is_cav]
        return {"fraction_observed": (tot_obs / tot) if tot else float("nan"),
                "fraction_by_lane": per_lane, "window_s": window,
                "density_vpkm_by_lane": density, "flow_vph_by_lane": flow,
                "n_tracks": len(self.tracks),
                "n_cav_tracks": sum(tr.is_cav for tr in self.tracks.values()),
                "mean_hdv_age_s": float(np.mean(ages)) if ages else float("nan")}

    # ------------------------------------------------------------------ internals
    def _q(self, tr: Track) -> np.ndarray:
        q, dt = (self.cfg.q_cav if tr.is_cav else self.cfg.q_hdv), self.cfg.dt_s
        return q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])

    def _predict(self) -> None:
        if not self.tracks:
            return
        # propagate the compact occupied-track state [1, N] (cheaper than all K slots)
        order = sorted(self.tracks.values(), key=lambda tr: tr.slot)
        st = self._pack(dict(enumerate(order)), len(order))
        nxt = self.propagate(st)
        dt = self.cfg.dt_s
        F = np.array([[1.0, dt], [0.0, 1.0]])
        n_s, n_v = nxt["s_m"][0].tolist(), nxt["v_mps"][0].tolist()
        n_l = nxt["lane"][0].long().tolist()
        n_p = nxt.get("present", st["present"])[0].bool().tolist()
        n_a = nxt["a_mps2"][0].tolist() if "a_mps2" in nxt else None
        for j, tr in enumerate(order):
            tid = tr.tid
            if not n_p[j]:
                self._drop(tid, "drops_exit")
                continue
            new_v = float(n_v[j])
            tr.a = float(n_a[j]) if n_a is not None else (new_v - tr.v) / dt
            tr.s, tr.v, tr.lane = float(n_s[j]), new_v, int(n_l[j])
            tr.P = F @ tr.P @ F.T + self._q(tr)
            if tr.s > self.cfg.exit_s_m:
                self._drop(tid, "drops_exit")
            elif tr.pos_std > self.cfg.max_pos_std_m:
                self._drop(tid, "drops_uncertain")
        self.t_s = (self.t_s or 0.0) + dt

    def _new_track(self, m: _Meas, *, is_cav: bool, a: float, length: float,
                   cav_id: Optional[str] = None) -> Optional[Track]:
        if not self._free_slots:
            self.stats["slot_overflow"] += 1
            return None
        slot = self._free_slots.pop(0)
        P = m.R.copy()
        P[1, 1] = max(P[1, 1], self.cfg.init_speed_std_mps ** 2 if not is_cav else P[1, 1])
        tr = Track(tid=self._next_tid, slot=slot, is_cav=is_cav, s=m.s, v=m.v, a=a,
                   lane=m.lane, length=length, P=P, born_t=self.t_s or 0.0,
                   last_seen_t=self.t_s or 0.0, cav_id=cav_id,
                   debug_id=_majority(m.debug_ids))
        self._next_tid += 1
        self.tracks[tr.tid] = tr
        self.stats["births"] += 1
        return tr

    def _drop(self, tid: int, reason: str) -> None:
        tr = self.tracks.pop(tid)
        self._free_slots.append(tr.slot)
        self._free_slots.sort()
        self.stats[reason] += 1

    def _kf_update(self, tr: Track, m: _Meas) -> None:
        x = np.array([tr.s, tr.v])
        y = np.array([m.s, m.v]) - x
        S = tr.P + m.R
        K = tr.P @ np.linalg.inv(S)
        x = x + K @ y
        tr.P = (np.eye(2) - K) @ tr.P
        tr.P = 0.5 * (tr.P + tr.P.T)
        tr.s, tr.v = float(x[0]), max(float(x[1]), 0.0)
        tr.lane = m.lane
        tr.last_seen_t = self.t_s or 0.0
        tr.hits += 1

    def _cost_matrix(self, hdv: list[Track], meas: list[_Meas]) -> np.ndarray:
        """Gated squared Mahalanobis distance on (s, v) + lane-change penalty."""
        if not hdv or not meas:
            return np.zeros((len(hdv), len(meas)))
        ts = np.array([[tr.s, tr.v] for tr in hdv])
        tl = np.array([tr.lane for tr in hdv])
        tP = np.stack([tr.P for tr in hdv])
        ms = np.array([[m.s, m.v] for m in meas])
        ml = np.array([m.lane for m in meas])
        mR = np.stack([m.R for m in meas])
        S = tP[:, None] + mR[None]                                   # [T,M,2,2]
        y = ms[None] - ts[:, None]                                   # [T,M,2]
        det = S[..., 0, 0] * S[..., 1, 1] - S[..., 0, 1] * S[..., 1, 0]
        d2 = (S[..., 1, 1] * y[..., 0] ** 2 - 2.0 * S[..., 0, 1] * y[..., 0] * y[..., 1]
              + S[..., 0, 0] * y[..., 1] ** 2) / np.maximum(det, 1e-12)
        move = np.zeros_like(d2, dtype=bool)
        for a, b in _LANE_MOVES:
            move |= (tl[:, None] == a) & (ml[None] == b)
        same = tl[:, None] == ml[None]
        d2 = d2 + np.where(same, 0.0, self.cfg.lane_change_cost)
        return np.where((same | move) & (d2 <= self.cfg.gate_chi2), d2, _INF)

    def _fuse(self, reports: Sequence[CavReport],
              rsu: Optional[Sequence[RsuDetection]] = None) -> None:
        cfg = self.cfg
        R_self = np.diag([cfg.self_pos_std_m ** 2, cfg.self_speed_std_mps ** 2])
        cav_by_id = {tr.cav_id: tr for tr in self.tracks.values() if tr.is_cav}
        dets: list[_Meas] = []
        self._observed_intervals = []
        empty: list[tuple[int, float, float]] = []
        for rep in reports:
            age = max(float(rep.age_s), 0.0)
            s_c = rep.s_m + rep.v_mps * age + 0.5 * rep.a_mps2 * age ** 2
            v_c = max(rep.v_mps + rep.a_mps2 * age, 0.0)
            R_age = np.diag([(0.5 * cfg.age_accel_std_mps2 * age ** 2) ** 2,
                             (cfg.age_accel_std_mps2 * age) ** 2])
            m = _Meas(s_c, v_c, int(rep.lane), R_self + R_age)
            tr = cav_by_id.get(rep.cav_id)
            if tr is None:
                tr = self._new_track(m, is_cav=True, a=rep.a_mps2, length=rep.length_m,
                                     cav_id=rep.cav_id)
                if tr is not None:
                    cav_by_id[rep.cav_id] = tr
            else:
                self._kf_update(tr, m)
                tr.a = rep.a_mps2
            R_det = np.diag([cfg.det_pos_std_m ** 2, cfg.det_speed_std_mps ** 2]) + R_age
            rng = float(rep.sensor_range_m)
            hi = s_c + rng + cfg.assumed_length_m
            lo = s_c - rep.length_m - rng
            if rep.front is not None:
                s_f = s_c + rep.front.gap_m + cfg.assumed_length_m
                dets.append(_Meas(s_f, max(v_c + rep.front.dv_mps, 0.0), int(rep.lane),
                                  R_det.copy(), [rep.front.debug_id]))
                hi = s_f
            if rep.rear is not None:
                s_r = s_c - rep.length_m - rep.rear.gap_m
                dets.append(_Meas(s_r, max(v_c + rep.rear.dv_mps, 0.0), int(rep.lane),
                                  R_det.copy(), [rep.rear.debug_id]))
                lo = s_r
            self._observed_intervals.append((int(rep.lane), lo, hi))
            empty.append((int(rep.lane), lo + cfg.negative_margin_m,
                          hi - cfg.negative_margin_m))
        if rsu is not None:
            R_rsu = np.diag([cfg.rsu_pos_std_m ** 2, cfg.rsu_speed_std_mps ** 2])
            for d in rsu:
                age = max(float(d.age_s), 0.0)
                R_age = np.diag([(0.5 * cfg.age_accel_std_mps2 * age ** 2) ** 2,
                                 (cfg.age_accel_std_mps2 * age) ** 2])
                dets.append(_Meas(d.s_m + d.v_mps * age + 0.5 * d.a_mps2 * age ** 2,
                                  max(d.v_mps + d.a_mps2 * age, 0.0), int(d.lane),
                                  R_rsu + R_age, [d.debug_id]))
            for lane, (lo, hi) in cfg.rsu_zones.by_lane:
                self._observed_intervals.append((int(lane), lo, hi))
                empty.append((int(lane), lo + cfg.negative_margin_m, hi - cfg.negative_margin_m))
        # detections of reported/tracked CAVs are those CAVs, not HDVs
        cavs = [tr for tr in self.tracks.values() if tr.is_cav]
        dets = [d for d in dets if not any(c.lane == d.lane and abs(c.s - d.s) < cfg.cav_dedupe_m
                                           for c in cavs)]
        meas = _cluster(dets, cfg.cluster_ds_m, cfg.cluster_dv_mps)
        hdv = [tr for tr in self.tracks.values() if not tr.is_cav]
        matched_tracks: set[int] = set()
        pairs = _assign(self._cost_matrix(hdv, meas))
        used = set()
        for i, j in pairs:
            tr, m = hdv[i], meas[j]
            self._kf_update(tr, m)
            new_id = _majority(m.debug_ids)
            if tr.debug_id is not None and new_id is not None and new_id != tr.debug_id:
                tr.debug_switches += 1
            if new_id is not None:
                tr.debug_id = new_id
            matched_tracks.add(tr.tid)
            used.add(j)
            self.stats["associations"] += 1
        for j, m in enumerate(meas):
            if j not in used:
                self._new_track(m, is_cav=False, a=0.0, length=cfg.assumed_length_m)
        # negative information: unmatched HDV tracks predicted inside observed-empty space
        for tr in hdv:
            if tr.tid in matched_tracks or tr.tid not in self.tracks:
                continue
            if any(lane == tr.lane and lo < tr.s < hi for lane, lo, hi in empty):
                self._drop(tr.tid, "drops_negative")
        t = self.t_s or 0.0
        for tid in [tid for tid, tr in self.tracks.items()
                    if tr.is_cav and t - tr.last_seen_t > cfg.cav_timeout_s]:
            self._drop(tid, "drops_cav_timeout")
        for lane, lo, hi in self._observed_intervals:
            self._mark_cover(lane, lo, hi, t)

    def _mark_cover(self, lane: int, lo: float, hi: float, t: float) -> None:
        if not 0 <= lane < len(self._cover) or self._cover[lane].size == 0:
            return
        base = self.cfg.lane_extents[lane][0]
        c = self.cfg.coverage_cell_m
        i0 = max(0, int(math.ceil((lo - base) / c - 0.5)))
        i1 = min(self._cover[lane].size, int(math.floor((hi - base) / c - 0.5)) + 1)
        if i1 > i0:
            self._cover[lane][i0:i1] = t

    def _record(self, t: float) -> None:
        k = self.cfg.max_slots
        kin = np.zeros((k, KIN_F), dtype=np.float32)
        valid = np.zeros(k, dtype=bool)
        tid = np.full(k, -1, dtype=np.int64)
        gap = np.full(k, GAP_FREE_M, dtype=np.float32)
        trs = list(self.tracks.values())
        for tr in trs:
            j = tr.slot
            kin[j, 0] = tr.s
            if 0 <= tr.lane <= 2:
                kin[j, 1 + tr.lane] = 1.0
            kin[j, 4], kin[j, 5], kin[j, 6] = tr.v, tr.a, tr.length
            valid[j] = (t - tr.last_seen_t) <= self.cfg.history_valid_age_s + 1e-9
            tid[j] = tr.tid
            ahead = [o.s - o.length - tr.s for o in trs if o.lane == tr.lane and o.s > tr.s]
            if ahead:
                gap[j] = max(min(ahead), GAP_FLOOR_M)
        self._history.append({"kin": kin, "valid": valid, "tid": tid, "gap": gap})
        if len(self._history) > self.cfg.history_steps:
            del self._history[0]


# ---------------------------------------------------------------------------- helpers
def _majority(ids: Sequence[Any]) -> Any:
    ids = [i for i in ids if i is not None]
    return max(set(ids), key=ids.count) if ids else None


def _union_length(ivs: Sequence[tuple[float, float]]) -> float:
    total, cur_lo, cur_hi = 0.0, None, None
    for lo, hi in sorted(ivs):
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None:
                total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    if cur_hi is not None:
        total += cur_hi - cur_lo
    return total


def _cluster(dets: list[_Meas], ds_m: float, dv_mps: float) -> list[_Meas]:
    """Merge detections of the same vehicle by several CAVs (inverse-variance mean)."""
    out: list[_Meas] = []
    for d in sorted(dets, key=lambda m: (m.lane, m.s)):
        last = out[-1] if out else None
        if last is not None and last.lane == d.lane and abs(d.s - last.s) < ds_m \
                and abs(d.v - last.v) < dv_mps:
            w1, w2 = 1.0 / np.diag(last.R), 1.0 / np.diag(d.R)
            mean = (w1 * np.array([last.s, last.v]) + w2 * np.array([d.s, d.v])) / (w1 + w2)
            last.s, last.v = float(mean[0]), float(mean[1])
            last.R = np.diag(1.0 / (w1 + w2))
            last.debug_ids.extend(d.debug_ids)
        else:
            out.append(_Meas(d.s, d.v, d.lane, d.R.copy(), list(d.debug_ids)))
    return out


def _assign(cost: np.ndarray) -> list[tuple[int, int]]:
    """Gated min-cost assignment (``>= _INF`` = forbidden)."""
    if cost.size == 0:
        return []
    if _lsa is not None:
        rows, cols = _lsa(np.minimum(cost, _INF))
        return [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < _INF]
    pairs, used_r, used_c = [], set(), set()
    for flat in np.argsort(cost, axis=None):
        i, j = np.unravel_index(flat, cost.shape)
        if cost[i, j] >= _INF:
            break
        if i not in used_r and j not in used_c:
            pairs.append((int(i), int(j)))
            used_r.add(i)
            used_c.add(j)
    return pairs


# ---------------------------------------------------------------------------- oracle / synth
def detection_mask(state: Mapping[str, Tensor], *,
                   sensor_range_m: float = SENSOR_RANGE_M) -> Tensor:
    """``[B,K]`` HDVs that are the nearest same-lane front/rear neighbour (bumper gap
    <= ``sensor_range_m``) of some present CAV.  Evaluation helper on a TRUE state."""
    s, lane = state["s_m"], state["lane"].long()
    present = state.get("present", torch.ones_like(s, dtype=torch.bool)).bool()
    is_cav = state.get("is_cav", torch.zeros_like(present)).bool() & present
    length = state.get("length_m", torch.full_like(s, DEFAULT_LENGTH_M))
    ds = s.unsqueeze(-2) - s.unsqueeze(-1)                       # [B,i,j] = s_j - s_i
    same = present.unsqueeze(-2) & (lane.unsqueeze(-2) == lane.unsqueeze(-1))
    eye = torch.eye(s.shape[-1], dtype=torch.bool, device=s.device)
    same = same & ~eye
    out = torch.zeros_like(present)
    for ahead in (True, False):
        bump = (ds - length.unsqueeze(-2)) if ahead else (-ds - length.unsqueeze(-1))
        ok = same & ((ds > 0) if ahead else (ds < 0))
        gaps = torch.where(ok, bump, torch.full_like(ds, _INF))
        gap, idx = gaps.min(-1)                                   # nearest for each i
        hit = is_cav & (gap <= sensor_range_m)
        for b in range(s.shape[0]):
            sel = idx[b][hit[b]]
            out[b, sel] = True
    return out & present & ~is_cav


def oracle_mask_state(true_state: Mapping[str, Tensor],
                      cav_detect_mask: Tensor) -> dict[str, Tensor]:
    """"Sensed-only" ablation: keep CAVs + currently detected HDVs, mark the rest absent."""
    out = dict(true_state)
    present = true_state.get("present",
                             torch.ones_like(true_state["s_m"], dtype=torch.bool)).bool()
    is_cav = true_state.get("is_cav", torch.zeros_like(present)).bool()
    out["present"] = present & (is_cav | cav_detect_mask.bool())
    return out


def reports_from_state(state: Mapping[str, Tensor], *, batch: int = 0,
                       age_s: float = 0.0,
                       sensor_range_m: float = SENSOR_RANGE_M) -> list[CavReport]:
    """Synthesise RSU reports from a TRUE WM-frame state (tests / SUMO glue).

    ``debug_id`` is the true slot index (evaluation only)."""
    s = state["s_m"][batch].tolist()
    v = state["v_mps"][batch].tolist()
    a = state.get("a_mps2", torch.zeros_like(state["s_m"]))[batch].tolist()
    lane = state["lane"][batch].long().tolist()
    present = state.get("present", torch.ones_like(state["s_m"], dtype=torch.bool))[
        batch].bool().tolist()
    is_cav = state.get("is_cav", torch.zeros_like(state["s_m"], dtype=torch.bool))[
        batch].bool().tolist()
    length = state.get("length_m", torch.full_like(state["s_m"], DEFAULT_LENGTH_M))[
        batch].tolist()
    reports = []
    for i in range(len(s)):
        if not (present[i] and is_cav[i]):
            continue
        front = rear = None
        best_f = best_r = sensor_range_m
        for j in range(len(s)):
            if j == i or not present[j] or lane[j] != lane[i]:
                continue
            if s[j] > s[i]:
                g = s[j] - length[j] - s[i]
                if g <= best_f:
                    best_f, front = g, Detection(g, v[j] - v[i], debug_id=j)
            elif s[j] < s[i]:
                g = s[i] - length[i] - s[j]
                if g <= best_r:
                    best_r, rear = g, Detection(g, v[j] - v[i], debug_id=j)
        reports.append(CavReport(cav_id=f"cav{i}", s_m=s[i], lane=lane[i], v_mps=v[i],
                                 a_mps2=a[i], age_s=age_s, length_m=length[i],
                                 front=front, rear=rear, sensor_range_m=sensor_range_m))
    return reports


def rsu_detections_from_state(state: Mapping[str, Tensor], *, batch: int = 0,
                              zones: Optional[RsuZones] = None,
                              age_s: float = 0.0) -> list[RsuDetection]:
    """Synthesise the RSU's own frame from a TRUE WM-frame state: every present vehicle
    (HDV or CAV) inside the RSU zones.  ``debug_id`` = true slot (evaluation only)."""
    zones = zones or RsuZones()
    s = state["s_m"][batch].tolist()
    v = state["v_mps"][batch].tolist()
    a = state.get("a_mps2", torch.zeros_like(state["s_m"]))[batch].tolist()
    lane = state["lane"][batch].long().tolist()
    present = state.get("present", torch.ones_like(state["s_m"], dtype=torch.bool))[
        batch].bool().tolist()
    return [RsuDetection(s[i], lane[i], v[i], a[i], age_s=age_s, debug_id=i)
            for i in range(len(s)) if present[i] and zones.contains(lane[i], s[i])]
