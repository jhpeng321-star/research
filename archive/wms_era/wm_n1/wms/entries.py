"""Observed-counter vehicle ENTRIES for the torch WM-S rollout (default OFF).

Why: the t0-snapshot WM-S rollout animates only the vehicles present at t0.
RUN1_DIAG shows 99% of the merge-section vehicles at 270-300 s are post-t0
entrants, so without entries no model can reproduce merge congestion.  This
module ports the numpy rollout's observed-counter entry model
(``closedloop/rollout_entries.py``) to the torch rollout.

RSU-observable only.  The arrival rate comes from the *entrance counters at
t0* (``counter [2, 30]``, 10 s departure bins, bin 0 most recent): per-entrance
rate = mean count over the most recent ``window_bins`` bins / 10 s.  The process
never reads a recorded future arrival.  Poisson draws per 0.5 s step use a
seeded generator shared by every batch row (common random numbers), so
plan-vs-native comparisons are not diluted by arrival noise.

Insertion.  Mainline arrivals are split over lanes 0/1 by the observed t0
mainline lane share; ramp arrivals go to lane 2.  CAV vs HDV per arrival uses
the observed t0 CAV share (never a configured penetration or hidden driver
type).  An arrival enters at the lane start (``ENTRY_S_M``, the physical lane
start in the s_down frame) only when the bumper gap to the lane's rearmost
vehicle is at least ``min_gap + headway * v_new + closing-braking``; otherwise
it is kept pending (FIFO, <= ``max_pending``) and retried on the next step.
Newcomers occupy free slots from the caller's padding pool and use vType-default
physics (nominal 5 m length, IDM/MOBIL + residual, initial speed = mean of the
three rearmost vehicles on that lane).

Entrants are marked in the cumulative ``entrant`` mask returned by the rollout:
they have no recorded target, so a caller excludes them from the per-car loss;
they remain ordinary present vehicles for traffic-level metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

N_COUNTER_BINS = 30
COUNTER_BIN_S = 10.0
NOMINAL_LENGTH_M = 5.0
#: s_down lane-start (entrance) positions = physical lane start (s=0) minus the
#: frozen lane offset (``features.geometry.HERE_OFFSETS``), lanes 0/1/2.
ENTRY_S_M = (-2645.67, -2644.22, -378.66)

#: Model-frame (``s_down``) section boundaries, shared with
#: ``wms_eval.metrics.SECTIONS`` and
#: ``wms.physics_torch.MODEL_FRAME_MERGE_{START,END}_M``: a slot is *upstream*
#: for ``s < 0``, in the *merge* for ``0 <= s <= 75.44`` and *downstream* above
#: it.  Exposed so a traffic-level loss can bin entrants and t0 vehicles alike.
SECTION_MERGE_START_M = 0.0
SECTION_MERGE_END_M = 75.44
SECTION_UPSTREAM, SECTION_MERGE, SECTION_DOWNSTREAM = 0, 1, 2
SECTION_NAMES = ("upstream", "merge", "downstream")


def section_ids(s_m, present=None):
    """Model-frame section id per position: 0 upstream, 1 merge, 2 downstream.

    ``present`` (optional, broadcastable) sets ``-1`` where a slot is absent so
    an entrant is only section-labelled once it exists.  Accepts numpy arrays or
    torch tensors and returns the same kind (long dtype).  Mirrors
    ``wms_eval.metrics._section`` exactly on the model frame.
    """
    if torch.is_tensor(s_m):
        section = torch.full_like(s_m, SECTION_MERGE, dtype=torch.long)
        section = torch.where(s_m < SECTION_MERGE_START_M,
                              torch.full_like(section, SECTION_UPSTREAM), section)
        section = torch.where(s_m > SECTION_MERGE_END_M,
                              torch.full_like(section, SECTION_DOWNSTREAM), section)
        if present is not None:
            section = torch.where(present, section, torch.full_like(section, -1))
        return section
    s = np.asarray(s_m, np.float64)
    out = np.where(s < SECTION_MERGE_START_M, SECTION_UPSTREAM,
                   np.where(s <= SECTION_MERGE_END_M, SECTION_MERGE,
                            SECTION_DOWNSTREAM))
    if present is not None:
        out = np.where(np.asarray(present, bool), out, -1)
    return out.astype(np.int64)


@dataclass
class EntryConfig:
    window_bins: int = 12          # most recent bins (x10 s) for the rate
    min_gap_m: float = 2.0
    headway_s: float = 1.0
    decel_mps2: float = 3.0
    max_pending: int = 6
    n_nearby: int = 3
    default_speed_mps: float = 20.0
    cav_share: Optional[float] = None   # None -> observed in the t0 snapshot
    rate_scale: float = 1.0


@dataclass
class EntryStats:
    inserted: int = 0
    inserted_cav: int = 0
    deferred_steps: int = 0        # (entrance, row) steps with a blocked arrival
    dropped_pending: int = 0
    no_slot: int = 0
    per_entrance: list = field(default_factory=lambda: [0, 0])


def observed_rates(counter, window_bins: int = 12) -> np.ndarray:
    """Per-entrance arrival rate [2] in veh/s from the t0 entrance counter.

    Only the ``window_bins`` most recent bins are read; every other bin of the
    counter is ignored, so no older (or future) bin can leak into the rate.
    """
    c = np.asarray(counter, np.float64).reshape(2, N_COUNTER_BINS)
    n = max(1, min(int(window_bins), N_COUNTER_BINS))
    return c[:, :n].sum(axis=1) / (n * COUNTER_BIN_S)


def safe_to_insert(gap_m: float, v_new: float, v_lead: float,
                   cfg: EntryConfig) -> bool:
    """SUMO-like safe-gap test: min gap + headway * v + closing brake."""
    brake = max(0.0, v_new - v_lead) ** 2 / (2.0 * cfg.decel_mps2)
    return gap_m >= cfg.min_gap_m + cfg.headway_s * v_new + brake


class TorchEntryProcess:
    """Seeded arrival process for the torch WM-S rollout (one state tensor set).

    ``insert(state)`` writes due arrivals into free slots of ``state`` (a WM-S
    state mapping with ``[B, K]`` ``s_m``/``v_mps``/``a_mps2``/``lane``/
    ``length_m``/``present``/``is_cav`` tensors) and returns the ``[B, K]`` bool
    mask of slots inserted on this step.  The caller must provide enough free
    padding slots; slot allocation prefers never-used slots for CAVs.
    """

    def __init__(self, counter, *, dt_s: float = 0.5, seed: int = 0,
                 batch: int = 1, cfg: Optional[EntryConfig] = None,
                 entry_s_m=ENTRY_S_M, pool=None) -> None:
        self.cfg = cfg or EntryConfig()
        self.rates = observed_rates(counter, self.cfg.window_bins) \
            * float(self.cfg.rate_scale)
        self.dt = float(dt_s)
        self.entry_s = tuple(float(x) for x in entry_s_m)
        self.batch = int(batch)
        self.rng = np.random.default_rng(int(seed))
        self.pending = np.zeros((self.batch, 2), np.int64)
        self.stats = [EntryStats() for _ in range(self.batch)]
        # ``pool`` restricts newcomers to the caller's padding slots.  Without
        # it a vacated t0 slot could be reused, but the recorded ``present_seq``
        # of that t0 row would drop the entrant on the next step.
        self.pool = None if pool is None else np.asarray(pool, bool)
        self.virgin = None             # never-used padding slots
        self._cav_slots = None         # padding slots that ever held a spawned CAV
        self.p_lane1 = None
        self.cav_share = self.cfg.cav_share

    def _pool_row(self, b: int, n_slots: int) -> np.ndarray:
        if self.pool is None:
            return np.ones(int(n_slots), bool)
        if self.pool.ndim == 1:
            return np.broadcast_to(self.pool, (int(n_slots),))
        return self.pool[b]

    # ------------------------------------------------------------ helpers
    def _observe(self, state: dict) -> None:
        """Lane split, CAV share and padding pool from the t0 snapshot (once)."""
        if self.p_lane1 is not None:
            return
        present = state["present"].detach().bool().cpu().numpy()
        lane = state["lane"].detach().cpu().numpy()
        n0 = int((present[0] & (lane[0] == 0)).sum())
        n1 = int((present[0] & (lane[0] == 1)).sum())
        self.p_lane1 = 0.5 if n0 + n1 == 0 else n1 / float(n0 + n1)
        if self.cav_share is None:
            is_cav = state.get("is_cav")
            cav = (is_cav.detach().bool().cpu().numpy() if is_cav is not None
                   else np.zeros_like(present))
            n_all = int(present[0].sum())
            self.cav_share = (float((cav[0] & present[0]).sum()) / n_all
                              if n_all else 0.0)
        self.virgin = ~present & self._pool_row(0, present.shape[1])[None, :]
        self._cav_slots = np.zeros_like(present)

    def _draw(self) -> list:
        """Per-step shared draws: (entrance, lane-noise, cav-noise) per arrival."""
        n = self.rng.poisson(self.rates * self.dt)
        draws = []
        for entrance in range(2):
            for _ in range(int(n[entrance])):
                draws.append((entrance, float(self.rng.random()),
                              float(self.rng.random())))
        return draws

    @staticmethod
    def _lane_rear(s, v, length, present, lane, b, ln, n_nearby):
        """(rear s, its length, rear speed, mean n_nearby speeds) on lane ``ln``."""
        rows = np.flatnonzero(present[b] & (lane[b] == ln))
        if rows.size == 0:
            return None
        order = rows[np.argsort(s[b, rows])]
        near = order[:n_nearby]
        return (float(s[b, order[0]]), float(length[b, order[0]]),
                float(v[b, order[0]]), float(np.mean(v[b, near])))

    def _other_lane_speed(self, s, v, length, present, lane, b, ln):
        if ln not in (0, 1):
            return None
        other = self._lane_rear(s, v, length, present, lane, b, 1 - ln,
                                self.cfg.n_nearby)
        return None if other is None else other[3]

    # ------------------------------------------------------------- step
    def insert(self, state: dict) -> torch.Tensor:
        """Insert due arrivals into ``state`` in place; return the new-slot mask."""
        self._observe(state)
        device = state["present"].device
        new_mask = torch.zeros(state["present"].shape, dtype=torch.bool,
                               device=device)
        s = state["s_m"].detach().cpu().numpy()
        v = state["v_mps"].detach().cpu().numpy()
        length = state["length_m"].detach().cpu().numpy()
        lane = state["lane"].detach().cpu().numpy()
        present = state["present"].detach().cpu().numpy()
        is_cav = state["is_cav"].detach().bool().cpu().numpy()
        b_n, k_n = present.shape
        # local working copies so repeated batch rows stay independent
        s, v, length, lane = s.copy(), v.copy(), length.copy(), lane.copy()
        present, is_cav = present.copy(), is_cav.copy()
        draws = self._draw()
        new_by_e = [[], []]
        for entrance, u_lane, u_cav in draws:
            new_by_e[entrance].append((u_lane, u_cav))
        inserted_slots: list = []
        for b in range(b_n):
            stt = self.stats[b]
            for e in range(2):
                queue = list(new_by_e[e])
                total = int(self.pending[b, e]) + len(queue)
                if total == 0:
                    continue
                # carried-over arrivals try either lane and spawn as HDVs
                items = [(0.0, 2.0)] * int(self.pending[b, e]) + queue
                done = 0
                blocked = False
                for u_lane, u_cav in items:
                    if blocked:
                        break
                    if e == 0:
                        lanes_try = [1, 0] if u_lane < self.p_lane1 else [0, 1]
                    else:
                        lanes_try = [2]
                    placed = False
                    for ln in lanes_try:
                        rear = self._lane_rear(s, v, length, present, lane, b,
                                               ln, self.cfg.n_nearby)
                        s_e = self.entry_s[ln]
                        if rear is None:
                            v_new, gap, v_lead = self.cfg.default_speed_mps, 1e9, 1e9
                            other = self._other_lane_speed(s, v, length, present,
                                                           lane, b, ln)
                            if other is not None:
                                v_new = other
                        else:
                            s_r, l_r, v_lead, v_new = rear
                            gap = s_r - l_r - s_e
                        v_new = max(0.0, float(v_new))
                        if not safe_to_insert(gap, v_new, v_lead, self.cfg):
                            continue
                        free = np.flatnonzero(~present[b]
                                              & self._pool_row(b, present.shape[1]))
                        cav = bool(u_cav < float(self.cav_share or 0.0))
                        if cav:
                            cand = free[self.virgin[b, free]
                                        & ~self._cav_slots[b, free]]
                        else:
                            cand = free[~self._cav_slots[b, free]]
                        if cand.size == 0:
                            stt.no_slot += 1
                            continue
                        k = int(cand[0])
                        s[b, k], v[b, k] = s_e, v_new
                        length[b, k], lane[b, k] = NOMINAL_LENGTH_M, ln
                        present[b, k], is_cav[b, k] = True, bool(cav)
                        if cav:
                            self._cav_slots[b, k] = True
                            stt.inserted_cav += 1
                        stt.inserted += 1
                        stt.per_entrance[e] += 1
                        inserted_slots.append((b, k))
                        placed = True
                        break
                    if placed:
                        done += 1
                    else:
                        blocked = True
                        stt.deferred_steps += 1
                left = total - done
                if left > self.cfg.max_pending:
                    stt.dropped_pending += left - self.cfg.max_pending
                    left = self.cfg.max_pending
                self.pending[b, e] = left
        if inserted_slots:
            with torch.no_grad():
                for b, k in inserted_slots:
                    state["s_m"][b, k] = float(s[b, k])
                    state["v_mps"][b, k] = float(v[b, k])
                    state["a_mps2"][b, k] = 0.0
                    state["lane"][b, k] = int(lane[b, k])
                    state["length_m"][b, k] = NOMINAL_LENGTH_M
                    state["present"][b, k] = True
                    state["is_cav"][b, k] = bool(is_cav[b, k])
                    new_mask[b, k] = True
        return new_mask

    def summary(self, b: int = 0) -> dict:
        s = self.stats[b]
        return {"rates_veh_s": [float(x) for x in self.rates],
                "p_lane1": self.p_lane1, "cav_share_observed": self.cav_share,
                "inserted": s.inserted, "inserted_cav": s.inserted_cav,
                "per_entrance": list(s.per_entrance),
                "deferred_steps": s.deferred_steps,
                "dropped_pending": s.dropped_pending, "no_slot": s.no_slot}


# ===========================================================================
# Entries v2: observed-history arrival process and HOW vehicles enter.
#
# v1 reads only the mean of the recent t0 counter bins and forces every arrival
# through a single pooled Poisson intensity, a snapshot lane split, a snapshot
# CAV share and a snapshot-mean speed.  v2 additionally estimates, from the
# *observed* pre-t0 entry history (departure/insertion times, entry speeds,
# classes and vehicle types; RSU-observable, t <= t0 only):
#
#  * arrival COUNT -> a recency-weighted (EWMA) rate with a bounded trend
#    extrapolation (weighted linear fit of the 10 s counter bins),
#  * arrival PROCESS -> a gamma renewal fit to the observed inter-arrival
#    times (mean + coefficient of variation), so platoons/burstiness survive
#    instead of an exponential/Poisson process,
#  * HOW they enter -> empirical entry-speed ratio (entry speed / local traffic
#    speed), entry gap to the vehicle ahead, entry lane and vehicle class mix
#    (CAV vs HDV, with the HDV vType length when it is observable),
#
# all shrunk towards network-wide defaults when the observed history is short.
# Sampling is seeded, so with a shared seed and history every compared
# controller receives the identical entrant stream (common random numbers).
# ===========================================================================

N_ENTRANCES = 2
VTYPES = ("normal_car", "cautious_car", "aggressive_car", "truck")
VTYPE_LENGTH_M = (5.0, 5.0, 5.0, 7.5)
N_VTYPES = len(VTYPES)
DEFAULT_RATE_VEH_S = 0.1
DEFAULT_HEADWAY_CV = 1.0
DEFAULT_SPEED_RATIO = 1.0
DEFAULT_SPEED_RATIO_STD = 0.15
DEFAULT_GAP_M = 10.0
DEFAULT_GAP_STD_M = 5.0
DEFAULT_CAV_SHARE = 0.0
DEFAULT_LANE_P1 = 0.5
_ENTRANCE_NAMES = {"mainline": 0, "ramp": 1}
_FEATURE_KEYS = ("rate", "trend", "headway_mean", "headway_cv", "speed_mean",
                 "speed_std", "gap_mean", "gap_std", "cav_share", "n_obs")
FEATURE_LEN = N_ENTRANCES * len(_FEATURE_KEYS) + N_ENTRANCES * N_VTYPES + 1


@dataclass
class EntranceHistory:
    """Per-entrance summary of the observed pre-t0 entry history.

    Arrays are length ``2`` (mainline, ramp) unless noted.  ``rate`` is veh/s,
    ``headway_mean`` seconds and ``headway_cv`` the observed coefficient of
    variation (>= 0).  ``speed_mean``/``speed_std`` describe the entry-speed to
    local-traffic-speed ratio; ``gap_mean``/``gap_std`` the entry gap (m).
    ``vtype_share`` is ``[2, N_VTYPES]``.  ``lane_p1`` is the observed mainline
    lane-1 share.  ``n_obs`` is the observed entry count per entrance.
    """

    rate: np.ndarray
    trend: np.ndarray
    headway_mean: np.ndarray
    headway_cv: np.ndarray
    speed_mean: np.ndarray
    speed_std: np.ndarray
    gap_mean: np.ndarray
    gap_std: np.ndarray
    cav_share: np.ndarray
    n_obs: np.ndarray
    vtype_share: np.ndarray
    lane_p1: float = DEFAULT_LANE_P1

    @classmethod
    def defaults(cls) -> "EntranceHistory":
        vtype = np.zeros((N_ENTRANCES, N_VTYPES), np.float64)
        vtype[:, 0] = 1.0
        return cls(
            rate=np.full(N_ENTRANCES, DEFAULT_RATE_VEH_S),
            trend=np.ones(N_ENTRANCES),
            headway_mean=1.0 / DEFAULT_RATE_VEH_S * np.ones(N_ENTRANCES),
            headway_cv=np.full(N_ENTRANCES, DEFAULT_HEADWAY_CV),
            speed_mean=np.full(N_ENTRANCES, DEFAULT_SPEED_RATIO),
            speed_std=np.full(N_ENTRANCES, DEFAULT_SPEED_RATIO_STD),
            gap_mean=np.full(N_ENTRANCES, DEFAULT_GAP_M),
            gap_std=np.full(N_ENTRANCES, DEFAULT_GAP_STD_M),
            cav_share=np.full(N_ENTRANCES, DEFAULT_CAV_SHARE),
            n_obs=np.zeros(N_ENTRANCES),
            vtype_share=vtype,
            lane_p1=DEFAULT_LANE_P1,
        )

    def to_features(self) -> np.ndarray:
        per = np.stack([getattr(self, key) for key in _FEATURE_KEYS], axis=0)
        return np.concatenate([per.reshape(-1), self.vtype_share.reshape(-1),
                               np.asarray([self.lane_p1], np.float64)]).astype(np.float64)

    @classmethod
    def from_features(cls, vec) -> "EntranceHistory":
        v = np.asarray(vec, np.float64).reshape(-1)
        if v.size != FEATURE_LEN:
            raise ValueError(f"entry feature vector has {v.size} values, expected {FEATURE_LEN}")
        n_keys = len(_FEATURE_KEYS)
        per = v[:N_ENTRANCES * n_keys].reshape(n_keys, N_ENTRANCES)
        vt = v[N_ENTRANCES * n_keys:-1].reshape(N_ENTRANCES, N_VTYPES)
        return cls(**{key: per[i].copy() for i, key in enumerate(_FEATURE_KEYS)},
                   vtype_share=vt.copy(), lane_p1=float(v[-1]))


def _recency_trend(counter, window_bins: int, halflife_bins: float):
    """Recency-weighted rate (veh/s) and bounded trend from 10 s counter bins.

    Bin 0 is the most recent.  A decayed linear fit extrapolates one bin ahead
    (``j = -1``); the rate is the mean of the recency-weighted mean and that
    one-step extrapolation, so an increasing arrival count raises the rate.
    """
    c = np.asarray(counter, np.float64).reshape(N_ENTRANCES, N_COUNTER_BINS)
    n = max(2, min(int(window_bins), N_COUNTER_BINS))
    cw = c[:, :n]
    bins = np.arange(n, dtype=np.float64)
    w = 0.5 ** (bins / max(float(halflife_bins), 1.0e-6))
    wsum = max(w.sum(), 1.0e-9)
    mean_per_bin = (cw * w).sum(1) / wsum
    xbar = (bins * w).sum() / wsum
    denom = max(((bins - xbar) ** 2 * w).sum(), 1.0e-9)
    slope = (((bins - xbar) * w)[None, :] * (cw - mean_per_bin[:, None])).sum(1) / denom
    next_bin = np.clip(mean_per_bin - slope, 0.0, None)
    rate = (mean_per_bin + next_bin) / (2.0 * COUNTER_BIN_S)
    trend = np.clip(next_bin / np.maximum(mean_per_bin, 1.0e-9), 0.25, 4.0)
    return rate, trend, mean_per_bin / COUNTER_BIN_S


def _counts_counter(times, entrance, t0_s, n_bins: int = N_COUNTER_BINS):
    """Build a [2,30] 10 s departure counter from observed entry times <= t0."""
    counter = np.zeros((N_ENTRANCES, N_COUNTER_BINS), np.float64)
    ts = np.asarray(times, np.float64)
    en = np.asarray(entrance, np.int64)
    if ts.size == 0:
        return counter
    age = float(t0_s) - ts
    keep = (age >= 0.0) & (age < N_COUNTER_BINS * COUNTER_BIN_S)
    b = np.floor(age[keep] / COUNTER_BIN_S).astype(np.int64)
    e = en[keep]
    np.add.at(counter, (e, b), 1.0)
    return counter


def build_entrance_history(*, counter=None, times=None, entrance=None, speed=None,
                           local_speed=None, lane=None, gap=None, gap_entrance=None,
                           is_cav=None, vtype=None, mainline_lane_share=None, t0_s=None,
                           window_bins: int = 12, halflife_bins: float = 4.0,
                           shrink_k: float = 8.0,
                           defaults: Optional[EntranceHistory] = None) -> EntranceHistory:
    """Estimate an :class:`EntranceHistory` from t0-observable entry evidence.

    ``counter`` is the [2,30] t0 counter.  ``times``/``entrance`` are observed
    entry times/entrances (t <= t0); ``speed``/``local_speed`` the entry and
    local traffic speeds; ``lane`` the entry lane; ``gap`` the entry gap; and
    ``is_cav``/``vtype`` the observed class mix.  Every empirical quantity is
    shrunk to ``defaults`` by ``n_obs / (n_obs + shrink_k)``.
    """
    base = defaults if defaults is not None else EntranceHistory.defaults()
    en = None if entrance is None else np.asarray(entrance, np.int64)
    ge = en if gap_entrance is None else np.asarray(gap_entrance, np.int64)
    ts = None if times is None else np.asarray(times, np.float64)
    n_obs = np.zeros(N_ENTRANCES, np.float64)
    if en is not None:
        for e in range(N_ENTRANCES):
            n_obs[e] = float((en == e).sum())
    if counter is not None:
        emp_rate, trend, _ = _recency_trend(counter, window_bins, halflife_bins)
        w_count = np.ones(N_ENTRANCES)
    elif ts is not None and ts.size:
        t_ref = float(t0_s) if t0_s is not None else float(ts.max())
        cnt = _counts_counter(ts, en, t_ref)
        emp_rate, trend, _ = _recency_trend(cnt, window_bins, halflife_bins)
        w_count = n_obs / (n_obs + max(float(shrink_k), 1.0e-9))
    else:
        emp_rate = base.rate.copy()
        trend = base.trend.copy()
        w_count = np.zeros(N_ENTRANCES)

    hw_mean = base.headway_mean.copy()
    hw_cv = base.headway_cv.copy()
    sp_mean = base.speed_mean.copy()
    sp_std = base.speed_std.copy()
    gp_mean = base.gap_mean.copy()
    gp_std = base.gap_std.copy()
    cav_share = base.cav_share.copy()
    vtype_share = base.vtype_share.copy()
    for e in range(N_ENTRANCES):
        sel = None if en is None else (en == e)
        if sel is not None and ts is not None and int(sel.sum()) >= 2:
            t_e = np.sort(ts[sel])
            h = np.diff(t_e)
            h = h[h > 0.0]
            if h.size:
                mu = float(h.mean())
                hw_mean[e] = mu
                hw_cv[e] = float(h.std() / mu) if mu > 0 else DEFAULT_HEADWAY_CV
        if sel is not None and speed is not None and local_speed is not None:
            sp = np.asarray(speed, np.float64)[sel]
            lv = np.asarray(local_speed, np.float64)[sel]
            ok = lv > 1.0e-3
            if ok.any():
                ratio = sp[ok] / lv[ok]
                if ratio.size:
                    sp_mean[e] = float(ratio.mean())
                    sp_std[e] = float(ratio.std()) if ratio.size > 1 else 0.0
        if sel is not None and gap is not None:
            gsel = (ge == e) if ge is not None else sel
            g = np.asarray(gap, np.float64)[gsel]
            g = g[np.isfinite(g)]
            if g.size:
                gp_mean[e] = float(g.mean())
                gp_std[e] = float(g.std()) if g.size > 1 else 0.0
        if sel is not None and is_cav is not None and int(sel.sum()):
            cav_share[e] = float(np.asarray(is_cav, bool)[sel].mean())
        if sel is not None and vtype is not None and int(sel.sum()):
            counts = np.zeros(N_VTYPES, np.float64)
            for t in np.asarray(vtype)[sel]:
                counts[_vtype_index(t)] += 1.0
            vtype_share[e] = counts / max(counts.sum(), 1.0)

    w = n_obs / (n_obs + max(float(shrink_k), 1.0e-9))
    rate = w_count * emp_rate + (1.0 - w_count) * base.rate
    trend = w_count * np.clip(trend, 0.25, 4.0) + (1.0 - w_count) * base.trend
    hw_cv = w * hw_cv + (1.0 - w) * DEFAULT_HEADWAY_CV
    sp_mean = w * sp_mean + (1.0 - w) * base.speed_mean
    sp_std = w * sp_std + (1.0 - w) * base.speed_std
    gp_mean = w * gp_mean + (1.0 - w) * base.gap_mean
    gp_std = w * gp_std + (1.0 - w) * base.gap_std
    cav_share = w * cav_share + (1.0 - w) * base.cav_share
    vtype_share = w[:, None] * vtype_share + (1.0 - w)[:, None] * base.vtype_share
    if lane is not None and en is not None:
        ml = np.asarray(lane, np.int64)[en == 0]
        n_main = float(ml.size)
        wl = n_main / (n_main + max(float(shrink_k), 1.0e-9))
        emp_p1 = float((ml == 1).mean()) if n_main else DEFAULT_LANE_P1
    elif mainline_lane_share is not None:
        n_main = float(n_obs[0])
        wl = n_main / (n_main + max(float(shrink_k), 1.0e-9))
        emp_p1 = float(mainline_lane_share)
    else:
        wl, emp_p1 = 0.0, DEFAULT_LANE_P1
    lane_p1 = wl * emp_p1 + (1.0 - wl) * base.lane_p1
    return EntranceHistory(
        rate=np.clip(rate, 0.0, None), trend=trend, headway_mean=hw_mean,
        headway_cv=np.clip(hw_cv, 0.05, 8.0), speed_mean=np.clip(sp_mean, 0.0, None),
        speed_std=np.clip(sp_std, 0.0, None), gap_mean=np.clip(gp_mean, 0.0, None),
        gap_std=np.clip(gp_std, 0.0, None), cav_share=np.clip(cav_share, 0.0, 1.0),
        n_obs=n_obs, vtype_share=vtype_share, lane_p1=float(np.clip(lane_p1, 0.0, 1.0)))


def _vtype_index(name) -> int:
    text = str(name or "").strip().lower()
    if text in VTYPES:
        return VTYPES.index(text)
    if "truck" in text or "bus" in text:
        return VTYPES.index("truck")
    if "cautious" in text:
        return VTYPES.index("cautious_car")
    if "aggress" in text:
        return VTYPES.index("aggressive_car")
    return VTYPES.index("normal_car")


def read_insertion_history(base_run_dir, t0_s: float) -> dict:
    """Observed pre-t0 insertion records of a base run (READ-ONLY).

    Reads ``run/insertions.jsonl`` and keeps only rows with ``inserted_s`` at
    or before ``t0_s`` (the same source the t0 entrance counter is built
    from), never a scheduled future departure.  Returns arrays ``times``,
    ``entrance`` (0 mainline / 1 ramp), ``speed`` (m/s), ``is_cav`` and
    ``vtype`` (string array).
    """
    import json
    from pathlib import Path

    path = Path(base_run_dir) / "run" / "insertions.jsonl"
    times: list = []
    entrance: list = []
    speed: list = []
    is_cav: list = []
    vtype: list = []
    if not path.is_file():
        return {"times": np.zeros(0), "entrance": np.zeros(0, np.int64),
                "speed": np.zeros(0), "is_cav": np.zeros(0, bool),
                "vtype": np.asarray([], dtype=object)}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = float(row.get("inserted_s", row.get("scheduled_depart_s", 0.0)))
            if t > float(t0_s) + 1.0e-9:
                continue
            ent = _ENTRANCE_NAMES.get(str(row.get("entrance", "mainline")), 0)
            times.append(t)
            entrance.append(ent)
            speed.append(float(row.get("depart_speed_mps", 0.0)))
            is_cav.append(bool(row.get("is_cav", False)))
            vtype.append(str(row.get("type_id", "normal_car")))
    order = np.argsort(np.asarray(times, np.float64))
    return {"times": np.asarray(times, np.float64)[order],
            "entrance": np.asarray(entrance, np.int64)[order],
            "speed": np.asarray(speed, np.float64)[order],
            "is_cav": np.asarray(is_cav, bool)[order],
            "vtype": np.asarray(vtype, dtype=object)[order]}


class TorchEntryProcessV2:
    """Seeded observed-history entry process for the torch WM-S rollout.

    Arrivals are a per-entrance gamma renewal fit to the observed inter-arrival
    times (shape ``1/cv^2``), with a recency-weighted rate and a bounded trend.
    Each arrival samples its lane, entry-speed ratio (relative to the current
    local traffic at the entrance), required entry gap and class from the
    observed distributions (shrunk to defaults).  The insertion/slot mechanics
    match :class:`TorchEntryProcess`, so a rollout can swap the two.
    """

    #: Test-only switch: when True the per-step host-materialisation fast-path is
    #: disabled so the pre-vectorisation behaviour can be compared for parity.
    _FORCE_MATERIALISE = False

    def __init__(self, counter=None, *, history: Optional[EntranceHistory] = None,
                 histories: Optional[list] = None, dt_s: float = 0.5, seed: int = 0,
                 batch: int = 1, cfg: Optional[EntryConfig] = None,
                 entry_s_m=ENTRY_S_M, pool=None,
                 defaults: Optional[EntranceHistory] = None) -> None:
        self.cfg = cfg or EntryConfig()
        self.dt = float(dt_s)
        self.entry_s = tuple(float(x) for x in entry_s_m)
        if histories is None:
            if history is None:
                history = build_entrance_history(counter=counter, defaults=defaults)
            histories = [history]
        self.histories = list(histories)
        self.batch = max(int(batch), len(self.histories))
        self.rng = np.random.default_rng(int(seed))
        self.pending = np.zeros((self.batch, N_ENTRANCES), np.int64)
        self.stats = [EntryStats() for _ in range(self.batch)]
        self.pool = None if pool is None else np.asarray(pool, bool)
        self.virgin = None
        self._cav_slots = None
        self.now = np.zeros(self.batch)
        self.t_next = np.full((self.batch, N_ENTRANCES), np.inf)
        self._started = np.zeros(self.batch, bool)

    def _hist(self, b: int) -> EntranceHistory:
        return self.histories[b] if b < len(self.histories) else self.histories[-1]

    def _pool_row(self, b: int, n_slots: int) -> np.ndarray:
        if self.pool is None:
            return np.ones(int(n_slots), bool)
        if self.pool.ndim == 1:
            return np.broadcast_to(self.pool, (int(n_slots),))
        return self.pool[b]

    def _headway(self, b: int, e: int) -> float:
        h = self._hist(b)
        rate = max(float(h.rate[e]), 1.0e-9)
        cv = float(np.clip(h.headway_cv[e], 0.2, 5.0))
        k = 1.0 / (cv * cv)
        scale = (1.0 / rate) / k
        draw = float(self.rng.gamma(k, scale))
        return max(draw, 0.05)

    def _observe(self, state: dict) -> None:
        if self.virgin is not None:
            return
        present = state["present"].detach().bool().cpu().numpy()
        b_n = present.shape[0]
        if b_n > self.batch:
            self.batch = b_n
            self.pending = np.zeros((b_n, N_ENTRANCES), np.int64)
            self.stats = [EntryStats() for _ in range(b_n)]
            self.now = np.zeros(b_n)
            self.t_next = np.full((b_n, N_ENTRANCES), np.inf)
            self._started = np.zeros(b_n, bool)
        self.virgin = ~present.copy()
        self._cav_slots = np.zeros_like(present)

    def _emit_due(self, b: int) -> list:
        self.now[b] += self.dt
        if not self._started[b]:
            for e in range(N_ENTRANCES):
                self.t_next[b, e] = self.now[b] + self._headway(b, e)
            self._started[b] = True
        out = []
        for e in range(N_ENTRANCES):
            guard = 0
            while self.t_next[b, e] <= self.now[b] and guard < 1000:
                out.append(e)
                self.t_next[b, e] += self._headway(b, e)
                guard += 1
        return out

    @staticmethod
    def _lane_rear(s, v, length, present, lane, b, ln, n_nearby):
        rows = np.flatnonzero(present[b] & (lane[b] == ln))
        if rows.size == 0:
            return None
        order = rows[np.argsort(s[b, rows])]
        near = order[:n_nearby]
        return (float(s[b, order[0]]), float(length[b, order[0]]),
                float(v[b, order[0]]), float(np.mean(v[b, near])))

    def _local_speed(self, s, v, length, present, lane, b, ln):
        rear = self._lane_rear(s, v, length, present, lane, b, ln, self.cfg.n_nearby)
        if rear is not None:
            return rear[3]
        if ln in (0, 1):
            other = self._lane_rear(s, v, length, present, lane, b, 1 - ln,
                                    self.cfg.n_nearby)
            if other is not None:
                return other[3]
        return None

    def insert(self, state: dict) -> torch.Tensor:
        self._observe(state)
        device = state["present"].device
        new_mask = torch.zeros(state["present"].shape, dtype=torch.bool, device=device)
        # Advance the renewal clock first; ``_emit_due`` touches no state tensor.
        # Only materialise the (six) host copies when an arrival is actually due
        # or a previously deferred insertion is still pending.  This removes the
        # per-step device->host copies and their syncs on the common no-arrival
        # step while leaving the insertion numerics byte-identical.
        b_n = int(state["present"].shape[0])
        due = [self._emit_due(b) for b in range(b_n)]
        if (not self._FORCE_MATERIALISE and not any(due)
                and not bool(np.any(self.pending))):
            return new_mask
        s = state["s_m"].detach().cpu().numpy()
        v = state["v_mps"].detach().cpu().numpy()
        length = state["length_m"].detach().cpu().numpy()
        lane = state["lane"].detach().cpu().numpy()
        present = state["present"].detach().cpu().numpy()
        is_cav = state["is_cav"].detach().bool().cpu().numpy()
        s, v = s.copy(), v.copy()
        length, lane = length.copy(), lane.copy()
        present, is_cav = present.copy(), is_cav.copy()
        inserted_slots: list = []
        for b in range(b_n):
            hist = self._hist(b)
            stt = self.stats[b]
            for e in range(N_ENTRANCES):
                total = int(self.pending[b, e]) + due[b].count(e)
                if total == 0:
                    continue
                done = 0
                blocked = False
                for _ in range(total):
                    if blocked:
                        break
                    u_cav = float(self.rng.random())
                    u_lane = float(self.rng.random())
                    u_sr = float(self.rng.normal(hist.speed_mean[e],
                                                 max(hist.speed_std[e], 1.0e-3)))
                    u_gap = float(self.rng.normal(hist.gap_mean[e],
                                                  max(hist.gap_std[e], 1.0e-3)))
                    vt = (int(self.rng.choice(N_VTYPES, p=_norm(hist.vtype_share[e])))
                          if hist.vtype_share[e].sum() > 0 else 0)
                    if e == 0:
                        lanes_try = [1, 0] if u_lane < hist.lane_p1 else [0, 1]
                    else:
                        lanes_try = [2]
                    placed = False
                    for ln in lanes_try:
                        rear = self._lane_rear(s, v, length, present, lane, b, ln,
                                               self.cfg.n_nearby)
                        s_e = self.entry_s[ln]
                        local = self._local_speed(s, v, length, present, lane, b, ln)
                        local = self.cfg.default_speed_mps if local is None else local
                        v_new = max(0.0, float(u_sr) * float(local))
                        if rear is None:
                            gap, v_lead = 1.0e9, 1.0e9
                        else:
                            s_r, l_r, v_lead, _ = rear
                            gap = s_r - l_r - s_e
                        if not safe_to_insert(gap, v_new, v_lead, self.cfg):
                            continue
                        if gap < max(0.0, u_gap):
                            continue
                        free = np.flatnonzero(~present[b]
                                              & self._pool_row(b, present.shape[1]))
                        cav = bool(u_cav < float(hist.cav_share[e]))
                        if cav:
                            cand = free[self.virgin[b, free]
                                        & ~self._cav_slots[b, free]]
                        else:
                            cand = free[~self._cav_slots[b, free]]
                        if cand.size == 0:
                            stt.no_slot += 1
                            continue
                        k = int(cand[0])
                        s[b, k], v[b, k] = s_e, v_new
                        length[b, k] = float(VTYPE_LENGTH_M[vt])
                        lane[b, k] = ln
                        present[b, k], is_cav[b, k] = True, bool(cav)
                        if cav:
                            self._cav_slots[b, k] = True
                            stt.inserted_cav += 1
                        stt.inserted += 1
                        stt.per_entrance[e] += 1
                        inserted_slots.append((b, k))
                        placed = True
                        break
                    if placed:
                        done += 1
                    else:
                        blocked = True
                        stt.deferred_steps += 1
                left = total - done
                if left > self.cfg.max_pending:
                    stt.dropped_pending += left - self.cfg.max_pending
                    left = self.cfg.max_pending
                self.pending[b, e] = left
        if inserted_slots:
            with torch.no_grad():
                for b, k in inserted_slots:
                    state["s_m"][b, k] = float(s[b, k])
                    state["v_mps"][b, k] = float(v[b, k])
                    state["a_mps2"][b, k] = 0.0
                    state["lane"][b, k] = int(lane[b, k])
                    state["length_m"][b, k] = float(length[b, k])
                    state["present"][b, k] = True
                    state["is_cav"][b, k] = bool(is_cav[b, k])
                    new_mask[b, k] = True
        return new_mask

    def summary(self, b: int = 0) -> dict:
        hist = self._hist(b)
        stt = self.stats[b]
        return {"version": 2,
                "rate_veh_s": [float(x) for x in hist.rate],
                "trend": [float(x) for x in hist.trend],
                "headway_mean_s": [float(x) for x in hist.headway_mean],
                "headway_cv": [float(x) for x in hist.headway_cv],
                "speed_ratio_mean": [float(x) for x in hist.speed_mean],
                "gap_mean_m": [float(x) for x in hist.gap_mean],
                "cav_share_observed": [float(x) for x in hist.cav_share],
                "lane_p1": float(hist.lane_p1),
                "n_obs": [int(x) for x in hist.n_obs],
                "inserted": stt.inserted, "inserted_cav": stt.inserted_cav,
                "per_entrance": list(stt.per_entrance),
                "deferred_steps": stt.deferred_steps,
                "dropped_pending": stt.dropped_pending, "no_slot": stt.no_slot}


def _norm(probs) -> np.ndarray:
    p = np.asarray(probs, np.float64)
    total = p.sum()
    if total <= 0:
        out = np.zeros_like(p)
        out[0] = 1.0
        return out
    return p / total
