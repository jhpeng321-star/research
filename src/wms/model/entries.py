"""Observed-counter vehicle entries for the torch WM-S rollout (default off).

The t0-snapshot rollout animates only the vehicles present at t0, yet most
merge-section vehicles at long horizons are post-t0 entrants.  This module ports the
observed-counter entry model of the numpy rollout to the torch rollout.

RSU-observable only.  Arrival rates come from the *entrance counters at t0*
(``counter [2, 30]``, 10 s departure bins, bin 0 most recent), never from a recorded
future arrival.  Arrival draws use a seeded generator shared by every batch row
(common random numbers), so plan-vs-native comparisons are not diluted by arrival
noise.

Mainline arrivals are split over lanes 0/1 by the observed t0 mainline lane share and
ramp arrivals go to lane 2.  An arrival enters at the lane start only when the bumper
gap to the lane's rearmost vehicle is at least ``min_gap + headway * v_new +
closing-braking``; otherwise it stays pending (FIFO, at most ``max_pending``) and is
retried on the next step.  Newcomers occupy free slots from the caller's padding pool.

Two processes share the placement mechanics:

* :class:`TorchEntryProcess` - Poisson arrivals at the recent-bin mean rate;
* :class:`TorchEntryProcessV2` - per-entrance gamma renewal fit to the observed
  pre-t0 entry history (:class:`EntranceHistory`), which also samples entry speed,
  entry gap, lane and vehicle class from the observed distributions.

Entrants are flagged in the rollout's cumulative ``entrant`` mask: they have no
recorded target, so callers exclude them from per-car losses while traffic metrics
still count them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

N_COUNTER_BINS = 30
COUNTER_BIN_S = 10.0
NOMINAL_LENGTH_M = 5.0
#: ``s_down`` lane-start (entrance) positions of lanes 0/1/2.
ENTRY_S_M = (-2645.67, -2644.22, -378.66)

#: Model-frame section boundaries: *upstream* for ``s < 0``, *merge* for
#: ``0 <= s <= 75.44`` and *downstream* above it.
SECTION_MERGE_START_M = 0.0
SECTION_MERGE_END_M = 75.44
SECTION_UPSTREAM, SECTION_MERGE, SECTION_DOWNSTREAM = 0, 1, 2
SECTION_NAMES = ("upstream", "merge", "downstream")

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
#: Insertion rules of :class:`TorchEntryProcessV2` (``EntryConfig.insert_rule``).
INSERT_RULES = ("v2", "sumo_like", "sumo_like_v2")
#: SUMO lane speed limits (m/s) of the entrance edges per model lane (network bytes,
#: ``1191554636_{0,1}`` and ``11553048#1_0``); the ``sumo_like`` departSpeed="max" cap.
SUMO_LANE_SPEED_MPS = (31.29, 31.29, 22.22)
#: SUMO vType parameters per ``VTYPES`` entry (scenarios.py route template; truck = SUMO
#: defaults): speedFactor, minGap (m), tau (s), decel (m/s^2).
VTYPE_SPEED_FACTOR = (1.0, 0.92, 1.08, 1.0)
VTYPE_MIN_GAP_M = (2.5, 3.5, 1.5, 2.5)
VTYPE_TAU_S = (1.1, 1.5, 0.95, 1.0)
VTYPE_DECEL_MPS2 = (4.5, 3.0, 4.5, 4.0)
#: SUMO ``departPos="base"`` puts the FRONT at ``lane start + length + POSITION_EPS``.
SUMO_POSITION_EPS_M = 0.1
#: sumo_like_v2 (V6ENTRY): vType (tau s, decel m/s^2, minGap m) of the v5 "textbook"
#: population medians (``traffic_shepherd/c4/population.py``; CAVs ``cav_typical``:
#: tau 1.2, decel 2.0, minGap 2.0) used for the admission test and the stop-speed
#: departure.  On the saturated v5 recordings SUMO's admission boundary is the
#: end-of-step bumper gap against these minGaps (measured true accept 0.83 / false
#: accept 0.004; the legacy :data:`VTYPE_MIN_GAP_M` 2.5 admits only 0.58), ENTRY_DIAG.md.
SUMO_V2_TAU_S = (1.2, 1.2, 1.2, 1.0)
SUMO_V2_DECEL_MPS2 = (2.0, 2.0, 2.0, 4.0)
SUMO_V2_MIN_GAP_M = (2.0, 2.0, 2.0, 2.5)
_ENTRANCE_NAMES = {"mainline": 0, "ramp": 1}
_FEATURE_KEYS = ("rate", "trend", "headway_mean", "headway_cv", "speed_mean",
                 "speed_std", "gap_mean", "gap_std", "cav_share", "n_obs")
FEATURE_LEN = N_ENTRANCES * len(_FEATURE_KEYS) + N_ENTRANCES * N_VTYPES + 1
#: Placement outcome for an arrival that found no safe gap or no free slot.
_NO_GAP = 1.0e9


def section_ids(s_m, present=None):
    """Model-frame section id per position: 0 upstream, 1 merge, 2 downstream.

    ``present`` (optional, broadcastable) sets ``-1`` where a slot is absent.
    Accepts numpy arrays or torch tensors and returns the same kind (int64).
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
    # Insertion rule of TorchEntryProcessV2 (ENTRYFIX_NOTES.md).  "v2" (default) = the
    # original rule, bit-identical.  "sumo_like" = SUMO's departSpeed="max" /
    # departLane="best" / departPos="base" insertion: the entrant's speed is lowered to
    # the safe insertion speed behind the lane's rearmost vehicle (no sampled required
    # gap, no fixed speed ratio), any entrance lane that admits the vehicle may be used, the
    # front is placed at lane start + length, and FIFO arrivals keep inserting in one step
    # while a lane admits them (an entrance stops at its first refusal, as in SUMO).
    #
    # "sumo_like_v2" (V6ENTRY, default off) additionally reproduces SUMO 1.27's *saturated*
    # entrance behaviour measured on the v5 textbook recordings (ENTRY_DIAG.md): the
    # departure test runs at the END of the insertion step (the lane's rearmost vehicle
    # moves ``v_lead * dt`` while the entrant, departing near the stop-speed, does not),
    # and the entrant's speed is the STOP speed behind that end-of-step rear position
    # (the leader is treated as stationary; SUMO inserts queued mainline entrants at a
    # median 0.16 m/s although the rearmost vehicle already moves at ~3.4 m/s).  Admission
    # stays "end-of-step bumper gap >= vType minGap"; "sumo_like" instead tests the
    # pre-step gap (refusing most of a saturated queue's actual insertions) and credits
    # the leader's speed in the safe-speed formula (entering 1.7-2.5 m/s instead of
    # ~0.15), so its entrance service runs ~30 % faster than SUMO's and the imagination
    # never grows an entry queue.
    insert_rule: str = "v2"
    #: sumo_like lane choice: "any" (default; try every entrance lane, largest gap first --
    #: admits 99% of SUMO's recorded insertions, ENTRYFIX_NOTES.md) or "best_free" (only the
    #: least-occupied entrance lane by vehicle-length sum; admits 65-96%).
    sumo_lane_mode: str = "any"
    #: sumo_like: refuse when the safe insertion speed is below this (SUMO "max": 0).
    sumo_min_speed_mps: float = 0.0
    sumo_lane_speed_mps: tuple = SUMO_LANE_SPEED_MPS

    def __post_init__(self) -> None:
        if self.insert_rule not in INSERT_RULES:
            raise ValueError(f"insert_rule must be one of {INSERT_RULES}, got {self.insert_rule!r}")
        if self.sumo_lane_mode not in ("best_free", "any"):
            raise ValueError("sumo_lane_mode must be 'best_free' or 'any'")


@dataclass
class EntryStats:
    inserted: int = 0
    inserted_cav: int = 0
    deferred_steps: int = 0        # (entrance, row) steps with a blocked arrival
    dropped_pending: int = 0
    no_slot: int = 0
    per_entrance: list = field(default_factory=lambda: [0, 0])


def observed_rates(counter, window_bins: int = 12) -> np.ndarray:
    """Per-entrance arrival rate ``[2]`` (veh/s) from the t0 entrance counter.

    Only the ``window_bins`` most recent bins are read, so no older or future bin
    can leak into the rate.
    """
    c = np.asarray(counter, np.float64).reshape(2, N_COUNTER_BINS)
    n = max(1, min(int(window_bins), N_COUNTER_BINS))
    return c[:, :n].sum(axis=1) / (n * COUNTER_BIN_S)


def safe_to_insert(gap_m: float, v_new: float, v_lead: float, cfg: EntryConfig) -> bool:
    """SUMO-like safe-gap test: min gap + headway * v + closing brake."""
    brake = max(0.0, v_new - v_lead) ** 2 / (2.0 * cfg.decel_mps2)
    return gap_m >= cfg.min_gap_m + cfg.headway_s * v_new + brake


def sumo_safe_insert_speed(gap_m: float, v_lead: float, tau_s: float, decel_mps2: float,
                           min_gap_m: float) -> float:
    """SUMO insertion safe speed (``maximumSafeFollowSpeed``, Euler, equal decelerations).

    The largest ``v`` with ``tau * v + v^2 / (2 b) <= g + v_lead^2 / (2 b)``,
    ``g = gap - minGap``; ``-1`` when ``g < 0`` (SUMO refuses an insertion inside minGap).
    """
    g = float(gap_m) - float(min_gap_m)
    if g < 0.0:
        return -1.0
    b = max(float(decel_mps2), 1.0e-6)
    tb = float(tau_s) * b
    vl = max(0.0, float(v_lead))
    return float(-tb + np.sqrt(tb * tb + vl * vl + 2.0 * b * g))


def sumo_insert_ok(gap_m: float, v_new: float, v_lead: float, tau_s: float,
                   decel_mps2: float, min_gap_m: float) -> bool:
    """SUMO insertion safety at ``v_new``: ``gap >= minGap + max(0, tau v + (v^2 - v_lead^2)/(2b))``."""
    b = max(float(decel_mps2), 1.0e-6)
    vl = max(0.0, float(v_lead))
    need = float(tau_s) * v_new + (v_new * v_new - vl * vl) / (2.0 * b)
    return float(gap_m) >= float(min_gap_m) + max(0.0, need) - 1.0e-6


class _Working:
    """Host-side working copies of a state, edited while arrivals are placed."""

    def __init__(self, state: dict) -> None:
        def host(name: str) -> np.ndarray:
            return state[name].detach().cpu().numpy().copy()

        self.s, self.v = host("s_m"), host("v_mps")
        self.length, self.lane = host("length_m"), host("lane")
        self.present = host("present")
        self.is_cav = state["is_cav"].detach().bool().cpu().numpy().copy()
        self.inserted: list[tuple[int, int]] = []


class _EntryProcessBase:
    """Placement mechanics shared by the entry processes."""

    def __init__(self, cfg: Optional[EntryConfig], dt_s: float, entry_s_m, batch: int,
                 pool) -> None:
        self.cfg = cfg or EntryConfig()
        self.dt = float(dt_s)
        self.entry_s = tuple(float(x) for x in entry_s_m)
        self.batch = int(batch)
        self.pending = np.zeros((self.batch, N_ENTRANCES), np.int64)
        self.stats = [EntryStats() for _ in range(self.batch)]
        # ``pool`` restricts newcomers to the caller's padding slots; without it a
        # vacated t0 slot could be reused although the recorded ``present_seq`` of
        # that row would drop the entrant on the next step.
        self.pool = None if pool is None else np.asarray(pool, bool)
        self.virgin = None             # never-used padding slots
        self._cav_slots = None         # padding slots that ever held a spawned CAV

    def _pool_row(self, b: int, n_slots: int) -> np.ndarray:
        if self.pool is None:
            return np.ones(int(n_slots), bool)
        if self.pool.ndim == 1:
            return np.broadcast_to(self.pool, (int(n_slots),))
        return self.pool[b]

    def _lane_rear(self, work: _Working, b: int, ln: int):
        """``(rear s, its length, rear speed, mean speed of n_nearby rearmost)`` on ``ln``."""
        rows = np.flatnonzero(work.present[b] & (work.lane[b] == ln))
        if rows.size == 0:
            return None
        order = rows[np.argsort(work.s[b, rows])]
        near = order[:self.cfg.n_nearby]
        return (float(work.s[b, order[0]]), float(work.length[b, order[0]]),
                float(work.v[b, order[0]]), float(np.mean(work.v[b, near])))

    def _place(self, work: _Working, b: int, e: int, ln: int, cav: bool, v_new: float,
               length_new: float) -> bool:
        """Put the arrival into a free padding slot; False when none is available."""
        stt = self.stats[b]
        free = np.flatnonzero(~work.present[b] & self._pool_row(b, work.present.shape[1]))
        if cav:
            cand = free[self.virgin[b, free] & ~self._cav_slots[b, free]]
        else:
            cand = free[~self._cav_slots[b, free]]
        if cand.size == 0:
            stt.no_slot += 1
            return False
        k = int(cand[0])
        work.s[b, k], work.v[b, k] = self.entry_s[ln], v_new
        work.length[b, k], work.lane[b, k] = length_new, ln
        work.present[b, k], work.is_cav[b, k] = True, bool(cav)
        if cav:
            self._cav_slots[b, k] = True
            stt.inserted_cav += 1
        stt.inserted += 1
        stt.per_entrance[e] += 1
        work.inserted.append((b, k))
        return True

    def backlog(self) -> np.ndarray:
        """Per-row entry queue ``[B]``: due arrivals not yet inserted (pending, both entrances).

        Arrivals dropped beyond ``cfg.max_pending`` are NOT included; callers that need the
        full queue (the C4 imagination, SRCWAIT_NOTES.md) use a large ``max_pending``."""
        return self.pending.sum(-1).astype(np.float64)

    def _carry_over(self, b: int, e: int, total: int, done: int) -> None:
        """Keep unplaced arrivals pending, dropping beyond ``max_pending``."""
        left = total - done
        if left > self.cfg.max_pending:
            self.stats[b].dropped_pending += left - self.cfg.max_pending
            left = self.cfg.max_pending
        self.pending[b, e] = left

    @staticmethod
    def _commit(state: dict, work: _Working, new_mask: torch.Tensor) -> torch.Tensor:
        """Write inserted slots back into ``state`` (in place) and flag them."""
        if work.inserted:
            with torch.no_grad():
                for b, k in work.inserted:
                    state["s_m"][b, k] = float(work.s[b, k])
                    state["v_mps"][b, k] = float(work.v[b, k])
                    state["a_mps2"][b, k] = 0.0
                    state["lane"][b, k] = int(work.lane[b, k])
                    state["length_m"][b, k] = float(work.length[b, k])
                    state["present"][b, k] = True
                    state["is_cav"][b, k] = bool(work.is_cav[b, k])
                    new_mask[b, k] = True
        return new_mask


class TorchEntryProcess(_EntryProcessBase):
    """Seeded Poisson arrival process for the torch WM-S rollout.

    ``insert(state)`` writes due arrivals into free slots of ``state`` (a WM-S state
    mapping with ``[B, K]`` ``s_m``/``v_mps``/``a_mps2``/``lane``/``length_m``/
    ``present``/``is_cav`` tensors) and returns the ``[B, K]`` bool mask of slots
    inserted on this step.  The caller must provide enough free padding slots; slot
    allocation prefers never-used slots for CAVs.
    """

    def __init__(self, counter, *, dt_s: float = 0.5, seed: int = 0,
                 batch: int = 1, cfg: Optional[EntryConfig] = None,
                 entry_s_m=ENTRY_S_M, pool=None) -> None:
        super().__init__(cfg, dt_s, entry_s_m, batch, pool)
        self.rates = observed_rates(counter, self.cfg.window_bins) \
            * float(self.cfg.rate_scale)
        self.rng = np.random.default_rng(int(seed))
        self.p_lane1 = None
        self.cav_share = self.cfg.cav_share

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
        """Per-step shared draws: ``(entrance, lane-noise, cav-noise)`` per arrival."""
        n = self.rng.poisson(self.rates * self.dt)
        draws = []
        for entrance in range(N_ENTRANCES):
            for _ in range(int(n[entrance])):
                draws.append((entrance, float(self.rng.random()),
                              float(self.rng.random())))
        return draws

    def _other_lane_speed(self, work: _Working, b: int, ln: int):
        if ln not in (0, 1):
            return None
        other = self._lane_rear(work, b, 1 - ln)
        return None if other is None else other[3]

    def insert(self, state: dict) -> torch.Tensor:
        """Insert due arrivals into ``state`` in place; return the new-slot mask."""
        self._observe(state)
        new_mask = torch.zeros(state["present"].shape, dtype=torch.bool,
                               device=state["present"].device)
        work = _Working(state)
        new_by_e: list[list] = [[] for _ in range(N_ENTRANCES)]
        for entrance, u_lane, u_cav in self._draw():
            new_by_e[entrance].append((u_lane, u_cav))
        for b in range(work.present.shape[0]):
            for e in range(N_ENTRANCES):
                self._insert_entrance(work, b, e, new_by_e[e])
        return self._commit(state, work, new_mask)

    def _insert_entrance(self, work: _Working, b: int, e: int, queue: list) -> None:
        total = int(self.pending[b, e]) + len(queue)
        if total == 0:
            return
        # Carried-over arrivals try either lane and spawn as HDVs.
        items = [(0.0, 2.0)] * int(self.pending[b, e]) + list(queue)
        done = 0
        for u_lane, u_cav in items:
            if e == 0:
                lanes_try = [1, 0] if u_lane < self.p_lane1 else [0, 1]
            else:
                lanes_try = [2]
            if not any(self._try_lane(work, b, e, ln, u_cav) for ln in lanes_try):
                self.stats[b].deferred_steps += 1
                break
            done += 1
        self._carry_over(b, e, total, done)

    def _try_lane(self, work: _Working, b: int, e: int, ln: int, u_cav: float) -> bool:
        rear = self._lane_rear(work, b, ln)
        if rear is None:
            v_new, gap, v_lead = self.cfg.default_speed_mps, _NO_GAP, _NO_GAP
            other = self._other_lane_speed(work, b, ln)
            if other is not None:
                v_new = other
        else:
            s_r, l_r, v_lead, v_new = rear
            gap = s_r - l_r - self.entry_s[ln]
        v_new = max(0.0, float(v_new))
        if not safe_to_insert(gap, v_new, v_lead, self.cfg):
            return False
        cav = bool(u_cav < float(self.cav_share or 0.0))
        return self._place(work, b, e, ln, cav, v_new, NOMINAL_LENGTH_M)

    def summary(self, b: int = 0) -> dict:
        s = self.stats[b]
        return {"rates_veh_s": [float(x) for x in self.rates],
                "p_lane1": self.p_lane1, "cav_share_observed": self.cav_share,
                "inserted": s.inserted, "inserted_cav": s.inserted_cav,
                "per_entrance": list(s.per_entrance),
                "deferred_steps": s.deferred_steps,
                "dropped_pending": s.dropped_pending, "no_slot": s.no_slot}


# ---------------------------------------------------------------------------
# Entries v2: observed-history arrival process
#
# v1 reads only the mean of the recent t0 counter bins and forces every arrival
# through a pooled Poisson intensity, a snapshot lane split, CAV share and speed.
# v2 estimates from the *observed* pre-t0 entry history (insertion times, entry
# speeds, classes, vehicle types; RSU-observable, t <= t0 only) the arrival count
# (recency-weighted rate with bounded trend), the arrival process (gamma renewal
# with the observed coefficient of variation) and HOW vehicles enter (entry-speed
# ratio, entry gap, lane and class mix), all shrunk towards defaults when the
# history is short.  With a shared seed and history every compared controller
# receives the identical entrant stream.
# ---------------------------------------------------------------------------
@dataclass
class EntranceHistory:
    """Per-entrance summary of the observed pre-t0 entry history.

    Arrays are length 2 (mainline, ramp) unless noted.  ``rate`` is veh/s,
    ``headway_mean`` seconds and ``headway_cv`` the observed coefficient of
    variation.  ``speed_mean``/``speed_std`` describe the entry-speed to
    local-traffic-speed ratio, ``gap_mean``/``gap_std`` the entry gap (m).
    ``vtype_share`` is ``[2, N_VTYPES]``, ``lane_p1`` the observed mainline lane-1
    share and ``n_obs`` the observed entry count per entrance.
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


def _counts_counter(times, entrance, t0_s):
    """Build a ``[2, 30]`` 10 s departure counter from observed entry times <= t0."""
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


def build_entrance_history(*, counter=None, times=None, entrance=None, speed=None,
                           local_speed=None, lane=None, gap=None, gap_entrance=None,
                           is_cav=None, vtype=None, mainline_lane_share=None, t0_s=None,
                           window_bins: int = 12, halflife_bins: float = 4.0,
                           shrink_k: float = 8.0,
                           defaults: Optional[EntranceHistory] = None) -> EntranceHistory:
    """Estimate an :class:`EntranceHistory` from t0-observable entry evidence.

    ``counter`` is the ``[2, 30]`` t0 counter.  ``times``/``entrance`` are observed
    entry times/entrances (t <= t0); ``speed``/``local_speed`` the entry and local
    traffic speeds; ``lane`` the entry lane; ``gap`` the entry gap; ``is_cav``/
    ``vtype`` the observed class mix.  Every empirical quantity is shrunk to
    ``defaults`` by ``n_obs / (n_obs + shrink_k)``.
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


def read_insertion_history(base_run_dir, t0_s: float) -> dict:
    """Observed pre-t0 insertion records of a base run (read-only).

    Reads ``run/insertions.jsonl`` and keeps only rows with ``inserted_s`` at or
    before ``t0_s``, never a scheduled future departure.  Returns arrays ``times``,
    ``entrance`` (0 mainline / 1 ramp), ``speed`` (m/s), ``is_cav`` and ``vtype``.
    """
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
            times.append(t)
            entrance.append(_ENTRANCE_NAMES.get(str(row.get("entrance", "mainline")), 0))
            speed.append(float(row.get("depart_speed_mps", 0.0)))
            is_cav.append(bool(row.get("is_cav", False)))
            vtype.append(str(row.get("type_id", "normal_car")))
    order = np.argsort(np.asarray(times, np.float64))
    return {"times": np.asarray(times, np.float64)[order],
            "entrance": np.asarray(entrance, np.int64)[order],
            "speed": np.asarray(speed, np.float64)[order],
            "is_cav": np.asarray(is_cav, bool)[order],
            "vtype": np.asarray(vtype, dtype=object)[order]}


def _norm(probs) -> np.ndarray:
    p = np.asarray(probs, np.float64)
    total = p.sum()
    if total <= 0:
        out = np.zeros_like(p)
        out[0] = 1.0
        return out
    return p / total


class TorchEntryProcessV2(_EntryProcessBase):
    """Seeded observed-history entry process for the torch WM-S rollout.

    Arrivals are a per-entrance gamma renewal fit to the observed inter-arrival
    times (shape ``1/cv^2``) with a recency-weighted rate.  Each arrival samples its
    lane, entry-speed ratio (relative to the local traffic at the entrance), required
    entry gap and class from the observed distributions (shrunk to defaults).  The
    placement mechanics match :class:`TorchEntryProcess`, so a rollout can swap them.
    """

    #: Test-only switch: when True the host-materialisation fast path is disabled so
    #: the unoptimised behaviour can be compared for parity.
    _FORCE_MATERIALISE = False

    def __init__(self, counter=None, *, history: Optional[EntranceHistory] = None,
                 histories: Optional[list] = None, dt_s: float = 0.5, seed: int = 0,
                 batch: int = 1, cfg: Optional[EntryConfig] = None,
                 entry_s_m=ENTRY_S_M, pool=None,
                 defaults: Optional[EntranceHistory] = None,
                 row_seeds: Optional[list] = None, couple_draws: bool = False) -> None:
        if histories is None:
            if history is None:
                history = build_entrance_history(counter=counter, defaults=defaults)
            histories = [history]
        super().__init__(cfg, dt_s, entry_s_m, max(int(batch), len(histories)), pool)
        self.histories = list(histories)
        self.rng = np.random.default_rng(int(seed))
        # Optional per-row generators (``row_seeds``): rows with the same seed and history draw
        # the same arrival stream as long as their traffic (blocking) agrees - common random
        # numbers for counterfactual rows of one start (C4 imagination).  None: one shared rng.
        self.row_rngs = (None if row_seeds is None
                         else [np.random.default_rng(int(x)) for x in row_seeds])
        self.now = np.zeros(self.batch)
        self.t_next = np.full((self.batch, N_ENTRANCES), np.inf)
        self._started = np.zeros(self.batch, bool)
        # Paired training: separate renewal and candidate-identity streams. Merely
        # sharing a row seed loses CRN as soon as one arm defers an insertion.
        self.couple_draws = bool(couple_draws)
        if self.couple_draws:
            if row_seeds is None or len(row_seeds) != self.batch:
                raise ValueError("coupled entry draws require one row seed per batch row")
            self._pair_seeds = list(row_seeds)
            self._headway_index = np.zeros((self.batch, N_ENTRANCES), dtype=np.int64)
            self._candidate_index = np.zeros((self.batch, N_ENTRANCES), dtype=np.int64)

    def _hist(self, b: int) -> EntranceHistory:
        return self.histories[b] if b < len(self.histories) else self.histories[-1]

    def _rng(self, b: int):
        if self.row_rngs is None:
            return self.rng
        return self.row_rngs[b] if b < len(self.row_rngs) else self.row_rngs[-1]

    def _headway(self, b: int, e: int) -> float:
        h = self._hist(b)
        rate = max(float(h.rate[e]), 1.0e-9)
        cv = float(np.clip(h.headway_cv[e], 0.2, 5.0))
        k = 1.0 / (cv * cv)
        scale = (1.0 / rate) / k
        if self.couple_draws:
            rng = np.random.default_rng([self._pair_seeds[b], e, 0,
                                         int(self._headway_index[b, e])])
            self._headway_index[b, e] += 1
        else:
            rng = self._rng(b)
        draw = float(rng.gamma(k, scale))
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
        """Advance row ``b``'s renewal clock one step; entrances with a due arrival."""
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

    def _local_speed(self, work: _Working, b: int, ln: int):
        rear = self._lane_rear(work, b, ln)
        if rear is not None:
            return rear[3]
        if ln in (0, 1):
            other = self._lane_rear(work, b, 1 - ln)
            if other is not None:
                return other[3]
        return None

    def insert(self, state: dict) -> torch.Tensor:
        self._observe(state)
        new_mask = torch.zeros(state["present"].shape, dtype=torch.bool,
                               device=state["present"].device)
        # Advance the renewal clock first; ``_emit_due`` touches no state tensor.
        # Host copies are only materialised when an arrival is due or a deferred
        # insertion is still pending, which avoids device syncs on quiet steps.
        b_n = int(state["present"].shape[0])
        due = [self._emit_due(b) for b in range(b_n)]
        if (not self._FORCE_MATERIALISE and not any(due)
                and not bool(np.any(self.pending))):
            return new_mask
        work = _Working(state)
        for b in range(b_n):
            for e in range(N_ENTRANCES):
                self._insert_entrance(work, b, e, due[b].count(e))
        return self._commit(state, work, new_mask)

    def _insert_entrance(self, work: _Working, b: int, e: int, n_due: int) -> None:
        hist = self._hist(b)
        total = int(self.pending[b, e]) + n_due
        if total == 0:
            return
        done = 0
        rng = self._rng(b)
        for attempt in range(total):
            if self.couple_draws:
                rng = np.random.default_rng([self._pair_seeds[b], e, 1,
                                             int(self._candidate_index[b, e]) + attempt])
            u_cav = float(rng.random())
            u_lane = float(rng.random())
            u_sr = float(rng.normal(hist.speed_mean[e],
                                    max(hist.speed_std[e], 1.0e-3)))
            u_gap = float(rng.normal(hist.gap_mean[e],
                                     max(hist.gap_std[e], 1.0e-3)))
            vt = (int(rng.choice(N_VTYPES, p=_norm(hist.vtype_share[e])))
                  if hist.vtype_share[e].sum() > 0 else 0)
            if e == 0:
                lanes_try = [1, 0] if u_lane < hist.lane_p1 else [0, 1]
            else:
                lanes_try = [2]
            cav = bool(u_cav < float(hist.cav_share[e]))
            if self.cfg.insert_rule != "v2":
                # Same random draws as v2 (common random numbers across rules); the
                # speed ratio and the sampled required gap are not used.
                placed = self._insert_sumo_like(work, b, e, lanes_try, cav, vt)
            else:
                placed = any(self._try_lane(work, b, e, ln, cav, u_sr, u_gap, vt)
                             for ln in lanes_try)
            if not placed:
                self.stats[b].deferred_steps += 1
                break
            done += 1
        dropped = self.stats[b].dropped_pending
        self._carry_over(b, e, total, done)
        if self.couple_draws:
            self._candidate_index[b, e] += done + self.stats[b].dropped_pending - dropped

    def _try_lane(self, work: _Working, b: int, e: int, ln: int, cav: bool,
                  speed_ratio: float, required_gap: float, vtype: int) -> bool:
        rear = self._lane_rear(work, b, ln)
        local = self._local_speed(work, b, ln)
        local = self.cfg.default_speed_mps if local is None else local
        v_new = max(0.0, float(speed_ratio) * float(local))
        if rear is None:
            gap, v_lead = _NO_GAP, _NO_GAP
        else:
            s_r, l_r, v_lead, _ = rear
            gap = s_r - l_r - self.entry_s[ln]
        if not safe_to_insert(gap, v_new, v_lead, self.cfg):
            return False
        if gap < max(0.0, required_gap):
            return False
        return self._place(work, b, e, ln, cav, v_new, float(VTYPE_LENGTH_M[vtype]))

    # -- sumo_like / sumo_like_v2 insertion (EntryConfig.insert_rule) -----------------
    def _entrance_gap(self, work: _Working, b: int, ln: int, length_new: float):
        """``(bumper gap, leader speed)`` for a SUMO departPos="base" entrant on ``ln``."""
        rear = self._lane_rear(work, b, ln)
        if rear is None:
            return _NO_GAP, _NO_GAP
        s_r, l_r, v_lead, _ = rear
        front = self.entry_s[ln] + float(length_new) + SUMO_POSITION_EPS_M
        return s_r - l_r - front, v_lead

    def _sumo_v2(self) -> bool:
        return self.cfg.insert_rule == "sumo_like_v2"

    def _lane_occupancy(self, work: _Working, b: int, ln: int) -> float:
        """Vehicle-length sum on the entrance edge of ``ln`` (SUMO departLane="best" key)."""
        sel = (work.present[b] & (work.lane[b] == ln) & (work.s[b] >= self.entry_s[ln])
               & (work.s[b] - work.length[b] <= SECTION_MERGE_START_M))
        return float(work.length[b, sel].sum())

    def _insert_sumo_like(self, work: _Working, b: int, e: int, lanes_try: list,
                          cav: bool, vtype: int) -> bool:
        length_new = float(VTYPE_LENGTH_M[vtype])
        if len(lanes_try) > 1:
            if self.cfg.sumo_lane_mode == "best_free":
                # Least-occupied entrance lane only; ties keep the sampled order.
                occ = [self._lane_occupancy(work, b, ln) for ln in lanes_try]
                lanes_try = [lanes_try[int(np.argmin(occ))]]
            else:
                gaps = [self._entrance_gap(work, b, ln, length_new)[0] for ln in lanes_try]
                lanes_try = [lanes_try[i] for i in np.argsort(-np.asarray(gaps), kind="stable")]
        return any(self._try_lane_sumo(work, b, e, ln, cav, vtype) for ln in lanes_try)

    def _try_lane_sumo(self, work: _Working, b: int, e: int, ln: int, cav: bool,
                       vtype: int) -> bool:
        length_new = float(VTYPE_LENGTH_M[vtype])
        gap, v_lead = self._entrance_gap(work, b, ln, length_new)
        lane_speeds = tuple(self.cfg.sumo_lane_speed_mps)
        v_cap = float(lane_speeds[min(ln, len(lane_speeds) - 1)]) * VTYPE_SPEED_FACTOR[vtype]
        if gap >= _NO_GAP:
            v_new = v_cap
        else:
            if self._sumo_v2():
                # SUMO 1.27 tests the departure at the END of the insertion step (the
                # lane's rearmost vehicle keeps moving) and departs at the STOP speed
                # behind that end-of-step rear position (the leader gets no speed
                # credit), with the v5 textbook vType settings; measured on the
                # saturated-entrance recordings, see ENTRY_DIAG.md.  Admission stays
                # "bumper gap >= vType minGap".
                gap = gap + max(0.0, float(v_lead)) * self.dt
                v_lead = 0.0
                tau_vt = SUMO_V2_TAU_S[vtype]
                decel_vt = SUMO_V2_DECEL_MPS2[vtype]
                min_gap_vt = SUMO_V2_MIN_GAP_M[vtype]
            else:
                tau_vt = VTYPE_TAU_S[vtype]
                decel_vt = VTYPE_DECEL_MPS2[vtype]
                min_gap_vt = VTYPE_MIN_GAP_M[vtype]
            v_safe = sumo_safe_insert_speed(gap, v_lead, tau_vt, decel_vt, min_gap_vt)
            if v_safe < max(0.0, float(self.cfg.sumo_min_speed_mps)):
                return False
            v_new = min(v_cap, v_safe)
        if not self._place(work, b, e, ln, cav, v_new, length_new):
            return False
        k = work.inserted[-1][1]
        work.s[b, k] = self.entry_s[ln] + length_new + SUMO_POSITION_EPS_M
        return True

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
