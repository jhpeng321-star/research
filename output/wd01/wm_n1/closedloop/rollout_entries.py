"""Open-road ENTRIES for the closed-loop A11 rollout (flag-gated, default OFF).

Why: the t0-snapshot rollout animates only the vehicles present at t0. This
module adds vehicles at the network entrances (mainline lanes 0/1 at the lane
start, on-ramp / accel lane 2 at its start) as the rollout runs.

Source of the arrival process (findings, 2026-10-05):
* The engine's own arrival path (``Rollout2Config.arrivals_enabled`` +
  ``bundle.arrival`` head + ``ArrivalSampler``) exists, but (a) the A11 member's
  arrival head is DEGENERATE (count_mu = 0, speed_mu = 1, log_sigma = -6 on
  the TRAIN smoke counter: it would insert nothing), (b) it inserts generic
  HDV-only tokens "behind the rearmost occupant" of lane 1 / 2 only (no
  mainline lane 0, not at the entrance, no safe-gap test).
* So this module draws arrivals from the OBSERVED entrance counters of the t0
  token (``state.counter [2, 30]``, 10 s bins, bin 0 most recent): rate per
  entrance = mean count over the most recent ``window_bins`` bins / 10 s;
  Poisson draws per 0.5 s step with a seeded generator (deterministic per
  state; identical draws for every plan of a batch = common random numbers, so
  plan-vs-native comparisons are not diluted by arrival noise).
* Mainline arrivals are split over lanes 0 / 1 by the observed share of
  vehicles on the two mainline lanes in the snapshot. CAV vs HDV per arrival
  from the OBSERVED CAV share of the snapshot (never configured penetration
  or hidden driver types).
* Initial speed = mean speed of the (up to) 3 rearmost present vehicles on
  that lane (the nearby observed traffic), else the lane mean, else the other
  mainline lane, else 20 m/s; capped by a safe-gap rule. Insert only if the
  bumper gap to the lane's rearmost vehicle >= min_gap + headway * v +
  closing-braking distance; otherwise the arrival is kept pending (<=
  ``max_pending`` per entrance, FIFO) and retried on the next step.
* Spawned HDVs are ordinary HDV rows (same IDM/MOBIL + residual). Spawned CAVs
  are flagged ``is_cav`` and run NATIVE (action code 0): per-CAV plans are
  held-constant codes for the t0 CAVs and carry no fleet policy for newcomers.
* The recorded ``cav_plan_ch`` is "alive since t0" (``present0 * cumprod``
  of the target mask), so a vehicle that enters after t0 is NOT counted in the
  120 s plan summary, recorded or rollout; ``long_plan_from_rollout`` already
  implements that (spawned slots have msk[0] False). Entries therefore act on
  the plan summary only indirectly, through the traffic they add to the HDV
  dynamics; the n_cav decay of recorded plans comes from exits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

NONE_CODE = -1
NATIVE_CODE = 0
NOMINAL_LENGTH_M = 5.0
N_COUNTER_BINS = 30
COUNTER_BIN_S = 10.0


@dataclass
class EntryConfig:
    window_bins: int = 12          # most recent bins (x10 s) for the rate
    min_gap_m: float = 2.0
    headway_s: float = 1.0
    decel_mps2: float = 3.0
    max_pending: int = 6
    n_nearby: int = 3
    default_speed_mps: float = 20.0
    cav_share: Optional[float] = None   # None -> observed in the snapshot
    rate_scale: float = 1.0


@dataclass
class EntryStats:
    inserted: int = 0
    inserted_cav: int = 0
    deferred_steps: int = 0        # (entrance, row) steps with a blocked arrival
    dropped_pending: int = 0
    no_slot: int = 0
    per_entrance: list = field(default_factory=lambda: [0, 0])


def observed_rates(counter: np.ndarray, window_bins: int = 12) -> np.ndarray:
    """Per-entrance arrival rate [2] in veh/s from the t0 entrance counter."""
    c = np.asarray(counter, np.float64).reshape(2, N_COUNTER_BINS)
    n = max(1, min(int(window_bins), N_COUNTER_BINS))
    return c[:, :n].sum(axis=1) / (n * COUNTER_BIN_S)


class EntryProcess:
    """Seeded arrival process shared by all rows of one batched rollout."""

    def __init__(self, state, params, *, seed: int = 0,
                 cfg: Optional[EntryConfig] = None, batch: int = 1):
        self.cfg = cfg or EntryConfig()
        self.rates = observed_rates(state.counter, self.cfg.window_bins) \
            * float(self.cfg.rate_scale)
        self.dt = float(params.dt)
        off = np.asarray(params.off, np.float64)
        # entrance start position per lane (world s coordinate = -offset)
        self.entry_s = {0: -float(off[0]), 1: -float(off[1]), 2: -float(off[2])}
        mask = np.asarray(state.veh_mask, bool)
        if hasattr(state, "rollout_mask") and state.rollout_mask is not None:
            mask = np.asarray(state.rollout_mask, bool)
        lane = np.asarray(state.veh_kin)[:, 2].astype(np.int64)
        n0 = int((mask & (lane == 0)).sum())
        n1 = int((mask & (lane == 1)).sum())
        self.p_lane1 = 0.5 if n0 + n1 == 0 else n1 / float(n0 + n1)
        n_all = int(mask.sum())
        self.cav_share = (float(self.cfg.cav_share) if self.cfg.cav_share is not None
                          else (float((np.asarray(state.is_cav, bool) & mask).sum())
                                / n_all if n_all else 0.0))
        self.rng = np.random.default_rng(int(seed))
        self.batch = int(batch)
        self.pending = np.zeros((self.batch, 2), np.int64)
        self.stats = [EntryStats() for _ in range(self.batch)]
        self.virgin = None             # slots never present (set on first step)
        self._cav_slots = None

    # ------------------------------------------------------------ helpers
    @staticmethod
    def safe_to_insert(gap_m: float, v_new: float, v_lead: float, cfg: EntryConfig) -> bool:
        brake = max(0.0, v_new - v_lead) ** 2 / (2.0 * cfg.decel_mps2)
        return gap_m >= cfg.min_gap_m + cfg.headway_s * v_new + brake

    def _lane_rear(self, s, v, length, present, lane, b, ln):
        """(rear s, its length, speeds of the n_nearby rearmost) on lane ln."""
        rows = np.flatnonzero(present[b] & (lane[b] == ln))
        if rows.size == 0:
            return None
        order = rows[np.argsort(s[b, rows])]
        near = order[: self.cfg.n_nearby]
        return float(s[b, order[0]]), float(length[b, order[0]]), \
            float(v[b, order[0]]), float(np.mean(v[b, near])), float(np.mean(v[b, rows]))

    # ------------------------------------------------------------- step
    def draw(self):
        """Per-step shared draws: arrivals per entrance, lane/cav/speed noise."""
        n = self.rng.poisson(self.rates * self.dt)
        draws = []
        for e in range(2):
            for _ in range(int(n[e])):
                draws.append((e, self.rng.random(), self.rng.random()))
        return draws

    def step(self, w, acts, st) -> None:
        """Insert due arrivals into the (float64) world ``w`` before a step."""
        s, v, a = w.s_m.numpy(), w.v_mps.numpy(), w.a_mps2.numpy()
        length, lane = w.length_m.numpy(), w.lane.numpy()
        present, is_cav = w.present.numpy(), w.is_cav.numpy()
        slot_id = w.slot_id.numpy()
        wg = w.wall_group.numpy() if getattr(w, "wall_group", None) is not None else None
        codes = acts.action_code.numpy()
        b_n, k_n = present.shape
        if self.virgin is None:
            self.virgin = (~present) & (slot_id < 0)          # [B, K]
            self._cav_slots = np.zeros((b_n, k_n), bool)
        draws = self.draw()
        new_by_e = [[], []]
        for e, u_lane, u_cav in draws:
            new_by_e[e].append((u_lane, u_cav))
        for b in range(b_n):
            stt = self.stats[b]
            for e in range(2):
                queue = [(u_lane, u_cav) for (u_lane, u_cav) in new_by_e[e]]
                total = int(self.pending[b, e]) + len(queue)
                if total == 0:
                    continue
                # carried-over arrivals try either lane and spawn as HDVs
                items = [(0.0, 2.0)] * int(self.pending[b, e]) + queue
                done = 0
                blocked = False
                for (u_lane, u_cav) in items:
                    if blocked:
                        break
                    if e == 0:
                        lanes_try = [1, 0] if u_lane < self.p_lane1 else [0, 1]
                    else:
                        lanes_try = [2]
                    placed = False
                    for ln in lanes_try:
                        rear = self._lane_rear(s, v, length, present, lane, b, ln)
                        s_e = self.entry_s[ln]
                        if rear is None:
                            v_new, gap, v_lead = self.cfg.default_speed_mps, 1e9, 1e9
                            other = self._lane_rear(s, v, length, present, lane, b, 1 - ln) \
                                if ln in (0, 1) else None
                            if other is not None:
                                v_new = other[3]
                        else:
                            s_r, l_r, v_lead, v_new, _ = rear
                            gap = s_r - l_r - s_e
                        v_new = max(0.0, float(v_new))
                        if not self.safe_to_insert(gap, v_new, v_lead, self.cfg):
                            continue
                        free = np.flatnonzero(~present[b])
                        cav = bool(u_cav < self.cav_share)
                        if cav:
                            cand = free[self.virgin[b, free] & ~self._cav_slots[b, free]]
                        else:
                            cand = free[~self._cav_slots[b, free]]
                        if cand.size == 0:
                            stt.no_slot += 1
                            continue
                        k = int(cand[0])
                        s[b, k], v[b, k], a[b, k] = s_e, v_new, 0.0
                        length[b, k], lane[b, k] = NOMINAL_LENGTH_M, ln
                        present[b, k], is_cav[b, k] = True, bool(cav)
                        if wg is not None:
                            wg[b, k] = -1
                        slot_id[b, k] = int(st.next_slot_id[b])
                        st.next_slot_id[b] += 1
                        codes[b, k] = NATIVE_CODE if cav else NONE_CODE
                        st.held_target_mps[b, k] = float("nan")
                        st.held_reason[b, k] = 0
                        st.lc_last[b, k] = -1e9
                        if cav:
                            self._cav_slots[b, k] = True
                            stt.inserted_cav += 1
                        stt.inserted += 1
                        stt.per_entrance[e] += 1
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

    def summary(self, b: int = 0) -> dict:
        s = self.stats[b]
        return {"rates_veh_s": [float(x) for x in self.rates], "p_lane1": self.p_lane1,
                "cav_share_observed": self.cav_share, "inserted": s.inserted,
                "inserted_cav": s.inserted_cav, "per_entrance": list(s.per_entrance),
                "deferred_steps": s.deferred_steps, "dropped_pending": s.dropped_pending,
                "no_slot": s.no_slot}
