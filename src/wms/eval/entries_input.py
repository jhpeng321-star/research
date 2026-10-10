"""Seeded vehicle-entry processes for rollouts with entries ON (default OFF).

Entries are driven by the RSU-observable t0 entrance counter only (never a scheduled
future arrival).  v1 resamples from the observed rate; v2 additionally conditions on the
observed pre-t0 entry history.  New vehicles are confined to the padding pool
(indices >= ``n_t0``): a vacated t0 slot carries the recorded presence sequence that would
drop an entrant on the next step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .backend import RolloutTools
from .errors import PredictorError
from .normalize import DT_S

MAX_PAD_SLOTS = 1024
PAD_SAFETY_FACTOR = 1.5
PAD_MARGIN = 16
COUNTER_SHAPE = (2, 30)
DEFAULT_LOCAL_SPEED_MPS = 20.0


def _entries(tools: RolloutTools) -> Any:
    if tools.entries is None:
        raise PredictorError("entries requested but the backend provides no entry module")
    return tools.entries


def resolve_counter(state: Mapping[str, Any], t0_s: float, tools: RolloutTools) -> np.ndarray:
    """RSU-observable t0 entrance counter ``[2, 30]``."""
    counter = state.get("counter")
    if counter is not None:
        return np.asarray(counter, np.float32).reshape(COUNTER_SHAPE)
    base = state.get("base_run_dir")
    if not base:
        raise PredictorError("entries require state.counter or state.base_run_dir (t0 counters)")
    if tools.counter_reader is None or not Path(str(base)).is_dir():
        raise PredictorError(f"entries base_run_dir unreadable: {base}")
    return tools.counter_reader(Path(str(base)), t0_s)


def _pad_size(total_rate: float, horizon_s: float) -> int:
    """1.5x the expected arrivals + 16, bounded; overflow is counted as no_slot."""
    n_pad = min(MAX_PAD_SLOTS,
                int(np.ceil(float(total_rate) * float(horizon_s) * PAD_SAFETY_FACTOR)) + PAD_MARGIN)
    return max(1, n_pad)


def _pool(n_t0: int, n_pad: int) -> np.ndarray:
    pool = np.zeros(n_t0 + n_pad, dtype=bool)
    pool[n_t0:] = True
    return pool


def entry_process_v1(state: Mapping[str, Any], t0_s: float, horizon_s: float, seed: int,
                     n_t0: int, tools: RolloutTools):
    """Observed-counter entry process and its padding size."""
    mod = _entries(tools)
    cfg = mod.EntryConfig()
    counter = resolve_counter(state, t0_s, tools)
    rates = mod.observed_rates(counter, cfg.window_bins)
    n_pad = _pad_size(rates.sum(), horizon_s)
    proc = mod.TorchEntryProcess(counter, dt_s=DT_S, seed=int(seed), batch=1, cfg=cfg,
                                 pool=_pool(n_t0, n_pad))
    return proc, n_pad


def _entry_history(state: Mapping[str, Any], t0_s: float, kin: np.ndarray,
                   tools: RolloutTools):
    """Observed t<=t0 entry history (recency rate, headways, how-enter)."""
    mod = _entries(tools)
    counter = resolve_counter(state, t0_s, tools)
    lane = kin[:, 1:4].argmax(-1)
    mainline = (lane == 0) | (lane == 1)
    ml = lane[mainline]
    lane_p1 = float((ml == 1).mean()) if ml.size else 0.5

    def rear_speed(sel):
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            return None
        idx = idx[np.argsort(kin[idx, 0])]
        return float(kin[idx[:3], 4].mean())

    local = np.array([rear_speed(mainline) or DEFAULT_LOCAL_SPEED_MPS,
                      rear_speed(lane == 2) or DEFAULT_LOCAL_SPEED_MPS], dtype=np.float64)
    gap_vals, gap_ent = [], []
    for entrance, sel in ((0, mainline), (1, lane == 2)):
        idx = np.flatnonzero(sel)
        idx = idx[np.argsort(kin[idx, 0])]
        for a_i, b_i in zip(idx[:-1], idx[1:]):
            gap = float(kin[b_i, 0] - kin[a_i, 0] - kin[a_i, 6])
            if gap > 0.0:
                gap_vals.append(gap)
                gap_ent.append(entrance)
    gaps = np.asarray(gap_vals, dtype=np.float64)
    gap_e = np.asarray(gap_ent, dtype=np.int64)

    base = state.get("base_run_dir")
    base_dir = Path(str(base)) if base and Path(str(base)).is_dir() else None
    obs = mod.read_insertion_history(base_dir, t0_s) if base_dir is not None else None
    if obs is not None and obs["times"].size:
        return mod.build_entrance_history(
            counter=counter, times=obs["times"], entrance=obs["entrance"], speed=obs["speed"],
            local_speed=local[obs["entrance"]], is_cav=obs["is_cav"], vtype=obs["vtype"],
            mainline_lane_share=lane_p1, gap=gaps, gap_entrance=gap_e, t0_s=t0_s)
    return mod.build_entrance_history(counter=counter, mainline_lane_share=lane_p1,
                                      gap=gaps, gap_entrance=gap_e, t0_s=t0_s)


def entry_process_v2(state: Mapping[str, Any], t0_s: float, horizon_s: float, seed: int,
                     kin: np.ndarray, tools: RolloutTools):
    """Observed-history entry process and its padding size."""
    mod = _entries(tools)
    cfg = mod.EntryConfig()
    history = _entry_history(state, t0_s, kin, tools)
    n_pad = _pad_size(history.rate.sum(), horizon_s)
    proc = mod.TorchEntryProcessV2(history=history, dt_s=DT_S, seed=int(seed), batch=1,
                                   cfg=cfg, pool=_pool(int(kin.shape[0]), n_pad))
    return proc, n_pad
