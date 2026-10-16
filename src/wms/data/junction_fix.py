"""JUNCTIONFIX: junction-transit rows of TRAIN pack windows in the evaluation convention.

A SUMO junction internal lane (``:``-prefixed edge, at the edge boundaries s ~ 0 m and
s ~ 75.4 m, i.e. inside the merge area) has no modelled lane.  The packs store such a row with
an all-zero lane one-hot, the vehicle's REAL speed and its LAST on-lane ``s`` (frozen while the
vehicle crosses the junction); the canonical window decodes the lane as ``-1``.

``_PackWindowLoader`` keeps the slot present and excludes the row from the per-car loss
(``loss_valid``), but ``_states_from_teacher`` clips lane ``-1`` to lane 0, so with the switch
off every junction row sits in ``states`` as a lane-0 car with a frozen ``s``:

* a vehicle that is on a junction at t0 STARTS the rollout on lane 0 (fake lane, possibly a
  fake lane-0 neighbour of the real lane-0 cars), while its later loss targets are on its real
  lane (lane 1 / 2) -> the lane loss asks for a change that never happened;
* a CAV on a junction is teacher-forced with lane ``-1`` (invisible to followers) and a frozen
  ``s`` that then jumps by about two steps of travel;
* anything that reads ``states`` with ``present`` (not ``loss_valid``) sees fake
  1 -> 0 -> 1 "lane changes" at the edge boundaries (the GAPRULE census artefact).

Evaluation (``wms.eval.rollout_inputs.recorded_teacher``) and deployment
(``wms.control.lanes.LaneDwell``) keep the vehicle's previous lane through a transit and skip
the loss on the transit step.  ``junction_fix = "ffill"`` applies that convention to the
training windows:

* lane: the last on-lane lane of the same vehicle (forward fill; for a vehicle that is on a
  junction at t0 the last on-lane row BEFORE t0 is read from the pack, same-vehicle continuity
  via ``target_mask``, at most ``LOOKBACK_TICKS`` back and never across the run start);
* ``s``: linear interpolation in time between the bracketing on-lane rows of the same vehicle
  (the route coordinate is continuous across the junction; the internal-lane length itself is
  not part of it, so the interpolation is the coordinate the WM frame would see); held at the
  last on-lane ``s`` when no later on-lane row exists inside the window (evaluation convention);
* ``v`` / ``a``: unchanged (the recorded values are real SUMO measurements);
* ``loss_valid``: unchanged (the transit step is still not supervised - its ``s`` is filled);
* a t0 vehicle on a junction without any on-lane pre-t0 row is dropped from the window
  (evaluation drops every t0 junction vehicle, ``wms.eval.rollout_inputs.t0_vehicles``) and,
  when the window carries the entries-v2 road presence, removed from it for as long as it stays
  in its slot, so the aggregate targets count exactly the population the model simulates.

The filled lane is also written into ``cav_teacher`` (the aggregate loss reads its recorded
lane from there), so recorded RSU cell fields count a transiting vehicle at its filled lane and
``s`` - as the model does.  Pack bytes are never modified.
"""
from __future__ import annotations

from typing import Any

import numpy as np

#: Allowed values of ``data.junction_fix``.
JUNCTION_FIX_MODES = ("off", "ffill")
#: How far before t0 (0.5 s ticks) the pack is searched for the last on-lane row of a vehicle
#: that is on a junction at t0 (transits last a few ticks; 40 ticks = 20 s).
LOOKBACK_TICKS = 40

_S, _LANE, _V, _A = 0, 1, 2, 3


def pre_t0_rows(corpus: Any, tick: int, slots: np.ndarray, *,
                lookback: int = LOOKBACK_TICKS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Last on-lane pre-t0 row of the t0 vehicle of each slot: ``(lane, s, age_ticks)`` ``[K]``.

    Only rows of the SAME vehicle count (``target_mask[u]`` = the slot's vehicle is still there
    at ``u+1``); the search stops at the first absent row, at the run start and after
    ``lookback`` ticks.  ``lane = -1`` where none exists (and for slots not in ``slots``).
    """
    slots = np.asarray(slots, dtype=bool)
    k = slots.shape[0]
    lane = np.full(k, -1, dtype=np.int64)
    s = np.zeros(k, dtype=np.float32)
    age = np.zeros(k, dtype=np.int64)
    lo = max(int(corpus.run_start(int(tick))), int(tick) - int(lookback))
    n = int(tick) - lo
    if n <= 0 or not slots.any():
        return lane, s, age
    kin = np.asarray(corpus.array("veh_kin")[lo:tick], np.float32)            # [n,K,7]
    occ = np.asarray(corpus.array("veh_mask")[lo:tick]).astype(bool)          # [n,K]
    same = np.asarray(corpus.array("target_mask")[lo:tick]).astype(bool)      # [n,K]
    open_ = slots.copy()                     # still on the same-vehicle chain, nothing found
    for j in range(n - 1, -1, -1):
        open_ &= same[j] & occ[j]            # row j belongs to the t0 vehicle
        hit = open_ & (kin[j, :, 1:4].sum(-1) > 0)
        lane = np.where(hit, kin[j, :, 1:4].argmax(-1), lane)
        s = np.where(hit, kin[j, :, 0], s)
        age = np.where(hit, int(tick) - (lo + j), age)
        open_ &= ~hit
        if not open_.any():
            break
    return lane, s, age


def fill_junction_rows(teacher: np.ndarray, occupied: np.ndarray, same_next: np.ndarray,
                       pre_lane: np.ndarray, pre_s: np.ndarray, pre_age: np.ndarray):
    """Forward-fill lanes and interpolate ``s`` over junction rows (``lane < 0``).

    ``teacher [T,K,4]`` = (s, lane, v, a) with lane ``-1`` on a junction row; ``occupied [T,K]``
    = the slot holds a vehicle (``veh_mask``; False beyond the window); ``same_next [T-1,K]`` =
    the same vehicle is in the slot at ``t+1`` (``target_mask``); ``pre_* [K]`` = the last
    on-lane pre-t0 row of the t0 vehicle (:func:`pre_t0_rows`).

    Returns ``(teacher_fixed, filled [T,K], unfilled [T,K])``: ``unfilled`` marks junction rows
    without any earlier on-lane row of the same vehicle (left unchanged, lane ``-1``).
    """
    teacher = np.array(teacher, dtype=np.float32, copy=True)
    t_n, k = teacher.shape[:2]
    occupied = np.asarray(occupied, dtype=bool)
    same_next = np.asarray(same_next, dtype=bool)
    lane = np.rint(teacher[..., _LANE]).astype(np.int64)
    s = teacher[..., _S]
    junction = occupied & (lane < 0)
    on_lane = occupied & (lane >= 0)
    if not junction.any():
        return teacher, np.zeros((t_n, k), bool), np.zeros((t_n, k), bool)

    def continuous(t: int) -> np.ndarray:    # rows t-1 and t hold the same vehicle
        return occupied[t - 1] & occupied[t] & same_next[t - 1]

    prev_lane = np.full((t_n, k), -1, dtype=np.int64)
    prev_s = np.zeros((t_n, k), dtype=np.float64)
    prev_t = np.zeros((t_n, k), dtype=np.float64)
    last_lane = np.where(occupied[0], np.asarray(pre_lane, np.int64), -1)
    last_s = np.asarray(pre_s, np.float64).copy()
    last_t = -np.asarray(pre_age, np.float64)
    for t in range(t_n):
        if t > 0:
            last_lane = np.where(continuous(t), last_lane, -1)
        prev_lane[t], prev_s[t], prev_t[t] = last_lane, last_s, last_t
        last_lane = np.where(on_lane[t], lane[t], last_lane)
        last_s = np.where(on_lane[t], s[t], last_s)
        last_t = np.where(on_lane[t], float(t), last_t)

    next_ok = np.zeros((t_n, k), dtype=bool)
    next_s = np.zeros((t_n, k), dtype=np.float64)
    next_t = np.zeros((t_n, k), dtype=np.float64)
    have = np.zeros(k, dtype=bool)
    cur_s = np.zeros(k, dtype=np.float64)
    cur_t = np.zeros(k, dtype=np.float64)
    for t in range(t_n - 1, -1, -1):
        if t < t_n - 1:
            have &= continuous(t + 1)
        next_ok[t], next_s[t], next_t[t] = have, cur_s, cur_t
        have = have | on_lane[t]
        cur_s = np.where(on_lane[t], s[t], cur_s)
        cur_t = np.where(on_lane[t], float(t), cur_t)

    filled = junction & (prev_lane >= 0)
    unfilled = junction & ~filled
    times = np.arange(t_n, dtype=np.float64)[:, None]
    span = np.maximum(next_t - prev_t, 1.0)
    frac = np.clip((times - prev_t) / span, 0.0, 1.0)
    s_new = np.where(next_ok, prev_s + (next_s - prev_s) * frac, prev_s)
    teacher[..., _LANE] = np.where(filled, prev_lane, teacher[..., _LANE])
    teacher[..., _S] = np.where(filled, s_new, teacher[..., _S]).astype(np.float32)
    return teacher, filled, unfilled


def run_prefix(present: np.ndarray, cont: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """``[T,K]`` rows of ``slots`` that still hold the slot's t0 vehicle (``present`` chain)."""
    present = np.asarray(present, dtype=bool)
    cont = np.asarray(cont, dtype=bool)
    out = np.zeros_like(present)
    alive = present[0] & np.asarray(slots, dtype=bool)
    out[0] = alive
    for t in range(1, present.shape[0]):
        alive = alive & cont[t - 1] & present[t]
        out[t] = alive
    return out
