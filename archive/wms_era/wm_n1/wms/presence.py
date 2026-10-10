"""Junction-aware slot presence for WM-S training and offline evaluation.

A SUMO junction internal lane (``:``-prefixed) is a physical transit, not a
departure: the vehicle is still on the modelled road and keeps its identity, it
just has no modelled lane id for one or more 0.5 s steps.  The recorded per-step
presence used by the packs/teacher is combined with a cumulative AND, so one
junction step used to permanently drop the slot from the loss and from the
evaluation matched set.

:func:`junction_carry_presence` rebuilds the occupancy so a slot is only
permanently absent when the vehicle truly leaves the modelled road or its
identity changes.  A junction step is carried with the last known state and is
reported separately (``junction_step``) so callers can skip the loss on that
step (its lane target is undefined) while keeping the vehicle present after it.

Both the training windows and the offline evaluator consume this helper so the
presence contract is identical on every path.  Only current and past
information is used: the returned occupancy is monotone non-increasing, so no
future step can revive a departed slot (no future leakage).
"""
from __future__ import annotations

import numpy as np


def junction_carry_presence(present, junction):
    """Monotone occupancy carrying junction internal-lane transits.

    ``present`` and ``junction`` are matching boolean ``[T, K]`` arrays:
    ``present[t, k]`` is the recorded occupancy on a modelled lane;
    ``junction[t, k]`` is a recorded junction internal-lane transit for the same
    vehicle (row exists, identity unchanged, ``lane < 0``).  A slot already
    present at ``t-1`` stays present at ``t`` when either ``present`` or
    ``junction`` holds, and is otherwise permanently dropped.

    Returns ``(occupancy, junction_step)`` where ``junction_step = occupancy &
    junction`` marks the carried steps whose loss must be skipped.
    """
    present = np.asarray(present, dtype=bool)
    junction = np.asarray(junction, dtype=bool)
    if present.ndim != 2 or present.shape != junction.shape:
        raise ValueError("present and junction must be matching [T,K] arrays")
    occupancy = np.zeros_like(present)
    if present.shape[0] == 0:
        return occupancy, occupancy
    occupancy[0] = present[0]
    for t in range(1, present.shape[0]):
        occupancy[t] = occupancy[t - 1] & (present[t] | junction[t])
    return occupancy, (occupancy & junction)


def junction_mask_from_lane(lane_one_hot, occupied, *, same_vehicle=None):
    """Junction transits from a lane one-hot and an occupancy mask.

    ``lane_one_hot`` ``[T, K, L]`` is the pack lane encoding; a junction row has
    all zeros (no modelled lane).  ``occupied`` ``[T, K]`` marks a slot that
    still carries a vehicle.  ``same_vehicle`` ``[T, K]`` (optional) forces the
    transit to follow an unchanged identity so a slot reused by a new vehicle is
    never carried.
    """
    lane = np.asarray(lane_one_hot)
    occupied = np.asarray(occupied, dtype=bool)
    no_lane = lane.sum(-1) <= 0
    junction = occupied & no_lane
    if same_vehicle is not None:
        junction = junction & np.asarray(same_vehicle, dtype=bool)
    return junction
