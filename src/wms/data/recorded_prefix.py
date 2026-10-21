"""Raw full-scene, native-tick history for first-passage survival conditioning.

The final row is the t0 anchor, not a decision to replay. ``continue[t,k]``
certifies identity continuity from row t to t+1. Feature histories cannot replace
this contract: they omit historical neighbours and native-tick continuity.
"""
from __future__ import annotations

import numpy as np


def recorded_prefix(corpus, tick: int, *, cav_array: str = "veh_step") -> dict:
    """Read only this run's recorded observations through t0 (no outcome arrays)."""
    bounds = corpus.run_bounds(tick)
    if bounds is None:
        raise ValueError("recorded prefix requires explicit run bounds")
    lo = int(bounds[0])
    states = np.asarray(corpus.array("veh_kin")[lo:tick + 1], np.float32).copy()
    present = np.asarray(corpus.array("veh_mask")[lo:tick + 1], bool).copy()
    continuity = np.asarray(corpus.array("target_mask")[lo:tick], bool).copy()
    # Ownership is a current observation, not a driver parameter/truth table.
    cav = (np.asarray(corpus.array("veh_step")[lo:tick + 1, :, 7]) > .5
           if cav_array == "veh_step" else
           np.asarray(corpus.array(cav_array)[lo:tick + 1], bool))
    return dict(states=states, present=present, is_cav=cav & present,
                **{"continue": continuity})


def array_prefix(arrays: dict, i: int, n: int, t0: np.ndarray) -> dict | None:
    """Explicit raw array history, including t0; no inference from feature history."""
    if not all(key in arrays for key in ("history_kin", "history_mask", "history_target_mask")):
        return None
    states = np.asarray(arrays["history_kin"], np.float32)
    present = np.asarray(arrays["history_mask"], bool)
    continuity = np.asarray(arrays["history_target_mask"], bool)
    if states.ndim == 3:
        states, present, continuity = states[None], present[None], continuity[None]
    if (states.shape[0] != n or states.shape[2:] != t0.shape
            or present.shape != states.shape[:-1]
            or continuity.shape != (n, states.shape[1] - 1, states.shape[2])):
        raise ValueError("raw recorded prefix shape mismatch")
    if not np.array_equal(states[i, -1], t0):
        raise ValueError("raw recorded prefix must end at the window t0 state")
    cav = arrays.get("history_is_cav")
    if cav is None:
        # Ownership must be observed, including neighbours that left before t0.
        return None
    cav = np.asarray(cav, bool)
    if cav.ndim == 2:
        cav = cav[None]
    if cav.shape != present.shape:
        raise ValueError("history_is_cav shape mismatch")
    result = dict(states=states[i].copy(), present=present[i].copy(),
                  is_cav=cav[i].copy(), **{"continue": continuity[i].copy()})
    timelines = arrays.get("history_action_timeline")
    if timelines is not None:
        result["action_timeline"] = {
            key: (value if key == "control_dt_s" else np.asarray(value)[i])
            for key, value in timelines.items()}
    return result
