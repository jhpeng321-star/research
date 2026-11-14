"""V9ENTRY evaluation mode: the SUMO entrance (source) queue of an evaluation state record.

Entries ON + ``entry_queue_state`` (``RolloutFlags.entry_queue_state``) seeds the imagined entrance
queue of a rollout with the recorded SUMO backlog and the recorded request (demand) rate, as the
C4 imagination does since V9ENTRY (``wms.model.entries.EntranceQueue``).  The queue vector of a
record is, in order of preference:

1. ``state["entry_queue"]`` (6 or 4 values ``[backlog main, ramp, demand main, ramp(, CAV share
   main, ramp)]``), written by a state builder that had the episode's request ledger;
2. derived from the record itself (:func:`derive_entry_queue`): every C4 scenario's request file is
   a deterministic function of (seed, demand, penetration)
   (``traffic_shepherd.scenarios.request_schedule(..., role="exposed")``, the call
   ``traffic_shepherd.c4.env`` builds every C4 scenario with, wallregen worlds included).  SUMO
   inserts the requests of one entrance in request order, so the vehicles inserted by t0 are the
   requests up to the newest one seen on the road (t0 frame + pre-t0 history ids); the backlog is
   the due requests after it.  The demand is the request rate over the fully observed 10 s bins of
   the last 300 s (exactly ``wms.data.c4_episodes.queue_features``); the CAV share is the CAV
   fraction of the requests inserted in the same bins.

Privileged information: the backlog and the demand are recorded SUMO quantities that the RSU
cannot observe (ENTRY_FIX.md section 2).  This mode is an offline WM check, not a deployable input.
Evaluation seeds 964000-964199 are refused.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

QUEUE_KEY = "entry_queue"
COUNTER_BINS, COUNTER_BIN_S = 30, 10.0
EVAL_SEEDS = range(964000, 964200)
_WORLD_RE = re.compile(r"^w(\d+)m(\d+)p(\d+)t(\d+)")
_ID_RE = re.compile(r"^(mainline|ramp)\.(\d+)$")


def world_case(state: Mapping[str, Any]) -> tuple[int, int, float]:
    """``(seed, demand veh/h, penetration)`` of a C4 record (``world_meta`` or the world name)."""
    meta = state.get("world_meta") or {}
    if meta.get("world_id") is not None and meta.get("demand") is not None \
            and meta.get("penetration_pct") is not None:
        return int(meta["world_id"]), int(meta["demand"]), float(meta["penetration_pct"]) / 100.0
    m = _WORLD_RE.match(str(state.get("world", "")))
    if m is None:
        raise ValueError(f"cannot read (seed, demand, penetration) of world {state.get('world')!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)) / 100.0


@lru_cache(maxsize=64)
def _requests(seed: int, demand: int, penetration: float):
    from traffic_shepherd.scenarios import request_schedule

    reqs = request_schedule(int(seed), int(demand), penetration=float(penetration),
                            role="exposed")
    out = {}
    for stream, ent in (("mainline", 0), ("ramp", 1)):
        rows = [r for r in reqs if _ID_RE.match(r.vehicle_id)
                and r.vehicle_id.split(".")[0] == stream]
        rows.sort(key=lambda r: int(r.vehicle_id.split(".")[1]))
        out[ent] = (np.asarray([int(r.vehicle_id.split(".")[1]) for r in rows], np.int64),
                    np.asarray([r.depart_s for r in rows], np.float64),
                    np.asarray([bool(r.is_cav) for r in rows], bool))
    return out


def seen_ids(state: Mapping[str, Any]) -> set[str]:
    """Vehicle ids on the road at or before t0 (t0 frame + pre-t0 history)."""
    ids = {str(v["id"]) for v in (state.get("initial_state") or {}).get("vehicles", [])}
    hist = state.get("history") or {}
    ids.update(str(x) for x in hist.get("ids", []) if x)
    return ids


def derive_entry_queue(state: Mapping[str, Any], t0_s: float) -> tuple[np.ndarray, dict]:
    """6-value queue vector of a record from its regenerated requests (see the module doc)."""
    seed, demand, pen = world_case(state)
    if seed in EVAL_SEEDS:
        raise ValueError(f"evaluation seed {seed} is refused")
    reqs = _requests(seed, demand, round(pen, 6))
    seen: dict[int, list[int]] = {0: [], 1: []}
    for vid in seen_ids(state):
        m = _ID_RE.match(vid)
        if m:
            seen[0 if m.group(1) == "mainline" else 1].append(int(m.group(2)))
    n = max(1, min(COUNTER_BINS, int(float(t0_s) // COUNTER_BIN_S)))
    window = n * COUNTER_BIN_S
    backlog, rate, share = np.zeros(2), np.zeros(2), np.zeros(2)
    info: dict = {"seed": seed, "demand_vph": demand, "penetration": pen, "window_s": window}
    for ent in (0, 1):
        idx, dep, cav = reqs[ent]
        depart_of = dict(zip(idx.tolist(), dep.tolist()))
        due = dep <= float(t0_s) + 1e-9
        unknown = [i for i in seen[ent] if i not in depart_of]
        if unknown:
            raise ValueError(f"entrance {ent}: vehicles {unknown[:3]} are not requests of {seed}")
        last = max(seen[ent]) if seen[ent] else -1
        late = [i for i in seen[ent] if depart_of[i] > float(t0_s) + 1e-6]
        if late:
            raise ValueError(f"entrance {ent}: vehicles seen before their request time: {late[:3]}")
        inserted = idx <= last
        backlog[ent] = float((due & ~inserted).sum())
        age = float(t0_s) - dep
        recent = due & (age < window)
        rate[ent] = float(recent.sum()) / window
        ins_recent = recent & inserted
        share[ent] = float(cav[ins_recent].mean()) if ins_recent.any() else 0.0
        info[f"last_inserted_{ent}"] = int(last)
    vec = np.concatenate([backlog, rate, share])
    info["vector"] = vec.tolist()
    return vec, info


def state_entry_queue(state: Mapping[str, Any], t0_s: float) -> tuple[np.ndarray, str]:
    """``(queue vector, source)``: the record's ``entry_queue`` or :func:`derive_entry_queue`."""
    vec = state.get(QUEUE_KEY)
    if vec is not None:
        return np.asarray(vec, np.float64).reshape(-1), "record"
    return derive_entry_queue(state, t0_s)[0], "derived_requests"
