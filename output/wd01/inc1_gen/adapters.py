"""Thin call-site adapters + the two-call-site identity harness (A2.6).

The ONE shared generator module must yield IDENTICAL commands whether the
SUMO arm calls it through a traci-like state rendering or the WM rollout
calls it through a tensor-like (array) rendering, on the same synthetic state
sequence (``INC1_GENERATORS.json`` gate, addendum 2 A2.6).  The adapters here
are the only place call-site data shapes are interpreted; the generator sees
only :class:`~inc1_gen.states.VehState` records.

SUMO-side rendering: a dict of traci-like per-vehicle state dicts, as the INC1
runner collects them (``getSpeed`` / ``getLanePosition`` / ``getLaneIndex`` /
``getAcceleration`` / ``getTypeID`` plus the local leader gap).

WM-side rendering: tensor-like arrays (numpy or anything ``numpy.asarray``
accepts, e.g. torch CPU tensors converted by the caller), one value per slot,
with ``NaN`` marking "no leader" and an unknown native acceleration.

Numpy-only.  No traci, no torch.  Zero SUMO.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np

from .generators import GeneratorStep
from .states import ObstacleRecord, VehState

# ------------------------------------------------------------ SUMO-side shim
_SPEED_KEYS = ("speed", "speed_mps", "v")
_POS_KEYS = ("lane_position", "lane_position_m", "position", "s_down", "s")
_LANE_KEYS = ("lane_index", "lane", "lane_phys")
_ACCEL_KEYS = ("acceleration", "acceleration_mps2", "a")
_NATIVE_ACCEL_KEYS = ("native_accel", "native_accel_mps2")
_LEADER_GAP_KEYS = ("leader_gap", "leader_gap_m")
_LEADER_V_KEYS = ("leader_speed", "leader_v", "leader_speed_mps")


def _first(mapping: Mapping, keys: Iterable[str], default=None):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def states_from_traci_like(vehicles: Mapping[str, Mapping], *,
                           cav_ids: Iterable[str] | None = None
                           ) -> list[VehState]:
    """Render a dict of traci-like per-vehicle state dicts into ``VehState``.

    Each inner dict may use traci-ish snake-case keys (``speed``,
    ``lane_position``, ``lane_index``, ``acceleration``, ``type``) or the
    canonical ``VehState`` names.  ``acceleration`` is the measured
    acceleration (-> ``a``); the native IDM bound must be passed explicitly
    as ``native_accel`` (what the runner captures from ``getAcceleration``
    for a native-running vehicle).  ``is_cav`` comes from an explicit
    ``is_cav``/``cav`` entry, else from ``cav_ids``, else from a ``type``
    string containing ``"cav"``.  A ``leader`` entry may be a
    ``(gap, leader_v)`` pair.  Vehicles listed in the mapping but flagged
    ``present=False`` are rendered as not-present (left the network).
    """
    cav = set(cav_ids or ())
    out = []
    for vid in sorted(vehicles):
        row = vehicles[vid]
        if "is_cav" in row:
            is_cav = bool(row["is_cav"])
        elif "cav" in row:
            is_cav = bool(row["cav"])
        elif vid in cav:
            is_cav = True
        else:
            type_id = str(_first(row, ("type", "type_id", "vtype"), ""))
            is_cav = "cav" in type_id.lower()
        leader = row.get("leader")
        if leader is not None:
            leader_gap, leader_v = float(leader[0]), float(leader[1])
        else:
            leader_gap = _first(row, _LEADER_GAP_KEYS)
            leader_v = _first(row, _LEADER_V_KEYS)
            leader_gap = None if leader_gap is None else float(leader_gap)
            leader_v = None if leader_v is None else float(leader_v)
        out.append(VehState(
            vehicle_id=str(vid),
            is_cav=is_cav,
            lane_phys=int(_first(row, _LANE_KEYS, -1)),
            s_down=float(_first(row, _POS_KEYS)),
            v=float(_first(row, _SPEED_KEYS)),
            a=float(_first(row, ("acceleration", "a"), 0.0) or 0.0),
            leader_gap=leader_gap,
            leader_v=leader_v,
            native_accel=_maybe_float(_first(row, _NATIVE_ACCEL_KEYS)),
            present=bool(row.get("present", True)),
        ))
    return out


def _maybe_float(value):
    return None if value is None else float(value)


# -------------------------------------------------------------- WM-side shim
def states_from_wm_arrays(vehicle_ids: Iterable[str], *, s_down, v, a,
                          lane_phys, is_cav, present,
                          native_accel=None, leader_gap=None,
                          leader_v=None) -> list[VehState]:
    """Render tensor-like (array) call-site state into ``VehState``.

    Every array is one value per slot, aligned with ``vehicle_ids``; any
    array-like accepted by ``numpy.asarray`` works (WM rollouts pass their
    float64 tensors' ``detach().cpu().numpy()`` views).  ``NaN`` in
    ``leader_gap`` / ``leader_v`` / ``native_accel`` means "none/unknown".
    """
    ids = [str(x) for x in vehicle_ids]
    columns = {
        "s_down": _col(s_down, len(ids)),
        "v": _col(v, len(ids)),
        "a": _col(a, len(ids)),
        "lane_phys": _col(lane_phys, len(ids)),
        "is_cav": _col(is_cav, len(ids)),
        "present": _col(present, len(ids)),
        "native_accel": (_col(native_accel, len(ids))
                         if native_accel is not None else [None] * len(ids)),
        "leader_gap": (_col(leader_gap, len(ids))
                       if leader_gap is not None else [None] * len(ids)),
        "leader_v": (_col(leader_v, len(ids))
                     if leader_v is not None else [None] * len(ids)),
    }
    out = []
    for i, vid in enumerate(ids):
        gap = _nan_to_none(columns["leader_gap"][i])
        l_v = _nan_to_none(columns["leader_v"][i])
        out.append(VehState(
            vehicle_id=vid,
            is_cav=bool(columns["is_cav"][i]),
            lane_phys=int(columns["lane_phys"][i]),
            s_down=float(columns["s_down"][i]),
            v=float(columns["v"][i]),
            a=float(columns["a"][i]),
            leader_gap=gap,
            leader_v=l_v,
            native_accel=_nan_to_none(columns["native_accel"][i]),
            present=bool(columns["present"][i]),
        ))
    # canonical id-sorted order, identical to the SUMO-side shim
    return sorted(out, key=lambda s: s.vehicle_id)


def _col(values, n: int):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != n:
        raise ValueError(f"array length {arr.size} != {n} slots")
    return arr.tolist()


def _nan_to_none(value):
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


# --------------------------------------------------- two-call-site identity
def run_two_site_identity(arm: str, t0_sumo, frames_sumo, t0_wm, frames_wm,
                          *, obstacle: ObstacleRecord, t0_s: float,
                          control_dt_s: float = 1.0) -> dict:
    """The A2.6 harness: one generator, two call sites, identical commands.

    Runs the SAME scripted arm over the SAME synthetic state sequence rendered
    twice: ``t0_sumo`` / ``frames_sumo`` are traci-like renderings
    (:func:`states_from_traci_like` inputs); ``t0_wm`` / ``frames_wm`` are
    ``(vehicle_ids, kwargs)`` renderings for :func:`states_from_wm_arrays`.
    Each frame entry is ``(now_s, frame)``.

    Returns a report dict; ``identical`` is True iff the t0 states from both
    shims are equal records, both sites latch the same feasibility, and every
    per-tick command dict and native set match exactly (bit-for-bit floats).
    A later wiring task runs this over its real call-site renderings before
    the gated launch.
    """
    from .generators import make_generator

    states_a0 = states_from_traci_like(t0_sumo)
    ids_b0, kwargs_b0 = t0_wm
    states_b0 = states_from_wm_arrays(ids_b0, **kwargs_b0)
    gen_a = make_generator(arm, states_a0, obstacle, t0_s=t0_s,
                           control_dt_s=control_dt_s)
    gen_b = make_generator(arm, states_b0, obstacle, t0_s=t0_s,
                           control_dt_s=control_dt_s)
    report: dict = {"arm": arm, "ticks": 0, "identical": True,
                    "mismatches": []}
    if not states_equal(states_a0, states_b0):
        report["identical"] = False
        report["mismatches"].append({"tick": "t0", "kind": "states"})
    if gen_a.feasible != gen_b.feasible or (gen_a.infeasible_reason
                                            != gen_b.infeasible_reason):
        report["identical"] = False
        report["mismatches"].append({"tick": "t0", "kind": "feasibility"})
    for (now_a, frame_a), (now_b, frame_b) in zip(frames_sumo, frames_wm):
        step_a = gen_a.step(float(now_a), states_from_traci_like(frame_a))
        ids_b, kwargs_b = frame_b
        step_b = gen_b.step(float(now_b),
                            states_from_wm_arrays(ids_b, **kwargs_b))
        report["ticks"] += 1
        if float(now_a) != float(now_b) or not steps_equal(step_a, step_b):
            report["identical"] = False
            report["mismatches"].append({"tick": report["ticks"],
                                         "kind": "step",
                                         "t_a": float(now_a),
                                         "t_b": float(now_b)})
    report["n_mismatches"] = len(report["mismatches"])
    return report


def steps_equal(a: GeneratorStep, b: GeneratorStep) -> bool:
    """Exact (bit-for-bit) equality of two generator steps."""
    return (a.t_s == b.t_s
            and a.commands.keys() == b.commands.keys()
            and all(a.commands[k] == b.commands[k] for k in a.commands)
            and a.native == b.native
            and a.feasible == b.feasible
            and a.infeasible_reason == b.infeasible_reason)


def states_equal(a: list[VehState], b: list[VehState]) -> bool:
    """Exact equality of two rendered state lists (ids and every field)."""
    if len(a) != len(b):
        return False
    for sa, sb in zip(a, b):
        if sa != sb:
            return False
    return True
