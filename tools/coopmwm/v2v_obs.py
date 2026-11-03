"""V2V intent-sharing per-CAV observation block (C4 part 1: OBSERVATION).

Shared contract for the parallel C4 v2v workstreams (branches ``c4/v2v-intent``,
``c4/v2v-model``, ``c4/v2v-bank``): the ``v2v`` block gives each CAV direct
partner awareness of the ``K`` nearest OTHER CAVs within comm range.  Per
partner: longitudinal gap, lane offset, speed, acceleration and the partner's
LAST ISSUED command (acceleration from the previous control step), plus a
presence bit, zero-padded.  The block is plain numpy in PHYSICAL units (the
``wms.control`` convention: networks normalise, observations do not).

Import this module from the model / bank workstreams; ``V2V_DIM`` and
``v2v_features`` are the contract.  Everything is gated by ``V2VConfig.v2v``
(default False): when the flag is off the block is all zeros and the wired call
sites add nothing at all, so the pre-v2v observation is byte-identical.

Layout (partner-major, nearest first, ``K = V2VConfig.k``):
    [gap_m, lane_offset, v_mps, a_mps2, last_cmd, present] * K   -> V2V_DIM = K * 6

- ``gap_m``       = partner_s - ego_s (signed longitudinal, front positive; NOT a
                    bumper gap -- radio distance, no length correction).
- ``lane_offset`` = partner_lane - ego_lane (junction rows use the carried lane).
- ``last_cmd``    = partner's last issued acceleration (previous control step),
                    from ``last_commands``; 0.0 when never commanded.
- ``present``     = 1.0 for a real partner, 0.0 for a zero-padded slot.

Partners are ranked by ``abs(gap_m)`` ascending (ties by mapping insertion
order, i.e. stable), restricted to ``abs(gap_m) <= V2VConfig.range_m``.  The
default range mirrors ``wms.control.contract.SENSOR_RANGE_M`` (100 m), kept as
a plain float here so the module stays torch-free for SUMO-worker startup; a
unit test asserts the equality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

#: Nearest OTHER CAVs per ego (default; overridable via V2VConfig.k).
V2V_K = 4
#: Per-partner feature order inside one 6-wide block (presence bit last).
V2V_PARTNER_FEATURES = ("gap_m", "lane_offset", "v_mps", "a_mps2", "last_cmd", "present")
#: Fixed output size of :func:`v2v_features` (float32).
V2V_DIM = V2V_K * len(V2V_PARTNER_FEATURES)
#: Mirror of ``wms.control.contract.SENSOR_RANGE_M`` (unit test asserts equality).
V2V_RANGE_M = 100.0


@dataclass(frozen=True)
class V2VConfig:
    """v2v block configuration; ``v2v=False`` (default) keeps observations byte-identical."""

    v2v: bool = False
    k: int = V2V_K
    range_m: float = V2V_RANGE_M

    def __post_init__(self):
        if int(self.k) < 0:
            raise ValueError(f"k must be >= 0, got {self.k}")
        if not float(self.range_m) > 0.0:
            raise ValueError(f"range_m must be > 0, got {self.range_m}")


def _scalar(record: Mapping, key: str, default: float = 0.0) -> float:
    value = record.get(key, default)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def _last_command(last_commands: Mapping | None, vid: str) -> float:
    """Partner's last issued command; floats directly, records via their field."""
    row = (last_commands or {}).get(vid)
    if row is None:
        return 0.0
    if isinstance(row, Mapping):
        for key in ("issued_acceleration_mps2", "cmd_accel_mps2", "accel"):
            if key in row and row[key] is not None:
                return _scalar(row, key)
        return 0.0
    return float(np.asarray(row, dtype=np.float64).reshape(-1)[0])


def v2v_features(state, cav_id, last_commands, cfg=None) -> np.ndarray:
    """Fixed-size ``float32`` v2v block for one CAV: shape ``(V2V_DIM,)``.

    ``state`` maps vehicle id -> record with numeric ``s_m`` (front-bumper
    longitudinal position, WM frame), ``v_mps``, ``a_mps2`` and ``lane`` (0/1/2;
    default 0 when absent).  Records may carry ``is_cav`` (default True); HDV
    rows are skipped -- v2v is CAV-to-CAV.  ``last_commands`` maps vehicle id ->
    last issued acceleration command (float) or a record mapping exposing
    ``issued_acceleration_mps2``; missing / None -> 0.0.  ``cfg`` is a
    :class:`V2VConfig`; when ``cfg.v2v`` is False the block is all zeros (and
    wired call sites add nothing, keeping the observation byte-identical).
    Unknown egos (absent from ``state``) also return zeros.
    """
    cfg = cfg if cfg is not None else V2VConfig()
    k = int(cfg.k)
    out = np.zeros(k * len(V2V_PARTNER_FEATURES), np.float32)
    if not cfg.v2v:
        return out
    ego = state.get(cav_id)
    if ego is None:
        return out
    ego_s = _scalar(ego, "s_m")
    ego_lane = _scalar(ego, "lane")
    ranked = []
    for position, (vid, record) in enumerate(state.items()):
        if vid == cav_id or not record.get("is_cav", True):
            continue
        gap = _scalar(record, "s_m") - ego_s
        if abs(gap) <= float(cfg.range_m):
            ranked.append((abs(gap), position, vid))
    ranked.sort()
    for block, (_distance, _position, vid) in enumerate(ranked[:k]):
        record = state[vid]
        base = block * len(V2V_PARTNER_FEATURES)
        out[base + 0] = _scalar(record, "s_m") - ego_s
        out[base + 1] = _scalar(record, "lane") - ego_lane
        out[base + 2] = _scalar(record, "v_mps")
        out[base + 3] = _scalar(record, "a_mps2")
        out[base + 4] = _last_command(last_commands, vid)
        out[base + 5] = 1.0
    return out


def v2v_slot_block(s_m, lane, v_mps, a_mps2, present, is_cav, slot_ids, last_commands,
                   cfg=None) -> np.ndarray:
    """Slot-aligned per-CAV v2v block ``[K, V2V_DIM]`` (``float32``).

    ``K`` = number of slots (any array length); ``slot_ids`` maps each slot to a
    vehicle id or None.  Rows for absent / non-CAV / id-less slots are zeros.
    ``lane`` must already be effective (junction rows pre-carried by the caller,
    ``traffic_shepherd.c4.env``).  ``cfg.v2v`` False -> all zeros.
    """
    cfg = cfg if cfg is not None else V2VConfig()
    s_arr = np.asarray(s_m, dtype=np.float64).reshape(-1)
    out = np.zeros((s_arr.size, int(cfg.k) * len(V2V_PARTNER_FEATURES)), np.float32)
    if not cfg.v2v:
        return out
    present = np.asarray(present, dtype=bool).reshape(-1)
    is_cav = np.asarray(is_cav, dtype=bool).reshape(-1)
    lane = np.asarray(lane, dtype=np.float64).reshape(-1)
    v_arr = np.asarray(v_mps, dtype=np.float64).reshape(-1)
    a_arr = np.asarray(a_mps2, dtype=np.float64).reshape(-1)
    table = {}
    for slot in range(min(s_arr.size, len(slot_ids))):
        vid = slot_ids[slot]
        if vid is None or not present[slot] or not is_cav[slot]:
            continue
        table[vid] = {"s_m": float(s_arr[slot]), "lane": float(lane[slot]),
                      "v_mps": float(v_arr[slot]), "a_mps2": float(a_arr[slot])}
    for slot in range(min(s_arr.size, len(slot_ids))):
        vid = slot_ids[slot]
        if vid is None or not present[slot] or not is_cav[slot]:
            continue
        out[slot] = v2v_features(table, vid, last_commands, cfg)
    return out
