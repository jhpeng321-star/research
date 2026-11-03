"""V2V intent-sharing per-CAV observation block - REHEARSAL STUB (part 5, branch c4/v2v-rehearsal).

LOCAL STUB of the part-1 observation contract (branch c4/v2v-intent,
tools/coopmwm/v2v_obs.py): same interface (``V2V_DIM``, ``V2VConfig``,
``v2v_features(state, cav_id, last_commands, cfg)``) and same documented
layout, so the integrated module can replace this file unchanged. Used only
while c4/v2v-int has not published V2V_INT_READY.json; the rehearsal discloses
the stub in V2V_REHEARSAL_READY.json.

Contract (partner-major, nearest first, ``K`` partners):
    [gap_m, lane_offset, v_mps, a_mps2, last_cmd, present] * K   -> V2V_DIM = K * 6

- ``gap_m``       = partner_s - ego_s (signed longitudinal, front positive).
- ``lane_offset`` = partner_lane - ego_lane.
- ``a_mps2``      = partner acceleration at that second; falls back to the
                    recorded ``issued_acceleration_mps2`` when the record has
                    no ``a_mps2`` field (branch-trace rows), 0.0 when absent.
- ``last_cmd``    = partner's last ISSUED acceleration (previous control step)
                    from ``last_commands``; 0.0 when never commanded.
- ``present``     = 1.0 for a real partner, 0.0 for a zero-padded slot.

Partners are the nearest OTHER CAVs by ``abs(gap_m)`` ascending (stable on
insertion order for ties) within ``range_m``. Plain numpy in physical units,
torch-free. ``cfg`` accepts a :class:`V2VConfig` or a plain mapping; a mapping
without a ``v2v`` key (the bank-wiring convention, ``max_partners``) is
treated as ENABLED, a mapping with ``v2v: False`` or a ``V2VConfig(v2v=False)``
as disabled (all zeros). ``cfg=None`` is disabled (flag-off byte identity).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

#: Nearest OTHER CAVs per ego (default; overridable via cfg).
V2V_K = 4
#: Per-partner feature order inside one 6-wide block (presence bit last).
V2V_PARTNER_FEATURES = ("gap_m", "lane_offset", "v_mps", "a_mps2", "last_cmd", "present")
#: Fixed output size of :func:`v2v_features` (float32).
V2V_DIM = V2V_K * len(V2V_PARTNER_FEATURES)
#: Communication range in metres (mirror of wms.control.contract.SENSOR_RANGE_M).
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


def _resolve_cfg(cfg) -> dict:
    """Accept V2VConfig, a plain mapping (``v2v`` or ``max_partners`` keys) or None."""
    if cfg is None:
        return dict(v2v=False, k=V2V_K, range_m=V2V_RANGE_M)
    if isinstance(cfg, V2VConfig):
        return dict(v2v=bool(cfg.v2v), k=int(cfg.k), range_m=float(cfg.range_m))
    cfg = dict(cfg)
    enabled = bool(cfg["v2v"]) if "v2v" in cfg else True
    k = int(cfg.get("k", cfg.get("max_partners", V2V_K)))
    return dict(v2v=enabled, k=max(0, k), range_m=float(cfg.get("range_m", V2V_RANGE_M)))


def _records(state):
    """Normalise ``state`` to a vid->record mapping.

    Accepts a vid->record mapping (observation contract) or a branch-trace
    frame ``{'time_s': ..., 'vehicles': [...]}`` (bank-wiring convention).
    """
    if isinstance(state, Mapping) and "vehicles" in state:
        return {v["vid"]: v for v in state["vehicles"] if isinstance(v, Mapping) and "vid" in v}
    if isinstance(state, Mapping):
        return state
    return {v["vid"]: v for v in state if isinstance(v, Mapping) and "vid" in v}


def _last_command(last_commands, vid: str) -> float:
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

    ``state`` maps vehicle id -> record (or is a trace frame with a
    ``vehicles`` list) carrying numeric ``s_m``, ``v_mps``, ``lane`` and
    optional ``is_cav`` (default True; HDV rows are skipped - v2v is
    CAV-to-CAV). ``last_commands`` maps vehicle id -> last issued acceleration
    (float) or a record exposing ``issued_acceleration_mps2``. Disabled cfg,
    unknown ego, or ``cfg=None`` return zeros.
    """
    resolved = _resolve_cfg(cfg)
    k = resolved["k"]
    out = np.zeros(k * len(V2V_PARTNER_FEATURES), np.float32)
    if not resolved["v2v"]:
        return out
    table = _records(state)
    ego = table.get(cav_id)
    if ego is None:
        return out
    ego_s = _scalar(ego, "s_m")
    ego_lane = _scalar(ego, "lane")
    ranked = []
    for position, (vid, record) in enumerate(table.items()):
        if vid == cav_id or not record.get("is_cav", True):
            continue
        gap = _scalar(record, "s_m") - ego_s
        if abs(gap) <= resolved["range_m"]:
            ranked.append((abs(gap), position, vid))
    ranked.sort()
    for block, (_distance, _position, vid) in enumerate(ranked[:k]):
        record = table[vid]
        accel = record.get("a_mps2", record.get("issued_acceleration_mps2"))
        base = block * len(V2V_PARTNER_FEATURES)
        out[base + 0] = _scalar(record, "s_m") - ego_s
        out[base + 1] = _scalar(record, "lane") - ego_lane
        out[base + 2] = _scalar(record, "v_mps")
        out[base + 3] = _scalar(accel, "a_mps2") if isinstance(accel, Mapping) else (
            float(np.asarray(accel, dtype=np.float64).reshape(-1)[0]) if accel is not None else 0.0)
        out[base + 4] = _last_command(last_commands, vid)
        out[base + 5] = 1.0
    return out
