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

Planned-command intent sharing (option ``V2VConfig.v2v_intent``, DEFAULT OFF)
extends every partner slot with the partner's broadcast PLAN: its ``H``
planned absolute-acceleration commands for the next ``H`` control steps (1 s
each, same units/clip as the action) plus an ``intent_valid`` bit.  Each ego
reads only broadcasts its partners published at step ``t-1`` (one-step V2V
latency, so no simultaneous dependency); stale or absent broadcasts are
masked (intent zeros + ``intent_valid = 0``).  The broadcast store is
:class:`V2VIntentBuffer`; the runner calls :func:`publish_intents` after each
policy step.  With ``v2v_intent=False`` the intent functions return EXACTLY
the pre-intent blocks above, so callers can keep one entry point.
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

#: Planned-command horizon (control steps of 1 s; option ``v2v_intent_horizon``).
V2V_INTENT_H = 5
#: Broadcasts are read at t-1, never t (one-step V2V latency).
V2V_INTENT_LATENCY = 1
#: Per-partner intent feature order appended after ``V2V_PARTNER_FEATURES``.
V2V_INTENT_PARTNER_FEATURES = V2V_PARTNER_FEATURES \
    + tuple(f"intent_{h}" for h in range(V2V_INTENT_H)) + ("intent_valid",)
#: Fixed output size of the intent block :func:`v2v_intent_features` (float32).
V2V_INTENT_DIM = V2V_K * len(V2V_INTENT_PARTNER_FEATURES)


@dataclass(frozen=True)
class V2VConfig:
    """v2v block configuration; ``v2v=False`` (default) keeps observations byte-identical.

    ``v2v_intent`` (default False) appends the intent payload to every partner
    slot (block width :data:`V2V_INTENT_DIM` at the default horizon); it only
    takes effect together with ``v2v=True``.  ``v2v_intent_horizon`` is the
    planned-command horizon ``H`` (default 5, >= 1).
    """

    v2v: bool = False
    k: int = V2V_K
    range_m: float = V2V_RANGE_M
    v2v_intent: bool = False
    v2v_intent_horizon: int = V2V_INTENT_H

    def __post_init__(self):
        if int(self.k) < 0:
            raise ValueError(f"k must be >= 0, got {self.k}")
        if not float(self.range_m) > 0.0:
            raise ValueError(f"range_m must be > 0, got {self.range_m}")
        if int(self.v2v_intent_horizon) < 1:
            raise ValueError(f"v2v_intent_horizon must be >= 1, got {self.v2v_intent_horizon}")


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


def intent_block_width(cfg=None) -> int:
    """Width of the v2v block under ``cfg``: :data:`V2V_DIM` when the intent
    option is off (default) or :data:`V2V_INTENT_DIM`-shaped ``K*(6+H+1)`` when
    ``cfg.v2v_intent`` is on (any horizon, ``H = cfg.v2v_intent_horizon``).
    """
    cfg = cfg if cfg is not None else V2VConfig()
    k = int(cfg.k)
    if not cfg.v2v or not cfg.v2v_intent:
        return k * len(V2V_PARTNER_FEATURES)
    return k * (len(V2V_PARTNER_FEATURES) + int(cfg.v2v_intent_horizon) + 1)


def is_intent_block_width(width, k: int = V2V_K) -> bool:
    """True when ``width == K*(6+H+1)`` for some horizon ``H >= 1`` (intent block)."""
    k = int(k)
    rest = int(width) - k * len(V2V_PARTNER_FEATURES)
    return k > 0 and rest > 0 and rest % k == 0


class V2VIntentBuffer:
    """Latest planned-command broadcast per CAV id: ``(step_index, H values)``.

    The runner publishes after each policy step (:func:`publish_intents`);
    readers at step ``t`` accept ONLY broadcasts whose ``step_index == t - 1``
    (:meth:`fresh`), so a same-step publish can never leak into the same step's
    observation.  ``reset()`` on episode start; :meth:`drop` / :meth:`retain`
    remove despawned CAVs so a stale id cannot resurface after respawn.
    Values are stored as ``float32(H,)`` in absolute-acceleration units (the
    caller pre-clips to the action bounds, or passes ``clip=(lo, hi)`` here).
    """

    def __init__(self, horizon: int = V2V_INTENT_H):
        if int(horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.horizon = int(horizon)
        self._latest: dict[str, tuple[int, np.ndarray]] = {}

    def reset(self) -> None:
        """Episode start: drop every broadcast."""
        self._latest.clear()

    def publish(self, step_index: int, intents: Mapping, clip=None) -> int:
        """Store ``{cav_id: H planned accelerations}`` for ``step_index``.

        ``None`` values (or rows of the wrong length) are skipped / rejected:
        a wrong-length row raises, an explicit ``None`` simply publishes
        nothing for that id.  ``clip=(lo, hi)`` clips to the action bounds.
        Returns the number of CAVs published.
        """
        published = 0
        for vid, values in dict(intents).items():
            if values is None:
                continue
            row = np.asarray(values, dtype=np.float64).reshape(-1)
            if row.size != self.horizon:
                raise ValueError(
                    f"intent row for {vid!r} must have horizon={self.horizon} values, got {row.size}")
            if clip is not None:
                lo, hi = float(clip[0]), float(clip[1])
                row = np.clip(row, lo, hi)
            self._latest[str(vid)] = (int(step_index), row.astype(np.float32))
            published += 1
        return published

    def drop(self, vid) -> None:
        """One despawned CAV stops broadcasting."""
        self._latest.pop(str(vid), None)

    def retain(self, vids) -> None:
        """Despawned CAVs drop out: keep only ids in ``vids``."""
        keep = {str(v) for v in vids}
        for vid in [v for v in self._latest if v not in keep]:
            del self._latest[vid]

    def fresh(self, vid, step_index: int):
        """The partner's intent vector iff broadcast at ``step_index - 1``.

        Returns ``float32(H,)`` or ``None`` (absent, stale, or same-step --
        the one-step-latency gate).  ``t`` NEVER sees a ``t`` publish.
        """
        row = self._latest.get(str(vid))
        if row is None or row[0] != int(step_index) - V2V_INTENT_LATENCY:
            return None
        return row[1]

    def ids(self) -> set:
        return set(self._latest)

    def __iter__(self):
        return iter(self._latest)

    def __contains__(self, vid) -> bool:
        return str(vid) in self._latest

    def __len__(self) -> int:
        return len(self._latest)


def publish_intents(buffer: V2VIntentBuffer, step_index: int, intents: Mapping,
                    clip=None) -> int:
    """Runner entry point after each policy step: bulk :meth:`V2VIntentBuffer.publish`.

    ``intents`` maps CAV id -> the ``H`` planned absolute-acceleration commands
    (same units/clip as the action; ``clip=(lo, hi)`` optionally enforced here).
    Returns the number of CAVs published.
    """
    return buffer.publish(step_index, intents, clip=clip)


def v2v_intent_features(state, cav_id, last_commands, intent_buffer, step_index,
                        cfg=None) -> np.ndarray:
    """Intent-augmented v2v block for one CAV: shape ``(intent_block_width(cfg),)``.

    With ``cfg.v2v_intent`` False (default) this returns EXACTLY
    :func:`v2v_features` (byte-identical, width ``V2V_DIM``).  When on (and
    ``cfg.v2v``), every partner slot extends the 6 v2v features with the
    partner's ``H`` planned commands broadcast at ``t-1`` and an
    ``intent_valid`` bit: 1.0 iff the broadcast is FRESH (exactly one step
    old), 0.0 when absent/stale (intent values then zero).  Partner ranking,
    range and zero-padding match :func:`v2v_features` slot for slot.
    """
    cfg = cfg if cfg is not None else V2VConfig()
    if not cfg.v2v_intent:
        return v2v_features(state, cav_id, last_commands, cfg)
    k = int(cfg.k)
    h = int(cfg.v2v_intent_horizon)
    stride = len(V2V_PARTNER_FEATURES) + h + 1
    out = np.zeros(k * stride, np.float32)
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
        base = block * stride
        out[base + 0] = _scalar(record, "s_m") - ego_s
        out[base + 1] = _scalar(record, "lane") - ego_lane
        out[base + 2] = _scalar(record, "v_mps")
        out[base + 3] = _scalar(record, "a_mps2")
        out[base + 4] = _last_command(last_commands, vid)
        out[base + 5] = 1.0
        plan = None if intent_buffer is None else intent_buffer.fresh(vid, step_index)
        if plan is not None:
            row = np.asarray(plan, dtype=np.float64).reshape(-1)
            out[base + 6:base + 6 + min(h, row.size)] = row[:h]
            out[base + stride - 1] = 1.0
    return out


def v2v_intent_slot_block(s_m, lane, v_mps, a_mps2, present, is_cav, slot_ids,
                          last_commands, intent_buffer, step_index,
                          cfg=None) -> np.ndarray:
    """Slot-aligned per-CAV intent block ``[n, intent_block_width(cfg)]``.

    Slot semantics match :func:`v2v_slot_block` exactly (same partner slots,
    carried lane, zero rows for absent / non-CAV / id-less slots); the intent
    payload is appended per partner slot.  ``cfg.v2v_intent`` False (default)
    returns EXACTLY :func:`v2v_slot_block` (byte-identical).
    """
    cfg = cfg if cfg is not None else V2VConfig()
    if not cfg.v2v_intent:
        return v2v_slot_block(s_m, lane, v_mps, a_mps2, present, is_cav, slot_ids,
                              last_commands, cfg)
    s_arr = np.asarray(s_m, dtype=np.float64).reshape(-1)
    out = np.zeros((s_arr.size, intent_block_width(cfg)), np.float32)
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
        out[slot] = v2v_intent_features(table, vid, last_commands, intent_buffer,
                                        step_index, cfg)
    return out


def intent_coverage_stats(intent_block, horizon: int = V2V_INTENT_H) -> dict:
    """Telemetry for one step's intent slot block: partner-intent coverage.

    ``intent_block`` is ``[n, K*(6+H+1)]`` (as produced by
    :func:`v2v_intent_slot_block` with the intent option on).  Per ego row the
    coverage is ``fresh partner slots / present partner slots``; the step value
    is the mean over egos that have at least one present partner (egos without
    partners are excluded, an empty block reports coverage 0.0).
    """
    block = np.asarray(intent_block, dtype=np.float32)
    if block.ndim == 1:
        block = block.reshape(1, -1)
    stride = len(V2V_PARTNER_FEATURES) + int(horizon) + 1
    present_at, valid_at = 5, stride - 1
    n = block.shape[0] if block.size else 0
    k = (block.shape[1] // stride) if n and block.shape[1] >= stride else 0
    covered_rows = present_slots = fresh_slots = 0
    coverage = 0.0
    for row in range(n):
        present = fresh = 0
        for slot in range(k):
            base = slot * stride
            if block[row, base + present_at] > 0.0:
                present += 1
                if block[row, base + valid_at] > 0.0:
                    fresh += 1
        present_slots += present
        fresh_slots += fresh
        if present:
            covered_rows += 1
            coverage += fresh / present
    return dict(coverage=float(coverage / covered_rows) if covered_rows else 0.0,
                egos_with_partners=covered_rows, rows=int(n), present_slots=present_slots,
                fresh_slots=fresh_slots)


def intent_coverage(intent_block, horizon: int = V2V_INTENT_H) -> float:
    """Per-step mean partner-intent coverage (the telemetry scalar)."""
    return float(intent_coverage_stats(intent_block, horizon=horizon)["coverage"])
