"""Observable merge-pressure trigger inputs (W2, default OFF via ``wall_pressure_features``).

The trigger head decides wall / no wall from the frozen MAPPO actor trunk plus the move
descriptor; it never saw merge pressure. This block adds, per CAV, ONLY quantities the
CAV can know from its own sensing packet at ``now`` and the selected V2V partners'
direct-sensing packets published exactly one control step earlier (the SAME packet
authorization as ``merge_features``: ego position required, nearest ``V2V_K`` valid
partners within ``V2V_RANGE_M``, no forwarding, no RSU/global field, no hidden or
driver parameters), plus the ego's own wall-commitment history.

Regions reuse registered constants: the observable merge zone ``variety_merge_zone_m``
(ZONE_S0..ZONE_S1, where walls must have released), merge start/end ``MERGE_S0_M``/
``MERGE_S1_M``, ramp lane 2, mainline lanes 0/1, queue speed ``QUEUE_SPEED_MPS``.

Features (physical units; flags are 0/1; every group carries a report age and a missing
bit; missing values are 0 with the bit set -- never imputed):
  ramp_*  on-ramp queue (lane 2, v < 5 m/s, ZONE_S0 <= s <= MERGE_S1) and arrival count
          (lane 2, ZONE_S0 <= s < MERGE_S0); missing when no accepted packet senses lane 2
          over that region; age = oldest contributing packet (0 own, 1 s partner).
  main_*  mainline occupancy near the merge = sum of visible lane-0/1 vehicle lengths in
          the SENSED part of [ZONE_S0, MERGE_S1] / (sensed length summed over lanes 0/1);
          mean speed of those vehicles; sensed coverage fraction; age; missing bits.
  dist_*  ego distance to the merge-zone start and to the merge start (own position).
  wall_*  seconds since the ego's own last wall commitment ended, wall-active flag,
          missing bit when no wall is known for this ego (e.g. a restart mid-episode).
Local coverage is a lower bound, never a claim that the whole region is observable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .merge_features import (CONTROL_DT_S, MERGE_S0_M, MERGE_S1_M, N_LANES, QUEUE_SPEED_MPS,
                             SENSOR_RANGE_M, SensorReport, _local_rows, _valid_report)
from .v2v_obs import V2V_K, V2V_RANGE_M
from .variety import VARIETY_REGISTERED

ZONE_S0_M, ZONE_S1_M = (float(x) for x in VARIETY_REGISTERED['variety_merge_zone_m'])
RAMP_LANE = 2
MAIN_LANES = (0, 1)

PRESSURE_FEATURES = (
    'ramp_queue_veh', 'ramp_arrival_count', 'ramp_report_age_s', 'ramp_missing',
    'main_occupancy', 'main_mean_speed_mps', 'main_coverage_frac', 'main_report_age_s',
    'main_missing', 'main_speed_missing',
    'dist_to_merge_zone_m', 'dist_to_merge_m', 'in_merge_zone', 'position_missing',
    'time_since_last_wall_s', 'wall_active', 'last_wall_missing',
)
PRESSURE_DIM = len(PRESSURE_FEATURES)
# Registered input normalisers for the additive trigger head (physical block / scale).
PRESSURE_SCALE = dict(ramp_queue_veh=10., ramp_arrival_count=10., main_mean_speed_mps=30.,
                      dist_to_merge_zone_m=1000., dist_to_merge_m=1000., time_since_last_wall_s=600.)
SCALE = np.array([PRESSURE_SCALE.get(name, 1.) for name in PRESSURE_FEATURES], dtype=np.float32)
_MISSING_BITS = ('ramp_missing', 'main_missing', 'main_speed_missing', 'position_missing',
                 'last_wall_missing')


@dataclass(frozen=True)
class WallMemory:
    """The ego's own wall-commitment history (its own commands; never another CAV's)."""

    last_end_s: float | None = None
    active: bool = False


def missing_block():
    """Explicitly-missing block (no packet / bank record without sensing)."""
    out = np.zeros(PRESSURE_DIM, dtype=np.float32)
    for name in _MISSING_BITS:
        out[PRESSURE_FEATURES.index(name)] = 1.
    return out


def _union_length(intervals):
    total, end = 0., -math.inf
    for lo, hi in sorted(intervals):
        if hi <= end:
            continue
        total += hi-max(lo, end)
        end = hi
    return total


def _clip(lo, hi):
    lo, hi = max(lo, ZONE_S0_M), min(hi, MERGE_S1_M)
    return (lo, hi) if hi > lo else None


def pressure_features(own: SensorReport | None, cav_id: str,
                      partner_reports: Mapping[str, SensorReport] | None = None, *,
                      now_s: float, partner_ids: Sequence[str] = (),
                      memory: WallMemory | None = None) -> np.ndarray:
    """``[PRESSURE_DIM]`` float32 from local packets only (see the module docstring)."""
    if not np.isfinite(now_s):
        raise ValueError('now_s must be finite')
    out = missing_block()
    f = {name: i for i, name in enumerate(PRESSURE_FEATURES)}
    if memory is not None:
        if memory.active:
            out[f['wall_active']], out[f['last_wall_missing']] = 1., 0.
        elif memory.last_end_s is not None and math.isfinite(memory.last_end_s) \
                and memory.last_end_s <= now_s:
            out[f['time_since_last_wall_s']] = now_s-memory.last_end_s
            out[f['last_wall_missing']] = 0.
    if not _valid_report(own, cav_id, now_s):
        return out
    ego = own.source
    out[f['position_missing']] = 0.
    out[f['dist_to_merge_zone_m']] = ZONE_S0_M-ego.s_m
    out[f['dist_to_merge_m']] = MERGE_S0_M-ego.s_m
    out[f['in_merge_zone']] = float(ZONE_S0_M <= ego.s_m <= ZONE_S1_M)
    partners = []
    for vid in sorted(set(partner_ids)):
        if vid == cav_id:
            continue
        report = (partner_reports or {}).get(vid)
        if not _valid_report(report, vid, now_s-CONTROL_DT_S):
            continue
        distance = abs(report.source.s_m-ego.s_m)
        if distance <= V2V_RANGE_M:
            partners.append((distance, vid, report))
    partners.sort(key=lambda item: (item[0], item[1]))
    accepted = [(0., own)] + [(CONTROL_DT_S, item[2]) for item in partners[:V2V_K]]
    visible, ages = {}, {}
    for age, report in accepted:
        for row in _local_rows(report):
            if row.vehicle_id not in visible:
                visible[row.vehicle_id], ages[row.vehicle_id] = row, age
    covered = {lane: [] for lane in range(N_LANES)}
    cover_age = {lane: [] for lane in range(N_LANES)}
    for age, report in accepted:
        span = _clip(report.source.s_m-SENSOR_RANGE_M, report.source.s_m+SENSOR_RANGE_M)
        if span is None:
            continue
        for lane in set(report.observed_lanes):
            covered[lane].append(span)
            cover_age[lane].append(age)

    def inside(row, lane):
        return any(lo <= row.s_m <= hi for lo, hi in covered[lane])

    rows = [visible[v] for v in sorted(visible)]
    if covered[RAMP_LANE]:
        ramp = [r for r in rows if r.lane == RAMP_LANE and inside(r, RAMP_LANE)]
        out[f['ramp_queue_veh']] = sum(r.v_mps < QUEUE_SPEED_MPS for r in ramp)
        out[f['ramp_arrival_count']] = sum(r.s_m < MERGE_S0_M for r in ramp)
        out[f['ramp_report_age_s']] = max(cover_age[RAMP_LANE])
        out[f['ramp_missing']] = 0.
    sensed = sum(_union_length(covered[lane]) for lane in MAIN_LANES)
    if sensed > 0:
        main = [r for r in rows if r.lane in MAIN_LANES and inside(r, r.lane)]
        out[f['main_occupancy']] = sum(r.length_m for r in main)/sensed
        out[f['main_coverage_frac']] = sensed/(len(MAIN_LANES)*(MERGE_S1_M-ZONE_S0_M))
        out[f['main_report_age_s']] = max(a for lane in MAIN_LANES for a in cover_age[lane])
        out[f['main_missing']] = 0.
        if main:
            out[f['main_mean_speed_mps']] = float(np.mean([r.v_mps for r in main]))
            out[f['main_speed_missing']] = 0.
    return out


def scaled(block):
    """Head input: the physical block divided by the registered normalisers."""
    value = np.asarray(block, dtype=np.float32)
    if value.shape != (PRESSURE_DIM,) or not np.isfinite(value).all():
        raise ValueError('merge-pressure block must be finite [PRESSURE_DIM]')
    return value/SCALE


def update_memory(memory, entries, now_s):
    """Fold the ego-owned wall commitments ``[(members, start_s, until_s)]`` into memory.

    ``memory`` maps CAV id -> last known wall end (latest until of a started wall).
    Returns {cav: WallMemory} as of ``now_s``; never reads other CAVs' commands.
    """
    for members, start, until in entries:
        if start <= now_s:
            for vid in members:
                memory[vid] = max(float(until), memory.get(vid, -math.inf))
    return {vid: WallMemory(last_end_s=end if end <= now_s else None, active=now_s < end)
            for vid, end in memory.items()}
