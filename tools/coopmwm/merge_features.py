"""Observable-only, opt-in cooperation-layer inputs (T2F).

This module accepts *local sensor packets*, never a simulation-wide state. The
caller supplies the ego's current sensing packet and the IDs of its selected V2V
partners. A partner packet must have been published exactly one control step
earlier, by that partner, within radio range. Each packet contributes only rows
within the publisher's sensing range and explicitly sensed lanes. No forwarding
of another partner's reports, global queue, RSU field, or hidden parameters is
accepted by this interface.

The float32 block is in physical units, lane-major (0/1 mainline, 2 ramp).
``queue_length_veh`` counts visible vehicles at speed < 5 m/s, upstream of or
inside the merge. ``approaching_count`` and its mean speed cover vehicles
strictly upstream of merge start (including stopped vehicles). Both include
the ego where applicable and are LOCAL LOWER BOUNDS, not network totals.
``rear_gap_m`` is the nearest visible rear vehicle's bumper gap relative to
the ego's longitudinal position, separately for each lane, floored at zero;
``rear_speed_mps`` is that vehicle's absolute speed. No extrapolation is used.

Own sensing wins duplicate vehicle IDs; otherwise the nearest publisher wins
(ties by publisher ID). This deliberately mixes current local sensing with
t-1 remote sensing, with explicit report-age and source-missingness features.
``lane_missing`` means no accepted packet declares sensing coverage of that
lane; coverage is local, never a claim that the whole lane is observable.
Missing speeds/gaps are zero with their respective missingness bits set.

Constants mirror the existing observation/V2V contracts and rsu_field queue
definition; contract tests pin them. Keeping plain constants here makes this
module numpy-only. There are no simulator, torch, filesystem or job APIs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from tools.coopmwm.v2v_obs import V2V_K, V2V_RANGE_M

SENSOR_RANGE_M = 100.0                 # wms.control.contract.SENSOR_RANGE_M
CONTROL_DT_S = 1.0                    # wms.control.contract.CONTROL_DT_S
MERGE_S0_M = 0.0                      # wms.control.contract.MERGE_S0_M
MERGE_S1_M = 75.44                    # wms.control.contract.MERGE_S1_M
QUEUE_SPEED_MPS = 5.0                 # wms.control.rsu_field.QUEUE_SPEED_MPS
N_LANES = 3                          # wms.control.contract.N_LANES

MERGE_LANE_FEATURES = (
    "queue_length_veh", "approaching_count", "approaching_mean_speed_mps",
    "rear_gap_m", "rear_speed_mps", "lane_missing",
    "approaching_speed_missing", "rear_missing",
)
MERGE_SOURCE_FEATURES = (
    "own_missing", "partner_reports_missing", "partner_report_count",
    "partner_report_age_s", "visible_vehicle_count",
)
MERGE_FEATURES = tuple(
    f"lane_{lane}_{name}" for lane in range(N_LANES) for name in MERGE_LANE_FEATURES
) + MERGE_SOURCE_FEATURES
MERGE_DIM = len(MERGE_FEATURES)


@dataclass(frozen=True)
class MergeFeaturesConfig:
    """Add a separate observation block only when explicitly enabled."""

    merge_features: bool = False


@dataclass(frozen=True)
class ObservedVehicle:
    """One sensor observation; length is measured geometry, not a driver parameter."""

    vehicle_id: str
    s_m: float
    lane: int
    v_mps: float
    length_m: float


@dataclass(frozen=True)
class SensorReport:
    """Atomic local sensing at ``timestamp_s`` in the shared WM road frame.

    ``source`` is the reporting CAV's own measurement. ``vehicles`` contains
    only its direct sensor observations (HDVs and CAVs alike), not relayed
    reports. ``observed_lanes`` must explicitly state actual sensor coverage;
    an empty covered lane differs from a lane with no sensing coverage. The
    publisher's own row is always known, independent of sensor lane coverage.
    The timestamp applies to every row, including the publisher's position.
    """

    timestamp_s: float
    source: ObservedVehicle
    observed_lanes: tuple[int, ...]
    vehicles: tuple[ObservedVehicle, ...] = ()


def _valid_vehicle(row: ObservedVehicle) -> bool:
    if not isinstance(row, ObservedVehicle):
        return False
    try:
        return (
            isinstance(row.vehicle_id, str) and bool(row.vehicle_id)
            and row.lane in range(N_LANES)
            and np.isfinite([row.s_m, row.v_mps, row.length_m]).all()
            and row.v_mps >= 0.0 and row.length_m > 0.0
        )
    except (TypeError, ValueError):
        return False


def _valid_report(report: SensorReport, source_id: str, timestamp_s: float) -> bool:
    if not isinstance(report, SensorReport) or not _valid_vehicle(report.source):
        return False
    if report.source.vehicle_id != source_id:
        return False
    try:
        if not np.isfinite(report.timestamp_s) or report.timestamp_s != timestamp_s:
            return False
        if any(lane not in range(N_LANES) for lane in report.observed_lanes):
            return False
        if not all(_valid_vehicle(row) for row in report.vehicles):
            return False
        # An ambiguous atomic packet is missing, rather than a fabricated zero.
        ids = [row.vehicle_id for row in report.vehicles]
        return len(ids) == len(set(ids))
    except (TypeError, ValueError):
        return False


def _local_rows(report: SensorReport):
    yield report.source
    for row in report.vehicles:
        if (row.lane in report.observed_lanes
                and abs(row.s_m - report.source.s_m) <= SENSOR_RANGE_M):
            yield row


def merge_features(
    own: SensorReport | None,
    cav_id: str,
    partner_reports: Mapping[str, SensorReport] | None = None,
    *,
    now_s: float,
    partner_ids: Sequence[str] = (),
    cfg: MergeFeaturesConfig | None = None,
    k: int = V2V_K,
    range_m: float = V2V_RANGE_M,
) -> np.ndarray:
    """Return ``[MERGE_DIM]`` float32, default-off zeros.

    Enabled but absent/invalid own sensing produces missingness bits, with no
    partner fallback: the ego position is required to authorize radio range.
    Only the nearest ``V2V_K`` explicitly selected, valid partner publishers
    within ``V2V_RANGE_M`` contribute. Membership and geometry use the supplied
    observable packets only. Same-step, stale, future, misidentified and
    malformed partner packets are unavailable. Time is compared exactly in
    seconds, without truncation, so fractional control times are supported.
    The function does not mutate or cache any input. ``k``/``range_m`` default to the
    registered V2V contract; only the opt-in W1 coop awareness block passes others.
    """
    cfg = cfg if cfg is not None else MergeFeaturesConfig()
    out = np.zeros(MERGE_DIM, dtype=np.float32)
    if not cfg.merge_features:
        return out
    lanes = out[:N_LANES * len(MERGE_LANE_FEATURES)].reshape(N_LANES, -1)
    lanes[:, 5:] = 1.0
    sources = out[N_LANES * len(MERGE_LANE_FEATURES):]
    sources[:2] = 1.0
    if not np.isfinite(now_s):
        raise ValueError("now_s must be finite")
    if not _valid_report(own, cav_id, now_s):
        return out
    sources[0] = 0.0
    ego = own.source
    partners = []
    for vid in sorted(set(partner_ids)):
        if vid == cav_id:
            continue
        report = (partner_reports or {}).get(vid)
        if not _valid_report(report, vid, now_s - CONTROL_DT_S):
            continue
        distance = abs(report.source.s_m - ego.s_m)
        if distance <= range_m:
            partners.append((distance, vid, report))
    partners.sort(key=lambda item: (item[0], item[1]))
    accepted = [own] + [item[2] for item in partners[:k]]
    sources[2] = len(accepted) - 1
    if len(accepted) > 1:
        sources[1] = 0.0
        sources[3] = CONTROL_DT_S
    visible = {}
    covered = set()
    for report in accepted:
        covered.update(report.observed_lanes)
        for row in _local_rows(report):
            visible.setdefault(row.vehicle_id, row)
    sources[4] = len(visible)
    # Sort IDs so reductions and tie handling are independent of packet order.
    rows = [visible[vid] for vid in sorted(visible)]
    for lane in range(N_LANES):
        lane_rows = [row for row in rows if row.lane == lane]
        lanes[lane, 0] = sum(
            row.s_m <= MERGE_S1_M and row.v_mps < QUEUE_SPEED_MPS for row in lane_rows
        )
        approaching = [row for row in lane_rows if row.s_m < MERGE_S0_M]
        lanes[lane, 1] = len(approaching)
        if approaching:
            lanes[lane, 2] = np.mean([row.v_mps for row in approaching])
            lanes[lane, 6] = 0.0
        lanes[lane, 5] = float(lane not in covered)
        rear = [row for row in lane_rows
                if row.vehicle_id != cav_id and row.s_m <= ego.s_m]
        if rear:
            nearest = min(rear, key=lambda row: (ego.s_m - row.s_m, row.vehicle_id))
            lanes[lane, 3] = max(0.0, ego.s_m - nearest.s_m - ego.length_m)
            lanes[lane, 4] = nearest.v_mps
            lanes[lane, 7] = 0.0
    return out


def with_merge_features(
    observation: Mapping,
    own: SensorReport | None,
    cav_id: str,
    partner_reports: Mapping[str, SensorReport] | None = None,
    *,
    now_s: float,
    partner_ids: Sequence[str] = (),
    cfg: MergeFeaturesConfig | None = None,
) -> Mapping:
    """Add a separate ``merge_features`` block for the cooperation layer.

    Off: return the EXACT original mapping without reading any sensor inputs.
    On: shallow-copy the mapping, preserving every existing value byte-for-byte;
    do not widen the frozen MAPPO input. Integration owns per-CAV/slot stacking.
    """
    cfg = cfg if cfg is not None else MergeFeaturesConfig()
    if not cfg.merge_features:
        return observation
    return {**observation, "merge_features": merge_features(
        own, cav_id, partner_reports, now_s=now_s, partner_ids=partner_ids, cfg=cfg,
    )}
