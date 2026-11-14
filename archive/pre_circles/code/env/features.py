"""Observation features shared by Selfish / Flat-MARL / Shepherd.

Contents (previously the ``env/features/`` package):
  - Neighbor extraction:   `collect_neighbors`, `NeighborFeatures`, `turn_signal_intent`
  - Trust weighting:       `TrustTracker` (EMA, Trust-MARL style)
  - Merge-zone stats:      `build_zone_features`, `distance_to_merge`
  - Observation assembly:  `ObservationAssembler`, `raw_obs_dim`

The raw observation vector packs:
  ego[9] (lane, speed/max, accel, time_to_merge, density_id, leader_gap,
          leader_rel_speed, queue_ahead, queue_behind) |
  zone[12] (pre_occ, pre_spd/max, merge_occ, merge_spd/max, ramp_occ,
           ramp_spd/max, ramp_queue, merge_outflow,
           corridor_delay_excess, fair_gap_excess, main_delay_rate,
           ramp_delay_rate) |
  up to max_neighbors * (4 features + 1 trust_weight)

All speed features are normalized by the highway max speed (35 m/s) at
assembly time; distances stay in meters. The queue/outflow features are the
congestion detectors: queue_ahead/queue_behind count slow same-edge vehicles
within 300 m, and merge_outflow is the shared EMA discharge of the merge
edge (see env.rewards.MergeOutflowTracker).

This keeps the Gym Box observation fixed-size while the PyTorch encoder
performs trust-weighted neighbor embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import traci

from config.constants import (
    MERGE_EDGE,
    PRE_MERGE_EDGE,
    PRE_MERGE_LANE_LENGTH,
    RAMP_EDGE,
    RAMP_LANE_LENGTH,
)
from env.snapshot import get_active_snapshot
from env.traci_query import leader_info

# ---------------------------------------------------------------------------
# Neighbor feature extraction
# ---------------------------------------------------------------------------

# SUMO vehicle signal bit flags.
VEH_SIGNAL_BLINKER_RIGHT = 1
VEH_SIGNAL_BLINKER_LEFT = 2


@dataclass
class NeighborFeatures:
    vehicle_id: str
    rel_distance: float
    rel_speed: float
    rel_lane: float
    turn_signal: float  # -1 left, 0 none, +1 right
    abs_distance: float


@dataclass
class ZoneConfig:
    """Physical zone definitions for bottleneck detection.

    Defines three zones by physical distance from the bottleneck (meters).
    Used to compute zone features without relying on SUMO edge IDs.
    """

    bottleneck_x: float = 5000.0  # x-coordinate of bottleneck junction
    pre_start: float = 0.0  # pre zone start (upstream)
    pre_end: float = 4000.0  # pre zone end
    merge_start: float = 4000.0  # merge zone start (near bottleneck)
    merge_end: float = 5000.0  # merge zone end (at bottleneck)
    ramp_lane: str = "upstream_2"  # dropped lane identifier


def _zone_occupancy(
    vehicle_ids: List[str], zone: Tuple[float, float], lane: Optional[str] = None
) -> float:
    """Count vehicles in a physical zone (x-range), optionally filtered by lane."""
    if not vehicle_ids:
        return 0.0
    x_min, x_max = zone
    count = 0
    for vid in vehicle_ids:
        x = _veh_x(vid)
        if x is None:
            continue
        if lane is not None:
            lane_id = _veh_lane(vid)
            if lane_id != lane:
                continue
        if x_min <= x <= x_max:
            count += 1
    return float(count)


def _zone_mean_speed(
    vehicle_ids: List[str],
    zone: Tuple[float, float],
    lane: Optional[str] = None,
    fallback: float = 0.0,
) -> float:
    """Average speed of vehicles in a physical zone, optionally filtered by lane."""
    x_min, x_max = zone
    speeds = []
    for vid in vehicle_ids:
        x = _veh_x(vid)
        if x is None:
            continue
        if lane is not None:
            lane_id = _veh_lane(vid)
            if lane_id != lane:
                continue
        if x_min <= x <= x_max:
            speed = _veh_speed(vid)
            if speed is not None:
                speeds.append(speed)
    if not speeds:
        return fallback
    return float(np.mean(speeds))


def _veh_x(veh_id: str) -> Optional[float]:
    """Vehicle x-coordinate (position along road)."""
    snap = get_active_snapshot()
    if snap is not None:
        pos = snap.position(veh_id)
        if pos is not None:
            return float(pos[0])
        return None
    try:
        return float(traci.vehicle.getPosition(veh_id)[0])
    except Exception:
        return None


def _veh_lane(veh_id: str) -> Optional[str]:
    """Vehicle lane ID."""
    snap = get_active_snapshot()
    if snap is not None:
        return snap.lane_id(veh_id)
    try:
        return traci.vehicle.getLaneID(veh_id)
    except Exception:
        return None


def _veh_lane_pos(veh_id: str) -> Optional[float]:
    """Vehicle longitudinal position on its current lane (m)."""
    snap = get_active_snapshot()
    if snap is not None:
        return snap.lane_pos(veh_id)
    try:
        return float(traci.vehicle.getLanePosition(veh_id))
    except Exception:
        return None


def _veh_speed(veh_id: str) -> Optional[float]:
    """Vehicle speed; None mirrors the direct getter raising."""
    snap = get_active_snapshot()
    if snap is not None:
        return snap.speed(veh_id)
    try:
        return traci.vehicle.getSpeed(veh_id)
    except Exception:
        return None


def _all_vehicle_ids() -> List[str]:
    """All vehicle IDs in the network."""
    snap = get_active_snapshot()
    if snap is not None:
        return list(snap.id_set)
    try:
        return list(traci.vehicle.getIDList())
    except Exception:
        return []


def turn_signal_intent(veh_id: str) -> float:
    """Encode blinker state from traci.vehicle.getSignals as {-1, 0, +1}."""
    snap = get_active_snapshot()
    if snap is not None:
        raw = snap.signals(veh_id)
        if raw is None:
            return 0.0
        signals = int(raw)
    else:
        try:
            signals = int(traci.vehicle.getSignals(veh_id))
        except Exception:
            return 0.0
    left = bool(signals & VEH_SIGNAL_BLINKER_LEFT)
    right = bool(signals & VEH_SIGNAL_BLINKER_RIGHT)
    if left and not right:
        return -1.0
    if right and not left:
        return 1.0
    return 0.0


def _approx_distance(cav_id: str, other_id: str) -> Optional[float]:
    """Longitudinal / Euclidean hybrid distance between two vehicles."""
    snap = get_active_snapshot()
    if snap is not None:
        c = snap.position(cav_id)
        o = snap.position(other_id)
        if c is None or o is None:
            return None
        (cx, cy), (ox, oy) = c, o
    else:
        try:
            cx, cy = traci.vehicle.getPosition(cav_id)
            ox, oy = traci.vehicle.getPosition(other_id)
        except Exception:
            return None
    return float(((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5)


def collect_neighbors(
    cav_id: str,
    radius: float = 100.0,
    max_neighbors: int = 20,
) -> List[NeighborFeatures]:
    """Return up to max_neighbors nearest vehicles within radius."""
    snap = get_active_snapshot()
    if snap is not None:
        if not snap.has(cav_id):
            return []
        ego_speed = snap.speed(cav_id)
        ego_lane = snap.lane_index(cav_id)
        if ego_speed is None or ego_lane is None:
            return []
        ego_road = snap.road(cav_id)
        ego_pos = snap.lane_pos(cav_id)

        candidates_snap: List[NeighborFeatures] = []
        for other_id in snap.ordered_ids():
            if other_id == cav_id:
                continue
            dist = _approx_distance(cav_id, other_id)
            if dist is None or dist > radius:
                continue
            other_speed = snap.speed(other_id)
            other_lane = snap.lane_index(other_id)
            if other_speed is None or other_lane is None:
                continue

            # Signed longitudinal distance along same edge if possible.
            rel_distance = dist
            other_road = snap.road(other_id)
            other_pos = snap.lane_pos(other_id)
            if (
                ego_road is not None
                and ego_road == other_road
                and other_pos is not None
                and ego_pos is not None
            ):
                rel_distance = other_pos - ego_pos

            candidates_snap.append(
                NeighborFeatures(
                    vehicle_id=other_id,
                    rel_distance=float(rel_distance),
                    rel_speed=float(other_speed - ego_speed),
                    rel_lane=float(other_lane - ego_lane),
                    turn_signal=turn_signal_intent(other_id),
                    abs_distance=dist,
                )
            )

        candidates_snap.sort(key=lambda n: n.abs_distance)
        return candidates_snap[:max_neighbors]

    if cav_id not in traci.vehicle.getIDList():
        return []

    try:
        ego_speed = traci.vehicle.getSpeed(cav_id)
        ego_lane = traci.vehicle.getLaneIndex(cav_id)
    except Exception:
        return []

    candidates: List[NeighborFeatures] = []
    for other_id in traci.vehicle.getIDList():
        if other_id == cav_id:
            continue
        dist = _approx_distance(cav_id, other_id)
        if dist is None or dist > radius:
            continue
        try:
            other_speed = traci.vehicle.getSpeed(other_id)
            other_lane = traci.vehicle.getLaneIndex(other_id)
        except Exception:
            continue

        # Signed longitudinal distance along same edge if possible.
        rel_distance = dist
        try:
            if traci.vehicle.getRoadID(cav_id) == traci.vehicle.getRoadID(other_id):
                rel_distance = traci.vehicle.getLanePosition(
                    other_id
                ) - traci.vehicle.getLanePosition(cav_id)
        except Exception:
            pass

        candidates.append(
            NeighborFeatures(
                vehicle_id=other_id,
                rel_distance=float(rel_distance),
                rel_speed=float(other_speed - ego_speed),
                rel_lane=float(other_lane - ego_lane),
                turn_signal=turn_signal_intent(other_id),
                abs_distance=dist,
            )
        )

    candidates.sort(key=lambda n: n.abs_distance)
    return candidates[:max_neighbors]


# ---------------------------------------------------------------------------
# Trust tracking (Trust-MARL style)
# ---------------------------------------------------------------------------

TrustKey = Tuple[str, str]


class TrustTracker:
    """Maintain per-(ego, neighbor) trust in [0, 1] with EMA updates."""

    def __init__(self, lambda_ema: float = 0.2, cav_ids: Optional[Set[str]] = None):
        self.lambda_ema = float(lambda_ema)
        self._trust: Dict[TrustKey, float] = {}
        self._cav_ids: Set[str] = set(cav_ids or [])

    def reset(self) -> None:
        self._trust.clear()

    def set_cav_ids(self, cav_ids: Iterable[str]) -> None:
        self._cav_ids = set(cav_ids)

    def get(self, ego_id: str, other_id: str) -> float:
        return self._trust.get((ego_id, other_id), 0.5)

    def type_prior(self, other_id: str, vtype: Optional[str] = None) -> float:
        """Prior trust by vehicle class / HDV type."""
        if other_id in self._cav_ids:
            return 0.8
        if vtype is None:
            return 0.5
        if "cautious" in vtype:
            return 0.7
        if "aggressive" in vtype:
            return 0.3
        if "normal" in vtype:
            return 0.5
        return 0.5

    def instantaneous_score(
        self,
        neighbor: NeighborFeatures,
        other_is_cav: bool = False,
        vtype: Optional[str] = None,
    ) -> float:
        """Score in [0, 1] from gap/TTC-ish safety, speed smoothness, and type prior."""
        # Gap safety: prefer larger absolute longitudinal gaps.
        gap = abs(neighbor.rel_distance)
        gap_score = float(np.clip(gap / 50.0, 0.0, 1.0))

        # Closing speed: if ahead and we are faster, trust drops.
        closing = 0.0
        if neighbor.rel_distance > 0:
            # neighbor ahead: positive rel_speed means they pull away (safer)
            closing = max(0.0, -neighbor.rel_speed)
        else:
            # neighbor behind: positive rel_speed means they approach from behind
            closing = max(0.0, neighbor.rel_speed)
        close_score = float(np.clip(1.0 - closing / 15.0, 0.0, 1.0))

        prior = 0.8 if other_is_cav else self.type_prior(neighbor.vehicle_id, vtype)
        score = 0.4 * gap_score + 0.4 * close_score + 0.2 * prior
        return float(np.clip(score, 0.0, 1.0))

    def update_pair(
        self,
        ego_id: str,
        neighbor: NeighborFeatures,
        other_is_cav: bool = False,
        vtype: Optional[str] = None,
    ) -> float:
        """EMA-update trust for one pair and return the new value."""
        key = (ego_id, neighbor.vehicle_id)
        s = self.instantaneous_score(neighbor, other_is_cav=other_is_cav, vtype=vtype)
        if key not in self._trust:
            self._trust[key] = (
                self.type_prior(neighbor.vehicle_id, vtype) if not other_is_cav else 0.8
            )
        old = self._trust[key]
        new = (1.0 - self.lambda_ema) * old + self.lambda_ema * s
        self._trust[key] = float(np.clip(new, 0.0, 1.0))
        return self._trust[key]

    def update_and_weights(
        self,
        ego_id: str,
        neighbors: Iterable[NeighborFeatures],
        cav_ids: Optional[Set[str]] = None,
        vtype_lookup: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Update trust for all neighbors and return normalized weights summing to 1."""
        if cav_ids is not None:
            self.set_cav_ids(cav_ids)
        vtype_lookup = vtype_lookup or {}
        neighbors = list(neighbors)
        if not neighbors:
            return {}

        raw: Dict[str, float] = {}
        for nbr in neighbors:
            oid = nbr.vehicle_id
            t = self.update_pair(
                ego_id,
                nbr,
                other_is_cav=oid in self._cav_ids,
                vtype=vtype_lookup.get(oid),
            )
            raw[oid] = t

        total = sum(raw.values()) + 1e-8
        return {k: v / total for k, v in raw.items()}

    def prune(self, active_ids: Set[str]) -> None:
        """Drop trust entries for vehicles that have left the network."""
        stale = [k for k in self._trust if k[0] not in active_ids or k[1] not in active_ids]
        for k in stale:
            del self._trust[k]


# ---------------------------------------------------------------------------
# Zone features (Z-Merge style) for the highway merge map
# ---------------------------------------------------------------------------

ZONE_FEATURE_DIM = 12  # 8 classic + 4 delay / vs-no_op slots

# Versioned RSU observation contracts. The legacy contract deliberately keeps
# honoring ``zone_comm_range`` so old configurations and checkpoints retain
# their exact semantics. The diagnostic contract is an explicitly named,
# fixed-range replay contract; it must never be enabled by changing the legacy
# default range silently.
RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1 = "classic_rsu_configured_range_v1"
RSU_CONTRACT_LIVE_2000_DIAGNOSTIC_V1 = "classic_rsu_live_2000m_diagnostic_v1"
RSU_DIAGNOSTIC_RANGE_M = 2000.0
RSU_OBSERVATION_CONTRACTS = (
    RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
    RSU_CONTRACT_LIVE_2000_DIAGNOSTIC_V1,
)


def resolve_rsu_communication_range(
    contract: str, configured_range_m: float
) -> float:
    """Resolve and validate a versioned RSU communication contract."""
    name = str(contract)
    configured = float(configured_range_m)
    if not np.isfinite(configured) or configured < 0.0:
        raise ValueError("zone_comm_range must be finite and nonnegative")
    if name == RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1:
        return configured
    if name == RSU_CONTRACT_LIVE_2000_DIAGNOSTIC_V1:
        if abs(configured - RSU_DIAGNOSTIC_RANGE_M) > 1e-9:
            raise ValueError(
                f"{name} requires zone_comm_range={RSU_DIAGNOSTIC_RANGE_M:g} m"
            )
        return RSU_DIAGNOSTIC_RANGE_M
    raise ValueError(f"unknown RSU observation contract: {name!r}")


def delay_zone_slots(
    corridor_delay_rate: Optional[float] = None,
    delay_rate_ref: Optional[float] = None,
    delay_budget_scale: float = 0.20,
    stream_delay_gap: Optional[float] = None,
    stream_delay_gap_ref: float = 0.0,
    stream_fairness_gap_scale: float = 0.40,
    ema_main_delay: Optional[float] = None,
    ema_ramp_delay: Optional[float] = None,
    zone_delay_abs_scale: float = 0.50,
) -> np.ndarray:
    """Four appended zone slots: vs-no_op excesses + live main/ramp delay rates.

    Layout:
      [0] signed corridor delay excess vs paired no_op
          clip((rate - ref) / delay_budget_scale, -1, 1); 0 if unknown
      [1] fair-gap excess vs paired no_op
          clip((|G| - G_ref)_+ / stream_fairness_gap_scale, 0, 1)
      [2] live ema_main delay rate clipped to [0, 1]
      [3] live ema_ramp delay rate clipped to [0, 1]
    """
    scale_b = max(float(delay_budget_scale), 1e-6)
    if corridor_delay_rate is None or delay_rate_ref is None:
        excess_d = 0.0
    else:
        excess_d = float(
            np.clip(
                (float(corridor_delay_rate) - float(delay_rate_ref)) / scale_b,
                -1.0,
                1.0,
            )
        )

    scale_g = max(float(stream_fairness_gap_scale), 1e-6)
    if stream_delay_gap is None:
        excess_g = 0.0
    else:
        gap_excess = max(
            abs(float(stream_delay_gap)) - max(float(stream_delay_gap_ref), 0.0),
            0.0,
        )
        excess_g = float(min(gap_excess / scale_g, 1.0))

    scale_abs = max(float(zone_delay_abs_scale), 1e-6)
    main_n = (
        float(np.clip(float(ema_main_delay) / scale_abs, 0.0, 1.0))
        if ema_main_delay is not None
        else 0.0
    )
    ramp_n = (
        float(np.clip(float(ema_ramp_delay) / scale_abs, 0.0, 1.0))
        if ema_ramp_delay is not None
        else 0.0
    )
    return np.array([excess_d, excess_g, main_n, ramp_n], dtype=np.float32)


def _edge_occupancy(edge_id: str) -> float:
    """Return mean lane occupancy on an edge in [0, 1].

    Memoized per step when a snapshot is active (the value cannot change
    within a step); the computation itself is unchanged.
    """
    snap = get_active_snapshot()
    if snap is not None:
        return snap.memo(("edge_occupancy", edge_id), lambda: _edge_occupancy_direct(edge_id))
    return _edge_occupancy_direct(edge_id)


def _edge_occupancy_direct(edge_id: str) -> float:
    try:
        n_lanes = traci.edge.getLaneNumber(edge_id)
    except Exception:
        return 0.0
    if n_lanes <= 0:
        return 0.0
    occ = []
    for i in range(n_lanes):
        lane_id = f"{edge_id}_{i}"
        try:
            occ.append(traci.lane.getLastStepOccupancy(lane_id))
        except Exception:
            continue
    if not occ:
        return 0.0
    return float(np.mean(occ))


def _edge_mean_speed(edge_id: str, fallback_speed: float) -> float:
    """Average speed of vehicles on edge; fallback if empty."""
    snap = get_active_snapshot()
    if snap is not None:
        return snap.memo(
            ("edge_mean_speed", edge_id),
            lambda: _edge_mean_speed_direct(edge_id, fallback_speed),
        )
    return _edge_mean_speed_direct(edge_id, fallback_speed)


def _edge_mean_speed_direct(edge_id: str, fallback_speed: float) -> float:
    try:
        veh_ids = traci.edge.getLastStepVehicleIDs(edge_id)
    except Exception:
        return fallback_speed
    if not veh_ids:
        return fallback_speed
    speeds = []
    for vid in veh_ids:
        speed = _veh_speed(vid)
        if speed is None:
            continue
        speeds.append(speed)
    if not speeds:
        return fallback_speed
    return float(np.mean(speeds))


def _ramp_slow_count(edge_id: str, slow_threshold: float) -> float:
    """Count vehicles on the ramp with speed below threshold."""
    snap = get_active_snapshot()
    if snap is not None:
        return snap.memo(
            ("ramp_slow_count", edge_id),
            lambda: _ramp_slow_count_direct(edge_id, slow_threshold),
        )
    return _ramp_slow_count_direct(edge_id, slow_threshold)


def _ramp_slow_count_direct(edge_id: str, slow_threshold: float) -> float:
    try:
        veh_ids = traci.edge.getLastStepVehicleIDs(edge_id)
    except Exception:
        return 0.0
    count = 0
    for vid in veh_ids:
        speed = _veh_speed(vid)
        if speed is None:
            continue
        if speed < slow_threshold:
            count += 1
    return float(count)


def _lane_slow_count(lane_id: str, slow_threshold: float) -> float:
    """Count vehicles on a specific lane with speed below threshold."""
    snap = get_active_snapshot()
    if snap is not None:
        return snap.memo(
            ("lane_slow_count", lane_id),
            lambda: _lane_slow_count_direct(lane_id, slow_threshold),
        )
    return _lane_slow_count_direct(lane_id, slow_threshold)


def _lane_slow_count_direct(lane_id: str, slow_threshold: float) -> float:
    try:
        veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
    except Exception:
        return 0.0
    count = 0
    for vid in veh_ids:
        speed = _veh_speed(vid)
        if speed is None:
            continue
        if speed < slow_threshold:
            count += 1
    return float(count)


def _lane_mean_speed(lane_id: str, fallback_speed: float) -> float:
    """Average speed of vehicles on a specific lane."""
    snap = get_active_snapshot()
    if snap is not None:
        return snap.memo(
            ("lane_mean_speed", lane_id),
            lambda: _lane_mean_speed_direct(lane_id, fallback_speed),
        )
    return _lane_mean_speed_direct(lane_id, fallback_speed)


def _lane_mean_speed_direct(lane_id: str, fallback_speed: float) -> float:
    try:
        veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
    except Exception:
        return fallback_speed
    if not veh_ids:
        return fallback_speed
    speeds = []
    for vid in veh_ids:
        speed = _veh_speed(vid)
        if speed is None:
            continue
        speeds.append(speed)
    if not speeds:
        return fallback_speed
    return float(np.mean(speeds))


def _lane_occupancy(lane_id: str) -> float:
    """Occupancy (fraction) of a specific lane."""
    snap = get_active_snapshot()
    if snap is not None:
        return snap.memo(
            ("lane_occupancy", lane_id),
            lambda: _lane_occupancy_direct(lane_id),
        )
    return _lane_occupancy_direct(lane_id)


def _lane_occupancy_direct(lane_id: str) -> float:
    try:
        return float(traci.lane.getLastStepOccupancy(lane_id))
    except Exception:
        return 0.0


def _lane_length(lane_id: str, fallback: float) -> float:
    """Lane length; static per episode, so memoized across steps."""
    snap = get_active_snapshot()
    if snap is not None:
        return snap.memo(
            ("lane_length", lane_id),
            lambda: _lane_length_direct(lane_id, fallback),
            static=True,
        )
    return _lane_length_direct(lane_id, fallback)


def _lane_length_direct(lane_id: str, fallback: float) -> float:
    try:
        return traci.lane.getLength(lane_id)
    except Exception:
        return fallback


def distance_to_merge(
    cav_id: str, merge_edge: str = MERGE_EDGE, pre_edge: str = PRE_MERGE_EDGE
) -> float:
    """Approximate longitudinal distance (m) from ego to the merge junction.

    Positive means still upstream of the merge (before / approaching).
    Zero or negative once on the merge edge or beyond.
    """
    snap = get_active_snapshot()
    if snap is not None:
        edge = snap.road(cav_id)
        lane_pos = snap.lane_pos(cav_id)
        if edge is None or lane_pos is None:
            return float("inf")
    else:
        try:
            edge = traci.vehicle.getRoadID(cav_id)
            lane_pos = traci.vehicle.getLanePosition(cav_id)
        except Exception:
            return float("inf")

    if edge == merge_edge or edge.startswith(":"):
        # Already at / past the physical join.
        if edge == merge_edge:
            return 0.0
        return 0.0

    if edge == pre_edge:
        lane_len = _lane_length(f"{edge}_0", PRE_MERGE_LANE_LENGTH)
        return max(0.0, lane_len - lane_pos)

    if edge == RAMP_EDGE:
        lane_len = _lane_length(f"{edge}_0", RAMP_LANE_LENGTH)
        return max(0.0, lane_len - lane_pos)

    # Downstream of merge: treat as inside range.
    return 0.0


def rsu_observation_metadata(
    cav_id: str,
    *,
    contract: str,
    communication_range_m: float,
    observation_timestamp_s: float,
    merge_edge: str = MERGE_EDGE,
    pre_edge: str = PRE_MERGE_EDGE,
) -> dict:
    """Describe causal RSU availability for one assembled observation.

    The current implementation samples SUMO state synchronously, so an
    available update has age zero. Keeping timestamp and age explicit prevents
    a future delayed-broadcast implementation from being mistaken for the
    present same-step contract.
    """
    resolved_range = resolve_rsu_communication_range(
        contract, communication_range_m
    )
    absolute_distance = float(distance_to_merge(cav_id, merge_edge, pre_edge))
    available = bool(
        np.isfinite(absolute_distance) and absolute_distance <= resolved_range
    )
    classic_state = "live" if available else "padded"
    return {
        "schema": "idm_rsu_observation_contract/1",
        "observation_schema_version": str(contract),
        "rsu_communication_range_m": resolved_range,
        "rsu_available": available,
        "candidate_absolute_distance_to_merge_m": absolute_distance,
        "observation_timestamp_s": float(observation_timestamp_s),
        "observation_age_s": 0.0 if available else None,
        "zone_blocks": {
            "pre_merge_count_speed": classic_state,
            "merge_count_speed": classic_state,
            "ramp_count_speed": classic_state,
            "ramp_queue_merge_outflow": classic_state,
            # These four slots are assembled before the classic-block cutoff.
            "shared_delay_service": "live",
        },
    }


# TTM normalization horizon (s) and minimum speed (m/s) used to avoid div-by-0.
TTM_HORIZON = 60.0
TTM_MIN_SPEED = 0.5


def time_to_merge(
    cav_id: str, speed: float, merge_edge: str = MERGE_EDGE, pre_edge: str = PRE_MERGE_EDGE
) -> float:
    """Normalized time (s, in [0, 1]) for ego to reach the merge junction.

    Z-Merge-style onboard feature: `distance_to_merge / max(speed, 0.5)`,
    clipped to [0, TTM_HORIZON] and divided by the horizon. Zero at or past
    the merge; 1.0 when far away (or the distance query fails, which yields
    an infinite distance and saturates the clip).
    """
    dist = distance_to_merge(cav_id, merge_edge, pre_edge)
    ttm = np.clip(dist / max(speed, TTM_MIN_SPEED), 0.0, TTM_HORIZON)
    return float(ttm / TTM_HORIZON)


def _leader_features(cav_id: str, ego_speed: float) -> Tuple[float, float]:
    """(leader_gap, leader_rel_speed) in normalized units.

    leader_gap: min(gap, 200)/200 — 1.0 when no leader within 200 m.
    leader_rel_speed: clip((v_leader - v_ego)/15, -1, 1) — 0.0 with no leader.
    """
    info = leader_info(cav_id)
    if info is None:
        return 1.0, 0.0
    _, gap, leader_speed = info
    gap_n = min(max(float(gap), 0.0), 200.0) / 200.0
    rel_n = float(np.clip((leader_speed - ego_speed) / 15.0, -1.0, 1.0))
    return gap_n, rel_n


def _queue_counts(
    cav_id: str,
    radius: float = 300.0,
    slow: float = 5.0,
    norm: float = 20.0,
) -> Tuple[float, float]:
    """(queue_ahead, queue_behind): slow same-edge vehicles within ``radius``.

    Counts vehicles with speed < ``slow`` m/s ahead of / behind the CAV on
    the same edge, each normalized by ``norm`` and clipped to 1.0. These are
    the congestion detectors the old obs lacked: queue_ahead sees the queue
    tail before reaching it, queue_behind sees the delay the CAV's own
    slowing is causing.
    """
    snap = get_active_snapshot()
    ahead = 0
    behind = 0
    if snap is not None:
        ego_edge = snap.road(cav_id)
        ego_pos = snap.lane_pos(cav_id)
        if ego_edge is None or ego_pos is None:
            return 0.0, 0.0
        for vid in snap.ordered_ids():
            if vid == cav_id:
                continue
            if snap.road(vid) != ego_edge:
                continue
            pos = snap.lane_pos(vid)
            speed = snap.speed(vid)
            if pos is None or speed is None or speed >= slow:
                continue
            rel = pos - ego_pos
            if 0.0 < rel <= radius:
                ahead += 1
            elif -radius <= rel < 0.0:
                behind += 1
    else:
        try:
            ego_edge = traci.vehicle.getRoadID(cav_id)
            ego_pos = traci.vehicle.getLanePosition(cav_id)
        except Exception:
            return 0.0, 0.0
        try:
            ids = traci.vehicle.getIDList()
        except Exception:
            ids = []
        for vid in ids:
            if vid == cav_id:
                continue
            try:
                if traci.vehicle.getRoadID(vid) != ego_edge:
                    continue
                pos = traci.vehicle.getLanePosition(vid)
                speed = traci.vehicle.getSpeed(vid)
            except Exception:
                continue
            if speed >= slow:
                continue
            rel = pos - ego_pos
            if 0.0 < rel <= radius:
                ahead += 1
            elif -radius <= rel < 0.0:
                behind += 1
    return (
        min(ahead / norm, 1.0),
        min(behind / norm, 1.0),
    )


def _list_mean_speed(vehicle_ids: list, fallback: float) -> float:
    """Mean speed of a list of vehicles; ``fallback`` if empty or all failed."""
    if not vehicle_ids:
        return fallback
    speeds = []
    for vid in vehicle_ids:
        s = _veh_speed(vid)
        if s is not None:
            speeds.append(s)
    if not speeds:
        return fallback
    return float(np.mean(speeds))


def _zone_vehicles_by_distance(
    pre_edge: str,
    merge_edge: str,
    ramp_edge: str,
    ramp_lane: Optional[str] = None,
    pre_range: tuple = (50.0, 400.0),
    merge_range: tuple = (-75.0, 75.0),
    ramp_range: tuple = (0.0, 150.0),
):
    """Return vehicle-ID lists for pre/merge/ramp zones by distance from junction.

    Uses snapshot data when available (no TraCI calls); falls back to
    direct TraCI getters otherwise.
    """
    # ---- edge lengths (memoized) ----
    pre_len = _lane_length(f"{pre_edge}_0", 1000.0)
    if ramp_lane is not None:
        ramp_len = _lane_length(ramp_lane, 1000.0)
    else:
        ramp_len = _lane_length(f"{ramp_edge}_0", 1000.0)

    pre_vehs: List[str] = []
    merge_vehs: List[str] = []
    ramp_vehs: List[str] = []

    snap = get_active_snapshot()
    if snap is not None:
        # Fast path: iterate cached snapshot data — zero TraCI calls.
        for vid in snap.ordered_ids():
            rec = snap.data.get(vid)
            if rec is None:
                continue
            road = rec.get(80)  # VAR_ROAD_ID
            pos = rec.get(86)  # VAR_LANEPOSITION
            if road is None or pos is None:
                continue
            pos = float(pos)
            if road == pre_edge:
                dist = pre_len - pos
                pl, ph = pre_range
                if pl < dist <= ph:
                    pre_vehs.append(vid)
                elif merge_range[0] <= dist <= merge_range[1]:
                    merge_vehs.append(vid)
            elif road == merge_edge:
                dist = -pos
                if merge_range[0] <= dist <= merge_range[1]:
                    merge_vehs.append(vid)
            elif road == ramp_edge or (ramp_lane is not None and rec.get(81) == ramp_lane):
                # rec.get(81) is VAR_LANE_ID; ramp_lane filter handles lane-drop scenarios
                if ramp_lane is not None and rec.get(81) != ramp_lane:
                    continue
                dist = ramp_len - pos
                r_low, r_high = ramp_range
                if r_low <= dist <= r_high:
                    ramp_vehs.append(vid)
        return pre_vehs, merge_vehs, ramp_vehs

    # Fallback: direct TraCI getters (no snapshot).
    # --- pre-edge vehicles ---
    try:
        for vid in traci.edge.getLastStepVehicleIDs(pre_edge):
            pos = _veh_lane_pos(vid)
            if pos is None:
                continue
            dist = pre_len - pos
            pre_low, pre_high = pre_range
            if pre_low < dist <= pre_high:
                pre_vehs.append(vid)
            elif merge_range[0] <= dist <= merge_range[1]:
                merge_vehs.append(vid)
    except Exception:
        pass

    # --- merge-edge vehicles ---
    try:
        for vid in traci.edge.getLastStepVehicleIDs(merge_edge):
            pos = _veh_lane_pos(vid)
            if pos is None:
                continue
            dist = -pos  # past junction → negative
            if merge_range[0] <= dist <= merge_range[1]:
                merge_vehs.append(vid)
    except Exception:
        pass

    # --- ramp / dropping-lane vehicles ---
    try:
        if ramp_lane is not None:
            vids = traci.lane.getLastStepVehicleIDs(ramp_lane)
        else:
            vids = traci.edge.getLastStepVehicleIDs(ramp_edge)
        r_low, r_high = ramp_range
        for vid in vids:
            pos = _veh_lane_pos(vid)
            if pos is None:
                continue
            dist = ramp_len - pos
            if r_low <= dist <= r_high:
                ramp_vehs.append(vid)
    except Exception:
        pass

    return pre_vehs, merge_vehs, ramp_vehs


def build_zone_features(
    cav_id: str,
    highway_speed_limit: float,
    zone_comm_range: float = 500.0,
    slow_speed_threshold: float = 5.0,
    pre_edge: str = PRE_MERGE_EDGE,
    merge_edge: str = MERGE_EDGE,
    ramp_edge: str = RAMP_EDGE,
    ramp_lane: Optional[str] = None,
    zone_config: Optional[ZoneConfig] = None,
    merge_outflow: float = 1.0,
    corridor_delay_rate: Optional[float] = None,
    delay_rate_ref: Optional[float] = None,
    delay_budget_scale: float = 0.20,
    stream_delay_gap: Optional[float] = None,
    stream_delay_gap_ref: float = 0.0,
    stream_fairness_gap_scale: float = 0.40,
    ema_main_delay: Optional[float] = None,
    ema_ramp_delay: Optional[float] = None,
    zone_delay_abs_scale: float = 0.50,
) -> np.ndarray:
    """Return 12-d zone feature vector with hard RSU-range cut-off.

    Layout: [pre_occ, pre_speed, merge_occ, merge_speed, ramp_occ, ramp_speed,
             ramp_queue, merge_outflow,
             corridor_delay_excess, fair_gap_excess, main_delay_rate,
             ramp_delay_rate]
    Zone mean speeds are normalized by ``highway_speed_limit``; ``ramp_queue``
    is a raw count; ``merge_outflow`` is the shared merge-edge discharge
    (q_norm, ~1.0 healthy, ->0 in a jam). Delay slots use the same scales as
    the shepherd reward (paired no_op refs).

    Inside ``zone_comm_range`` (meters to merge): live zone statistics.
    Outside range: free-flow pad on classic slots; delay slots still carry
    corridor/vs-no_op signals (RSU-style shared state).

    When ``ramp_lane`` is set, ramp occupancy/speed/queue are computed on that
    specific lane instead of the whole ``ramp_edge`` (useful for lane-drop
    scenarios where the dropped lane is part of the pre-edge).
    """
    delay = delay_zone_slots(
        corridor_delay_rate=corridor_delay_rate,
        delay_rate_ref=delay_rate_ref,
        delay_budget_scale=delay_budget_scale,
        stream_delay_gap=stream_delay_gap,
        stream_delay_gap_ref=stream_delay_gap_ref,
        stream_fairness_gap_scale=stream_fairness_gap_scale,
        ema_main_delay=ema_main_delay,
        ema_ramp_delay=ema_ramp_delay,
        zone_delay_abs_scale=zone_delay_abs_scale,
    )
    vmax = max(float(highway_speed_limit), 1e-6)
    dist = distance_to_merge(cav_id, merge_edge, pre_edge)
    if dist > zone_comm_range:
        return np.concatenate([padded_zone_features_classic(highway_speed_limit), delay]).astype(
            np.float32
        )

    if zone_config is not None:
        # Physical coordinate-based zones (scenario-agnostic)
        all_vehicles = _all_vehicle_ids()
        pre_occ = _zone_occupancy(all_vehicles, (zone_config.pre_start, zone_config.pre_end))
        pre_spd = _zone_mean_speed(
            all_vehicles, (zone_config.pre_start, zone_config.pre_end), fallback=highway_speed_limit
        )
        merge_occ = _zone_occupancy(all_vehicles, (zone_config.merge_start, zone_config.merge_end))
        merge_spd = _zone_mean_speed(
            all_vehicles,
            (zone_config.merge_start, zone_config.merge_end),
            fallback=highway_speed_limit,
        )
        ramp_occ = _zone_occupancy(
            all_vehicles,
            (zone_config.merge_start, zone_config.merge_end),
            lane=zone_config.ramp_lane,
        )
        ramp_spd = _zone_mean_speed(
            all_vehicles,
            (zone_config.merge_start, zone_config.merge_end),
            lane=zone_config.ramp_lane,
            fallback=highway_speed_limit,
        )
        ramp_queue = _zone_occupancy(
            all_vehicles,
            (zone_config.merge_start, zone_config.merge_end),
            lane=zone_config.ramp_lane,
        )
    else:
        # Distance-based zones: fixed distance ranges from merge/bottleneck junction.
        # Memoized per step — zones don't change within a single step, and they
        # are identical for all CAVs (referenced to the junction, not the ego).
        memo_key = f"zone_vehs_{pre_edge}_{merge_edge}_{ramp_edge}_{ramp_lane}"
        snap = get_active_snapshot()
        if snap is not None:
            pre_v, merge_v, ramp_v = snap.memo(
                (memo_key,),
                lambda: _zone_vehicles_by_distance(
                    pre_edge,
                    merge_edge,
                    ramp_edge,
                    ramp_lane=ramp_lane,
                ),
            )
        else:
            pre_v, merge_v, ramp_v = _zone_vehicles_by_distance(
                pre_edge,
                merge_edge,
                ramp_edge,
                ramp_lane=ramp_lane,
            )
        pre_occ = float(len(pre_v)) / 350.0  # veh/m in [50, 400]
        pre_spd = _list_mean_speed(pre_v, highway_speed_limit)
        merge_occ = float(len(merge_v)) / 150.0  # veh/m in [-75, 75]
        merge_spd = _list_mean_speed(merge_v, highway_speed_limit)
        ramp_occ = float(len(ramp_v)) / 150.0  # veh/m in [0, 150]
        ramp_spd = _list_mean_speed(ramp_v, highway_speed_limit)
        ramp_queue = float(
            sum(1 for vid in ramp_v if (_veh_speed(vid) or 99.0) < slow_speed_threshold)
        )

    classic = np.array(
        [
            pre_occ,
            pre_spd / vmax,
            merge_occ,
            merge_spd / vmax,
            ramp_occ,
            ramp_spd / vmax,
            ramp_queue,
            merge_outflow,
        ],
        dtype=np.float32,
    )
    return np.concatenate([classic, delay]).astype(np.float32)


def padded_zone_features_classic(highway_speed_limit: float) -> np.ndarray:
    """Fixed free-flow pad for the classic 8 zone slots (outside RSU range)."""
    del highway_speed_limit
    return np.array(
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        dtype=np.float32,
    )


def padded_zone_features(highway_speed_limit: float) -> np.ndarray:
    """Fixed free-flow pad used when the CAV is outside RSU / zone_comm_range.

    Classic 8 slots mean "no V2X update"; delay slots zero (unknown / neutral).
    """
    return np.concatenate(
        [
            padded_zone_features_classic(highway_speed_limit),
            np.zeros(4, dtype=np.float32),
        ]
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Observation assembly
# ---------------------------------------------------------------------------

EGO_DIM = 9
MAX_NEIGHBORS_DEFAULT = 20
# per neighbor: rel_dist, rel_speed, rel_lane, turn_signal, trust_weight
NEIGHBOR_SLOT = 5
# V2V coordination feature (Option B, 2026-08-06): how many CAVs are
# currently metering — lets the policy learn to coast when load is high
# (penetration-invariant coordination).
MET_LOAD_DIM = 1
WAVE_SIGNAL_DIM = 2  # [wave_risk, wave_forming] — RSU compound shockwave detector

# Optional physical context is appended after the fixed ego/zone/neighbor
# blocks. Action-mask bits are deliberately excluded from this schema.
CONTEXT_FEATURE_ORDER = (
    "meter_load",
    "cav_count",
    "wave_risk",
    "wave_forming",
    "metastable_health",
    "injection_dv",
    "runway_margin",
    "idm_absorber_gate",
    "idm_nominal_acceleration",
    "idm_filtered_reference",
    "idm_reference_error",
    "idm_projection_delta",
    "idm_severe_level",
    "idm_wave_energy_level",
    "idm_severe_short_ema",
    "idm_wave_short_ema",
    "idm_severe_trend",
    "idm_wave_energy_trend",
)


def context_feature_names(
    *,
    include_meter_load: bool = False,
    include_cav_count: bool = False,
    include_wave_signal: bool = False,
    include_health: bool = False,
    include_dv: bool = False,
    include_margin: bool = False,
    include_idm_absorber_gate: bool = False,
    include_idm_controller_context: bool = False,
) -> tuple[str, ...]:
    """Return enabled optional physical fields in packed-vector order."""
    names: list[str] = []
    if include_meter_load:
        names.append("meter_load")
    if include_cav_count:
        names.append("cav_count")
    if include_wave_signal:
        names.extend(("wave_risk", "wave_forming"))
    if include_health:
        names.append("metastable_health")
    if include_dv:
        names.append("injection_dv")
    if include_margin:
        names.append("runway_margin")
    if include_idm_absorber_gate:
        names.append("idm_absorber_gate")
    if include_idm_controller_context:
        names.extend(
            (
                "idm_nominal_acceleration",
                "idm_filtered_reference",
                "idm_reference_error",
                "idm_projection_delta",
                "idm_severe_level",
                "idm_wave_energy_level",
                "idm_severe_short_ema",
                "idm_wave_short_ema",
                "idm_severe_trend",
                "idm_wave_energy_trend",
            )
        )
    result = tuple(names)
    expected = tuple(name for name in CONTEXT_FEATURE_ORDER if name in result)
    if result != expected:
        raise AssertionError(f"context feature order drifted: {result} != {expected}")
    return result


def wave_risk_signal(
    cav_id: str,
    look_ahead: float = 500.0,
    back: float = 150.0,
    slow_thresh: float = 6.0,
    merge_thresh: float = 10.0,
    merge_edge: str = MERGE_EDGE,
) -> tuple:
    """RSU compound shockwave detector (validated on the chaotic seeds).

    risk = clamp((slow_thresh - down_band)/6,0,1) * clamp((merge_thresh - merge_spd)/8,0,1)
    -> high only when the downstream band is NEARLY STOPPED AND the merge is
    congested (the real-onset signature); low on false-positive states.
    Returns (wave_risk, wave_forming).
    """
    snap = get_active_snapshot()
    if snap is None or not snap.has(cav_id):
        return (0.0, 0.0)
    lane = snap.lane_id(cav_id)
    pos = snap.lane_pos(cav_id)
    down = []
    up = []
    if lane is not None and pos is not None:
        for vid in snap.ordered_ids():
            if vid == cav_id or snap.lane_id(vid) != lane:
                continue
            p = snap.lane_pos(vid)
            sp = snap.speed(vid)
            if p is None or sp is None:
                continue
            d = p - pos
            if 0.0 < d <= look_ahead:
                down.append(float(sp))
            elif -back <= d < 0.0:
                up.append(float(sp))
    down_band = sum(down) / len(down) if down else None
    merge_spd = None
    try:
        spds = []
        for vid in snap.ordered_ids():
            if snap.road(vid) == merge_edge:
                s = snap.speed(vid)
                if s is not None:
                    spds.append(float(s))
        merge_spd = sum(spds) / len(spds) if spds else None
    except Exception:
        merge_spd = None
    term_d = 0.0 if down_band is None else min(1.0, max(0.0, (slow_thresh - down_band) / 6.0))
    term_m = 0.0 if merge_spd is None else min(1.0, max(0.0, (merge_thresh - merge_spd) / 8.0))
    risk = float(term_d * term_m)
    forming = 1.0 if risk > 0.4 else 0.0
    return (risk, forming)


def raw_obs_dim(
    max_neighbors: int = MAX_NEIGHBORS_DEFAULT,
    include_meter_load: bool = False,
    include_cav_count: bool = False,
    include_wave_signal: bool = False,
    include_health: bool = False,
    include_dv: bool = False,
    include_margin: bool = False,
    include_idm_absorber_gate: bool = False,
    include_idm_controller_context: bool = False,
) -> int:
    extra = len(
        context_feature_names(
            include_meter_load=include_meter_load,
            include_cav_count=include_cav_count,
            include_wave_signal=include_wave_signal,
            include_health=include_health,
            include_dv=include_dv,
            include_margin=include_margin,
            include_idm_absorber_gate=include_idm_absorber_gate,
            include_idm_controller_context=include_idm_controller_context,
        )
    )
    return EGO_DIM + ZONE_FEATURE_DIM + max_neighbors * NEIGHBOR_SLOT + extra


RAW_OBS_DIM = raw_obs_dim()


class ObservationAssembler:
    """Build fixed-size raw observation vectors using trust and zone features."""

    def __init__(
        self,
        trust_tracker: TrustTracker,
        neighbor_radius: float = 100.0,
        max_neighbors: int = 20,
        zone_comm_range: float = 500.0,
        rsu_observation_contract: str = RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
        max_speed: float = 35.0,
        slow_speed_threshold: float = 5.0,
        include_meter_load: bool = False,
        include_cav_count: bool = False,
        include_wave_signal: bool = False,
        include_health: bool = False,
        include_dv: bool = False,
        include_margin: bool = False,
        include_idm_absorber_gate: bool = False,
        include_idm_controller_context: bool = False,
    ):
        self.trust = trust_tracker
        self.neighbor_radius = neighbor_radius
        self.max_neighbors = max_neighbors
        self.rsu_observation_contract = str(rsu_observation_contract)
        self.zone_comm_range = resolve_rsu_communication_range(
            self.rsu_observation_contract, zone_comm_range
        )
        self.max_speed = max_speed
        self.slow_speed_threshold = slow_speed_threshold
        self.include_meter_load = include_meter_load
        self.include_cav_count = include_cav_count
        self.include_wave_signal = include_wave_signal
        self.include_health = include_health
        self.include_dv = include_dv
        self.include_margin = include_margin
        self.include_idm_absorber_gate = include_idm_absorber_gate
        self.include_idm_controller_context = include_idm_controller_context
        self.context_feature_names = context_feature_names(
            include_meter_load=include_meter_load,
            include_cav_count=include_cav_count,
            include_wave_signal=include_wave_signal,
            include_health=include_health,
            include_dv=include_dv,
            include_margin=include_margin,
            include_idm_absorber_gate=include_idm_absorber_gate,
            include_idm_controller_context=include_idm_controller_context,
        )
        self.obs_dim = raw_obs_dim(
            max_neighbors,
            include_meter_load,
            include_cav_count,
            include_wave_signal,
            include_health,
            include_dv,
            include_margin,
            include_idm_absorber_gate,
            include_idm_controller_context,
        )

    def reset(self) -> None:
        self.trust.reset()

    def rsu_metadata(
        self,
        cav_id: str,
        *,
        observation_timestamp_s: float,
        merge_edge: str = MERGE_EDGE,
        pre_edge: str = PRE_MERGE_EDGE,
    ) -> dict:
        return rsu_observation_metadata(
            cav_id,
            contract=self.rsu_observation_contract,
            communication_range_m=self.zone_comm_range,
            observation_timestamp_s=observation_timestamp_s,
            merge_edge=merge_edge,
            pre_edge=pre_edge,
        )

    def build(
        self,
        cav_id: str,
        cav_ids: Optional[Set[str]] = None,
        vtype_lookup: Optional[Dict[str, str]] = None,
        density_id: float = 0.5,
        pre_edge: str = PRE_MERGE_EDGE,
        merge_edge: str = MERGE_EDGE,
        ramp_edge: str = RAMP_EDGE,
        ramp_lane: Optional[str] = None,
        zone_config: Optional[ZoneConfig] = None,
        merge_outflow: float = 1.0,
        corridor_delay_rate: Optional[float] = None,
        delay_rate_ref: Optional[float] = None,
        delay_budget_scale: float = 0.20,
        stream_delay_gap: Optional[float] = None,
        stream_delay_gap_ref: float = 0.0,
        stream_fairness_gap_scale: float = 0.40,
        ema_main_delay: Optional[float] = None,
        ema_ramp_delay: Optional[float] = None,
        metering_load: float = 0.0,
        metering_load_scale: float = 8.0,
        cav_count_norm: float = 0.0,
        metastable_health: Optional[float] = None,
        merge_dv: Optional[float] = None,
        runway_margin: Optional[float] = None,
        idm_absorber_gate: float = 0.0,
        idm_nominal_acceleration: float = 0.0,
        idm_filtered_reference: float = 0.0,
        idm_reference_error: float = 0.0,
        idm_projection_delta: float = 0.0,
        idm_severe_level: float = 0.0,
        idm_wave_energy_level: float = 0.0,
        idm_severe_short_ema: float = 0.0,
        idm_wave_short_ema: float = 0.0,
        idm_severe_trend: float = 0.0,
        idm_wave_energy_trend: float = 0.0,
        zone_delay_abs_scale: float = 0.50,
    ) -> np.ndarray:
        """Return a fixed-size float32 observation for one CAV."""
        cav_ids = cav_ids or set()
        vtype_lookup = vtype_lookup or {}

        snap = get_active_snapshot()
        if snap is not None:
            lane = snap.lane_index(cav_id)
            speed = snap.speed(cav_id)
            accel = snap.accel(cav_id)
            if lane is None or speed is None or accel is None:
                return np.zeros(self.obs_dim, dtype=np.float32)
            lane, speed, accel = float(lane), float(speed), float(accel)
        else:
            try:
                lane = float(traci.vehicle.getLaneIndex(cav_id))
                speed = float(traci.vehicle.getSpeed(cav_id))
                accel = float(traci.vehicle.getAcceleration(cav_id))
            except Exception:
                return np.zeros(self.obs_dim, dtype=np.float32)

        vmax = max(self.max_speed, 1e-6)
        if EGO_DIM <= 4:
            # Legacy layout (pre time_to_merge): lane, speed, accel, density_id.
            ego = np.array([lane, speed, accel, density_id], dtype=np.float32)
        elif EGO_DIM == 5:
            # v2 layout: lane, speed, accel, time_to_merge, density_id.
            ego = np.array(
                [
                    lane,
                    speed,
                    accel,
                    time_to_merge(cav_id, speed, merge_edge, pre_edge),
                    density_id,
                ],
                dtype=np.float32,
            )
        else:
            # v3 layout: speeds normalized; congestion detectors appended.
            leader_gap, leader_rel = _leader_features(cav_id, speed)
            queue_ahead, queue_behind = _queue_counts(cav_id)
            ego = np.array(
                [
                    lane,
                    speed / vmax,
                    accel,
                    time_to_merge(cav_id, speed, merge_edge, pre_edge),
                    density_id,
                    leader_gap,
                    leader_rel,
                    queue_ahead,
                    queue_behind,
                ],
                dtype=np.float32,
            )
        if ego.shape[0] != EGO_DIM:
            raise ValueError(f"ego length {ego.shape[0]} != EGO_DIM={EGO_DIM}")
        zone = build_zone_features(
            cav_id,
            highway_speed_limit=self.max_speed,
            zone_comm_range=self.zone_comm_range,
            slow_speed_threshold=self.slow_speed_threshold,
            pre_edge=pre_edge,
            merge_edge=merge_edge,
            ramp_edge=ramp_edge,
            ramp_lane=ramp_lane,
            zone_config=zone_config,
            merge_outflow=merge_outflow,
            corridor_delay_rate=corridor_delay_rate,
            delay_rate_ref=delay_rate_ref,
            delay_budget_scale=delay_budget_scale,
            stream_delay_gap=stream_delay_gap,
            stream_delay_gap_ref=stream_delay_gap_ref,
            stream_fairness_gap_scale=stream_fairness_gap_scale,
            ema_main_delay=ema_main_delay,
            ema_ramp_delay=ema_ramp_delay,
            zone_delay_abs_scale=zone_delay_abs_scale,
        )

        neighbors = collect_neighbors(cav_id, self.neighbor_radius, self.max_neighbors)
        weights = self.trust.update_and_weights(
            cav_id, neighbors, cav_ids=cav_ids, vtype_lookup=vtype_lookup
        )

        slots = np.zeros((self.max_neighbors, NEIGHBOR_SLOT), dtype=np.float32)
        for i, nbr in enumerate(neighbors):
            slots[i, 0] = nbr.rel_distance
            slots[i, 1] = nbr.rel_speed / vmax
            slots[i, 2] = nbr.rel_lane
            slots[i, 3] = nbr.turn_signal
            slots[i, 4] = weights.get(nbr.vehicle_id, 0.0)

        if self.include_meter_load:
            obs = np.concatenate(
                [
                    ego,
                    zone,
                    slots.reshape(-1),
                    np.array([metering_load / metering_load_scale], dtype=np.float32),
                ]
            )
        else:
            obs = np.concatenate([ego, zone, slots.reshape(-1)])
        if self.include_cav_count:
            obs = np.concatenate([obs, np.array([cav_count_norm], dtype=np.float32)])
        if self.include_wave_signal:
            risk, forming = wave_risk_signal(cav_id)
            obs = np.concatenate([obs, np.array([risk, forming], dtype=np.float32)])
        if self.include_health:
            # RSU merge-zone health h in [0,1] (shared corridor scalar, same
            # value for every CAV). None (merge empty / no snapshot) -> 0.5
            # neutral so the policy never mistakes "unknown" for "collapsed".
            h_val = 0.5 if metastable_health is None else float(metastable_health)
            obs = np.concatenate([obs, np.array([h_val], dtype=np.float32)])
        if self.include_dv:
            # Injection velocity delta |v_mean_merge - v_ramp_approach| in m/s
            # (retry 2026-08-13): the KWG breakdown trigger is the injection
            # speed difference; None -> 0.0 (no ramp vehicle near the
            # junction / no perturbation to see).
            dv_val = 0.0 if merge_dv is None else float(merge_dv)
            obs = np.concatenate([obs, np.array([dv_val], dtype=np.float32)])
        if self.include_margin:
            # Runway margin L_pre - L_min(rho) in m (feasibility, 2026-08-13):
            # positive = this highway has room to control at this density,
            # negative = futile (coast). None -> 0.0 neutral.
            m_val = 0.0 if runway_margin is None else float(runway_margin)
            obs = np.concatenate([obs, np.array([m_val], dtype=np.float32)])
        if self.include_idm_absorber_gate:
            gate = 1.0 if idm_absorber_gate > 0.5 else 0.0
            obs = np.concatenate([obs, np.array([gate], dtype=np.float32)])
        if self.include_idm_controller_context:
            context = np.array(
                [
                    float(np.clip(idm_nominal_acceleration, -1.0, 1.0)),
                    float(
                        np.clip(
                            idm_filtered_reference / max(self.max_speed, 1e-6),
                            0.0,
                            1.5,
                        )
                    ),
                    float(np.clip(idm_reference_error / 10.0, -1.0, 1.0)),
                    float(np.clip(idm_projection_delta, -1.0, 1.0)),
                    float(np.clip(idm_severe_level / 2.0, 0.0, 1.5)),
                    float(np.clip(idm_wave_energy_level / 2.0, 0.0, 1.5)),
                    float(np.clip(idm_severe_short_ema / 2.0, 0.0, 1.5)),
                    float(np.clip(idm_wave_short_ema / 2.0, 0.0, 1.5)),
                    float(np.clip(idm_severe_trend, -1.0, 1.0)),
                    float(np.clip(idm_wave_energy_trend, -1.0, 1.0)),
                ],
                dtype=np.float32,
            )
            obs = np.concatenate([obs, context])
        assert obs.shape[0] == self.obs_dim
        # TraCI can emit inf/nan (teleports, missing edges); keep the policy finite.
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
