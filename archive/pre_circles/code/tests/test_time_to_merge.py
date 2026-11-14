"""Unit tests for the time-to-merge (TTM) ego observation feature.

TTM is a Z-Merge-style onboard feature: time for the ego CAV to reach the
merge junction, `dist / max(speed, 0.5)` clipped to [0, 60] s and normalized
by 60, so the value lies in [0, 1].
"""

from unittest.mock import MagicMock, patch

import numpy as np

from config.constants import MERGE_EDGE, PRE_MERGE_EDGE
from env.features import (
    EGO_DIM,
    ObservationAssembler,
    TrustTracker,
    raw_obs_dim,
    time_to_merge,
)

# Edge just past the merge junction (see distance_to_merge docstring).
DOWNSTREAM_EDGE = "1487606178"


def _traci_on_edge(edge: str, lane_pos: float, lane_len: float) -> MagicMock:
    """Minimal traci mock placing one vehicle on `edge` at `lane_pos`."""
    traci = MagicMock()
    traci.vehicle.getRoadID.return_value = edge
    traci.vehicle.getLanePosition.return_value = lane_pos
    traci.lane.getLength.return_value = lane_len
    return traci


def test_ego_dim_includes_ttm():
    assert EGO_DIM == 9
    assert raw_obs_dim(20) == 9 + 12 + 20 * 5
    assert raw_obs_dim(20) == 121


def test_ttm_normal_speed():
    # 400 m lane, 100 m driven -> 300 m to go; 300 / 20 = 15 s -> 15 / 60.
    traci = _traci_on_edge(PRE_MERGE_EDGE, lane_pos=100.0, lane_len=400.0)
    with patch("env.features.traci", traci):
        ttm = time_to_merge("cav_0", speed=20.0)
    assert abs(ttm - 0.25) < 1e-6


def test_ttm_low_speed_clips_to_one():
    # v = 0 -> max(speed, 0.5); 300 / 0.5 = 600 s -> clip to 60 s -> 1.0.
    traci = _traci_on_edge(PRE_MERGE_EDGE, lane_pos=100.0, lane_len=400.0)
    with patch("env.features.traci", traci):
        ttm = time_to_merge("cav_0", speed=0.0)
    assert ttm == 1.0


def test_ttm_downstream_is_zero():
    # Beyond the merge junction distance_to_merge returns 0 -> ttm 0.
    traci = _traci_on_edge(DOWNSTREAM_EDGE, lane_pos=50.0, lane_len=500.0)
    with patch("env.features.traci", traci):
        assert time_to_merge("cav_0", speed=20.0) == 0.0
    # On the merge edge itself: also 0.
    traci = _traci_on_edge(MERGE_EDGE, lane_pos=10.0, lane_len=100.0)
    with patch("env.features.traci", traci):
        assert time_to_merge("cav_0", speed=20.0) == 0.0


def test_build_places_ttm_in_ego_block():
    assembler = ObservationAssembler(TrustTracker())
    traci = _traci_on_edge(PRE_MERGE_EDGE, lane_pos=100.0, lane_len=400.0)
    traci.vehicle.getLaneIndex.return_value = 1
    traci.vehicle.getSpeed.return_value = 20.0
    traci.vehicle.getAcceleration.return_value = 0.5
    traci.vehicle.getIDList.return_value = ["cav_0"]  # no neighbors
    traci.edge.getLaneNumber.return_value = 0  # zone occupancy -> 0
    traci.edge.getLastStepVehicleIDs.return_value = []
    with patch("env.features.traci", traci):
        obs = assembler.build("cav_0")
    assert obs.shape == (assembler.obs_dim,)
    assert obs[0] == 1.0  # lane
    assert abs(obs[1] - 20.0 / 35.0) < 1e-6  # speed, normalized by max_speed
    assert obs[2] == 0.5  # accel
    assert abs(obs[3] - 0.25) < 1e-6  # ttm
    assert obs[5] == 1.0  # leader_gap (no leader)
    assert obs[6] == 0.0  # leader_rel_speed (no leader)
    assert obs[7] == 0.0  # queue_ahead
    assert obs[8] == 0.0  # queue_behind


def test_build_traci_error_returns_zero_vector():
    assembler = ObservationAssembler(TrustTracker())
    traci = MagicMock()
    traci.vehicle.getLaneIndex.side_effect = Exception("no such vehicle")
    with patch("env.features.traci", traci):
        obs = assembler.build("cav_0")
    assert obs.shape == (assembler.obs_dim,)
    assert np.all(obs == 0.0)
