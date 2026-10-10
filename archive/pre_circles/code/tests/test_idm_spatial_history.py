from copy import deepcopy

import pytest

from idm.spatial_history import (
    SpatialHistoryContract,
    SpatialRSUHistory,
    _fit_front_velocity,
    capture_corridor_frame,
)
from tools.run_idm_decentralized_authority_verification import (
    _observation_fingerprint,
    _physical_state_fingerprint,
    _treatment_fingerprint,
)


class _Engine:
    down_edge = "downstream"


class _Env:
    _pre_edge = "mainline"
    _ramp_edge = "ramp"
    _merge_edge = "merge"
    _shepherd_engine = _Engine()


class _Snapshot:
    def __init__(self, rows):
        self.rows = rows

    def ordered_ids(self):
        return tuple(self.rows)

    def road(self, vehicle_id):
        return self.rows[vehicle_id]["road"]

    def lane_id(self, vehicle_id):
        return self.rows[vehicle_id]["lane"]

    def lane_pos(self, vehicle_id):
        return self.rows[vehicle_id]["position"]

    def speed(self, vehicle_id):
        return self.rows[vehicle_id]["speed"]

    def accel(self, vehicle_id):
        return self.rows[vehicle_id]["acceleration"]

    def leader(self, vehicle_id):
        return self.rows[vehicle_id].get("leader")


def _frame(rows, *, time_s=1.0, contract=None):
    return capture_corridor_frame(
        _Env(),
        contract or SpatialHistoryContract(),
        snapshot=_Snapshot(rows),
        simulation_time_s=time_s,
        lane_lengths={"mainline_0": 1000.0, "mainline_1": 1000.0, "ramp_0": 400.0},
    )


def test_fixed_band_layout_uses_live_lane_length_and_null_empty_statistics():
    contract = SpatialHistoryContract(
        mainline_upstream_coverage_m=200.0,
        band_width_m=50.0,
    )
    frame = _frame(
        {
            "a": {
                "road": "mainline",
                "lane": "mainline_0",
                "position": 925.0,
                "speed": 25.0,
                "acceleration": 0.0,
            }
        },
        contract=contract,
    )
    bands = frame["mainline"]["by_lane"]["mainline_0"]
    assert len(bands) == 4
    assert bands[1]["vehicle_count"] == 1
    assert bands[1]["mean_speed_mps"] == pytest.approx(25.0)
    assert bands[0]["vehicle_count"] == 0
    assert bands[0]["mean_speed_mps"] is None
    assert bands[0]["density_speed_flow_proxy_veh_per_s"] is None


def test_single_slow_vehicle_does_not_create_queue_or_wave_front():
    frame = _frame(
        {
            "a": {
                "road": "mainline",
                "lane": "mainline_0",
                "position": 900.0,
                "speed": 10.0,
                "acceleration": -1.0,
            }
        }
    )
    front = frame["fronts_by_lane"]["mainline_0"]
    assert front["queue_tail_distance_m"] is None
    assert front["detached_wave_front_distance_m"] is None
    assert front["front_detection_reason"] == "no_qualified_slow_cluster"


def test_connected_slow_cluster_produces_queue_tail():
    frame = _frame(
        {
            "a": {
                "road": "mainline",
                "lane": "mainline_0",
                "position": 900.0,
                "speed": 10.0,
                "acceleration": -1.0,
            },
            "b": {
                "road": "mainline",
                "lane": "mainline_0",
                "position": 850.0,
                "speed": 11.0,
                "acceleration": -0.5,
            },
        }
    )
    front = frame["fronts_by_lane"]["mainline_0"]
    assert front["queue_tail_distance_m"] == pytest.approx(150.0)
    assert front["detached_wave_front_distance_m"] is None


def test_detached_slow_cluster_is_wave_not_queue():
    frame = _frame(
        {
            "a": {
                "road": "mainline",
                "lane": "mainline_0",
                "position": 400.0,
                "speed": 10.0,
                "acceleration": -1.0,
            },
            "b": {
                "road": "mainline",
                "lane": "mainline_0",
                "position": 350.0,
                "speed": 11.0,
                "acceleration": -0.5,
            },
        }
    )
    front = frame["fronts_by_lane"]["mainline_0"]
    assert front["queue_tail_distance_m"] is None
    assert front["detached_wave_front_distance_m"] == pytest.approx(650.0)
    assert front["front_detection_reason"] == "detached_slow_cluster"


def test_positive_upstream_propagation_has_positive_velocity():
    fit = _fit_front_velocity([(0.0, 100.0), (1.0, 110.0), (2.0, 120.0)])
    assert fit["velocity_mps"] == pytest.approx(10.0)
    assert fit["sample_count"] == 3
    assert fit["r_squared"] == pytest.approx(1.0)


def test_history_fits_are_causal_and_never_use_future_frames():
    contract = SpatialHistoryContract(sample_interval_s=1.0)
    history = SpatialRSUHistory(contract)

    def frame(time_s, position):
        return {
            "schema": "idm_spatial_rsu_history/1",
            "simulation_time_s": time_s,
            "fronts_by_lane": {
                "mainline_0": {
                    "queue_tail_distance_m": None,
                    "detached_wave_front_distance_m": position,
                }
            },
        }

    history.observe(frame(0.0, 100.0))
    at_one = history.observe(frame(1.0, 110.0))
    assert at_one["front_velocity_by_lane"]["mainline_0"]["10s"]["sample_count"] == 2
    at_two = history.observe(frame(2.0, 120.0))
    assert at_two["front_velocity_by_lane"]["mainline_0"]["10s"]["sample_count"] == 3
    assert at_two["front_velocity_by_lane"]["mainline_0"]["10s"]["velocity_mps"] == pytest.approx(10.0)


def test_capture_is_read_only_with_respect_to_snapshot_records():
    rows = {
        "a": {
            "road": "mainline",
            "lane": "mainline_0",
            "position": 900.0,
            "speed": 10.0,
            "acceleration": -1.0,
        }
    }
    before = deepcopy(rows)
    _frame(rows)
    assert rows == before


def test_diagnostic_history_is_excluded_from_all_decision_fingerprints():
    decision = {
        "simulation_time_s": 10.0,
        "lane_id": "mainline_0",
        "action_set": "activation_speed_ceiling_v1",
        "terminal_reference_percentile": "p30",
        "candidates": [{"slot": 1, "vehicle_id": "cav", "lane_id": "mainline_0"}],
        "pattern_context": {"idm_wave_energy_level": 1.0},
        "candidate_observations": {"cav": [0.0, 1.0]},
        "candidate_observation_metadata": {"cav": {"schema": "legacy"}},
    }
    with_history = {
        **decision,
        "diagnostic_spatial_history": [{"simulation_time_s": 9.0, "bands": [1, 2]}],
    }
    assert _physical_state_fingerprint(decision) == _physical_state_fingerprint(with_history)
    assert _treatment_fingerprint(decision) == _treatment_fingerprint(with_history)
    assert _observation_fingerprint(decision) == _observation_fingerprint(with_history)
