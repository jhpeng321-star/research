from tools.audit_idm_phase5_decision_state_sufficiency import _population_metrics
from tools.analyze_idm_phase5_spatial_response import _paired_level_effects


class _Snapshot:
    def __init__(self):
        self._speed = {"a": 10.0, "b": 8.0}
        self._accel = {"a": -4.0, "b": 2.0}

    def speed(self, vehicle_id):
        return self._speed.get(vehicle_id)

    def accel(self, vehicle_id):
        return self._accel.get(vehicle_id)

    def leader(self, vehicle_id):
        return ("b", 6.0) if vehicle_id == "a" else None


def test_population_metrics_capture_spatial_safety_and_wave_response():
    result = _population_metrics(
        _Snapshot(),
        {"a", "b"},
        previous_acceleration={"a": -2.0, "b": 1.0},
        severe_threshold_mps2=-3.0,
    )

    assert result["vehicle_count"] == 2
    assert result["speed_variance_m2ps2"] == 1.0
    assert result["acceleration_energy_m2ps4"] == 20.0
    assert result["integrated_negative_acceleration_mps"] == 4.0
    assert result["severe_braking_count"] == 1
    assert result["mean_abs_jerk_mps3"] == 1.5
    assert result["minimum_ttc_s"] == 3.0


def test_spatial_effects_are_paired_and_positive_for_improvement():
    zero = {
        "spatial_trace": [
            {
                "relative_step": 1,
                "absorber": {
                    "acceleration_energy_m2ps4": 4.0,
                    "integrated_negative_acceleration_mps": 2.0,
                    "severe_braking_count": 1,
                    "speed_variance_m2ps2": 2.0,
                    "mean_abs_jerk_mps3": 1.0,
                    "minimum_ttc_s": 1.0,
                    "mean_speed_mps": 10.0,
                    "mean_acceleration_mps2": -2.0,
                },
            }
        ]
    }
    action = {
        "spatial_trace": [
            {
                "relative_step": 1,
                "absorber": {
                    "acceleration_energy_m2ps4": 1.0,
                    "integrated_negative_acceleration_mps": 1.0,
                    "severe_braking_count": 0,
                    "speed_variance_m2ps2": 1.0,
                    "mean_abs_jerk_mps3": 0.5,
                    "minimum_ttc_s": 2.0,
                    "mean_speed_mps": 11.0,
                    "mean_acceleration_mps2": -1.0,
                },
            }
        ]
    }

    result = _paired_level_effects(zero, action, horizon_steps=1)["absorber"]
    assert result["acceleration_energy_m2ps4_improvement_sum"] == 3.0
    assert result["minimum_ttc_improvement_mean_s"] == 1.0
    assert result["mean_speed_response_mps"] == 1.0
