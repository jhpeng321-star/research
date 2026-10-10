import numpy as np

from idm.temporal_option_context import (
    TEMPORAL_OPTION_CONTEXT_NAMES,
    frame_levels,
    temporal_option_context,
)


def _frame(time_s: float, speed: float, front: float) -> dict:
    band = {
        "start_distance_m": 0.0,
        "end_distance_m": 50.0,
        "vehicle_count": 10,
        "mean_speed_mps": speed,
        "p10_speed_mps": speed - 2.0,
        "speed_variance_m2ps2": 4.0,
        "acceleration_energy_m2ps4": 5.0,
        "severe_braking_count": 1,
        "low_ttc_count": 2,
    }
    empty = {
        "vehicle_count": 0,
        "mean_speed_mps": None,
        "acceleration_energy_m2ps4": None,
        "severe_braking_count": 0,
    }
    return {
        "snapshot_available": True,
        "simulation_time_s": time_s,
        "mainline": {"all_lanes": [band]},
        "ramp": {"summary": empty},
        "merge": empty,
        "downstream": empty,
        "fronts_by_lane": {
            "pre_0": {
                "detached_wave_front_distance_m": front,
                "queue_tail_distance_m": None,
            }
        },
        "front_velocity_by_lane": {
            "pre_0": {"30s": {"velocity_mps": 2.0, "sample_count": 3, "r_squared": 0.8}}
        },
    }


def test_temporal_context_is_fixed_finite_and_causal():
    frames = [_frame(float(step), 30.0 - step, 100.0 + 2.0 * step) for step in range(4)]
    values = temporal_option_context(frames)
    assert len(values) == len(TEMPORAL_OPTION_CONTEXT_NAMES) == 52
    assert np.all(np.isfinite(values))
    assert values[1] == (30.0 - 3.0) / 35.0
    assert values[22 + 1] < 0.0
    assert values[-3] == (100.0 + 2.0 * 3.0) / 2000.0


def test_frame_levels_handles_empty_corridor():
    empty = _frame(0.0, 30.0, 100.0)
    empty["mainline"]["all_lanes"] = []
    values = frame_levels(empty)
    assert len(values) == 22
    assert np.all(np.isfinite(values))
