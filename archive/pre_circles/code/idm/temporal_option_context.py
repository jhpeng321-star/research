"""Causal temporal context for the IDM binary damping option.

The feature contract deliberately summarizes only the read-only RSU history
available at the declared decision.  It is a development diagnostic until the
observation-sufficiency gate explicitly promotes it into the actor contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import math

import numpy as np


TEMPORAL_OPTION_CONTEXT_SCHEMA = "idm_temporal_option_context/1"

_LEVEL_NAMES = (
    "mainline_vehicle_count_over_150",
    "mainline_mean_speed_over_35mps",
    "mainline_weighted_p10_speed_over_35mps",
    "mainline_min_band_mean_speed_over_35mps",
    "mainline_speed_variance_over_400m2ps2",
    "mainline_acceleration_energy_over_500m2ps4",
    "mainline_severe_braking_count_over_20",
    "mainline_low_ttc_count_over_20",
    "mainline_slow_vehicle_fraction",
    "mainline_slowest_band_distance_over_2000m",
    "ramp_vehicle_count_over_40",
    "ramp_mean_speed_over_35mps",
    "ramp_acceleration_energy_over_150m2ps4",
    "ramp_severe_braking_count_over_10",
    "merge_vehicle_count_over_40",
    "merge_mean_speed_over_35mps",
    "merge_acceleration_energy_over_150m2ps4",
    "merge_severe_braking_count_over_10",
    "downstream_vehicle_count_over_40",
    "downstream_mean_speed_over_35mps",
    "downstream_acceleration_energy_over_150m2ps4",
    "downstream_severe_braking_count_over_10",
)

TEMPORAL_LEVEL_NAMES = tuple(f"latest_{name}" for name in _LEVEL_NAMES)
TEMPORAL_TREND_NAMES = tuple(f"causal_10s_change_{name}" for name in _LEVEL_NAMES)
TEMPORAL_FRONT_NAMES = (
    "front_30s_velocity_mean_over_10mps",
    "front_30s_velocity_min_over_10mps",
    "front_30s_velocity_max_over_10mps",
    "front_30s_velocity_lane_coverage",
    "front_30s_velocity_mean_r_squared",
    "front_current_nearest_distance_over_2000m",
    "front_current_farthest_distance_over_2000m",
    "front_current_lane_detection_fraction",
)
TEMPORAL_OPTION_CONTEXT_NAMES = TEMPORAL_LEVEL_NAMES + TEMPORAL_TREND_NAMES + TEMPORAL_FRONT_NAMES


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _weighted(bands: Sequence[Mapping[str, Any]], field: str) -> float:
    pairs = [
        (_finite(row.get(field)), int(row.get("vehicle_count", 0)))
        for row in bands
        if int(row.get("vehicle_count", 0)) > 0 and row.get(field) is not None
    ]
    weight = sum(count for _, count in pairs)
    return sum(value * count for value, count in pairs) / weight if weight else 0.0


def _road_values(summary: Mapping[str, Any], count_scale: float) -> tuple[float, ...]:
    energy_scale = 150.0
    return (
        _finite(summary.get("vehicle_count")) / count_scale,
        _finite(summary.get("mean_speed_mps")) / 35.0,
        _finite(summary.get("acceleration_energy_m2ps4")) / energy_scale,
        _finite(summary.get("severe_braking_count")) / 10.0,
    )


def frame_levels(frame: Mapping[str, Any]) -> tuple[float, ...]:
    """Return the normalized, prespecified 22-field level summary."""

    bands = tuple((frame.get("mainline") or {}).get("all_lanes") or ())
    occupied = [row for row in bands if int(row.get("vehicle_count", 0)) > 0]
    total_count = sum(int(row.get("vehicle_count", 0)) for row in bands)
    slow_count = sum(
        int(row.get("vehicle_count", 0))
        for row in occupied
        if _finite(row.get("p10_speed_mps"), 35.0) < 15.0
    )
    slowest = min(
        occupied,
        key=lambda row: _finite(row.get("mean_speed_mps"), 35.0),
        default=None,
    )
    slowest_distance = (
        0.5 * (_finite(slowest.get("start_distance_m")) + _finite(slowest.get("end_distance_m")))
        if slowest is not None
        else 2000.0
    )
    values = (
        total_count / 150.0,
        _weighted(bands, "mean_speed_mps") / 35.0,
        _weighted(bands, "p10_speed_mps") / 35.0,
        min(
            (_finite(row.get("mean_speed_mps"), 35.0) for row in occupied),
            default=35.0,
        )
        / 35.0,
        _weighted(bands, "speed_variance_m2ps2") / 400.0,
        sum(_finite(row.get("acceleration_energy_m2ps4")) for row in bands) / 500.0,
        sum(_finite(row.get("severe_braking_count")) for row in bands) / 20.0,
        sum(_finite(row.get("low_ttc_count")) for row in bands) / 20.0,
        slow_count / max(total_count, 1),
        slowest_distance / 2000.0,
        *_road_values((frame.get("ramp") or {}).get("summary") or {}, 40.0),
        *_road_values(frame.get("merge") or {}, 40.0),
        *_road_values(frame.get("downstream") or {}, 40.0),
    )
    if len(values) != len(_LEVEL_NAMES) or not np.all(np.isfinite(values)):
        raise ValueError("temporal level context is incomplete or non-finite")
    return tuple(float(np.clip(value, -5.0, 5.0)) for value in values)


def _causal_trends(frames: Sequence[Mapping[str, Any]], levels: np.ndarray) -> tuple[float, ...]:
    times = np.asarray([_finite(frame.get("simulation_time_s")) for frame in frames])
    if len(times) < 2 or float(np.ptp(times)) <= 1e-9:
        return (0.0,) * len(_LEVEL_NAMES)
    centered = times - float(np.mean(times))
    denominator = float(centered @ centered)
    slopes = centered @ (levels - np.mean(levels, axis=0)) / denominator
    # Report the fitted change over ten seconds in the already normalized unit.
    return tuple(float(np.clip(value * 10.0, -5.0, 5.0)) for value in slopes)


def _front_context(frame: Mapping[str, Any]) -> tuple[float, ...]:
    fits = frame.get("front_velocity_by_lane") or {}
    velocities: list[float] = []
    r_squared: list[float] = []
    for lane in fits.values():
        fit = (lane or {}).get("30s") or {}
        if fit.get("velocity_mps") is not None:
            velocities.append(_finite(fit.get("velocity_mps")))
            if fit.get("r_squared") is not None:
                r_squared.append(_finite(fit.get("r_squared")))
    fronts = frame.get("fronts_by_lane") or {}
    positions = []
    for row in fronts.values():
        position = row.get("detached_wave_front_distance_m")
        if position is None:
            position = row.get("queue_tail_distance_m")
        if position is not None:
            positions.append(_finite(position))
    lane_count = max(len(fronts), len(fits), 1)
    values = (
        (float(np.mean(velocities)) if velocities else 0.0) / 10.0,
        (min(velocities) if velocities else 0.0) / 10.0,
        (max(velocities) if velocities else 0.0) / 10.0,
        len(velocities) / lane_count,
        float(np.mean(r_squared)) if r_squared else 0.0,
        (min(positions) if positions else 2000.0) / 2000.0,
        (max(positions) if positions else 2000.0) / 2000.0,
        len(positions) / lane_count,
    )
    return tuple(float(np.clip(value, -5.0, 5.0)) for value in values)


def temporal_option_context(
    predecision_window: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    """Summarize a causal 30-second spatial window into 52 fixed fields."""

    frames = tuple(
        sorted(
            (row for row in predecision_window if row.get("snapshot_available", True)),
            key=lambda row: _finite(row.get("simulation_time_s")),
        )
    )
    if not frames:
        raise ValueError("predecision spatial window is empty")
    latest_time = _finite(frames[-1].get("simulation_time_s"))
    causal = tuple(
        row for row in frames if latest_time - _finite(row.get("simulation_time_s")) <= 30.0 + 1e-9
    )
    levels = np.asarray([frame_levels(row) for row in causal], dtype=float)
    values = (
        *levels[-1].tolist(),
        *_causal_trends(causal, levels),
        *_front_context(causal[-1]),
    )
    if len(values) != len(TEMPORAL_OPTION_CONTEXT_NAMES):
        raise AssertionError("temporal feature contract length changed")
    if not np.all(np.isfinite(values)):
        raise ValueError("temporal option context contains non-finite values")
    return tuple(float(value) for value in values)
