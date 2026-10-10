"""Pure contracts for the timing observation-repair phase.

Frozen 40-feature provenance table, causal window summaries and identity
checks.  No TraCI calls.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# family dims: A=8, B=14, C=10, D=8 -> 40 features in total
FAMILIES = ("A", "B", "C", "D")
FAMILY_DIMS = {"A": 8, "B": 14, "C": 10, "D": 8}
WINDOWS = (5, 20)

PROVENANCE: tuple[tuple[str, str, str], ...] = (
    ("stage_index", "A", "recurrence index at the decision instant"),
    ("landmark_wave", "A", "wave_onset indicator of the landmark kind"),
    ("persistence_steps", "A", "frozen landmark persistence count"),
    ("runway_m", "A", "candidate runway to the detected disturbance"),
    ("predicted_dose_s", "A", "predicted action dose from runway/speed"),
    ("current_instant_s", "A", "causal simulation time at the instant"),
    ("lane_1", "A", "target lane one-hot index 1"),
    ("lane_0", "A", "target lane one-hot index 0"),
    ("ego_speed", "B", "target speed at the instant"),
    ("ego_accel", "B", "target acceleration at the instant"),
    ("leader_gap", "B", "target leader gap at the instant"),
    ("leader_rel_speed", "B", "leader speed minus target speed, at the instant"),
    ("follower_gap", "B", "target follower gap at the instant"),
    ("follower_closing", "B", "follower speed minus target speed, at the instant"),
    ("speed_std_5s", "B", "std of target speed over the last 5 s (causal)"),
    ("speed_std_20s", "B", "std of target speed over the last 20 s (causal)"),
    ("min_gap_5s", "B", "min target leader gap over the last 5 s"),
    ("min_gap_20s", "B", "min target leader gap over the last 20 s"),
    ("max_closing_5s", "B", "max target-lane leader closing over the last 5 s"),
    ("max_closing_20s", "B", "max target-lane leader closing over the last 20 s"),
    ("severe_count_5s", "B", "count of sub--3 m/s2 events in the target lane (5 s)"),
    ("severe_count_20s", "B", "count of sub--3 m/s2 events in the target lane (20 s)"),
    ("merge_lane0_mean_speed", "C", "mean speed in 0-300 m zone, lane 0"),
    ("merge_lane1_mean_speed", "C", "mean speed in 0-300 m zone, lane 1"),
    ("merge_lane0_speed_std", "C", "speed std in 0-300 m zone, lane 0"),
    ("merge_lane1_speed_std", "C", "speed std in 0-300 m zone, lane 1"),
    ("merge_zone_density", "C", "vehicles per km in the 0-300 m zone"),
    ("ramp_count", "C", "vehicles on the ramp edge at the instant"),
    ("ramp_slow_count", "C", "ramp vehicles below 3 m/s at the instant"),
    ("merge_discharge_rate_10s", "C", "mean per-second discharge over the last 10 s"),
    ("zone_severe_count_10s", "C", "zone sub--3 m/s2 events over the last 10 s"),
    ("zone_low_ttc_count_10s", "C", "zone sub-1.5 s pair events over the last 10 s"),
    ("corridor_mean_speed", "D", "pre-edge mean speed"),
    ("corridor_speed_std", "D", "pre-edge speed std"),
    ("corridor_fraction_slow", "D", "fraction of pre-edge vehicles below 15 m/s"),
    ("gradient_upstream_merge", "D", "mean speed [300,800) m minus [0,300) m"),
    ("gradient_merge_downstream", "D", "downstream-edge mean speed minus [0,300) m"),
    ("idm_wave_energy_level", "D", "frozen causal corridor wave-energy level"),
    ("idm_severe_short_ema", "D", "frozen causal corridor severe EMA"),
    ("cav_upstream_count", "D", "CAVs on the pre-edge (fleet deployment knowledge)"),
)
PROVENANCE_BY_FAMILY = {"A": [], "B": [], "C": [], "D": []}
for _name, _family, _source in PROVENANCE:
    PROVENANCE_BY_FAMILY[_family].append(_name)


def feature_indeces(families: Iterable[str]) -> list[int]:
    """Feature column indices for a family subset (order A, B, C, D)."""
    chosen = [family for family in FAMILIES if family in set(families)]
    offsets = {"A": 0, "B": FAMILY_DIMS["A"], "C": FAMILY_DIMS["A"] + FAMILY_DIMS["B"], "D": sum(FAMILY_DIMS.values()) - FAMILY_DIMS["D"]}
    indices = []
    for family in chosen:
        indices.extend(range(offsets[family], offsets[family] + FAMILY_DIMS[family]))
    return indices


def summarize_window(series: Sequence[float]) -> dict[str, float]:
    """Causal summaries for one series (values are already window-filtered)."""
    values = [float(value) for value in series]
    if not values:
        return {"std": 0.0, "min": 0.0, "max": 0.0, "count": 0, "available": 0.0}
    return {
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "count": len(values),
        "available": 1.0,
    }


def window_values(
    series: Sequence[tuple[float, float]], now: float, window_s: float
) -> list[float]:
    """Values with sample time in (now - window_s, now], causal only."""
    return [value for t, value in series if float(t) <= now and float(t) > now - window_s]


def identity_check(
    reconstructed: Mapping[str, Any], stored: Mapping[str, Any]
) -> dict[str, bool]:
    """Exact identity between a reconstructed landmark and the frozen schedule."""
    return {
        "step": int(reconstructed["step"]) == int(stored["step"]),
        "landmark": str(reconstructed.get("landmark")) == str(stored.get("landmark")),
        "candidate_id": str(reconstructed.get("candidate_id"))
        == str(stored.get("candidate_id")),
        "lane_id": str(reconstructed.get("lane_id")) == str(stored.get("lane_id")),
        "observation_fingerprint": str(reconstructed.get("observation_hash"))
        == str(stored.get("observation_hash")),
        "physical_fingerprint": str(reconstructed.get("raw_decision_context_fingerprint"))
        == str(stored.get("raw_decision_context_fingerprint")),
        "config_hash": str(reconstructed.get("config_hash"))
        == str(stored.get("config_hash")),
    }


def mean_fill(matrix: np.ndarray, means: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Training-fold-only mean imputation; returns filled matrix and means."""
    filled = np.array(matrix, dtype=float)
    if means is None:
        means = np.where(np.isnan(filled), np.nan, filled).mean(axis=0)
        means = np.where(np.isnan(means), 0.0, means)
    filled = np.where(np.isnan(filled), means, filled)
    return filled, means
