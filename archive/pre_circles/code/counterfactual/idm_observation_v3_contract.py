"""Physics-selected predecision safety-history features for Gate-B repair v3."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

from counterfactual.idm_predictor_contract import (
    FORBIDDEN_FEATURE_TOKENS,
    IDM_DECISION_FEATURES,
)


CONTRACT_VERSION = "idm_counterfactual_escalation_observation/3"


CURRENT_SAFETY_FEATURES = (
    "decision_min_low_ttc_s",
    "merge_density_ema_scaled",
    "merge_inflow_scaled",
    "merge_mean_speed_scaled",
    "pre_merge_queue_scaled",
    "queue_spillback",
    "network_mean_speed_mps",
    "formation_soft_cav_count",
    "formation_follower_gap_mean_m",
    "formation_leader_gap_mean_m",
    "formation_follower_cav_fraction",
    "formation_leader_cav_fraction",
)


LOCAL_KINEMATIC_FEATURES = tuple(
    f"local_bin_{band}_{metric}"
    for band in ("0_100m", "100_200m", "200_400m", "400_800m")
    for metric in (
        "cav_count",
        "cav_fraction",
        "min_speed_mps",
        "mean_acceleration_mps2",
        "min_acceleration_mps2",
    )
)


LANE_RESOLVED_FEATURES = tuple(
    f"local_lane_{lane}_{metric}"
    for lane in (0, 1, 2)
    for metric in (
        "vehicle_count",
        "cav_count",
        "cav_fraction",
        "queue_count",
        "mean_speed_mps",
        "min_speed_mps",
        "speed_std_mps",
        "mean_acceleration_mps2",
        "min_acceleration_mps2",
        "hard_brake_count",
    )
)


PATH_SHAPE_FEATURES = (
    "pattern60_hard_brake_vehicle_steps",
    "pattern60_hard_brake_onset_count",
    "pattern60_low_ttc_pair_steps",
    "pattern60_inband_available_fraction",
    "pattern60_queue_count_change",
    "pattern60_queue_count_range",
    "pattern60_queue_count_max",
    "pattern60_pre_merge_speed_mps_change",
    "pattern60_pre_merge_speed_mps_range",
    "pattern60_pre_merge_speed_mps_min",
    "pattern60_merge_speed_mps_change",
    "pattern60_merge_speed_mps_range",
    "pattern60_merge_speed_mps_min",
    "pattern60_inflow_scaled_change",
    "pattern60_inflow_scaled_range",
    "pattern60_inflow_scaled_max",
    "pattern60_pre_merge_speed_std_mps_change",
    "pattern60_pre_merge_speed_std_mps_range",
    "pattern60_pre_merge_speed_std_mps_max",
)


LAGGED_STATE_FEATURES = tuple(
    f"traffic_lag_{lag}_{metric}"
    for lag in (5, 10, 20, 30, 60)
    for metric in (
        "density_scaled",
        "inflow_scaled",
        "merge_mean_speed_mps",
        "queue_count",
        "pre_merge_mean_speed_mps",
        "active_cav_pre_merge_lane_count",
    )
)


OBSERVATION_V3_ADDITIONS = tuple(
    dict.fromkeys(
        (
            *CURRENT_SAFETY_FEATURES,
            *LOCAL_KINEMATIC_FEATURES,
            *LANE_RESOLVED_FEATURES,
            *PATH_SHAPE_FEATURES,
            *LAGGED_STATE_FEATURES,
        )
    )
)


IDM_OBSERVATION_V3_FEATURES = tuple(
    dict.fromkeys((*IDM_DECISION_FEATURES, *OBSERVATION_V3_ADDITIONS))
)


def _names_sha256(names: Sequence[str]) -> str:
    encoded = json.dumps(tuple(names), separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return sha256(encoded).hexdigest().upper()


def feature_contract_manifest() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "feature_names": list(IDM_OBSERVATION_V3_FEATURES),
        "feature_count": len(IDM_OBSERVATION_V3_FEATURES),
        "feature_names_sha256": _names_sha256(IDM_OBSERVATION_V3_FEATURES),
        "added_feature_names": list(
            name for name in OBSERVATION_V3_ADDITIONS if name not in IDM_DECISION_FEATURES
        ),
        "source": "already_persisted_common_control_prefix_features",
        "latest_timestamp": "declared_decision_inclusive",
        "future_samples": 0,
        "new_sumo_outcomes": False,
        "selection_basis": "physics_selected_safety_history_before_v3_screen",
    }


def validate_observation_v3_features(row: Mapping[str, Any]) -> tuple[float, ...]:
    features = row.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("row lacks a decision-time features mapping")
    values = []
    for name in IDM_OBSERVATION_V3_FEATURES:
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS):
            raise ValueError(f"feature {name!r} resembles a postdecision field")
        if name == "context_decision_step":
            value = row.get("decision_step")
        elif name == "context_cav_penetration":
            value = row.get("cav_penetration", row.get("scenario", {}).get("cav_penetration"))
        else:
            value = features.get(name)
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"v3 feature {name!r} is missing or nonnumeric") from None
        if not math.isfinite(number):
            raise ValueError(f"v3 feature {name!r} is non-finite")
        values.append(number)
    return tuple(values)


__all__ = [
    "CONTRACT_VERSION",
    "IDM_OBSERVATION_V3_FEATURES",
    "OBSERVATION_V3_ADDITIONS",
    "feature_contract_manifest",
    "validate_observation_v3_features",
]

