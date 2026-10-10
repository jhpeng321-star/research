"""Frozen, leakage-safe data contract for the IDM escalation predictor.

This module defines targets and validates rows; it deliberately does not fit a
model or authorize treatment.  All learned inputs must exist at the declared
decision instant.  Treatment dose and paired outcomes are targets only.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

from counterfactual.actionability import dose_qualified
from counterfactual.escalation_v2_model import TRAFFIC_PLACEMENT_LOCAL
from counterfactual.headroom import TreatmentRepairCriteria


CONTRACT_VERSION_V1 = "idm_counterfactual_escalation_predictor/1"
CONTRACT_VERSION = "idm_counterfactual_escalation_predictor/2"

PREFLIGHT_FEATURES = (
    "preflight_active_cav_count",
    "preflight_damping_command_count",
    "preflight_member_count",
    "preflight_required_lane_count",
    "preflight_initial_span_m",
    "preflight_predicted_max_span_m",
    "preflight_predicted_terminal_span_m",
    "preflight_min_front_gap_m",
    "preflight_min_rear_gap_m",
    "preflight_min_front_ttc_s_capped",
    "preflight_min_rear_ttc_s_capped",
    "preflight_min_acceleration_mps2",
)

# Context is intentionally small.  Every remaining value is produced by
# counterfactual.pilot.features_from_trace from the control prefix only.
IDM_DECISION_FEATURES_V1 = tuple(
    dict.fromkeys(
        (
            "context_decision_step",
            "context_cav_penetration",
            *(name for name in TRAFFIC_PLACEMENT_LOCAL if name != "context_is_hyper"),
            "local_bin_400_800m_vehicle_count",
            "local_bin_400_800m_mean_speed_mps",
            "local_bin_400_800m_speed_std_mps",
            "local_bin_400_800m_queue_count",
            "local_bin_400_800m_hard_brake_count",
            *PREFLIGHT_FEATURES,
        )
    )
)

OBSERVATION_V2_FEATURES = (
    "obs_v2_postmerge_vehicle_count",
    "obs_v2_postmerge_density_veh_per_km_lane",
    "obs_v2_postmerge_occupancy_proxy",
    "obs_v2_postmerge_mean_speed_mps",
    "obs_v2_postmerge_harmonic_speed_mps",
    "obs_v2_postmerge_speed_variance_m2ps2",
    "obs_v2_postmerge_hard_brake_count",
    "obs_v2_postmerge_low_ttc_pair_count",
    "obs_v2_postmerge_discharge_rate_20s_vph",
    "obs_v2_postmerge_discharge_rate_60s_vph",
    "obs_v2_postmerge_density_slope_60s",
    "obs_v2_postmerge_speed_slope_60s_mps_per_sample",
    "obs_v2_postmerge_speed_variance_slope_60s",
    "obs_v2_postmerge_hard_brake_active_fraction_60s",
    "obs_v2_postmerge_low_ttc_active_fraction_60s",
    "obs_v2_postmerge_discharge_short_minus_long_vph",
    "obs_v2_premerge_speed_variance_20s",
    "obs_v2_premerge_speed_lag1_autocorr_20s",
    "obs_v2_premerge_speed_lag1_autocorr_60s",
    "obs_v2_premerge_gap_available_20s",
    "obs_v2_premerge_gap_variance_20s_m2",
    "obs_v2_premerge_gap_lag1_autocorr_20s",
    "obs_v2_premerge_gap_lag1_autocorr_60s",
    "obs_v2_dose_proposed_wall_speed_mps",
    "obs_v2_dose_release_boundary_margin_m",
    "obs_v2_dose_upstream_flow_60s_vph",
    "obs_v2_dose_flow_speed_product",
)

IDM_DECISION_FEATURES = tuple(
    dict.fromkeys((*IDM_DECISION_FEATURES_V1, *OBSERVATION_V2_FEATURES))
)

OBSERVATION_V2_PROVENANCE = {
    name: {
        "source": (
            "frozen_treatment_context_and_control_prefix"
            if name.startswith("obs_v2_dose_")
            else "common_control_prefix_vehicle_snapshots"
        ),
        "latest_timestamp": "declared_decision_inclusive",
        "future_samples": 0,
        "branch_identity": "identical_through_decision",
        "window_units": "simulation_seconds",
    }
    for name in OBSERVATION_V2_FEATURES
}

# Physics-informed primary inputs. The expanded allow-list above is retained for
# predeclared sensitivity analysis, not automatic feature selection.
IDM_PRIMARY_FEATURES = (
    "context_decision_step",
    "health",
    "health_change_30",
    "decision_queue_count",
    "decision_pre_merge_mean_speed_mps",
    "decision_merge_mean_speed_mps",
    "decision_hard_brake_vehicle_count",
    "decision_low_ttc_pair_count",
    "pattern60_queue_count_slope_per_step",
    "pattern60_pre_merge_speed_slope_mps_per_step",
    "pattern60_pre_merge_speed_std_mps_mean",
    "pattern60_hard_brake_active_fraction",
    "pattern60_low_ttc_active_fraction",
    "placement_strong_band_cav_count",
    "placement_strong_band_lane_coverage_fraction",
    "local_downstream_minus_upstream_speed_mps",
    "local_bin_0_100m_mean_speed_mps",
    "local_bin_100_200m_mean_speed_mps",
    "local_bin_200_400m_mean_speed_mps",
    "local_bin_400_800m_mean_speed_mps",
    "preflight_member_count",
    "preflight_required_lane_count",
    "preflight_predicted_max_span_m",
    "preflight_predicted_terminal_span_m",
    "preflight_min_front_gap_m",
    "preflight_min_rear_gap_m",
    "preflight_min_front_ttc_s_capped",
    "preflight_min_rear_ttc_s_capped",
    "preflight_min_acceleration_mps2",
    *OBSERVATION_V2_FEATURES,
)


def _names_sha256(names: Sequence[str]) -> str:
    encoded = json.dumps(tuple(names), separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return sha256(encoded).hexdigest().upper()


def feature_contract_manifest() -> dict[str, Any]:
    """Return a stable identifier for the frozen predictor input schema."""
    return {
        "contract_version": CONTRACT_VERSION,
        "supersedes_contract_version": CONTRACT_VERSION_V1,
        "primary_feature_names": list(IDM_PRIMARY_FEATURES),
        "primary_feature_count": len(IDM_PRIMARY_FEATURES),
        "primary_feature_names_sha256": _names_sha256(IDM_PRIMARY_FEATURES),
        "expanded_feature_names": list(IDM_DECISION_FEATURES),
        "expanded_feature_count": len(IDM_DECISION_FEATURES),
        "expanded_feature_names_sha256": _names_sha256(IDM_DECISION_FEATURES),
        # Backward-compatible aliases identify the persisted expanded schema.
        "feature_names": list(IDM_DECISION_FEATURES),
        "feature_count": len(IDM_DECISION_FEATURES),
        "feature_names_sha256": _names_sha256(IDM_DECISION_FEATURES),
        "feature_time_scope": "control prefix through declared decision only",
        "observation_v2_feature_names": list(OBSERVATION_V2_FEATURES),
        "observation_v2_feature_count": len(OBSERVATION_V2_FEATURES),
        "observation_v2_provenance": OBSERVATION_V2_PROVENANCE,
        "feature_window_units": "simulation_seconds",
        "feature_branch_contract": "computed once on common prefix; copied to both arms",
    }


FORBIDDEN_FEATURE_TOKENS = (
    "outcome",
    "benefit",
    "effect",
    "treatment",
    "override",
    "realized",
    "abort",
    "engaged",
    "completed",
    "collision_excess",
    "teleport_excess",
)


@dataclass(frozen=True)
class StageTargets:
    """Targets for the feasibility/safety head and conditional effect heads."""

    deterministic_admitted: bool
    dose_qualified: bool
    safety_admissible: bool
    operationally_admissible: bool
    joint_safe_beneficial: bool
    hard_brake_onset_reduction: float | None
    hard_brake_exposure_reduction_vehicle_steps: float | None
    low_ttc_exposure_reduction_steps: float | None
    merge_discharge_gain: float | None
    delay_benefit_s: float | None


def preflight_features(scan: Mapping[str, Any]) -> dict[str, float]:
    """Flatten pre-override formation diagnostics into deployable features.

    The deterministic supervisor already computes these values before applying
    a wall command. A missing TTC means there is no closing encounter; it is
    represented by the frozen 120 s cap rather than an infinity.
    """
    formation = scan.get("formation") or {}
    safety = scan.get("activation_safety") or {}

    def number(value: Any, *, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    def capped_ttc(value: Any) -> float:
        return min(max(number(value, default=120.0), 0.0), 120.0)

    return {
        "preflight_active_cav_count": number(scan.get("active_cav_count")),
        "preflight_damping_command_count": number(scan.get("damping_command_count")),
        "preflight_member_count": float(len(formation.get("controlled_cav_ids") or ())),
        "preflight_required_lane_count": float(len(formation.get("required_lane_ids") or ())),
        "preflight_initial_span_m": number(formation.get("longitudinal_span_m")),
        "preflight_predicted_max_span_m": number(formation.get("predicted_max_span_m")),
        "preflight_predicted_terminal_span_m": number(formation.get("predicted_terminal_span_m")),
        "preflight_min_front_gap_m": number(safety.get("min_front_gap_m")),
        "preflight_min_rear_gap_m": number(safety.get("min_rear_gap_m")),
        "preflight_min_front_ttc_s_capped": capped_ttc(safety.get("min_front_ttc_s")),
        "preflight_min_rear_ttc_s_capped": capped_ttc(safety.get("min_rear_ttc_s")),
        "preflight_min_acceleration_mps2": number(safety.get("min_acceleration_mps2")),
    }


def complete_group_id(row: Mapping[str, Any]) -> str:
    """Keep every decision window from one physical seed in the same split."""
    demand = int(row.get("demand_mainline_vph", row.get("scenario", {}).get("demand", 0)))
    penetration = float(
        row.get("cav_penetration", row.get("scenario", {}).get("cav_penetration", 0.0))
    )
    model = str(row.get("hdv_model", row.get("scenario", {}).get("hdv_model", "IDM")))
    return f"m{demand}|p{penetration:.6f}|{model}|seed{int(row['seed'])}"


def validate_decision_features(
    row: Mapping[str, Any],
    feature_names: Sequence[str] = IDM_DECISION_FEATURES,
) -> tuple[float, ...]:
    """Return a finite vector and fail closed on a non-allow-listed input."""
    features = row.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("row lacks a decision-time features mapping")
    values = []
    for name in feature_names:
        if name not in IDM_DECISION_FEATURES:
            raise ValueError(f"feature {name!r} is outside the frozen allow-list")
        lowered = name.lower()
        if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
            raise ValueError(f"feature {name!r} resembles a postdecision field")
        if name == "context_decision_step":
            value = row.get("decision_step")
        elif name == "context_cav_penetration":
            value = row.get(
                "cav_penetration",
                row.get("scenario", {}).get("cav_penetration"),
            )
        else:
            value = features.get(name)
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"feature {name!r} is missing or nonnumeric") from None
        if not math.isfinite(number):
            raise ValueError(f"feature {name!r} is non-finite")
        values.append(number)
    return tuple(values)


def staged_targets(
    row: Mapping[str, Any],
    criteria: TreatmentRepairCriteria = TreatmentRepairCriteria(
        require_collapse_prevention_diversity=False
    ),
) -> StageTargets:
    """Build targets without folding treatment delivery into treatment benefit.

    Effect targets are defined only after a dose-qualified, safety-admissible
    treatment.  This prevents formation failure from becoming a false negative
    treatment-effect label.
    """
    scan = row.get("timing_scan") or {}
    deterministic_admitted = bool(scan.get("admitted", False))
    completed = dose_qualified(row, criteria)
    oracle = row.get("safety_service_oracle") or {}
    safe = bool(oracle.get("safety_admissible", False)) if completed else False
    admissible = bool(deterministic_admitted and completed and safe)
    joint_safe_beneficial = bool(
        deterministic_admitted and completed and oracle.get("wall_selected", False)
    )
    effects = oracle.get("effects") or {}
    if not admissible:
        return StageTargets(
            deterministic_admitted=deterministic_admitted,
            dose_qualified=completed,
            safety_admissible=safe,
            operationally_admissible=False,
            joint_safe_beneficial=False,
            hard_brake_onset_reduction=None,
            hard_brake_exposure_reduction_vehicle_steps=None,
            low_ttc_exposure_reduction_steps=None,
            merge_discharge_gain=None,
            delay_benefit_s=None,
        )
    hard_brake_onset_excess = effects.get("hard_brake_onset_excess")
    hard_brake_exposure_excess = effects.get("hard_brake_excess_vehicle_steps")
    return StageTargets(
        deterministic_admitted=True,
        dose_qualified=True,
        safety_admissible=True,
        operationally_admissible=True,
        joint_safe_beneficial=joint_safe_beneficial,
        hard_brake_onset_reduction=(
            -float(hard_brake_onset_excess)
            if hard_brake_onset_excess is not None
            else None
        ),
        hard_brake_exposure_reduction_vehicle_steps=(
            -float(hard_brake_exposure_excess)
            if hard_brake_exposure_excess is not None
            else None
        ),
        low_ttc_exposure_reduction_steps=-float(effects["low_ttc_excess_steps"]),
        merge_discharge_gain=float(effects["merge_discharge_benefit"]),
        delay_benefit_s=float(effects["delay_benefit_s"]),
    )
