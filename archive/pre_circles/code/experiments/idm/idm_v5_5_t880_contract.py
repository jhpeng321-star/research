"""Frozen timing-aligned Gate-F and conditional wall-pilot contract."""

from __future__ import annotations

from hashlib import sha256
import json

from experiments.idm.wall_v5_4_contract import profile_manifest
from policies.idm_analytical import frozen_idm_analytical_manifest


SCHEMA = "idm_v5_5_t880_wall_contract/1"
SCIENTIFIC_ROLE = "timing_aligned_late_congestion_wall_headroom_development"
DEMAND_MAINLINE_VPH = 2700
DEMAND_RAMP_VPH = 400
CAV_PENETRATION = 0.30
DEVELOPMENT_SEEDS = tuple(range(4401, 4407))
ACTIVATION_TIME_S = 880
RELEASE_TIME_S = 900
TREATMENT_DURATION_S = 20
ANALYSIS_END_S = 1080
DENSITY_MAPPER_SUMMARY = (
    "runs/probe/idm_v5_5_damping_density_mapper_p30_s4401_4406/summary.json"
)
DENSITY_MAPPER_SUMMARY_SHA256 = (
    "92c1727da054e1640ab62afc6cdb43ea18d593ee025f00e72aefb4b9eee9effb"
)
DENSITY_CONFIRMATIONS_S = (
    {"seed": 4405, "onset_s": 907, "confirmation_s": 966},
    {"seed": 4403, "onset_s": 916, "confirmation_s": 975},
    {"seed": 4402, "onset_s": 940, "confirmation_s": 999},
    {"seed": 4406, "onset_s": 940, "confirmation_s": 999},
)
PRIMARY_QUEUE_COUNT = 5
PRIMARY_SPEED_THRESHOLD_MPS = 15.0
PRIMARY_SUSTAIN_S = 60
SECONDARY_HEALTH_THRESHOLD = 0.15
SECONDARY_HEALTH_SUSTAIN_S = 60
MINIMUM_FEASIBLE_SEEDS = 4
MINIMUM_DURATION_FRACTION = 1.0
MINIMUM_REALIZED_DURATION_FRACTION = 1.0
MINIMUM_REALIZED_LANE_COVERAGE_FRACTION = 1.0
MINIMUM_SHORTENING_S = 10
REPEAT_SEED = 4401


def contract_manifest() -> dict:
    payload = {
        "schema": SCHEMA,
        "scientific_role": SCIENTIFIC_ROLE,
        "demand_mainline_vph": DEMAND_MAINLINE_VPH,
        "demand_ramp_vph": DEMAND_RAMP_VPH,
        "cav_penetration": CAV_PENETRATION,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "activation_time_s": ACTIVATION_TIME_S,
        "release_time_s": RELEASE_TIME_S,
        "treatment_duration_s": TREATMENT_DURATION_S,
        "timing_semantics": (
            "wall command is first applied at SUMO time 880 before the 880->881 "
            "transition; its twentieth command advances 899->900; time 900 is "
            "the planned release boundary back to identical damping"
        ),
        "analysis_end_s": ANALYSIS_END_S,
        "timing_selection_provenance": {
            "density_mapper_summary": DENSITY_MAPPER_SUMMARY,
            "density_mapper_summary_sha256": DENSITY_MAPPER_SUMMARY_SHA256,
            "incident_onset_and_confirmation_s": list(DENSITY_CONFIRMATIONS_S),
        },
        "primary_outcome": {
            "role": "prevention_or_shortening_of_sustained_premerge_congestion",
            "queue_count_at_least": PRIMARY_QUEUE_COUNT,
            "premerge_mean_speed_at_most_mps": PRIMARY_SPEED_THRESHOLD_MPS,
            "sustain_s": PRIMARY_SUSTAIN_S,
            "incidence_and_duration_reported": True,
            "recovery_reported": True,
        },
        "secondary_diagnostic": {
            "health_strictly_below": SECONDARY_HEALTH_THRESHOLD,
            "sustain_s": SECONDARY_HEALTH_SUSTAIN_S,
            "not_an_outcome_gate": True,
        },
        "gate_f": {
            "treatment_blind": True,
            "minimum_feasible_seeds": MINIMUM_FEASIBLE_SEEDS,
            "minimum_projected_duration_fraction": MINIMUM_DURATION_FRACTION,
            "require_all_occupied_lanes": True,
            "maximum_initial_and_predicted_span_m": 50.0,
            "minimum_terminal_runway_margin_m": 0.0,
            "require_frozen_front_rear_activation_safety": True,
        },
        "matched_pilot": {
            "common_endpoint_absolute_s": ANALYSIS_END_S,
            "natural_clearance_rule": (
                "terminal before 1080 is accepted only when the runner reports "
                "terminated, the final network vehicle count is zero, and the "
                "terminal expected vehicle count is zero; the empty remainder "
                "through 1080 contributes no congestion, delay, discharge or storage"
            ),
            "primary_window": "post_release_900_through_1080_or_natural_clearance",
            "full_reporting_window": "activation_880_through_1080_or_natural_clearance",
            "minimum_material_shortening_s": MINIMUM_SHORTENING_S,
            "materiality_rule": (
                "prevention or shortening additionally requires at least the "
                "minimum reduction in both raw congestion-predicate seconds and "
                "longest contiguous congestion run"
            ),
            "minimum_realized_duration_fraction": MINIMUM_REALIZED_DURATION_FRACTION,
            "minimum_realized_lane_coverage_fraction": (
                MINIMUM_REALIZED_LANE_COVERAGE_FRACTION
            ),
            "safety_and_service_margins": "idm_escalation_gate_b_safety_service/1",
            "service_failure_counts_as_safe_nonbeneficial_diversity": False,
            "require_all_feasible_service_noninferior_for_expansion": True,
            "minimum_independent_beneficial_seeds": 2,
            "minimum_safe_nonbeneficial_seeds": 2,
            "repeat_seed": REPEAT_SEED,
            "repeat_selection": "lowest treatment-blind admitted development seed",
        },
        "base_control_arm": frozen_idm_analytical_manifest(),
        "wall_profile": profile_manifest(),
        "frozen_treatment_fields": [
            "duration",
            "strength",
            "geometry",
            "release",
            "membership_and_handoff_rules",
        ],
        "treatment_outcomes_read_during_gate_f": False,
        "validation_or_final_seeds_opened": False,
        "predictor_training_authorized": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "contract_fingerprint_sha256": sha256(encoded).hexdigest()}


__all__ = [
    "ACTIVATION_TIME_S",
    "ANALYSIS_END_S",
    "CAV_PENETRATION",
    "DENSITY_CONFIRMATIONS_S",
    "DENSITY_MAPPER_SUMMARY",
    "DENSITY_MAPPER_SUMMARY_SHA256",
    "DEMAND_MAINLINE_VPH",
    "DEMAND_RAMP_VPH",
    "DEVELOPMENT_SEEDS",
    "MINIMUM_DURATION_FRACTION",
    "MINIMUM_FEASIBLE_SEEDS",
    "MINIMUM_REALIZED_DURATION_FRACTION",
    "MINIMUM_REALIZED_LANE_COVERAGE_FRACTION",
    "MINIMUM_SHORTENING_S",
    "PRIMARY_QUEUE_COUNT",
    "PRIMARY_SPEED_THRESHOLD_MPS",
    "PRIMARY_SUSTAIN_S",
    "RELEASE_TIME_S",
    "REPEAT_SEED",
    "SCIENTIFIC_ROLE",
    "SECONDARY_HEALTH_SUSTAIN_S",
    "SECONDARY_HEALTH_THRESHOLD",
    "TREATMENT_DURATION_S",
    "contract_manifest",
]
