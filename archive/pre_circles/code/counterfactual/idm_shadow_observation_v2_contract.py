"""Frozen development contract for the v5.4 shadow-observation v2 screen.

This module contains constants only.  It must not load treatment outcomes or
launch SUMO.  Capture and analysis runners bind their manifests to
``contract_manifest()`` so a resumed run cannot silently change the scientific
question.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from experiments.idm.wall_v5_4_predictor_contract import (
    DECISION_STEPS,
    DEVELOPMENT_SEEDS,
)


CONTRACT_VERSION = "idm_v5_4_shadow_observation_v2/1"
CAPTURE_SCHEMA = "idm_v5_4_shadow_observation_v2_capture/1"
SEED_SCHEMA = "idm_v5_4_shadow_observation_v2_seed/1"
ANALYSIS_SCHEMA = "idm_v5_4_shadow_observation_v2_analysis/1"

SOURCE_LABEL_ROOT = Path(
    "runs/eval/idm_wall_v5_4_predictor_development_p30_s4401_4520"
)
SOURCE_FEASIBILITY_ROOT = Path(
    "runs/probe/idm_wall_v5_4_predictor_development_feasibility_p30_s4401_4520"
)
DEVELOPMENT_ROOT = Path(
    "runs/eval/idm_wall_v5_4_shadow_observation_v2_development"
)
CAPTURE_ROOT = DEVELOPMENT_ROOT / "capture"
ANALYSIS_ROOT = DEVELOPMENT_ROOT / "analysis"

SOURCE_SHA256 = {
    "label_manifest": "7c4a83202d5e11cef259eea885ee5bc1254cf8d4aab23822a217a425b86fb62e",
    "label_summary": "79397267b77de354c54cc9848263258c4e3b5f57f324b3d5a8929c3ae1756e79",
    "feasibility_analysis": "f620186a2ad81e406b5d545b37bc9656d5a9aae1e3300f4223fdcf112c5bedc6",
    "v1_diagnosis": "f66ac74ee72dd097024ecf0a85a761ea2aced398440f45eb59547a7dd31c3bde",
}

SHADOW_HORIZON_S = 10
SHADOW_REPORT_TIMES_S = (1, 2, 5, 10)
HISTORY_CAPTURE_S = 60
PRIMARY_HISTORY_S = 20
HISTORY_SAMPLE_INTERVAL_S = 1
OUTER_FOLDS = 5
INNER_FOLDS = 4
RANDOM_STATE = 20260831
MAX_WORKERS = 4
PROGRESS_EVERY_SEEDS = 10

TARGETS = {
    "dose": {
        "population": "all_gate_f_admissions",
        "positive": "realized_duration_fraction_at_least_0.80",
    },
    "safety": {
        "population": "dose_qualified_gate_f_admissions",
        "positive": "frozen_safety_admissible",
    },
    "benefit": {
        "population": "safe_dose_qualified_gate_f_admissions",
        "positive": "frozen_joint_benefit",
    },
}

FEATURE_BLOCKS = (
    "engineered_temporal_baseline",
    "sectional_transitional_instability",
    "merge_pressure_breakdown_state",
    "trajectory_physical_20s",
    "proposed_wall_shadow_10s",
    "trajectory_plus_shadow",
)
TABULAR_MODEL_FAMILIES = (
    "regularized_logistic",
    "additive_spline_logistic",
    "shallow_gradient_boosting",
)
SEQUENCE_MODEL_FAMILY = "small_regularized_temporal_convolution"
SECONDARY_EFFECT_MODEL = "small_direct_paired_effect_regressor"

CONFIRMATORY_GATES = {
    "minimum_matched_auc_gain": 0.03,
    "minimum_selected_cases": 10,
    "minimum_selected_seeds": 8,
    "minimum_selected_safety_rate": 0.90,
    "minimum_joint_benefit_precision": 0.50,
    "minimum_selected_dose_success_rate": 0.80,
    "maximum_selected_excess_collisions": 0,
    "maximum_selected_excess_teleports": 0,
}
EXPLORATORY_REPORTING_FLOOR = {
    "minimum_selected_safety_rate": 0.80,
    "minimum_joint_benefit_precision": 0.45,
}

# Tokens are checked recursively in exported model-input names and provenance.
PROHIBITED_INPUT_TOKENS = (
    "seed",
    "vehicle_id",
    "absolute_vehicle_identity",
    "realized_duration",
    "actual_duration",
    "completion_reason",
    "release_reason",
    "abort_reason",
    "outcome",
    "oracle",
    "counterfactual_effect",
    "future_trajectory",
    "postdecision",
    "collision_outcome",
    "teleport_outcome",
    "wall_selected",
)


def contract_payload() -> dict[str, Any]:
    """Return the canonical JSON-safe contract without its own digest."""
    seeds = tuple(int(seed) for seed in DEVELOPMENT_SEEDS)
    decisions = tuple(int(step) for step in DECISION_STEPS)
    if seeds != tuple(range(4401, 4521)):
        raise AssertionError("development partition changed")
    if decisions != (620, 660, 680, 700):
        raise AssertionError("decision grid changed")
    return {
        "contract_version": CONTRACT_VERSION,
        "source_seeds": list(seeds),
        "decision_steps": list(decisions),
        "source_label_root": SOURCE_LABEL_ROOT.as_posix(),
        "source_feasibility_root": SOURCE_FEASIBILITY_ROOT.as_posix(),
        "source_sha256": dict(SOURCE_SHA256),
        "shadow": {
            "method": "deterministic_time_zero_exact_sumo_twin_replay",
            "horizon_s": SHADOW_HORIZON_S,
            "report_times_s": list(SHADOW_REPORT_TIMES_S),
            "common_episode_horizon_controller_steps": 710,
            "control": "frozen_idm_analytical_damping",
            "treatment": "identical_damping_plus_frozen_v5_4_wall",
            "prefix_identity_required": True,
            "lane_changes": "native_sumo_idm_lc2013_unchanged",
            "rng": "same_seed_route_and_prefix_random_history",
        },
        "history": {
            "captured_s": HISTORY_CAPTURE_S,
            "primary_screen_s": PRIMARY_HISTORY_S,
            "sample_interval_s": HISTORY_SAMPLE_INTERVAL_S,
            "sixty_second_sequence_is_conditional": True,
        },
        "evaluation": {
            "group_unit": "complete_seed",
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "random_state": RANDOM_STATE,
            "targets": TARGETS,
            "feature_blocks": list(FEATURE_BLOCKS),
            "tabular_models": list(TABULAR_MODEL_FAMILIES),
            "sequence_model": SEQUENCE_MODEL_FAMILY,
            "secondary_effect_model": SECONDARY_EFFECT_MODEL,
            "confirmatory_gates": dict(CONFIRMATORY_GATES),
            "exploratory_reporting_floor": dict(EXPLORATORY_REPORTING_FLOOR),
        },
        "execution": {
            "maximum_workers": MAX_WORKERS,
            "progress_every_seeds": PROGRESS_EVERY_SEEDS,
            "atomic_per_seed": True,
            "resumable": True,
            "capture_all_decisions": 480,
            "shadow_gate_f_admissions_only": 199,
        },
        "prohibited_input_tokens": list(PROHIBITED_INPUT_TOKENS),
        "sealed_partitions": {
            "existing_validation": list(range(4601, 4661)),
            "final_paper": list(range(5001, 5061)),
        },
    }


def contract_manifest() -> dict[str, Any]:
    payload = contract_payload()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "contract_sha256": sha256(encoded).hexdigest()}


__all__ = [
    "ANALYSIS_ROOT",
    "ANALYSIS_SCHEMA",
    "CAPTURE_ROOT",
    "CAPTURE_SCHEMA",
    "CONFIRMATORY_GATES",
    "CONTRACT_VERSION",
    "DEVELOPMENT_ROOT",
    "HISTORY_CAPTURE_S",
    "HISTORY_SAMPLE_INTERVAL_S",
    "INNER_FOLDS",
    "MAX_WORKERS",
    "OUTER_FOLDS",
    "PRIMARY_HISTORY_S",
    "PROHIBITED_INPUT_TOKENS",
    "PROGRESS_EVERY_SEEDS",
    "RANDOM_STATE",
    "SEED_SCHEMA",
    "SHADOW_HORIZON_S",
    "SHADOW_REPORT_TIMES_S",
    "SOURCE_FEASIBILITY_ROOT",
    "SOURCE_LABEL_ROOT",
    "SOURCE_SHA256",
    "contract_manifest",
    "contract_payload",
]
