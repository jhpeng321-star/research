"""Frozen v5.7.2 upstream-convergent, low-authority compact-wall treatment."""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from experiments.idm.wall_v5_7_1_original_corridor_contract import (
    CAV_PENETRATION, DECISION_ABSOLUTE_S, DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH, POST_RELEASE_S, SEEDS, WALL_BAND_HI_M, WALL_BAND_LO_M,
)
from experiments.idm.wall_v5_4_contract import overlay_overrides as v5_4_overrides
from policies.idm_analytical import frozen_idm_analytical_manifest


PROFILE_ID = "idm_v5_7_2_upstream_convergent_assembly_wall"
CONTRACT_VERSION = "idm_compact_moving_wall/5.7.2-upstream-convergent-assembly"
MAX_DECELERATION_MPS2 = 0.3
MAX_ASSEMBLY_S = 10.0
COMPACT_READINESS_SAMPLES = 3
COMPACT_SPAN_M = 50.0
COMPACT_SPEED_SPREAD_MPS = 0.5
HOLD_S = 40.0
TAPER_S = 8.0
TOTAL_TREATMENT_S = 58.0
OUTCOME_HORIZON_S = TOTAL_TREATMENT_S + POST_RELEASE_S
MATERIAL_EFFECT_S = 10


def overlay_overrides() -> dict[str, Any]:
    return {
        **v5_4_overrides(),
        "treatment_integrity_version": "v5.7.2",
        "wall_params": {"d_lo": WALL_BAND_LO_M, "d_hi": WALL_BAND_HI_M, "wall_speed": 10.0, "wall_tau": 3.0, "a_max_dec": MAX_DECELERATION_MPS2, "tau_ramp_step": 0.05},
        "geometry_mode": "compact_moving",
        "max_longitudinal_span_m": COMPACT_SPAN_M,
        "planned_treatment_duration_s": TOTAL_TREATMENT_S,
        "compact_release_boundary_m": COMPACT_SPAN_M,
        "compact_selection_mode": "upstream_convergent",
        "compact_assembly_duration_s": MAX_ASSEMBLY_S,
        "compact_min_assembly_steps": 0,
        "compact_requested_hold_duration_s": HOLD_S,
        "compact_assembly_speed_tolerance_mps": COMPACT_SPEED_SPREAD_MPS,
        "compact_readiness_samples": COMPACT_READINESS_SAMPLES,
        "compact_enforce_span_during_assembly": False,
        "compact_assembly_a_max_dec_mps2": MAX_DECELERATION_MPS2,
        "compact_control_mode": "adaptive_servo",
        "compact_servo_a_max_dec_mps2": MAX_DECELERATION_MPS2,
        "compact_initial_assembly_servo_a_max_dec_mps2": MAX_DECELERATION_MPS2,
        "compact_high_authority_assembly_steps": 0,
        "compact_settling_max_steps": 0,
        "compact_handoff_enabled": True, "max_handoffs_per_lane": 1,
        "compactness_enforcement_samples": 2,
        "taper_duration_s": TAPER_S, "taper_function": "linear",
        "adaptive_release_enabled": False,
        "diagnostic_formation_preview": "decision_scoped",
    }


def profile_manifest() -> dict[str, Any]:
    payload = {
        "contract_version": CONTRACT_VERSION, "profile_id": PROFILE_ID,
        "base_control_arm": frozen_idm_analytical_manifest(),
        "scenario": {"original_verified_long_idm_corridor": True, "derived_network_used": False, "mainline_vph": DEMAND_MAINLINE_VPH, "ramp_vph": DEMAND_RAMP_VPH, "cav_penetration": CAV_PENETRATION},
        "seeds": list(SEEDS), "decision_absolute_s": DECISION_ABSOLUTE_S,
        "band_distance_to_merge_m": [WALL_BAND_LO_M, WALL_BAND_HI_M],
        "overlay_overrides": overlay_overrides(),
        "lifecycle": {"maximum_assembly_s": MAX_ASSEMBLY_S, "compact_readiness_samples": COMPACT_READINESS_SAMPLES, "requested_hold_s": HOLD_S, "taper_s": TAPER_S, "maximum_total_s": TOTAL_TREATMENT_S, "post_release_s": POST_RELEASE_S, "outcome_horizon_s": OUTCOME_HORIZON_S},
        "selection": "only at frozen t=880; treatment-blind upstream-convergent full-lane safety feasibility",
        "assembly_is_compact_wall_dose": False,
        "outcome_selected_timing": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "profile_sha256": sha256(encoded).hexdigest()}
