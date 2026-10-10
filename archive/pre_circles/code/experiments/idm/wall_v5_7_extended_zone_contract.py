"""Frozen v5.7 extended-control-zone low-authority wall contract.

This is deliberately a new mechanism experiment.  It does not reinterpret
the v5.6.2 negative result and it contains no predictor or outcome-selected
field.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from experiments.idm.wall_v5_4_contract import overlay_overrides as v5_4_overrides
from policies.idm_analytical import frozen_idm_analytical_manifest


PROFILE_ID = "idm_v5_7_extended_zone_low_authority_wall"
CONTRACT_VERSION = "idm_compact_moving_wall/5.7-extended-zone-low-authority"
UPSTREAM_EXTENSION_EDGE = "idm_v5_7_upstream_extension"
CURRENT_PREMERGE_LENGTH_M = 2644.22
UPSTREAM_EXTENSION_M = 900.0
WALL_BAND_LO_M = 200.0
WALL_BAND_HI_M = 1900.0
MAX_DECELERATION_MPS2 = 0.3
ASSEMBLY_MAX_S = 10.0
ASSEMBLY_MIN_STEPS = 4
HOLD_S = 40.0
TAPER_S = 8.0
TOTAL_TREATMENT_S = ASSEMBLY_MAX_S + HOLD_S + TAPER_S
POST_RELEASE_OBSERVATION_S = 120
EVENT_HORIZON_S = int(TOTAL_TREATMENT_S + POST_RELEASE_OBSERVATION_S)


def overlay_overrides() -> dict[str, Any]:
    """Return the one frozen v5.7 profile without an inherited high-authority phase."""
    return {
        **v5_4_overrides(),
        "treatment_integrity_version": "v5.7",
        "approach_edge_distance_offsets_m": {
            UPSTREAM_EXTENSION_EDGE: CURRENT_PREMERGE_LENGTH_M,
        },
        "wall_params": {
            "d_lo": WALL_BAND_LO_M,
            "d_hi": WALL_BAND_HI_M,
            "wall_speed": 10.0,
            "wall_tau": 3.0,
            "a_max_dec": MAX_DECELERATION_MPS2,
            "tau_ramp_step": 0.05,
        },
        "geometry_mode": "compact_moving",
        "max_longitudinal_span_m": 50.0,
        "planned_treatment_duration_s": TOTAL_TREATMENT_S,
        "compact_release_boundary_m": 50.0,
        "compact_selection_mode": "trajectory_coherent",
        "compact_assembly_duration_s": ASSEMBLY_MAX_S,
        "compact_min_assembly_steps": ASSEMBLY_MIN_STEPS,
        "compact_requested_hold_duration_s": HOLD_S,
        "compact_assembly_speed_tolerance_mps": 2.0,
        "compact_assembly_a_max_dec_mps2": MAX_DECELERATION_MPS2,
        "compact_control_mode": "adaptive_servo",
        "compact_servo_a_max_dec_mps2": MAX_DECELERATION_MPS2,
        "compact_initial_assembly_servo_a_max_dec_mps2": MAX_DECELERATION_MPS2,
        "compact_high_authority_assembly_steps": 0,
        "compact_settling_max_steps": 0,
        "compact_handoff_enabled": True,
        "max_handoffs_per_lane": 1,
        "compactness_enforcement_samples": 2,
        "taper_duration_s": TAPER_S,
        "taper_function": "linear",
        "adaptive_release_enabled": False,
        "diagnostic_formation_preview": True,
    }


def profile_manifest() -> dict[str, Any]:
    payload = {
        "contract_version": CONTRACT_VERSION,
        "profile_id": PROFILE_ID,
        "base_control_arm": frozen_idm_analytical_manifest(),
        "parent_treatment_version": "idm_compact_moving_wall/5.6.2-phase-split-authority-settling",
        "overlay_overrides": overlay_overrides(),
        "geometry": {
            "derived_upstream_extension_m": UPSTREAM_EXTENSION_M,
            "eligible_distance_to_merge_m": [WALL_BAND_LO_M, WALL_BAND_HI_M],
            "downstream_merge_and_bottleneck": "unchanged",
        },
        "authority": {
            "all_assembly_hold_taper_cap_mps2": MAX_DECELERATION_MPS2,
            "high_authority_assembly_steps": 0,
            "minimum_low_authority_assembly_steps": ASSEMBLY_MIN_STEPS,
        },
        "requested_lifecycle_s": {
            "assembly_max": ASSEMBLY_MAX_S,
            "hold": HOLD_S,
            "taper": TAPER_S,
            "total_window": TOTAL_TREATMENT_S,
            "post_release_observation": POST_RELEASE_OBSERVATION_S,
        },
        "selection": "treatment-blind first full-lane safety-admissible distance-band state",
        "outcome_selection": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "profile_sha256": sha256(encoded).hexdigest()}
