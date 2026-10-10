"""Frozen v5.6.2 one-step assembly / jerk-settling / low-authority wall."""
from __future__ import annotations
from hashlib import sha256
import json
from typing import Any
from experiments.idm.wall_v5_4_contract import overlay_overrides as v5_4_overrides
from policies.idm_analytical import frozen_idm_analytical_manifest

PROFILE_ID = "adaptive_compact_wall_v5_6_2_phase_split_authority_settling"
CONTRACT_VERSION = "idm_compact_moving_wall/5.6.2-phase-split-authority-settling"

def overlay_overrides() -> dict[str, Any]:
    return {**v5_4_overrides(), "treatment_integrity_version": "v5.6.2",
        "compact_initial_assembly_servo_a_max_dec_mps2": 1.0,
        "compact_high_authority_assembly_steps": 1,
        "compact_settling_jerk_limit_mps3": 0.5,
        "compact_settling_max_steps": 2,
        "compact_servo_a_max_dec_mps2": 0.3,
        "compact_assembly_a_max_dec_mps2": 1.0}

def profile_manifest() -> dict[str, Any]:
    payload = {"contract_version": CONTRACT_VERSION, "profile_id": PROFILE_ID,
        "base_control_arm": frozen_idm_analytical_manifest(),
        "parent_treatment_version": "idm_compact_moving_wall/5.4",
        "overlay_overrides": overlay_overrides(),
        "authority_schedule": {
            "assembling": {"cap_mps2": 1.0, "max_command_steps": 1, "failure": "exact_damping"},
            "settling": {"jerk_limit_mps3": 0.5, "max_command_steps": 2, "actual_command_based": True, "tau_tightening": False},
            "holding": {"cap_mps2": 0.3},
            "tapering": {"cap_mps2": 0.3, "monotone_to": "exact_damping"},
        },
        "unchanged": ["t880 decision", "single compact full-lane wall", "membership/handoff", "20-second window", "five-second linear taper", "front/rear safety", "common t1080 horizon"]}
    encoded=json.dumps(payload,sort_keys=True,separators=(",", ":")).encode()
    return {**payload,"profile_sha256":sha256(encoded).hexdigest()}
