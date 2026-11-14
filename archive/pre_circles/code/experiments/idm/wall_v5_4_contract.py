"""Frozen treatment-integrity contract for IDM compact moving-wall v5.4."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from experiments.idm.wall_v5_3_contract import overlay_overrides as v5_3_overrides
from policies.idm_analytical import frozen_idm_analytical_manifest


PROFILE_ID = "adaptive_compact_wall_v5_4_emergency_handoff"
CONTRACT_VERSION = "idm_compact_moving_wall/5.4"
TAPER_DURATION_S = 5.0
TAPER_FUNCTION = "linear"


def overlay_overrides() -> dict[str, Any]:
    """Return v5.3 plus the explicit zero-authority emergency handoff."""

    return {
        **v5_3_overrides(),
        "treatment_integrity_version": "v5.4",
    }


def _payload() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "profile_id": PROFILE_ID,
        "base_control_arm": frozen_idm_analytical_manifest(),
        "parent_treatment_version": "idm_compact_moving_wall/5.3",
        "overlay_overrides": overlay_overrides(),
        "state_machine": ["DAMPING", "ASSEMBLING", "HOLDING", "TAPERING", "RELEASED"],
        "integrity_repairs": [
            "prelatched_ranked_same_lane_reserves",
            "at_most_one_handoff_per_lane",
            "no_partial_lane_hold",
            "two_sample_compactness_enforcement",
            "five_second_in_horizon_linear_taper_to_identical_damping",
            "hard_safety_abort_records_explicit_zero_authority_handoff",
        ],
        "emergency_release_rule": {
            "trigger": "frozen_runtime_hard_safety_abort",
            "behavior": "immediate_identical_damping_no_wall_actuation",
            "reported_taper_endpoint": 0.0,
            "may_interrupt_nominal_five_second_taper": True,
        },
        "taper_safety_claim": False,
    }


def profile_manifest() -> dict[str, Any]:
    payload = _payload()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "profile_sha256": sha256(encoded).hexdigest()}


__all__ = ["CONTRACT_VERSION", "overlay_overrides", "profile_manifest"]
