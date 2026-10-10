"""Frozen Candidate-A low-authority compact-wall treatment contract."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from experiments.idm.wall_v5_4_contract import overlay_overrides as v5_4_overrides
from policies.idm_analytical import frozen_idm_analytical_manifest


PROFILE_ID = "adaptive_compact_wall_v5_6_a_low_authority"
CONTRACT_VERSION = "idm_compact_moving_wall/5.6.a-low-authority"
DECELERATION_CAP_MPS2 = 0.3


def overlay_overrides() -> dict[str, Any]:
    """Return v5.4 with the audited active servo limits reduced to 0.3 m/s2."""

    return {
        **v5_4_overrides(),
        "treatment_integrity_version": "v5.6.a",
        # Both values are required: adaptive formation projection and the
        # runtime compact actuator use the servo envelope.
        "compact_assembly_a_max_dec_mps2": DECELERATION_CAP_MPS2,
        "compact_servo_a_max_dec_mps2": DECELERATION_CAP_MPS2,
    }


def profile_manifest() -> dict[str, Any]:
    payload = {
        "contract_version": CONTRACT_VERSION,
        "profile_id": PROFILE_ID,
        "base_control_arm": frozen_idm_analytical_manifest(),
        "parent_treatment_version": "idm_compact_moving_wall/5.4",
        "overlay_overrides": overlay_overrides(),
        "single_change": {
            "field_units": "m/s2",
            "compact_assembly_a_max_dec_mps2": {"from": 1.0, "to": DECELERATION_CAP_MPS2},
            "compact_servo_a_max_dec_mps2": {"from": 1.0, "to": DECELERATION_CAP_MPS2},
        },
        "unchanged": [
            "compact geometry and membership",
            "activation time and 20-second maximum window",
            "adaptive speed reference and tau law",
            "jerk and safety projection",
            "same-lane handoff",
            "five-second linear taper and exact damping return",
            "no support CAV and no second wall",
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "profile_sha256": sha256(encoded).hexdigest()}


__all__ = ["CONTRACT_VERSION", "DECELERATION_CAP_MPS2", "overlay_overrides", "profile_manifest"]
