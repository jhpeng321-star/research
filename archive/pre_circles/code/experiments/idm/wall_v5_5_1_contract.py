"""Frozen v5.5.1 adaptive follower-aware hold/release treatment contract."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from experiments.idm.wall_v5_4_contract import overlay_overrides as v5_4_overrides
from policies.idm_analytical import frozen_idm_analytical_manifest


PROFILE_ID = "adaptive_compact_wall_v5_5_1_follower_aware_release"
CONTRACT_VERSION = "idm_compact_moving_wall/5.5.1"

# These are inherited, pre-existing safety quantities.  They are not fitted on
# the four repair seeds: v5.5.1 merely turns their near-boundary approach into
# a planned, monotone release before the hard 1.5 s / -4.5 m/s2 abort layer.
SOFT_REAR_GAP_M = 20.0
SOFT_REAR_TTC_S = 6.0
SOFT_FOLLOWER_ACCEL_MPS2 = -3.0
SOFT_GAP_LOSS_M = 3.5
SOFT_CLOSING_INCREASE_MPS = 1.0
FOLLOWER_COUNT = 3
FOLLOWER_RANGE_M = 100.0
PERSISTENCE_SAMPLES = 1
RELEASE_DURATION_S = 5.0


def overlay_overrides() -> dict[str, Any]:
    return {
        **v5_4_overrides(),
        "treatment_integrity_version": "v5.5.1",
        "adaptive_release_enabled": True,
        "adaptive_release_soft_rear_gap_m": SOFT_REAR_GAP_M,
        "adaptive_release_soft_rear_ttc_s": SOFT_REAR_TTC_S,
        "adaptive_release_soft_follower_accel_mps2": SOFT_FOLLOWER_ACCEL_MPS2,
        "adaptive_release_soft_gap_loss_m": SOFT_GAP_LOSS_M,
        "adaptive_release_soft_closing_increase_mps": SOFT_CLOSING_INCREASE_MPS,
        "adaptive_release_persistence_samples": PERSISTENCE_SAMPLES,
        "adaptive_release_missing_follower_behavior": "release",
    }


def _payload() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "profile_id": PROFILE_ID,
        "base_control_arm": frozen_idm_analytical_manifest(),
        "parent_treatment_version": "idm_compact_moving_wall/5.4",
        "overlay_overrides": overlay_overrides(),
        "state_machine": ["DAMPING", "ASSEMBLING", "HOLDING", "TAPERING", "RELEASED"],
        "adaptive_release": {
            "follower_chain_per_lane": FOLLOWER_COUNT,
            "maximum_upstream_range_m": FOLLOWER_RANGE_M,
            "soft_trigger_any": {
                "rear_gap_m_at_most": SOFT_REAR_GAP_M,
                "rear_ttc_s_at_most": SOFT_REAR_TTC_S,
                "follower_acceleration_mps2_at_most": SOFT_FOLLOWER_ACCEL_MPS2,
                "identity_change_or_missing": "release",
            },
            "risk_gate": "rear_gap<=20m OR rear_ttc<=6s; then same-follower trend",
            "trend": {
                "gap_loss_m_strictly_greater_than": SOFT_GAP_LOSS_M,
                "closing_increase_mps_strictly_greater_than": SOFT_CLOSING_INCREASE_MPS,
            },
            "persistence_samples": PERSISTENCE_SAMPLES,
            "on_first_soft_sample": "freeze_speed_and_tau_tightening",
            "on_persistent_soft_trigger": "latch_irreversible_linear_taper",
            "release_duration_s": RELEASE_DURATION_S,
            "hard_abort_remains": {
                "follower_ttc_s_below": 1.5,
                "hard_brake_mps2_at_most": -4.5,
                "collision_or_teleport": True,
            },
            "completion_semantics": {
                "adaptive_taper": "policy_complete_reduced_realized_authority",
                "hard_abort": "not_policy_complete",
                "damping_handoff": "exact_frozen_base_controller",
            },
        },
        "taper_safety_claim": False,
    }


def profile_manifest() -> dict[str, Any]:
    payload = _payload()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "profile_sha256": sha256(encoded).hexdigest()}


__all__ = ["CONTRACT_VERSION", "overlay_overrides", "profile_manifest"]
