"""Frozen treatment-integrity contract for IDM compact moving-wall v5.3."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from experiments.idm.wall_v5_2_contract import overlay_overrides as v5_2_overrides
from policies.idm_analytical import frozen_idm_analytical_manifest


PROFILE_ID = "adaptive_compact_wall_v5_3_integrity"
CONTRACT_VERSION = "idm_compact_moving_wall/5.3"
STAGE1_CASES = (
    (4301, 660),
    (4302, 620),
    (4307, 680),
    (4308, 680),
    (4309, 620),
    (4311, 700),
    (4312, 660),
)
TAPER_DURATION_S = 5.0
TAPER_FUNCTION = "linear"
MAX_HANDOFFS_PER_LANE = 1
COMPACTNESS_EXCESS_SAMPLES = 2


def overlay_overrides() -> dict[str, Any]:
    """Return v5.2 plus only the owner-approved v5.3 integrity repairs."""

    return {
        **v5_2_overrides(),
        "treatment_integrity_version": "v5.3",
        "compact_handoff_enabled": True,
        "max_handoffs_per_lane": MAX_HANDOFFS_PER_LANE,
        "compactness_enforcement_samples": COMPACTNESS_EXCESS_SAMPLES,
        "taper_duration_s": TAPER_DURATION_S,
        "taper_function": TAPER_FUNCTION,
    }


def _payload() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "profile_id": PROFILE_ID,
        "base_control_arm": frozen_idm_analytical_manifest(),
        "parent_treatment_version": "adaptive_compact_wall_v5_2_prospective",
        "overlay_overrides": overlay_overrides(),
        "state_machine": ["DAMPING", "ASSEMBLING", "HOLDING", "TAPERING", "RELEASED"],
        "integrity_repairs": [
            "prelatched_ranked_same_lane_reserves",
            "at_most_one_handoff_per_lane",
            "no_partial_lane_hold",
            "two_sample_compactness_enforcement",
            "five_second_in_horizon_linear_taper_to_identical_damping",
        ],
        "taper_safety_claim": False,
        "treatment_blind_feasibility_scan_required_before_outcome_pilot": True,
        "stage1_cases": [list(case) for case in STAGE1_CASES],
    }


def profile_manifest() -> dict[str, Any]:
    payload = _payload()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "profile_sha256": sha256(encoded).hexdigest()}

