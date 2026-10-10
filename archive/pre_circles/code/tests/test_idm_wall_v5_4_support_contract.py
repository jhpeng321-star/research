import json

from experiments.idm.wall_v5_4_contract import profile_manifest
from experiments.idm.wall_v5_4_support_contract import (
    CAV_PENETRATION,
    SUPPORT_SEEDS,
    availability_rule,
    support_manifest,
    validate_support_seeds,
)
from tools.analyze_idm_wall_v5_4_feasibility import _computed_admission, _reject_outcomes
from experiments.idm.wall_v5_4_p30_support_contract import (
    CAV_PENETRATION as P30_CAV_PENETRATION,
    availability_rule as p30_availability_rule,
    support_manifest as p30_support_manifest,
)


def test_v5_4_profile_is_distinct_and_emergency_handoff_is_explicit() -> None:
    profile = profile_manifest()
    assert profile["contract_version"] == "idm_compact_moving_wall/5.4"
    assert profile["overlay_overrides"]["treatment_integrity_version"] == "v5.4"
    assert profile["emergency_release_rule"]["reported_taper_endpoint"] == 0.0
    assert profile["emergency_release_rule"]["may_interrupt_nominal_five_second_taper"] is True


def test_v5_4_support_cohort_and_hash_are_stable() -> None:
    assert SUPPORT_SEEDS == tuple(range(4333, 4353))
    assert validate_support_seeds(SUPPORT_SEEDS) == SUPPORT_SEEDS
    assert CAV_PENETRATION == 0.25
    assert availability_rule()["minimum_admitted_seed_windows"] == 20
    left = support_manifest()
    right = support_manifest()
    assert left == right
    assert len(left["support_contract_sha256"]) == 64


def test_v5_4_admission_reconstructs_and_rejects_outcomes() -> None:
    row = {
        "formation": {
            "feasible": True,
            "reason": "feasible",
            "longitudinal_span_m": 8.0,
            "predicted_max_span_m": 20.0,
            "occupied_lane_ids": ["lane0", "lane1"],
            "required_lane_ids": ["lane0", "lane1"],
        },
        "lane_coverage": {"selected_lane_ids": ["lane0", "lane1"]},
        "projected_dose": {
            "requested_duration_s": 20.0,
            "projected_duration_fraction": 1.0,
        },
        "activation_safety": {"safe": True},
    }
    assert _computed_admission(row) is True
    _reject_outcomes(json.loads(json.dumps(row)))


def test_v5_4_p30_is_a_treatment_blind_operating_point_amendment() -> None:
    assert P30_CAV_PENETRATION == 0.30
    assert p30_availability_rule()[
        "penetration_frozen_after_p25_availability_nogo_before_any_v5_4_outcome"
    ] == 0.30
    manifest = p30_support_manifest()
    assert manifest["amends"] == "v5.4_p25_availability_nogo_without_treatment_outcomes"
    assert manifest["wall_profile"] == profile_manifest()
