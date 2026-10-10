from experiments.idm.wall_v5_2_contract import overlay_overrides as v5_2_overrides
from experiments.idm.wall_v5_3_contract import (
    CONTRACT_VERSION,
    STAGE1_CASES,
    overlay_overrides,
    profile_manifest,
)


def test_v5_3_changes_only_the_frozen_integrity_fields() -> None:
    parent = v5_2_overrides()
    treatment = overlay_overrides()
    additions = {
        key: value for key, value in treatment.items() if key not in parent
    }
    assert {key: treatment[key] for key in parent} == parent
    assert additions == {
        "treatment_integrity_version": "v5.3",
        "compact_handoff_enabled": True,
        "max_handoffs_per_lane": 1,
        "compactness_enforcement_samples": 2,
        "taper_duration_s": 5.0,
        "taper_function": "linear",
    }


def test_v5_3_manifest_is_stable_and_requires_blind_scan() -> None:
    first = profile_manifest()
    assert first == profile_manifest()
    assert first["contract_version"] == CONTRACT_VERSION
    assert first["taper_safety_claim"] is False
    assert first["treatment_blind_feasibility_scan_required_before_outcome_pilot"] is True
    assert tuple(tuple(case) for case in first["stage1_cases"]) == STAGE1_CASES
    assert len(first["profile_sha256"]) == 64
