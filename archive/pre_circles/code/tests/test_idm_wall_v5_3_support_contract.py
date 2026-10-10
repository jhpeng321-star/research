from __future__ import annotations

import pytest

from experiments.idm.wall_v5_3_contract import CONTRACT_VERSION
from experiments.idm.wall_v5_3_support_contract import (
    SUPPORT_SEEDS,
    availability_rule,
    support_manifest,
    validate_support_seeds,
)
from tools.analyze_idm_wall_v5_3_feasibility import _computed_admission


def test_support_cohort_and_manifest_are_frozen() -> None:
    assert SUPPORT_SEEDS == tuple(range(4313, 4333))
    assert validate_support_seeds(SUPPORT_SEEDS) == SUPPORT_SEEDS
    with pytest.raises(ValueError):
        validate_support_seeds(SUPPORT_SEEDS[:-1])
    manifest = support_manifest()
    assert manifest["wall_profile"]["contract_version"] == CONTRACT_VERSION
    assert manifest["availability_rule"] == availability_rule()
    assert len(manifest["support_contract_sha256"]) == 64


def test_feasibility_admission_reconstructs_from_predecision_fields() -> None:
    row = {
        "formation": {
            "feasible": True,
            "reason": "feasible",
            "occupied_lane_ids": ["a", "b"],
            "required_lane_ids": ["a", "b"],
            "longitudinal_span_m": 20.0,
            "predicted_max_span_m": 30.0,
        },
        "lane_coverage": {"selected_lane_ids": ["a", "b"]},
        "projected_dose": {
            "requested_duration_s": 20.0,
            "projected_duration_fraction": 1.0,
        },
        "activation_safety": {"safe": True},
    }
    assert _computed_admission(row)["admitted"] is True
    row["formation"]["predicted_max_span_m"] = 51.0
    assert _computed_admission(row)["admitted"] is False

