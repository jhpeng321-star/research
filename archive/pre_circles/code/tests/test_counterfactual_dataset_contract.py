import pytest

from counterfactual.dataset_contract import (
    CONTRACT_VERSION,
    MIN_MODEL_DEVELOPMENT_GROUPS,
    MIN_WALL_SELECTED_CASES,
    MIN_WALL_SELECTED_GROUPS,
    cell_for,
    contract_manifest,
    validate_assigned_seeds,
)


def test_primary_hyper_p20_blocks_are_disjoint_and_frozen():
    cell = cell_for("hyper", 0.20)
    assert cell.model_development_seeds == tuple(range(1081, 1101))
    assert cell.locked_test_seeds == tuple(range(2041, 2051))
    assert set(cell.model_development_seeds).isdisjoint(cell.locked_test_seeds)


def test_model_role_rejects_protocol_and_out_of_cell_seeds():
    for seed in (123, 205, 404, 506, 601, 701, 801):
        with pytest.raises(ValueError, match="cannot enter"):
            validate_assigned_seeds(
                role="model_development",
                demand="hyper",
                cav_penetration=0.20,
                seeds=[seed],
            )
    with pytest.raises(ValueError, match="outside"):
        validate_assigned_seeds(
            role="model_development",
            demand="hyper",
            cav_penetration=0.20,
            seeds=[1001],
        )


def test_frozen_headroom_seeds_are_allowed_only_for_the_headroom_role():
    assert validate_assigned_seeds(
        role="headroom_gate",
        demand="hyper",
        cav_penetration=0.20,
        seeds=[601, 602],
    ) == (601, 602)


def test_model_role_accepts_resumable_subset_and_manifest_freezes_fit_gate():
    assert validate_assigned_seeds(
        role="model_development",
        demand="hyper",
        cav_penetration=0.20,
        seeds=[1081, 1082],
    ) == (1081, 1082)
    gate = contract_manifest()["model_fit_adequacy_gate"]
    assert CONTRACT_VERSION == "counterfactual_escalation_v2_dataset/5"
    amendment = contract_manifest()["prospective_headroom_amendment"]
    assert amendment["primary_estimand"] == "safety_service"
    assert amendment["collapse_prevention_role"] == "secondary_reported_outcome"
    assert amendment["global_safety_gate_unchanged"] is True
    assert amendment["predecision_supervisor"] == "local_support_density_coherence_v1"
    assert (
        contract_manifest()["outcome_oracle_amendment"][
            "original_delay_oracle_retained_as_sensitivity"
        ]
        is True
    )
    assert gate["minimum_independent_development_groups"] == MIN_MODEL_DEVELOPMENT_GROUPS
    assert gate["minimum_wall_selected_independent_groups"] == MIN_WALL_SELECTED_GROUPS
    assert gate["minimum_wall_selected_cases"] == MIN_WALL_SELECTED_CASES
