from counterfactual.idm_gate_labels import (
    CONTRACT_VERSION,
    classify_idm_gate_b_case,
    contract_manifest,
)


def _outcome(*, delay=1000.0, discharge=100, hard_steps=20, hard_onsets=10, low_ttc=4):
    return {
        "incremental_delay_s": delay,
        "queue_area_vehicle_s": 500.0,
        "merge_discharge_count": discharge,
        "hard_brake_vehicle_steps": hard_steps,
        "hard_brake_attribution": {"event_start_count": hard_onsets},
        "low_ttc_exposure_steps": low_ttc,
        "collision_unique_vehicle_count": 0,
        "starting_teleport_unique_vehicle_count": 0,
    }


def _row(control, treatment, *, engaged=True):
    return {
        "arms": {
            "0": {"outcomes": control, "wall_engaged": False},
            "1": {"outcomes": treatment, "wall_engaged": engaged},
        }
    }


def test_safety_improvement_is_meaningful_with_delay_only_as_guardrail() -> None:
    result = classify_idm_gate_b_case(
        _row(
            _outcome(),
            _outcome(delay=1100.0, hard_steps=14, hard_onsets=7),
        )
    )
    assert result["safety_benefit"] is True
    assert result["traffic_noninferior"] is True
    assert result["wall_selected"] is True
    assert result["delay_is_standalone_benefit"] is False


def test_delay_gain_alone_does_not_select_wall_and_queue_is_diagnostic() -> None:
    result = classify_idm_gate_b_case(
        _row(_outcome(), _outcome(delay=500.0))
    )
    assert result["meaningful_benefit"] is False
    assert result["wall_selected"] is False
    assert result["queue_area_is_selection_criterion"] is False


def test_discharge_gain_is_meaningful_subject_to_safety_and_service() -> None:
    result = classify_idm_gate_b_case(
        _row(_outcome(), _outcome(discharge=103))
    )
    assert result["throughput_benefit"] is True
    assert result["wall_selected"] is True


def test_gate_b_contract_is_versioned() -> None:
    manifest = contract_manifest()
    assert manifest["contract_version"] == CONTRACT_VERSION
    assert manifest["delay_is_standalone_benefit"] is False
    assert manifest["queue_area_is_selection_criterion"] is False
