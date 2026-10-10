from tools.relabel_idm_v5_4_gate_b import relabel_seed_payload


def _outcome(*, discharge=100, hard_steps=10, onsets=5):
    return {
        "incremental_delay_s": 1000.0,
        "queue_area_vehicle_s": 100.0,
        "merge_discharge_count": discharge,
        "hard_brake_vehicle_steps": hard_steps,
        "hard_brake_attribution": {"event_start_count": onsets},
        "low_ttc_exposure_steps": 0,
        "collision_unique_vehicle_count": 0,
        "starting_teleport_unique_vehicle_count": 0,
    }


def test_relabel_changes_only_derived_oracle_fields() -> None:
    row = {
        "case_id": "case",
        "arms": {
            "0": {"outcomes": _outcome(), "wall_engaged": False},
            "1": {"outcomes": _outcome(discharge=103), "wall_engaged": True},
        },
        "features": {"x": 1.0},
        "safety_service_oracle": {"legacy": True},
    }
    payload = {"cases": [row]}
    relabel_seed_payload(payload)
    assert row["features"] == {"x": 1.0}
    assert row["arms"]["1"]["outcomes"]["merge_discharge_count"] == 103
    assert row["safety_service_oracle"]["label_contract_version"].endswith("/1")
    assert row["safety_service_oracle"]["wall_selected"] is True
