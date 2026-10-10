import json

from tools.analyze_idm_p30_operating_domain import analyze


def test_operating_domain_separates_delivered_authority_from_outcome(tmp_path):
    root = tmp_path / "paired"
    (root / "discoveries").mkdir(parents=True)
    (root / "branches").mkdir()
    case_id = "m2200.seed1.t50"
    decision = {
        "decision_index": 1,
        "candidate_observations": {"cav": [0.0, 0.8]},
        "candidates": [
            {
                "slot": 1,
                "vehicle_id": "cav",
                "predicted_dose_s": 24.0,
                "preview_reference_speed_mps": 20.0,
            }
        ],
        "pattern_context": {
            "idm_severe_level": 1.0,
            "idm_wave_energy_level": 2.0,
        },
    }
    (root / "discoveries" / "m2200.seed1.json").write_text(
        json.dumps({"decision": decision}), encoding="utf-8"
    )
    effect = {
        "arm": "candidate1_speed_ceiling",
        "candidate_slot": 1,
        "classification": "harmful",
        "safety_composite_reference_units": -0.10,
        "constraints_pass": True,
        "constraints": {
            "dose_at_least_minimum": True,
            "zero_added_collisions": True,
        },
        "target_command_step_count": 20,
    }
    (root / "summary.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": case_id,
                        "seed": 1,
                        "decision_step": 50,
                        "effects": [effect],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    common = {
        "severe_braking_count_sum": 10.0,
        "acceleration_energy_sum": 100.0,
        "collapse_max_consecutive_steps": 3,
    }
    (root / "branches" / f"{case_id}.abstain.json").write_text(
        json.dumps(common), encoding="utf-8"
    )
    (root / "branches" / f"{case_id}.candidate1_speed_ceiling.json").write_text(
        json.dumps(
            {
                **common,
                "severe_braking_count_sum": 12.0,
                "acceleration_energy_sum": 120.0,
                "projection_survival_values": [0.8, 1.0],
            }
        ),
        encoding="utf-8",
    )

    result = analyze([root])
    group = result["by_demand"]["2200"]
    assert result["development_only"] is True
    assert group["adequate_dose_fraction"] == 1.0
    assert group["command_projection_survival"]["median"] == 0.9
    assert group["qualified_harmful_fraction"] == 1.0
    assert group["terminal_speed_drop_mps"]["median"] == 8.0
