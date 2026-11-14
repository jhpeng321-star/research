import json

import pytest

from tools.analyze_idm_damping_authority_map import analyze, roots_from_grid_summary


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _refs(tmp_path):
    path = tmp_path / "refs.json"
    _write_json(
        path,
        {
            "refs": {
                "m2000": {
                    "demand_mainline_vph": 2000,
                    "corridor_severe_braking_count_ref_per_s": 2.0,
                    "corridor_acceleration_energy_sum_ref_m2ps4": 10.0,
                }
            }
        },
    )
    return path


def _branch(*, severe, energy, control=False, replay_offset=0.0):
    spatial = {
        level: {
            "severe_braking_count": 0.0,
            "low_ttc_count": 0.0,
            "integrated_negative_acceleration_mps": 0.0,
        }
        for level in ("corridor", "immediate_follower")
    }
    spatial["corridor"]["severe_braking_count"] = float(severe)
    return {
        "decision_identity_with_discovery": True,
        "control_command_count": 0,
        "completed_horizon_steps": 60,
        "severe_braking_count_sum": float(severe) + replay_offset,
        "acceleration_energy_sum": float(energy),
        "ttc_count_sum": 3,
        "low_ttc_exposure_steps": 2,
        "mean_corridor_delay_rate": 0.25,
        "arrived_vehicles": 50,
        "merge_discharge_count": 20,
        "collision_observations": 0,
        "teleport_observations": 0,
        "collapse_max_consecutive_steps": 7,
        "target_command_step_count": 0 if control else 20,
        "command_rows": (
            []
            if control
            else [
                {
                    "nominal_acceleration_mps2": 0.4,
                    "desired_acceleration_mps2": -0.6,
                    "projected_acceleration_mps2": -0.1,
                }
            ]
        ),
        "projection_survival_values": [] if control else [0.5],
        "spatial_totals": spatial,
    }


def _source(tmp_path, refs, *, penetration=0.10, seeds=(4121, 4122)):
    root = tmp_path / f"source_{penetration:.2f}_{seeds[0]}"
    case_id = "m2000.seed4121.t100"
    decision = {
        "decision_index": 3,
        "decision_step": 100,
        "landmark": "wave_onset",
        "lane_id": "mainline_1",
        "candidates": [
            {
                "slot": 1,
                "vehicle_id": "cav.1",
                "lane_id": "mainline_1",
                "predicted_dose_s": 22.0,
                "runway_m": 600.0,
                "distance_to_merge_m": 500.0,
            }
        ],
        "pattern_context": {
            "idm_severe_level": 1.2,
            "idm_wave_energy_level": 1.4,
            "idm_severe_short_ema": 1.1,
            "idm_wave_short_ema": 1.3,
            "idm_severe_trend": 0.2,
            "idm_wave_energy_trend": 0.3,
        },
    }
    _write_json(root / "discoveries" / "m2000.seed4121.json", {"decision": decision})
    control = _branch(severe=10, energy=100, control=True)
    treatment = _branch(severe=6, energy=70)
    _write_json(root / "branches" / f"{case_id}.abstain.json", control)
    _write_json(root / "branches" / f"{case_id}.abstain_replay.json", control)
    _write_json(
        root / "branches" / f"{case_id}.candidate1_speed_ceiling.json", treatment
    )
    effect = {
        "arm": "candidate1_speed_ceiling",
        "candidate_slot": 1,
        "classification": "helpful",
        "safety_composite_reference_units": 0.10,
        "constraints_pass": True,
        "constraints": {
            "complete_horizon": True,
            "dose_at_least_minimum": True,
        },
        "target_command_step_count": 20,
    }
    refs_sha = __import__("hashlib").sha256(refs.read_bytes()).hexdigest()
    _write_json(
        root / "summary.json",
        {
            "protocol_hash": "protocol-hash",
            "protocol": {
                "schema": "idm_decentralized_authority/1",
                "development_only": True,
                "action_set": "activation_speed_ceiling_v1",
                "decision_index": 3,
                "option_steps": 20,
                "safety_horizon_steps": 60,
                "service_horizon_steps": 60,
                "candidate_ranking": "runway",
                "candidate_count": 1,
                "landmark": "risk_onset",
                "reward_refs_sha256": refs_sha,
                "demand": 2000,
                "cav_penetration": penetration,
                "seeds": list(seeds),
            },
            "cases": [
                {
                    "case_id": case_id,
                    "seed": 4121,
                    "decision_step": 100,
                    "all_branch_decision_identity": True,
                    "all_discovery_identity": True,
                    "effects": [effect],
                }
            ],
        },
    )
    return root


def test_authority_map_keeps_response_separate_from_absolute_residuals(tmp_path):
    refs = _refs(tmp_path)
    root = _source(tmp_path, refs)

    result = analyze([root], refs)

    assert result["development_only"] is True
    assert result["critical_density_defined"] is False
    assert result["binary_safe_but_insufficient_label_defined"] is False
    assert result["coverage"]["declared_seed_cell_count"] == 2
    assert result["coverage"]["paired_opportunity_count"] == 1
    assert result["coverage"]["no_decision_seed_cell_count"] == 1
    assert result["overall"]["response_class_counts"] == {"helpful": 1}
    row = result["rows"][0]
    assert row["response_class"] == "helpful"
    assert row["in_frozen_rule_evidence_domain"] is True
    assert row["treatment_absolute"]["severe_braking_absolute_reference_units"] == 0.05
    assert row["treatment_absolute"]["acceleration_energy_absolute_reference_units"] == pytest.approx(
        70 / 600
    )
    assert row["command_authority"]["mean_projection_survival"] == 0.5
    assert result["integrity"]["maximum_absolute_abstain_replay_delta_overall"] == 0.0


def test_authority_map_marks_out_of_domain_rows_as_diagnostics(tmp_path):
    refs = _refs(tmp_path)
    root = _source(tmp_path, refs, penetration=0.15)

    result = analyze([root], refs)

    assert result["frozen_rule_in_domain"]["paired_opportunity_count"] == 0
    assert result["out_of_domain_diagnostic"]["paired_opportunity_count"] == 1
    assert result["rows"][0]["evidence_role"] == "out_of_domain_diagnostic"


def test_authority_map_rejects_prospective_seeds(tmp_path):
    refs = _refs(tmp_path)
    root = _source(tmp_path, refs, seeds=(4201,))

    with pytest.raises(ValueError, match="prospective seed"):
        analyze([root], refs)


def test_grid_summary_requires_complete_development_grid(tmp_path):
    grid = tmp_path / "grid.json"
    _write_json(
        grid,
        {
            "schema": "idm_paired_option_penetration_grid/1",
            "development_only": True,
            "status": "complete",
            "cells": [
                {"status": "complete", "output": str(tmp_path / "one")},
                {"status": "complete", "output": str(tmp_path / "two")},
            ],
        },
    )

    assert roots_from_grid_summary(grid) == [
        (tmp_path / "one").resolve(),
        (tmp_path / "two").resolve(),
    ]
