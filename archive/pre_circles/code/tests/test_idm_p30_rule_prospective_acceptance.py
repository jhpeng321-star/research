import json
from pathlib import Path

import pytest

from idm.contract import DAMPING_ACCEPTANCE_SEEDS, validate_seed_role
from tools.run_idm_paired_option_penetration_grid import cell_command
from tools.analyze_idm_p30_rule_prospective_acceptance import (
    DEMANDS,
    PENETRATIONS,
    REWARD_REFS_SHA256,
    _acceptance_reward,
    _cell_directory,
    _constraints,
    _validate_protocol,
    analyze,
)


def _protocol(demand: int, penetration: float) -> dict:
    return {
        "development_only": False,
        "evidence_role": "damping_acceptance",
        "demand": demand,
        "seeds": list(DAMPING_ACCEPTANCE_SEEDS),
        "action_set": "activation_speed_ceiling_v1",
        "terminal_reference_percentile": "p30",
        "option_steps": 20,
        "minimum_dose_steps": 16,
        "cav_penetration": penetration,
        "decision_index": 3,
        "candidate_ranking": "runway",
        "candidate_count": 1,
        "safety_horizon_steps": 60,
        "service_horizon_steps": 60,
        "max_predecision_steps": 900,
        "minimum_decision_step": 0,
        "landmark": "risk_onset",
        "maximum_landmarks_per_cell": 3,
        "discover_only": False,
        "decision_source_path": None,
        "reward_refs_sha256": REWARD_REFS_SHA256,
    }


def _write_no_decision_grid(root: Path) -> None:
    for penetration in PENETRATIONS:
        for demand in DEMANDS:
            directory = _cell_directory(root, demand, penetration)
            (directory / "discoveries").mkdir(parents=True)
            (directory / "branches").mkdir()
            protocol = _protocol(demand, penetration)
            (directory / "summary.json").write_text(
                json.dumps(
                    {
                        "protocol": protocol,
                        "protocol_hash": f"{demand}-{penetration}",
                        "branch_count": 0,
                        "cases": [],
                    }
                ),
                encoding="utf-8",
            )
            (directory / "progress.json").write_text(
                json.dumps({"pending": 0, "complete": 0}), encoding="utf-8"
            )
            for seed in DAMPING_ACCEPTANCE_SEEDS:
                (directory / "discoveries" / f"m{demand}.seed{seed}.json").write_text(
                    json.dumps({"seed": seed, "decision": None}), encoding="utf-8"
                )


def test_acceptance_seed_partition_is_exact():
    assert DAMPING_ACCEPTANCE_SEEDS == tuple(range(4201, 4221))
    assert validate_seed_role("damping_acceptance", DAMPING_ACCEPTANCE_SEEDS) == (
        DAMPING_ACCEPTANCE_SEEDS
    )
    with pytest.raises(ValueError):
        validate_seed_role("damping_acceptance", [4221])


def test_grid_command_labels_prospective_evidence_role(tmp_path: Path):
    command = cell_command(
        python=Path("python.exe"),
        repo=tmp_path,
        output=tmp_path / "out",
        demand=2000,
        penetration=0.05,
        decision_index=3,
        seeds=list(DAMPING_ACCEPTANCE_SEEDS),
        workers=4,
        reward_refs=tmp_path / "refs.json",
        evidence_role="damping_acceptance",
    )
    joined = " ".join(command)
    assert "--evidence-role damping_acceptance" in joined
    assert "--decision-index 3" in joined
    assert "--action-set activation_speed_ceiling_v1" in joined


def test_development_grid_command_keeps_legacy_arguments(tmp_path: Path):
    command = cell_command(
        python=Path("python.exe"),
        repo=tmp_path,
        output=tmp_path / "out",
        demand=1900,
        penetration=0.10,
        decision_index=1,
        seeds=[4121],
        workers=1,
        reward_refs=tmp_path / "refs.json",
    )
    assert "--evidence-role" not in command


def test_protocol_validation_rejects_any_frozen_field_mismatch():
    protocol = _protocol(2000, 0.05)
    _validate_protocol(protocol, demand=2000, penetration=0.05)
    protocol["candidate_count"] = 2
    with pytest.raises(ValueError, match="candidate_count"):
        _validate_protocol(protocol, demand=2000, penetration=0.05)


def test_no_decision_cells_are_retained_as_zero_value(tmp_path: Path):
    _write_no_decision_grid(tmp_path)
    result = analyze(tmp_path)
    assert result["cell_count"] == 120
    assert result["activation_count"] == 0
    assert result["no_decision_count"] == 120
    assert result["mean_policy_reward"] == 0.0
    assert result["mean_policy_physical_effect"] == 0.0


def test_discovery_with_decision_cannot_be_silently_counted_as_no_decision(
    tmp_path: Path,
):
    _write_no_decision_grid(tmp_path)
    path = (
        _cell_directory(tmp_path, 1900, 0.05)
        / "discoveries"
        / "m1900.seed4201.json"
    )
    path.write_text(json.dumps({"seed": 4201, "decision": {"step": 10}}), encoding="utf-8")
    with pytest.raises(ValueError, match="discovery/case mismatch"):
        analyze(tmp_path)


def test_follower_braking_failure_is_retained_as_hard_failure():
    base = {
        "completed_horizon_steps": 60,
        "target_command_step_count": 20,
        "unexpected_command_count": 0,
        "collision_observations": 0,
        "teleport_observations": 0,
        "low_ttc_exposure_steps": 0,
        "mean_corridor_delay_rate": 1.0,
        "arrived_vehicles": 20,
        "merge_discharge_count": 10,
        "treatment_safety_abort_rows": [],
        "spatial_totals": {"immediate_follower": {"severe_braking_count": 0}},
    }
    treatment = json.loads(json.dumps(base))
    treatment["spatial_totals"]["immediate_follower"]["severe_braking_count"] = 1
    constraints = _constraints(base, treatment)
    assert constraints["immediate_follower_braking_noninferior"] is False
    assert _acceptance_reward(0.5, constraints) == -2.0
