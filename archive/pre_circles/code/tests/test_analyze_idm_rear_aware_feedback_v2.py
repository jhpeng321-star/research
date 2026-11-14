"""Focused tests for the P30-versus-v2 joined pilot analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.analyze_idm_rear_aware_feedback_v2 import (
    P30_ARM,
    V2_ARM,
    join_pilot_and_p30,
)


def _effect(*, arm: str, classification: str, composite: float, dose: int = 5) -> dict:
    return {
        "arm": arm,
        "mode": arm.split("_", 1)[1],
        "classification": classification,
        "safety_composite_reference_units": composite,
        "constraints": {
            "complete_horizon": True,
            "dose_at_least_minimum": dose >= 5,
            "zero_unexpected_commands": True,
            "zero_added_collisions": True,
            "zero_added_teleports": True,
            "low_ttc_within_budget": True,
            "delay_within_budget": True,
            "arrival_within_budget": True,
            "discharge_within_budget": True,
        },
        "constraints_pass": classification != "harmful",
        "target_command_step_count": dose,
        "low_ttc_exposure_step_delta": 0,
        "delay_rate_delta": 0.0,
        "arrival_delta": 0,
        "merge_discharge_count_delta": 0,
        "treatment_safety_abort_count": 0,
        "projection_survival_values": [1.0],
        "paired_projection_survival": 1.0,
    }


def _case(*, seed: int, step: int, arm: str, classification: str, composite: float) -> dict:
    return {
        "case_id": f"m2000.seed{seed}.t{step}",
        "seed": seed,
        "decision_step": step,
        "effects": [
            _effect(
                arm=arm,
                classification=classification,
                composite=composite,
            )
        ],
    }


def _summary(cases: list[dict]) -> dict:
    return {"schema": "idm_decentralized_authority/1", "cases": cases}


def _branch(tmp_path: Path, name: str, *, seed: int, step: int, target: str, arm: str) -> Path:
    row = {
        "arm": arm,
        "seed": seed,
        "decision_step": step,
        "target_vehicle_id": target,
        "completed_horizon_steps": 60,
        "feedback_release_rows": [],
        "treatment_safety_abort_rows": [],
        "treatment_dose": {
            "realized_active_steps": 5,
            "requested_probe_steps": 5,
            "requested_maximum_active_steps": 20,
            "release_reasons": [],
        },
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(row), encoding="utf-8")
    return path


def test_join_matches_exact_seed_step_and_target(tmp_path: Path):
    pilot = _summary([_case(seed=4123, step=292, arm=V2_ARM, classification="helpful", composite=0.12)])
    source = _summary([_case(seed=4123, step=292, arm=P30_ARM, classification="harmful", composite=-0.05)])
    pilot_branch = _branch(tmp_path, "pilot", seed=4123, step=292, target="ID.1", arm=V2_ARM)
    source_branch = _branch(tmp_path, "source", seed=4123, step=292, target="ID.1", arm=P30_ARM)

    report = join_pilot_and_p30(
        pilot_summary=pilot,
        pilot_discoveries=[],
        pilot_branch_paths=[pilot_branch],
        source_summary=source,
        source_branch_paths=[source_branch],
    )

    assert report["matched_case_count"] == 1
    assert report["excluded_count"] == 0
    row = report["matched"][0]
    assert row["seed"] == 4123
    assert row["target_match"] is True
    assert row["v2"]["classification"] == "helpful"
    assert row["p30"]["classification"] == "harmful"
    gates = report["gates"]
    assert gates["p30_and_v2_share_exact_identity"] is True
    assert gates["v2_fewer_harmful_than_p30"] is True
    assert gates["v2_benefit_beyond_deadband_in_at_least_two_seeds"] is False
    assert gates["at_least_two_nonbeneficial_seeds"] is False


def test_target_mismatch_is_excluded_never_compared(tmp_path: Path):
    pilot = _summary([_case(seed=4123, step=292, arm=V2_ARM, classification="helpful", composite=0.12)])
    source = _summary([_case(seed=4123, step=292, arm=P30_ARM, classification="harmful", composite=-0.05)])
    pilot_branch = _branch(tmp_path, "pilot", seed=4123, step=292, target="ID.1", arm=V2_ARM)
    source_branch = _branch(tmp_path, "source", seed=4123, step=292, target="ID.2", arm=P30_ARM)

    report = join_pilot_and_p30(
        pilot_summary=pilot,
        pilot_discoveries=[],
        pilot_branch_paths=[pilot_branch],
        source_summary=source,
        source_branch_paths=[source_branch],
    )

    assert report["matched_case_count"] == 0
    assert report["excluded_count"] == 1
    assert report["excluded"][0]["reason"] == "branch_identity"
    assert report["gates"]["p30_and_v2_share_exact_identity"] is False


def test_missing_source_case_is_reported_separately(tmp_path: Path):
    pilot = _summary([_case(seed=4130, step=460, arm=V2_ARM, classification="neutral", composite=0.0)])
    source = _summary([])
    pilot_branch = _branch(tmp_path, "pilot", seed=4130, step=460, target="ID.1", arm=V2_ARM)

    report = join_pilot_and_p30(
        pilot_summary=pilot,
        pilot_discoveries=[],
        pilot_branch_paths=[pilot_branch],
        source_summary=source,
        source_branch_paths=[],
    )

    assert report["matched_case_count"] == 0
    assert report["missing_source_case_count"] == 1
    assert report["missing_source_cases"][0]["reason"] == "no_source_case"


def test_no_decision_seeds_are_reported_and_never_joined(tmp_path: Path):
    pilot = _summary([_case(seed=4123, step=292, arm=V2_ARM, classification="neutral", composite=0.0)])
    source = _summary([_case(seed=4123, step=292, arm=P30_ARM, classification="neutral", composite=0.0)])
    pilot_branch = _branch(tmp_path, "pilot", seed=4123, step=292, target="ID.1", arm=V2_ARM)
    source_branch = _branch(tmp_path, "source", seed=4123, step=292, target="ID.1", arm=P30_ARM)
    discoveries = [{"seed": 4131, "decision": None, "matched_decision_source": {}}]

    report = join_pilot_and_p30(
        pilot_summary=pilot,
        pilot_discoveries=discoveries,
        pilot_branch_paths=[pilot_branch],
        source_summary=source,
        source_branch_paths=[source_branch],
    )

    assert report["no_decision_seeds"] == [4131]
    assert report["matched_case_count"] == 1


def test_early_release_tracked_with_documented_reason(tmp_path: Path):
    pilot_case = _case(
        seed=4132, step=880, arm=V2_ARM, classification="harmful", composite=-0.02
    )
    source = _summary([_case(seed=4132, step=880, arm=P30_ARM, classification="neutral", composite=0.0)])
    pilot_branch = _branch(tmp_path, "pilot", seed=4132, step=880, target="ID.9", arm=V2_ARM)
    source_branch = _branch(tmp_path, "source", seed=4132, step=880, target="ID.9", arm=P30_ARM)
    pilot_branch.write_text(
        json.dumps(
            {
                **json.loads(pilot_branch.read_text(encoding="utf-8")),
                "feedback_release_rows": [{"reason": "follower_identity_changed"}],
            }
        ),
        encoding="utf-8",
    )

    report = join_pilot_and_p30(
        pilot_summary=_summary([pilot_case]),
        pilot_discoveries=[],
        pilot_branch_paths=[pilot_branch],
        source_summary=source,
        source_branch_paths=[source_branch],
    )

    row = report["matched"][0]
    assert row["v2"]["release_reasons"][0]["reason"] == "follower_identity_changed"
    assert report["gates"]["release_reason_counts"] == {
        "follower_identity_changed": 1
    }
