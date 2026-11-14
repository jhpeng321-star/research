"""Focused tests for binary option-reward audit utilities."""

from __future__ import annotations

import json

import pytest

from tools.analyze_idm_binary_option_reward import (
    _correlation,
    _effect_deltas,
    _rank,
    write_outputs,
)


def test_rank_uses_average_rank_for_ties():
    assert _rank([3.0, 1.0, 1.0, 2.0]) == [4.0, 1.5, 1.5, 3.0]


def test_correlation_detects_exact_monotone_relationship():
    assert _correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert _correlation([1.0], [1.0]) != _correlation([1.0], [1.0])


def test_effect_delta_compares_independently_computed_components():
    computed = {
        "severe_braking_effect_reference_units": 0.3,
        "energy_effect_reference_units": -0.1,
        "physical_utility": 0.1,
    }
    stored = {
        "severe_braking_effect_reference_units": 0.3,
        "wave_effect_reference_units": -0.1,
        "safety_composite_reference_units": 0.1,
    }
    assert _effect_deltas(computed, stored) == {
        "severe": 0.0,
        "energy": 0.0,
        "composite": 0.0,
    }


def test_write_outputs_keeps_records_out_of_compact_summary(tmp_path):
    summary = {
        "valid_paired_binary_option_reward": True,
        "record_count": 1,
        "qualified_count": 1,
        "constraint_failed_count": 0,
        "pearson_reward_vs_effect": 1.0,
        "spearman_reward_vs_effect": 1.0,
        "maximum_absolute_neutral_reward": 0.0,
        "component_absolute_mass_shares": {
            "severe_braking": 0.5,
            "acceleration_energy": 0.5,
        },
        "gates": {"example": True},
        "records": [{"case_id": "case"}],
    }
    write_outputs(summary, tmp_path)
    compact = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "records" not in compact
    assert json.loads((tmp_path / "records.jsonl").read_text(encoding="utf-8")) == {
        "case_id": "case"
    }
    assert "Valid paired reward: `True`" in (tmp_path / "summary.md").read_text(
        encoding="utf-8"
    )
