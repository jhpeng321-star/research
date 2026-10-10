"""Planner-variant rule: lowest CAL dV CRPS, coverage constraint, tie."""
from __future__ import annotations

import json

import pytest

from combine.select_variant import (DEFAULT_CONTRASTS, DEFAULT_TAUS,
                                    VARIANTS, select_planner_variant,
                                    verify_planner_variant,
                                    write_planner_variant)


def _crps(values):
    return {v: {t: {c: float(values[v]) for c in DEFAULT_CONTRASTS}
                 for t in DEFAULT_TAUS} for v in VARIANTS}


def _cov(values):
    return {v: float(values[v]) for v in VARIANTS}


def test_variants_include_idea_a():
    assert VARIANTS == ("T", "A", "C", "N")


def test_lowest_eligible_crps_wins():
    rec = select_planner_variant(
        crps=_crps({"T": 1.20, "A": 1.15, "C": 1.00, "N": 1.10}),
        coverage=_cov({"T": 0.80, "A": 0.80, "C": 0.79, "N": 0.78}))
    assert rec["chosen"] == "C"


def test_tie_within_two_percent_goes_to_simpler():
    rec = select_planner_variant(
        crps=_crps({"T": 1.015, "A": 1.010, "C": 1.005, "N": 1.000}),
        coverage=_cov({"T": 0.80, "A": 0.80, "C": 0.80, "N": 0.80}))
    assert rec["chosen"] == "T"
    assert set(rec["tied"]) == {"T", "A", "C", "N"}


def test_near_tie_only_simplest_among_close():
    rec = select_planner_variant(
        crps=_crps({"T": 1.20, "A": 1.15, "C": 1.10, "N": 1.00}),
        coverage=_cov({"T": 0.80, "A": 0.80, "C": 0.80, "N": 0.80}))
    assert rec["chosen"] == "N"
    assert rec["tied"] == ["N"]


def test_idea_a_tie_order_beats_c_and_n():
    rec = select_planner_variant(
        crps=_crps({"T": 1.30, "A": 1.000, "C": 1.005, "N": 0.990}),
        coverage=_cov({"T": 0.80, "A": 0.80, "C": 0.80, "N": 0.80}))
    assert rec["chosen"] == "A"
    assert set(rec["tied"]) == {"A", "C", "N"}


def test_coverage_constraint_excludes_best():
    rec = select_planner_variant(
        crps=_crps({"T": 1.30, "A": 1.20, "C": 1.10, "N": 1.00}),
        coverage=_cov({"T": 0.70, "A": 0.90, "C": 0.80, "N": 0.92}))
    assert rec["chosen"] == "C"
    assert sorted(rec["eligible"]) == ["C"]


def test_no_variant_meets_coverage():
    rec = select_planner_variant(
        crps=_crps({"T": 1.3, "A": 1.25, "C": 1.2, "N": 1.1}),
        coverage=_cov({"T": 0.60, "A": 0.63, "C": 0.65, "N": 0.95}))
    assert rec["chosen"] is None
    assert "coverage" in rec["reason"]


def test_write_and_verify_sha(tmp_path):
    rec = select_planner_variant(
        crps=_crps({"T": 1.20, "A": 1.15, "C": 1.00, "N": 1.10}),
        coverage=_cov({"T": 0.80, "A": 0.80, "C": 0.79, "N": 0.78}))
    path = tmp_path / "PLANNER_VARIANT.json"
    sha = write_planner_variant(path, rec)
    assert len(sha) == 64
    assert verify_planner_variant(path) == sha
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["sha256"] == sha and data["chosen"] == "C"
    data["chosen"] = "N"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_planner_variant(path)
