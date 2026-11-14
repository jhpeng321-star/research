"""Comparator regression tests for the hold-extension reporting repair.

Covers the four required behaviors: wrong-reference selection is rejected,
the sign convention is treatment - reference, totals and per-request
quantities stay distinct and consistent, and guardrail classification
matches the registered criteria. A final artifact test pins the generated
COMPARISONS_V3.json to the independently derived expected values.
"""
import json

import pytest

from traffic_shepherd.hold_extend_comparisons import (
    CRITERIA, METRICS, ROOT, ReferenceMismatch, assert_total_consistent,
    build_comparison_record, classify_difference, cost_difference,
    per_request_costs, per_request_from_summary, require_same_conditions)

BASE = dict(service=10.0, braking=1.0, effort=100.0, ttc=1.0, wait=5.0)


def _costs(**overrides):
    return dict(BASE, **overrides)


# ---- wrong-reference selection ----

def test_reference_identity_is_declared_not_inferred():
    meta = dict(seed=4107, requested_population=1415, mainline_vph=2700,
                penetration=0.2, duration_s=2400)
    # same everything except the reference job id: rejected
    wrong = dict(meta, job_id="s4107_m2700_p20_t600_lifecycle_v2_wall",
                 expected_job_id="s4107_m2700_p20_t600_hold_adjust_up_wall")
    with pytest.raises(ReferenceMismatch, match="reference identity mismatch"):
        require_same_conditions(meta, wrong)
    require_same_conditions(meta, dict(meta, job_id="s4107_m2700_p20_t600_hold_adjust_up_wall",
                                       expected_job_id="s4107_m2700_p20_t600_hold_adjust_up_wall"))


def test_condition_mismatch_rejected_even_with_right_job_id():
    meta = dict(seed=4107, requested_population=1415, mainline_vph=2700,
                penetration=0.2, duration_s=2400)
    other = dict(meta, seed=4108, job_id="x", expected_job_id="x")
    with pytest.raises(ReferenceMismatch, match="seed"):
        require_same_conditions(meta, other)


def test_artifact_level_wrong_reference_is_caught_by_wall_vs_wall():
    """The historical error, reproduced and rejected: comparing the 4107
    extension against keep-24 does not reproduce the registered wall_vs_wall
    block, which pins the raise-26 H20 source."""
    record = build_comparison_record(ROOT)
    b = record["seeds"]["4107"]["B_extension_vs_same_prefix_raise26_H20"]
    registered_ttc = b["totals"]["ttc_vehicle_seconds"] / 1415
    keep24_wrong = -0.092226148409  # extension minus keep-24 (superseded table)
    assert abs(keep24_wrong - registered_ttc) > 0.05  # the two references differ materially
    with pytest.raises(ReferenceMismatch, match="wrong reference or arithmetic"):
        from traffic_shepherd.hold_extend_comparisons import _cross_check_wall_vs_wall
        _cross_check_wall_vs_wall(
            dict(_costs(ttc=1.6328621908127208 - 2441.0 / 1415.0)),
            dict(direction="new_target_minus_h20_source",
                 metrics={"braking_vehicle_seconds": dict(per_request=0.0, full_difference=0.0),
                          "acceleration_energy": dict(per_request=0.0, full_difference=0.0),
                          "ttc_below_3_vehicle_seconds": dict(per_request=0.0, full_difference=0.0),
                          "requested_not_arrived_vehicle_seconds": dict(per_request=0.0, full_difference=0.0),
                          "collision_involvements": dict(per_request=0.0, full_difference=0)}),
            1415, 4107)


# ---- sign convention ----

def test_sign_convention_treatment_minus_reference():
    worse = cost_difference(_costs(ttc=1.2), _costs())
    better = cost_difference(_costs(ttc=0.8), _costs())
    assert worse["ttc"] == pytest.approx(+0.2)  # positive = deterioration
    assert better["ttc"] == pytest.approx(-0.2)  # negative = improvement
    assert set(worse) == set(METRICS)


# ---- population conversion / totals vs per-request ----

def test_total_and_per_request_consistency():
    assert_total_consistent(2338.0 / 1415, 2338.0, 1415, "ttc")
    with pytest.raises(ReferenceMismatch, match="does not reproduce"):
        assert_total_consistent(2338.0 / 1415, 2383.0, 1415, "ttc")


def test_per_request_from_summary_matches_division():
    summary = dict(population=1415,
                   system_seconds_per_requested_vehicle=196.5,
                   source_wait_vehicle_seconds=58665.0,
                   totals=dict(braking_vehicle_seconds=1700.0,
                               acceleration_energy=142000.0,
                               ttc_below_3_vehicle_seconds=2338.0,
                               requested_not_arrived_vehicle_seconds=278055.0))
    costs = per_request_from_summary(summary)
    assert costs["braking"] == pytest.approx(1700.0 / 1415)
    assert costs["service"] == pytest.approx(196.5)
    assert costs["wait"] == pytest.approx(58665.0 / 1415)
    with pytest.raises(ValueError):
        per_request_costs(dict.fromkeys(
            ("system_seconds_per_request", "braking_seconds_per_request",
             "effort_per_request", "ttc_seconds_per_request",
             "source_wait_seconds_per_request"), -1.0))


# ---- criterion classification ----

def test_classification_4108_vs_damping_four_violations_wait_within():
    diff = dict(service=13.258990, braking=0.095256, effort=9.373357,
                ttc=0.081867, wait=0.005738)
    c = classify_difference(diff)
    assert c["violated_guardrails"] == ["braking", "effort", "service", "ttc"]
    assert c["increases_within_allowance"] == ["wait"]  # +0.005738 <= 0.1
    assert c["improvements"] == []


def test_classification_4107_ttc_only_violation_and_primaries():
    diff = dict(service=-8.488693, braking=-0.063958, effort=-4.214348,
                ttc=+0.020141, wait=-0.154417)
    c = classify_difference(diff)
    assert c["violated_guardrails"] == ["ttc"]
    assert sorted(c["improvements"]) == ["braking", "effort", "service", "wait"]
    assert c["material_primary_improvement"] == ["braking", "service"]
    # the razor-thin margin is explicit: +0.000141 over the 0.02 allowance
    assert diff["ttc"] - CRITERIA["maximum_cost_increase"]["ttc"] == pytest.approx(0.000141, abs=1e-6)


# ---- generated artifact (machine-generated comparison data) ----

EXPECTED_B = {
    "4107": dict(service=-0.0530035335689263, braking=-0.00636042402826864,
                 effort=1.53080151588944, ttc=-0.0194346289752652,
                 wait=-0.0208480565371048),
    "4108": dict(service=-0.000765110941102876, braking=0.000382555470543222,
                 effort=-0.0103792952841104, ttc=0.00114766641162967, wait=0.0),
}


def test_generated_comparisons_match_independently_derived_values():
    path = ROOT / "output/hold_extend_patch_v1/COMPARISONS_V3.json"
    assert path.exists(), "run python -m traffic_shepherd.hold_extend_comparisons"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema"] == "hold_extend_comparator/1"
    assert record["sign_convention"].startswith("cost difference = treatment - reference")
    for seed, expected in EXPECTED_B.items():
        block = record["seeds"][seed]["B_extension_vs_same_prefix_raise26_H20"]
        for metric, value in expected.items():
            assert block["difference_per_request"][metric] == pytest.approx(value, abs=1e-9)
    b4107 = record["seeds"]["4107"]["B_extension_vs_same_prefix_raise26_H20"]
    assert b4107["totals"]["ttc_vehicle_seconds"] == pytest.approx(-27.5)
    assert b4107["collision_difference"] == 2  # 0 -> 2 versus raise-H20
    b4108 = record["seeds"]["4108"]["B_extension_vs_same_prefix_raise26_H20"]
    assert b4108["totals"]["ttc_vehicle_seconds"] == pytest.approx(1.5)
    assert b4108["collision_difference"] == 0
    assert b4108["difference_per_request"]["wait"] == 0.0


def test_generated_comparisons_a_and_c_and_prefix():
    record = json.loads((ROOT / "output/hold_extend_patch_v1/COMPARISONS_V3.json")
                        .read_text(encoding="utf-8"))
    a4107 = record["seeds"]["4107"]["A_extension_vs_frozen_damping"]
    assert a4107["difference_per_request"]["ttc"] == pytest.approx(0.020141, abs=2e-6)
    assert a4107["absolute"]["treatment_collision_involvements"] == 2
    assert a4107["absolute"]["damping_collision_involvements"] == 4
    assert a4107["classification"]["violated_guardrails"] == ["ttc"]
    a4108 = record["seeds"]["4108"]["A_extension_vs_frozen_damping"]
    assert a4108["classification"]["violated_guardrails"] == \
        ["braking", "effort", "service", "ttc"]
    assert a4108["classification"]["increases_within_allowance"] == ["wait"]
    # comparison C: keep-24 reference, explicitly NOT "the extension alone"
    c4107 = record["seeds"]["4107"]["C_extension_vs_keep24_H20"]
    assert c4107["difference_per_request"]["service"] == pytest.approx(-4.334982, abs=2e-6)
    assert "NEVER describes the extension alone" in c4107["interpretation_note"]
    for seed in ("4107", "4108"):
        prefix = record["seeds"][seed]["B_extension_vs_same_prefix_raise26_H20"]["prefix_evidence"]
        assert prefix["extension_prefix_exact"] is True
        assert prefix["shared_prefix_through_s"] == prefix["extension_rows_before_branch_s"]
    claims = {c["claim"]: c for c in record["superseded_claims"]}
    assert any("withdrawn" in c["correction"] for c in claims.values())


def test_effect_carrier_matches_frozen_analyses_verbatim():
    """The verify-sources carrier embeds the frozen analysis rows verbatim;
    the frozen files stay authoritative."""
    carrier = json.loads((ROOT / "output/hold_extend_patch_v1/EXTENSION_EFFECTS_V3.json")
                         .read_text(encoding="utf-8"))
    assert carrier["schema"] == "hold_extend_effect_carrier/1"
    frozen = {seed: json.loads((ROOT / path).read_text(encoding="utf-8"))["results"][0]
              for seed, path in (("4107", "runs/active/hold_extend_v1/analysis.json"),
                                 ("4108", "runs/active/hold_extend_v2/analysis.json"))}
    for provenance_path, info in carrier["provenance"].items():
        current = json.loads((ROOT / provenance_path).read_text(encoding="utf-8"))["results"][info["row_index"]]
        assert current is not None  # frozen source still present
    assert carrier["results"] == [frozen["4107"], frozen["4108"]]
