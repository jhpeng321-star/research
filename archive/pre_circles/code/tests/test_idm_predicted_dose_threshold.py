import json

from tools.analyze_idm_predicted_dose_threshold import analyze


def _write_case(root, *, seed, predicted, realized, classification, effect, passed):
    case_id = f"m2000.seed{seed}.t50"
    (root / "discoveries").mkdir(parents=True, exist_ok=True)
    (root / "discoveries" / f"m2000.seed{seed}.json").write_text(
        json.dumps(
            {
                "decision": {
                    "decision_index": 1,
                    "candidates": [{"slot": 1, "predicted_dose_s": predicted}],
                }
            }
        ),
        encoding="utf-8",
    )
    return {
        "case_id": case_id,
        "seed": seed,
        "decision_step": 50,
        "effects": [
            {
                "candidate_slot": 1,
                "target_command_step_count": realized,
                "classification": classification,
                "safety_composite_reference_units": effect,
                "constraints_pass": passed,
                "constraints": {
                    "dose_at_least_minimum": realized >= 16,
                    "zero_added_collisions": True,
                    "arrival_within_budget": passed,
                },
            }
        ],
    }


def test_threshold_reports_coverage_and_does_not_relabel_outcomes(tmp_path):
    root = tmp_path / "paired"
    root.mkdir()
    cases = [
        _write_case(
            root,
            seed=1,
            predicted=18.0,
            realized=20,
            classification="helpful",
            effect=0.10,
            passed=True,
        ),
        _write_case(
            root,
            seed=2,
            predicted=22.0,
            realized=20,
            classification="harmful",
            effect=-0.10,
            passed=True,
        ),
    ]
    (root / "summary.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")

    result = analyze([root])

    assert result["post_hoc_diagnostic_only"] is True
    assert result["opportunity_count"] == 2
    sixteen, twenty = result["thresholds"]
    assert sixteen["admitted_count"] == 2
    assert sixteen["qualified_harmful_fraction"] == 0.5
    assert twenty["admitted_count"] == 1
    assert twenty["opportunity_coverage_fraction"] == 0.5
    assert twenty["qualified_harmful_fraction"] == 1.0
    assert twenty["excluded_classification_counts"] == {"helpful": 1}
