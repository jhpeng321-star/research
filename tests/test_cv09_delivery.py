import json
from pathlib import Path

from output.controller_verification_v9.delivery_analysis import analyze


def test_saved_cv08_delivery_recomputes_source_and_bridge():
    report = analyze()
    assert report["passed"] is True
    assert report["checks"]["raw_recurrence_matches"] is True
    assert report["checks"]["issued_bridge_matches"] is True
    assert len(report["rows"]) == 5
    assert report["covered_regions"] == {"catch": ["high", "interior"], "alpha": ["one"]}


def test_delivery_rows_are_finite_and_source_pinned():
    report = analyze()
    for row in report["rows"]:
        for key, value in row.items():
            if isinstance(value, float):
                assert value == value
                assert abs(value) < 1e308
    assert report["parameters"]["u_mps"] == 31.29
    assert report["artifact_sha256"]["result.json"]


def test_delivery_json_is_reproducible_artifact():
    report = analyze()
    output = Path("output/controller_verification_v9/DELIVERY.json")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["schema"] == "cv09_pi_delivery_analysis/1"
    assert saved == report
