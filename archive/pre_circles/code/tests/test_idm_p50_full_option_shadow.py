import json

from tools.analyze_idm_p50_full_option_shadow import analyze


def test_shadow_analyzer_separates_first_step_from_option(tmp_path):
    root = tmp_path / "run"
    branches = root / "branches"
    branches.mkdir(parents=True)
    rows = []
    for step, delta in ((1, 0.0), (2, 0.10)):
        rows.append(
            {
                "relative_step": step,
                "desired_acceleration_mps2": -0.5,
                "projected_acceleration_mps2": -0.5,
                "p50_shadow": {
                    "desired_acceleration_mps2": -0.5 + delta,
                    "projected_acceleration_mps2": -0.5 + delta,
                },
            }
        )
    (branches / "case.treatment.json").write_text(
        json.dumps(
            {
                "case_id": "case",
                "seed": 1,
                "is_control": False,
                "mode": "speed_ceiling",
                "command_rows": rows,
            }
        ),
        encoding="utf-8",
    )
    result = analyze([root])
    assert result["diagnostic_only"] is True
    assert result["all_commands"]["shadow_coverage_fraction"] == 1.0
    assert result["all_commands"]["projected_distinct_fraction"] == 0.5
    assert result["after_first_step"]["projected_distinct_fraction"] == 1.0
