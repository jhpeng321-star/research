from tools.analyze_idm_p50_projection_preview import analyze


def test_empty_preview_is_explicitly_diagnostic(tmp_path):
    result = analyze([tmp_path])
    assert result["diagnostic_only"] is True
    assert result["candidate_count"] == 0
    assert result["reference_distinctness"]["p50_vs_p30"]["distinct_fraction"] is None
    assert result["desired_command_distinctness"]["p50_vs_p30"]["distinct_fraction"] is None
    assert result["p50_vs_p30"]["distinct_fraction"] is None
    assert result["p50_vs_p70"]["distinct_fraction"] is None
