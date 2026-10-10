from tools.pool_idm_p30_p50_development_screen import _metrics


def _case(seed, p30_class, p30_value, p50_class, p50_value):
    return {
        "seed": seed,
        "p30_effect": {
            "classification": p30_class,
            "safety_composite_reference_units": p30_value,
        },
        "p50_effect": {
            "classification": p50_class,
            "safety_composite_reference_units": p50_value,
        },
    }


def test_pooled_metrics_count_seed_diversity_and_replay_responses():
    cases = [
        _case(1, "harmful", -0.1, "neutral", 0.0),
        _case(1, "helpful", 0.2, "helpful", 0.1),
        _case(2, "neutral", 0.0, "harmful", -0.2),
    ]
    metrics = _metrics(cases, "p50", 0.02)
    assert metrics["qualified_case_count"] == 3
    assert metrics["qualified_harmful_fraction"] == 1.0 / 3.0
    assert metrics["helpful_seeds"] == [1]
    assert metrics["nonbeneficial_seeds"] == [1, 2]
    assert metrics["effect_beyond_replay_fraction"] == 2.0 / 3.0
