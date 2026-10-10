"""tests/test_paired_pool_oversample.py"""

from env.core import paired_pool_sample_probs

# fair_gap == seed value, 1..10 -> easy seeds are the low ones.
REFS = {str(s): {"fair_gap": float(s)} for s in range(1, 11)}
POOL = list(range(1, 11))


def test_disabled_returns_none():
    assert paired_pool_sample_probs(POOL, REFS, None, 2.0) is None
    assert paired_pool_sample_probs(POOL, REFS, 0.4, 1.0) is None
    assert paired_pool_sample_probs([], REFS, 0.4, 2.0) is None


def test_easy_seeds_boosted_and_probs_normalized():
    probs = paired_pool_sample_probs(POOL, REFS, easy_quantile=0.4, easy_boost=3.0)
    assert probs is not None
    assert abs(sum(probs) - 1.0) < 1e-9
    # np.quantile([1..10], 0.4) = 4.6 -> seeds 1-4 easy: weight 3, others 1.
    assert abs(probs[0] - 3.0 / 18.0) < 1e-9  # seed 1 (easy)
    assert abs(probs[3] - 3.0 / 18.0) < 1e-9  # seed 4 (easy)
    assert abs(probs[4] - 1.0 / 18.0) < 1e-9  # seed 5 (hard)
    assert abs(probs[9] - 1.0 / 18.0) < 1e-9  # seed 10 (hard)


def test_missing_ref_treated_as_hard():
    refs = {"1": {"fair_gap": 1.0}}  # seeds 2,3 missing -> fair_gap 0.0
    probs = paired_pool_sample_probs([1, 2, 3], refs, 0.5, 2.0)
    assert probs is not None
    assert abs(sum(probs) - 1.0) < 1e-9
