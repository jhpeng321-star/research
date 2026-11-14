"""OE32 is a bounded development allocation, never an acceptance escape."""
import pytest
from traffic_shepherd.seeds import (OE32_OBSERVATION_DEVELOPMENT_SEEDS,
    PROSPECTIVE_EVALUATION_EXCLUDED_SEEDS, SEALED_SEEDS, require_development_case)


def test_named_oe32_panel_is_excluded_from_future_acceptance():
    assert len(OE32_OBSERVATION_DEVELOPMENT_SEEDS) == 12
    assert set(OE32_OBSERVATION_DEVELOPMENT_SEEDS) <= set(PROSPECTIVE_EVALUATION_EXCLUDED_SEEDS)
    assert not set(OE32_OBSERVATION_DEVELOPMENT_SEEDS) & SEALED_SEEDS
    for seed in OE32_OBSERVATION_DEVELOPMENT_SEEDS:
        require_development_case(seed, 2700, 'development')


@pytest.mark.parametrize('seed,loading,role', [(4149, 2600, 'development'),
    (4149, 2700, 'exposed'), (4161, 2700, 'development'), (4201, 2700, 'development')])
def test_named_allocation_does_not_expand_other_cases(seed, loading, role):
    with pytest.raises(ValueError):
        require_development_case(seed, loading, role)
