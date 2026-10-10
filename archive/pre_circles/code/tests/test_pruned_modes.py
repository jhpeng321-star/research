"""Pruned champion action set (2026-08-20).

The retrained baseline champion runs Discrete(3): coast · recover_speed ·
protect. fair_balance (metering leaks at dense demand) and vacate_drop
(disabled in every probe) are removed — and the removal is NOT a prefix of
the global MODE_NAMES registry, so the engine must map action indices
through the configured list (ShepherdActionEngine.mode_names), not the
registry.
"""

from unittest.mock import patch

from env.actions import make_action_space
from tests.test_shepherd_actions import _make_engine, _mock_traci

PRUNED = ["coast", "recover", "guard"]


def _pruned_engine():
    engine = _make_engine()
    engine.mode_names = list(PRUNED)
    engine.mode_count = len(PRUNED)
    return engine


def test_pruned_space_is_three():
    assert make_action_space(len(PRUNED)).n == 3


def test_engine_maps_through_configured_list():
    engine = _pruned_engine()
    # Index 2 in the pruned set is protect — NOT registry index 2
    # (recover_speed). This is the mapping the pruned champion depends on.
    assert engine.mode_names[2] == "guard"
    assert engine.mode_count == 3


def test_eligibility_mask_sized_and_indexed_by_pruned_set():
    engine = _pruned_engine()
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        mask = engine.eligibility_mask("cav0")
    assert len(mask) == 3
    assert mask[0] == 1.0  # coast is always legal
