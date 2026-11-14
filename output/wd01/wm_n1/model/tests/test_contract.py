"""The model constants must equal the frozen featurizer layout (FORMAT.md).

Read-only import of ``features.constants``: the model never imports featurizer
code at runtime, only this test pins the contract.
"""
from __future__ import annotations

from model import constants as M


def test_constants_match_frozen_featurizer():
    from features import constants as F
    pairs = [
        ("DT_S", M.DT_S, F.DT_S), ("K_MAX", M.K_MAX, F.K_MAX),
        ("VEH_F", M.VEH_F, F.VEH_F), ("PLAN_F", M.PLAN_F, F.PLAN_F),
        ("N_SEGMENTS", M.N_SEGMENTS, F.N_SEGMENTS),
        ("N_DECISION", M.N_DECISION, F.N_DECISION),
        ("MAX_WALLS", M.MAX_WALLS, F.MAX_WALLS),
        ("WALL_F", M.WALL_F, F.WALL_F),
        ("N_ENTRANCES", M.N_ENTRANCES, F.N_ENTRANCES),
        ("COUNTER_BINS", M.COUNTER_BINS, F.COUNTER_BINS),
        ("TAUS", M.TAUS, F.TAUS),
        ("ACTIONS", M.ACTIONS, F.ACTIONS),
    ]
    for name, mine, theirs in pairs:
        assert mine == theirs, name


def test_format_k_max_contract_matches_featurizer():
    """Amendment 5: K_max is a format parameter; both mirrors agree."""
    from features import constants as F

    assert (M.K_MAX_V2, M.K_MAX_V3) == (F.K_MAX_V2, F.K_MAX_V3) == (128, 192)
    assert M.FORMAT_K_MAX == F.FORMAT_K_MAX
    assert M.DEFAULT_FORMAT == F.DEFAULT_FORMAT == "v2"
    for fmt in sorted(F.FORMAT_K_MAX):
        assert M.k_max_of_format(fmt) == F.k_max_of_format(fmt)
    assert M.format_of_k_max(128) == F.format_of_k_max(128) == "v2"
    assert M.format_of_k_max(192) == F.format_of_k_max(192) == "v3"
    # the v2 legacy constant still mirrors the default format
    assert M.K_MAX == M.K_MAX_V2 == M.k_max_of_format(M.DEFAULT_FORMAT)


def test_feature_indices_match_featurizer():
    from features import constants as F
    assert (M.S_DOWN, M.LANE_LEFT, M.LANE_RIGHT, M.LANE_ACCEL) == (
        F.S_DOWN, F.LANE_LEFT, F.LANE_RIGHT, F.LANE_ACCEL)
    assert (M.V_IDX, M.A_IDX, M.LENGTH_IDX, M.IS_CAV_IDX) == (
        F.V_IDX, F.A_IDX, F.LENGTH_IDX, F.IS_CAV_IDX)
    assert (M.ACT_NATIVE, M.ACT_DAMPING, M.ACT_PI, M.ACT_WALL,
            M.ACT_UNKNOWN) == (F.ACT_NATIVE, F.ACT_DAMPING, F.ACT_PI,
                               F.ACT_WALL, F.ACT_UNKNOWN)
    assert (M.WALL_MEMBER_IDX, M.JUNCTION_IDX, M.PRESENT_IDX) == (
        F.WALL_MEMBER_IDX, F.JUNCTION_IDX, F.PRESENT_IDX)
