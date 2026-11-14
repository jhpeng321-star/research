"""P9 corpus passthrough interface (implemented)."""
from __future__ import annotations

import pytest

from corpus.idea_a import (IDEA_A_ARRAYS, IDEA_A_DTYPES, IDEA_A_DONE_NAME,
                           IDEA_A_SCHEMA, IdeaAPassthrough,
                           road_end_relabel_enabled, verify_idea_a,
                           write_idea_a_channels)


def test_surface_and_flag_reader():
    assert IDEA_A_ARRAYS == ("veh_kin", "hdv_base", "cav_plan_ch",
                             "event_tok", "event_mask")
    assert IDEA_A_DONE_NAME == "IDEA_A_DONE.json"
    assert IDEA_A_SCHEMA == "wm_n1_features_idea_a/1"
    assert IDEA_A_DTYPES["event_mask"] == "uint8"
    assert road_end_relabel_enabled({}) is False
    assert road_end_relabel_enabled({"amendment2_road_end_relabel": True}) \
        is True
    assert road_end_relabel_enabled({"amendment2_road_end_relabel": False}) \
        is False
    assert road_end_relabel_enabled({}, default=True) is True


def test_passthrough_empty_root(tmp_path):
    p = IdeaAPassthrough(tmp_path)
    assert len(p) == 0
    assert p.has_idea_a_channels() is False
    with pytest.raises(IndexError):
        p.get(0, "veh_kin")
    with pytest.raises(KeyError):
        p.get(0, "not_a_channel")
    with pytest.raises(IndexError):
        p._dir(0)
    assert callable(write_idea_a_channels)
    assert callable(verify_idea_a)


def test_road_end_relabel_reader_alias():
    from corpus.idea_a import road_end_relabel_enabled as r

    assert r({"amendment2_road_end_relabel": 1}) is True
