"""The final-HOLDOUT guard refuses the owner-excluded roots (PL-02a) and keeps
the disclosed format-only roots (WM-70 w8477, 2026-09-29; w8617, owner
ruling 2026-09-30)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import holdout_guard as hg  # noqa: E402


def test_exclusion_file_loads_and_is_consistent():
    data = hg.load_holdout_exclusions()
    excluded = hg.excluded_roots(data)
    kept = set(hg.disclosed_roots(data))
    assert len(excluded) == data["expected_excluded_count"]
    assert not (excluded & kept)
    # owner rulings: BOTH format-only-exposed worlds stay KEPT + disclosed
    # (w8477 WM-70 ruling 2026-09-29; w8617 ruling 2026-09-30)
    assert kept == {"w8477m2600p02t480", "w8617m2700p02t300"}
    # each kept world carries its own KEPT + disclosed note in the file
    for world in sorted(kept):
        assert any(world in note and "KEPT + disclosed" in note
                   for note in data["notes"]), world
    # the two WM-70 format-only roots that PL-02a also exposed are EXCLUDED
    assert {"w8463m2700p40t420", "w8407m2400p02t360"} <= excluded
    assert "w8451m2700p20t420" not in excluded  # the 14th, never analysed


def test_final_loader_refuses_every_excluded_root():
    data = hg.load_holdout_exclusions()
    for world in hg.excluded_roots(data):
        with pytest.raises(hg.HoldoutExclusionError):
            hg.assert_final_holdout_eligible([world], data)
        with pytest.raises(hg.HoldoutExclusionError):
            hg.assert_final_holdout_eligible([f"{world}__native"], data)
        assert hg.filter_final_holdout([world, "w9999m2700p10t300"], data) == ["w9999m2700p10t300"]


def test_disclosed_roots_stay_eligible():
    data = hg.load_holdout_exclusions()
    kept = [w for w in hg.disclosed_roots(data) if w not in hg.excluded_roots(data)]
    hg.assert_final_holdout_eligible(kept, data)
