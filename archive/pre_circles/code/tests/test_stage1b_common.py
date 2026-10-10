"""Focused tests for Stage 1b-prep shared helpers (pure logic)."""

from __future__ import annotations

import numpy as np

from experiments.idm.analyze_sustained_wall_v1_stage1b_band_locator import (
    _clopper_pearson,
)
from experiments.idm.stage1b_common import (
    LOCATION_BAND,
    LOCATION_MERGE,
    LOCATION_NEAR,
    LOCATION_OTHER,
    LOCATION_POST,
    LOCATION_RAMP,
    LOCATION_UPSTREAM,
    event_phase_bucket,
    location_bucket,
    phase_bucket,
    series_in_window,
)


def test_location_bucket_mainline_bands():
    # PRE_MERGE_LANE_LENGTH = 1144.95; distance = 1144.95 - lane_position.
    assert location_bucket("1191554636", 1144.95 - 500.0) == LOCATION_UPSTREAM
    assert location_bucket("1191554636", 1144.95 - 400.0) == LOCATION_BAND
    assert location_bucket("1191554636", 1144.95 - 200.0) == LOCATION_BAND
    assert location_bucket("1191554636", 1144.95 - 199.99) == LOCATION_NEAR
    assert location_bucket("1191554636", 1144.95 - 10.0) == LOCATION_NEAR


def test_location_bucket_edges_and_others():
    assert location_bucket("1487606179", None) == LOCATION_MERGE
    assert location_bucket(":junction_0", None) == LOCATION_MERGE
    assert location_bucket("1487606178", None) == LOCATION_POST
    assert location_bucket("11553048#1", None) == LOCATION_RAMP
    assert location_bucket("something_else", None) == LOCATION_OTHER


def test_phase_bucket_boundaries():
    assert phase_bucket(200, 200, 381) == "prefix"
    assert phase_bucket(201, 200, 381) == "during_wall"
    assert phase_bucket(380, 200, 381) == "during_wall"
    assert phase_bucket(381, 200, 381) == "release_60"
    assert phase_bucket(441, 200, 381) == "release_60"
    assert phase_bucket(442, 200, 381) == "after_60"
    assert event_phase_bucket({"relative_step": 380}, 381) == "during_wall"
    assert event_phase_bucket({"relative_step": 381}, 381) == "release_60"


def test_series_in_window():
    series = {"step": [0, 1, 2, 3, 4], "hard_brake_count": [0, 1, 2, 3, 4]}
    assert series_in_window(series, "hard_brake_count", 1, 3) == 6.0
    assert series_in_window(series, "hard_brake_count", 5, 10) == 0.0


def test_clopper_pearson_edges():
    lo, hi = _clopper_pearson(0, 30)
    assert abs(lo - 0.0) < 1e-12
    assert 0.10 < hi < 0.13  # exact bound ~ 0.116
    lo, hi = _clopper_pearson(30, 30)
    assert 0.87 < lo < 0.90
    assert abs(hi - 1.0) < 1e-12
    lo, hi = _clopper_pearson(24, 30)
    assert abs(lo - 0.6143) < 0.02 and abs(hi - 0.9229) < 0.02
