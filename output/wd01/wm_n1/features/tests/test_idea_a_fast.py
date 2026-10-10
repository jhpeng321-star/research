"""FEAT1 fast-baseline flag: byte-identical outputs, faster Idea A pass.

The ``fast=True`` path of ``features.idea_a.scene_baseline_channels``
caches the per-lane sorted rosters (rebuilt only after an accepted lane
change) and memoizes the frozen engine module.  These tests pin the
contract that the fast outputs are EXACTLY equal (no tolerance) to the
reference path on dense random scenes that exercise accepted mandatory
merges and discretionary MOBIL changes, and through
``idea_a_channels_from_shard`` on a synthetic shard.  Zero SUMO; no data
reads.
"""
from __future__ import annotations

import numpy as np
import pytest

from features.idea_a import (HDV_BASE_FIELDS, _ENGINE_MEMO, _engine,
                             _engine_cached, idea_a_channels_from_shard,
                             load_nominal_params, scene_baseline_channels)

K = 24
RNG = np.random.default_rng(20261001)


def _random_scene(rng):
    lane = np.full(K, -1, np.int64)
    s = np.zeros(K)
    v = np.zeros(K)
    length = np.full(K, 5.0)
    present = np.zeros(K, bool)
    n = int(rng.integers(2, K + 1))
    for j in range(n):
        li = int(rng.integers(0, 3))
        if li == 2:
            sj = float(rng.uniform(360.0, 460.0))
        else:
            sj = float(rng.uniform(300.0, 2700.0))
        lane[j], s[j], v[j], present[j] = (
            li, sj, float(rng.uniform(3.0, 33.0)), True)
    return lane, s, v, length, present


OUTPUT_KEYS = tuple(HDV_BASE_FIELDS) + ("lane_next", "s_next", "v_next",
                                        "merged")


def test_fast_matches_reference_on_random_scenes(params):
    rng = np.random.default_rng(20261001)
    n_merged = n_lc = 0
    for _ in range(48):
        lane, s, v, length, present = _random_scene(rng)
        ref = scene_baseline_channels(lane, s, v, length, present,
                                      params=params)
        fast = scene_baseline_channels(lane, s, v, length, present,
                                       params=params, fast=True)
        for key in OUTPUT_KEYS:
            a, b = ref[key], fast[key]
            assert a.dtype == b.dtype, key
            assert np.array_equal(a, b), key
        n_merged += int(ref["merged"].sum())
        n_lc += int((ref["lane_next"] >= 0).sum())
    assert n_merged > 0          # the scenes exercised accepted merges
    assert n_lc > n_merged       # ... and discretionary changes


def test_fast_matches_reference_single_vehicle_and_empty(params):
    lane = np.full(K, -1, np.int64)
    s = np.zeros(K)
    v = np.zeros(K)
    length = np.full(K, 5.0)
    present = np.zeros(K, bool)
    ref = scene_baseline_channels(lane, s, v, length, present, params=params)
    fast = scene_baseline_channels(lane, s, v, length, present,
                                   params=params, fast=True)
    for key in OUTPUT_KEYS:
        assert np.array_equal(ref[key], fast[key])
    present[0] = True
    lane[0], s[0], v[0] = 1, 2500.0, 25.0
    ref = scene_baseline_channels(lane, s, v, length, present, params=params)
    fast = scene_baseline_channels(lane, s, v, length, present,
                                   params=params, fast=True)
    for key in OUTPUT_KEYS:
        assert np.array_equal(ref[key], fast[key])


def test_engine_cached_returns_same_module():
    assert _engine_cached() is _engine()
    assert _ENGINE_MEMO and _ENGINE_MEMO[0] is _engine_cached()


def _synthetic_shard():
    n, k = 12, 4
    veh = np.zeros((n, k, 16), np.float16)
    mask = np.zeros((n, k), np.uint8)
    wall = np.zeros((n, 2, 11), np.float16)
    wall_mask = np.zeros((n, 2), np.uint8)
    rng = np.random.default_rng(7)
    for j in range(n):
        veh[j, 0, 0] = rng.uniform(100.0, 2500.0)
        veh[j, 0, 1] = 1.0                       # lane 1
        veh[j, 0, 4] = rng.uniform(10.0, 30.0)
        veh[j, 0, 5] = rng.uniform(-1.0, 1.0)
        veh[j, 0, 6] = 5.0
        veh[j, 0, 15] = 1.0
        veh[j, 1, 0] = rng.uniform(380.0, 440.0)
        veh[j, 1, 3] = 1.0                       # accel lane
        veh[j, 1, 4] = rng.uniform(10.0, 30.0)
        veh[j, 1, 5] = 0.0
        veh[j, 1, 6] = 5.0
        veh[j, 1, 15] = 1.0
        mask[j, :2] = 1
    wall[:, 0, 0] = 1.0                          # holding phase
    wall_mask[:, 0] = 1
    return dict(veh_step=veh, veh_mask=mask, wall=wall, wall_mask=wall_mask)


def test_idea_a_channels_fast_identical(params):
    shard = _synthetic_shard()
    ref = idea_a_channels_from_shard(shard, params=params, fast=False)
    fast = idea_a_channels_from_shard(shard, params=params, fast=True)
    for name in ("veh_kin", "hdv_base", "cav_plan_ch", "event_tok",
                 "event_mask"):
        assert np.array_equal(ref[name], fast[name]), name


@pytest.fixture(scope="module")
def params():
    return load_nominal_params()
