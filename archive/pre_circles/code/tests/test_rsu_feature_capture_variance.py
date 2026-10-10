"""Stage 1c-2 Part A regression test: the recording fix, feature by feature.

The p20 generation recorded ``cav_count_band`` / ``cav_count_upstream`` as
constant zero because the capture read a nonexistent env attribute
(``active_cavs``; the env only has ``_active_cavs``), and
``ramp_queue_length`` used a 15 m/s threshold that never fired at decision
times. This test replays the frozen analytical damping controller on the
dense transition scenario (2700/400) for three development seeds (4101-4103,
analytical-pilot role, excluded from evaluation) for 255 steps - long enough
for the full feature schema at t = 150 and t = 250, including the 120 s
backward differences - and asserts:

1. the two defect-broken CAV counts now vary across seeds and are integral;
2. the ramp queue is an integral count bounded by the ramp region count;
3. no feature is constant across the pooled rows except a documented
   allowlist of features that are constant by construction, not by defect:
   - ``lane_*_nearest_to_band_top`` is censored to 0 whenever the band
     holds a CAV (min distance inside the band clips to 0);
   - ``ramp_30s_entries``, ``upstream_inflow_30s`` and ``ramp_count`` are
     seed-invariant because the route template inserts vehicles with
     deterministic ``vehsPerHour`` headways (seeds vary car-following and
     CAV scatter, not arrival headways);
   - ``ramp_queue_length`` stays 0 on this corridor: even under merge
     gridlock (h = 0 by t = 250) ramp-edge vehicles do not drop below
     5 m/s at decision times.
"""

from __future__ import annotations

import statistics

import pytest

from experiments.idm import deterministic_sumo  # noqa: F401  (race-free launch first)
from experiments.idm.rerecord_stage1b_features import replay_d_v2
from experiments.idm.run_stage1b_generation import _extract_features
from experiments.idm.run_sustained_wall_v1_stage1a import DEFAULT_SUMO_EXE, env_overrides
from idm.contract import SCENARIO_ROOT as TRANSITION_SCENARIO_ROOT
from idm.contract import transition_scenario_config

SEEDS = (4101, 4102, 4103)
DEMAND_VPH = 2700
TOTAL_STEPS = 255
FEATURE_TIMES = (150, 250)
PROCESS_CONSTANT_ALLOWLIST = {
    "ramp_30s_entries",
    "upstream_inflow_30s",
    "ramp_count",
    "ramp_queue_length",
}


def _sumo_binary() -> str:
    return str(DEFAULT_SUMO_EXE) if DEFAULT_SUMO_EXE.exists() else "sumo"


@pytest.fixture(scope="module")
def feature_vectors():
    scenario_name = transition_scenario_config(DEMAND_VPH).name
    assert env_overrides()["cav_placement"] == "scatter"
    vectors = []
    for seed in SEEDS:
        trace = replay_d_v2(
            seed=seed, scenario_name=scenario_name, sumo_binary=_sumo_binary(),
            total_steps=TOTAL_STEPS, project_root=TRANSITION_SCENARIO_ROOT,
        )
        features = _extract_features(trace)
        assert set(features) == set(FEATURE_TIMES), sorted(features)
        vectors.extend(features[t] for t in FEATURE_TIMES)
    return vectors


def test_defect_broken_cav_counts_now_vary(feature_vectors):
    for key in ("cav_count_band", "cav_count_upstream"):
        values = [row[key] for row in feature_vectors]
        assert statistics.pvariance(values) > 0.0, (key, values)
        assert all(v >= 0.0 and float(v).is_integer() for v in values), key


def test_ramp_queue_is_an_integral_count_bounded_by_ramp_region(feature_vectors):
    for row in feature_vectors:
        queue = row["ramp_queue_length"]
        assert queue >= 0.0 and float(queue).is_integer(), queue
        assert queue <= row["ramp_count"] + 1e-9, (queue, row["ramp_count"])


def test_no_feature_constant_except_documented_allowlist(feature_vectors):
    keys = sorted(feature_vectors[0].keys())
    assert "cav_count_band" in keys and "ramp_queue_length" in keys
    constant = {
        key
        for key in keys
        if statistics.pvariance([row[key] for row in feature_vectors]) == 0.0
    }
    censored = {k for k in constant if k.endswith("_nearest_to_band_top")}
    unexpected = constant - PROCESS_CONSTANT_ALLOWLIST - censored
    assert not unexpected, f"constant features (recording defect): {sorted(unexpected)}"
