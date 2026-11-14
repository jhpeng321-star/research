"""Focused tests for the Stage 2 generation contract (pure logic, no SUMO)."""

from __future__ import annotations

import re

from experiments.idm.stage2_generation_contract import (
    demand_for_seed,
    per_lane_unclipped,
    stage2_feature_vector,
    write_seed_scenario,
)


def test_demand_draw_is_deterministic_and_in_range():
    for seed in (8001, 8123, 8360):
        d1 = demand_for_seed(seed)
        d2 = demand_for_seed(seed)
        assert d1 == d2
        assert 2200 <= d1 <= 2800 and float(d1).is_integer()


def test_scenario_writer_sorts_flows_and_keeps_ramp(tmp_path):
    scene = write_seed_scenario(8001, 2500, root=tmp_path)
    assert scene.endswith(".sumocfg")
    route = tmp_path / scene.replace(".sumocfg", ".rou.xml").replace("osm.dense.", "")
    text = route.read_text(encoding="utf-8")
    begins = [float(b) for b in re.findall(r'begin="([0-9.]+)"', text)]
    assert begins == sorted(begins)
    assert 'id="ramp"' in text and 'vehsPerHour="400"' in text
    # Plateau-decline segments: plateau at the drawn demand, tail at 1500.
    assert 'vehsPerHour="2500"' in text and 'vehsPerHour="1500"' in text


def test_per_lane_unclipped_records_negative_in_band():
    summary = per_lane_unclipped({
        "lane_0": [350.0, 420.0],   # one CAV inside the band
        "lane_1": [410.0],
        "lane_2": [],               # no CAVs -> dropped
    })
    assert set(summary) == {"lane_0", "lane_1"}
    assert summary["lane_0"]["nearest_to_band_top_unclipped"] == -50.0
    assert summary["lane_0"]["count_in_band"] == 1.0
    assert summary["lane_1"]["nearest_to_band_top_unclipped"] == 10.0


def _record(region_counts, ramp_entries=0.0, upstream_entries=0.0, merge_discharge=0.0):
    return {
        "regions": {
            name: {"count": count, "mean_speed": 20.0, "speed_std": 1.0}
            for name, count in region_counts.items()
        },
        "mainline_lane_imbalance": 0.0,
        "ramp_queue_length": 0.0,
        "cav_count_band": 0.0,
        "cav_count_upstream": 0.0,
        "ramp_entries": ramp_entries,
        "upstream_entries": upstream_entries,
        "merge_discharge": merge_discharge,
    }


def test_feature_vector_lags_present_only_with_history():
    history = {t: _record({"merge_edge": 5, "post_merge": 4}) for t in range(30, 200)}
    vector = stage2_feature_vector(history, {}, 150, 0.5)
    for region in ("merge_edge", "post_merge"):
        for lag in (10, 30, 60):
            assert vector[f"{region}_mean_speed_lag{lag}s"] == 20.0
    # 120 s diffs available (history reaches 30).
    assert "merge_density_diff_120s" in vector

    short_history = {t: _record({"merge_edge": 5}) for t in range(80, 160)}
    vector2 = stage2_feature_vector(short_history, {}, 150, 0.5)
    # 150 - 10/30/60 = 140/120/90, all inside [80, 160): every lag present.
    for lag in (10, 30, 60):
        assert f"merge_edge_mean_speed_lag{lag}s" in vector2
    # At t = 100 only the 10 s lag (90) is inside history; 70 and 40 are not.
    vector3 = stage2_feature_vector(short_history, {}, 100, 0.5)
    assert "merge_edge_mean_speed_lag10s" in vector3
    assert "merge_edge_mean_speed_lag30s" not in vector3
    assert "merge_edge_mean_speed_lag60s" not in vector3
    assert "merge_density_diff_120s" not in vector3  # 100 - 120 = -20 missing
