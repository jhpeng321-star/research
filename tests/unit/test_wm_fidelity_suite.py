"""scripts/c4/wm_fidelity_suite.py: pure C1-C6 metric helpers and verdict rules.

No torch, no checkpoint: the module is loaded via ``importlib.util`` so only the pure
helpers (kinematics, TTC, road-end count, lane share, free-flow speed, wave amplitude,
deviation/status) are exercised.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "wm_fidelity_suite", REPO / "scripts" / "c4" / "wm_fidelity_suite.py")
FS = importlib.util.module_from_spec(_spec)
sys.modules["wm_fidelity_suite"] = FS
_spec.loader.exec_module(FS)


def _v(vid, s, lane, v, is_cav=False):
    return {"id": vid, "s": float(s), "lane": int(lane), "v": float(v),
            "present": True, "is_cav": bool(is_cav)}


def _f(t, rows):
    return {"time_s": float(t), "vehicles": rows}


def _state(rows):
    return {"initial_state": {"time_s": 0.0, "vehicles": rows}}


def test_t0_roster_and_cav_ids():
    record = _state([_v("h", 10, 0, 20), _v("c", 20, 1, 20, is_cav=True)])
    frame, roster, cavs = FS._t0_roster(record)
    assert frame["time_s"] == 0.0
    assert roster == {"h", "c"} and cavs == {"c"}


def test_cav_behaviour_accel_hard_brake_and_gap():
    rows = [_v("c", 100, 0, 20, is_cav=True), _v("lead", 140, 0, 20)]
    frames = [_f(0.0, rows),
              _f(0.5, [_v("c", 106, 0, 18, is_cav=True), _v("lead", 150, 0, 20)]),
              _f(1.0, [_v("c", 106, 0, 16, is_cav=True), _v("lead", 160, 0, 20)])]
    out = FS.cav_behaviour(frames, 0.0, 1.0, {"c", "lead"}, {"c"})
    assert out["n_accel"] == 2
    assert out["accel_mean"] == -4.0
    assert out["hard_brake_share"] == 1.0
    assert out["gap_median"] == 44.0        # (150-5-106, 160-5-106)
    assert out["thw_median"] is not None


def test_ttc_near_miss_rates():
    rows = [_v("f", 0, 1, 20), _v("lead", 20, 1, 10)]
    frames = [_f(0.0, rows), _f(0.5, rows), _f(1.0, rows)]
    out = FS.ttc_distribution(frames, 0.0, 1.0, {"f", "lead"}, set())
    block = out["all_pairs"]
    assert block["n_pair_steps"] == 2
    assert block["ttc_median"] == 1.5
    assert block["near_miss_3_rate"] == 1.0
    assert block["near_miss_15_rate"] == 0.0      # strict < 1.5
    assert out["cav_follower_pairs"]["n_pair_steps"] == 0


def test_road_end_count_and_seconds():
    frames = [_f(0.0, [_v("a", 500, 0, 20), _v("b", 600, 0, 20)]),
              _f(0.5, [_v("a", 510, 0, 20), _v("b", 610, 0, 20)]),
              _f(1.0, [_v("a", 520, 0, 20), _v("b", 620, 0, 20)])]
    out = FS.road_end_exits(frames, 0.0, 1.0)
    assert out["n_vehicles_beyond"] == 1
    assert out["beyond_vehicle_steps"] == 3
    assert out["beyond_vehicle_seconds"] == 1.5
    assert out["top_ids"] == ["b"]


def test_road_end_code_path_reports_no_removal():
    out = FS.road_end_code_path()
    assert out["model_has_exit_gate"] is True
    assert out["predictor_sets_exit_s"] is False
    assert out["rollouts_remove_at_road_end"] is False
    assert "do NOT remove" in out["finding"]


def test_lane_share_and_per_lane_speed():
    rows = [_v("a", 10, 0, 20), _v("b", 20, 0, 30), _v("c", 30, 1, 10),
            _v("d", 40, 2, 5)]
    frames = [_f(0.0, rows), _f(60.0, rows)]
    out = FS.lane_share_speed(frames, 0.0, 60.0, {"a", "b", "c", "d"}, (0, 60))
    assert out["0"]["lane0_hdv_share"] == 2 / 3
    assert out["60"]["lane1_speed"] == 10.0
    assert out["60"]["lane2_speed"] == 5.0


def test_free_flow_speed_excludes_followers():
    rows = [_v("a", 0, 0, 20), _v("b", 30, 0, 20), _v("c", 300, 0, 30)]
    frames = [_f(0.5, rows), _f(1.0, rows)]
    out = FS.free_flow_speed(frames, 0.0, 1.0, {"a", "b", "c"}, gap_m=100)
    assert out["0"]["n"] == 4          # b and c are free; a follows b (gap 25 < 100)
    assert out["0"]["ff_mean"] == 25.0


def test_wave_amplitude_drop_and_ratio():
    frames = []
    for t, v_lead, v_fol in ((0.0, 20, 20), (0.5, 20, 20), (1.0, 15, 20),
                             (1.5, 10, 12), (2.0, 12, 11)):
        frames.append(_f(t, [_v("fol", t * 10, 1, v_fol), _v("lead", 100 + t * 20, 1,
                                                             v_lead)]))
    out = FS.wave_amplitude(frames, 0.0, 2.0, {"fol", "lead"}, set(), window_s=1.0)
    assert out["n_events"] >= 1
    assert out["drops"][0] is not None and out["drops"][0] > 0
    assert out["ratios"][0] is not None


def test_deviation_and_status_use_relative_tolerance():
    assert FS._deviation(130.0, 100.0, 1.0) == 0.3
    assert FS._deviation(10.0, 0.0, 1.0) == 10.0
    row = FS._compare("C1", "gap_median", 130.0, 100.0)
    assert row["flagged"] is True
    assert FS._status([FS._compare("C1", "gap_median", 110.0, 100.0)]) == "PASS"
    assert FS._status([FS._compare("C1", "gap_median", 140.0, 100.0)]) == "FLAG"


def test_sanitize_makes_json_safe():
    out = FS._sanitize({"a": float("nan"), "b": [float("inf"), 1.0]})
    assert out == {"a": None, "b": [None, 1.0]}
