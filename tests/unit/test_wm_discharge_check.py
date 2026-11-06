"""scripts/c4/wm_discharge_check.py: pure C11 discharge/zipper helpers.

No torch, no checkpoint: the module is loaded via ``importlib.util`` (like the
fidelity-suite test) so only the pure helpers (service events, queue series, headways,
zipper alternation, follower braking, bootstrap bands, guards) are exercised.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "wm_discharge_check", REPO / "scripts" / "c4" / "wm_discharge_check.py")
DC = importlib.util.module_from_spec(_spec)
sys.modules["wm_discharge_check"] = DC
_spec.loader.exec_module(DC)

MERGE_END = 75.44
HORIZON = 10.0


def _v(vid, s, lane, v):
    return {"id": vid, "s": float(s), "lane": int(lane), "v": float(v),
            "present": True, "is_cav": False}


def _m4_v(t):
    return 7.0 if t <= 2.0 else max(7.0 - 2.5 * (t - 2.0), 0.5)


def _rows(t):
    """The synthetic merge episode: r1/r3 ramp cars merge, r2 never does."""
    return [
        _v("r1", 30 + 5 * t if t < 2 else 40 + 5 * (t - 2), 2 if t < 2 else 1, 5.0),
        _v("r2", 10 + 1 * t, 2, 1.0),
        _v("r3", 40 + 5 * t if t < 3 else 55 + 4 * (t - 3), 2 if t < 3 else 1, 5.0),
        _v("m1", 40 + 10 * t, 1, 10.0),
        _v("m2", 50 + 10 * t, 0, 10.0),
        _v("m3", 60 + 1 * t, 1, 1.0),
        _v("m4", 20 + 7 * t, 1, _m4_v(t)),
    ]


def _frames():
    steps = int(HORIZON / 0.5)
    return [{"time_s": i * 0.5, "vehicles": _rows(i * 0.5)} for i in range(steps + 1)]


ROSTER = {"r1", "r2", "r3", "m1", "m2", "m3", "m4"}


def test_service_events_merges_and_crossings():
    events = DC.service_events(_frames(), 0.0, HORIZON, ROSTER, MERGE_END)
    merges = [(e["id"], e["t"]) for e in events["merges"]]
    assert merges == [("r1", 2.0), ("r3", 3.0)]
    assert events["first_lane"]["r1"] == 2 and events["first_lane"]["m1"] == 1
    crossings = [(e["id"], e["t"], e["origin"], e["lane"])
                 for e in events["crossings"]]
    # m2 crosses on lane 0, m1/m4 mainline on lane 1, r3/r1 after their merges
    assert crossings == [("m2", 3.0, "mainline", 0), ("m1", 4.0, "mainline", 1),
                         ("m4", 8.0, "mainline", 1), ("r3", 8.5, "ramp", 1),
                         ("r1", 9.5, "ramp", 1)]


def test_queue_series_and_queues_present():
    events = DC.service_events(_frames(), 0.0, HORIZON, ROSTER, MERGE_END)
    series = DC.queue_series(_frames(), 0.0, HORIZON, ROSTER, events, MERGE_END)
    at = {round(row["t"], 1): (row["ramp_q"], row["mainline_q"]) for row in series}
    assert at[0.0] == (3, 4)          # r1,r2,r3 on lane 2; m1,m2,m3,m4 unserved
    assert at[2.0] == (2, 4)          # r1 merged (m2 not yet across)
    assert at[3.0] == (1, 3)          # r3 merged, m2 crossed at 3.0
    assert at[9.5] == (1, 1)          # only r2 and m3 remain


def test_headway_stats_quantiles():
    out = DC.headway_stats([4.0, 8.0, 8.0, 9.5])
    assert out["n"] == 2                       # the simultaneous pair is dropped
    assert out["p50"] == 2.75
    assert DC.headway_stats([1.0])["n"] == 0
    assert DC.headway_stats([1.0])["p50"] is None


def test_zipper_alternation():
    events = DC.service_events(_frames(), 0.0, HORIZON, ROSTER, MERGE_END)
    series = DC.queue_series(_frames(), 0.0, HORIZON, ROSTER, events, MERGE_END)
    out = DC.zipper_stats(events["crossings"], series)
    assert out["n_events_lane1"] == 4
    assert out["n_pairs"] == 3                  # both queues present at 4.0/8.0/8.5/9.5
    assert out["same_stream_share"] == 2 / 3    # m1->m4 same, m4->r3 not, r3->r1 same
    assert out["run_len_median"] == 1.5         # runs [1, 2]
    assert out["n_runs"] == 2


def test_discharge_block_rates_dwell_queue():
    block = DC.discharge_block(_frames(), 0.0, HORIZON, ROSTER, MERGE_END,
                               queue_targets=(0, 10))
    assert block["n_ramp_t0"] == 3
    assert block["rates"]["ramp_merges_per_min"] == 12.0     # 2 in 1/6 min
    assert block["rates"]["mainline_passages_per_min"] == 18.0
    assert block["rates"]["lane1_crossings_per_min"] == 24.0
    assert block["headways"]["insertion"]["p50"] == 1.0
    # lane-1 discharge times 4.0/8.0/8.5/9.5 -> headways 4.0/0.5/1.0
    assert block["headways"]["discharge_lane1"]["n"] == 3
    assert block["headways"]["discharge_lane1"]["p50"] == 1.0
    assert block["dwell"]["n_served"] == 2
    assert block["dwell"]["served_p50"] == 2.5
    assert abs(block["dwell"]["never_served_share"] - 1 / 3) < 1e-9
    assert abs(block["queue"]["ramp_queue_mean"] - 31 / 21) < 1e-6
    assert block["queue"]["ramp_queue_at"]["0"] == 3.0
    assert block["queue"]["ramp_queue_at"]["10"] == 1.0
    assert block["merge_s_median_m"] == 47.5


def test_follower_decel_around_insertions():
    block = DC.discharge_block(_frames(), 0.0, HORIZON, ROSTER, MERGE_END)
    follower = block["follower"]
    # r1's insertion (s=40): nearest lane-1 follower is m4 (braking, min a=-2.5,
    # gap 1.0); r3's insertion (s=55): nearest follower is the merged r1 (constant
    # speed, min a=0.0, gap 5.0) - previously merged ramp cars count as followers.
    assert follower["n_events"] == 2
    assert follower["min_accel_p50"] == -1.25
    assert follower["gap_median_m"] == 3.0


def test_bootstrap_mean_deterministic_and_covers():
    values = [float(i) for i in range(8)]
    a = DC.bootstrap_mean(values, n_boot=500, seed=7)
    b = DC.bootstrap_mean(values, n_boot=500, seed=7)
    assert a == b
    assert a["lo"] <= a["mean"] <= a["hi"]
    const = DC.bootstrap_mean([3.0] * 4, n_boot=100, seed=1)
    assert const["lo"] == const["hi"] == const["mean"] == 3.0
    assert DC.bootstrap_mean([1.0], n_boot=10, seed=1) is None


def test_fleet_rows_flag_outside_band():
    states = []
    for value in (1.0, 1.0, 1.0, 1.0):
        states.append({
            "sumo": {"rates": {"ramp_merges_per_min": value}},
            "wm_blocks": {"wm1": {"wm": {"rates": {"ramp_merges_per_min": 5.0}}}},
        })
    rows = DC.fleet_rows(states, ["wm1"], n_boot=200, seed=3)
    row = rows["rates.ramp_merges_per_min"]
    assert row["sumo"]["lo"] == 1.0 and row["sumo"]["hi"] == 1.0
    assert row["wm"]["wm1"]["flagged"] is True
    assert row["wm"]["wm1"]["direction"] == "above"


def test_guards_refuse_restricted_and_eval_seeds():
    assert DC.refused_seed(964123) is True
    assert DC.refused_seed(964000) and DC.refused_seed(964199)
    assert DC.refused_seed(411) is False and DC.refused_seed(None) is False
    for bad in ("sealed/x.json", "data/holdout/states.json", "cal_split.json",
                "chunk5", "j4b", "inc1"):
        try:
            DC.refuse_restricted(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"restricted path accepted: {bad}")
    assert DC.refuse_restricted("delay_check/delay_states.json")


def test_per_start_values_leaf_lookup():
    block = {"rates": {"ramp_merges_per_min": 2.5}, "dwell": {"never_served_share": None}}
    vals = DC.per_start_values({"sumo": block}, "sumo")
    assert vals["rates.ramp_merges_per_min"] == 2.5
    assert vals["dwell.never_served_share"] is None
    assert not any(isinstance(v, bool) for v in vals.values())
